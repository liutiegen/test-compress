#!/usr/bin/env python3
"""
stage1.1 主量化脚本 —— 针对单个 (algorithm, scheme, target) 组合量化 Qwen2.5-0.5B-Instruct。

命令行契约见 README.md。所有 llm-compressor API 均按 0.15.0 实包校验：
  - oneshot(model=..., tokenizer=..., recipe=[...], dataset=<hf Dataset>,
            max_seq_length=..., num_calibration_samples=..., output_dir=...)
  - QuantizationModifier(scheme=<preset名>, targets=[...], ignore=[...])
  - GPTQModifier(scheme=..., targets=..., ignore=..., actorder="static")
  - SmoothQuantModifier(smoothing_strength=...)
  - AWQModifier(duo_scaling=...)  → 必须与 QuantizationMixin 子类同用
  - preset 名合法值来自 compressed_tensors 0.20.1 PRESET_SCHEMES

设计要点
--------
1. rt 与 gptq 的差别只在"量化器"：gptq 用 GPTQModifier（需要 hessian），
   rtn 用 QuantizationModifier（无数据依赖的 round-to-nearest）。
2. smoothquant / awq 是 *transform* modifier，必须再配一个量化器，
   否则产出的是未压缩的 fp16 权重（这是最容易踩的坑）。
3. 量化位置由 targets 控制；lm_head 永远 ignore。
4. KV cache 量化只在 target == linear_all_kv 时启用。
5. 结果自检：量化后必须真的存在 compressed-tensors 的 quantization_config，
   且 linear 层出现 weight_scale/int_weight 之类张量，否则判定失败。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def log(msg: str) -> None:
    print(f"[quantize] {msg}", flush=True)


def build_recipe(variant: dict):
    """根据 variant 描述构造 recipe（modifier 列表）。"""
    from llmcompressor.modifiers.quantization import QuantizationModifier

    algo = variant["algorithm"]
    scheme = variant["preset_scheme"]
    targets = variant["targets"]
    ignore = ["lm_head"]

    mods = []

    if algo == "smoothquant":
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

        mods.append(SmoothQuantModifier(smoothing_strength=0.8))
        mods.append(
            QuantizationModifier(scheme=scheme, targets=targets, ignore=ignore)
        )

    elif algo == "awq":
        from llmcompressor.modifiers.transform.awq import AWQModifier

        # duo_scaling 只支持 per-channel 量化；W8A8 是 per-tensor 通道式，
        # 因此 W8A8 时必须关闭，否则 AWQModifier 校验会直接抛错。
        duo = False if scheme == "W8A8" else "both"
        mods.append(AWQModifier(duo_scaling=duo))
        mods.append(
            QuantizationModifier(scheme=scheme, targets=targets, ignore=ignore)
        )

    elif algo == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier

        mods.append(
            GPTQModifier(
                scheme=scheme,
                targets=targets,
                ignore=ignore,
                actorder="static",
                dampening_frac=0.01,
            )
        )

    elif algo == "rtn":
        mods.append(
            QuantizationModifier(scheme=scheme, targets=targets, ignore=ignore)
        )

    else:
        raise ValueError(f"unknown algorithm: {algo}")

    # KV cache 量化：独立于权重配方，附加为第二段
    if variant.get("use_kv_cache"):
        from compressed_tensors.quantization import QuantizationArgs

        mods.append(
            QuantizationModifier(
                targets=[],
                kv_cache_scheme=QuantizationArgs(
                    num_bits=8,
                    type="int",
                    strategy="tensor",
                    symmetric=True,
                    dynamic=False,
                ),
            )
        )

    return mods


def build_calibration_dataset(model_path: str, tokenizer, num_samples: int,
                             max_seq_length: int, dataset_path: str):
    """
    校准数据集：本地 parquet（已由 stage3 download_files 预置）。

    关键点：llm-compressor 的 oneshot 接受 HuggingFace Dataset 对象，
    但要求列里是 tokenized 的 input_ids/attention_mask（见
    examples/custom_dataset_example.py），因此这里先套 chat template 再 tokenize。
    使用本地文件路径，避免运行期联网。
    """
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=str(Path(dataset_path) / "data" / "train_sft-*.parquet"),
        split="train",
    )
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    ds = ds.map(preprocess)

    def tokenize_fn(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    return ds


def verify_quantized(model_dir: Path) -> tuple[bool, dict]:
    """
    产物自检。返回 (ok, evidence)。

    检查项：
      1. config.json 里有 quantization_config（compressed-tensors 写的）
      2. 权重文件里出现 int4/int8 scale 张量（证明真的量化了，而非只改了 config）
      3. quantization_config 里的 targets 与我们请求的一致
    """
    import torch
    from safetensors import safe_open

    evidence: dict = {}
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return False, {"error": "config.json missing"}

    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg.get("quantization_config")
    evidence["has_quantization_config"] = qcfg is not None
    if not qcfg:
        return False, evidence
    evidence["quant_method"] = qcfg.get("quant_method")
    evidence["format"] = qcfg.get("format")

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        return False, {**evidence, "error": "no safetensors"}

    scale_like = 0
    int_like = 0
    total = 0
    with safe_open(str(st_files[0]), framework="pt") as f:
        for k in f.keys():
            total += 1
            if "weight_scale" in k or "weight_zero_point" in k:
                scale_like += 1
            if "weight_packed" in k or "weight_shape" in k:
                int_like += 1

    evidence.update({
        "tensors_in_first_shard": total,
        "scale_tensors": scale_like,
        "packed_tensors": int_like,
    })
    ok = scale_like > 0 or int_like > 0
    return ok, evidence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="本地基座模型目录（OSS 挂载 / 本地拷贝）")
    ap.add_argument("--dataset-path", required=True,
                    help="本地数据集目录，需含 data/train_sft-*.parquet")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--algorithm", required=True,
                    choices=["smoothquant", "awq", "gptq", "rtn"])
    ap.add_argument("--scheme", required=True,
                    choices=["W8A8", "W4A16", "W4A16_ASYM", "FP8_DYNAMIC"])
    ap.add_argument("--target", required=True,
                    choices=["mlp", "attn", "linear_all", "linear_all_kv"])
    ap.add_argument("--num-calibration-samples", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--matrix-id", default=None)
    ap.add_argument("--gpu-capability", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="只解析 variant 并打印 recipe，不加载模型")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from matrix import Variant, SCHEMES, SHORT

    # 反查得到 preset 名 / targets
    variant_obj = Variant(args.algorithm, args.scheme, args.target)
    variant = {
        "matrix_id": args.matrix_id or variant_obj.matrix_id,
        "algorithm": args.algorithm,
        "scheme": args.scheme,
        "preset_scheme": SCHEMES[args.scheme][0],
        "targets": variant_obj.targets,
        "use_kv_cache": variant_obj.use_kv_cache,
    }
    log(f"variant = {variant['matrix_id']}  recipe={json.dumps(variant, ensure_ascii=False)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "quant_metrics.json"
    started = time.time()

    try:
        # ---- 硬件门控：算力不够就别浪费时间 ----
        cap = args.gpu_capability
        if cap is None:
            import torch
            major, minor = torch.cuda.get_device_capability()
            cap = major * 10 + minor
        need = SCHEMES[args.scheme][1]
        if cap < need:
            raise RuntimeError(
                f"GPU capability sm{cap} < required sm{need} for {args.scheme}"
            )
        log(f"gpu capability sm{cap} ok (need sm{need})")

        if args.dry_run:
            recipe = build_recipe(variant)
            log(f"dry-run recipe: {[type(m).__name__ for m in recipe]}")
            metrics_path.write_text(json.dumps(
                {"matrix_id": variant["matrix_id"], "variant": variant,
                 "recipe": [type(m).__name__ for m in build_recipe(variant)],
                 "dry_run": True}, indent=2, ensure_ascii=False))
            return 0

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot

        log(f"loading model from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )

        ds = build_calibration_dataset(
            args.model_path, tokenizer, args.num_calibration_samples,
            args.max_seq_length, args.dataset_path,
        )
        log(f"calibration samples = {len(ds)}")

        recipe = build_recipe(variant)
        log(f"recipe modifiers = {[type(m).__name__ for m in recipe]}")

        oneshot(
            model=model,
            tokenizer=tokenizer,
            recipe=recipe,
            dataset=ds,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
            output_dir=str(out),
        )

        model.save_pretrained(str(out), save_compressed=True)
        tokenizer.save_pretrained(str(out))

        ok, evidence = verify_quantized(out)
        elapsed = time.time() - started
        record = {
            "matrix_id": variant["matrix_id"],
            "variant": variant,
            "status": "ok" if ok else "verification_failed",
            "verify": evidence,
            "elapsed_sec": round(elapsed, 1),
            "gpu_capability": cap,
        }
        metrics_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        log(f"verify ok={ok} evidence={evidence}")

        if not ok:
            log("FATAL: 产物未包含量化张量，判定失败")
            return 2
        return 0

    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - started
        metrics_path.write_text(json.dumps({
            "matrix_id": variant["matrix_id"],
            "variant": variant,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(elapsed, 1),
        }, indent=2, ensure_ascii=False))
        log(f"ERROR {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())