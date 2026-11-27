#!/usr/bin/env python3
"""
stage1.2 微调 —— 对 stage1.1 选出的最优量化模型做 QAD（Quantization-Aware
Distillation），进一步恢复/稳住 benchmark 表现。

为什么用 QAD 而不是普通 SFT：
  1. 模型已经是 compressed-tensors 量化权重（int4/int8）。普通 SFT 在 fp16 上
     训练会与推理时实际执行的量化内核错配，训练收益兑现不了。
  2. llm-compressor 0.15.0 自带 QADModifier：它做的是"逐 decoder block 的
     假量化输出对齐未量化输出"，不需要额外 teacher 模型（README 明确说明）。
     这与"进一步保证量化后模型性能"的需求精确对应。
  3. 单卡可跑（需求指定"只用一个 k8s 节点对量化模型做微调"）。

API 依据（已对 0.15.0 实包校验）：
  QADModifier(num_epochs=..., lr=..., gradient_accumulation_steps=...)
  oneshot(..., recipe=[quantizer, QADModifier(...)], pipeline="sequential",
          sequential_targets=["Qwen2DecoderLayer"], propagate_error=True)

注意：QAD 内部会对已量化层重新 observe 权重尺度，所以 recipe 里必须把
"产生量化配置的那个 modifier" 放在 QAD 之前（即 stage1.1 选出的量化器），
不能只传 QAD。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def log(m: str) -> None:
    print(f"[finetune] {m}", flush=True)


def build_quantizer(variant: dict):
    """
    重建 stage1.1 用的量化器（与 quantize.py 的 build_recipe 保持一致），
    QAD 需要它来确定量化配置。
    """
    from llmcompressor.modifiers.quantization import QuantizationModifier

    algo = variant["algorithm"]
    scheme = variant["preset_scheme"]
    targets = variant["targets"]
    ignore = ["lm_head"]

    if algo == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier
        return GPTQModifier(scheme=scheme, targets=targets, ignore=ignore,
                            actorder="static", dampening_frac=0.01)
    if algo == "smoothquant":
        # 已量化 checkpoint 已包含平滑结果，QAD 阶段只需量化器重建配置
        return QuantizationModifier(scheme=scheme, targets=targets, ignore=ignore)
    if algo == "awq":
        return QuantizationModifier(scheme=scheme, targets=targets, ignore=ignore)
    return QuantizationModifier(scheme=scheme, targets=targets, ignore=ignore)


def load_train_dataset(dataset_path: str, tokenizer, max_samples: int,
                       max_seq_length: int):
    """ultrachat_200k train_sft -> tokenized Dataset。"""
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=str(Path(dataset_path) / "data" / "train_sft-*.parquet"),
        split="train",
    )
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    def preprocess(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"],
                                                      tokenize=False)}
    ds = ds.map(preprocess)

    def tok(s):
        return tokenizer(s["text"], padding=False, max_length=max_seq_length,
                         truncation=True, add_special_tokens=False)
    return ds.map(tok, remove_columns=ds.column_names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="stage1.1 产出的量化模型目录")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant-json", default=None,
                    help="stage1.1 的 quant_metrics.json，用于还原量化配置")
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--targets-per-subgraph", type=int, default=1,
                    help="QAD 联合训练的连续 block 数；越大越吃显存")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "finetune_metrics.json"
    started = time.time()

    try:
        if not args.variant_json:
            raise ValueError("--variant-json 必填：需要 stage1.1 的量化配置")
        qm = json.loads(Path(args.variant_json).read_text())
        variant = qm.get("variant") or {}
        if not variant:
            raise ValueError(f"{args.variant_json} 中没有 variant 字段")
        log(f"还原 stage1.1 variant = {variant}")

        quantizer = build_quantizer(variant)
        log(f"quantizer = {type(quantizer).__name__}")

        if args.dry_run:
            metrics_path.write_text(json.dumps({
                "status": "dry_run", "variant": variant,
                "quantizer": type(quantizer).__name__}, indent=2,
                ensure_ascii=False))
            log("dry-run ok")
            return 0

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
        from llmcompressor.modifiers.qad import QADModifier

        log(f"loading quantized model {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                  trust_remote_code=True)
        ds = load_train_dataset(args.dataset_path, tokenizer,
                                args.num_samples, args.max_seq_length)
        log(f"train samples = {len(ds)}")

        qad = QADModifier(
            num_epochs=args.num_epochs,
            lr=args.lr,
            gradient_accumulation_steps=args.grad_accum,
        )

        oneshot(
            model=model,
            tokenizer=tokenizer,
            recipe=[quantizer, qad],
            dataset=ds,
            pipeline="sequential",
            sequential_targets=["Qwen2DecoderLayer"],
            sequential_targets_per_subgraph=args.targets_per_subgraph,
            propagate_error=True,
            num_calibration_samples=args.num_samples,
            max_seq_length=args.max_seq_length,
            batch_size=1,
            output_dir=str(out),
        )

        model.save_pretrained(str(out), save_compressed=True)
        tokenizer.save_pretrained(str(out))

        metrics_path.write_text(json.dumps({
            "status": "ok",
            "variant": variant,
            "hyperparams": {
                "num_epochs": args.num_epochs, "lr": args.lr,
                "grad_accum": args.grad_accum,
                "num_samples": args.num_samples,
                "max_seq_length": args.max_seq_length,
                "targets_per_subgraph": args.targets_per_subgraph,
            },
            "elapsed_sec": round(time.time() - started, 1),
        }, indent=2, ensure_ascii=False))
        log(f"微调完成，产出 {out}")
        return 0

    except Exception as exc:  # noqa: BLE001
        metrics_path.write_text(json.dumps({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - started, 1),
        }, indent=2, ensure_ascii=False))
        log(f"ERROR {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())