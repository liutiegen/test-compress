#!/usr/bin/env python3
"""
LoRA 有监督微调（SFT）—— 让模型在指定任务类型（数学推理）上表现提升。

设计要点（每条都对应一个真实失败模式）：

1. 只训 LoRA，不训全参：7B 全参在单卡 A10 24G 上放不下优化器状态。
2. target_modules 由 config._infer_target_modules 推断并校验非空 ——
   空目标会静默产生 0 个可训练参数，"训练成功"但模型没变。
3. fp16 而非 bf16：T4(sm75) 无 bf16；A10 有，但为了与 baseline 口径一致
   且能跨卡型复现，统一 fp16。
4. 每个退出路径都写 status.json —— 否则聚合阶段无法区分
   "训练失败" 与 "从未运行"。
5. 结束后立刻 merge_and_unload 并存盘：评测走纯 transformers/vllm，
   不依赖 peft 在推理时的动态 patch（vllm 对 LoRA 的支持更受限）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/workspace")
from stage1_common.config import (  # noqa: E402
    MOUNT_OUTPUT, TaskConfig, build_lora_config, load_task_config,
)


def log(msg: str) -> None:
    print(f"[sft {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_status(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    cfg: TaskConfig = load_task_config(os.environ.get("TASK_CONFIG"))
    run_name = os.environ.get("RUN_NAME", "sft_lora")
    out_dir = Path(MOUNT_OUTPUT) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    status = {
        "run_name": run_name,
        "kind": "sft_lora",
        "base_model": cfg.base_model_path,
        "dataset": cfg.train_dataset_path,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_status(out_dir, status)

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用 —— 训练必须跑在 GPU 上")

        # ---------------- 数据 ----------------
        # 训练集已预置为 HF datasets 目录（text 字段）。不从 Hub 拉：
        # Pod 内 HF_HUB_OFFLINE=1，联网拉取必然失败。
        log(f"loading dataset from {cfg.train_dataset_path}")
        ds_files = sorted(Path(cfg.train_dataset_path).glob("*.parquet"))
        if not ds_files:
            raise FileNotFoundError(f"{cfg.train_dataset_path} 下无 parquet")
        ds = load_dataset("parquet", data_files=[str(p) for p in ds_files], split="train")
        log(f"train samples: {len(ds)}  columns: {ds.column_names}")
        if cfg.dataset_text_field not in ds.column_names:
            raise KeyError(
                f"数据集缺少字段 '{cfg.dataset_text_field}'，实际字段 {ds.column_names}")

        # ---------------- 模型 ----------------
        log(f"loading tokenizer/model from {cfg.base_model_path}")
        tok = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path,
            torch_dtype=torch.float16,
            device_map=None,          # 单卡，交给 Trainer 放置
            trust_remote_code=False,
        )
        model.config.use_cache = False

        lora_cfg = build_lora_config(model, cfg.lora)
        log(f"lora targets={lora_cfg.target_modules} r={lora_cfg.r}")

        # ---------------- 训练 ----------------
        train_args = SFTConfig(
            output_dir=str(out_dir / "trainer_out"),
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            learning_rate=cfg.train.learning_rate,
            num_train_epochs=cfg.train.num_train_epochs,
            max_steps=cfg.train.max_steps,
            warmup_ratio=cfg.train.warmup_ratio,
            lr_scheduler_type=cfg.train.lr_scheduler_type,
            logging_steps=cfg.train.logging_steps,
            save_strategy=cfg.train.save_strategy,
            packing=cfg.train.packing,
            gradient_checkpointing=cfg.train.gradient_checkpointing,
            bf16=cfg.train.bf16,
            fp16=cfg.train.fp16,
            optim=cfg.train.optim,
            seed=cfg.train.seed,
            max_length=cfg.train.max_length,
            dataset_text_field=cfg.dataset_text_field,
            report_to=[],
            dataloader_num_workers=2,
        )

        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=ds,
            peft_config=lora_cfg,
            processing_class=tok,
        )

        n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in trainer.model.parameters())
        log(f"trainable={n_trainable:,} / total={n_total:,} "
            f"({100.0 * n_trainable / n_total:.3f}%)")
        if n_trainable == 0:
            raise RuntimeError("可训练参数为 0 —— LoRA 未生效")

        train_result = trainer.train()
        metrics = dict(train_result.metrics)
        log(f"train metrics: {metrics}")

        # ---------------- 合并并保存 ----------------
        log("merging LoRA into base weights")
        merged = trainer.model.merge_and_unload()
        merged_dir = out_dir / "merged_model"
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tok.save_pretrained(str(merged_dir))

        # adapter 也单独存一份（便于对比/调试，体积小）
        adapter_dir = out_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))

        # 记录实际生效的配置，便于溯源
        (out_dir / "resolved_config.json").write_text(cfg.to_json(), encoding="utf-8")

        status.update({
            "status": "ok",
            "train_metrics": metrics,
            "trainable_params": n_trainable,
            "total_params": n_total,
            "lora_targets": list(lora_cfg.target_modules),
            "merged_model_dir": str(merged_dir),
            "elapsed_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_status(out_dir, status)
        log("done")
        return 0

    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        status.update({
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-4000:],
            "elapsed_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_status(out_dir, status)
        return 1


if __name__ == "__main__":
    sys.exit(main())