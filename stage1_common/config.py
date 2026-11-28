#!/usr/bin/env python3
"""
训练/评测的公共配置层 —— 单一事实来源（single source of truth）。

为什么单独一个模块：prompt 要求"工作流采用模板方法模式和策略模式，
所需信息从 config 和 strategy 里找"。这里就是 config 的代码化落地：
所有阶段共享的路径、超参、任务定义只在这里出现一次，脚本从它派生，
避免 tag / 路径 / 任务名在不同文件里漂移（这是最常见的静默失败源）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# 路径（容器内约定）
# --------------------------------------------------------------------------
MOUNT_INPUT = "/data"          # OSS 只读挂载：模型 + 训练集 + 评测缓存
MOUNT_OUTPUT = "/output"       # OSS 读写挂载：产物 + 指标


@dataclass
class EvalSpec:
    """评测口径。改这里 = 改所有阶段的评测结论可比性。"""
    tasks: list[str] = field(default_factory=lambda: ["gsm8k"])
    # few-shot 按任务区分：gsm8k 用 5-shot 是论文默认口径
    num_fewshot: dict[str, int] = field(default_factory=lambda: {"gsm8k": 5})
    limit: int | None = None      # None = 全量；冒烟时设小值
    batch_size: str = "auto"
    max_model_len: int = 4096
    dtype: str = "auto"           # 绝不钉 bfloat16：T4/A10 上会失败


@dataclass
class LoraSpec:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)  # 空 = 由代码按架构推断
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainSpec:
    max_length: int = 1024
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 5
    save_strategy: str = "no"
    packing: bool = False
    gradient_checkpointing: bool = True
    bf16: bool = False           # T4(sm75) 无 bf16；A10 有，但统一用 fp16 保可比
    fp16: bool = True
    optim: str = "adamw_torch"
    seed: int = 42


@dataclass
class TaskConfig:
    """一次实验的完整配置。"""
    base_model_name: str = "Qwen2.5-7B-Instruct"
    train_dataset_name: str = "gsm8k_sft"
    # 训练集在这里是 jsonl/parquet 的 HF datasets 目录；用 text 字段
    dataset_text_field: str = "text"
    eval: EvalSpec = field(default_factory=EvalSpec)
    lora: LoraSpec = field(default_factory=LoraSpec)
    train: TrainSpec = field(default_factory=TrainSpec)

    @property
    def base_model_path(self) -> str:
        return f"{MOUNT_INPUT}/{self.base_model_name}"

    @property
    def train_dataset_path(self) -> str:
        return f"{MOUNT_INPUT}/{self.train_dataset_name}"

    @property
    def bench_cache_dir(self) -> str:
        return f"{MOUNT_INPUT}/bench_hf_cache"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _infer_target_modules(model) -> list[str]:
    """
    按模型架构推断 LoRA 目标模块，而不是硬编码 "all-linear"。

    为什么：Qwen2 系列的线性层名是 q_proj/k_proj/v_proj/o_proj/gate_proj/
    up_proj/down_proj。peft 的 target_modules="all-linear" 在部分版本/架构上
    会连带把 lm_head 也包进去，导致 LoRA 参数量暴涨且合并时分外麻烦。
    这里显式列出，并由测试保证不为空。
    """
    cand = []
    for name, _ in model.named_modules():
        tail = name.rsplit(".", 1)[-1]
        if tail in {"q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"}:
            if tail not in cand:
                cand.append(tail)
    # 兜底：老架构可能用 c_attn / c_proj
    if not cand:
        for name, _ in model.named_modules():
            tail = name.rsplit(".", 1)[-1]
            if tail in {"c_attn", "c_proj", "c_fc"} and tail not in cand:
                cand.append(tail)
    return cand


def build_lora_config(model, spec: LoraSpec):
    """构造 peft.LoraConfig。任何 ValueError 都在这里暴露，而不是训练中途。"""
    from peft import LoraConfig

    targets = spec.target_modules or _infer_target_modules(model)
    if not targets:
        raise RuntimeError(
            "无法推断 LoRA target_modules —— 模型结构与预期不符，"
            "拒绝以空目标继续训练（会产生 0 可训练参数且不报错）"
        )
    return LoraConfig(
        r=spec.r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=targets,
        bias=spec.bias,
        task_type=spec.task_type,
    )


def load_task_config(path: str | None = None) -> TaskConfig:
    """从 JSON 覆盖默认配置（便于同一镜像跑不同任务）。"""
    cfg = TaskConfig()
    if path and os.path.exists(path):
        raw = json.loads(Path(path).read_text())
        if "eval" in raw:
            cfg.eval = EvalSpec(**raw["eval"])
        if "lora" in raw:
            cfg.lora = LoraSpec(**raw["lora"])
        if "train" in raw:
            cfg.train = TrainSpec(**raw["train"])
        for k in ("base_model_name", "train_dataset_name", "dataset_text_field"):
            if k in raw:
                setattr(cfg, k, raw[k])
    return cfg


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}