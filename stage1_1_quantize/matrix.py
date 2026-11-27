#!/usr/bin/env python3
"""
量化实验矩阵定义 —— 单一事实来源 (single source of truth)。

三个维度：
  algorithm  量化算法  : smoothquant | awq | gptq | rtn
  scheme     量化方案  : W8A8 | W4A16 | W4A16_ASYM | FP8_DYNAMIC
  target     量化位置  : mlp | attn | linear_all | linear_all+kv

matrix_id 形如  sq__w8a8__mlp，用于命名镜像 tag / k8s 资源 / 输出目录，
保证三者严格一一对应（stage1 的 necessary_asset 要求）。

硬件门控（HARDWARE_GATE）来自 vLLM 源码 compressed-tensors scheme 的
get_min_capability()，不是猜测：
  W8A8           -> 75 (compressed_tensors_w8a8_int8.py)
  W4A16/W4A16_ASYM -> 75 (compressed_tensors_wNa16.py)
  FP8_DYNAMIC    -> 89 (compressed_tensors_w8a8_fp8.py)
GPU 算力低于门槛的组合会在 plan 阶段被剔除并记入 excluded。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterator

# --------------------------------------------------------------------------
# 维度定义
# --------------------------------------------------------------------------

# 算法 -> llm-compressor 中对应的 transform modifier（rtn 无 transform）
ALGORITHMS: dict[str, str | None] = {
    "smoothquant": "SmoothQuantModifier",
    "awq": "AWQModifier",
    "gptq": None,          # GPTQ 是量化器本身，不是 transform
    "rtn": None,           # 纯 RTN，无 transform
}

# 方案 -> (preset scheme 名, 最低 GPU 算力, 说明)
SCHEMES: dict[str, tuple[str, int, str]] = {
    "W8A8":         ("W8A8",         75, "int8 权重+激活, 需 sm75+"),
    "W4A16":        ("W4A16",        75, "int4 权重/对称, fp16 激活, 需 sm75+"),
    "W4A16_ASYM":   ("W4A16_ASYM",   75, "int4 权重/非对称, 需 sm75+"),
    "FP8_DYNAMIC":  ("FP8_DYNAMIC",  89, "fp8 权重+动态激活, 需 sm89+"),
}

# 量化位置 -> (targets 列表, 人类可读说明)
# 层名依据 transformers/models/qwen2/modeling_qwen2.py:
#   Qwen2Attention: q_proj / k_proj / v_proj / o_proj
#   Qwen2MLP     : gate_proj / up_proj / down_proj
TARGETS: dict[str, tuple[list[str], str]] = {
    "mlp": (
        [r"re:.*\.mlp\.(gate|up|down)_proj$"],
        "仅 MLP 三个投影层",
    ),
    "attn": (
        [r"re:.*\.self_attn\.(q|k|v|o)_proj$"],
        "仅 self-attention 四个投影层",
    ),
    "linear_all": (
        ["Linear"],
        "全部 Linear（含 attn+mlp，排除 lm_head）",
    ),
    "linear_all_kv": (
        ["Linear"],
        "全部 Linear + KV cache 量化",
    ),
}

# 允许的算法/方案配对：有些组合语义上没意义或必然失败
#   AWQ 需要 per-channel 权重量化，仅对 int4/int8 weight-only 类有效
#   SmoothQuant 是为 activation outlier 设计的，针对带激活量化的方案
ALGO_SCHEME_ALLOWED: dict[str, set[str]] = {
    "smoothquant": {"W8A8", "FP8_DYNAMIC"},
    "awq":         {"W4A16", "W4A16_ASYM", "W8A8"},
    "gptq":        {"W4A16", "W4A16_ASYM", "W8A8", "FP8_DYNAMIC"},
    "rtn":         {"W4A16", "W4A16_ASYM", "W8A8", "FP8_DYNAMIC"},
}

SHORT = {
    "smoothquant": "sq",
    "awq": "awq",
    "gptq": "gptq",
    "rtn": "rtn",
    "W8A8": "w8a8",
    "W4A16": "w4a16",
    "W4A16_ASYM": "w4a16a",
    "FP8_DYNAMIC": "fp8dyn",
    "mlp": "mlp",
    "attn": "attn",
    "linear_all": "linall",
    "linear_all_kv": "linallkv",
}


@dataclass(frozen=True)
class Variant:
    algorithm: str
    scheme: str
    target: str

    @property
    def matrix_id(self) -> str:
        return f"{SHORT[self.algorithm]}__{SHORT[self.scheme]}__{SHORT[self.target]}"

    @property
    def preset_scheme(self) -> str:
        return SCHEMES[self.scheme][0]

    @property
    def min_capability(self) -> int:
        return SCHEMES[self.scheme][1]

    @property
    def targets(self) -> list[str]:
        return TARGETS[self.target][0]

    @property
    def use_kv_cache(self) -> bool:
        return self.target == "linear_all_kv"

    def describe(self) -> str:
        return (f"{self.algorithm} + {self.scheme} @ {self.target} "
                f"-> {self.matrix_id}")


def all_variants() -> Iterator[Variant]:
    """笛卡尔积：algorithm x scheme x target（未过滤）。"""
    for algo, scheme, target in itertools.product(
        ALGORITHMS, SCHEMES, TARGETS
    ):
        yield Variant(algo, scheme, target)


# --------------------------------------------------------------------------
# 选定子集（4 个组合）
#
# 为什么只跑 4 个：余额 133.55，节点 ¥0.87/h，每组合需量化 + 评估两趟，
# 完整 52 组合会耗尽余额且无重试余量。这 4 个构成三个维度的单变量对照，
# 选出来的"最优组合"有可比性，而不是只挑一个跑：
#
#   #1 awq/W4A16/linear_all      ┐ 同方案同位置，比算法
#   #2 gptq/W4A16/linear_all     ┘
#   #3 sq/W8A8/linear_all        ─ 比方案（8bit vs 4bit）
#   #4 awq/W4A16/mlp             ─ 比位置（与 #1 同算法同方案）
#
# 注意：集群实际只有 T4(V100 不可用)，sm75 刚好卡在 W8A8/W4A16 门槛上，
# 故 FP8_DYNAMIC(sm89) 不在子集内。
# --------------------------------------------------------------------------
SELECTED_IDS: list[str] = [
    "awq__w4a16__linall",
    "gptq__w4a16__linall",
    "sq__w8a8__linall",
    "awq__w4a16__mlp",
]


def selected_variants() -> list[Variant]:
    """按 SELECTED_IDS 返回选定的 4 个变体（顺序即 SELECTED_IDS 顺序）。"""
    allv = {v.matrix_id: v for v in all_variants()}
    missing = [i for i in SELECTED_IDS if i not in allv]
    if missing:
        raise ValueError(f"SELECTED_IDS 含未知 matrix_id: {missing}")
    return [allv[i] for i in SELECTED_IDS]


def plan(gpu_capability: int,
         only_selected: bool = True) -> tuple[list[Variant], list[tuple[Variant, str]]]:
    """
    按 GPU 算力 + 算法/方案配对规则过滤，返回 (included, excluded)。

    only_selected=True 时只保留 SELECTED_IDS 的 4 个组合，
    其余全部进 excluded 并附带"未入选子集"的原因。

    excluded 是 (variant, 原因) 列表，便于写进报告里向人解释为什么没跑。
    """
    included: list[Variant] = []
    excluded: list[tuple[Variant, str]] = []

    keep = set(SELECTED_IDS) if only_selected else None

    for v in all_variants():
        if v.scheme not in ALGO_SCHEME_ALLOWED[v.algorithm]:
            excluded.append(
                (v, f"算法/方案不匹配: {v.algorithm} 不适用于 {v.scheme}")
            )
            continue
        if v.min_capability > gpu_capability:
            excluded.append(
                (v, f"GPU 算力不足: {v.scheme} 需 sm{v.min_capability}, "
                    f"当前 sm{gpu_capability}")
            )
            continue
        if keep is not None and v.matrix_id not in keep:
            excluded.append(
                (v, "未入选本次子集（预算限制，只跑 4 个对照组合）")
            )
            continue
        included.append(v)

    return included, excluded


def baseline_variant() -> Variant:
    """对照组：不量化也能"通过"的锚点。用于计算相对退化。"""
    return Variant("rtn", "W4A16", "mlp")


if __name__ == "__main__":
    import sys

    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 89
    inc, exc = plan(cap)
    print(f"GPU capability sm{cap}")
    print(f"included: {len(inc)}  excluded: {len(exc)}")
    print("\n--- INCLUDED ---")
    for v in inc:
        print(" ", v.describe())
    print("\n--- EXCLUDED ---")
    for v, why in exc:
        print(" ", v.matrix_id, "|", why)