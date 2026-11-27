#!/usr/bin/env python3
"""
stage1.1 汇总选优 —— 从所有 variant 的评估结果中选出最优
(算法, 方案, 位置) 组合。

选优规则（工程上必须显式、可复现、能解释）：
  1. 硬门槛: 任一任务相对 baseline 的绝对退化 > MAX_DROP_PP(默认 1.0 个百分点)
     或相对退化 > MAX_DROP_REL(默认 3%) => 该 variant 直接淘汰。
     这条对应需求里的"使得在<指定 benchmark>上的表现能稳住"。
  2. 在通过门槛的 variant 里，按 score 排序：
        score = mean(相对 baseline 的保留率) - compression_penalty
     其中保留率 = variant_primary / baseline_primary（越大越好）。
     compression_penalty 让同等精度下更省显存的方案胜出：
        penalty = 0.02 * (1 - bytes_ratio)
     bytes_ratio 用 scheme 的理论权重位宽 / 16 估算。
  3. 并列时依次比较: 更多任务胜出 > 更大压缩率 > 更小 matrix_id(确定性)。

输入: 结果根目录，期望结构
    <root>/baseline/eval_metrics.json
    <root>/<matrix_id>/eval_metrics.json
    <root>/<matrix_id>/quant_metrics.json
输出: <root>/selection.json + 打印人类可读表格
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 各 scheme 的相对位宽开销（相对 fp16=16bit），用于同等精度下偏好更省的方案
# 注意：这是 *权重* 位宽估算；W8A8/FP8 的激活也要算，故给个略保守的加权。
SCHEME_BYTES_RATIO = {
    "W8A8": 0.50,        # 8bit w + 8bit a
    "FP8_DYNAMIC": 0.50,  # 8bit w + 8bit a
    "W4A16": 0.28,       # 4bit w + 16bit a
    "W4A16_ASYM": 0.28,
}

MAX_DROP_PP = 1.0     # 绝对退化上限（百分点）
MAX_DROP_REL = 0.03   # 相对退化上限


def log(m: str) -> None:
    print(f"[select] {m}", flush=True)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def primary_map(eval_rec: dict) -> dict[str, float]:
    """{task: primary_value}，跳过 None。"""
    out = {}
    for t, r in (eval_rec.get("results") or {}).items():
        v = r.get("primary_value")
        if isinstance(v, (int, float)):
            out[t] = float(v)
    return out


def score_variant(base: dict[str, float], cur: dict[str, float],
                  scheme: str,
                  max_drop_pp: float = MAX_DROP_PP,
                  max_drop_rel: float = MAX_DROP_REL) -> dict:
    """计算保留率、退化、score、是否通过门槛。"""
    common = sorted(set(base) & set(cur))
    if not common:
        return {"comparable": False, "reason": "no overlapping tasks"}

    ratios, drops_pp, drops_rel, per_task = [], [], [], {}
    for t in common:
        b, c = base[t], cur[t]
        if b == 0:
            continue
        ratio = c / b
        drop_pp = (b - c) * 100.0
        drop_rel = (b - c) / b
        ratios.append(ratio)
        drops_pp.append(drop_pp)
        drops_rel.append(drop_rel)
        per_task[t] = {
            "baseline": round(b, 4),
            "variant": round(c, 4),
            "drop_pp": round(drop_pp, 3),
            "drop_rel_pct": round(drop_rel * 100, 3),
        }

    if not ratios:
        return {"comparable": False, "reason": "all baseline values zero"}

    mean_ratio = sum(ratios) / len(ratios)
    worst_drop_pp = max(drops_pp)
    worst_drop_rel = max(drops_rel)
    wins = sum(1 for r in ratios if r >= 1.0)

    bytes_ratio = SCHEME_BYTES_RATIO.get(scheme, 1.0)
    penalty = 0.02 * (1.0 - bytes_ratio)
    score = mean_ratio - penalty

    passed = (worst_drop_pp <= max_drop_pp) and (worst_drop_rel <= max_drop_rel)

    return {
        "comparable": True,
        "tasks": common,
        "mean_ratio": round(mean_ratio, 6),
        "worst_drop_pp": round(worst_drop_pp, 4),
        "worst_drop_rel_pct": round(worst_drop_rel * 100, 4),
        "tasks_won": wins,
        "bytes_ratio": bytes_ratio,
        "score": round(score, 6),
        "passed_threshold": passed,
        "per_task": per_task,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="结果根目录（OSS 挂载点）")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-drop-pp", type=float, default=MAX_DROP_PP)
    ap.add_argument("--max-drop-rel", type=float, default=MAX_DROP_REL)
    args = ap.parse_args()

    root = Path(args.root)
    base_rec = load(root / args.baseline_label / "eval_metrics.json")
    if not base_rec or base_rec.get("status") != "ok":
        log(f"FATAL: baseline 结果缺失或失败: "
            f"{root / args.baseline_label / 'eval_metrics.json'}")
        return 2
    base = primary_map(base_rec)
    if not base:
        log("FATAL: baseline 没有可用的主指标")
        return 2
    log(f"baseline tasks = {sorted(base)}")
    log(f"baseline = { {k: round(v,4) for k,v in sorted(base.items())} }")

    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == args.baseline_label:
            continue
        ev = load(d / "eval_metrics.json")
        if not ev or ev.get("status") != "ok":
            rows.append({"matrix_id": d.name, "status": "eval_missing_or_failed",
                         "comparable": False})
            continue
        # quant_metrics.json 由 quantize.py 写在 --output-dir 下，
        # 而 Job 里 --output-dir 是 $OUT/model，所以实际落在 model/ 子目录。
        # 两处都找，避免元信息丢失（丢了会导致 scheme=UNKNOWN、
        # 压缩率惩罚算不出来，进而影响排序）。
        qm = (load(d / "model" / "quant_metrics.json")
              or load(d / "quant_metrics.json") or {})
        variant = qm.get("variant") or ev.get("variant") or {}
        # 兼容早期产物：quant_metrics.json 里没有 target 字段。
        # matrix_id 形如 algo__scheme__target，可从中反推，避免重跑任务。
        if not variant.get("target"):
            parts = d.name.split("__")
            if len(parts) == 3:
                variant = {**variant, "_id_parts": parts}
        scheme = variant.get("scheme", "UNKNOWN")
        s = score_variant(base, primary_map(ev), scheme,
                          args.max_drop_pp, args.max_drop_rel)
        s.update({
            "matrix_id": d.name,
            "status": ev.get("status"),
            "algorithm": variant.get("algorithm"),
            "scheme": scheme,
            "target": variant.get("target") or (
                variant.get("_id_parts") or [None, None, None])[2],
            "quant_status": qm.get("status"),
        })
        rows.append(s)

    ok = [r for r in rows if r.get("comparable") and r.get("passed_threshold")]
    ok.sort(key=lambda r: (-r["score"], -r["tasks_won"], r["matrix_id"]))

    log("")
    log(f"{'matrix_id':34s} {'score':>8s} {'mean_ratio':>11s} "
        f"{'worst_drop_pp':>14s} {'pass':>5s}")
    log("-" * 78)
    for r in sorted([x for x in rows if x.get("comparable")],
                    key=lambda x: -x["score"]):
        log(f"{r['matrix_id']:34s} {r['score']:8.4f} {r['mean_ratio']:11.4f} "
            f"{r['worst_drop_pp']:14.3f} {str(r['passed_threshold']):>5s}")
    missing = [r["matrix_id"] for r in rows if not r.get("comparable")]
    if missing:
        log(f"\n无有效评估结果 ({len(missing)}): {missing}")

    result = {
        "baseline": base,
        "thresholds": {"max_drop_pp": args.max_drop_pp,
                        "max_drop_rel": args.max_drop_rel},
        "all_variants": rows,
        "passed": ok,
        "best": ok[0] if ok else None,
        "verdict": ("found" if ok else "no_variant_passed_threshold"),
    }

    out = Path(args.out) if args.out else root / "selection.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"\n写出 {out}")

    if not ok:
        log("WARNING: 没有任何 variant 通过门槛 —— 需要人工介入")
        return 3

    b = ok[0]
    log(f"\nBEST: {b['matrix_id']}  "
        f"(algorithm={b['algorithm']}, scheme={b['scheme']}, target={b['target']})")
    log(f"  score={b['score']}  mean_ratio={b['mean_ratio']}  "
        f"worst_drop={b['worst_drop_pp']}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())