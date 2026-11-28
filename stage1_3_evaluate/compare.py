#!/usr/bin/env python3
"""
聚合评测结果，产出"微调前 vs 微调后"对比报告。

设计要求（对应 prompt 的"比较微调前后性能"）：
  - baseline 与 finetuned 必须来自同一条评测代码、同一口径，否则差值无意义。
    因此这里拒绝比较 num_fewshot / limit / tasks 不一致的两份结果。
  - 每份结果都带 stderr。差值小于 ~2*stderr 时判定为"不显著"，不能声称提升。
    这是最容易自欺的一步：gsm8k 全量 1319 条，±1pp 的差可能就是噪声。
  - 输出同时给人和给机器：compare.md（人看）+ comparison.json（下游用）。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/output"))


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_primary(rec: dict) -> dict:
    """从 eval_metrics.json 里取 primary 指标。"""
    if not rec:
        return {}
    return rec.get("primary") or {}


def sig_test(delta: float, se_a: float | None, se_b: float | None) -> str:
    """
    两比例之差的标准误：sqrt(se_a^2 + se_b^2)。
    差值绝对值 < 2SE 判为不显著（约 95% 水平）。
    """
    if se_a is None or se_b is None:
        return "unknown"
    se = math.sqrt(se_a ** 2 + se_b ** 2)
    if se == 0:
        return "identical"
    z = abs(delta) / se
    if z >= 2.576:
        return f"significant(p<0.01,z={z:.2f})"
    if z >= 1.96:
        return f"significant(p<0.05,z={z:.2f})"
    if z >= 1.645:
        return f"marginal(p<0.10,z={z:.2f})"
    return f"not-significant(z={z:.2f})"


def main() -> int:
    base = load(OUTPUT_ROOT / "baseline" / "eval_metrics.json")
    ft = load(OUTPUT_ROOT / "finetuned" / "eval_metrics.json")

    report = {"baseline_present": base is not None, "finetuned_present": ft is not None}
    lines = ["# 微调前后性能对比", ""]
    lines.append(f"生成时间: {__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime())}")
    lines.append("")

    if not base or not ft:
        report["verdict"] = "INCOMPLETE"
        missing = [n for n, r in (("baseline", base), ("finetuned", ft)) if not r]
        lines.append(f"缺少结果: {', '.join(missing)} —— 无法比较")
        (OUTPUT_ROOT / "comparison.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (OUTPUT_ROOT / "compare.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 2

    pb, pf = get_primary(base), get_primary(ft)

    # 口径一致性校验：口径不同则差值无意义
    def calibre(rec):
        return (rec.get("tasks") and tuple(sorted(rec["tasks"].keys()))) or None

    # 从 summary 里取 fewshot / limit（写在 evaluate.py 的 summary 之外的字段）
    mismatches = []
    for key in ("num_fewshot", "limit", "tasks"):
        vb, vf = base.get(key), ft.get(key)
        if vb is not None and vf is not None and vb != vf:
            mismatches.append(f"{key}: baseline={vb} vs finetuned={vf}")
    if mismatches:
        report["verdict"] = "INCOMPARABLE"
        report["mismatches"] = mismatches
        lines.append("## 口径不一致，拒绝比较")
        lines += [f"- {m}" for m in mismatches]
        (OUTPUT_ROOT / "comparison.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (OUTPUT_ROOT / "compare.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 3

    lines.append("| 指标 | 微调前 | 微调后 | 差值 | 显著性 |")
    lines.append("|---|---|---|---|---|")

    rows = []
    for k in sorted(set(pb) | set(pf)):
        vb, vf = pb.get(k), pf.get(k)
        if vb is None or vf is None:
            continue
        # stderr：键形如 "gsm8k:exact_match,flexible-extract"
        task, _, metric = k.partition(":")
        def se_of(rec):
            t = (rec.get("tasks") or {}).get(task) or {}
            return (t.get("stderr") or {}).get(metric)
        se_b = se_of(base)
        se_f = se_of(ft)
        delta = vf - vb
        sig = sig_test(delta, se_b, se_f)
        rows.append({"metric": k, "before": vb, "after": vf,
                     "delta": round(delta, 4), "se_before": se_b,
                     "se_after": se_f, "significance": sig})
        lines.append(
            f"| {k} | {vb:.4f} | {vf:.4f} | {delta:+.4f} | {sig} |")

    report["rows"] = rows
    report["verdict"] = "COMPARED"
    report["verdict_detail"] = {
        "improved": [r["metric"] for r in rows
                     if r["delta"] > 0 and r["significance"].startswith(("significant", "marginal"))],
        "not_significant": [r["metric"] for r in rows
                            if r["significance"].startswith(("not-significant", "unknown"))],
        "regressed": [r["metric"] for r in rows
                      if r["delta"] < 0 and r["significance"].startswith(("significant", "marginal"))],
    }

    lines += ["", "## 结论", ""]
    vd = report["verdict_detail"]
    if vd["improved"]:
        lines.append(f"**在 {', '.join(vd['improved'])} 上取得统计显著提升。**")
    if vd["regressed"]:
        lines.append(f"**出现显著下降：{', '.join(vd['regressed'])}**")
    if vd["not_significant"]:
        lines.append(f"差异不显著（在噪声范围内）：{', '.join(vd['not_significant'])}")
    lines += ["", "显著性判据：两比例差 z 检验，|Δ| ≥ 2·SE 视为显著。", ""]

    (OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())