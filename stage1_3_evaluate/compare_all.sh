#!/usr/bin/env bash
# stage1.3 —— 系统性地横向对比 baseline / 各量化变体 / 微调后模型的 benchmark 表现。
#
# 与 stage1.1 的 evaluate.py 的区别：
#   stage1.1 的评估是"每个变体各自在自己的 Pod 里评"，结果分散在各目录；
#   stage1.3 是"等所有变体就绪后，用一个统一的部署把三者拉到同一口径下比较"，
#   避免不同 Pod 因 batch_size / max_model_len 差异导致不可比。
#
# 本脚本假设输入目录已由 OSS 挂载就绪：
#   /output/baseline/             baseline 评估结果（已存在或本次生成）
#   /output/<matrix_id>/         各变体（含 model/ 与 eval_metrics.json）
#   /output/best/                 选优结果 selection.json
#
# 产出：
#   /output/stage3_compare/summary.json     横向对比表
#   /output/stage3_compare/summary.md       人类可读报告
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/output}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-0.5B-Instruct}"
REPORT_DIR="${OUTPUT_ROOT}/stage3_compare"
mkdir -p "$REPORT_DIR"

log() { echo "[stage1_3] $*" >&2; }

log "轮询 baseline 与各变体结果是否齐备"
python - <<'PY'
import glob, json, os, sys, time

root = os.environ.get("OUTPUT_ROOT", "/output")
deadline = time.time() + int(os.environ.get("WAIT_TIMEOUT", "7200"))
required = ["baseline"]
extra = [d for d in glob.glob(os.path.join(root, "*"))
         if os.path.isdir(d) and os.path.basename(d) not in
         ("baseline", "stage3_compare", "output")]
required += [os.path.basename(d) for d in extra]

def ready(name):
    p = os.path.join(root, name, "eval_metrics.json")
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p)).get("status") == "ok"
    except Exception:
        return False

while time.time() < deadline:
    missing = [r for r in required if not ready(r)]
    if not missing:
        print(f"全部就绪: {required}")
        sys.exit(0)
    print(f"等待中，尚未就绪: {missing}")
    time.sleep(60)
print(f"超时，仍未就绪: {[r for r in required if not ready(r)]}", file=sys.stderr)
sys.exit(3)
PY

log "生成横向对比"
python - <<'PY'
import glob, json, os
from pathlib import Path

root = Path(os.environ.get("OUTPUT_ROOT", "/output"))
out = root / "stage3_compare"
out.mkdir(parents=True, exist_ok=True)

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def primary(rec):
    d = {}
    for t, r in (rec.get("results") or {}).items():
        v = r.get("primary_value")
        if isinstance(v, (int, float)):
            d[t] = float(v)
    return d

base_rec = load(root / "baseline" / "eval_metrics.json")
base = primary(base_rec) if base_rec else {}
if not base:
    raise SystemExit("baseline 缺失，无法横向对比")

rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    if d.name in ("baseline", "stage3_compare"):
        continue
    rec = load(d / "eval_metrics.json")
    if not rec or rec.get("status") != "ok":
        continue
    cur = primary(rec)
    common = sorted(set(base) & set(cur))
    if not common:
        continue
    per = {t: {"baseline": round(base[t], 4), "model": round(cur[t], 4),
               "delta_pp": round((cur[t] - base[t]) * 100, 3)} for t in common}
    mean_delta = sum(v["delta_pp"] for v in per.values()) / len(per)
    rows.append({
        "label": d.name,
        "mean_delta_pp": round(mean_delta, 3),
        "worst_delta_pp": round(min(v["delta_pp"] for v in per.values()), 3),
        "tasks": per,
    })

rows.sort(key=lambda r: -r["mean_delta_pp"])
summary = {"baseline_tasks": sorted(base), "models": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

lines = ["# 模型横向对比", "",
         f"baseline 任务: {', '.join(sorted(base))}", "",
         "| 模型 | 平均 Δ(pp) | 最差 Δ(pp) |",
         "|---|---|---|"]
for r in rows:
    lines.append(f"| {r['label']} | {r['mean_delta_pp']:+.3f} | {r['worst_delta_pp']:+.3f} |")
lines += ["", "注: Δ 为相对 baseline 的百分点差值，负值表示退化。"]
(out / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
PY

log "完成，报告在 $REPORT_DIR"