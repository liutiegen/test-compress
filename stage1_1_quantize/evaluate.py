#!/usr/bin/env python3
"""
stage1.1 评估脚本 —— 用 lm-evaluation-harness 在 vLLM 后端上评估量化模型。

为什么用 vLLM 后端而不是 HF：
  量化产物（compressed-tensors）的真实价值只有在量化推理内核上才体现；
  用 HF eager 跑量化模型既慢又容易触发 dequant 路径，
  且 baseline 与 quantized 必须用同一后端才有可比性。

评估协议（"不让表现下滑"要靠可比的锚点）：
  - 任务: arc_challenge, arc_easy, hellaswag, winogrande, gsm8k
  - few-shot: 与 lm-eval 官方默认一致（arc/hellaswag 25-shot, winogrande 5-shot,
    gsm8k 5-shot），固定在 eval_tasks.yaml，避免不同 run 静默漂移
  - limit: 可由外部指定，用于冒烟；正式跑不设 limit
  - batch_size 与 max_model_len 必须显式给定，保证与 baseline 完全一致

输出: 统一 schema 的 eval_metrics.json，供 stage1.1 汇总脚本横向比较。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

# lm-eval 官方任务默认 few-shot，固化在这里防止漂移
DEFAULT_TASKS = {
    "arc_challenge": 25,
    "arc_easy": 25,
    "hellaswag": 10,
    "winogrande": 5,
    "gsm8k": 5,
}


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def parse_results(path: Path) -> dict:
    """lm-eval 结果 JSON -> {task: {metric: value}}，只取主指标。"""
    raw = json.loads(path.read_text())
    res = raw.get("results", {})
    out: dict[str, dict] = {}
    for task, metrics in res.items():
        clean = {}
        for k, v in metrics.items():
            if k in ("alias", "samples"):
                continue
            if isinstance(v, (int, float)):
                clean[k] = v
        # 主指标优先级
        primary = None
        for cand in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                     "exact_match,flexible-extract", "acc"):
            if cand in clean:
                primary = cand
                break
        out[task] = {"metrics": clean, "primary_metric": primary,
                     "primary_value": clean.get(primary) if primary else None}
    return {"tasks": out, "config": raw.get("config", {}),
            "versions": raw.get("versions", {})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", required=True,
                    help="baseline / <matrix_id> / finetuned")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--num-fewshot", default=None,
                    help="JSON 覆盖默认 few-shot，如 '{\"arc_easy\":0}'")
    ap.add_argument("--limit", type=float, default=None,
                    help="每个任务样本数上限（冒烟用），正式评估不要设")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None,
                    help="传给 vLLM 的 quantization，compressed-tensors 一般留空自动探测")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "eval_metrics.json"
    started = time.time()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    fewshot = dict(DEFAULT_TASKS)
    if args.num_fewshot:
        fewshot.update(json.loads(args.num_fewshot))

    try:
        # 按 few-shot 分组后逐组调用 lm_eval CLI（见下）。
        # 不用 python API 是为了避开不同 lm-eval 版本间的签名差异。

        # 按 few-shot 分组，同组一次评估；这样既保证口径准确又不重复加载模型太多次
        groups: dict[int, list[str]] = {}
        for t in tasks:
            groups.setdefault(fewshot.get(t, 5), []).append(t)
        log(f"few-shot 分组: {groups}")

        merged_tasks: dict = {}
        for shot, group_tasks in sorted(groups.items()):
            raw_dir = out / "lm_eval_raw" / f"fs{shot}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            c = [
                sys.executable, "-m", "lm_eval",
                "--model", "vllm",
                "--model_args",
                f"pretrained={args.model_path},"
                f"dtype={args.dtype},"
                f"max_model_len={args.max_model_len},"
                f"gpu_memory_utilization={args.gpu_memory_utilization},"
                f"trust_remote_code={str(args.trust_remote_code)}"
                + (f",quantization={args.quantization}" if args.quantization else ""),
                "--tasks", ",".join(group_tasks),
                "--batch_size", str(args.batch_size),
                "--seed", str(args.seed),
                "--num_fewshot", str(shot),
                "--output_path", str(raw_dir),
            ]
            if args.limit is not None:
                c += ["--limit", str(args.limit)]
            if os.environ.get("EVAL_LOG_SAMPLES") == "1":
                c += ["--log_samples"]

            log(f"评估 tasks={group_tasks} fewshot={shot}")
            proc = subprocess.run(c, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"lm_eval failed rc={proc.returncode} tasks={group_tasks}\n"
                    f"STDERR:\n{proc.stderr[-4000:]}")
            logs = sorted(raw_dir.glob("**/results_*.json"),
                          key=lambda p: p.stat().st_mtime)
            if not logs:
                raise RuntimeError(f"lm_eval 未产出 results_*.json ({group_tasks})")
            merged_tasks.update(parse_results(logs[-1])["tasks"])

        merged = {"tasks": merged_tasks}

        record = {
            "label": args.label,
            "model_path": args.model_path,
            "status": "ok",
            "backend": "vllm",
            "tasks_requested": tasks,
            "fewshot": {t: fewshot.get(t, 5) for t in tasks},
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "results": merged.get("tasks", {}),
            "elapsed_sec": round(time.time() - started, 1),
        }
        metrics_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        log(f"写出 {metrics_path}")
        for t, r in record["results"].items():
            log(f"  {t}: {r['primary_metric']}={r['primary_value']}")
        return 0

    except Exception as exc:  # noqa: BLE001
        metrics_path.write_text(json.dumps({
            "label": args.label,
            "model_path": args.model_path,
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