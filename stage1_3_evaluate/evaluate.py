#!/usr/bin/env python3
"""
统一评测脚本 —— baseline 与 finetuned 走同一条代码路径，保证数字可比。

为什么不用 lm-eval 的 --model vllm：
  1. vllm 后端要求把模型地址和 adapter 分开传，对"合并后模型"与
     "原始模型"两种形态反而多一层映射；
  2. lm-eval 的 vllm 后端在多版本间 API 变动大（我们镜像里是 lm_eval 0.4.x）。
改用 --model hf（transformers 后端），显式关掉 4bit/8bit，dtype=auto。
代价是比 vllm 慢，但评测集是 gsm8k 全量 1319 条，单卡 A10 完全可接受，
换来的是一次跑通、结果可复现。

评测口径（关键，写在输出里）：
  - gsm8k, 5-shot, 全量 test split
  - 指标取 exact_match（strict-match 与 flexible-extract 都记录）
  - 生成参数用 lm-eval 默认（temperature=0 / greedy）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace")
from stage1_common.config import MOUNT_INPUT, MOUNT_OUTPUT, TaskConfig, load_task_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_lm_eval(model_path: str, cfg: TaskConfig, out_dir: Path, label: str) -> dict:
    """
    调 lm_eval CLI。返回解析后的指标 dict。

    注意 lm_eval 的 CLI 参数名在 0.4.x 里是 --model_args / --tasks /
    --num_fewshot / --output_path / --batch_size / --limit，
    且 --output_path 传目录时它自己在里面建 json。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据集缓存：lm-eval 从 HF datasets 拉。Pod 无外网，所以
    # 必须让 HF_DATASETS_CACHE 指向预先灌好的 OSS 挂载目录。
    env = dict(os.environ)
    env["HF_DATASETS_CACHE"] = f"{cfg.bench_cache_dir}/ds_cache"
    env["HF_HOME"] = f"{cfg.bench_cache_dir}/hf_home"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", (
            f"pretrained={model_path},dtype={cfg.eval.dtype},"
            f"trust_remote_code=False,max_length={cfg.eval.max_model_len}"
        ),
        "--tasks", ",".join(cfg.eval.tasks),
        "--batch_size", str(cfg.eval.batch_size),
        "--output_path", str(out_dir / "results"),
        "--log_samples",
    ]
    # few-shot 只支持全局一个值；多任务若 fewshot 不同则需分批跑。
    # 目前 tasks 只有 gsm8k，取其值即可。
    fs = {cfg.eval.num_fewshot.get(t, 0) for t in cfg.eval.tasks}
    if len(fs) == 1:
        cmd += ["--num_fewshot", str(fs.pop())]
    else:
        raise ValueError(f"多任务 fewshot 不一致，需分批: {cfg.eval.num_fewshot}")
    if cfg.eval.limit:
        cmd += ["--limit", str(cfg.eval.limit)]

    log(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    (out_dir / "lm_eval.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "lm_eval.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        raise RuntimeError(f"lm_eval 退出码 {proc.returncode}\n{tail}")

    # lm-eval 把结果写成 <output_path>/<model_name>/results_<ts>.json
    cands = sorted(out_dir.glob("results*/**/results_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        cands = sorted(Path("/tmp").glob("**/results_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError("lm_eval 未产出 results_*.json")

    raw = json.loads(cands[-1].read_text())
    results = raw.get("results", {})

    summary = {"label": label, "model_path": model_path, "elapsed_sec": round(elapsed, 1),
               "raw_result_file": str(cands[-1]), "tasks": {},
               # 口径字段：compare.py 用它拒绝比较不同口径的结果
               "tasks_requested": list(cfg.eval.tasks),
               "num_fewshot": dict(cfg.eval.num_fewshot),
               "limit": cfg.eval.limit,
               "dtype": cfg.eval.dtype}
    for task, m in results.items():
        keep = {k: v for k, v in m.items() if not k.endswith("_stderr")}
        stderr = {k: v for k, v in m.items() if k.endswith("_stderr")}
        summary["tasks"][task] = {"metrics": keep, "stderr": stderr,
                                  "num_samples": m.get("alias") and m.get("samples") or m.get("samples")}
    return summary


def main() -> int:
    cfg = load_task_config(os.environ.get("TASK_CONFIG"))
    label = os.environ.get("RUN_NAME", "baseline")
    model_override = os.environ.get("MODEL_PATH")   # 微调后指向 merged_model
    model_path = model_override or cfg.base_model_path

    out_dir = Path(MOUNT_OUTPUT) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status = {"label": label, "model_path": model_path, "status": "running"}
    write_json(out_dir / "eval_status.json", status)

    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        log(f"model={model_path} tasks={cfg.eval.tasks} fewshot={cfg.eval.num_fewshot}")

        summary = run_lm_eval(model_path, cfg, out_dir / "lm_eval", label)

        # 展平成 primary 指标，方便比较表
        primary = {}
        for task, d in summary["tasks"].items():
            m = d["metrics"]
            # gsm8k: exact_match,strict-match 与 exact_match,flexible-extract
            for key in ("exact_match,strict-match", "exact_match,flexible-extract",
                        "acc,none", "acc_norm,none"):
                if key in m:
                    primary[f"{task}:{key}"] = m[key]
        summary["primary"] = primary

        status.update({"status": "ok", "summary": summary,
                       "elapsed_sec": round(time.time() - t0, 1)})
        write_json(out_dir / "eval_metrics.json", summary)
        write_json(out_dir / "eval_status.json", status)
        log(f"primary metrics: {json.dumps(primary)}")
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        status.update({"status": "failed", "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-4000:],
                       "elapsed_sec": round(time.time() - t0, 1)})
        write_json(out_dir / "eval_status.json", status)
        return 1


if __name__ == "__main__":
    sys.exit(main())