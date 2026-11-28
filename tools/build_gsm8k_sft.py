#!/usr/bin/env python3
"""
把 GSM8K 转成"有监督微调用"的 SFT 格式。

为什么需要这一步：
  GSM8K 原始数据是 {question, answer}，answer 里带 "#### 42" 的答案标记。
  直接拿 answer 当训练目标，模型会学到输出中间推理 + #### 格式，
  但 chat 模板下应该包成 messages / 或纯 text。

本脚本产出两种字段，让训练脚本二选一（当前用 text）：
  text      : 已经套好 Qwen chat template 的完整字符串
  messages  : [{"role":"user",...},{"role":"assistant",...}] 原始结构

关键质量控制：
  - 丢弃 reasoning 为空或答案解析失败的样本（否则等于教模型学噪声）
  - 校验答案可解析出来，解析失败率高说明数据源有问题，直接报错
  - 记录样本长度分布，超 max_length 的比例要报出来（截断会丢答案）

用法（在**远程节点**上跑，本地禁止下载大文件）：
  python3 build_gsm8k_sft.py --out-dir /root/gsm8k_sft --split train
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ANSWER_RE = re.compile(r"####\s*([\-0-9\.,]+)")
# Qwen2.5 的默认 system prompt（与 eval 时用的一致，避免训练/推理分布漂移）
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Solve the math problem step by step, "
    "then give the final numeric answer after ####."
)


def parse_answer(answer: str) -> tuple[str, str] | None:
    """
    拆成 (reasoning, final)。失败返回 None。
    GSM8K 的标准格式是推理 + 换行 + "#### 数字"。
    """
    if not isinstance(answer, str) or "####" not in answer:
        return None
    m = ANSWER_RE.search(answer)
    if not m:
        return None
    final = m.group(1).replace(",", "").strip()
    reasoning = answer[: m.start()].strip()
    if not reasoning or not final:
        return None
    try:
        float(final)
    except ValueError:
        return None
    return reasoning, final


def build(records, tokenizer=None):
    """返回 (rows, stats)。tokenizer 可选：有则统计长度。"""
    rows, bad, lengths = [], 0, []
    for rec in records:
        q = (rec.get("question") or "").strip()
        parsed = parse_answer(rec.get("answer") or "")
        if not q or parsed is None:
            bad += 1
            continue
        reasoning, final = parsed
        assistant = f"{reasoning}\n\n#### {final}"

        if tokenizer is not None:
            msgs = [{"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": assistant}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
        else:
            # 离线构造等价文本（tokenizer 不可用时用），格式与 Qwen2.5 模板一致
            text = (f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
                    f"<|im_start|>user\n{q}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant}<|im_end|>\n")

        rows.append({
            "text": text,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant},
            ],
            "question": q,
            "final_answer": final,
        })
        if tokenizer is not None:
            lengths.append(len(tokenizer(text)["input_ids"]))

    stats = {"total_in": len(records), "kept": len(rows), "dropped_unparsable": bad}
    if lengths:
        lengths.sort()
        n = len(lengths)
        stats["tokens"] = {
            "min": lengths[0], "p50": lengths[n // 2], "p95": lengths[int(n * 0.95)],
            "max": lengths[-1], "mean": round(sum(lengths) / n, 1),
        }
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-dir", default="", help="可选：用它的 tokenizer 统计长度")
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"[build] loading gsm8k split={args.split}", flush=True)
    ds = load_dataset("gsm8k", "main", split=args.split)
    print(f"[build] raw rows={len(ds)}", flush=True)

    tokenizer = None
    if args.model_dir and Path(args.model_dir).exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        print(f"[build] tokenizer loaded from {args.model_dir}", flush=True)

    rows, stats = build(ds, tokenizer)

    # 质量控制：解析失败率不应超过 1%，否则说明数据源格式与预期不符
    fail_rate = stats["dropped_unparsable"] / max(stats["total_in"], 1)
    print(f"[build] stats={json.dumps(stats, ensure_ascii=False)}", flush=True)
    if fail_rate > 0.01:
        print(f"[build] FATAL: 解析失败率 {fail_rate:.2%} > 1%，数据源异常", file=sys.stderr)
        return 2

    if "tokens" in stats:
        over = sum(1 for r in rows
                   if len(tokenizer(r["text"])["input_ids"]) > args.max_length)
        stats["over_max_length"] = over
        stats["over_max_length_ratio"] = round(over / max(len(rows), 1), 4)
        print(f"[build] 超过 max_length={args.max_length} 的样本: {over} "
              f"({stats['over_max_length_ratio']:.1%})", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 存成 parquet：训练脚本按 parquet 读，避免依赖 datasets 的 arrow 缓存格式
    from datasets import Dataset
    Dataset.from_list(rows).to_parquet(str(out / "train.parquet"))
    (out / "build_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build] wrote {out}/train.parquet rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())