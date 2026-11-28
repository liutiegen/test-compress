#!/usr/bin/env bash
# ============================================================================
# 容器入口 —— 策略分派（strategy pattern）
#
# 一个镜像承载三个策略，由 MODE 选择：
#   MODE=baseline   评原始模型
#   MODE=sft        LoRA SFT 微调
#   MODE=finetuned  评合并后的微调模型
#   MODE=compare    生成前后对比报告
#
# 之所以一个镜像多策略：三个策略共享同一套依赖（transformers/trl/lm_eval），
# 拆成三个镜像会让 base 层多存 ~10GB x3，而 ACR 跨地域拉取是本管线的
# 主要时间成本。版本一致性也更可靠。
# ============================================================================
set -euo pipefail

MODE="${MODE:-baseline}"
echo "===== MODE=${MODE} ====="
echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "data: $(ls /data 2>/dev/null | head -20 | tr '\n' ' ')"
echo "output writable: $(touch /output/.wtest 2>/dev/null && echo yes && rm -f /output/.wtest || echo NO)"

case "$MODE" in
  baseline)
    exec python3 /workspace/stage1_3_evaluate/evaluate.py
    ;;
  sft)
    exec python3 /workspace/stage1_2_finetune/train_sft.py
    ;;
  finetuned)
    : "${MODEL_PATH:?MODE=finetuned 需要 MODEL_PATH 指向 merged_model}"
    export MODEL_PATH
    exec python3 /workspace/stage1_3_evaluate/evaluate.py
    ;;
  compare)
    exec python3 /workspace/stage1_3_evaluate/compare.py
    ;;
  *)
    echo "ERROR: unknown MODE=$MODE (baseline|sft|finetuned|compare)" >&2
    exit 1
    ;;
esac