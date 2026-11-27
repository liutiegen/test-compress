#!/usr/bin/env bash
# 通用 entrypoint —— stage1.1 / 1.2 / 1.3 三个镜像共用一个骨架，
# 通过 SUBCOMMAND 环境变量选择行为。
#
# 设计原则：
#   - 任何一步失败都立刻退出（set -euo pipefail），让 Pod 进入 Error 状态，
#     由 stage3 的 kubectl 侧脚本捕获，而不是"假装成功"往下跑。
#   - 所有产物写到 $OUTPUT_ROOT（OSS 挂载点），Pod 结束即销毁也无所谓。
#   - 结果 JSON 一定写，不管成功失败 —— 否则汇总阶段无法区分"失败"和"没跑"。
set -euo pipefail

log() { echo "[entrypoint] $*" >&2; }

OUTPUT_ROOT="${OUTPUT_ROOT:-/output}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-0.5B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/data/datasets/ultrachat_200k}"
MATRIX_ID="${MATRIX_ID:-unknown}"
SUBCOMMAND="${SUBCOMMAND:-quant_eval}"

log "SUBCOMMAND=$SUBCOMMAND MATRIX_ID=$MATRIX_ID"
log "MODEL_PATH=$MODEL_PATH"
log "DATASET_PATH=$DATASET_PATH"
log "OUTPUT_ROOT=$OUTPUT_ROOT"

# ---- 前置检查：挂载点存在且可写 ----
for p in "$MODEL_PATH" "$DATASET_PATH"; do
  if [[ ! -d "$p" ]]; then
    log "FATAL: 路径不存在 $p（OSS PV 未正确挂载？）"
    exit 10
  fi
done
mkdir -p "$OUTPUT_ROOT/$MATRIX_ID"
if ! touch "$OUTPUT_ROOT/$MATRIX_ID/.write_test" 2>/dev/null; then
  log "FATAL: 输出目录不可写 $OUTPUT_ROOT/$MATRIX_ID"
  exit 11
fi
rm -f "$OUTPUT_ROOT/$MATRIX_ID/.write_test"

log "GPU 信息:"
python - <<'PY' || log "WARNING: 无法读取 GPU 信息"
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device:", p.name, "cap:", f"sm{torch.cuda.get_device_capability()[0]}"
          f"{torch.cuda.get_device_capability()[1]}", "mem_GB:",
          round(p.total_memory / 1024**3, 1))
PY

RUN_DIR="$OUTPUT_ROOT/$MATRIX_ID"

case "$SUBCOMMAND" in
  quant_eval)
    log "=== 阶段1: 量化 ==="
    python /workspace/stage1_1_quantize/quantize.py \
      --model-path "$MODEL_PATH" \
      --dataset-path "$DATASET_PATH" \
      --output-dir "$RUN_DIR/model" \
      --algorithm "${ALGORITHM:?ALGORITHM required}" \
      --scheme "${SCHEME:?SCHEME required}" \
      --target "${TARGET:?TARGET required}" \
      --matrix-id "$MATRIX_ID" \
      --num-calibration-samples "${NUM_CALIBRATION_SAMPLES:-256}" \
      --max-seq-length "${MAX_SEQ_LENGTH:-2048}"
    log "=== 阶段2: 评估量化模型 ==="
    python /workspace/stage1_1_quantize/evaluate.py \
      --model-path "$RUN_DIR/model" \
      --output-dir "$RUN_DIR" \
      --label "$MATRIX_ID" \
      --limit "${EVAL_LIMIT:-}" || true
    # eval 允许失败（记录在 eval_metrics.json 里），但 quant 失败必须让 Pod 失败
    ;;

  eval_only)
    log "=== 仅评估（用于 baseline / finetuned） ==="
    python /workspace/stage1_1_quantize/evaluate.py \
      --model-path "${EVAL_MODEL_PATH:-$MODEL_PATH}" \
      --output-dir "$RUN_DIR" \
      --label "$MATRIX_ID" \
      --limit "${EVAL_LIMIT:-}"
    ;;

  finetune)
    log "=== QAD 微调 ==="
    python /workspace/stage1_2_finetune/finetune_qad.py \
      --model-path "${QUANT_MODEL_PATH:?QUANT_MODEL_PATH required}" \
      --dataset-path "$DATASET_PATH" \
      --output-dir "$RUN_DIR/model" \
      --variant-json "${VARIANT_JSON:?VARIANT_JSON required}" \
      --num-epochs "${FT_EPOCHS:-1}" \
      --lr "${FT_LR:-2e-6}" \
      --num-samples "${FT_SAMPLES:-512}"
    log "=== 评估微调后模型 ==="
    python /workspace/stage1_1_quantize/evaluate.py \
      --model-path "$RUN_DIR/model" \
      --output-dir "$RUN_DIR" \
      --label "$MATRIX_ID" \
      --limit "${EVAL_LIMIT:-}" || true
    ;;

  *)
    log "FATAL: 未知 SUBCOMMAND=$SUBCOMMAND"
    exit 12
    ;;
esac

log "=== 完成，产物在 $RUN_DIR ==="
ls -la "$RUN_DIR" || true