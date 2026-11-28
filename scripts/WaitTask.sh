#!/usr/bin/env bash
# 等待 aliyun 任务（节点池创建/扩容/删除、集群删除等）结束
# 用法: WaitTask.sh --repeat-command 'aliyun cs describe-task-info --task-id T-xxx --region cn-shanghai' \
#          --condition '"state": "failed"|"state": "success"' --poll-interval 30 --max-time 3000
set -uo pipefail
REPEAT_COMMAND=""; CONDITION=""; POLL_INTERVAL=5; MAX_TIME=300
START_TS=$(date +%s)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat-command) REPEAT_COMMAND="$2"; shift 2 ;;
        --condition)      CONDITION="$2"; shift 2 ;;
        --poll-interval)  POLL_INTERVAL="$2"; shift 2 ;;
        --max-time)       MAX_TIME="$2"; shift 2 ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$REPEAT_COMMAND" && -n "$CONDITION" ]] || {
    echo "Usage: $0 --repeat-command <cmd> --condition <ere> [--poll-interval s] [--max-time s]" >&2; exit 1; }
while true; do
    NOW_TS=$(date +%s)
    if (( NOW_TS - START_TS >= MAX_TIME )); then
        echo "ERROR: timeout after ${MAX_TIME}s, condition [$CONDITION] not matched" >&2; exit 2
    fi
    CMD_OUT=$(eval "$REPEAT_COMMAND" 2>&1)
    if printf '%s\n' "$CMD_OUT" | grep -qE -- "$CONDITION"; then
        printf '%s\n' "$CMD_OUT"; exit 0
    fi
    sleep "$POLL_INTERVAL"
done