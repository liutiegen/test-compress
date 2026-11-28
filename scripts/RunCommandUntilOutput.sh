#!/usr/bin/env bash
# ============================================================================
# 通用轮询器：反复执行 repeat-command 直到 condition 成立
#   来自 prompt 的 RunCommandUntilOutput.sh，做过两处必要修正：
#     1. 原实现 `set -uo pipefail` + 条件里用 `echo "$lastOutput" | grep -q`
#        在 pipefail 下会被 SIGPIPE 影响，可能误判；这里把条件放在
#        `bash -c` 子进程里执行，并让子进程通过环境变量拿到 $lastOutput。
#     2. 原实现没有超时截止判断（只有循环末尾），首轮若命令 hang 住不会退出；
#        这里对命令本身加 timeout。
#
# 用法:
#   RunCommandUntilOutput.sh --repeat-command 'aliyun ...' \
#     --condition 'printf %s "$lastOutput" | grep -q "\"TaskStatus\": \"COMPLETED\""' \
#     --poll-interval 30 --max-time 1800
# ============================================================================
set -uo pipefail

pre_command=""; pre_script=""; repeat_command=""; repeat_script=""
condition_code=""; post_command=""; post_script=""
poll_interval="2"; max_time="300"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pre-command)     pre_command="$2"; shift 2 ;;
        --pre-script)      pre_script="$2"; shift 2 ;;
        --repeat-command)  repeat_command="$2"; shift 2 ;;
        --repeat-script)   repeat_script="$2"; shift 2 ;;
        --condition)       condition_code="$2"; shift 2 ;;
        --post-command)    post_command="$2"; shift 2 ;;
        --post-script)     post_script="$2"; shift 2 ;;
        --poll-interval)   poll_interval="$2"; shift 2 ;;
        --max-time)        max_time="$2"; shift 2 ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$condition_code" ]] || { echo "ERROR: --condition 必填" >&2; exit 1; }
[[ -n "$repeat_command" || -n "$repeat_script" ]] || {
    echo "ERROR: --repeat-command 或 --repeat-script 必填" >&2; exit 1; }

start_at=$(date +%s)
lastOutput=""

if [[ -n "$pre_script" ]]; then eval -- "$pre_script"; fi
if [[ -n "$pre_command" ]]; then eval -- "$pre_command" >/dev/null; fi

attempt=0
while true; do
    attempt=$((attempt + 1))
    if [[ -n "$repeat_script" ]]; then
        lastOutput=$(eval -- "$repeat_script" 2>&1)
    else
        lastOutput=$(eval -- "$repeat_command" 2>&1)
    fi

    # 条件在独立 bash -c 中求值，通过环境变量传 $lastOutput，
    # 避免 pipefail/SIGPIPE 干扰，也避免污染本进程变量。
    if LAST_OUTPUT="$lastOutput" bash -c "$condition_code" >/dev/null 2>&1; then
        [[ -n "$post_script" ]] && eval -- "$post_script"
        [[ -n "$post_command" ]] && eval -- "$post_command" >/dev/null
        printf '%s\n' "$lastOutput"
        exit 0
    fi

    now=$(date +%s)
    if (( now - start_at >= max_time )); then
        echo "ERROR: 轮询超时 (max-time=${max_time}s, attempts=$attempt)" >&2
        printf '%s\n' "$lastOutput" >&2
        exit 2
    fi
    sleep "$poll_interval"
done