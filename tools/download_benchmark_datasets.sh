#!/usr/bin/env bash
# download_files —— 预置 benchmark 数据集到 OSS
#
# 为什么需要：lm-eval 的任务数据集（winogrande/arc/hellaswag/gsm8k）默认从
# HuggingFace Hub 拉，而 Pod 里设了 HF_HUB_OFFLINE=1（防止运行期联网），
# 直接跑必然 ConnectionError: Couldn't reach 'allenai/winogrande' (OfflineModeIsEnabled)。
#
# 办法：在有公网的节点上，用 hf-mirror.com 把数据集下载到 HF cache 布局，
# 再同步到 OSS；Pod 里设 HF_DATASETS_CACHE 指向挂载目录即可离线读取。
#
# 速度要求（对应 prompt）：下载必须走 mirror 且 >500KB/s。
# hf-mirror.com 实测可用（huggingface.co 在本环境不可达）。
#
# 用法：
#   ./tools/download_benchmark_datasets.sh --instance-id i-xxx --oss-prefix benchmark_hf_cache
set -uo pipefail

REGION="${REGION:-cn-qingdao}"
BIZ_REGION="${BIZ_REGION:-$REGION}"
INSTANCE_ID=""
OSS_BUCKET="${OSS_BUCKET:-test-llm-compressor}"
OSS_PREFIX="${OSS_PREFIX:-benchmark_hf_cache}"
BUCKET_REGION="${BUCKET_REGION:-cn-shanghai}"

# lm-eval 的任务数据集（与 evaluate.py DEFAULT_TASKS 对应）
DATASETS="${DATASETS:-allenai/winogrande AI2-ARC/ARC-Easy AI2-ARC/ARC-Challenge Rowan/hellaswag openai/gsm8k}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --oss-bucket) OSS_BUCKET="$2"; shift 2 ;;
        --oss-prefix) OSS_PREFIX="$2"; shift 2 ;;
        --region) REGION="$2"; BIZ_REGION="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[download] $*" >&2; }
[[ -n "$INSTANCE_ID" ]] || { log "FATAL: --instance-id 必填"; exit 1; }

run_short() {
    local cmd="$1"
    local inv out st
    for attempt in 1 2 3; do
        inv=$(aliyun ecs run-command --biz-region-id "$BIZ_REGION" --type RunShellScript \
                --instance-id "$INSTANCE_ID" --region "$REGION" \
                --timeout 300 --command-content "$cmd" 2>/dev/null \
              | python3 -c "import json,sys; print(json.load(sys.stdin).get('InvokeId',''))" 2>/dev/null)
        [[ -n "$inv" ]] && break
        sleep 5
    done
    [[ -z "$inv" ]] && { echo ""; return 1; }
    for _ in $(seq 1 100); do
        sleep 6
        out=$(aliyun ecs describe-invocation-results --content-encoding PlainText \
                --biz-region-id "$BIZ_REGION" --invoke-id "$inv" --region "$REGION" 2>/dev/null)
        st=$(echo "$out" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    rs=d.get('Invocation',{}).get('InvocationResults',{}).get('InvocationResult',[])
    print(rs[0].get('InvocationStatus','') if rs else '')
except Exception: print('')" 2>/dev/null)
        case "$st" in
            Success|Failed|Timeout)
                echo "$out" | python3 -c "
import json,sys
d=json.load(sys.stdin)
rs=d.get('Invocation',{}).get('InvocationResults',{}).get('InvocationResult',[])
print((rs[0].get('Output') or '').strip() if rs else '')" 2>/dev/null
                return 0 ;;
        esac
    done
    echo ""; return 1
}

# ---- 1. 确认 mirror 可达 ----
log "确认 hf-mirror 可达"
code=$(run_short 'timeout 15 curl -s -o /dev/null -w "%{http_code}" https://hf-mirror.com')
log "hf-mirror HTTP: $code"
[[ "$code" == "200" ]] || { log "FATAL: hf-mirror 不可达"; exit 2; }

# ---- 2. 在节点上装 huggingface_hub 并用 mirror 下载 ----
log "确保 huggingface_hub / datasets 可用"
run_short 'python3 -c "import datasets" 2>/dev/null || pip install -q datasets huggingface_hub' >/dev/null

DS_JSON=$(python3 -c "
import json,sys
ds='''$DATASETS'''.split()
print(json.dumps(ds))")
log "待下载数据集: $DATASETS"

# 用 hf_transfer 加速（可选）；设 HF_ENDPOINT 指向 mirror
log "开始下载（后台，写入 /root/benchmark_hf_cache）"
run_short "nohup sh -c \"HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0 \
  python3 - <<'PYEOF' > /tmp/ds_download.log 2>&1
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset
ds = $DS_JSON
for repo in ds:
    print('downloading', repo, flush=True)
    try:
        # 只下元数据+数据文件到 cache，不全部载入内存
        load_dataset(repo, split=None, trust_remote_code=False)
        print('OK', repo, flush=True)
    except Exception as e:
        print('FAIL', repo, type(e).__name__, e, flush=True)
print('ALL DONE', flush=True)
PYEOF
\" >/dev/null 2>&1 & echo started" >/dev/null

log "轮询下载完成（最多 30 分钟）..."
end=$(( $(date +%s) + 1800 )); ok=0
while [[ $(date +%s) -lt $end ]]; do
    sleep 30
    tail=$(run_short 'tail -3 /tmp/ds_download.log 2>/dev/null')
    if echo "$tail" | grep -q "ALL DONE"; then ok=1; break; fi
    log "  ...$(echo "$tail" | tail -1 | cut -c1-80)"
done
[[ $ok -eq 1 ]] || { log "FATAL: 下载超时"; run_short 'tail -20 /tmp/ds_download.log' >&2; exit 3; }

log "下载日志："
run_short 'grep -E "^(OK|FAIL)" /tmp/ds_download.log 2>/dev/null' >&2
log "缓存大小："
run_short 'du -sh /root/benchmark_hf_cache ~/.cache/huggingface 2>/dev/null' >&2

# ---- 3. 上传到 OSS（用节点上的 ossutil）----
log "检查节点是否有 ossutil"
has_ossutil=$(run_short 'command -v ossutil >/dev/null && echo yes || echo no')
if [[ "$has_ossutil" != "yes" ]]; then
    log "节点无 ossutil —— 请用本机 ossutil 上传（见下方命令）"
    log "缓存位置在节点 /root/benchmark_hf_cache 或 ~/.cache/huggingface"
    log "在本机执行：ossutil cp -r <本地缓存> oss://$OSS_BUCKET/$OSS_PREFIX/ -f"
    exit 0
fi

log "上传到 oss://$OSS_BUCKET/$OSS_PREFIX/"
run_short "ossutil cp -r /root/benchmark_hf_cache oss://$OSS_BUCKET/$OSS_PREFIX/ -f 2>&1 | tail -3" >&2

log "完成。Pod 中设置 HF_DATASETS_CACHE 指向该目录即可离线读取。"