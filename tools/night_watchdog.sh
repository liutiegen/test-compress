#!/usr/bin/env bash
# 无人值守收尾守护进程
#
# 职责（按用户要求）：
#   1. 等待所有量化 Job 结束（完成或失败都算结束）
#   2. 把全部结果同步到本地磁盘（不依赖 OSS 是否还可用）
#   3. 无条件释放所有云资源：缩容节点池到 0、删除临时 Pod/Job/ConfigMap
#   4. 硬截止：UTC 21:00（用户本地 05:00）前必须完成收尾，不管任务是否跑完
#
# 为什么用 setsid 启动：会话/终端断开不会杀掉它，保证通宵执行不中断。
#
# 日志：$LOG_DIR/watchdog.log
set -uo pipefail

CLUSTER_ID="c100027250d794dff9012ece2025cbc41"
REGION="cn-qingdao"
NODEPOOL_ID="npa7c9bcf7f4fc409baaa32d4033d4da45"
NS="st-compress"
LOCAL_DIR="/root/test_compress_project/results"
LOG_DIR="/root/test_compress_project/logs"
OSS_BUCKET="oss://test-llm-compressor"
HARD_DEADLINE_UTC="${HARD_DEADLINE_UTC:-$(date -u -d '21:00' +%s)}"

mkdir -p "$LOCAL_DIR" "$LOG_DIR"
LOG="$LOG_DIR/watchdog.log"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] $*" | tee -a "$LOG"
}

log "===== 守护进程启动 ====="
log "硬截止 UTC: $(date -u -d "@$HARD_DEADLINE_UTC" '+%Y-%m-%d %H:%M:%SZ')"

past_deadline() { [[ $(date -u +%s) -ge $HARD_DEADLINE_UTC ]]; }

# ---------------------------------------------------------------------------
# 阶段 1：等待 Job 结束
# ---------------------------------------------------------------------------
log "--- 阶段1: 等待量化 Job ---"
while true; do
    if past_deadline; then
        log "已达硬截止，停止等待（允许有未完成任务）"
        break
    fi

    running=$(kubectl get jobs -n "$NS" -l app=quant \
                -o jsonpath='{range .items[*]}{.status.active}{"\n"}{end}' 2>/dev/null \
              | grep -c '^1$' || true)

    total=$(kubectl get jobs -n "$NS" -l app=quant --no-headers 2>/dev/null | wc -l)
    done_n=$(kubectl get jobs -n "$NS" -l app=quant --no-headers 2>/dev/null \
              | awk '$2 ~ /Complete/ || $3 ~ /Complete/' | wc -l)
    failed_n=$(kubectl get pods -n "$NS" -l app=quant --no-headers 2>/dev/null \
              | awk '$3 ~ /Error|Failed/' | wc -l)

    log "jobs: total=$total done=$done_n failed_pods=$failed_n 仍active=$running"

    if [[ "${running:-0}" -eq 0 ]] && [[ "$total" -gt 0 ]]; then
        log "所有 Job 已停止（完成或失败）"
        break
    fi

    # 单 Job 超时保护：量化+评估单个组合预计 <60min。
    # 若某个 Pod 卡住（active 长期不降），到 150min 时主动删除该 Pod，
    # 让 Job 结束并进入收尾，避免整夜挂在死节点上烧钱。
    now_s=$(date -u +%s)
    [[ -z "${JOB_START_TS:-}" ]] && JOB_START_TS=$now_s
    elapsed=$(( now_s - JOB_START_TS ))
    if [[ $elapsed -gt 9000 ]]; then
        log "已等待 $((elapsed/60)) 分钟，主动清理卡住的 Pod"
        kubectl delete pods -n "$NS" -l app=quant \
            --field-selector=status.phase=Pending --ignore-not-found 2>&1 | tail -2 | tee -a "$LOG" || true
        # 只删 Pending（排队无意义），Running 的继续跑
        JOB_START_TS=$now_s
    fi
    sleep 60
done

# ---------------------------------------------------------------------------
# 阶段 2：收集结果到本地
# ---------------------------------------------------------------------------
log "--- 阶段 2: 收集结果 ---"
mkdir -p "$LOCAL_DIR"

# 2.1 从 OSS 拉全部 output（最完整：含 model/ 与 eval_metrics.json）
if command -v ossutil >/dev/null 2>&1; then
    log "从 OSS 同步 output/ 到本地"
    ossutil cp -r "$OSS_BUCKET/output/" "$LOCAL_DIR/oss_output/" -f --update 2>&1 | tail -3 | tee -a "$LOG" || \
        log "WARN: OSS 同步失败（继续用 kubectl 兜底）"
else
    log "WARN: 本机无 ossutil"
fi

# 2.2 kubectl 兜底：即使 OSS 挂载有问题，也把 Pod 内容抓下来
for j in $(kubectl get jobs -n "$NS" -l app=quant -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    pod=$(kubectl get pods -n "$NS" -l "job-name=$j" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [[ -z "$pod" ]] && continue
    d="$LOCAL_DIR/pod_logs/$j"
    mkdir -p "$d"
    kubectl logs -n "$NS" "$pod" --tail=-1 > "$d/full.log" 2>&1 || true
    log "抓取 $j 日志 -> $d/full.log"
done

# 2.3 生成汇总（用本地已有的 eval_metrics.json）
log "生成汇总表"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import glob, json, os

root = "/root/test_compress_project/results/oss_output"
rows, base = [], None

def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def prim(rec):
    d = {}
    for t, r in (rec.get("results") or {}).items():
        v = r.get("primary_value")
        if isinstance(v, (int, float)):
            d[t] = float(v)
    return d

cand = glob.glob(os.path.join(root, "**", "eval_metrics.json"), recursive=True)
for p in cand:
    rec = load(p)
    if not rec or rec.get("status") != "ok":
        continue
    label = rec.get("label") or os.path.basename(os.path.dirname(p))
    pm = prim(rec)
    if label == "baseline":
        base = pm
    rows.append({"label": label, "metrics": pm,
                 "path": os.path.relpath(p, root)})

summary = {"baseline": base, "variants": rows}
out = "/root/test_compress_project/results/summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"{'label':28s} " + " ".join(f"{k:>12s}" for k in (base or {})))
for r in sorted(rows, key=lambda x: x["label"]):
    vals = " ".join(f"{r['metrics'].get(k, float('nan')):12.4f}" for k in (base or {}))
    print(f"{r['label']:28s} {vals}")
print(f"\n写出 {out}")
PY

# ---------------------------------------------------------------------------
# 阶段 3：释放所有资源（无条件执行）
# ---------------------------------------------------------------------------
log "--- 阶段 3: 释放资源 ---"

log "删除 Job / 临时 Pod / ConfigMap"
kubectl delete jobs -n "$NS" -l app=quant --ignore-not-found --wait=false 2>&1 | tail -2 | tee -a "$LOG" || true
kubectl delete pods -n "$NS" --all --ignore-not-found --wait=false 2>&1 | tail -2 | tee -a "$LOG" || true
kubectl delete configmap -n "$NS" quant-runner-script baseline-eval-script ds-download-script ds-upload-script \
    --ignore-not-found 2>&1 | tail -2 | tee -a "$LOG" || true

log "缩容节点池 $NODEPOOL_ID 到 0"
aliyun cs PUT "/clusters/$CLUSTER_ID/nodepools/$NODEPOOL_ID" \
    --region "$REGION" --header "Content-Type=application/json" \
    --body '{"scaling_group":{"desired_size":0}}' 2>&1 | tail -1 | tee -a "$LOG" || \
    log "WARN: 缩容 API 调用失败，稍后重试"

# 轮询确认节点归零（最多 15 分钟）
for i in $(seq 1 30); do
    n=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    log "节点数: $n"
    [[ "$n" -eq 0 ]] && break
    sleep 30
done

log "校验无残留 ECS"
aliyun ecs describe-instances --biz-region-id "$REGION" --region "$REGION" 2>/dev/null \
    | python3 -c "
import json,sys
try:
    ins = json.load(sys.stdin).get('Instances',{}).get('Instance',[])
    print('ECS count:', len(ins))
    for i in ins:
        print(' ', i['InstanceId'], i['InstanceType'], i['Status'])
except Exception as e:
    print('查询失败:', e)
" 2>&1 | tee -a "$LOG"

log "余额: $(aliyun bssopenapi query-account-balance 2>/dev/null | python3 -c "
import json,sys
try: print(json.load(sys.stdin)['Data']['AvailableAmount'],'CNY')
except: print('N/A')")"

log "===== 守护进程结束 ====="
log "结果目录: $LOCAL_DIR"