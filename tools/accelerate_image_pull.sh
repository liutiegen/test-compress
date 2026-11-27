#!/usr/bin/env bash
# accelerate_image_pull —— 预拉镜像并固化到数据盘快照
#
# 为什么需要：业务镜像各约 10GB，跨地域从 ACR 拉到目标集群很慢，且每个新 Pod
# 都要重拉。把镜像预先写进数据盘并打快照，后续 GPU 节点池挂这个快照后，
# containerd 启动即可命中本地镜像。
#
# 为什么用 CPU 机器：拉镜像纯 IO，不需要 GPU。在 GPU 机型（T4 ¥17/h）上做这件事
# 是纯浪费；CPU 机型不到 ¥1/h。
#
# 关键实现细节（都是踩过的坑）：
#   * `aliyun ecs run-command` 拉 10GB 必然 Timeout（默认超时太短），
#     必须 `nohup ... &` 放后台，再轮询 `crictl images` 判断完成。
#   * 校验必须看 `crictl images` 里真的出现该 tag；InvocationStatus=Success
#     只说明命令执行了，不代表拉取成功。
#   * containerd 必须落在数据盘（`findmnt /var/lib/containerd` → /dev/vdb），
#     否则快照拍的是系统盘，镜像白拉。
#   * 节点池 JSON 里 `data_disks[].encrypted` 是字符串 "false"，传 bool 会报
#     `Unmarshal type error: expected=string, got=bool`。
#
# 用法:
#   ./tools/accelerate_image_pull.sh \
#     --instance-id i-xxx \
#     --images "reg/ns/repo:tag1 reg/ns/repo:tag2" \
#     --creds "user:password"
set -uo pipefail

REGION="${REGION:-cn-qingdao}"
BIZ_REGION="${BIZ_REGION:-$REGION}"
INSTANCE_ID=""
IMAGES=""
CREDS=""
PULL_TIMEOUT="${PULL_TIMEOUT:-3600}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --images) IMAGES="$2"; shift 2 ;;
        --creds) CREDS="$2"; shift 2 ;;
        --region) REGION="$2"; BIZ_REGION="$2"; shift 2 ;;
        --pull-timeout) PULL_TIMEOUT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[accelerate] $*" >&2; }
[[ -n "$INSTANCE_ID" ]] || { log "FATAL: --instance-id 必填"; exit 1; }
[[ -n "$IMAGES" ]] || { log "FATAL: --images 必填"; exit 1; }

# 在实例上执行短命令并返回 stdout（自带重试，规避 API EOF）
run_short() {
    local cmd="$1" inv out st
    for attempt in 1 2 3; do
        inv=$(aliyun ecs run-command --biz-region-id "$BIZ_REGION" --type RunShellScript \
                --instance-id "$INSTANCE_ID" --region "$REGION" \
                --command-content "$cmd" 2>/dev/null \
              | python3 -c "import json,sys; print(json.load(sys.stdin).get('InvokeId',''))" 2>/dev/null)
        [[ -n "$inv" ]] && break
        sleep 5
    done
    [[ -z "$inv" ]] && { echo ""; return 1; }

    for _ in $(seq 1 40); do
        sleep 3
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
    echo ""
    return 1
}

# ---- 0. containerd 必须位于数据盘 ----
log "确认 containerd 存储位置"
mnt=$(run_short 'findmnt -no SOURCE /var/lib/containerd')
log "containerd device: $mnt"
case "$mnt" in
    /dev/vd*) log "OK: containerd 在数据盘上" ;;
    "") log "WARNING: 无法确认挂载点" ;;
    *) log "FATAL: containerd 不在数据盘（$mnt），快照会白做"; exit 2 ;;
esac

# ---- 1-2. 逐个后台拉取 ----
for img in $IMAGES; do
    short="${img##*/}"; name="${short%%:*}"
    log "拉取 $short"

    if run_short "crictl images 2>/dev/null | awk '{print \$1\":\"\$2}' | grep -qx '$img' && echo FOUND" | grep -q FOUND; then
        log "  已存在，跳过"
        continue
    fi

    run_short "nohup sh -c \"crictl pull --creds '$CREDS' $img > /tmp/pull_$name.log 2>&1\" >/dev/null 2>&1 & echo started" >/dev/null
    log "  后台拉取中，轮询本地镜像列表..."

    end=$(( $(date +%s) + PULL_TIMEOUT )); ok=0
    while [[ $(date +%s) -lt $end ]]; do
        sleep 30
        cnt=$(run_short "crictl images 2>/dev/null | awk '{print \$1\":\"\$2}' | grep -cx '$img' || true" | tr -dc '0-9')
        if [[ "${cnt:-0}" -ge 1 ]]; then log "  完成"; ok=1; break; fi
        log "    ...仍在拉取"
    done
    [[ $ok -eq 1 ]] || { log "FATAL: $short 拉取超时"; log " 查 /tmp/pull_$name.log"; exit 3; }
done

# ---- 最终校验：逐个 tag 真实存在 ----
log "最终校验所有 tag"
missing=""
for img in $IMAGES; do
    cnt=$(run_short "crictl images 2>/dev/null | awk '{print \$1\":\"\$2}' | grep -cx '$img' || true" | tr -dc '0-9')
    [[ "${cnt:-0}" -lt 1 ]] && missing="$missing $img"
done
[[ -n "$missing" ]] && { log "FATAL: 未找到:$missing"; exit 4; }
log "全部镜像已就位"

# ---- 3. 数据盘快照 ----
log "定位数据盘（Type=data）"
DISK_ID=$(aliyun ecs describe-disks --biz-region-id "$BIZ_REGION" \
            --instance-id "$INSTANCE_ID" --region "$REGION" 2>/dev/null \
          | python3 -c "
import json,sys
d=json.load(sys.stdin)
for k in d.get('Disks',{}).get('Disk',[]):
    if k.get('Type')=='data': print(k['DiskId']); break")
[[ -n "$DISK_ID" ]] || { log "FATAL: 未找到 data 盘"; exit 5; }
log "数据盘: $DISK_ID"

log "创建快照（耗时较长）"
SNAP_ID=$(aliyun ecs create-snapshot --disk-id "$DISK_ID" --region "$REGION" 2>/dev/null \
          | python3 -c "import json,sys; print(json.load(sys.stdin).get('SnapshotId',''))")
[[ -n "$SNAP_ID" ]] || { log "FATAL: 快照创建失败"; exit 6; }
log "快照: $SNAP_ID"

# ---- 4. 等快照完成 ----
log "等待快照完成"
WaitTask.sh --repeat-command \
    "aliyun ecs describe-snapshots --biz-region-id $BIZ_REGION --snapshot-ids '[\"$SNAP_ID\"]' --region $REGION" \
    --condition '"Status": ?"accomplished"|"Status": ?"failed"' \
    --poll-interval 30 --max-time 7200 2>&1 | tail -c 400

log ""
log "SNAPSHOT_ID=$SNAP_ID"
log "下一步：GPU 节点池设 data_disks[0].snapshot_id=$SNAP_ID，然后删除本 CPU 节点池。"