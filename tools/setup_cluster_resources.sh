#!/usr/bin/env bash
# 部署前的一次性准备：创建 PV/PVC 与两个 Secret。
#
# 为什么单独一个脚本：
#   凭证（OSS AK、ACR 密码）绝不进 git 仓库，所以 k8s yaml 里只有引用关系。
#   这个脚本负责从本机已有的配置文件读取并按需创建，让"yaml 可公开"
#   和"部署可复现"同时成立。
#
# 用法:
#   OSS_AK_ID=... OSS_AK_SECRET=... \
#   ACR_USER=... ACR_PASSWORD=... \
#   ./tools/setup_cluster_resources.sh
#
# 若不提供环境变量，则尝试从 /root/.ossutilconfig 读取 OSS 凭证。
set -euo pipefail

NAMESPACE="${NAMESPACE:-st-compress}"
ACR_REGISTRY="${ACR_REGISTRY:-test-acr-business-registry.cn-shanghai.cr.aliyuncs.com}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo "[setup] $*" >&2; }

# ---- OSS AK：优先环境变量，其次 ossutilconfig ----
if [[ -z "${OSS_AK_ID:-}" || -z "${OSS_AK_SECRET:-}" ]]; then
    CFG="${OSSUTIL_CONFIG:-/root/.ossutilconfig}"
    if [[ -f "$CFG" ]]; then
        OSS_AK_ID="$(grep -E '^accessKeyID' "$CFG" | cut -d= -f2- | tr -d ' ')"
        OSS_AK_SECRET="$(grep -E '^accessKeySecret' "$CFG" | cut -d= -f2- | tr -d ' ')"
        log "从 $CFG 读取 OSS 凭证"
    fi
fi
if [[ -z "${OSS_AK_ID:-}" || -z "${OSS_AK_SECRET:-}" ]]; then
    log "FATAL: 缺少 OSS_AK_ID / OSS_AK_SECRET，且无法从 ossutilconfig 读取"
    exit 1
fi

# ---- 1. 命名空间 ----
kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || {
    log "创建 namespace $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
}

# ---- 2. PV / PVC / StorageClass ----
log "apply pv_stage_1_1.yaml"
kubectl apply -f "$HERE/stage1_1_quantize/k8s/pv_stage_1_1.yaml"
log "apply pvc_stage_1_1.yaml"
kubectl apply -f "$HERE/stage1_1_quantize/k8s/pvc_stage_1_1.yaml"

# ---- 3. OSS 凭证 Secret ----
log "创建 oss-ak-secret（幂等）"
kubectl create secret generic oss-ak-secret -n "$NAMESPACE" \
    --from-literal=akId="$OSS_AK_ID" \
    --from-literal=akSecret="$OSS_AK_SECRET" \
    --dry-run=client -o yaml | kubectl apply -f -

# ---- 4. ACR 拉取凭证 ----
if [[ -n "${ACR_USER:-}" && -n "${ACR_PASSWORD:-}" ]]; then
    log "创建 acr-cred（幂等）"
    # 必须用 create secret docker-registry，不要手写 base64：
    # 手工拼的 auth 字段用户名极易写错，表现为 kubelet 401 Unauthorized
    kubectl create secret docker-registry acr-cred -n "$NAMESPACE" \
        --docker-server="$ACR_REGISTRY" \
        --docker-username="$ACR_USER" \
        --docker-password="$ACR_PASSWORD" \
        --dry-run=client -o yaml | kubectl apply -f -
else
    log "WARNING: 未提供 ACR_USER/ACR_PASSWORD，跳过 acr-cred（拉镜像会失败）"
fi

# ---- 5. 校验绑定 ----
log "PVC 绑定状态："
kubectl get pvc -n "$NAMESPACE"

for p in pvc-stage-1-1-input pvc-stage-1-1-quant; do
    phase="$(kubectl get pvc "$p" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo Missing)"
    if [[ "$phase" != "Bound" ]]; then
        log "FATAL: $p 未 Bound（当前 $phase）"
        log "常见原因：PV 上残留旧 claimRef —— 用以下命令清除后重试："
        log "  kubectl patch pv <pv-name> --type=json -p '[{\"op\":\"remove\",\"path\":\"/spec/claimRef\"}]'"
        exit 2
    fi
done

log "全部就绪"