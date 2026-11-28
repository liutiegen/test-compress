#!/usr/bin/env bash
# ============================================================================
# 通用校验器 —— 把"每个校验项"统一成可执行、可入清单的检查
#
# 设计：所有校验都是有序的 (id, 描述, 命令) 三元组，失败即记录并可阻断。
#   输出：一张 Markdown 清单 + 机器可读 JSON，供 stage 门禁消费。
#
# 用法: ./checklist.sh <phase> [--json <out.json>] [--md <out.md>]
#   phase ∈ stage1 | stage2 | stage3
# 退出码: 0 全部通过；3 有 FAIL（阻断）
# ============================================================================
set -uo pipefail
PHASE="${1:?usage: checklist.sh <stage1|stage2|stage3>}"
shift
JSON_OUT=""; MD_OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json) JSON_OUT="$2"; shift 2 ;;
        --md)   MD_OUT="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CTX="$ROOT/artifacts/ctx.env"
[[ -f "$CTX" ]] && set -a && . "$CTX" && set +a

RESULTS=()
NFAIL=0
NWARN=0

# check <id> <desc> <severity:block|warn> <cmd...>
check() {
    local id="$1" desc="$2" sev="$3"; shift 3
    local out rc
    out=$("$@" 2>&1); rc=$?
    local status="PASS"
    if [[ $rc -ne 0 ]]; then
        if [[ "$sev" == "block" ]]; then status="FAIL"; NFAIL=$((NFAIL+1));
        else status="WARN"; NWARN=$((NWARN+1)); fi
    fi
    # JSON 转义
    local esc_out esc_desc
    esc_out=$(printf '%s' "$out" | head -c 600 | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')
    esc_desc=$(printf '%s' "$desc" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')
    RESULTS+=("{\"id\":\"$id\",\"desc\":$esc_desc,\"severity\":\"$sev\",\"status\":\"$status\",\"exit\":$rc,\"detail\":$esc_out}")
    printf '[%-4s] %-38s %s\n' "$status" "$id" "$desc"
    if [[ "$status" != "PASS" ]]; then
        printf '       -> %s\n' "$(printf '%s' "$out" | head -4 | tr '\n' ' ' | cut -c1-300)"
    fi
}

# ---------------- 校验体 ----------------
c_local_tools() {
    local miss=()
    for t in git uv ossutil gh aliyun kubectl jq curl; do
        command -v "$t" >/dev/null 2>&1 || miss+=("$t")
    done
    [[ ${#miss[@]} -eq 0 ]] || { echo "missing: ${miss[*]}"; return 1; }
    echo "all present"
}
c_no_docker_local() {
    # 本地禁止拉镜像：docker/crictl 不应存在
    if command -v docker >/dev/null 2>&1 || command -v crictl >/dev/null 2>&1; then
        echo "docker/crictl present -> 违反 forbidden_ops"; return 1
    fi
    echo "no docker/crictl locally (as designed)"
}
c_aliyun_identity() {
    local id
    id=$(aliyun sts get-caller-identity 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["UserId"])' 2>/dev/null)
    [[ -n "$id" ]] || { echo "无法获取身份"; return 1; }
    echo "uid=$id"; [[ "$id" == "${EXPECT_UID:-202791490665844832}" ]] || { echo "uid 不匹配"; return 1; }
}
c_balance() {
    local amt
    amt=$(aliyun bssopenapi query-account-balance 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["Data"]["AvailableAmount"])')
    echo "balance=${amt} CNY"
    python3 -c "import sys;sys.exit(0 if float('$amt')>${MIN_BALANCE:-30} else 1)" || {
        echo "余额低于 ${MIN_BALANCE:-30} CNY"; return 1; }
}
c_oss_bucket() {
    ossutil ls "oss://${OSS_BUCKET}/" -e "${OSS_ENDPOINT}" >/dev/null 2>&1 || { echo "bucket 不可访问"; return 1; }
    echo "oss://${OSS_BUCKET} reachable (${OSS_ENDPOINT})"
}
c_model_in_oss() {
    local n
    n=$(ossutil ls "oss://${OSS_BUCKET}/${MODEL_NAME}/" -e "${OSS_ENDPOINT}" 2>/dev/null | grep -c 'model.safetensors')
    [[ "$n" -ge 1 ]] || { echo "未找到 ${MODEL_NAME}/model.safetensors"; return 1; }
    local sz
    sz=$(ossutil ls "oss://${OSS_BUCKET}/${MODEL_NAME}/" -e "${OSS_ENDPOINT}" 2>/dev/null | awk '/model-[0-9]+-of-|model\.safetensors/{s+=$3}END{print s}')
    echo "model present, ~$(( ${sz:-0} / 1073741824 )) GiB"
}
c_dataset_in_oss() {
    local n
    n=$(ossutil ls "oss://${OSS_BUCKET}/${TRAIN_DATASET_NAME}/" -e "${OSS_ENDPOINT}" 2>/dev/null | grep -c '\.parquet')
    [[ "$n" -ge 1 ]] || { echo "训练集 parquet 缺失"; return 1; }
    echo "train parquet files: $n"
}
c_bench_in_oss() {
    local n
    n=$(ossutil ls "oss://${OSS_BUCKET}/${BENCH_CACHE_PREFIX}/" -e "${OSS_ENDPOINT}" 2>/dev/null | grep -c 'gsm8k.*\.arrow')
    [[ "$n" -ge 1 ]] || { echo "gsm8k arrow 缺失"; return 1; }
    echo "gsm8k arrow files: $n"
}
c_acr_subscription() {
    local out
    out=$(aliyun cr list-artifact-subscription-rule --instance-id "$ACR_INSTANCE" --region "$ACR_REGION" 2>/dev/null)
    local doms
    doms=$(printf '%s' "$out" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(" ".join(r.get("SourceDomain","") for r in d.get("Rules",[])))')
    echo "rules: $doms"
    for want in $AVAILABLE_SOURCE_REGISTRIES; do
        case " $doms " in *" $want "*) ;; *) echo "缺少源 $want"; return 1 ;; esac
    done
}
c_image_tag_exists() {
    local tag="$1" out
    out=$(aliyun cr list-repo-tag --instance-id "$ACR_INSTANCE" --repo-id "$ACR_REPO_ID" --region "$ACR_REGION" 2>/dev/null)
    printf '%s' "$out" | python3 -c "import json,sys;d=json.load(sys.stdin);tags=[i['Tag'] for i in d.get('Images',[])];sys.exit(0 if '$tag' in tags else 1)" \
        || { echo "tag '$tag' 不在 ACR"; return 1; }
    echo "tag '$tag' present"
}
c_cluster_running() {
    local st
    st=$(aliyun cs GET "/clusters/${CLUSTER_ID}" --region "$CLUSTER_REGION" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("state",""))')
    [[ "$st" == "running" ]] || { echo "cluster state=$st"; return 1; }
    echo "cluster ${CLUSTER_ID} running"
}
c_kubectl_works() {
    kubectl get ns >/dev/null 2>&1 || { echo "kubectl 不可用"; return 1; }
    echo "kubectl ok: $(kubectl version -o json 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["serverVersion"]["gitVersion"])' 2>/dev/null)"
}
c_kubeconfig_points_at() {
    local cur
    cur=$(kubectl config current-context 2>/dev/null)
    echo "context=$cur"
    [[ "$cur" == *"$CLUSTER_ID"* ]] || { echo "kubeconfig 未指向 ${CLUSTER_ID}"; return 1; }
}
c_gpu_nodes_ready() {
    local n
    n=$(kubectl get nodes -l "node.kubernetes.io/instance-type=${GPU_INSTANCE_TYPE}" --no-headers 2>/dev/null | grep -c ' Ready')
    [[ "$n" -ge "${GPU_NODES_REQUIRED:-1}" ]] || { echo "Ready GPU nodes=$n < ${GPU_NODES_REQUIRED:-1}"; return 1; }
    echo "Ready GPU nodes: $n"
}
c_gpu_resource_advertised() {
    local n
    n=$(kubectl get nodes -o json 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
c=0
for i in d.get("items",[]):
    a=i.get("status",{}).get("allocatable",{})
    if "nvidia.com/gpu" in a: c+=1
print(c)')
    [[ "$n" -ge 1 ]] || { echo "无节点 advertise nvidia.com/gpu"; return 1; }
    echo "nodes advertising nvidia.com/gpu: $n"
}
c_no_nodes_when_idle() {
    local n
    n=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    [[ "$n" -eq 0 ]] || { echo "仍有 $n 个节点未释放"; return 1; }
    echo "0 nodes (all released)"
}
c_nodepool_zero() {
    local d
    d=$(aliyun cs GET "/clusters/${CLUSTER_ID}/nodepools/${NODEPOOL_ID}" --region "$CLUSTER_REGION" 2>/dev/null \
        | python3 -c 'import json,sys;print(json.load(sys.stdin).get("scaling_group",{}).get("desired_size"))')
    [[ "$d" == "0" ]] || { echo "desired_size=$d (应为 0)"; return 1; }
    echo "nodepool desired_size=0"
}
c_no_ecs_left() {
    local n
    n=$(aliyun ecs describe-instances --region "$CLUSTER_REGION" --biz-region-id "$CLUSTER_REGION" 2>/dev/null \
        | python3 -c 'import json,sys;print(len(json.load(sys.stdin).get("Instances",{}).get("Instance",[])))')
    [[ "$n" -eq 0 ]] || { echo "仍有 $n 台 ECS"; return 1; }
    echo "0 ECS instances"
}

# ---------------- 各阶段清单 ----------------
case "$PHASE" in
  stage1)
    check "S1-01" "本地必备 CLI 齐全" block c_local_tools
    check "S1-02" "本地无 docker/crictl（禁止本地拉镜像）" block c_no_docker_local
    check "S1-03" "阿里云身份正确" block c_aliyun_identity
    check "S1-04" "账户余额充足" block c_balance
    check "S1-05" "OSS bucket 可访问" block c_oss_bucket
    check "S1-06" "OSS 中存在基座模型" block c_model_in_oss
    check "S1-07" "OSS 中存在训练数据集" block c_dataset_in_oss
    check "S1-08" "OSS 中存在评测集缓存" block c_bench_in_oss
    check "S1-09" "代码静态语法通过" block bash -c "python3 '$ROOT/tools/verify_syntax.py'"
    check "S1-10" "依赖版本窗口自洽" warn bash -c "python3 '$ROOT/tools/check_dep_window.py'"
    ;;
  stage2)
    check "S2-01" "ACR 制品订阅源含可用 registry" block c_acr_subscription
    check "S2-02" "base 镜像已在 ACR" block c_image_tag_exists "$BASE_TAG"
    check "S2-03" "目标镜像已在 ACR" block c_image_tag_exists "$TARGET_TAG"
    ;;
  stage3)
    check "S3-01" "集群处于 running" block c_cluster_running
    check "S3-02" "kubeconfig 指向目标集群" block c_kubeconfig_points_at
    check "S3-03" "kubectl 可执行" block c_kubectl_works
    check "S3-04" "目标镜像已在 ACR" block c_image_tag_exists "$TARGET_TAG"
    check "S3-05" "GPU 节点 Ready 且 advertise gpu" block c_gpu_resource_advertised
    ;;
  stage3-cleanup)
    check "S3-C1" "节点池 desired_size=0" block c_nodepool_zero
    check "S3-C2" "集群内无节点" block c_no_nodes_when_idle
    check "S3-C3" "无残留 ECS" block c_no_ecs_left
    ;;
  *) echo "unknown phase $PHASE" >&2; exit 1 ;;
esac

# ---------------- 输出 ----------------
if [[ -n "$JSON_OUT" ]]; then
    printf '{"phase":"%s","checks":[%s],"fail":%d,"warn":%d}\n' \
        "$PHASE" "$(IFS=,; echo "${RESULTS[*]}")" "$NFAIL" "$NWARN" > "$JSON_OUT"
fi
if [[ -n "$MD_OUT" ]]; then
    {
        echo "# 校验清单 — $PHASE"
        echo
        echo "生成时间: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo
        echo "| 状态 | ID | 校验项 | 严重级 |"
        echo "|---|---|---|---|"
        for r in "${RESULTS[@]}"; do
            st=$(printf '%s' "$r" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["status"])')
            id=$(printf '%s' "$r" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["id"])')
            de=$(printf '%s' "$r" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["desc"])')
            sv=$(printf '%s' "$r" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["severity"])')
            echo "| $st | $id | $de | $sv |"
        done
        echo
        echo "FAIL=$NFAIL  WARN=$NWARN"
    } > "$MD_OUT"
fi

echo "---- FAIL=$NFAIL WARN=$NWARN"
[[ $NFAIL -eq 0 ]] || exit 3
exit 0