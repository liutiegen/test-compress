#!/usr/bin/env bash
# 创建 ACK 集群（cn-shanghai，与 OSS 同地域 -> 可走内网 endpoint）
set -uo pipefail
REGION="${REGION:-cn-shanghai}"
VPC="${VPC:-vpc-uf6dq409wdbh1iu5f4nry}"
VSW_M="${VSW_M:-vsw-uf6aam6f6r7dlkfaz6pjk}"   # cn-shanghai-m
VSW_L="${VSW_L:-vsw-uf6holdzzx32p5kduekhv}"   # cn-shanghai-l
NAME="${NAME:-cluster-training}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BODY=$(cat <<JSON
{
  "name": "$NAME",
  "cluster_type": "ManagedKubernetes",
  "kubernetes_version": "1.37.2-aliyun.1",
  "region_id": "$REGION",
  "vpcid": "$VPC",
  "vswitch_ids": ["$VSW_L", "$VSW_M"],
  "pod_vswitch_ids": ["$VSW_L", "$VSW_M"],
  "snat_entry": true,
  "endpoint_public_access": true,
  "deletion_protection": false,
  "proxy_mode": "ipvs",
  "profile": "Default",
  "cis_enable_risk_check": true,
  "timezone": "Asia/Shanghai",
  "addons": [
    {"name": "metrics-server"},
    {"name": "coredns"},
    {"name": "ack-kubeskoop"},
    {"name": "terway-controlplane", "config": "{\"ENITrunking\":\"true\"}"},
    {"name": "terway-eniip", "config": "{\"IPVlan\":\"false\",\"NetworkPolicy\":\"false\",\"ENITrunking\":\"true\"}"},
    {"name": "csi-plugin"},
    {"name": "managed-csiprovisioner"},
    {"name": "storage-operator", "config": "{\"CnfsOssEnable\":\"false\",\"CnfsNasEnable\":\"false\"}"},
    {"name": "loongcollector", "config": "{\"IngressDashboardEnabled\":\"true\"}"},
    {"name": "nginx-ingress-controller", "disabled": true},
    {"name": "ack-node-local-dns"}
  ],
  "support_policy": "Standard",
  "cluster_spec": "ack.pro.small",
  "enable_rrsa": false,
  "zone_ids": ["cn-shanghai-l", "cn-shanghai-m"],
  "service_cidr": "172.20.0.0/16",
  "ip_stack": "ipv4",
  "audit_log_config": {"enabled": false},
  "maintenance_window": {"enable": false},
  "operation_policy": {"cluster_auto_upgrade": {"enabled": false}},
  "is_enterprise_security_group": true
}
JSON
)

echo "[create-cluster] POST /clusters"
aliyun cs POST /clusters --region "$REGION" \
  --header "Content-Type=application/json" --body "$BODY" 2>&1 | tee "$ROOT/artifacts/cluster_create.json"