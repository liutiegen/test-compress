# 环境勘察结论（实测）

采集时间: 2026-11-27T08:33Z
采集方式: 全部为实际命令输出，非推测。

## 1. 本地环境

| 项 | 实测值 | 结论 |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS | 符合 ubuntu_24.04 |
| git / uv / ossutil / gh / aliyun / kubectl | 全部存在 | 符合 |
| docker / crictl | **不存在** | 符合"禁止本地拉镜像"，本地构建不可行 → 必须用 ACR 构建 |
| 磁盘 | 872G 可用 | 充足 |
| RAM / CPU | 11G / 16 vCPU | 只能做 CPU 级校验，不能本地跑量化 |
| 代码仓库 | /root/compress/{vllm,llm-compressor,lm-evaluation-harness,transformers,trl} 全部存在 | 符合；版本见下 |

代码仓库版本（git describe）:
- llm-compressor: `0.15.0-72-g18b10e2d7`
- transformers: `v4.56.1-Vault-Gemma-preview-4592`  ← **注意：远低于 llm-compressor 0.15.0 要求的 `>=5.24.0`**
- trl: `v1.18.0-139`
- vllm: `v0.34.1rc0-90`
- lm-evaluation-harness: `v0.4.13-11`

## 2. 云凭据与静态资源

| 项 | 实测值 | 结论 |
|---|---|---|
| aliyun AK | 有效，default profile，region=cn-shanghai | OK |
| gh | 已登录 liutiegen，scope 含 repo/workflow | OK |
| ossutil | /root/.ossutilconfig 有效，endpoint=oss-cn-shanghai | OK |
| ECS 密钥 | /root/ecs_secrets/cn_qingdao/ecs_secret.pem | **只有 cn_qingdao，cn-shanghai 无 ecs_secret 私钥** |
| 余额 | AvailableAmount = **136.85 CNY** | 硬约束 |

## 3. OSS 桶

| 项 | 实测值 |
|---|---|
| bucket | test-llm-compressor |
| Location | **oss-cn-shanghai** |
| IntranetEndpoint | oss-cn-shanghai-internal.aliyuncs.com |
| Qwen2.5-0.5B-Instruct/ | 64 对象，model.safetensors 988,097,824 B (0.92GiB) |
| ultrachat_200k/ | 87 对象，data/{train_sft,test_sft,train_gen,test_gen}-*.parquet |
| output/ | 已存在 |

## 4. ACK 集群（关键冲突）

账号内共有 2 个集群：

| cluster_id | 名字 | region | state | size | VPC |
|---|---|---|---|---|---|
| `cf6b1acd52ac74f189f659e74ea56096b` | cluster-Fw9Mkv | **cn-shanghai** | running | 0 | vpc-uf6ozdrez09gvl0dg6nll |
| `c100027250d794dff9012ece2025cbc41` | cluster-0PfjGJ | **cn-qingdao** | running | 0 | vpc-m5ecf7qno5fnl46bjuzy4 |

**冲突**：`~/.kube/config` 当前指向 **cn-qingdao** 那个（server 47.105.74.72:6443）。

**判定 cn-shanghai 才是正确目标的硬证据**（4 条互相独立）：
1. 配置里 PV 的 `url: https://oss-cn-shanghai-internal.aliyuncs.com` —— 内网 endpoint **只可能**从 cn-shanghai VPC 内访问；用 cn-qingdao 集群必然挂载失败。
2. OSS 桶 Location = oss-cn-shanghai。
3. 配置 nodepool 的 vswitch_ids `vsw-uf62g8dnn8armcfssjl8r / vsw-uf6h2ezyrvmujr0pl4acm / vsw-uf6bsoxrror7eki2g5vtj` 全部属于 vpc-uf6ozdrez09gvl0dg6nll = **cn-shanghai 集群的 VPC**。
4. RAM 角色 `rrsa` 的信任策略 OIDC issuer =
   `oidc-ack-cn-shanghai.oss-cn-shanghai.aliyuncs.com/cf6b1acd52ac74f189f659e74ea56096b`
   → 只对 **cn-shanghai 集群** `cf6b1acd...` 生效。cn-qingdao 集群无 RRSA 绑定。

cn-shanghai 集群附加组件齐备（csi-plugin / managed-csiprovisioner / ack-pod-identity-webhook /
ack-ram-authenticator / ack-nvidia-device-plugin 均在），0 个节点池，deletion_protection=True。

## 5. ACR

| 项 | 实测值 |
|---|---|
| instance | cri-49hkfmbw85y3jjt4 (test-acr-business, Enterprise_Basic, RUNNING) |
| namespace | test-llm-compressor (PRIVATE) |
| repo | test, repo_id = crr-bcqdjvq1iygwoxtf |
| 现有 tag | vllm-v0.34.0-base (10.04GB), stage1_{1,2,3}-*-v2.0.0 (各约 10.5GB) |

已有制品订阅规则 2 条：
- `crasr-38x338pagxitp3jb`: docker.m.daocloud.io / vllm/vllm-openai:v0.34.0 → test-llm-compressor/test
- `crasr-4xjcniskabgkpsy7`: nvcr.io / nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 → test-llm-compressor/test

`available_source_registry = ["docker.m.daocloud.io"]` **匹配第 1 条**，无阻塞。

已有构建规则 3 条，绑定的 GitHub 仓库 `liutiegen/test-compress` **已不存在**（gh repo view 报 not found）。

## 6. GPU 供给（决定性约束）

cn-shanghai 可达分区: cn-shanghai-m / n / o（o 无 GPU），vswitch 都在配置指定的 VPC 内。

| 分区 | 可用 GPU 机型 | GPU | 算力 | 显存 | 时价 |
|---|---|---|---|---|---|
| m | ecs.gn6v-c8g1.2xlarge | Tesla V100 | sm70 | 16G | ¥26.47/h |
| m | ecs.gn6e-c12g1.3xlarge | Tesla V100 | sm70 | 32G | ¥19.75/h |
| n | ecs.gn8ia.4xlarge | **L20** | **sm89** | 48G | **¥15.81/h** |
| n | ecs.gn7t-c16g1.4xlarge | G39 | - | 24G | 无报价 |

**决定性发现**：vLLM 的 compressed-tensors 各 scheme 最低算力要求
（源码 `compressed_tensors_wNa16.py` / `compressed_tensors_w8a8_int8.py` = 75，`w8a8_fp8` = 89）：
- W4A16 / W8A8-int8 / AWQ 均需 **sm75+**
- FP8_DYNAMIC 需 **sm89+**
- V100 = **sm70** → 配置出的量化模型 **V100 上一律无法加载**（连 W4A16 都不行）

且 V100 无原生 BF16，而 llm-compressor 0.15.0 链路默认 bf16。

→ **必须用 cn-shanghai-n 的 L20 (sm89)**。巧合的是 L20 单价还低于 V100（15.81 vs 26.47），
  并且 sm89 让 FP8_DYNAMIC 也可入矩阵。

## 7. 预算测算

余额 136.85 CNY。
- L20 单节点 ¥15.81/h → 单节点可跑 **8.65 小时**
- 若按 8 节点并行 → ¥126.5/h → **约 1.08 小时** 就用尽，且无重试余量

这是需要人工拍板的点（见 CHECKLIST.md 的 BLOCK-3）。

## 8. GitHub

`liutiegen/test-compress` 不存在；账号下有 test-repo / demo-project / test 等。
按 strategy.prepare_github_repo 「已有则不创建」，需 `gh repo create liutiegen/test-compress --public` 新建。