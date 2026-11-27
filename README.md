# 量化流水线：Qwen2.5-0.5B-Instruct

在 Aliyun ACK 上用 llm-compressor 对 Qwen2.5-0.5B-Instruct 做多组量化实验，
在 benchmark 上选优，再对最优方案做 QAD 微调。

集群：cn-qingdao / `c100027250d794dff9012ece2025cbc41`
镜像仓库：`test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test`
模型与数据集：OSS `oss://test-llm-compressor/`

## 目录

```
stage1_1_quantize/     量化 + 评估 + 选优
  matrix.py            矩阵定义（单一事实来源）
  quantize.py          单个变体的量化worker
  evaluate.py          lm-eval 评估（vLLM 后端）
  select_best.py       跨变体选优
  Dockerfile
  k8s/                 pv / pvc / deployment(Job模板)
stage1_2_finetune/     QAD 微调
stage1_3_evaluate/     baseline/量化/微调 三者横向对比
stage1_common/         共用 entrypoint
tools/                 Job 生成器 + API 契约校验器
tests/                 PV/PVC 挂载验证
```

## 实验矩阵

三个维度：算法 × 方案 × 位置。受预算约束只跑 4 个组合，构成三次单变量对照：

| # | 组合 | 对照维度 |
|---|---|---|
| 1 | `awq__w4a16__linall` | 算法（vs #2） |
| 2 | `gptq__w4a16__linall` | 算法（vs #1） |
| 3 | `sq__w8a8__linall` | 方案 8bit vs 4bit |
| 4 | `awq__w4a16__mlp` | 位置（vs #1） |

其余 60 个组合被程序化排除并记录原因：
- 12 个 FP8_DYNAMIC 需 sm89，集群只有 T4(sm75)
- 12 个算法/方案语义不匹配（smoothquant 不适用 W4A16 等）
- 36 个因预算限制未入选

查看完整排除清单：

```bash
cd stage1_1_quantize && python3 matrix.py 75
```

## 关键约束（踩过的坑）

1. **GPU 算力门控**：W4A16/W8A8 需 sm75+，FP8_DYNAMIC 需 sm89+。
   集群只有 T4(sm75)，所以 FP8 不可用。V100 是 sm70，任何方案都不行。

2. **OSS 走公网**：集群在 cn-qingdao，bucket 在 cn-shanghai，内网 endpoint 不可达。
   公网读速约 893 kB/s，4 个 Pod 并行时会被摊薄，故校准样本限制在 128。

3. **ossfs 版本差异**：
   - `ossfs2` 不支持 `authType: accessKey` → 去掉 `fuseType` 走 ossfs 1.0
   - `close_to_open` 是 ossfs2 专有参数，ossfs 1.0 报 `fuse: unknown option`

4. **集群无出网到 docker.io**：所有镜像必须来自 ACR，且需要 `imagePullSecrets`。

5. **ACR 凭证不要手工拼 base64**，用
   `kubectl create secret docker-registry`，否则用户名极易写错导致 401。

6. **一个 PV 只能绑一个 PVC**：重复定义会导致后绑的 PVC 永远 Pending。
   若 PV 上残留旧 `claimRef`，需 `kubectl patch --type=json -p '[{"op":"remove","path":"/spec/claimRef"}]'`。

## 依赖窗口

```
vllm v0.34.1         : transformers >= 5.16.1, < 5.25.0
llmcompressor 0.15.0 : transformers >= 5.15.0, <= 5.21.0
=> 交集 [5.16.1, 5.21.0]，Dockerfile 钉 5.21.0
```

详见 `requirements.txt`。

## 代码校验

本地无 GPU，无法真跑量化，所以用静态校验兜住 API 误用：

```bash
python3 tools/verify_api_contract.py \
    <llmcompressor.whl> <compressed_tensors.whl>
```

它会解析 wheel 源码 AST，校验脚本里每个 import 路径与构造参数是否真实存在。
**必须同时做负向测试**（喂一个参数名写错的脚本，确认能报错），否则校验器可能空过。

## 部署

```bash
# 1. PV/PVC（顺序不能反）
kubectl apply -f stage1_1_quantize/k8s/pv_stage_1_1.yaml
kubectl apply -f stage1_1_quantize/k8s/pvc_stage_1_1.yaml

# 2. 生成 Job（镜像 tag 必须与 ACR 构建产物一致）
python3 tools/gen_jobs.py \
    --image <registry>/test:stage1_1-quant-<tag> \
    --gpu-capability 75 \
    --out stage1_1_quantize/k8s/jobs_generated.yaml

# 3. 提交
kubectl apply -f stage1_1_quantize/k8s/jobs_generated.yaml
```

## 选优规则

`select_best.py` 的双门槛：

- 任一任务相对 baseline 绝对退化 > 1.0 个百分点 → 淘汰
- 或相对退化 > 3% → 淘汰

通过者按 `mean(保留率) - 0.02*(1-压缩率)` 排序，
即同等精度下偏好更省显存的方案。阈值可用
`--max-drop-pp` / `--max-drop-rel` 覆盖。