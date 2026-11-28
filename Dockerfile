# ============================================================================
# 训练 + 评测镜像（本项目的唯一业务镜像）
#
# 为什么 FROM 用 -vpc 域名：ACR 云端构建机在 ACR 的 VPC 内，只能解析 vpc 域名。
#   用公网域名会报 `tls: bad certificate`；反过来集群节点拉镜像要用公网域名
#   （或用 VPC 打通后的 vpc 域名，见 manifests）。两处不能"统一"。
#
# 基础镜像 vllm-v0.34.0-base 已在 ACR（由制品订阅从 docker.m.daocloud.io 同步），
# 已经带 CUDA 12.8 runtime + python3 + vllm 依赖树。镜像里**没有** `python`
# 别名，所有脚本必须用 python3。
#
# 依赖窗口（人工核对，tools/check_dep_window.py 会验证）：
#   vllm 0.34.0 需要 transformers >=5.16.1,<5.25.0
#   trl 1.19.0  需要 transformers >=4.56.2（宽）
#   => 取 5.21.0（在 vllm 窗口内，且 llmcompressor 未安装所以无需更严）
# ============================================================================
ARG BASE_IMAGE=test-acr-business-registry-vpc.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:vllm-v0.34.0-base
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ARG TRANSFORMERS_VERSION=5.21.0
ARG TRL_VERSION=1.19.0
ARG PEFT_VERSION=0.22.0
ARG LMEVAL_VERSION=0.4.9.2

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HOME=/tmp/hf_home

# 运行期完全离线：模型/数据集都在 /data 挂载里。
# 这里之所以设离线标志，是因为 Pod 无外网；留着联网会卡在 hf 下载超时上。

RUN set -eux; \
    pip install --no-cache-dir \
        "transformers==${TRANSFORMERS_VERSION}" \
        "trl==${TRL_VERSION}" \
        "peft==${PEFT_VERSION}" \
        "accelerate>=1.15.0" \
        "datasets>=4.8.4" \
        "lm_eval==${LMEVAL_VERSION}"; \
    python3 -c "import transformers, trl, peft, datasets, torch, lm_eval; \
print('transformers', transformers.__version__); \
print('trl', trl.__version__); \
print('peft', peft.__version__); \
print('datasets', datasets.__version__); \
print('torch', torch.__version__); \
print('cuda', torch.version.cuda); \
print('lm_eval', lm_eval.__version__ if hasattr(lm_eval,'__version__') else 'n/a')"

# 构建期自检：把「依赖能 import」变成构建失败，而不是运行到一半才炸。
# 这里只 import，不加载权重（构建机上没有 GPU，也不该有）。
RUN python3 - <<'PY'
import importlib
for mod in ("torch", "transformers", "trl", "peft", "datasets", "lm_eval"):
    importlib.import_module(mod)
# trl 的 SFTTrainer 在部分版本里是 lazy 导入，显式拉一次
from trl import SFTTrainer, SFTConfig  # noqa: F401
print("import check OK")
PY

WORKDIR /workspace
COPY stage1_common/ /workspace/stage1_common/
COPY stage1_2_finetune/ /workspace/stage1_2_finetune/
COPY stage1_3_evaluate/ /workspace/stage1_3_evaluate/
COPY tools/ /workspace/tools/

# 静态语法校验放进构建：语法错误不该等到 Pod 里才暴露
RUN python3 -c "import ast,sys;[ast.parse(open(f).read(),f) for f in sys.argv[1:]]" \
        /workspace/stage1_common/config.py \
        /workspace/stage1_2_finetune/train_sft.py \
        /workspace/stage1_3_evaluate/evaluate.py \
        /workspace/stage1_3_evaluate/compare.py

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]