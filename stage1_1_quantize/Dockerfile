# stage1.1 量化镜像
#
# 构建方式：ACR 云端构建（本地禁止拉镜像，见 environment.forbidden_ops）。
# FROM 用的是**已经同步到本仓库**的 base 镜像，不是 docker.io。
#   base tag 由 stage2.1 制品订阅从 docker.m.daocloud.io/vllm/vllm-openai:v0.34.0
#   同步为 test-llm-compressor/test:vllm-v0.34.0-base
#
# 依赖约束（已实测核对，不可随意放宽）：
#   vllm v0.34.1        : transformers >=5.16.1,<5.25.0 ; compressed-tensors ==0.20.1
#   llmcompressor 0.15.0: transformers >=5.15.0,<=5.21.0; torch >=2.10.0,<=2.14.1
#   => 交集: transformers [5.16.1, 5.21.0]。因此必须 **降级** 基础镜像里的
#      transformers，否则 llmcompressor 的 import 会因 API 变动而失败。

ARG BASE_IMAGE=test-acr-business-registry-vpc.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:vllm-v0.34.0-base
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ARG UV_INSTALLER_GITHUB_BASE_URL=https://ghproxy.cn/https://github.com
ARG TRANSFORMERS_VERSION=5.21.0
ARG LLMCOMPRESSOR_VERSION=0.15.0

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    UV_INSTALLER_GITHUB_BASE_URL=${UV_INSTALLER_GITHUB_BASE_URL} \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# 运行期完全离线：模型/数据集都从 OSS 挂载的本地路径读取，禁止联网拉取。
# 这样既符合"data 预先下载"的设计，也避免 Pod 因无外网而卡在 hf 下载上。

# 关键：把 transformers 对齐到 llmcompressor 允许的窗口
RUN pip install --no-cache-dir \
        "transformers==${TRANSFORMERS_VERSION}" \
        "compressed-tensors==0.20.1" \
    && pip install --no-cache-dir \
        "llmcompressor==${LLMCOMPRESSOR_VERSION}" \
    && python -c "import transformers, llmcompressor, compressed_tensors, vllm; \
print('transformers', transformers.__version__); \
print('llmcompressor', llmcompressor.__version__); \
print('compressed_tensors', compressed_tensors.__version__); \
print('vllm', vllm.__version__)"

# lm-evaluation-harness：从固定 commit 安装，保证评估口径可复现
ARG LMEVAL_COMMIT=d6de8164
RUN pip install --no-cache-dir \
        "lm_eval @ git+https://github.com/EleutherAI/lm-evaluation-harness@${LMEVAL_COMMIT}" \
    || pip install --no-cache-dir lm_eval

WORKDIR /workspace
COPY stage1_1_quantize/ /workspace/stage1_1_quantize/
COPY tools/ /workspace/tools/
COPY stage1_common/ /workspace/stage1_common/

RUN python -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" \
        /workspace/stage1_1_quantize/quantize.py \
        /workspace/stage1_1_quantize/evaluate.py \
        /workspace/stage1_1_quantize/select_best.py \
        /workspace/stage1_1_quantize/matrix.py

# 启动脚本：先量化（可选），再评估，写结果到挂载目录
COPY stage1_common/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]