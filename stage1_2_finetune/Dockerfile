# stage1.2 微调镜像（QAD）
#
# 与 stage1.1 共用 base，但目标不同：
#   stage1.1 需要能"跑量化"（llmcompressor + GPTQ/AWQ/SmoothQuant）
#   stage1.2 需要能"跑训练"（trl + accelerate + QAD 的反向传播）
# 两者依赖窗口一致，因此 FROM 同一个 base，只额外补 trl。
ARG BASE_IMAGE=test-acr-business-registry-vpc.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:vllm-v0.34.0-base
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ARG TRANSFORMERS_VERSION=5.21.0
ARG LLMCOMPRESSOR_VERSION=0.15.0
ARG TRL_VERSION=1.18.0

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# transformers 必须钉在 llmcompressor 允许的窗口内（见 stage1.1 Dockerfile 注释）
RUN pip install --no-cache-dir \
        "transformers==${TRANSFORMERS_VERSION}" \
        "compressed-tensors==0.20.1" \
    && pip install --no-cache-dir \
        "llmcompressor==${LLMCOMPRESSOR_VERSION}" \
        "trl==${TRL_VERSION}" \
    && python3 -c "import transformers, llmcompressor, trl; \
print('transformers', transformers.__version__); \
print('llmcompressor', llmcompressor.__version__); \
print('trl', trl.__version__)"

WORKDIR /workspace
COPY stage1_2_finetune/ /workspace/stage1_2_finetune/
COPY stage1_1_quantize/matrix.py /workspace/stage1_1_quantize/matrix.py
COPY stage1_1_quantize/evaluate.py /workspace/stage1_1_quantize/evaluate.py
COPY tools/ /workspace/tools/
COPY stage1_common/ /workspace/stage1_common/

RUN python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" \
        /workspace/stage1_2_finetune/finetune_qad.py \
        /workspace/stage1_1_quantize/matrix.py

COPY stage1_common/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]