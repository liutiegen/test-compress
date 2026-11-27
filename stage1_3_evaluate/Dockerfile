# stage1.3 部署评估镜像
#
# 与 1.1/1.2 共用 base；这里只需要 vLLM + lm-eval（不需要 llmcompressor，
# 因为评估是对已量化好的产物做推理，不是再量化）。
ARG BASE_IMAGE=test-acr-business-registry-vpc.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:vllm-v0.34.0-base
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ARG LMEVAL_COMMIT=d6de8164

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

RUN pip install --no-cache-dir \
        "lm_eval @ git+https://github.com/EleutherAI/lm-evaluation-harness@${LMEVAL_COMMIT}" \
    || pip install --no-cache-dir lm_eval

WORKDIR /workspace
COPY stage1_1_quantize/ /workspace/stage1_1_quantize/
COPY stage1_3_evaluate/ /workspace/stage1_3_evaluate/
COPY tools/ /workspace/tools/

RUN python -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" \
        /workspace/stage1_1_quantize/evaluate.py

RUN chmod +x /workspace/stage1_3_evaluate/compare_all.sh

ENTRYPOINT ["/workspace/stage1_3_evaluate/compare_all.sh"]