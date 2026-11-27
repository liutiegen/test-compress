#!/usr/bin/env python3
"""
生成 4 个量化 Job 的清单，把已发现的运行时问题全部在 Job 层绕过。

为什么用 ConfigMap 注入脚本而不是重建镜像：
  镜像里的 evaluate.py 默认 dtype=bfloat16（T4 不支持）、entrypoint 也没有
  处理数据集只读问题。重建镜像要重走 ACR 构建 + 数据盘快照（约 40 分钟 +
  一笔 CPU 机器费用）。冒烟阶段先用 ConfigMap 覆盖，确认链路无误后再重建。

已绕过的问题：
  1. bfloat16 -> T4(sm75) 不支持，必须 --dtype auto
  2. OSS 只读挂载 -> datasets 要写 .lock，必须拷到可写 emptyDir
  3. 数据集需预置 -> 已在 OSS benchmark_hf_cache/，挂载后拷到本地 cache

用法:
  python3 tools/gen_quant_jobs.py --image <registry>/test:tag --out <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_1_quantize"))
from matrix import plan  # noqa: E402

JOB_TMPL = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: quant-{slug}
  namespace: st-compress
  labels:
    app: quant
    matrix-id: "{matrix_id}"
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: quant
        matrix-id: "{matrix_id}"
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: acr-cred
      containers:
        - name: quant
          image: {image}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "/scripts/run.sh"]
          env:
            - name: MATRIX_ID
              value: "{matrix_id}"
            - name: ALGORITHM
              value: "{algorithm}"
            - name: SCHEME
              value: "{scheme}"
            - name: TARGET
              value: "{target}"
            - name: NUM_CALIBRATION_SAMPLES
              value: "{calib}"
            - name: MAX_SEQ_LENGTH
              value: "{seqlen}"
            - name: EVAL_LIMIT
              value: "{limit}"
            - name: PYTHONUNBUFFERED
              value: "1"
          volumeMounts:
            - name: input
              mountPath: /data
              readOnly: true
            - name: output
              mountPath: /output
            - name: scripts
              mountPath: /scripts
            - name: hfcache
              mountPath: /tmp/hfcache
            - name: shm
              mountPath: /dev/shm
          resources:
            limits:
              nvidia.com/gpu: 1
              cpu: "8"
              memory: 40Gi
            requests:
              nvidia.com/gpu: 1
              cpu: "6"
              memory: 28Gi
      volumes:
        - name: input
          persistentVolumeClaim:
            claimName: pvc-stage-1-1-input
            readOnly: true
        - name: output
          persistentVolumeClaim:
            claimName: pvc-stage-1-1-quant
        - name: scripts
          configMap:
            name: quant-runner-script
        - name: hfcache
          emptyDir:
            sizeLimit: 2Gi
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
"""

CONFIGMAP = """---
apiVersion: v1
kind: ConfigMap
metadata:
  name: quant-runner-script
  namespace: st-compress
data:
  run.sh: |
    #!/bin/bash
    set -eo pipefail
    OUT=/output/$MATRIX_ID
    mkdir -p "$OUT"

    # datasets 库要在 cache 目录写 .lock，而输入 PVC 只读，必须拷到可写目录
    cp -r /data/benchmark_hf_cache/ds_cache /tmp/hfcache/ds_cache 2>/dev/null || true
    cp -r /data/benchmark_hf_cache/hf_home /tmp/hfcache/hf_home 2>/dev/null || true
    export HF_DATASETS_CACHE=/tmp/hfcache/ds_cache
    export HF_HOME=/tmp/hfcache/hf_home

    # 本地模型/数据（离线）
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    echo "=== quantize $MATRIX_ID ==="
    python3 /workspace/stage1_1_quantize/quantize.py \\
      --model-path /data/Qwen2.5-0.5B-Instruct \\
      --dataset-path /data/ultrachat_200k \\
      --output-dir "$OUT/model" \\
      --algorithm "$ALGORITHM" \\
      --scheme "$SCHEME" \\
      --target "$TARGET" \\
      --matrix-id "$MATRIX_ID" \\
      --num-calibration-samples "$NUM_CALIBRATION_SAMPLES" \\
      --max-seq-length "$MAX_SEQ_LENGTH"
    echo "=== quantize done, rc=$? ==="

    echo "=== evaluate ==="
    # dtype 必须 auto：T4(sm75) 不支持 bfloat16，写死会在 vLLM 初始化时 ValueError
    python3 /workspace/stage1_1_quantize/evaluate.py \\
      --model-path "$OUT/model" \\
      --output-dir "$OUT" \\
      --label "$MATRIX_ID" \\
      --dtype auto \\
      --limit "$EVAL_LIMIT" \\
      --tasks arc_easy,hellaswag || echo "EVAL FAILED (see eval_metrics.json)"
    echo "=== all done ==="
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gpu-capability", type=int, default=75)
    ap.add_argument("--calib", type=int, default=128,
                    help="校准样本数。公网/跨地域 IO 慢，不宜过大")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    inc, exc = plan(args.gpu_capability)
    docs = [CONFIGMAP]
    for v in inc:
        docs.append(JOB_TMPL.format(
            slug=v.matrix_id.replace("_", "-"),
            matrix_id=v.matrix_id,
            algorithm=v.algorithm,
            scheme=v.scheme,
            target=v.target,
            image=args.image,
            calib=args.calib,
            seqlen=args.seqlen,
            limit=args.limit,
        ))

    text = "".join(docs)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"生成 {len(inc)} 个量化 Job + 1 个 ConfigMap -> {args.out}")
        for v in inc:
            print(f"  - {v.matrix_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())