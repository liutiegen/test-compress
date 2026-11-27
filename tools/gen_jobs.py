#!/usr/bin/env python3
"""
把 matrix.py 的变体列表展开成可 apply 的 k8s Job 清单。

为什么需要生成而不是手写：
  52 个变体手写 52 份 yaml 必然出现 copy-paste 漂移（image tag、
  env 名、matrix-id 对不上），而"镜像 tag 与 k8s 中 image 必须一致"
  是 stage1 的硬性交付要求。用程序生成能保证一致性并可校验。

用法:
  python3 tools/gen_jobs.py --image test-.../test:stage1_1-quant-<sha> \
      --gpu-capability 89 --out stage1_1_quantize/k8s/jobs_generated.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "stage1_1_quantize"))
from matrix import plan  # noqa: E402

TEMPLATE = """---
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
      nodeSelector:
        kubernetes.io/os: linux
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: quant
          image: {image}
          imagePullPolicy: IfNotPresent
          env:
            - {{name: SUBCOMMAND, value: quant_eval}}
            - {{name: MATRIX_ID, value: "{matrix_id}"}}
            - {{name: ALGORITHM, value: "{algorithm}"}}
            - {{name: SCHEME, value: "{scheme}"}}
            - {{name: TARGET, value: "{target}"}}
            - {{name: MODEL_PATH, value: /data/Qwen2.5-0.5B-Instruct}}
            - {{name: DATASET_PATH, value: /data/ultrachat_200k}}
            - {{name: OUTPUT_ROOT, value: /output}}
            - {{name: NUM_CALIBRATION_SAMPLES, value: "128"}}
            - {{name: MAX_SEQ_LENGTH, value: "1024"}}
            - {{name: EVAL_LIMIT, value: "{eval_limit}"}}
          resources:
            limits:
              nvidia.com/gpu: 1
              cpu: "8"
              memory: 48Gi
            requests:
              nvidia.com/gpu: 1
              cpu: "6"
              memory: 32Gi
          volumeMounts:
            - name: input
              mountPath: /data
              readOnly: true
            - name: output
              mountPath: /output
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: input
          persistentVolumeClaim:
            claimName: pvc-stage-1-1-input
            readOnly: true
        - name: output
          persistentVolumeClaim:
            claimName: pvc-stage-1-1-quant
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gpu-capability", type=int, default=89)
    ap.add_argument("--out", default="-")
    ap.add_argument("--eval-limit", default="")
    ap.add_argument("--only", default=None,
                    help="逗号分隔的 matrix_id，仅生成这些（用于分批/冒烟）")
    args = ap.parse_args()

    inc, exc = plan(args.gpu_capability)
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        missing = keep - {v.matrix_id for v in inc}
        if missing:
            print(f"ERROR: 指定了不在可用集合里的 matrix_id: {sorted(missing)}",
                  file=sys.stderr)
            return 2
        inc = [v for v in inc if v.matrix_id in keep]

    docs = []
    for v in inc:
        docs.append(TEMPLATE.format(
            slug=v.matrix_id.replace("_", "-"),
            matrix_id=v.matrix_id,
            algorithm=v.algorithm,
            scheme=v.scheme,
            target=v.target,
            image=args.image,
            eval_limit=args.eval_limit,
        ))

    text = "".join(docs)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"生成 {len(inc)} 个 Job -> {args.out}")
        print(f"（另有 {len(exc)} 个变体因算法/方案不匹配或算力不足被排除）")
        for v, why in exc:
            print(f"  - {v.matrix_id}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())