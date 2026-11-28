#!/usr/bin/env python3
"""
静态语法校验 —— 本地唯一能做的"代码正确性"检查（环境规定禁止本地跑重活）。

不只 parse：还做几项能抓住真实 bug 的语义检查：
  1. 所有 .py 能被 ast.parse
  2. shell 脚本通过 bash -n
  3. 禁止 bare `python`（只允许 python3）—— 镜像里没有 python 别名
  4. 容器内脚本不得引用 /root 或绝对本地路径（会跑不进容器）
  5. yaml 能被解析，且 Job/Pod 里引用的镜像与 ctx.env 的 TARGET_TAG 一致
     （tag 漂移是本项目历史上最常见的静默失败）

用法: python3 tools/verify_syntax.py [--root /root/test_training]
退出码 0 = 通过
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "logs", "artifacts", "results", ".venv"}


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    root = Path(args.root)

    problems: list[str] = []
    checked = {"py": 0, "sh": 0, "yaml": 0}

    py_files, sh_files, yaml_files = [], [], []
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            if p.suffix == ".py":
                py_files.append(p)
            elif p.suffix == ".sh":
                sh_files.append(p)
            elif p.suffix in {".yaml", ".yml"}:
                yaml_files.append(p)

    # 1. Python 语法
    for p in py_files:
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            checked["py"] += 1
        except SyntaxError as e:
            problems.append(f"[py] {p.relative_to(root)}:{e.lineno} {e.msg}")

    # 2. Shell 语法
    for p in sh_files:
        rc, out = sh(["bash", "-n", str(p)])
        if rc != 0:
            problems.append(f"[sh] {p.relative_to(root)}: {out.strip()[:300]}")
        else:
            checked["sh"] += 1

    # 3. 禁止 bare python
    for p in py_files + sh_files:
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            # 匹配 "python " 但不是 python3 / python -m 之后带 3
            if re.search(r"(?<![\w./-])python\s+-", line) or \
               re.search(r"(?<![\w./-])python\s+[a-zA-Z/]", line):
                problems.append(f"[py-bin] {p.relative_to(root)}:{i} 使用了 bare python: {s[:120]}")

    # 4. 容器内不该出现的本地绝对路径（用容器内 /workspace 或 /data 替代）
    container_scope = ["stage1_common", "stage1_1_quantize", "stage1_2_finetune",
                       "stage1_3_evaluate"]
    for d in container_scope:
        dp = root / d
        if not dp.exists():
            continue
        for p in list(dp.rglob("*.py")) + list(dp.rglob("*.sh")):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            txt = p.read_text(encoding="utf-8")
            for i, line in enumerate(txt.splitlines(), 1):
                s = line.strip()
                if s.startswith("#"):
                    continue
                if "/root/test_training" in line:
                    problems.append(
                        f"[path] {p.relative_to(root)}:{i} 容器内脚本引用了本地路径 {s[:100]}")

    # 5. YAML 解析 + 镜像 tag 一致性
    target_tag = ""
    ctx = root / "artifacts" / "ctx.env"
    if ctx.exists():
        for line in ctx.read_text().splitlines():
            if line.startswith("TARGET_TAG="):
                target_tag = line.split("=", 1)[1].strip().strip('"')

    try:
        import yaml  # type: ignore
        have_yaml = True
    except Exception:
        have_yaml = False

    if have_yaml:
        for p in yaml_files:
            try:
                docs = list(yaml.safe_load_all(p.read_text(encoding="utf-8")))
                checked["yaml"] += 1
            except Exception as e:
                problems.append(f"[yaml] {p.relative_to(root)}: {e}")
                continue
            if target_tag and "k8s/" in str(p):
                imgs = []
                def walk(o):
                    if isinstance(o, dict):
                        for k, v in o.items():
                            if k == "image" and isinstance(v, str):
                                imgs.append(v)
                            walk(v)
                    elif isinstance(o, list):
                        for v in o:
                            walk(v)
                walk(docs)
                for img in imgs:
                    if ":" not in img.rsplit("/", 1)[-1]:
                        problems.append(f"[img] {p.relative_to(root)}: '{img}' 没有 tag")
                    elif not img.endswith(":" + target_tag):
                        problems.append(
                            f"[img] {p.relative_to(root)}: '{img}' 与 TARGET_TAG={target_tag} 不一致")
    else:
        problems.append("[yaml] 本机无 PyYAML，跳过 yaml 校验（warn，不是 block）"
                        .replace("[yaml]", "[yaml-warn]"))

    result = {
        "checked": checked,
        "problems": problems,
        "ok": not any(not x.startswith("[yaml-warn]") for x in problems),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())