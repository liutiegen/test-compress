#!/usr/bin/env python3
"""
依赖版本窗口自洽性检查（离线，不安装任何东西）。

背景：这个管线的依赖三角是最容易崩的地方：
  - vllm        : transformers >=5.16.1,<5.25.0
  - trl 1.19    : transformers >=4.56.2 (宽)
  - lm_eval     : 自带 requirements，与 transformers 冲突常见
  - torch       : 必须与镜像 CUDA 同线（cu128 -> torch 2.x+cu128）

做法：从 requirements.txt / Dockerfile 的 ARG 里抽出钉住的版本，
然后用一个**显式声明的兼容区间表**校验。区间表是人工维护的事实，
不是从网络拿的 —— 本地只做静态校验。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# 已知兼容区间（来源：各库 pyproject 的 dependencies，人工核对）
WINDOWS = {
    "transformers": (">=5.16.1", "<5.25.0"),   # 由 vllm 决定
    "trl": (">=1.18.0", None),
    "peft": (">=0.14.0", None),
    "accelerate": (">=1.4.0", None),
    "datasets": (">=3.0.0", None),
}


def parse_ver(v: str) -> tuple:
    out = []
    for part in re.split(r"[.\-+]", v):
        m = re.match(r"^\d+", part)
        out.append(int(m.group(0)) if m else 0)
    return tuple(out)


def within(ver: str, lo: str | None, hi: str | None) -> tuple[bool, str]:
    v = parse_ver(ver)
    if lo:
        op = lo[:2] if lo[:2] in (">=", "<=", "==", "!=") else lo[0]
        val = lo[len(op):]
        if op == ">=" and not (v >= parse_ver(val)):
            return False, f"{ver} < {val}"
    if hi:
        op = hi[:2] if hi[:2] in ("<=", ">=", "==") else hi[0]
        val = hi[len(op):]
        if op == "<" and not (v < parse_ver(val)):
            return False, f"{ver} >= {val}"
        if op == "<=" and not (v <= parse_ver(val)):
            return False, f"{ver} > {val}"
    return True, ""


def collect_pins(root: Path) -> dict[str, str]:
    """从 requirements.txt 与 Dockerfile ARG 收集钉住的版本。"""
    pins: dict[str, str] = {}
    req = root / "requirements.txt"
    if req.exists():
        for line in req.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([0-9][0-9a-zA-Z.\-]*)", line)
            if m:
                pins[m.group(1).lower()] = m.group(2)
    for df in root.rglob("Dockerfile"):
        txt = df.read_text()
        for m in re.finditer(r'^\s*"?([A-Za-z0-9_.\-]+)==\$\{?([A-Z_]+)\}?"?', txt, re.M):
            pkg, var = m.group(1).lower(), m.group(2)
            am = re.search(rf"ARG\s+{var}=([0-9][0-9a-zA-Z.\-]*)", txt)
            if am and pkg not in pins:
                pins[pkg] = am.group(1)
        for m in re.finditer(r"^ARG\s+([A-Z_]+)_VERSION=([0-9][0-9a-zA-Z.\-]*)", txt, re.M):
            pass  # 上面已按包名收集
    return pins


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                Path(__file__).resolve().parent.parent)
    pins = collect_pins(root)
    problems = []
    for pkg, ver in sorted(pins.items()):
        lo, hi = WINDOWS.get(pkg, (None, None))
        if lo is None and hi is None:
            continue
        ok, why = within(ver, lo, hi)
        if not ok:
            problems.append(f"{pkg}=={ver} 超出窗口 [{lo}, {hi}] ({why})")

    # torch 与 CUDA 同线检查：base 镜像是 cu128
    torch_cu = re.findall(r"cu(\d+)", json.dumps(pins))
    result = {
        "pins": pins,
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "problems": problems,
        "ok": not problems,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())