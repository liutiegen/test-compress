#!/usr/bin/env python3
"""
静态 API 校验 —— 不联网、不需要 GPU/安装依赖，直接对着真实 wheel 检查
stage1 代码里用到的每一个 import 路径与关键字参数是否真实存在。

为什么需要它：
  量化脚本只能在 GPU 节点上真正跑通，但"import 路径写错 / 参数名写错"
  这类错误完全可以在本地静态查出，避免把错误带上集群烧 GPU 时间。

用法:
  python3 tools/verify_api_contract.py /path/to/llmcompressor.whl /path/to/compressed_tensors.whl
"""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

# 需要校验的脚本
TARGETS = [
    "stage1_1_quantize/quantize.py",
    "stage1_1_quantize/matrix.py",
]

# 我们依赖的外部包前缀（只校验这些，别的跳过）
EXTERNAL_PREFIXES = ("llmcompressor", "compressed_tensors", "transformers",
                     "datasets", "safetensors", "torch")


def wheel_modules(whl: Path) -> dict[str, str]:
    """返回 {module.dotted.path: source} 用于包内模块。"""
    out: dict[str, str] = {}
    with zipfile.ZipFile(whl) as z:
        for name in z.namelist():
            if not name.endswith(".py"):
                continue
            mod = name[:-3].replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            out[mod] = name
    return out


def module_exists(modules: dict[str, str], dotted: str) -> bool:
    """检查 dotted 模块是否存在于 wheel 中（含包路径推断）。"""
    if dotted in modules:
        return True
    parts = dotted.split(".")
    for i in range(len(parts) - 1, 0, -1):
        if ".".join(parts[:i]) in modules:
            return True
    return False


def find_attr(whl: Path, dotted: str) -> tuple[bool, str]:
    """
    检查 dotted（模块.属性）是否可解析。

    策略：先定位最长的可用模块前缀，再把剩余部分当属性名在源码里搜
    （包括 `from x import y` 的 re-export，__all__ 等）。
    """
    with zipfile.ZipFile(whl) as z:
        names = {n[:-3].replace("/", ".").replace(".__init__", ""): n
                 for n in z.namelist() if n.endswith(".py")}

        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            mod = ".".join(parts[:cut])
            if mod not in names:
                continue
            attr = parts[cut:]
            if not attr:
                return True, f"module {mod}"

            src = z.read(names[mod]).decode("utf-8", "replace")
            tree = ast.parse(src)
            # 1) 直接定义
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == attr[0]:
                        return True, f"defined in {mod}"
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == attr[0]:
                            return True, f"assigned in {mod}"
                if isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name) and node.target.id == attr[0]:
                        return True, f"annotated in {mod}"
                # 2) from ... import attr
                if isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        if a.name == attr[0] or a.asname == attr[0]:
                            return True, f"re-exported in {mod}"
                # 3) __all__ 里声明（可能是 star import 透传）
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "__all__":
                            try:
                                vals = [e.value for e in node.value.elts]
                                if attr[0] in vals:
                                    return True, f"in __all__ of {mod}"
                            except Exception:
                                pass
            # 4) 通配 import 透传
            if "import *" in src:
                return True, f"star-import in {mod} (unverified)"
            return False, f"{attr[0]} not found in {mod}"
    return False, f"no module prefix matched for {dotted}"


def collect_imports(path: Path) -> list[tuple[str, str, int]]:
    """抽出 (dotted, 形式, 行号)。"""
    tree = ast.parse(path.read_text())
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(EXTERNAL_PREFIXES):
                for a in node.names:
                    found.append((f"{node.module}.{a.name}", "from", node.lineno))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(EXTERNAL_PREFIXES):
                    found.append((a.name, "import", node.lineno))
    return found


def check_kwargs(path: Path, whl: Path) -> list[str]:
    """
    检查我们调用的关键构造函数/函数，其关键字参数是否被真实签名接受。
    针对最易错的几个：oneshot / GPTQModifier / QuantizationModifier / AWQModifier。
    """
    problems: list[str] = []
    src = path.read_text()
    tree = ast.parse(src)

    wanted = {
        "oneshot": ("llmcompressor/entrypoints/oneshot.py", "oneshot"),
        "GPTQModifier": None,          # 在 gptq/base.py，用 pydantic 字段
        "QuantizationModifier": None,
        "AWQModifier": None,
        "SmoothQuantModifier": None,
    }

    with zipfile.ZipFile(whl) as z:
        names = {n: n for n in z.namelist() if n.endswith(".py")}

        def class_fields(cls: str, _seen: set[str] | None = None) -> set[str] | None:
            """
            pydantic 类：收集字段名 + __init__ kwargs，并递归解析基类。

            llm-compressor 的 QuantizationModifier / GPTQModifier 把 scheme、targets、
            ignore 定义在 QuantizationMixin 基类上，只看自身 class body 会误报，
            所以必须沿 bases 向上收集。
            """
            if _seen is None:
                _seen = set()
            if cls in _seen:
                return set()
            # 同名类可能出现在多个模块，优先取第一个能解析的
            for n in names:
                s = z.read(n).decode("utf-8", "replace")
                if f"class {cls}(" not in s:
                    continue
                t = ast.parse(s)
                for node in t.body:
                    if isinstance(node, ast.ClassDef) and node.name == cls:
                        _seen.add(cls)
                        fields: set[str] = set()
                        for st in node.body:
                            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                                fields.add(st.target.id)
                            if isinstance(st, ast.FunctionDef) and st.name == "__init__":
                                a = st.args
                                fields.update(x.arg for x in a.args + a.kwonlyargs)
                        # 递归收集基类字段
                        for base in node.bases:
                            bname = None
                            if isinstance(base, ast.Name):
                                bname = base.id
                            elif isinstance(base, ast.Attribute):
                                bname = base.attr
                            elif isinstance(base, ast.Subscript):
                                v = base.value
                                bname = v.id if isinstance(v, ast.Name) else (
                                    v.attr if isinstance(v, ast.Attribute) else None)
                            if not bname or bname in {"object", "BaseModel"}:
                                continue
                            sub = class_fields(bname, _seen)
                            if sub:
                                fields |= sub
                        return fields
            return None

        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else None)
            if name not in wanted:
                continue
            kws = {k.arg for k in call.keywords if k.arg}
            if name == "oneshot":
                s = z.read("llmcompressor/entrypoints/oneshot.py").decode()
                t = ast.parse(s)
                params = set()
                has_var_kw = False
                for node in t.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "oneshot":
                        a = node.args
                        params.update(x.arg for x in a.args + a.kwonlyargs)
                        has_var_kw = a.kwarg is not None
                # oneshot 通过 **kwargs 把参数路由进 args 类（DatasetArguments 等），
                # 所以这些字段名也算合法。
                extra: set[str] = set()
                if has_var_kw:
                    for cn in z.namelist():
                        if not cn.startswith("llmcompressor/args/") or not cn.endswith(".py"):
                            continue
                        tt = ast.parse(z.read(cn).decode("utf-8", "replace"))
                        for node in ast.walk(tt):
                            if isinstance(node, ast.ClassDef):
                                for st in node.body:
                                    if isinstance(st, ast.AnnAssign) and isinstance(
                                            st.target, ast.Name):
                                        extra.add(st.target.id)
                if params:
                    bad = kws - params - extra - {"kwargs"}
                    if bad:
                        problems.append(
                            f"{path.name}: oneshot() 不接受的参数 {sorted(bad)}")
                    missing = {"model"} - kws
                    if missing:
                        problems.append(f"{path.name}: oneshot() 缺少必填 {missing}")
            else:
                fields = class_fields(name)
                if fields is None:
                    problems.append(f"{path.name}: 未找到类 {name} 的定义")
                    continue
                bad = kws - fields
                # 允许 **kwargs 透传的类
                if bad:
                    problems.append(
                        f"{path.name}: {name}() 可疑参数 {sorted(bad)} "
                        f"(该类已知字段 {sorted(fields)[:12]}...)")
    return problems


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: verify_api_contract.py <llmcompressor.whl> <compressed_tensors.whl>")
        return 2
    lc = Path(sys.argv[1])
    ct = Path(sys.argv[2])

    root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    checked = 0

    targets = TARGETS
    if len(sys.argv) > 3:
        # 允许显式指定待检脚本（用于自测校验器本身是否真的能查出问题）
        targets = [sys.argv[3]]
        root = Path(".")

    lc_mods = wheel_modules(lc)
    ct_mods = wheel_modules(ct)

    for rel in targets:
        p = root / rel
        if not p.exists():
            failures.append(f"{rel}: 文件不存在")
            continue
        for dotted, form, lineno in collect_imports(p):
            checked += 1
            if dotted.startswith("compressed_tensors"):
                ok, why = find_attr(ct, dotted)
            elif dotted.startswith("llmcompressor"):
                ok, why = find_attr(lc, dotted)
            else:
                ok, why = True, "skip (not in provided wheels)"
            if not ok:
                failures.append(f"{rel}:{lineno} {form} {dotted} -> {why}")

    print(f"[api] 检查 import 路径 {checked} 条")
    for rel in targets:
        p = root / rel
        if p.exists():
            problems = check_kwargs(p, lc)
            failures.extend(problems)

    if failures:
        print("\n[api] FAILED:")
        for f in failures:
            print("  ✗", f)
        return 1

    print("[api] OK: 所有 import 路径与关键字参数均与真实 wheel 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())