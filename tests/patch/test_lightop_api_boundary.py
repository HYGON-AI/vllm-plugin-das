# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Fail-closed production boundary for the public categorized LightOp API."""

from __future__ import annotations

import ast
import importlib
import sys
from collections import Counter
from importlib import invalidate_caches
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
PUBLIC_CATEGORIES = {
    "activation",
    "attention",
    "gemm_ops",
    "moe",
    "norm",
    "quant",
    "sampling",
    "tensor",
}
CLAMP_OWNER = (
    "vllm_hcu/model_executor/layers/fused_moe/experts/"
    "dpsk_v4_deep_gemm_moe.py"
)
ALLOWED_TOP_LEVEL = {
    (CLAMP_OWNER, "fuse_silu_mul_clamp_quant"),
    (CLAMP_OWNER, "fuse_silu_mul_clamp_quant_ep"),
}


def _attribute_parts(node: ast.Attribute) -> tuple[ast.expr, list[str]]:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    parts.reverse()
    return current, parts


class _LightOpVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        # Also recognize an injected/global ``lightop`` name so a call such as
        # ``lightop.op.foo()`` cannot evade the policy by omitting its import
        # from the scanned file.
        self.root_aliases: set[str] = {"lightop"}
        self.category_aliases: dict[str, str] = {}
        self.used: set[tuple[str, str]] = set()
        self.allowed_calls: list[tuple[str, str, int]] = []
        self.violations: list[str] = []
        self._violation_keys: set[tuple[int, str]] = set()
        self._functions: list[str] = []

    def _violate(self, node: ast.AST, detail: str) -> None:
        line = getattr(node, "lineno", 1)
        key = (line, detail)
        if key not in self._violation_keys:
            self._violation_keys.add(key)
            self.violations.append(f"{self.relative_path}:{line}: {detail}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        saved_root_aliases = self.root_aliases.copy()
        saved_category_aliases = self.category_aliases.copy()
        arguments = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            self.root_aliases.discard(argument.arg)
            self.category_aliases.pop(argument.arg, None)
        self._functions.append(node.name)
        self.generic_visit(node)
        self._functions.pop()
        self.root_aliases = saved_root_aliases
        self.category_aliases = saved_category_aliases

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        saved_root_aliases = self.root_aliases.copy()
        saved_category_aliases = self.category_aliases.copy()
        self.generic_visit(node)
        self.root_aliases = saved_root_aliases
        self.category_aliases = saved_category_aliases

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name
            if module == "lmslim" or module.startswith("lmslim."):
                self._violate(node, f"external LMSlim import {module!r}")
                continue
            if module == "lightop":
                local = alias.asname or "lightop"
                self.root_aliases.add(local)
                if self.relative_path != CLAMP_OWNER:
                    self._violate(
                        node,
                        "top-level 'lightop' import outside the clamp owner",
                    )
                continue
            if not module.startswith("lightop."):
                continue
            suffix = module.removeprefix("lightop.")
            if suffix in {"op", "gemmopt"}:
                self._violate(node, f"obsolete LightOp namespace {module!r}")
            elif suffix in PUBLIC_CATEGORIES:
                local = alias.asname or suffix
                self.category_aliases[local] = suffix
                if alias.asname is None:
                    self.root_aliases.add("lightop")
            else:
                self._violate(node, f"non-public LightOp module {module!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "lmslim" or module.startswith("lmslim."):
            self._violate(node, f"external LMSlim import from {module!r}")
        elif module == "lightop":
            for alias in node.names:
                if alias.name in PUBLIC_CATEGORIES:
                    self.category_aliases[alias.asname or alias.name] = alias.name
                else:
                    self._violate(
                        node,
                        f"moved top-level LightOp import {alias.name!r}",
                    )
        elif module.startswith("lightop."):
            suffix = module.removeprefix("lightop.")
            if suffix in {"op", "gemmopt"}:
                self._violate(node, f"obsolete LightOp namespace {module!r}")
            elif suffix not in PUBLIC_CATEGORIES:
                self._violate(node, f"non-public LightOp module {module!r}")
            else:
                for alias in node.names:
                    if alias.name == "*":
                        self._violate(node, f"wildcard import from {module!r}")
                    else:
                        self.used.add((module, alias.name))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root, parts = _attribute_parts(node)
        if isinstance(root, ast.Name) and parts:
            if root.id in self.root_aliases:
                namespace = parts[0]
                if namespace in {"op", "gemmopt"}:
                    self._violate(
                        node, f"obsolete LightOp namespace 'lightop.{namespace}'"
                    )
                elif namespace in PUBLIC_CATEGORIES:
                    if len(parts) >= 2:
                        self.used.add((f"lightop.{namespace}", parts[1]))
                else:
                    self._violate(
                        node, f"top-level LightOp attribute 'lightop.{namespace}'"
                    )
            elif root.id in self.category_aliases:
                self.used.add((f"lightop.{self.category_aliases[root.id]}", parts[0]))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            root, parts = _attribute_parts(node.func)
            if isinstance(root, ast.Name) and parts:
                if root.id in self.root_aliases and len(parts) >= 2:
                    if parts[0] in PUBLIC_CATEGORIES:
                        self.used.add((f"lightop.{parts[0]}", parts[1]))
                elif root.id in self.category_aliases:
                    self.used.add(
                        (f"lightop.{self.category_aliases[root.id]}", parts[0])
                    )

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_lightop_activation"
            and self.relative_path == CLAMP_OWNER
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.used.add(("lightop.activation", node.args[0].value))

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_lightop_clamp"
            and self.relative_path == CLAMP_OWNER
        ):
            if (
                len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                self.allowed_calls.append(
                    (self.relative_path, node.args[0].value, node.lineno)
                )
            else:
                self._violate(node, "clamp resolver call must use one literal name")

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.root_aliases
        ):
            allowed_resolver = (
                self.relative_path == CLAMP_OWNER
                and self._functions[-1:] == ["_lightop_clamp"]
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "name"
            )
            if not allowed_resolver:
                self._violate(node, "dynamic top-level LightOp attribute lookup")
        self.generic_visit(node)


def _clamp_resolver_is_exact(tree: ast.Module) -> bool:
    expected = {symbol for _, symbol in ALLOWED_TOP_LEVEL}
    resolvers = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_lightop_clamp"
    ]
    if len(resolvers) != 1:
        return False
    resolver = resolvers[0]
    for node in ast.walk(resolver):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "name"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], (ast.Set, ast.Tuple))
        ):
            values = {
                element.value
                for element in node.comparators[0].elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            }
            return values == expected and len(node.comparators[0].elts) == len(expected)
    return False


def _scan(root: Path) -> tuple[list[str], set[tuple[str, str]]]:
    violations: list[str] = []
    used: set[tuple[str, str]] = set()
    allowed_calls: list[tuple[str, str, int]] = []
    owner_tree: ast.Module | None = None
    repository = root.parent

    for path in sorted(root.rglob("*.py")):
        relative_path = path.relative_to(repository).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _LightOpVisitor(relative_path)
        visitor.visit(tree)
        violations.extend(visitor.violations)
        used.update(visitor.used)
        allowed_calls.extend(visitor.allowed_calls)
        if relative_path == CLAMP_OWNER:
            owner_tree = tree

    actual = Counter((path, symbol) for path, symbol, _ in allowed_calls)
    expected = Counter(ALLOWED_TOP_LEVEL)
    if actual != expected:
        details = {
            "missing": sorted((expected - actual).elements()),
            "extra": sorted((actual - expected).elements()),
        }
        violations.append(f"{CLAMP_OWNER}:1: clamp allowlist mismatch: {details}")
    if owner_tree is None or not _clamp_resolver_is_exact(owner_tree):
        violations.append(
            f"{CLAMP_OWNER}:1: clamp resolver guard is missing or broader than "
            f"{sorted(symbol for _, symbol in ALLOWED_TOP_LEVEL)}"
        )

    return sorted(violations), used


def scan_lightop_imports(root: Path) -> list[str]:
    return _scan(root)[0]


def categorized_symbols(root: Path) -> set[tuple[str, str]]:
    return _scan(root)[1]


def installed_public_exports() -> set[tuple[str, str]]:
    exports: set[tuple[str, str]] = set()
    for category in sorted(PUBLIC_CATEGORIES):
        module_name = f"lightop.{category}"
        module = importlib.import_module(module_name)
        public = getattr(module, "__all__", None)
        assert isinstance(public, (list, tuple)), f"{module_name} has no public __all__"
        exports.update((module_name, symbol) for symbol in public)
    return exports


@pytest.fixture
def isolated_lightop_modules():
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "lightop" or name.startswith("lightop.")
    }
    for name in tuple(original_modules):
        sys.modules.pop(name, None)
    invalidate_caches()
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "lightop" or name.startswith("lightop."):
                sys.modules.pop(name, None)
        invalidate_caches()
        sys.modules.update(original_modules)


def test_production_uses_public_lightop_categories_only() -> None:
    violations = scan_lightop_imports(REPOSITORY / "vllm_hcu")
    assert violations == []


def test_scanner_rejects_forbidden_forms_with_locations(tmp_path: Path) -> None:
    root = tmp_path / "vllm_hcu"
    owner = tmp_path / CLAMP_OWNER
    owner.parent.mkdir(parents=True)
    owner.write_text(
        "import lightop\n"
        "def _lightop_clamp(name):\n"
        "    if name not in {\"fuse_silu_mul_clamp_quant\", "
        "\"fuse_silu_mul_clamp_quant_ep\"}:\n"
        "        raise AttributeError(name)\n"
        "    return getattr(lightop, name)\n"
        "def clamp(*args):\n"
        "    return _lightop_clamp(\"fuse_silu_mul_clamp_quant\")(*args)\n"
        "def clamp_ep(*args):\n"
        "    return _lightop_clamp(\"fuse_silu_mul_clamp_quant_ep\")(*args)\n",
        encoding="utf-8",
    )
    mutation = root / "mutation.py"
    mutation.write_text(
        "import lmslim\n"
        "from lightop.op import old\n"
        "import lightop as lo\n"
        "lo.gemmopt.foo()\n"
        "lo.moved()\n"
        "from lightop import moved\n",
        encoding="utf-8",
    )

    violations = scan_lightop_imports(root)

    expected = {
        "vllm_hcu/mutation.py:1: external LMSlim import 'lmslim'",
        "vllm_hcu/mutation.py:2: obsolete LightOp namespace 'lightop.op'",
        "vllm_hcu/mutation.py:3: top-level 'lightop' import outside the clamp owner",
        "vllm_hcu/mutation.py:4: obsolete LightOp namespace 'lightop.gemmopt'",
        "vllm_hcu/mutation.py:5: top-level LightOp attribute 'lightop.moved'",
        "vllm_hcu/mutation.py:6: moved top-level LightOp import 'moved'",
    }
    assert expected <= set(violations)


@pytest.mark.usefixtures("isolated_lightop_modules")
def test_installed_category_exports_cover_production_symbols(monkeypatch) -> None:
    device_properties = SimpleNamespace(
        gcnArchName="gfx936:sramecc+:xnack-",
        multi_processor_count=80,
        name="HYGON HCU",
        major=9,
        minor=3,
        total_memory=64 << 30,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: device_properties,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    used = categorized_symbols(REPOSITORY / "vllm_hcu")
    exported = installed_public_exports()
    assert used - exported == set()
