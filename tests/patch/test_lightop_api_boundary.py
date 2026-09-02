# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Fail-closed production boundary for the public categorized LightOp API."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


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
        # Production must not use an injected/global LMSlim binding either;
        # requiring a local import would leave dynamic bindings unscanned.
        self.lmslim_root_aliases: set[str] = {"lmslim"}
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
        saved_lmslim_root_aliases = self.lmslim_root_aliases.copy()
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
            self.lmslim_root_aliases.discard(argument.arg)
            self.category_aliases.pop(argument.arg, None)
        self._functions.append(node.name)
        self.generic_visit(node)
        self._functions.pop()
        self.root_aliases = saved_root_aliases
        self.lmslim_root_aliases = saved_lmslim_root_aliases
        self.category_aliases = saved_category_aliases

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        saved_root_aliases = self.root_aliases.copy()
        saved_lmslim_root_aliases = self.lmslim_root_aliases.copy()
        saved_category_aliases = self.category_aliases.copy()
        self.generic_visit(node)
        self.root_aliases = saved_root_aliases
        self.lmslim_root_aliases = saved_lmslim_root_aliases
        self.category_aliases = saved_category_aliases

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name
            if module == "lmslim" or module.startswith("lmslim."):
                self.lmslim_root_aliases.add(
                    alias.asname or module.split(".", maxsplit=1)[0]
                )
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
            if root.id in self.lmslim_root_aliases:
                reference = ".".join((root.id, *parts))
                self._violate(
                    node, f"external LMSlim attribute {reference!r}"
                )
                return
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

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.lmslim_root_aliases:
            self._violate(node, f"external LMSlim root {node.id!r}")

    def visit_Call(self, node: ast.Call) -> None:
        direct_lmslim_call: str | None = None
        if isinstance(node.func, ast.Name):
            if node.func.id in self.lmslim_root_aliases:
                direct_lmslim_call = node.func.id
        elif isinstance(node.func, ast.Attribute):
            root, parts = _attribute_parts(node.func)
            if isinstance(root, ast.Name) and root.id in self.lmslim_root_aliases:
                direct_lmslim_call = ".".join((root.id, *parts))
        if direct_lmslim_call is not None:
            self._violate(
                node, f"external LMSlim call {direct_lmslim_call!r}"
            )
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return

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
        ):
            if (
                len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                self.used.add(("lightop.activation", node.args[0].value))
            else:
                self._violate(
                    node, "activation resolver call must use one literal name"
                )

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_lightop_clamp"
            and self.relative_path == CLAMP_OWNER
        ):
            if (
                len(node.args) == 1
                and not node.keywords
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
        ):
            owner = node.args[0].id
            if owner in self.root_aliases:
                allowed_resolver = (
                    self.relative_path == CLAMP_OWNER
                    and self._functions[-1:] == ["_lightop_clamp"]
                    and len(node.args) == 2
                    and not node.keywords
                    and isinstance(node.args[1], ast.Name)
                    and node.args[1].id == "name"
                )
                if not allowed_resolver:
                    self._violate(node, "dynamic top-level LightOp attribute lookup")
            elif owner in self.category_aliases:
                category = self.category_aliases[owner]
                if (
                    len(node.args) == 2
                    and not node.keywords
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    self.used.add((f"lightop.{category}", node.args[1].value))
                else:
                    allowed_activation_resolver = (
                        self.relative_path == CLAMP_OWNER
                        and self._functions[-1:] == ["_lightop_activation"]
                        and category == "activation"
                        and len(node.args) == 2
                        and not node.keywords
                        and isinstance(node.args[1], ast.Name)
                        and node.args[1].id == "name"
                    )
                    if not allowed_activation_resolver:
                        self._violate(
                            node,
                            "dynamic categorized LightOp attribute lookup",
                        )
        self.generic_visit(node)


def _resolver(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    resolvers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return resolvers[0] if len(resolvers) == 1 else None


def _resolver_body(resolver: ast.FunctionDef) -> list[ast.stmt]:
    body = list(resolver.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _has_exact_name_argument(resolver: ast.FunctionDef) -> bool:
    arguments = resolver.args
    return (
        not arguments.posonlyargs
        and len(arguments.args) == 1
        and arguments.args[0].arg == "name"
        and arguments.vararg is None
        and not arguments.kwonlyargs
        and arguments.kwarg is None
        and not arguments.defaults
        and not arguments.kw_defaults
    )


def _has_exact_lru_cache_decorator(resolver: ast.FunctionDef) -> bool:
    if len(resolver.decorator_list) != 1:
        return False
    decorator = resolver.decorator_list[0]
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "functools"
        and decorator.func.attr == "lru_cache"
        and not decorator.args
        and len(decorator.keywords) == 1
        and decorator.keywords[0].arg == "maxsize"
        and isinstance(decorator.keywords[0].value, ast.Constant)
        and decorator.keywords[0].value.value is None
    )


def _is_exact_getattr_return(
    node: ast.stmt,
    *,
    owner: str,
) -> bool:
    if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "getattr"
        and len(call.args) == 2
        and not call.keywords
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == owner
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "name"
    )


def _clamp_resolver_is_exact(tree: ast.Module) -> bool:
    expected = {symbol for _, symbol in ALLOWED_TOP_LEVEL}
    resolver = _resolver(tree, "_lightop_clamp")
    if (
        resolver is None
        or not _has_exact_name_argument(resolver)
        or not _has_exact_lru_cache_decorator(resolver)
    ):
        return False
    body = _resolver_body(resolver)
    if len(body) != 3:
        return False
    guard, import_node, return_node = body
    if not (
        isinstance(guard, ast.If)
        and not guard.orelse
        and isinstance(guard.test, ast.Compare)
        and isinstance(guard.test.left, ast.Name)
        and guard.test.left.id == "name"
        and len(guard.test.ops) == 1
        and isinstance(guard.test.ops[0], ast.NotIn)
        and len(guard.test.comparators) == 1
        and isinstance(guard.test.comparators[0], ast.Set)
    ):
        return False
    elements = guard.test.comparators[0].elts
    values = {
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    if values != expected or len(elements) != len(expected):
        return False
    if len(guard.body) != 1 or not isinstance(guard.body[0], ast.Raise):
        return False
    raised = guard.body[0]
    if not (
        isinstance(raised.exc, ast.Call)
        and isinstance(raised.exc.func, ast.Name)
        and raised.exc.func.id == "AttributeError"
        and len(raised.exc.args) == 1
        and not raised.exc.keywords
        and isinstance(raised.exc.args[0], ast.Name)
        and raised.exc.args[0].id == "name"
        and raised.cause is None
    ):
        return False
    if not (
        isinstance(import_node, ast.Import)
        and len(import_node.names) == 1
        and import_node.names[0].name == "lightop"
        and import_node.names[0].asname is None
    ):
        return False
    return _is_exact_getattr_return(return_node, owner="lightop")


def _activation_resolver_is_exact(tree: ast.Module) -> bool:
    resolver = _resolver(tree, "_lightop_activation")
    if (
        resolver is None
        or not _has_exact_name_argument(resolver)
        or not _has_exact_lru_cache_decorator(resolver)
    ):
        return False
    body = _resolver_body(resolver)
    if len(body) != 2:
        return False
    import_node, return_node = body
    return (
        isinstance(import_node, ast.ImportFrom)
        and import_node.module == "lightop"
        and import_node.level == 0
        and len(import_node.names) == 1
        and import_node.names[0].name == "activation"
        and import_node.names[0].asname is None
        and _is_exact_getattr_return(return_node, owner="activation")
    )


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
            f"{CLAMP_OWNER}:1: clamp resolver body is not the exact guarded "
            "top-level lookup"
        )
    if owner_tree is None or not _activation_resolver_is_exact(owner_tree):
        violations.append(
            f"{CLAMP_OWNER}:1: activation resolver body is not the exact "
            "categorized lookup"
        )

    return sorted(violations), used


def scan_lightop_imports(root: Path) -> list[str]:
    return _scan(root)[0]


def categorized_symbols(root: Path) -> set[tuple[str, str]]:
    return _scan(root)[1]


def installed_public_exports(
    required: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    exports: set[tuple[str, str]] = set()
    module_names = sorted({module_name for module_name, _ in required})
    for module_name in module_names:
        module = importlib.import_module(module_name)
        public = getattr(module, "__all__", None)
        assert isinstance(public, (list, tuple)), f"{module_name} has no public __all__"
        assert all(isinstance(name, str) for name in public), (
            f"{module_name}.__all__ contains non-string entries"
        )
        assert len(public) == len(set(public)), (
            f"{module_name}.__all__ contains duplicates"
        )
        module_required = {
            symbol for owner, symbol in required if owner == module_name
        }
        not_public = sorted(module_required - set(public))
        assert not not_public, (
            f"{module_name} required exports are not public: {not_public}"
        )
        not_bound = sorted(
            symbol for symbol in module_required if not hasattr(module, symbol)
        )
        assert not not_bound, (
            f"{module_name} public exports are not bound: {not_bound}"
        )
        exports.update((module_name, symbol) for symbol in public)
    return exports


def test_production_uses_public_lightop_categories_only() -> None:
    violations = scan_lightop_imports(REPOSITORY / "vllm_hcu")
    assert violations == []


def _write_mutation_owner(
    tmp_path: Path,
    *,
    clamp_body: str | None = None,
    activation_call: str | None = None,
    include_second_clamp_call: bool = True,
) -> Path:
    root = tmp_path / "vllm_hcu"
    owner = tmp_path / CLAMP_OWNER
    owner.parent.mkdir(parents=True)
    resolver = clamp_body or (
        "    if name not in {\"fuse_silu_mul_clamp_quant\", "
        "\"fuse_silu_mul_clamp_quant_ep\"}:\n"
        "        raise AttributeError(name)\n"
        "    import lightop\n"
        "    return getattr(lightop, name)\n"
    )
    second_call = (
        "def clamp_ep(*args):\n"
        "    return _lightop_clamp(\"fuse_silu_mul_clamp_quant_ep\")(*args)\n"
        if include_second_clamp_call
        else ""
    )
    activation_wrapper = (
        "def activation_wrapper(name):\n"
        f"    return _lightop_activation({activation_call})\n"
        if activation_call is not None
        else ""
    )
    owner.write_text(
        "import functools\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def _lightop_activation(name):\n"
        "    from lightop import activation\n"
        "    return getattr(activation, name)\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def _lightop_clamp(name):\n"
        f"{resolver}"
        "def clamp(*args):\n"
        "    return _lightop_clamp(\"fuse_silu_mul_clamp_quant\")(*args)\n"
        f"{second_call}"
        f"{activation_wrapper}",
        encoding="utf-8",
    )
    return root


def test_scanner_rejects_forbidden_forms_with_locations(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path)
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


def test_scanner_rejects_injected_direct_lmslim_call_with_location(
    tmp_path: Path,
) -> None:
    root = _write_mutation_owner(tmp_path)
    mutation = root / "injected_lmslim.py"
    mutation.write_text(
        "def invoke():\n"
        "    return lmslim.foo()\n",
        encoding="utf-8",
    )

    violations = scan_lightop_imports(root)

    assert (
        "vllm_hcu/injected_lmslim.py:2: external LMSlim call 'lmslim.foo'"
        in violations
    )


@pytest.mark.parametrize(
    ("clamp_body", "include_second_clamp_call"),
    [
        pytest.param(
            "    if name not in {\"fuse_silu_mul_clamp_quant\", "
            "\"fuse_silu_mul_clamp_quant_ep\"}:\n"
            "        raise AttributeError(name)\n"
            "    import lightop\n"
            "    name = \"fuse_silu_mul_quant\"\n"
            "    return getattr(lightop, name)\n",
            True,
            id="reassigned-name",
        ),
        pytest.param(
            "    if name not in {\"fuse_silu_mul_clamp_quant\", "
            "\"fuse_silu_mul_clamp_quant_ep\"}:\n"
            "        raise AttributeError(name)\n"
            "    import lightop\n"
            "    if name == \"fuse_silu_mul_clamp_quant\":\n"
            "        pass\n"
            "    return getattr(lightop, name)\n",
            True,
            id="extra-branch",
        ),
        pytest.param(
            "    if name not in {\"fuse_silu_mul_clamp_quant\", "
            "\"fuse_silu_mul_clamp_quant_ep\", \"broadened\"}:\n"
            "        raise AttributeError(name)\n"
            "    import lightop\n"
            "    return getattr(lightop, name)\n",
            True,
            id="broadened-guard",
        ),
        pytest.param(
            None,
            False,
            id="stale-missing-call",
        ),
    ],
)
def test_scanner_rejects_mutated_clamp_boundary(
    tmp_path: Path,
    clamp_body: str | None,
    include_second_clamp_call: bool,
) -> None:
    root = _write_mutation_owner(
        tmp_path,
        clamp_body=clamp_body,
        include_second_clamp_call=include_second_clamp_call,
    )

    violations = scan_lightop_imports(root)

    assert any(
        "clamp resolver" in item or "clamp allowlist" in item
        for item in violations
    )


def test_scanner_rejects_nonliteral_activation_resolver_call(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path, activation_call="name")

    violations = scan_lightop_imports(root)

    assert any(
        "activation resolver call must use one literal name" in item
        for item in violations
    )


def test_scanner_rejects_dynamic_category_getattr(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path)
    mutation = root / "dynamic_category.py"
    mutation.write_text(
        "from lightop import activation\n"
        "def resolve(name):\n"
        "    return getattr(activation, name)\n",
        encoding="utf-8",
    )

    violations = scan_lightop_imports(root)

    assert any(
        "dynamic categorized LightOp attribute lookup" in item
        for item in violations
    )


def test_scanner_records_literal_category_getattr(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path)
    mutation = root / "literal_category.py"
    mutation.write_text(
        "from lightop import quant\n"
        "kernel = getattr(quant, \"literal_quant_kernel\")\n",
        encoding="utf-8",
    )

    violations = scan_lightop_imports(root)
    used = categorized_symbols(root)

    assert violations == []
    assert ("lightop.quant", "literal_quant_kernel") in used


def test_installed_category_exports_cover_production_symbols() -> None:
    used = categorized_symbols(REPOSITORY / "vllm_hcu")
    env = dict(os.environ)
    env["LIGHTOP_REQUIRED_EXPORTS"] = json.dumps(sorted(used))
    env["ROCM_HOME"] = env.get("ROCM_HOME", env.get("ROCM_PATH", "/opt/dtk"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, json, os, subprocess, torch; "
            "from types import SimpleNamespace; "
            "subprocess.run = lambda *_a, **_k: "
            "SimpleNamespace(stdout="
            "'Name: gfx936\\nCompute Unit: 80\\n' "
            "if _k.get('text') else b'26.04'); "
            "torch.cuda.get_device_properties = lambda *_a, **_k: "
            "SimpleNamespace(gcnArchName='gfx936:sramecc+:xnack-', "
            "multi_processor_count=80, name='HYGON HCU', major=9, minor=3, "
            "total_memory=64 << 30); "
            "torch.cuda.current_device = lambda: 0; "
            "required=json.loads(os.environ['LIGHTOP_REQUIRED_EXPORTS']); "
            "modules={name: importlib.import_module(name) "
            "for name, _symbol in required}; "
            "missing_public=[(name, symbol) for name, symbol in required "
            "if symbol not in modules[name].__all__]; "
            "missing_bound=[(name, symbol) for name, symbol in required "
            "if not hasattr(modules[name], symbol)]; "
            "assert not missing_public, f'not public: {missing_public}'; "
            "assert not missing_bound, f'not bound: {missing_bound}'",
        ],
        cwd=REPOSITORY,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_dynamic_export_audit_rejects_public_but_unbound_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling = SimpleNamespace(
        __all__=["top_k_top_p_sampling_from_probs"],
    )
    def import_sampling(name: str) -> SimpleNamespace:
        assert name == "lightop.sampling"
        return sampling

    monkeypatch.setattr(importlib, "import_module", import_sampling)

    with pytest.raises(AssertionError, match="public exports are not bound"):
        installed_public_exports(
            {("lightop.sampling", "top_k_top_p_sampling_from_probs")}
        )
