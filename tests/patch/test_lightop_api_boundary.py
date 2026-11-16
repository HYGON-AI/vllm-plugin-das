# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Source policy for public LightOp imports, with an exact legacy fallback.

The PCP compatibility owner permits reviewed static imports and one existing
fixed-target vLLM import. Other loader and namespace capabilities are rejected
at their source, without alias analysis. This is a source policy, not a Python
sandbox for external modules that mutate interpreter state.
"""

from __future__ import annotations

import ast
import copy
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
# This versioned fast path has a maintained fallback when older wheels omit it.
# If installed, it must still be exported by the public category.
OPTIONAL_CATEGORIZED_FALLBACKS = {
    (
        "lightop.attention",
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp",
    ): (
        "op",
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp",
    ),
}
CLAMP_OWNER = (
    "vllm_hcu/model_executor/layers/fused_moe/experts/"
    "dpsk_v4_deep_gemm_moe.py"
)
ALLOWED_TOP_LEVEL = {
    (CLAMP_OWNER, "fuse_silu_mul_clamp_quant"),
    (CLAMP_OWNER, "fuse_silu_mul_clamp_quant_ep"),
}
LEGACY_TOP_LEVEL_IMPORTS = {
    (
        "vllm_hcu/patch/worker/op_opt/patch_custom_ops.py",
        "gather_and_maybe_dequant_cache",
        "lightop_gather",
    ),
}
LEGACY_TOP_LEVEL_MODULE_IMPORTS = {
    (
        "vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_pcp.py",
        "op",
        "legacy_native",
    ),
}
LEGACY_MODULE_OWNERS = {
    path for path, _symbol, _alias in LEGACY_TOP_LEVEL_MODULE_IMPORTS
}
LEGACY_MODULE_ALLOWED_SYMBOLS = {
    (
        "vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_pcp.py",
        "legacy_native",
    ): frozenset({
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp",
    }),
}


# Reviewed syntax for the sole legacy dispatcher fallback. Comments and source
# positions are intentionally ignored; executable changes require a new audit.
PCP_RESOLVER_SOURCE = """\
@functools.cache
def _pcp_insert_op():
    try:
        from lightop import attention as native
        op = native.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp
    except (ImportError, AttributeError):
        op = None
    if callable(op):
        return op
    try:
        from lightop import op as legacy_native
        op = legacy_native.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp
    except (ImportError, AttributeError):
        return None
    return op if callable(op) else None
"""


def _constant_string(node: ast.AST) -> str | None:
    """Fold literal concatenation and plain literal-only f-strings safely."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _constant_string(node.left), _constant_string(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.FormattedValue):
        if node.conversion == -1 and node.format_spec is None:
            return _constant_string(node.value)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            part = _constant_string(value)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


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
        # Global names are resolved when a function is called, not when its
        # body is defined. Seed configured aliases so helpers appearing before
        # the approved import cannot evade the legacy boundary.
        self.legacy_module_aliases: dict[str, frozenset[str]] = {
            alias: symbols
            for (path, alias), symbols in LEGACY_MODULE_ALLOWED_SYMBOLS.items()
            if path == relative_path
        }
        self.used: set[tuple[str, str]] = set()
        self.allowed_calls: list[tuple[str, str, int]] = []
        self.allowed_legacy_imports: list[tuple[str, str, str | None, int]] = []
        self.violations: list[str] = []
        self._violation_keys: set[tuple[int, str]] = set()
        self._functions: list[str] = []
        self._audited_legacy_nodes: set[int] = set()

    def _violate(self, node: ast.AST, detail: str) -> None:
        line = getattr(node, "lineno", 1)
        key = (line, detail)
        if key not in self._violation_keys:
            self._violation_keys.add(key)
            self.violations.append(f"{self.relative_path}:{line}: {detail}")


    def visit_Module(self, node: ast.Module) -> None:
        if self.relative_path in LEGACY_MODULE_OWNERS:
            resolver = _resolver(node, "_pcp_insert_op")
            if resolver is None or ast.dump(resolver) != ast.dump(
                ast.parse(PCP_RESOLVER_SOURCE).body[0]
            ):
                self._violate(node, "PCP resolver is not the exact audited fallback")
            else:
                self._audited_legacy_nodes = {id(n) for n in ast.walk(resolver)}
            self._audit_legacy_owner(node)
        self.generic_visit(node)

    def _audit_legacy_owner(self, tree: ast.Module) -> None:
        # Static imports plus one existing, fixed-target vLLM import suffice
        # for this compatibility owner. Reject other dynamic capabilities at
        # their source instead of implementing aliases or Python data flow.
        static_modules = {
            "__future__", "functools", "importlib", "contextvars", "types", "torch",
            "vllm.logger", "._common", "lightop", "vllm_hcu.platforms",
            "vllm.forward_context", "vllm.models.deepseek_v4",
            "vllm.models.deepseek_v4.attention",
            "vllm.models.deepseek_v4.compressor",
            "vllm.v1.attention.backends.mla.sparse_swa",
            "vllm_hcu.model_executor.layers.attention.pcp",
            "vllm_hcu.v1.glm53_pcp",
        }
        forbidden = {
            "importlib", "import_module", "__import__", "globals", "locals",
            "vars", "sys", "builtins", "__builtins__", "__globals__",
            "func_globals", "f_globals", "f_locals", "f_builtins",
            "eval", "exec", "compile", "load_exact_module", "__dict__",
            "__getattribute__", "__setattr__", "__delattr__",
        }
        protected = {
            "getattr", "callable", "ImportError", "AttributeError",
            "functools", "_pcp_insert_op", "TARGET_MODULE", "load_exact_module",
            "_MARKER",
        }
        reflection = {"getattr", "setattr", "delattr"}
        forbidden |= reflection
        protected |= reflection
        # Existing wrapper/marker operations are the only reviewed uses of
        # dynamic attribute names. Other lookups/writes must use constant
        # strings; do not try to evaluate arbitrary string-building calls.
        reviewed_reflection = {
            ast.dump(stmt.value) for stmt in ast.parse("""
getattr(cls, _MARKER, False)
getattr(getattr(cls, name), _MARKER, False)
getattr(cls, name)
getattr(rocm.rocm_sparse_attn_prefill, _MARKER, False)
getattr(quant, _MARKER, False)
setattr(local_indexer_quant, _MARKER, True)
setattr(wrapper, _MARKER, True)
setattr(cls, name, wrapper)
setattr(indexer_wrapper, _MARKER, True)
setattr(sparse_prefill, _MARKER, True)
setattr(cls, _MARKER, True)
""").body
        }
        # These exceptions apply only to the existing calls in the one
        # top-level patch entry point, not matching calls appended elsewhere.
        entry = _resolver(tree, "apply_to_module")
        reflection_nodes: set[int] = set()
        for pattern in reviewed_reflection:
            matches = [
                node for node in ast.walk(entry) if isinstance(node, ast.Call)
                and ast.dump(node) == pattern
            ] if entry is not None else []
            if len(matches) == 1:
                reflection_nodes.add(id(matches[0].func))
        parents = {
            id(child): parent for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        expected_import = ast.dump(ast.parse(
            'importlib.import_module("vllm.models.deepseek_v4.attention")'
        ).body[0].value)
        reviewed_loader_nodes = {
            id(child) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.dump(node) == expected_import
            for child in ast.walk(node)
        }
        # The imported coordinator helper is a loader too. Permit only its
        # existing call shape, with the target pinned to the reviewed module.
        expected_loader = ast.dump(ast.parse(
            "load_exact_module(TARGET_MODULE, module)"
        ).body[0].value)
        reviewed_exact_loaders = {
            id(node.func) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.dump(node) == expected_loader
        }
        expected_target = ast.dump(ast.parse(
            'TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"'
        ).body[0])
        targets = [node for node in tree.body if ast.dump(node) == expected_target]
        if reviewed_exact_loaders and len(targets) != 1:
            self._violate(tree, "module loader requires the exact TARGET_MODULE")
        reviewed_target_nodes = {
            id(child) for node in targets for child in ast.walk(node)
        }
        expected_marker = ast.dump(ast.parse(
            '_MARKER = "_vllm_hcu_deepseek_v4_pcp_applied"'
        ).body[0])
        markers = [node for node in tree.body if ast.dump(node) == expected_marker]
        if reflection_nodes and len(markers) != 1:
            self._violate(tree, "dynamic reflection requires the exact patch marker")
        reviewed_target_nodes.update(
            id(child) for node in markers for child in ast.walk(node)
        )
        for node in ast.walk(tree):
            parent = parents.get(id(node))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [a.name for a in node.names] if isinstance(node, ast.Import)
                    else ["." * node.level + (node.module or "")]
                )
                if any(module not in static_modules for module in modules):
                    self._violate(node, "legacy owner requires reviewed static imports")
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module in {"functools", "importlib"}
                ) or (
                    isinstance(node, ast.Import) and any(
                        a.name == "functools" and a.asname for a in node.names
                    )
                ):
                    self._violate(node, "legacy owner requires plain dependency imports")
            literal = _constant_string(node)
            token = literal
            if isinstance(node, ast.Name):
                token = node.id
            elif isinstance(node, ast.Attribute):
                token = node.attr
            elif isinstance(node, ast.alias):
                token = node.name
            plain_importlib = (
                isinstance(node, ast.alias) and node.name == "importlib"
                and node.asname is None and isinstance(parent, ast.Import)
                and len(parent.names) == 1 and parent in tree.body
            )
            plain_exact_loader = (
                isinstance(node, ast.alias) and node.name == "load_exact_module"
                and node.asname is None and isinstance(parent, ast.ImportFrom)
                and parent.level == 1 and parent.module == "_common"
            )
            direct_reflection = (
                isinstance(node, ast.Name) and node.id in reflection
                and isinstance(parent, ast.Call) and parent.func is node
            )
            if direct_reflection and (
                len(parent.args) < 2
                or _constant_string(parent.args[1]) is None
            ) and id(node) not in reflection_nodes:
                self._violate(node, "nonliteral reflective access in legacy owner")
            if (
                isinstance(token, str) and token in forbidden
                and id(node) not in reviewed_loader_nodes
                and not plain_importlib and not plain_exact_loader
                and not direct_reflection
                and id(node) not in reviewed_exact_loaders
            ):
                self._violate(node, "dynamic namespace capability in legacy owner")
            if literal == "_pcp_insert_op":
                self._violate(node, "indirect PCP resolver reference in legacy owner")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id == "_pcp_insert_op" and not (
                    isinstance(parent, ast.Call) and parent.func is node
                ):
                    self._violate(node, "bare PCP fallback function escape")
                if node.id == "functools" and not (
                    isinstance(parent, ast.Attribute) and parent.value is node
                    and parent.attr in {"cache", "wraps"}
                    and isinstance(parent.ctx, ast.Load)
                    and (parent.attr == "wraps"
                         or id(parent) in self._audited_legacy_nodes)
                ):
                    self._violate(node, "PCP fallback decorator escape")
            if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
                if node.attr in protected | {"cache", "wraps"}:
                    self._violate(node, "PCP fallback dependency mutation")
            binding = None
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                binding = node.id
            elif isinstance(node, (ast.arg, ast.keyword)):
                binding = node.arg
            elif isinstance(node, ast.alias):
                binding = node.asname or node.name.split(".")[0]
            elif isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ast.ExceptHandler, ast.MatchAs, ast.MatchStar,
            )):
                binding = node.name
            elif isinstance(node, ast.MatchMapping):
                binding = node.rest
            if (
                binding in protected and id(node) not in self._audited_legacy_nodes
                and id(node) not in reviewed_target_nodes and not plain_exact_loader
            ):
                if not (isinstance(parent, ast.Import) and binding == "functools"
                        and node.name == "functools" and node.asname is None
                        and parent in tree.body):
                    self._reject_protected_binding(node, binding)

    def _reject_protected_binding(self, node: ast.AST, name: str) -> None:
        if name == "getattr":
            self._violate(node, "builtin 'getattr' is shadowed in legacy owner")
        else:
            self._violate(node, f"PCP fallback dependency {name!r} is rebound")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        saved_root_aliases = self.root_aliases.copy()
        saved_lmslim_root_aliases = self.lmslim_root_aliases.copy()
        saved_category_aliases = self.category_aliases.copy()
        saved_legacy_module_aliases = self.legacy_module_aliases.copy()
        arguments = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)

        # Decorators, defaults, and annotations are evaluated in the outer
        # scope, before parameter names shadow imported aliases in the body.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for parameter in getattr(node, "type_params", ()):
            self.visit(parameter)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)

        for argument in arguments:
            self.root_aliases.discard(argument.arg)
            self.lmslim_root_aliases.discard(argument.arg)
            self.category_aliases.pop(argument.arg, None)
            self.legacy_module_aliases.pop(argument.arg, None)
        self._functions.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self._functions.pop()
        self.root_aliases = saved_root_aliases
        self.lmslim_root_aliases = saved_lmslim_root_aliases
        self.category_aliases = saved_category_aliases
        self.legacy_module_aliases = saved_legacy_module_aliases

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        saved_root_aliases = self.root_aliases.copy()
        saved_lmslim_root_aliases = self.lmslim_root_aliases.copy()
        saved_category_aliases = self.category_aliases.copy()
        saved_legacy_module_aliases = self.legacy_module_aliases.copy()
        self.generic_visit(node)
        self.root_aliases = saved_root_aliases
        self.lmslim_root_aliases = saved_lmslim_root_aliases
        self.category_aliases = saved_category_aliases
        self.legacy_module_aliases = saved_legacy_module_aliases

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
        for alias in node.names:
            if (
                alias.name == "*"
                and self.relative_path in LEGACY_MODULE_OWNERS
            ):
                self._violate(
                    node,
                    "wildcard import can shadow builtins in legacy owner",
                )
        if module == "lmslim" or module.startswith("lmslim."):
            self._violate(node, f"external LMSlim import from {module!r}")
        elif node.level and (
            module == "lightop" or module.startswith("lightop.")
        ):
            self._violate(node, "relative LightOp import is not a public import")
        elif module == "lightop":
            for alias in node.names:
                if alias.name in PUBLIC_CATEGORIES:
                    self.category_aliases[alias.asname or alias.name] = alias.name
                elif node.level == 0 and (
                    self.relative_path,
                    alias.name,
                    alias.asname,
                ) in LEGACY_TOP_LEVEL_IMPORTS | LEGACY_TOP_LEVEL_MODULE_IMPORTS:
                    self.allowed_legacy_imports.append(
                        (
                            self.relative_path,
                            alias.name,
                            alias.asname,
                            node.lineno,
                        )
                    )
                    module_key = (
                        self.relative_path,
                        alias.asname or alias.name,
                    )
                    allowed_symbols = LEGACY_MODULE_ALLOWED_SYMBOLS.get(module_key)
                    if (
                        allowed_symbols is not None
                        and id(node) not in self._audited_legacy_nodes
                    ):
                        self._violate(
                            node, "legacy import outside exact PCP fallback",
                        )
                    if allowed_symbols is not None:
                        self.legacy_module_aliases[alias.asname or alias.name] = (
                            allowed_symbols
                        )
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
            elif root.id in self.legacy_module_aliases:
                allowed = self.legacy_module_aliases[root.id]
                if (
                    id(node) not in self._audited_legacy_nodes
                    or not isinstance(node.ctx, ast.Load)
                    or len(parts) != 1
                    or parts[0] not in allowed
                ):
                    reference = ".".join((root.id, *parts))
                    self._violate(
                        node,
                        f"unapproved legacy LightOp module attribute {reference!r}",
                    )
                # Do not visit the root Name: this exact attribute lookup is
                # the only non-getattr use permitted for the legacy module.
                return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.lmslim_root_aliases:
            self._violate(node, f"external LMSlim root {node.id!r}")
        if node.id in self.legacy_module_aliases:
            self._violate(
                node,
                f"bare legacy LightOp module use {node.id!r}",
            )

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
            elif owner in self.legacy_module_aliases:
                # No getattr exemption: a rebound builtin can ignore even an
                # allowlisted literal. Only the audited direct attribute is
                # permitted. Still inspect every remaining argument.
                self._violate(
                    node, "unapproved legacy LightOp module attribute lookup",
                )
                self.visit(node.func)
                for argument in node.args[1:]:
                    self.visit(argument)
                for keyword in node.keywords:
                    self.visit(keyword.value)
                return
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


def _resolver_matches(tree: ast.Module, name: str, body: str) -> bool:
    resolver = _resolver(tree, name)
    if resolver is None:
        return False
    resolver = copy.deepcopy(resolver)
    resolver.body = _resolver_body(resolver)
    # Type annotations do not change these import/dispatch boundaries.
    for argument in resolver.args.args:
        argument.annotation = None
    resolver.returns = None
    expected = ast.parse(
        f"@functools.lru_cache(maxsize=None)\ndef {name}(name):\n" + body
    ).body[0]
    return ast.dump(resolver) == ast.dump(expected)


def _clamp_resolver_is_exact(tree: ast.Module) -> bool:
    return _resolver_matches(tree, "_lightop_clamp", (
        '    if name not in {"fuse_silu_mul_clamp_quant", '
        '"fuse_silu_mul_clamp_quant_ep"}:\n'
        '        raise AttributeError(name)\n'
        '    import lightop\n'
        '    return getattr(lightop, name)\n'
    ))


def _activation_resolver_is_exact(tree: ast.Module) -> bool:
    return _resolver_matches(tree, "_lightop_activation", (
        '    from lightop import activation\n'
        '    return getattr(activation, name)\n'
    ))


def _scan(root: Path) -> tuple[list[str], set[tuple[str, str]]]:
    violations: list[str] = []
    used: set[tuple[str, str]] = set()
    allowed_calls: list[tuple[str, str, int]] = []
    allowed_legacy_imports: list[tuple[str, str, str | None, int]] = []
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
        allowed_legacy_imports.extend(visitor.allowed_legacy_imports)
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
    actual_legacy = Counter(
        (path, symbol, alias)
        for path, symbol, alias, _ in allowed_legacy_imports
    )
    expected_legacy = Counter(
        LEGACY_TOP_LEVEL_IMPORTS | LEGACY_TOP_LEVEL_MODULE_IMPORTS
    )
    if actual_legacy != expected_legacy:
        details = {
            "missing": sorted((expected_legacy - actual_legacy).elements()),
            "extra": sorted((actual_legacy - expected_legacy).elements()),
        }
        violations.append(
            "vllm_hcu:1: legacy top-level LightOp import allowlist mismatch: "
            f"{details}"
        )
    expected_module_aliases = {
        (path, alias or symbol)
        for path, symbol, alias in LEGACY_TOP_LEVEL_MODULE_IMPORTS
    }
    actual_module_aliases = set(LEGACY_MODULE_ALLOWED_SYMBOLS)
    if actual_module_aliases != expected_module_aliases:
        details = {
            "missing": sorted(expected_module_aliases - actual_module_aliases),
            "extra": sorted(actual_module_aliases - expected_module_aliases),
        }
        violations.append(
            "vllm_hcu:1: legacy LightOp module symbol allowlist mismatch: "
            f"{details}"
        )
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
    for relative_path, symbol, alias in LEGACY_TOP_LEVEL_IMPORTS:
        legacy_owner = tmp_path / relative_path
        legacy_owner.parent.mkdir(parents=True, exist_ok=True)
        legacy_owner.write_text(
            f"from lightop import {symbol} as {alias}\n",
            encoding="utf-8",
        )
    for relative_path, symbol, alias in LEGACY_TOP_LEVEL_MODULE_IMPORTS:
        legacy_owner = tmp_path / relative_path
        legacy_owner.parent.mkdir(parents=True, exist_ok=True)
        with legacy_owner.open("a", encoding="utf-8") as stream:
            stream.write("import functools\n" + PCP_RESOLVER_SOURCE)
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


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize(
    ("relative_path", "symbol", "alias"),
    sorted(LEGACY_TOP_LEVEL_IMPORTS | LEGACY_TOP_LEVEL_MODULE_IMPORTS),
)
def test_scanner_rejects_relative_legacy_imports(
    tmp_path: Path,
    level: int,
    relative_path: str,
    symbol: str,
    alias: str,
) -> None:
    root = _write_mutation_owner(tmp_path)
    owner = tmp_path / relative_path
    owner.write_text(
        f"from {'.' * level}lightop import {symbol} as {alias}\n",
        encoding="utf-8",
    )

    violations = scan_lightop_imports(root)

    assert any(
        "legacy top-level LightOp import allowlist mismatch" in item
        for item in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "legacy_native.any_private_kernel()",
        "alias = legacy_native",
        "consume(legacy_native)",
        "legacy_native.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp",
        "legacy_native.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp = None",
        "getattr(legacy_native, 'fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp', None)",
        "def helper(legacy_native=legacy_native):\n    pass",
        "def getattr(*args):\n    pass",
        "from helpers import resolve as getattr",
        "from helpers import *",
        "globals()['getattr'] = replacement",
        "import builtins as bi; bi.getattr = replacement",
        'import sys; sys.modules["builtins"].getattr = replacement',
        '_can_use_local_queries.__globals__["_pcp_insert_op"] = replacement',
        'lookup = getattr; lookup(helper, "__globals__")["_pcp_insert_op"] = replacement',
        "import importlib; importlib.import_module(module_name)",
        "from importlib import import_module as load; load(module_name)",
        "load = __import__; load(module_name)",
        'getattr(importlib, "import_module")(module_name)',
        'lookup = getattr; load = lookup(importlib, "import_module"); load(module_name)',
        'importlib.import_module("vllm_hcu.patch.worker.core_fix.patch_deepseek_v4_pcp")',
        "import vllm_hcu.patch.worker.core_fix.patch_deepseek_v4_pcp as owner",
        "from . import patch_deepseek_v4_pcp as owner",
        "owner._pcp_insert_op = replacement",
        "functools.cache = replacement",
        "cache = functools.cache",
        "_pcp_insert_op = replacement",
        "_pcp_insert_op.__wrapped__ = replacement",
        "function = _pcp_insert_op",
        "callable = replacement",
    ],
)
def test_legacy_owner_rejects_review_regressions(
    tmp_path: Path, source: str,
) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write(f"\n{source}\n")

    assert scan_lightop_imports(root)


@pytest.mark.parametrize("source", [
    "from ._common import require_callable",
    'import importlib\nimportlib.import_module("vllm.models.deepseek_v4.attention")',
])
def test_legacy_owner_allows_reviewed_imports(tmp_path: Path, source: str) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write(f"\n{source}\n")

    assert scan_lightop_imports(root) == []


@pytest.mark.parametrize("source", [
    "load_exact_module(__name__, None)",
    "load_exact_module(TARGET_MODULE, None)",
    "load_exact_module(target=__name__, module=None)",
    "load_exact_module(*(__name__, None))",
    "load = load_exact_module; load(__name__, None)",
    "from ._common import load_exact_module as load; load(__name__, None)",
    'getattr(common, "load_exact_module")(__name__, None)',
])
def test_legacy_owner_restricts_exact_module_loader(tmp_path: Path, source: str) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write(f"\n{source}\n")

    assert scan_lightop_imports(root)


@pytest.mark.parametrize("source", [
    "load_exact_module(__name__, None).__dict__.update(_pcp_insert_op=lambda: None)",
    "owner.__dict__.update(_pcp_insert_op=lambda: None)",
    "owner.__dict__.update(callable=replacement)",
    "owner.__dict__.update(TARGET_MODULE=__name__)",
    "TARGET_MODULE = __name__",
])
def test_legacy_owner_rejects_indirect_dependency_updates(
    tmp_path: Path, source: str,
) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write(f"\n{source}\n")

    assert scan_lightop_imports(root)


@pytest.mark.parametrize("target", [
    'TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"',
    "TARGET_MODULE = __name__",
    "",
])
def test_legacy_owner_pins_exact_loader_target(tmp_path: Path, target: str) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write(
            "\nfrom ._common import load_exact_module\n"
            f"{target}\n"
            "def resolve(module):\n"
            "    return load_exact_module(TARGET_MODULE, module)\n"
        )

    violations = scan_lightop_imports(root)

    assert (violations == []) == (
        target == 'TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"'
    )


@pytest.mark.parametrize("source", [
    'getattr(_can_use_local_queries, "__glob" + "als__")["_pcp_" + "insert_op"] = lambda: None',
    'owner.__dict__["_pcp_" + "insert_op"] = lambda: None',
    'getattr(helper, "__glo" + ("bals" + "__"))',
    'key = "_pcp_" + "insert_" + "op"; owner.__dict__[key] = replacement',
    'lookup = getattr; lookup(helper, "__glob" + "als__")["_pcp_" + "insert_op"] = replacement',
    'getattr(helper, f"{\'__glob\'}{\'als__\'}")["_pcp_" + "insert_op"] = replacement',
])
def test_legacy_owner_rejects_computed_protected_names(
    tmp_path: Path, source: str,
) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write(f"\n{source}\n")

    assert scan_lightop_imports(root)


def test_legacy_owner_allows_benign_string_concatenation(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write('\nlabel = "public_" + "kernel"\n')

    assert scan_lightop_imports(root) == []


@pytest.mark.parametrize("source", [
    'getattr(_can_use_local_queries, "__gloXbals__".replace("X", ""))["_pcp_Xinsert_op".replace("X", "")] = lambda: None',
    'getattr(helper, attribute_name)',
    'getattr(cls, name)',
    '_MARKER = "__gloXbals__".replace("X", "")',
    'lookup = getattr; lookup(helper, attribute_name)',
    'helper.__getattribute__(attribute_name)',
    'setattr(owner, attribute_name, replacement)',
    'delattr(owner, attribute_name)',
    'namespace = owner.__dict__; namespace[key_name] = replacement',
])
def test_legacy_owner_rejects_dynamic_reflection(tmp_path: Path, source: str) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write(f"\n{source}\n")

    assert scan_lightop_imports(root)


def test_legacy_owner_allows_literal_attribute_lookup(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    with (tmp_path / relative_path).open("a", encoding="utf-8") as stream:
        stream.write('\ngetattr(layer, "head_" + "dim", None)\n')

    assert scan_lightop_imports(root) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("@functools.cache", "@replacement"),
        ("def _pcp_insert_op():", "def _pcp_insert_op(getattr=replacement):"),
        ("from lightop import op as legacy_native",
         "from .lightop import op as legacy_native"),
        ("op = legacy_native.", "op = other."),
        ("    if callable(op):", "    if False:"),
        ("    return op if callable(op) else None", "    return replacement"),
        ("    try:\n        from lightop import op",
         "    replacement()\n    try:\n        from lightop import op"),
    ],
)
def test_scanner_rejects_changed_pcp_resolver(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    owner = tmp_path / relative_path
    source = owner.read_text(encoding="utf-8")
    assert old in source
    owner.write_text(source.replace(old, new, 1), encoding="utf-8")

    assert any(
        "PCP resolver is not the exact audited fallback" in item
        for item in scan_lightop_imports(root)
    )


def test_scanner_allows_pcp_resolver_comments(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    owner = tmp_path / relative_path
    owner.write_text(
        owner.read_text(encoding="utf-8").replace(
            "def _pcp_insert_op():",
            "def _pcp_insert_op():\n    # Documentation is not executable.",
        ),
        encoding="utf-8",
    )

    assert scan_lightop_imports(root) == []


@pytest.mark.parametrize("module", ["lightop", "lightop.attention"])
def test_scanner_rejects_relative_category_import(
    tmp_path: Path,
    module: str,
) -> None:
    root = _write_mutation_owner(tmp_path)
    (root / "relative.py").write_text(
        f"from .{module} import attention\n", encoding="utf-8",
    )

    assert any(
        "relative LightOp import is not a public import" in item
        for item in scan_lightop_imports(root)
    )


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_scanner_checks_function_type_parameter_bounds(tmp_path: Path) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    owner = tmp_path / relative_path
    with owner.open("a", encoding="utf-8") as stream:
        stream.write(
            "\ndef forbidden[T: legacy_native.any_private_kernel]():\n"
            "    pass\n"
        )

    assert any(
        "unapproved legacy LightOp module attribute" in item
        for item in scan_lightop_imports(root)
    )


@pytest.mark.parametrize("definition", ["def", "async def"])
@pytest.mark.parametrize(
    ("decorator", "detail"),
    [
        (
            "legacy_native.any_private_kernel",
            "unapproved legacy LightOp module attribute",
        ),
        (
            "getattr(legacy_native, name)",
            "unapproved legacy LightOp module attribute lookup",
        ),
        (
            "getattr(lightop, name)",
            "dynamic top-level LightOp attribute lookup",
        ),
        (
            "getattr(attention, name)",
            "dynamic categorized LightOp attribute lookup",
        ),
    ],
)
def test_scanner_rejects_forbidden_function_decorators(
    tmp_path: Path,
    definition: str,
    decorator: str,
    detail: str,
) -> None:
    root = _write_mutation_owner(tmp_path)
    relative_path, _, _ = next(iter(LEGACY_TOP_LEVEL_MODULE_IMPORTS))
    owner = tmp_path / relative_path
    with owner.open("a", encoding="utf-8") as stream:
        stream.write(
            "from lightop import attention\n"
            f"@{decorator}\n"
            f"{definition} forbidden(legacy_native, lightop, attention):\n"
            "    pass\n"
        )
    decorator_line = next(
        index
        for index, line in enumerate(owner.read_text(encoding="utf-8").splitlines(), 1)
        if line == f"@{decorator}"
    )

    violations = scan_lightop_imports(root)

    assert any(
        item.startswith(f"{relative_path}:{decorator_line}: {detail}")
        for item in violations
    )


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


def test_installed_lightop_exports_cover_production_symbols() -> None:
    used = categorized_symbols(REPOSITORY / "vllm_hcu")
    env = dict(os.environ)
    env["LIGHTOP_REQUIRED_EXPORTS"] = json.dumps(sorted(used))
    env["LIGHTOP_OPTIONAL_FALLBACKS"] = json.dumps([
        (*public, *legacy)
        for public, legacy in sorted(OPTIONAL_CATEGORIZED_FALLBACKS.items())
    ])
    env["LIGHTOP_REQUIRED_LEGACY_ROOT_EXPORTS"] = json.dumps(
        sorted({symbol for _, symbol, _ in LEGACY_TOP_LEVEL_IMPORTS})
    )
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
            "legacy_required=json.loads("
            "os.environ['LIGHTOP_REQUIRED_LEGACY_ROOT_EXPORTS']); "
            "optional_fallbacks=json.loads("
            "os.environ['LIGHTOP_OPTIONAL_FALLBACKS']); "
            "modules={name: importlib.import_module(name) "
            "for name, _symbol in required}; "
            "lightop=importlib.import_module('lightop'); "
            "optional_keys={(name, symbol) for name, symbol, _module, "
            "_fallback in optional_fallbacks}; "
            "required_public=[(name, symbol) for name, symbol in required "
            "if (name, symbol) not in optional_keys]; "
            "bound_optional=[(name, symbol) for name, symbol, _module, "
            "_fallback in optional_fallbacks "
            "if callable(getattr(modules[name], symbol, None))]; "
            "audited_public=required_public + bound_optional; "
            "missing_public=[(name, symbol) for name, symbol in audited_public "
            "if symbol not in modules[name].__all__]; "
            "missing_bound=[(name, symbol) for name, symbol in required_public "
            "if not hasattr(modules[name], symbol)]; "
            "missing_legacy=[symbol for symbol in legacy_required "
            "if not callable(getattr(lightop, symbol, None))]; "
            "missing_fallbacks=[(name, symbol) for name, symbol, legacy_module, "
            "legacy_symbol in optional_fallbacks "
            "if not (callable(getattr(modules[name], symbol, None)) or "
            "callable(getattr(getattr(lightop, legacy_module, None), "
            "legacy_symbol, None)))]; "
            "assert not missing_public, f'not public: {missing_public}'; "
            "assert not missing_bound, f'not bound: {missing_bound}'; "
            "assert not missing_legacy, "
            "f'legacy root exports are not callable: {missing_legacy}'; "
            "assert not missing_fallbacks, "
            "f'optional public/legacy fallbacks are not callable: "
            "{missing_fallbacks}'",
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
