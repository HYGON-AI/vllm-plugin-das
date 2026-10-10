# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Allow Qwen4Exp HC layers to run with sequence-parallel MoE (DP+TP+EP).

The AMD Qwen4Exp constructors reject ``use_sequence_parallel_moe`` up front.
The MoE block inherited from Qwen3Next already handles that mode itself: it
chunks the replicated tokens across TP ranks, runs expert-parallel dispatch on
the local chunk and all-gathers the result.  The HC state and attention stay
TP-replicated, so removing the two guards is sufficient for correctness.  The
guards never fire for other parallel layouts, so the rebuilt constructors are
behaviorally identical there.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import textwrap
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.models.qwen4_exp.amd.model"
PATCH_ID = "worker.core_fix.qwen4_exp.sequence_parallel_moe"
TARGETS = (
    f"{TARGET_MODULE}.Qwen4ExpSparseMoeBlock.__init__",
    f"{TARGET_MODULE}.Qwen4ExpDecoderLayer.__init__",
)
_MARKER = "_vllm_hcu_qwen4_exp_sp_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_sp_wrapper"
_SOURCE_SHA256 = {
    "Qwen4ExpSparseMoeBlock": (
        "9e84d4d730d14443c55886ea7d75fd76d16ce61a59f92774c0ca4b20048af404"
    ),
    "Qwen4ExpDecoderLayer": (
        "074a90fb40855a585a08d7b125b12df63ff3ddc6759248e45d628bc9b5572c19"
    ),
}


def _is_sp_moe_guard(node: ast.stmt) -> bool:
    """Match ``if <...>.use_sequence_parallel_moe: raise NotImplementedError``."""
    if not isinstance(node, ast.If) or node.orelse or len(node.body) != 1:
        return False
    test, body = node.test, node.body[0]
    if not (
        isinstance(test, ast.Attribute) and test.attr == "use_sequence_parallel_moe"
    ):
        return False
    if not isinstance(body, ast.Raise) or not isinstance(body.exc, ast.Call):
        return False
    func = body.exc.func
    return isinstance(func, ast.Name) and func.id == "NotImplementedError"


def _rebuild_init(owner: type, target: str):
    init = vars(owner).get("__init__")
    if not callable(init):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    try:
        source = textwrap.dedent(inspect.getsource(init))
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint could not be "
            "computed"
        ) from exc
    expected = _SOURCE_SHA256[owner.__name__]
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != expected:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint mismatch: "
            f"expected sha256={expected}, actual sha256={actual}"
        )

    function_def = ast.parse(source).body[0]
    assert isinstance(function_def, ast.FunctionDef)
    kept = [stmt for stmt in function_def.body if not _is_sp_moe_guard(stmt)]
    if len(function_def.body) - len(kept) != 1:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has "
            f"{len(function_def.body) - len(kept)} sequence-parallel MoE guards; "
            "expected 1"
        )
    function_def.body = kept

    # Zero-argument super() needs a ``__class__`` cell; a factory parameter of
    # that name provides it outside the class body.
    factory = ast.FunctionDef(
        name="_factory",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="__class__")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[function_def, ast.Return(value=ast.Name(id="__init__", ctx=ast.Load()))],
        decorator_list=[],
    )
    module_ast = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace: dict[str, object] = {}
    filename = inspect.getsourcefile(init) or init.__code__.co_filename
    exec(compile(module_ast, filename, "exec"), init.__globals__, namespace)
    rebuilt = namespace["_factory"](owner)
    functools.update_wrapper(rebuilt, init)
    setattr(rebuilt, _WRAPPER, True)
    return init, rebuilt


def apply_to_module(module: ModuleType) -> bool:
    model_module = load_exact_module(TARGET_MODULE, module)
    owners = tuple(
        require_class(model_module, name, f"{TARGET_MODULE}.{name}")
        for name in ("Qwen4ExpSparseMoeBlock", "Qwen4ExpDecoderLayer")
    )
    if getattr(model_module, _MARKER, False):
        if not all(getattr(vars(owner)["__init__"], _WRAPPER, False) for owner in owners):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False
    if any(getattr(vars(owner).get("__init__"), _WRAPPER, False) for owner in owners):
        raise PatchCompatibilityError(
            "refusing a partial Qwen4Exp sequence-parallel MoE patch"
        )

    # Rebuild both before installing either so a mismatch leaves no partial state.
    rebuilt = [_rebuild_init(owner, target) for owner, target in zip(owners, TARGETS)]
    try:
        for owner, (_, new_init) in zip(owners, rebuilt):
            owner.__init__ = new_init
        setattr(model_module, _MARKER, True)
    except BaseException:
        for owner, (original, _) in zip(owners, rebuilt):
            owner.__init__ = original
        if hasattr(model_module, _MARKER):
            delattr(model_module, _MARKER)
        raise
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
