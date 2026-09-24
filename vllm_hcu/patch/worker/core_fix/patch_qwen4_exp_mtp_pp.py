# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run Qwen4Exp MTP from target hidden states on the last PP stage."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import textwrap
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.mtp"
PATCH_ID = "worker.core_fix.qwen4_exp.mtp_last_stage_input"
TARGETS = (f"{TARGET_MODULE}.Qwen4ExpMultiTokenPredictor.forward",)
_MARKER = "_vllm_hcu_qwen4_exp_mtp_pp_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_mtp_pp_wrapper"
_ORIGINAL = "_vllm_hcu_original_forward"
_V028_FORWARD_SOURCE_SHA256 = (
    "0cfccb8c3f57aa1f4886803723e263d95551159ee8467e2a4c96a97a338cb673"
)


def _is_first_pp_rank_test(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "is_first_rank"
        and isinstance(node.value, ast.Call)
        and not node.value.args
        and not node.value.keywords
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "get_pp_group"
    )


class _UseAvailableHiddenStates(ast.NodeTransformer):
    replacements = 0

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        if _is_first_pp_rank_test(node.test):
            node.test = ast.copy_location(
                ast.Compare(
                    left=ast.Name(id="hidden_states", ctx=ast.Load()),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                node.test,
            )
            self.replacements += 1
        return node


def _rebuild_forward(forward):
    try:
        source = textwrap.dedent(inspect.getsource(forward))
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} source fingerprint could "
            "not be computed"
        ) from exc
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != _V028_FORWARD_SOURCE_SHA256:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} source fingerprint "
            f"mismatch: expected sha256={_V028_FORWARD_SOURCE_SHA256}, "
            f"actual sha256={actual}"
        )

    tree = ast.parse(source)
    transformer = _UseAvailableHiddenStates()
    tree = transformer.visit(tree)
    if transformer.replacements != 1:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has "
            f"{transformer.replacements} first-PP-rank guards; expected 1"
        )
    ast.fix_missing_locations(tree)

    namespace: dict[str, object] = {}
    filename = inspect.getsourcefile(forward) or forward.__code__.co_filename
    exec(compile(tree, filename, "exec"), forward.__globals__, namespace)
    rebuilt = namespace.get("forward")
    if not callable(rebuilt):
        raise PatchCompatibilityError(
            f"could not rebuild required HCU patch target {TARGETS[0]}"
        )
    functools.update_wrapper(rebuilt, forward)
    setattr(rebuilt, _WRAPPER, True)
    return rebuilt


def apply_to_module(module: ModuleType) -> bool:
    mtp_module = load_exact_module(TARGET_MODULE, module)
    predictor = require_class(
        mtp_module,
        "Qwen4ExpMultiTokenPredictor",
        f"{TARGET_MODULE}.Qwen4ExpMultiTokenPredictor",
    )
    forward = require_callable(predictor, "forward", TARGETS[0])
    if getattr(mtp_module, _MARKER, False):
        if not getattr(forward, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    if getattr(forward, _WRAPPER, False):
        raise PatchCompatibilityError(
            f"refusing a partial Qwen4Exp MTP PP patch for {TARGETS[0]}"
        )

    require_exact_signature(
        forward,
        TARGETS[0],
        positional=(
            "self",
            "input_ids",
            "positions",
            "hidden_states",
            "intermediate_tensors",
            "inputs_embeds",
            "spec_step_idx",
        ),
        defaults={
            "hidden_states": None,
            "intermediate_tensors": None,
            "inputs_embeds": None,
            "spec_step_idx": 0,
        },
    )
    rebuilt = _rebuild_forward(forward)
    setattr(predictor, _ORIGINAL, forward)
    setattr(predictor, "forward", rebuilt)
    setattr(mtp_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
