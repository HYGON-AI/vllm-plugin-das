# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Launch Qwen4Exp PLE INT8 UVA lookup before the owning decoder layer."""

from __future__ import annotations

import functools
import inspect
import os
from types import ModuleType
from weakref import WeakSet

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.models.qwen4_exp.amd.model"
PATCH_ID = "worker.core_fix.qwen4_exp.ple_prefetch_stream"
TARGETS = (
    f"{TARGET_MODULE}.Qwen4ExpModel.__init__",
    f"{TARGET_MODULE}.Qwen4ExpModel.forward",
    f"{TARGET_MODULE}.Qwen4ExpForCausalLM.process_weights_after_loading",
    f"{TARGET_MODULE}.Qwen4ExpForConditionalGeneration.process_weights_after_loading",
)
_MODULE_MARKER = "_vllm_hcu_qwen4_exp_ple_prefetch_applied"
_WRAPPER_MARKER = "_vllm_hcu_qwen4_exp_ple_prefetch_wrapper"
_CUSTOM_OP_REGISTERED = False
_PREFETCH_OWNERS: WeakSet[object] = WeakSet()


def _requested() -> bool:
    return os.environ.get("VLLM_HCU_PLE_PREFETCH_STREAM", "False").lower() in (
        "true",
        "1",
    )


def _prefetch_owner(layer_name: str):
    from vllm.forward_context import get_forward_context

    owner = get_forward_context().no_compile_layers[layer_name]
    ngram = getattr(owner, "ple_embedding", None)
    if ngram is None or not callable(getattr(ngram, "_start_prefetch_impl", None)):
        raise TypeError(f"{layer_name} is not an HCU prefetch-capable PLE owner")
    return ngram


def _hcu_qwen4_exp_ple_prefetch(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    ids_buffer: torch.Tensor,
    rows_buffer: torch.Tensor,
    layer_name: str,
) -> None:
    ngram = _prefetch_owner(layer_name)
    if ids_buffer is not ngram._hcu_prefetch_ids_buffer:
        raise RuntimeError(f"PLE ids workspace mismatch for {layer_name}")
    if rows_buffer is not ngram._hcu_prefetch_rows_buffer:
        raise RuntimeError(f"PLE rows workspace mismatch for {layer_name}")
    ngram._start_prefetch_impl(input_ids, query_start_loc, ngram_context)


def _ensure_custom_op_registered() -> None:
    global _CUSTOM_OP_REGISTERED
    if _CUSTOM_OP_REGISTERED:
        return
    direct_register_custom_op(
        op_name="hcu_qwen4_exp_ple_prefetch",
        op_func=_hcu_qwen4_exp_ple_prefetch,
        mutates_args=["ids_buffer", "rows_buffer"],
    )
    _CUSTOM_OP_REGISTERED = True


def _run_prefetch_custom_op(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    ngram,
) -> None:
    torch.ops.vllm.hcu_qwen4_exp_ple_prefetch(
        input_ids,
        query_start_loc,
        ngram_context,
        ngram._hcu_prefetch_ids_buffer,
        ngram._hcu_prefetch_rows_buffer,
        ngram.layer_name,
    )


def join_pending_prefetches() -> None:
    """Join outstanding PLE side-stream work before graph capture starts."""
    main = torch.cuda.current_stream()
    for ngram in tuple(_PREFETCH_OWNERS):
        if not getattr(ngram, "_hcu_prefetch_pending", False):
            continue
        event = getattr(ngram, "_hcu_prefetch_event", None)
        if event is None:
            raise RuntimeError(
                f"pending PLE prefetch for {ngram.layer_name} has no event"
            )
        main.wait_event(event)


def _bind_model_ple_chain(model) -> tuple[object, ...]:
    local = []
    layers = model.layers
    for index in range(model.start_layer, model.end_layer):
        layer = layers[index]
        ple = getattr(layer, "ple", None)
        ngram = getattr(ple, "ple_embedding", None)
        if ngram is not None:
            required = (
                "_hcu_prefetch_enabled",
                "_hcu_prefetch_ids_buffer",
                "_hcu_prefetch_rows_buffer",
                "_start_prefetch_impl",
                "prepare_prefetch",
            )
            missing = [name for name in required if not hasattr(ngram, name)]
            if missing:
                raise PatchCompatibilityError(
                    "Qwen4Exp PLE module exchange is missing prefetch contract: "
                    + ", ".join(missing)
                )
            local.append(ngram)
    for current, successor in zip(local, local[1:]):
        current._hcu_prefetch_successor = successor
    if local:
        local[-1]._hcu_prefetch_successor = None
    for ngram in local:
        if ngram._hcu_prefetch_enabled:
            _PREFETCH_OWNERS.add(ngram)
    model._vllm_hcu_local_ple_chain = tuple(local)
    model._vllm_hcu_first_local_ple = local[0] if local else None
    return tuple(local)


def _require_signature(function, target: str, names: tuple[str, ...]) -> None:
    if not callable(function) or tuple(inspect.signature(function).parameters) != names:
        raise PatchCompatibilityError(
            f"required target {target} has an incompatible signature"
        )


def apply_to_module(module: ModuleType) -> bool:
    model_module = load_exact_module(TARGET_MODULE, module)
    if not _requested():
        return False
    _ensure_custom_op_registered()

    model_class = require_class(
        model_module, "Qwen4ExpModel", f"{TARGET_MODULE}.Qwen4ExpModel"
    )
    causal_class = require_class(
        model_module,
        "Qwen4ExpForCausalLM",
        f"{TARGET_MODULE}.Qwen4ExpForCausalLM",
    )
    conditional_class = require_class(
        model_module,
        "Qwen4ExpForConditionalGeneration",
        f"{TARGET_MODULE}.Qwen4ExpForConditionalGeneration",
    )
    init = vars(model_class).get("__init__")
    forward = vars(model_class).get("forward")
    process_weights = vars(causal_class).get("process_weights_after_loading")
    conditional_process_weights = vars(conditional_class).get(
        "process_weights_after_loading"
    )
    _require_signature(
        init,
        TARGETS[0],
        ("self", "args", "vllm_config", "prefix", "kwargs"),
    )
    _require_signature(
        forward,
        TARGETS[1],
        (
            "self",
            "input_ids",
            "positions",
            "intermediate_tensors",
            "inputs_embeds",
            "query_start_loc",
            "ngram_context",
            "deepstack_input_embeds",
        ),
    )
    if getattr(model_module, _MODULE_MARKER, False):
        if not all(
            getattr(function, _WRAPPER_MARKER, False)
            for function in (
                init,
                forward,
                process_weights,
                conditional_process_weights,
            )
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False
    if process_weights is not None:
        raise PatchCompatibilityError(
            f"required target {TARGETS[2]} unexpectedly already exists"
        )
    if conditional_process_weights is not None:
        raise PatchCompatibilityError(
            f"required target {TARGETS[3]} unexpectedly already exists"
        )
    if any(
        getattr(function, _WRAPPER_MARKER, False)
        for function in (init, forward)
    ):
        raise PatchCompatibilityError("refusing a partial Qwen4Exp PLE prefetch patch")

    @functools.wraps(init)
    def hcu_init(self, *args, vllm_config=None, prefix="", **kwargs):
        init(
            self,
            *args,
            vllm_config=vllm_config,
            prefix=prefix,
            **kwargs,
        )
        _bind_model_ple_chain(self)

    @functools.wraps(forward)
    def hcu_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        query_start_loc=None,
        ngram_context=None,
        deepstack_input_embeds=None,
    ):
        first = self._vllm_hcu_first_local_ple
        if (
            first is not None
            and first._hcu_prefetch_enabled
            and input_ids is not None
            and query_start_loc is not None
            and ngram_context is not None
        ):
            _run_prefetch_custom_op(
                input_ids, query_start_loc, ngram_context, first
            )
        return forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            query_start_loc,
            ngram_context,
            deepstack_input_embeds,
        )

    def hcu_process_weights_after_loading(self):
        model = self.model
        for ngram in model._vllm_hcu_local_ple_chain:
            ngram.prepare_prefetch()

    def hcu_conditional_process_weights_after_loading(self):
        model = self.language_model.model
        for ngram in model._vllm_hcu_local_ple_chain:
            ngram.prepare_prefetch()

    for function in (
        hcu_init,
        hcu_forward,
        hcu_process_weights_after_loading,
        hcu_conditional_process_weights_after_loading,
    ):
        setattr(function, _WRAPPER_MARKER, True)

    try:
        model_class.__init__ = hcu_init
        model_class.forward = hcu_forward
        causal_class.process_weights_after_loading = hcu_process_weights_after_loading
        conditional_class.process_weights_after_loading = (
            hcu_conditional_process_weights_after_loading
        )
        setattr(model_module, _MODULE_MARKER, True)
    except BaseException:
        model_class.__init__ = init
        model_class.forward = forward
        if hasattr(causal_class, "process_weights_after_loading"):
            delattr(causal_class, "process_weights_after_loading")
        if hasattr(conditional_class, "process_weights_after_loading"):
            delattr(conditional_class, "process_weights_after_loading")
        if hasattr(model_module, _MODULE_MARKER):
            delattr(model_module, _MODULE_MARKER)
        raise
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
