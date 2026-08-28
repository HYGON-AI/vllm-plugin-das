# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Backport memory-efficient generation head dtype to vLLM v0.25.1."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.logits_processor"
PATCH_ID = "platform.core_fix.logits_processor_head_dtype"
TARGETS = (
    f"{TARGET_MODULE}.LogitsProcessor.__init__",
    f"{TARGET_MODULE}.LogitsProcessor._get_logits",
    f"{TARGET_MODULE}.LogitsProcessor.get_top_tokens",
)
_MARKER = "_vllm_hcu_logits_processor_head_dtype_applied"


def apply_to_module(module: ModuleType) -> bool:
    logits_module = load_exact_module(TARGET_MODULE, module)
    if getattr(logits_module, _MARKER, False):
        return False

    processor = getattr(logits_module, "LogitsProcessor", None)
    if not isinstance(processor, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.LogitsProcessor is missing"
        )
    original_init = require_callable(processor, "__init__", TARGETS[0])
    original_get_logits = require_callable(processor, "_get_logits", TARGETS[1])
    original_get_top_tokens = require_callable(
        processor,
        "get_top_tokens",
        TARGETS[2],
    )
    require_positional_signature(
        original_init,
        TARGETS[0],
        (
            "self",
            "vocab_size",
            "org_vocab_size",
            "scale",
            "logits_as_input",
            "soft_cap",
        ),
    )
    require_positional_signature(
        original_get_logits,
        TARGETS[1],
        ("self", "hidden_states", "lm_head", "embedding_bias"),
    )
    require_positional_signature(
        original_get_top_tokens,
        TARGETS[2],
        ("self", "lm_head", "hidden_states", "embedding_bias"),
    )

    import torch
    import torch.nn.functional as F

    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    current_platform = getattr(logits_module, "current_platform", None)
    if current_platform is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.current_platform is missing"
        )

    @functools.wraps(original_init)
    def hcu_init(
        self,
        vocab_size,
        org_vocab_size=None,
        scale=1.0,
        logits_as_input=False,
        soft_cap=None,
    ):
        original_init(
            self,
            vocab_size,
            org_vocab_size,
            scale,
            logits_as_input,
            soft_cap,
        )
        model_config = get_current_vllm_config().model_config
        self.head_dtype = (
            model_config.head_dtype if model_config is not None else None
        )

    def hcu_apply_head(self, lm_head, hidden_states, embedding_bias):
        head_dtype = self.head_dtype
        if head_dtype is None or head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply(
                lm_head,
                hidden_states,
                bias=embedding_bias,
            )
        if not isinstance(
            lm_head.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            raise ValueError(
                "A head_dtype different from the model dtype is only "
                "supported for an unquantized lm_head."
            )
        if (
            head_dtype == torch.float32
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
        ):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.mm(flat, lm_head.weight.t(), out_dtype=head_dtype)
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(head_dtype)
            return logits.reshape(*hidden_states.shape[:-1], -1)
        return F.linear(
            hidden_states.to(head_dtype),
            lm_head.weight.to(head_dtype),
            embedding_bias.to(head_dtype) if embedding_bias is not None else None,
        )

    @functools.wraps(original_get_logits)
    def hcu_get_logits(self, hidden_states, lm_head, embedding_bias):
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    @functools.wraps(original_get_top_tokens)
    def hcu_get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = logits_module.get_tensor_model_parallel_world_size()
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        local_max_vals, local_max_indices = logits.max(dim=-1)
        global_indices = (
            local_max_indices + lm_head.shard_indices.org_vocab_start_index
        )
        if tp_size == 1:
            return global_indices
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()],
            dim=-1,
        )
        gathered = logits_module.tensor_model_parallel_all_gather(
            local_pair,
            dim=-1,
        )
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    setattr(processor, "_vllm_hcu_original_init", original_init)
    setattr(processor, "_vllm_hcu_original_get_logits", original_get_logits)
    setattr(processor, "_vllm_hcu_original_get_top_tokens", original_get_top_tokens)
    setattr(processor, "__init__", hcu_init)
    setattr(processor, "_apply_head", hcu_apply_head)
    setattr(processor, "_get_logits", hcu_get_logits)
    setattr(processor, "get_top_tokens", hcu_get_top_tokens)
    setattr(logits_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    logits_module = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=logits_module,
        marker=_MARKER,
        callback=lambda: apply_to_module(logits_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
