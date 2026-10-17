# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use dense varlen FlashAttention for context-free Qwen4Exp prefills.

QSA side-cache updates remain owned by the upstream indexer.  Only its top-k
selection and the sparse main-attention reader are bypassed during a pure
prefill step. Decode, speculative decode, and mixed batches retain the
existing QSA path.
"""

from __future__ import annotations

import functools
from functools import lru_cache
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

logger = init_logger(f"vllm.{__name__}")

TARGET_MODULE = "vllm.models.qwen4_exp.amd.qsa"
PATCH_ID = "worker.core_fix.qwen4_exp.qsa.dense_prefill"
_INDEXER_FORWARD_TARGET = (
    "vllm.models.qwen4_exp.amd.indexer_qsa.QSAIndexer.forward"
)
_INDEXER_SELECT_TARGET = (
    "vllm.models.qwen4_exp.amd.indexer_qsa.QSAIndexer._select"
)
_ATTENTION_FORWARD_TARGET = (
    f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionImpl.forward_qsa"
)
TARGETS = (
    _INDEXER_FORWARD_TARGET,
    _INDEXER_SELECT_TARGET,
    _ATTENTION_FORWARD_TARGET,
)

_MARKER = "_vllm_hcu_qsa_dense_prefill_applied"
_INDEXER_FORWARD_WRAPPER = "_vllm_hcu_qsa_dense_prefill_indexer_forward"
_INDEXER_SELECT_WRAPPER = "_vllm_hcu_qsa_dense_prefill_indexer_select"
_ATTENTION_FORWARD_WRAPPER = "_vllm_hcu_qsa_dense_prefill_attention_forward"
_STEP_ATTR = "_vllm_hcu_qsa_dense_prefill_step"


def _is_pure_prefill(metadata) -> bool:
    """Use dense Q/K/V only when the whole batch starts without context."""

    num_actual_tokens = int(getattr(metadata, "num_actual_tokens", 0))
    num_decode_tokens = int(getattr(metadata, "num_decode_tokens", -1))
    num_prefill_tokens = int(getattr(metadata, "num_prefill_tokens", -1))
    max_query_len = int(getattr(metadata, "max_query_len", -1))
    max_seq_len = int(getattr(metadata, "max_seq_len", -1))
    return (
        num_actual_tokens > 0
        and num_decode_tokens == 0
        and num_prefill_tokens == num_actual_tokens
        and max_query_len > 0
        and max_seq_len == max_query_len
    )


@lru_cache(maxsize=1)
def _load_dense_flash_attention_symbols():
    from vllm_hcu.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    return flash_attn_varlen_func, get_flash_attn_version


def run_qsa_dense_varlen_prefill(
    impl,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense varlen attention from the current context-free BF16 Q/K/V."""

    del layer, kv_cache
    if output_scale is not None or output_block_scale is not None:
        raise NotImplementedError(
            "QSA dense prefill does not support fused output quantization"
        )
    if impl.alibi_slopes is not None or impl.sinks is not None:
        raise NotImplementedError(
            "QSA dense prefill does not support ALiBi or attention sinks"
        )
    if impl.sliding_window != (-1, -1):
        raise NotImplementedError(
            "QSA dense prefill does not support sliding-window attention"
        )

    num_tokens = attn_metadata.num_actual_tokens
    output.zero_()
    if num_tokens == 0:
        return output

    flash_attn_varlen_func, get_flash_attn_version = (
        _load_dense_flash_attention_symbols()
    )

    logger.info_once(
        "Qwen4Exp QSA routing: context-free prefill uses dense varlen "
        "FlashAttention; prefix/chunked prefill and decode remain QSA",
        scope="local",
    )
    flash_attn_varlen_func(
        q=query[:num_tokens],
        k=key[:num_tokens],
        v=value[:num_tokens],
        out=output[:num_tokens],
        cu_seqlens_q=attn_metadata.query_start_loc,
        cu_seqlens_k=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=None,
        max_seqlen_k=attn_metadata.max_query_len,
        softmax_scale=impl.scale,
        causal=attn_metadata.causal,
        alibi_slopes=impl.alibi_slopes,
        window_size=[-1, -1],
        block_table=None,
        softcap=impl.logits_soft_cap,
        scheduler_metadata=None,
        # QSA constructs the official vLLM implementation, whose ROCm
        # version probe is unset. Resolve against the HCU vendor wheel here.
        fa_version=get_flash_attn_version(),
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=impl.sinks,
    )
    return output


def _already_applied(indexer_cls: type, impl_cls: type) -> bool:
    if not getattr(impl_cls, _MARKER, False):
        return False
    wrapped = (
        (indexer_cls, "forward", _INDEXER_FORWARD_WRAPPER),
        (indexer_cls, "_select", _INDEXER_SELECT_WRAPPER),
        (impl_cls, "forward_qsa", _ATTENTION_FORWARD_WRAPPER),
    )
    for owner, name, wrapper_marker in wrapped:
        function = require_callable(owner, name, f"{owner.__name__}.{name}")
        if not getattr(function, wrapper_marker, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {owner.__name__}.{name} is stale; "
                "restart the process"
            )
    return True


def apply_to_module(module: ModuleType) -> bool:
    owner = load_exact_module(TARGET_MODULE, module)
    indexer_cls = require_class(owner, "QSAIndexer", _INDEXER_FORWARD_TARGET)
    impl_cls = require_class(
        owner,
        "Qwen4ExpQSAFlashAttentionImpl",
        _ATTENTION_FORWARD_TARGET,
    )
    if _already_applied(indexer_cls, impl_cls):
        return False

    original_indexer_forward = require_callable(
        indexer_cls, "forward", _INDEXER_FORWARD_TARGET
    )
    require_exact_signature(
        original_indexer_forward,
        _INDEXER_FORWARD_TARGET,
        positional=("self", "hidden_states", "positions", "out"),
        defaults={"out": None},
    )
    original_indexer_select = require_callable(
        indexer_cls, "_select", _INDEXER_SELECT_TARGET
    )
    require_exact_signature(
        original_indexer_select,
        _INDEXER_SELECT_TARGET,
        positional=("self", "q", "metadata", "out"),
    )
    original_attention_forward = require_callable(
        impl_cls, "forward_qsa", _ATTENTION_FORWARD_TARGET
    )
    require_exact_signature(
        original_attention_forward,
        _ATTENTION_FORWARD_TARGET,
        positional=(
            "self",
            "layer",
            "query",
            "key",
            "value",
            "kv_cache",
            "attn_metadata",
            "output",
            "token_to_req",
            "output_scale",
            "output_block_scale",
        ),
        defaults={"output_scale": None, "output_block_scale": None},
    )

    @functools.wraps(original_indexer_forward)
    def hcu_indexer_forward(self, hidden_states, positions, out=None):
        setattr(self, _STEP_ATTR, False)
        return original_indexer_forward(self, hidden_states, positions, out)

    @functools.wraps(original_indexer_select)
    def hcu_indexer_select(self, q, metadata, out):
        if henvs.qsa_dense_prefill_enabled() and _is_pure_prefill(metadata):
            setattr(self, _STEP_ATTR, True)
            if out is not None:
                return out
            return torch.empty(
                q.shape[0],
                self.output_width,
                dtype=torch.int32,
                device=q.device,
            )
        return original_indexer_select(self, q, metadata, out)

    @functools.wraps(original_attention_forward)
    def hcu_attention_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        token_to_req,
        output_scale=None,
        output_block_scale=None,
    ):
        dense_prefill = bool(
            henvs.qsa_dense_prefill_enabled()
            and getattr(layer.indexer, _STEP_ATTR, False)
        )
        if not dense_prefill:
            return original_attention_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                token_to_req,
                output_scale,
                output_block_scale,
            )
        return run_qsa_dense_varlen_prefill(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    setattr(hcu_indexer_forward, _INDEXER_FORWARD_WRAPPER, True)
    setattr(hcu_indexer_select, _INDEXER_SELECT_WRAPPER, True)
    setattr(hcu_attention_forward, _ATTENTION_FORWARD_WRAPPER, True)

    try:
        setattr(indexer_cls, "forward", hcu_indexer_forward)
        setattr(indexer_cls, "_select", hcu_indexer_select)
        setattr(impl_cls, "forward_qsa", hcu_attention_forward)
        setattr(impl_cls, _MARKER, True)
    except BaseException:
        setattr(indexer_cls, "forward", original_indexer_forward)
        setattr(indexer_cls, "_select", original_indexer_select)
        setattr(impl_cls, "forward_qsa", original_attention_forward)
        if hasattr(impl_cls, _MARKER):
            delattr(impl_cls, _MARKER)
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
    "run_qsa_dense_varlen_prefill",
]
