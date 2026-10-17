# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use dense varlen FlashAttention for Qwen4Exp prefill-only steps.

QSA side-cache updates remain owned by the upstream indexer.  Only its top-k
selection and the sparse main-attention reader are bypassed during prefill.
Decode, speculative decode, and mixed batches retain the existing QSA path.
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
_METADATA_BUILD_TARGET = (
    f"{TARGET_MODULE}.Qwen4ExpQSAMetadataBuilder.build"
)
TARGETS = (
    _INDEXER_FORWARD_TARGET,
    _INDEXER_SELECT_TARGET,
    _ATTENTION_FORWARD_TARGET,
    _METADATA_BUILD_TARGET,
)

_MARKER = "_vllm_hcu_qsa_dense_prefill_applied"
_INDEXER_FORWARD_WRAPPER = "_vllm_hcu_qsa_dense_prefill_indexer_forward"
_INDEXER_SELECT_WRAPPER = "_vllm_hcu_qsa_dense_prefill_indexer_select"
_ATTENTION_FORWARD_WRAPPER = "_vllm_hcu_qsa_dense_prefill_attention_forward"
_METADATA_BUILD_WRAPPER = "_vllm_hcu_qsa_dense_prefill_metadata_build"
_STEP_ATTR = "_vllm_hcu_qsa_dense_prefill_step"
_SELECTION_SKIPPED_ATTR = "_vllm_hcu_qsa_selection_skipped"
_CONTEXT_FREE_PREFILL_ATTR = "_vllm_hcu_qsa_context_free_prefill"
_HISTORY_DENSE_SAFE_ATTR = "_vllm_hcu_qsa_history_dense_safe"
_MAX_HISTORY_WORKSPACE_AMPLIFICATION = 4
_MAX_HISTORY_WORKSPACE_BYTES = 1 << 30
_RUNTIME_ENABLED = henvs.qsa_dense_prefill_enabled()


def configure(enabled: bool) -> None:
    """Latch the master-gated policy serialized by the API process."""

    if not isinstance(enabled, bool):
        raise TypeError("QSA dense-prefill worker policy must be bool")
    global _RUNTIME_ENABLED
    _RUNTIME_ENABLED = enabled


def _enabled() -> bool:
    return _RUNTIME_ENABLED


def _is_prefill_only(metadata) -> bool:
    """Whether every real token in this step belongs to prefill."""

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
        and max_seq_len >= max_query_len
    )


def _can_skip_qsa_selection(metadata) -> bool:
    """Prove a uniform context-free batch from side metadata alone."""

    if not _is_prefill_only(metadata):
        return False
    seq_lens = getattr(metadata, "seq_lens", None)
    if not isinstance(seq_lens, torch.Tensor):
        return False
    num_reqs = int(seq_lens.shape[0])
    return (
        num_reqs > 0
        and metadata.max_query_len == metadata.max_seq_len
        and metadata.num_actual_tokens == num_reqs * metadata.max_seq_len
    )


def _history_workspace_fits(impl, attn_metadata) -> bool:
    if not bool(getattr(attn_metadata, _HISTORY_DENSE_SAFE_ATTR, False)):
        return False
    num_reqs = int(attn_metadata.seq_lens.shape[0])
    num_slots = num_reqs * int(attn_metadata.max_seq_len)
    bytes_per_element = 1 if impl.kv_cache_dtype.startswith("fp8") else 2
    # K and V gathers coexist. FP8 history additionally materializes BF16
    # dequantized K and V before the native attention call.
    bytes_per_channel = 2 * bytes_per_element
    if impl.kv_cache_dtype.startswith("fp8"):
        bytes_per_channel += 2 * torch.bfloat16.itemsize
    workspace_bytes = (
        num_slots
        * int(impl.num_kv_heads)
        * int(impl.head_size)
        * bytes_per_channel
    )
    return workspace_bytes <= _MAX_HISTORY_WORKSPACE_BYTES


@lru_cache(maxsize=1)
def _load_dense_flash_attention_symbols():
    from vllm_hcu.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        flash_attn_varlen_fixed_stride_func,
        get_flash_attn_version,
    )

    return (
        flash_attn_varlen_func,
        flash_attn_varlen_fixed_stride_func,
        get_flash_attn_version,
    )


def _gather_paged_kv_fixed_stride(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather manager pages without dynamic output shapes.

    Each request owns ``max_seq_len`` rows in the returned tensor. Padding
    rows repeat the request's last valid token and are excluded by
    ``seqused_k`` in FlashAttention. This keeps allocations and addresses
    stable for piecewise/full graph capture while avoiding reads through
    unused block-table entries.
    """

    if kv_cache.ndim != 4:
        raise ValueError(
            "QSA main KV cache must be [blocks, heads, tokens, channels], "
            f"got {tuple(kv_cache.shape)}"
        )
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    num_reqs = seq_lens.shape[0]
    block_size = kv_cache.shape[2]
    token_positions = torch.arange(
        max_seq_len,
        dtype=torch.int64,
        device=seq_lens.device,
    ).unsqueeze(0)
    # Prefill requests have at least one live token. Clamp defensively so a
    # zero-length padded row still performs an in-bounds, subsequently masked
    # read rather than indexing an unused page.
    last_positions = seq_lens[:num_reqs].to(torch.int64).clamp_min(1).unsqueeze(1) - 1
    safe_positions = torch.minimum(token_positions, last_positions)
    block_columns = torch.div(safe_positions, block_size, rounding_mode="floor")
    block_ids = torch.gather(
        block_table[:num_reqs].to(torch.int64),
        1,
        block_columns,
    )
    offsets = torch.remainder(safe_positions, block_size)

    # QSA's bound cache is [block, head, token, 2 * head_size]. Split before
    # advanced indexing so the gather produces contiguous K and V tensors
    # directly instead of a combined temporary followed by two copies.
    token_major_cache = kv_cache.transpose(1, 2)
    key_cache, value_cache = token_major_cache.chunk(2, dim=-1)
    dense_key = key_cache[block_ids, offsets].flatten(0, 1)
    dense_value = value_cache[block_ids, offsets].flatten(0, 1)
    cu_seqlens_k = torch.arange(
        num_reqs + 1,
        dtype=torch.int32,
        device=seq_lens.device,
    ) * max_seq_len
    return dense_key, dense_value, cu_seqlens_k


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
    """Run dense varlen attention from current or paged Qwen4Exp K/V."""

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

    (
        flash_attn_varlen_func,
        flash_attn_varlen_fixed_stride_func,
        get_flash_attn_version,
    ) = _load_dense_flash_attention_symbols()
    context_free = bool(
        getattr(attn_metadata, _CONTEXT_FREE_PREFILL_ATTR, False)
    )
    if context_free:
        dense_key = key[:num_tokens]
        dense_value = value[:num_tokens]
        cu_seqlens_k = attn_metadata.query_start_loc
        seqused_k = None
        max_seqlen_k = attn_metadata.max_query_len
        block_table = None
        k_descale = None
        v_descale = None
        attention_func = flash_attn_varlen_func
        logger.info_once(
            "Qwen4Exp QSA routing: context-free prefill uses current BF16 "
            "K/V with dense varlen FlashAttention",
            scope="local",
        )
    else:
        dense_key, dense_value, cu_seqlens_k = _gather_paged_kv_fixed_stride(
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.max_seq_len,
        )
        if impl.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            dense_key = (
                dense_key.view(torch.float8_e4m3fn).to(query.dtype)
                * layer._k_scale
            ).to(query.dtype)
            dense_value = (
                dense_value.view(torch.float8_e4m3fn).to(query.dtype)
                * layer._v_scale
            ).to(query.dtype)
        elif impl.kv_cache_dtype == "fp8_e5m2":
            dense_key = (
                dense_key.view(torch.float8_e5m2).to(query.dtype)
                * layer._k_scale
            ).to(query.dtype)
            dense_value = (
                dense_value.view(torch.float8_e5m2).to(query.dtype)
                * layer._v_scale
            ).to(query.dtype)
        elif impl.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "QSA dense history prefill does not support KV cache dtype "
                f"{impl.kv_cache_dtype}"
            )
        seqused_k = attn_metadata.seq_lens
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = None
        k_descale = None
        v_descale = None
        attention_func = flash_attn_varlen_fixed_stride_func
        logger.info_once(
            "Qwen4Exp QSA routing: prefix/chunked prefill gathers manager "
            "pages into contiguous K/V for dense varlen FlashAttention; "
            "decode remains QSA",
            scope="local",
        )
    attention_func(
        q=query[:num_tokens],
        k=dense_key,
        v=dense_value,
        out=output[:num_tokens],
        cu_seqlens_q=attn_metadata.query_start_loc,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=impl.scale,
        causal=attn_metadata.causal,
        alibi_slopes=impl.alibi_slopes,
        window_size=[-1, -1],
        block_table=block_table,
        softcap=impl.logits_soft_cap,
        scheduler_metadata=None,
        # QSA constructs the official vLLM implementation, whose ROCm
        # version probe is unset. Resolve against the HCU vendor wheel here.
        fa_version=get_flash_attn_version(),
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=impl.sinks,
    )
    return output


def _already_applied(
    indexer_cls: type,
    impl_cls: type,
    metadata_builder_cls: type,
) -> bool:
    if not getattr(impl_cls, _MARKER, False):
        return False
    wrapped = (
        (indexer_cls, "forward", _INDEXER_FORWARD_WRAPPER),
        (indexer_cls, "_select", _INDEXER_SELECT_WRAPPER),
        (impl_cls, "forward_qsa", _ATTENTION_FORWARD_WRAPPER),
        (metadata_builder_cls, "build", _METADATA_BUILD_WRAPPER),
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
    metadata_builder_cls = require_class(
        owner,
        "Qwen4ExpQSAMetadataBuilder",
        _METADATA_BUILD_TARGET,
    )
    if _already_applied(indexer_cls, impl_cls, metadata_builder_cls):
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
    original_metadata_build = require_callable(
        metadata_builder_cls,
        "build",
        _METADATA_BUILD_TARGET,
    )
    require_exact_signature(
        original_metadata_build,
        _METADATA_BUILD_TARGET,
        positional=(
            "self",
            "common_prefix_len",
            "common_attn_metadata",
            "fast_build",
        ),
        defaults={"fast_build": False},
    )

    @functools.wraps(original_metadata_build)
    def hcu_metadata_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
    ):
        metadata = original_metadata_build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        if not _enabled():
            return metadata
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        context_free = False
        history_dense_safe = False
        if seq_lens_cpu is not None:
            num_reqs = int(common_attn_metadata.num_reqs)
            query_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
            )
            context_free = torch.equal(
                seq_lens_cpu[:num_reqs],
                query_lens_cpu,
            )
            if not context_free:
                live_kv_tokens = int(
                    seq_lens_cpu[:num_reqs].to(torch.int64).sum().item()
                )
                fixed_stride_tokens = num_reqs * int(metadata.max_seq_len)
                history_dense_safe = (
                    live_kv_tokens > 0
                    and fixed_stride_tokens
                    <= _MAX_HISTORY_WORKSPACE_AMPLIFICATION * live_kv_tokens
                )
        setattr(metadata, _CONTEXT_FREE_PREFILL_ATTR, context_free)
        setattr(metadata, _HISTORY_DENSE_SAFE_ATTR, history_dense_safe)
        return metadata

    @functools.wraps(original_indexer_forward)
    def hcu_indexer_forward(self, hidden_states, positions, out=None):
        setattr(self, _STEP_ATTR, False)
        setattr(self, _SELECTION_SKIPPED_ATTR, False)
        return original_indexer_forward(self, hidden_states, positions, out)

    @functools.wraps(original_indexer_select)
    def hcu_indexer_select(self, q, metadata, out):
        if _enabled() and _is_prefill_only(metadata):
            setattr(self, _STEP_ATTR, True)
            if _can_skip_qsa_selection(metadata):
                setattr(self, _SELECTION_SKIPPED_ATTR, True)
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
            _enabled() and getattr(layer.indexer, _STEP_ATTR, False)
        )
        context_free = bool(
            getattr(attn_metadata, _CONTEXT_FREE_PREFILL_ATTR, False)
            or getattr(layer.indexer, _SELECTION_SKIPPED_ATTR, False)
        )
        history_workspace_fits = _history_workspace_fits(self, attn_metadata)
        if not dense_prefill or not (context_free or history_workspace_fits):
            if dense_prefill and not context_free:
                logger.info_once(
                    "Qwen4Exp QSA routing: prefix/chunked prefill remains QSA "
                    "because fixed-stride dense K/V would exceed the history "
                    "workspace safety limit",
                    scope="local",
                )
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
    setattr(hcu_metadata_build, _METADATA_BUILD_WRAPPER, True)

    try:
        setattr(indexer_cls, "forward", hcu_indexer_forward)
        setattr(indexer_cls, "_select", hcu_indexer_select)
        setattr(impl_cls, "forward_qsa", hcu_attention_forward)
        setattr(metadata_builder_cls, "build", hcu_metadata_build)
        setattr(impl_cls, _MARKER, True)
    except BaseException:
        setattr(indexer_cls, "forward", original_indexer_forward)
        setattr(indexer_cls, "_select", original_indexer_select)
        setattr(impl_cls, "forward_qsa", original_attention_forward)
        setattr(metadata_builder_cls, "build", original_metadata_build)
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
    "configure",
    "run_qsa_dense_varlen_prefill",
]
