# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Full-KV PCP for the native AMD DeepSeek V4 execution path.

Reuse the GLM5Next PCP batch binding (``vllm_hcu.v1.glm53_pcp``): attention
runs on the restored global batch with full KV, while the input/query/output
projections and sparse prefill queries stay partitioned. mHC, hash routing
and MoE remain local.
"""

from __future__ import annotations

import functools
from contextvars import ContextVar
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError, load_exact_module, require_callable, require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4.full_kv_pcp"
_MARKER = "_vllm_hcu_deepseek_v4_pcp_applied"
_WRAPPED = (
    "forward", "_run_parallel_input_projections", "_wq_b_proj", "_o_proj",
    "_forward_prefill",
)
_ATTENTION_BATCH = ContextVar("deepseek_v4_pcp_attention_batch", default=None)
_PREFILL_ROWS = ContextVar("deepseek_v4_pcp_prefill_rows", default=None)


def current_pcp_batch():
    """PCP batch of the DeepSeek V4 attention layer currently executing."""
    return _ATTENTION_BATCH.get()


def _run_local_prefill(kernel, batch, root_q, decode_tokens, **kwargs):
    """Run only this rank's query rows while preserving global KV coordinates."""
    q = kwargs["q"]
    start = (q.storage_offset() - root_q.storage_offset()) // root_q.stride(0)
    selected = batch.owned_rows(
        start + decode_tokens, start + decode_tokens + q.shape[0],
    )
    full_output = kwargs["output"]
    full_output.zero_()
    if selected.numel() == 0:
        return
    for name in ("q", "indices", "topk_length"):
        kwargs[name] = kwargs[name].index_select(0, selected).contiguous()
    kwargs["output"] = torch.empty_like(kwargs["q"])
    kernel(**kwargs)
    full_output.index_copy_(0, selected, kwargs["output"])


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    cls = require_class(rocm, "DeepseekV4ROCMAiterMLAAttention", TARGET_MODULE)
    if getattr(cls, _MARKER, False):
        if not all(getattr(getattr(cls, name), _MARKER, False) for name in _WRAPPED) \
                or not getattr(rocm.rocm_sparse_attn_prefill, _MARKER, False):
            raise PatchCompatibilityError("stale DeepSeek V4 PCP patch")
        return False
    originals = {
        name: require_callable(cls, name, f"{TARGET_MODULE}.{name}")
        for name in _WRAPPED
    }
    kernel = require_callable(rocm, "rocm_sparse_attn_prefill", TARGET_MODULE)
    signatures = {
        "forward": (("self", "positions", "hidden_states", "llama_4_scaling"),
                    {"llama_4_scaling": None}),
        "_run_parallel_input_projections": (("self", "hidden_states"), {}),
        "_wq_b_proj": (("self", "qr", "qr_scale"), {"qr_scale": None}),
        "_o_proj": (("self", "o", "positions"), {}),
        "_forward_prefill": (("self", "q", "positions", "compressed_k_cache",
                              "swa_k_cache", "output", "attn_metadata", "swa_metadata"), {}),
    }
    for name, (positional, defaults) in signatures.items():
        require_exact_signature(originals[name], f"{TARGET_MODULE}.{name}",
                                positional=positional, defaults=defaults)
    require_exact_signature(
        kernel, f"{TARGET_MODULE}.rocm_sparse_attn_prefill",
        positional=("q", "kv", "indices", "topk_length", "scale", "head_dim",
                    "nope_head_dim", "rope_head_dim", "attn_sink", "output",
                    "ragged_indices", "ragged_indptr"),
        defaults={"ragged_indices": None, "ragged_indptr": None},
    )
    from vllm.models.deepseek_v4.attention import DeepseekV4Indexer
    from vllm.models.deepseek_v4.compressor import CompressorBackend
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

    indexer_projection = require_callable(
        DeepseekV4Indexer, "_wq_b_proj", "DeepseekV4Indexer._wq_b_proj"
    )
    require_exact_signature(indexer_projection, "DeepseekV4Indexer._wq_b_proj",
                            positional=("self", "qr", "qr_scale"),
                            defaults={"qr_scale": None})

    @functools.wraps(originals["forward"])
    def forward(self, positions, hidden_states, llama_4_scaling=None):
        from vllm_hcu.model_executor.layers.attention.pcp import (
            replicated_mtp_batch_scope,
        )
        from vllm_hcu.v1.glm53_pcp import _current_pcp_batch

        batch = _current_pcp_batch()
        if batch is None:
            return originals["forward"](self, positions, hidden_states, llama_4_scaling)
        token = _ATTENTION_BATCH.set(batch)
        try:
            full_hidden = batch.gather(hidden_states)
            # Native indexer/compressor calls already see complete KV. Reuse
            # the existing logical-PCP scope to suppress a second cache gather.
            with replicated_mtp_batch_scope():
                return originals["forward"](
                    self, batch.global_batch.positions[:full_hidden.shape[0]],
                    full_hidden, llama_4_scaling,
                )
        finally:
            _ATTENTION_BATCH.reset(token)

    @functools.wraps(originals["_run_parallel_input_projections"])
    def projections(self, hidden_states):
        batch = _ATTENTION_BATCH.get()
        if batch is None:
            return originals["_run_parallel_input_projections"](self, hidden_states)
        results = originals["_run_parallel_input_projections"](
            self, batch.localize(hidden_states),
        )
        return tuple(None if value is None else batch.gather(value) for value in results)

    def partition_query_projection(original):
        @functools.wraps(original)
        def query_projection(self, qr, qr_scale=None):
            batch = _ATTENTION_BATCH.get()
            if batch is None:
                return original(self, qr, qr_scale)
            result = original(
                self, batch.localize(qr),
                None if qr_scale is None else batch.localize(qr_scale),
            )
            return batch.gather(result)
        return query_projection

    @functools.wraps(originals["_o_proj"])
    def output_projection(self, output, positions):
        batch = _ATTENTION_BATCH.get()
        if batch is not None:
            output = batch.localize(output)
            positions = batch.localize(positions)
        return originals["_o_proj"](self, output, positions)

    @functools.wraps(originals["_forward_prefill"])
    def prefill(self, q, positions, compressed_k_cache, swa_k_cache,
                output, attn_metadata, swa_metadata):
        batch = _ATTENTION_BATCH.get()
        token = None
        if batch is not None:
            token = _PREFILL_ROWS.set((batch, q, swa_metadata.num_decode_tokens))
        try:
            return originals["_forward_prefill"](
                self, q, positions, compressed_k_cache, swa_k_cache,
                output, attn_metadata, swa_metadata,
            )
        finally:
            if token is not None:
                _PREFILL_ROWS.reset(token)

    @functools.wraps(kernel)
    def sparse_prefill(q, kv, indices, topk_length, scale, head_dim,
                       nope_head_dim, rope_head_dim, attn_sink, output,
                       ragged_indices=None, ragged_indptr=None):
        selection = _PREFILL_ROWS.get()
        if selection is None:
            return kernel(q, kv, indices, topk_length, scale, head_dim,
                          nope_head_dim, rope_head_dim, attn_sink, output,
                          ragged_indices, ragged_indptr)
        batch, root_q, decode_tokens = selection
        # The native V4 prefill call provides dense combined indices. Keep
        # the installed HCU/FlashMLA provider authoritative for selected rows.
        if ragged_indices is not None or ragged_indptr is not None:
            raise ValueError("DeepSeek V4 PCP prefill requires dense combined indices")
        return _run_local_prefill(
            kernel, batch, root_q, decode_tokens, q=q, kv=kv, indices=indices,
            topk_length=topk_length, scale=scale, head_dim=head_dim,
            nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
            attn_sink=attn_sink, output=output,
        )

    # These are V4-only cache backends; all of them now consume full KV on
    # each rank. Do not advertise PCP on generic MLA/SWA backends.
    for backend in (
        rocm.DeepseekV4ROCMAiterMLASparseBackend,
        CompressorBackend, DeepseekSparseSWABackend,
    ):
        backend.supports_pcp = classmethod(lambda cls: True)
    for name, wrapper in (
        ("forward", forward), ("_run_parallel_input_projections", projections),
        ("_wq_b_proj", partition_query_projection(originals["_wq_b_proj"])),
        ("_o_proj", output_projection),
        ("_forward_prefill", prefill),
    ):
        setattr(wrapper, _MARKER, True)
        setattr(cls, name, wrapper)
    indexer_wrapper = partition_query_projection(indexer_projection)
    setattr(indexer_wrapper, _MARKER, True)
    DeepseekV4Indexer._wq_b_proj = indexer_wrapper
    setattr(sparse_prefill, _MARKER, True)
    rocm.rocm_sparse_attn_prefill = sparse_prefill
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module", "current_pcp_batch"]
