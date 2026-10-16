# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Full-KV PCP for the native AMD DeepSeek V4 execution path.

Reuse the GLM5Next PCP batch binding (``vllm_hcu.v1.glm53_pcp``): attention
consumes full KV in the original request order, while the input/query/output
projections and sparse prefill queries stay partitioned. mHC, hash routing
and MoE remain local.

For pure prefill steps with the LightOp PCP cache-insert kernel, queries stay
rank-local end to end and only KV is gathered. Otherwise the compatibility
path restores hidden states and runs native attention on the global batch.
"""

from __future__ import annotations

import functools
import importlib
from contextvars import ContextVar
from types import ModuleType

import torch

from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError, load_exact_module, require_callable, require_class,
    require_exact_signature,
)

logger = init_logger(__name__)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4.full_kv_pcp"
_MARKER = "_vllm_hcu_deepseek_v4_pcp_applied"
_WRAPPED = (
    "forward", "_run_parallel_input_projections", "_wq_b_proj", "_o_proj",
    "_forward_prefill", "forward_mqa", "_fused_qnorm_rope_kv_insert",
)
_ATTENTION_BATCH = ContextVar("deepseek_v4_pcp_attention_batch", default=None)
_PREFILL_ROWS = ContextVar("deepseek_v4_pcp_prefill_rows", default=None)
_LOCAL_QUERIES = ContextVar("deepseek_v4_pcp_local_queries", default=None)
_LOCAL_MQA = ContextVar("deepseek_v4_pcp_local_mqa", default=None)


def current_pcp_batch():
    """PCP batch of the DeepSeek V4 attention layer currently executing."""
    return _ATTENTION_BATCH.get()


def local_queries_enabled(batch) -> bool:
    return batch is not None and _LOCAL_QUERIES.get() is batch


@functools.cache
def _pcp_insert_op():
    # LightOp's PCP entry has separate Q/KV lengths and position vectors.
    # KVNorm remains the caller's responsibility, unlike the combined op.
    try:
        import lightop
    except ImportError:
        return None
    native = getattr(lightop, "op", None)
    if native is None:
        return None
    op = getattr(native, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp",
                 None)
    return op if callable(op) else None


def _can_use_local_queries(layer, batch) -> bool:
    # Native decode consumes global metadata and replicated queries. Keep its
    # established path, including mixed prefill/decode and replicated MTP.
    from vllm_hcu.platforms import envs as henvs

    eligible = (
        henvs.optional_custom_op_enabled(
            henvs.VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES
        )
        and layer.head_dim == 512
        and layer.n_local_heads == layer.padded_heads
        and layer.swa_cache_layer.kv_cache.dtype == torch.uint8
        and bool(batch.global_batch.is_prefilling_np.all())
    )
    if not eligible:
        return False
    from vllm.forward_context import get_forward_context
    metadata = get_forward_context().attn_metadata
    swa = (metadata.get(layer.swa_cache_layer.prefix)
           if isinstance(metadata, dict) else None)
    # A short prefill can be classified as decode by the attention builder
    # despite the request-level is_prefilling flag. Never pass shape-only Q
    # views to a decode kernel.
    if swa is None or getattr(swa, "num_decodes", 0) or getattr(swa, "num_decode_tokens", 0):
        return False
    if _pcp_insert_op() is None:
        logger.warning_once(
            "DeepSeek V4 PCP local queries unavailable: LightOp does not "
            "export fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp; "
            "using the full-query compatibility path."
        )
        return False
    return True


def _run_local_prefill(kernel, batch, root_q, decode_tokens, **kwargs):
    """Run only this rank's query rows while preserving global KV coordinates."""
    q = kwargs["q"]
    start = (q.storage_offset() - root_q.storage_offset()) // root_q.stride(0)
    selected, local_rows = batch.local_query_rows(
        start + decode_tokens, start + decode_tokens + q.shape[0],
    )
    local_mqa = _LOCAL_MQA.get()
    full_output = kwargs["output"]
    if local_mqa is None:
        full_output.zero_()
    if selected.numel() == 0:
        return
    if local_mqa is None:
        local_q = q.index_select(0, selected).contiguous()
    else:
        local_q = local_mqa[0].index_select(0, local_rows).contiguous()
    for name in ("indices", "topk_length"):
        kwargs[name] = kwargs[name].index_select(0, selected).contiguous()
    kwargs["q"] = local_q
    kwargs["output"] = torch.empty_like(local_q)
    kernel(**kwargs)
    if local_mqa is None:
        full_output.index_copy_(0, selected, kwargs["output"])
    else:
        local_mqa[1].index_copy_(0, local_rows, kwargs["output"])


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
        "forward_mqa": (("self", "q", "kv", "positions", "output"), {}),
        "_fused_qnorm_rope_kv_insert": (
            ("self", "q", "kv", "positions", "attn_metadata"), {}),
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

    def local_query_forward(self, batch, positions, hidden_states):
        from vllm_hcu.model_executor.layers.attention.pcp import (
            replicated_mtp_batch_scope,
        )

        logger.info_once("DeepSeek V4 PCP uses local queries and full KV.")
        # Do not restore hidden states or QR/Q merely to adapt to a
        # full-query API. Compressors still consume full KV scores.
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            originals["_run_parallel_input_projections"](self, hidden_states)
        )
        qr, raw_kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self.q_norm(qr)
        # KVNorm is token-local too. Normalize only this rank's KV before
        # gathering; the PCP cache writer must not repeat it.
        raw_kv = batch.gather(self.kv_norm(raw_kv).contiguous())
        kv_score = None if kv_score is None else batch.gather(kv_score)
        indexer_kv_score = (None if indexer_kv_score is None
                            else batch.gather(indexer_kv_score))
        full_tokens = batch.global_batch.num_tokens
        # Native indexer uses hidden_states only for metadata/shape; all
        # numerical projections above consumed the local tensor.
        shape_only_hidden = hidden_states[:1].expand(full_tokens, -1)
        output = hidden_states.new_zeros(
            (hidden_states.shape[0], self.padded_heads, self.head_dim)
        )
        with replicated_mtp_batch_scope():
            self._prepare_and_attn_fn(
                shape_only_hidden, qr, raw_kv, None, kv_score,
                indexer_kv_score, indexer_weights,
                batch.global_batch.positions[:full_tokens], output,
            )
        return originals["_o_proj"](self, output[:, :self.n_local_heads], positions)

    @functools.wraps(originals["forward"])
    def forward(self, positions, hidden_states, llama_4_scaling=None):
        from vllm_hcu.model_executor.layers.attention.pcp import (
            replicated_mtp_batch_scope,
        )
        from vllm_hcu.v1.glm53_pcp import _current_pcp_batch

        batch = _current_pcp_batch()
        if batch is None:
            return originals["forward"](self, positions, hidden_states, llama_4_scaling)
        replicated_decode = not batch.manager._global_has_prefill
        # Pure decode already has identical, globally ordered rows on every
        # PCP peer. Hide the partition from projection/indexer wrappers so
        # they do not gather those replicated inputs again.
        token = _ATTENTION_BATCH.set(None if replicated_decode else batch)
        local_token = _LOCAL_QUERIES.set(
            batch if not replicated_decode and _can_use_local_queries(self, batch)
            else None
        )
        try:
            if replicated_decode:
                num_tokens = batch.global_batch.num_tokens
                with replicated_mtp_batch_scope():
                    output = originals["forward"](
                        self, batch.global_batch.positions[:num_tokens],
                        hidden_states[:num_tokens], llama_4_scaling,
                    )
                # Native attention sees only valid rows. Restore the caller's
                # padded layout once, after all native output projections.
                return batch.localize(output)
            if local_queries_enabled(batch):
                return local_query_forward(self, batch, positions, hidden_states)
            full_hidden = batch.gather(hidden_states)
            # Native indexer/compressor calls already see complete KV. Reuse
            # the existing logical-PCP scope to suppress a second cache gather.
            with replicated_mtp_batch_scope():
                return originals["forward"](
                    self, batch.global_batch.positions[:full_hidden.shape[0]],
                    full_hidden, llama_4_scaling,
                )
        finally:
            _LOCAL_QUERIES.reset(local_token)
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
            if batch is None or local_queries_enabled(batch):
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

    @functools.wraps(originals["_fused_qnorm_rope_kv_insert"])
    def local_insert(self, q, kv, positions, attn_metadata):
        batch = _ATTENTION_BATCH.get()
        if not local_queries_enabled(batch):
            return originals["_fused_qnorm_rope_kv_insert"](
                self, q, kv, positions, attn_metadata,
            )
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata[self.swa_cache_layer.prefix]
        q = q.contiguous()
        # The PCP kernel has no KVNorm. local_query_forward() already
        # applied the learned norm on local KV before gathering.
        _pcp_insert_op()(
            q, kv.contiguous(),
            self.swa_cache_layer.kv_cache.view(
                self.swa_cache_layer.kv_cache.shape[0], -1,
            ),
            metadata.slot_mapping[:kv.shape[0]].to(dtype=torch.int64).contiguous(),
            batch.localize(positions).to(dtype=torch.int64),
            positions.to(dtype=torch.int64), self.rotary_emb.cos_sin_cache,
            self.eps, metadata.block_size,
        )
        return q

    @functools.wraps(originals["forward_mqa"])
    def local_mqa(self, q, kv, positions, output):
        batch = _ATTENTION_BATCH.get()
        if not local_queries_enabled(batch):
            return originals["forward_mqa"](self, q, kv, positions, output)
        # The native prefill loop owns global chunk/KV metadata. Give it
        # tiny shape-only views with a nonzero token stride so chunk
        # offsets remain recoverable, rather than a 2-GiB global Q buffer.
        full_tokens = batch.global_batch.num_tokens
        q_proxy = q.new_empty(full_tokens).as_strided(
            (full_tokens, q.shape[1], q.shape[2]), (1, 0, 0),
        )
        out_proxy = output.new_empty(full_tokens).as_strided(
            q_proxy.shape, (1, 0, 0),
        )
        token = _LOCAL_MQA.set((q, output))
        try:
            return originals["forward_mqa"](
                self, q_proxy, kv, positions, out_proxy,
            )
        finally:
            _LOCAL_MQA.reset(token)

    wrappers = [
        ("forward", forward), ("_run_parallel_input_projections", projections),
        ("_wq_b_proj", partition_query_projection(originals["_wq_b_proj"])),
        ("_o_proj", output_projection),
        ("_forward_prefill", prefill),
        ("_fused_qnorm_rope_kv_insert", local_insert), ("forward_mqa", local_mqa),
    ]

    attention_module = importlib.import_module("vllm.models.deepseek_v4.attention")
    quant = getattr(attention_module, "fused_indexer_q_rope_quant", None)
    if quant is not None and not getattr(quant, _MARKER, False):
        @functools.wraps(quant)
        def local_indexer_quant(*args, **kwargs):
            batch = _ATTENTION_BATCH.get()
            if local_queries_enabled(batch):
                if args:
                    args = (batch.localize(args[0]), *args[1:])
                else:
                    kwargs["positions"] = batch.localize(kwargs["positions"])
            return quant(*args, **kwargs)
        setattr(local_indexer_quant, _MARKER, True)
        attention_module.fused_indexer_q_rope_quant = local_indexer_quant

    # These are V4-only cache backends; all of them now consume full KV on
    # each rank. Do not advertise PCP on generic MLA/SWA backends.
    for backend in (
        rocm.DeepseekV4ROCMAiterMLASparseBackend,
        CompressorBackend, DeepseekSparseSWABackend,
    ):
        backend.supports_pcp = classmethod(lambda cls: True)
    for name, wrapper in wrappers:
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


__all__ = [
    "PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module", "current_pcp_batch",
    "local_queries_enabled",
]
