# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU Triton MLA backend with DSpark non-causal draft support.

The wheel's Triton MLA backend only serves single-token causal decode. DSpark
draft models (including the K3 DSpark draft) attend non-causally over their
whole ``(1 + num_speculative_tokens)`` block, so the selector rejects the
wheel backend for the draft's attention groups.  The evaluated fork enables
this at three points (fork 877e6996 ``vllm/v1/attention/backends/mla/
triton_mla.py``): a non-causal capability flag for the backend selector, a
spec-aware reorder threshold so multi-token decode blocks are admitted, and a
per-token flattening of the decode rows in ``forward_mqa``.  This module
ports those three pieces onto the wheel (0.25.1) builder/impl classes.

Known deviation from the fork, RESOLVED with an origin stamp: the wheel
``MLACommonMetadata`` carries no ``causal`` flag, so the producer-side patch
``patch_dspark_non_causal_stamp`` stamps draft-built metadata objects with
``NON_CAUSAL_DRAFT_STAMP`` (fork parity for the ``causal`` field).  Only
stamped (non-causal draft) batches are flattened; unstamped multi-token
batches -- i.e. causal speculative verify routed to this backend -- are
expanded with exact per-row causal lengths instead, so an early query never
reads the KV of tokens committed after its own position (review 2026-10-12
P1).  Row counts alone decide nothing about semantics anymore.

Fork parity note (measured 2026-10-10): the flattened rows must stay LOCAL
to the forward call exactly as in the fork -- the shared ``attn_metadata``
is consumed by every attention layer of the step and must keep its intact
per-request rows.  Mutating it (an earlier port) re-flattens per layer and
aliases sibling requests onto the first request's KV blocks, which is the
bs>=2 draft acceptance collapse (per-position healthy prefix then deep
falloff, growing with concurrency; serial requests are immune).
"""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm import envs
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.triton_mla import (
    _compute_num_kv_splits,
    decode_attention_fwd,
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

# Producer-side stamp: ``patch_dspark_non_causal_stamp`` sets this attribute
# on every per-group metadata object built for the speculative draft step
# (fork ``causal=False`` semantics).  Defined in the dependency-free
# ``_stamps`` leaf module (the worker patch layer imports it without torch)
# and re-exported here for the forward path.
from vllm_hcu.v1.attention.backends.mla._stamps import (
    NON_CAUSAL_DRAFT_STAMP,
)


def expand_causal_rows(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int,
    num_decodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token rows with EXACT causal lengths for verify batches.

    A causal multi-token batch routes here when the scheduler packs
    ``num_decode_tokens == num_decodes * q`` query tokens for speculative
    verification.  Query row ``i`` of a request (token order 0..q-1) may read
    the committed prefix up to and including its own token: with ``q`` rows
    per request and ``s = seq_lens[req]`` (all ``q`` tokens already written
    to the KV cache by the common MLA pre-write), lengths are
    ``s - q + 1, ..., s - 1, s``.  Full-``s`` rows for every query -- the
    non-causal shape -- would let early queries read future tokens and poison
    the logits (review 2026-10-12 P1 repro: prefix 2 + 2 queries -> [3, 4],
    not [4, 4]).  Single-token batches pass through untouched.
    """

    if num_decodes <= 0 or num_decode_tokens <= num_decodes:
        return block_table, seq_lens
    query_len = num_decode_tokens // num_decodes
    if query_len <= 1:
        return block_table, seq_lens
    num_reqs = seq_lens.shape[0]
    inner = torch.arange(query_len, device=seq_lens.device, dtype=seq_lens.dtype)
    offsets = (query_len - 1) - inner  # 0 per-token shift, oldest row first
    causal_lens = (
        seq_lens.repeat_interleave(query_len) - offsets.repeat(num_reqs)
    ).clamp_min(0)  # padded tail requests carry len 0/1, never negative
    return (
        block_table.repeat_interleave(query_len, dim=0),
        causal_lens,
    )


def flatten_non_causal_rows(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int,
    num_decodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decode row per query token (fork 877e6996 ``forward_mqa`` parity).

    Returns (possibly new) LOCAL row tables; inputs are never mutated.  Every
    flattened row attends to the committed KV prefix of its own request and
    never to sibling block tokens, which is the DSpark non-causal block
    semantics.  Row counts already per-token are a no-op, so causal decode
    (one token per request) passes through untouched.
    """

    if num_decodes <= 0 or num_decode_tokens <= num_decodes:
        return block_table, seq_lens
    query_len = num_decode_tokens // num_decodes
    if query_len <= 1:
        return block_table, seq_lens
    return (
        block_table.repeat_interleave(query_len, dim=0),
        seq_lens.repeat_interleave(query_len),
    )


class HcuTritonMLAMetadataBuilder(TritonMLAMetadataBuilder):
    """Builder admitting non-causal multi-token decode (DSpark draft)."""

    # Wheel default is UNIFORM_BATCH; the explicit override documents that a
    # non-causal multi-token block still satisfies the contract (the forward
    # path never device->host syncs).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # Opt the builder into the spec-aware reorder threshold: the base
    # ``__init__`` raises ``reorder_batch_threshold`` to the speculative
    # block length when ``query_len_support`` is not SINGLE_ONLY.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    supports_non_causal_multi_token_decode: ClassVar[bool] = True

    def _reserve_attn_logits_workspace(self) -> None:
        """Wheel reservation scaled for flattened non-causal decode rows.

        The reserved decode-row count must cover every flattened query row
        (``max_num_seqs * reorder_batch_threshold``), otherwise the first
        non-causal forward would have to grow the workspace after it is
        locked (fork ``_reserve_attn_logits_workspace`` parity).
        """

        if not is_workspace_manager_initialized():
            return
        threshold = self.reorder_batch_threshold or 1
        B = self.vllm_config.scheduler_config.max_num_seqs * threshold
        # DCP all-gathers the query heads before forward_mqa.
        q_num_heads = self.num_heads * self.dcp_world_size
        max_splits = _compute_num_kv_splits(
            self.model_config.max_model_len,
            current_platform.num_compute_units(),
        )
        lse_dim = self.mla_dims.kv_lora_rank + 1
        current_workspace_manager().get_simultaneous(
            ((B, q_num_heads, max_splits, lse_dim), torch.float32),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ):
        """Reject DCP>1 multi-token DRAFT batches before the wheel split.

        With ``decode_context_parallel_size > 1`` the wheel resets
        ``reorder_batch_threshold`` to 1 (``supports_dcp_with_varlen``
        defaults to False on this backend; see
        ``AttentionMetadataBuilder._init_reorder_batch_threshold``), so the
        spec-aware threshold our ``query_len_support=UNIFORM`` requested is
        silently discarded.  A multi-token DRAFT batch -- the DSpark block,
        ``q = 1 + num_speculative_tokens`` -- is then routed to the PREFILL
        branch of the inherited build (``split_decodes_and_prefills`` sees
        a first query len > 1), where it dies on
        ``assert seq_lens_cpu is not None``: the draft metadata comes from
        ``build_attn_metadata`` with its ``seq_lens_cpu_upper_bound``
        parameter left at the ``None`` default (the speculator
        ``_build_draft_attn_metadata`` chain never passes it), and the
        ``MLACommonMetadataBuilder.build`` prefill branch asserts the
        bound.  Under full capture the wheel
        ``build_for_cudagraph_capture`` own query-len assert fires first
        instead.  The forward-time DCP guard never runs for those batches
        (review 2026-10-14 P2).

        Scope against over-rejection: TARGET/plain PREFILL batches under
        DCP>1 are NOT rejected -- they carry a real
        ``seq_lens_cpu_upper_bound`` and stay on the wheel's supported
        chunked-prefill path.  The guard only fires when BOTH the collapse
        trigger (``max_query_len > 1``) AND the missing-bound signature of
        a draft/probe batch (``seq_lens_cpu_upper_bound is None``) hold.
        Pure q=1 decode keeps the inherited wheel behaviour; the
        forward-time guard remains the second barrier for unstamped
        multi-token batches that already reached forward_mqa.
        """

        if (
            self.dcp_world_size > 1
            and common_attn_metadata.max_query_len > 1
            # The sentinel (not ``None``) distinguishes "field ABSENT"
            # (bare test stubs / a wheel without the field) from a draft
            # batch whose bound is genuinely None: only the latter matches
            # the crash signature, an absent field fails OPEN by design.
            and getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", "missing")
            is None
        ):
            raise RuntimeError(
                "HCU triton MLA: multi-token draft batches (max query len "
                f"{common_attn_metadata.max_query_len}) are not supported "
                "with decode_context_parallel_size > 1: without DCP varlen "
                "support the reorder threshold collapses to 1, the batch is "
                "mis-routed to the prefill builder and would crash on the "
                "missing seq_lens_cpu_upper_bound. Disable DCP for "
                "speculative decoding or use a DCP-aware backend."
            )
        return super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )


class HcuTritonMLAImpl(TritonMLAImpl):
    """Impl flattening non-causal multi-token blocks to per-token rows.

    The kernel body is the wheel ``(0.25.1)`` ``forward_mqa`` verbatim with
    the fork's LOCAL row flattening inserted before the kernel launch.  The
    shared ``attn_metadata`` object is never written: every attention layer
    of the step flattens from the same intact per-request rows.
    """

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            num_kv_splits = _compute_num_kv_splits(
                attn_metadata.max_seq_len, self._sm_count
            )

        # NOTE: the +1 stores the LogSumExp (LSE) that the stage2 kernel uses to
        # merge partial attention outputs across splits. The scratch is served
        # from the shared workspace (reserved at max in the metadata builder),
        # so there is no per-call allocation on the decode hot path.
        logits_shape = (B, q_num_heads, num_kv_splits, self.kv_lora_rank + 1)
        if is_workspace_manager_initialized():
            (attn_logits,) = current_workspace_manager().get_simultaneous(
                (logits_shape, torch.float32),
            )
        else:
            attn_logits = torch.empty(
                logits_shape, dtype=torch.float32, device=q.device
            )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Row semantics keyed on the producer stamp, NOT row counts:
        #  - stamped -> non-causal DSpark draft block, fork-parity flatten;
        #  - unstamped -> causal batch (plain decode or spec verify):
        #    single-token rows pass through, uniform multi-token verify rows
        #    get exact per-row causal lengths.
        # DCP>1 + unstamped multi-token is REJECTED loudly (latest review
        # P1, 2026-10-15): under decode-context parallelism the query is
        # all-gathered (mla_attention.py:807) while each rank owns only its
        # interleaved KV slice, so global-lens causal expansion cannot
        # produce per-shard masks -- the lse reduction (cp_lse_ag_out_rs)
        # then folds in wrong per-rank partitions and corrupts logits.
        # ``expand_causal_rows`` is our own addition (the fork has no causal
        # expansion and has never validated this pairing), so the fork-
        # parity exemption does not apply. Non-causal stamped batches keep
        # the fork-parity flatten (repeat_interleave is rank-symmetric).
        # The guard keys on UNSTAMPED multi-token shape; non-uniform packs
        # (`num_decode_tokens` is not a multiple of `num_decodes`) are also
        # rejected rather than passed through: their row counts already
        # cannot map back to requests, so failing loudly is the safer
        # contract under DCP (their unpadded path is expand-passthrough).
        is_non_causal = getattr(attn_metadata, NON_CAUSAL_DRAFT_STAMP, False)
        unstamped_multi_token = (
            not is_non_causal
            and attn_metadata.num_decode_tokens > attn_metadata.num_decodes
        )
        if unstamped_multi_token and self.dcp_world_size > 1:
            raise RuntimeError(
                "HCU triton MLA: causal multi-token verification batches "
                "are not supported with decode_context_parallel_size > 1 "
                "(per-shard causal masks would be derived from global "
                "seq_lens); disable DCP for speculative decoding or use a "
                "DCP-aware backend."
            )
        if is_non_causal:
            block_table, seq_lens = flatten_non_causal_rows(
                attn_metadata.decode.block_table,
                attn_metadata.decode.seq_lens,
                attn_metadata.num_decode_tokens,
                attn_metadata.num_decodes,
            )
        else:
            block_table, seq_lens = expand_causal_rows(
                attn_metadata.decode.block_table,
                attn_metadata.decode.seq_lens,
                attn_metadata.num_decode_tokens,
                attn_metadata.num_decodes,
            )

        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            block_table,
            seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            # Latest review P2 (2026-10-15): pass _k_scale for BOTH sides.
            # The MLA combined c_kv cache is written with _k_scale only, so
            # any reachable path that descales those bytes must use the same
            # factor: the kernel's normal MHA fallback (kv_group_num == 1,
            # one local q head) loads this very cache as V and multiplies by
            # v_scale, while the grouped MLA path merely transposes k and
            # ignores v_scale entirely.  _v_scale only suits true GQA V
            # caches; for this cache a separately calibrated _v_scale would
            # dequantize with the wrong factor (fork 877e6996 passes
            # _k_scale for both sides, as does upstream).
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse


class HcuTritonMLABackend(TritonMLABackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_non_causal(cls) -> bool:
        """Non-causal blocks are flattened to per-token decode rows in
        ``HcuTritonMLAImpl.forward_mqa`` (the decode kernel has no intra-block
        causal masking). Enables the non-causal DSpark draft path."""

        return True

    @staticmethod
    def get_builder_cls() -> type[HcuTritonMLAMetadataBuilder]:
        return HcuTritonMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[HcuTritonMLAImpl]:
        return HcuTritonMLAImpl
