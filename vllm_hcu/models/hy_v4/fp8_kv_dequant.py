# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather + dequantize HY V4's ``fp8_ds_mla`` KV cache to BF16 via LightOp.

The fp8 FlashMLA sparse kernel hardcodes DeepSeek's ``fp8_ds_mla`` geometry, so
opting out of it (via ``VLLM_HCU_HYV4_FP8_KV_DEQUANT``) means the BF16 sparse
kernel must run instead, which needs a BF16 KV cache. This module delegates the
fused gather + FP8->BF16 up-convert + compact-index remap to LightOp's
vendor-optimized HCU op ``decode_gather_and_up_convert_with_indices``, reading
only the ``topk`` slots the sparse kernel actually reads into a compact BF16
buffer and keeping a fixed output shape so the decode path stays
CUDA-graph-safe.

The ``fp8_ds_mla`` per-token layout is 656 bytes, written by
``concat_and_cache_ds_mla_kernel`` (csrc/hcu_cache_kernel.cu):

    bytes [0,   512):  512 x fp8_e4m3   NoPE, 4 tiles of 128, one fp32 scale each
    bytes [512, 528):    4 x fp32       per-tile NoPE scales (tile_scale=amax/448)
    bytes [528, 656):   64 x bf16       RoPE, copied verbatim

LightOp reconstructs the BF16 output row ``[NoPE(512) || RoPE(64)]`` = 576,
matching the layout the fp8 *prefill* path produces and what
``flash_mla_sparse_fwd`` consumes (``d_v=512``). The op also writes the compact
row indices (``token t``'s ``k``-th slot -> row ``t*topk+k``, -1 preserved),
identical to the standalone Triton implementation and verified bit-exact against
it on realistic fp8 data.
"""

import torch

# fp8_ds_mla layout constants (see module docstring / concat_and_cache_ds_mla).
_ROW_BYTES = 656
_NOPE_DIM = 512
_ROPE_DIM = 64

_LIGHTOP_GATHER = None


def _resolve_lightop_gather():
    """Resolve LightOp's gather/up-convert op, failing closed if unavailable.

    The experimental path is deliberately opt-in, so a missing LightOp or a
    LightOp build without the expected op is a hard error rather than a silent
    fallback to the fp8 kernel.
    """
    global _LIGHTOP_GATHER
    if _LIGHTOP_GATHER is not None:
        return _LIGHTOP_GATHER
    try:
        from lightop import op as lightop_op
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_HCU_HYV4_FP8_KV_DEQUANT is set but LightOp could not be "
            "imported; install LightOp or unset the env var."
        ) from exc
    gather = getattr(lightop_op, "decode_gather_and_up_convert_with_indices", None)
    if gather is None:
        raise RuntimeError(
            "Installed LightOp does not expose "
            "decode_gather_and_up_convert_with_indices; upgrade LightOp or "
            "unset VLLM_HCU_HYV4_FP8_KV_DEQUANT."
        )
    _LIGHTOP_GATHER = gather
    return gather


def gather_dequantize_fp8_ds_mla_cache(
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tokens_per_request: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize only the topk-selected ``fp8_ds_mla`` slots to a compact BF16.

    Delegates to LightOp's fused gather/up-convert HCU op. Only the
    ``num_tokens * topk`` slots the sparse kernel reads are dequantized, into a
    compact ``(num_tokens * topk, kv_lora_rank + qk_rope_head_dim)`` BF16 buffer.
    Both ``topk`` (constant) and ``num_tokens`` (fixed per CUDA-graph batch
    bucket) are static, so the output shape is fixed and the decode path stays
    CUDA-graph-capturable.

    Args:
        kv_cache: ``(num_blocks, block_size, 656)`` cache, ``fp8_ds_mla`` layout.
        topk_indices: ``(num_tokens, topk)`` global cache slots
            (``block * block_size + pos``), with -1 marking unfilled entries.
        kv_lora_rank: NoPE dim (512 for HY V4).
        qk_rope_head_dim: RoPE dim (64 for HY V4).
        tokens_per_request: Query rows each request contributes in this batch,
            i.e. the MTP width (``speculative_config.num_speculative_tokens``,
            so MTP3 -> 3) and 1 when MTP is off. LightOp uses it to group rows
            back into requests, matching ``q``'s
            ``(num_requests * tokens_per_request, topk)`` layout.

    Returns:
        ``(kv_bf16, new_indices)`` where ``kv_bf16`` is BF16
        ``(num_tokens * topk, kv_lora_rank + qk_rope_head_dim)`` laid out so
        token ``t``'s ``k``-th selected slot is row ``t * topk + k``, and
        ``new_indices`` is ``(num_tokens, topk)`` int32 addressing those rows
        (-1 preserved for unfilled entries), ready for ``flash_mla_sparse_fwd``.
    """
    assert kv_lora_rank == _NOPE_DIM and qk_rope_head_dim == _ROPE_DIM, (
        f"gather_dequantize_fp8_ds_mla_cache hardcodes the DeepSeek fp8_ds_mla "
        f"geometry (NoPE={_NOPE_DIM}, RoPE={_ROPE_DIM}), got "
        f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}."
    )
    gather = _resolve_lightop_gather()

    cache_u8 = kv_cache.view(torch.uint8).reshape(-1, _ROW_BYTES)
    assert cache_u8.shape[1] == _ROW_BYTES, (
        f"expected {_ROW_BYTES}-byte fp8_ds_mla rows, got {cache_u8.shape[1]}."
    )
    num_tokens, topk = topk_indices.shape
    out_dim = kv_lora_rank + qk_rope_head_dim
    idx32 = topk_indices.to(torch.int32).contiguous()
    gathered = torch.empty(
        (num_tokens, topk, out_dim), dtype=torch.bfloat16, device=kv_cache.device
    )
    compact_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=kv_cache.device
    )
    # LightOp caps validity on the topk axis at cache_seqlens[q]. The fp8 kernel
    # itself is fed cache_lens = max_model_len (an upper bound, not the true
    # length), so real validity comes from the -1 entries in topk_indices. A
    # full-width cap (>= topk) reproduces that exactly: the op then masks only
    # the -1 slots, matching the fp8 kernel and the Triton path bit-for-bit.
    cache_seqlens = torch.full(
        (num_tokens,), topk, dtype=torch.int32, device=kv_cache.device
    )
    gather(
        cache_u8, idx32, gathered, cache_seqlens, compact_indices, tokens_per_request
    )

    return gathered.reshape(num_tokens * topk, out_dim), compact_indices
