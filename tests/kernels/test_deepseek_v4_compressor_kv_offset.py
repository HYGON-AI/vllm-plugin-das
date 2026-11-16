# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache as ops
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_compressor_kv_offset


@pytest.mark.hcu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
def test_finalize_kernel_preserves_cache_pointer_range():
    patch_deepseek_v4_compressor_kv_offset.apply(ops)
    assert not patch_deepseek_v4_compressor_kv_offset.apply(ops)

    slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    scratch = torch.ones(512, device="cuda")
    weight = torch.ones(512, dtype=torch.bfloat16, device="cuda")
    cache = torch.empty(1200, dtype=torch.uint8, device="cuda")
    kernel = ops._finalize_norm_rope_quant_store_sparse_attn
    compiled = kernel.warmup(
        scratch, 512, slot, slot, weight, 1e-6, scratch, 64, cache, slot, 2,
        HEAD_SIZE=512, TRITON_BLOCK_SIZE=512, COMPRESS_RATIO=128,
        ROPE_HEAD_DIM=64, FP8_MAX=448., QUANT_BLOCK=64, TOKEN_STRIDE=576,
        SCALE_DIM=8, KV_BLOCK_STRIDE=1677888, SANITIZE_CACHE_NANS=False,
        grid=(1,),
    )
    # Even a small warmup tensor must not specialize this kernel's KV pointer.
    signature = next(line for line in compiled.asm["ttir"].splitlines()
                     if "tt.func public" in line)
    cache_arg = signature.split("%k_cache_ptr:", 1)[1].split("%kv_slot_mapping_ptr:", 1)[0]
    assert "tt.pointer_range" not in cache_arg


@pytest.mark.hcu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
def test_gather_kernel_preserves_cache_pointer_range():
    from vllm.models.deepseek_v4.common.ops import cache_utils
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_gather_kv_offset

    patch_deepseek_v4_gather_kv_offset.apply(cache_utils)
    assert not patch_deepseek_v4_gather_kv_offset.apply(cache_utils)
    slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.empty(512, dtype=torch.bfloat16, device="cuda")
    cache = torch.empty(1200, dtype=torch.uint8, device="cuda")
    compiled = cache_utils._dequantize_and_gather_k_kernel.warmup(
        output, 512, 512, cache, slot, slot, 0, None, max_blocks_per_seq=1,
        fp8_dim=448, bf16_dim=64, scale_dim=8, quant_block=64,
        cache_block_size=2, token_data_size=576, block_stride=1677888,
        output_dim=512, fp8_max=448., n_quant_blocks=7, grid=(1, 1),
    )
    signature = next(line for line in compiled.asm["ttir"].splitlines()
                     if "tt.func public" in line)
    cache_arg = signature.split("%k_cache_ptr:", 1)[1].split("%seq_lens_ptr:", 1)[0]
    assert "tt.pointer_range" not in cache_arg


def test_compressor_state_cache_specialization():
    from importlib import import_module
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_state_cache_offset

    target = import_module(patch_deepseek_v4_state_cache_offset.TARGET_MODULE)
    patch_deepseek_v4_state_cache_offset.apply(target)
    kernel = target._save_partial_states_kernel
    assert kernel.params[kernel.arg_names.index("state_cache_ptr")].do_not_specialize

    patch_deepseek_v4_compressor_kv_offset.apply(ops)
    kernel = ops._fused_kv_compress_norm_rope_insert_sparse_attn
    for pointer in ("state_cache_ptr", "k_cache_ptr"):
        assert kernel.params[kernel.arg_names.index(pointer)].do_not_specialize


@pytest.mark.hcu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
@pytest.mark.parametrize("bf16", [False, True], ids=["two_stage_fp8", "plain_bf16"])
def test_compress_and_gather_beyond_two_gib(bf16):
    from importlib import import_module
    from types import SimpleNamespace

    from vllm.models.deepseek_v4.common.ops import cache_utils
    from vllm_hcu.patch.worker.core_fix import (
        patch_deepseek_v4_gather_kv_offset,
        patch_deepseek_v4_state_cache_offset,
    )
    from vllm_hcu.v1.attention.ops.deepseek_v4_bf16_cache import (
        bf16_sparse_attn_decode, gather_bf16_k_cache,
    )
    from vllm_hcu.v1.attention.ops.deepseek_v4_bf16_compressor import (
        compress_norm_rope_store_bf16,
    )

    state_ops = import_module(patch_deepseek_v4_state_cache_offset.TARGET_MODULE)
    patch_deepseek_v4_state_cache_offset.apply(state_ops)
    patch_deepseek_v4_compressor_kv_offset.apply(ops)
    patch_deepseek_v4_gather_kv_offset.apply(cache_utils)

    # Physical block 2 begins at 2 GiB + 8192 bytes in each real allocation.
    page_stride_bytes = 2**30 + 4096
    if torch.cuda.mem_get_info()[0] < 4 * page_stride_bytes + 512 * 1024**2:
        pytest.skip("requires 4.5 GiB free HCU memory for two strided caches")
    block = 2
    ratio = 128
    dim = 512
    state = torch.empty_strided(
        (3, ratio, 2 * dim), (page_stride_bytes // 4, 2 * dim, 1),
        device="cuda", dtype=torch.float32,
    )
    width = dim if bf16 else 584
    dtype = torch.bfloat16 if bf16 else torch.uint8
    cache = torch.empty_strided(
        (3, 2, width), (page_stride_bytes // dtype.itemsize, width, 1),
        device="cuda", dtype=dtype,
    )
    # The two lower physical blocks must stay untouched.
    for page in (0, 1):
        state[page].fill_(23)
        cache[page].fill_(23)
    positions = torch.arange(ratio, device="cuda")
    slots = block * ratio + positions
    mapping = torch.tensor([[block]], dtype=torch.int32, device="cuda")
    sign = (torch.arange(dim, device="cuda") % 2 * 2 - 1).float()
    # Only half the channels vary across rows. Reading a single row or
    # duplicating the last row cannot produce the expected normalized vector.
    varying = (torch.arange(dim, device="cuda") % 4 < 2).float()
    values = sign + ((2 * positions[:, None] - (ratio - 1)) / ratio) * varying
    zeros = torch.zeros_like(values)
    state_ops.save_partial_states(
        values, zeros, zeros, positions, state, slots, ratio, dim, ratio,
    )
    torch.testing.assert_close(state[block, :, :dim], values, rtol=0, atol=0)
    assert not state[block, :, dim:].count_nonzero()
    rope = torch.zeros((ratio, 64), device="cuda")
    rope[:, :32] = 1
    kv_slots = torch.full_like(slots, -1)
    kv_slots[-1] = block * cache.shape[1]
    args = dict(
        state_cache=state, num_actual=ratio, positions=positions, slot_mapping=slots,
        token_to_req_indices=torch.zeros_like(slots, dtype=torch.int32),
        block_table=mapping, block_size=ratio, state_width=dim,
        cos_sin_cache=rope, kv_cache=cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=kv_slots),
        rms_norm_weight=torch.ones_like(sign), rms_norm_eps=1e-6,
        head_dim=dim, rope_head_dim=64, compress_ratio=ratio, overlap=False,
    )
    lens = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, 1, dim), float("nan"), dtype=torch.bfloat16, device="cuda")
    if bf16:
        compress_norm_rope_store_bf16(**args)
        gather_bf16_k_cache(out, cache, lens, None, mapping, 2, 0)
        # Exercise both BF16 attention cache pointers as well as the prefill gather.
        q = torch.zeros((1, 16, dim), dtype=out.dtype, device="cuda")
        decoded = torch.empty_like(q)
        index = kv_slots[-1:].int()
        indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        bf16_sparse_attn_decode(
            q, cache, cache, False, None, None, index, lens, index, indptr,
            index, indptr, None, 1., dim, 448, 64, decoded,
        )
        torch.testing.assert_close(decoded, sign.to(out.dtype).expand_as(q), rtol=0, atol=0)
    else:
        ops.compress_norm_rope_store_two_stage_triton(
            **args, pdl_kwargs={}, use_fp4_cache=False, quant_block=64,
            token_stride=576, scale_dim=8, num_decode_tokens=0,
            compress_scratch=torch.empty_like(values),
        )
        cache_utils.dequantize_and_gather_k_cache_triton(out, cache, lens, None, mapping, 2, 0)
    # Uniform scores average the 128 rows. RMSNorm reduces them to alternating
    # +/-1, exactly representable in both output formats; RoPE is the identity.
    torch.testing.assert_close(out.flatten(), sign.to(out.dtype), rtol=0, atol=0)
    for page in (0, 1):
        assert (state[page] == 23).all() and (cache[page] == 23).all()
