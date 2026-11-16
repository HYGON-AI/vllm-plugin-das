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
