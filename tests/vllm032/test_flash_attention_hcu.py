from __future__ import annotations

import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("VLLM032_HCU_TEST") != "1",
    reason="set VLLM032_HCU_TEST=1 on a prepared HCU node",
)


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1).float()
    expanded_value = value.repeat_interleave(groups, dim=1).float()
    scores = torch.einsum("qhd,khd->qhk", query.float(), expanded_key)
    scores *= 1.0 / math.sqrt(query.shape[-1])
    query_positions = key.shape[0] - query.shape[0] + torch.arange(
        query.shape[0], device=query.device
    )
    key_positions = torch.arange(key.shape[0], device=query.device)
    scores.masked_fill_(
        key_positions[None, None, :] > query_positions[:, None, None],
        float("-inf"),
    )
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,khd->qhd", probabilities, expanded_value).to(
        query.dtype
    )


@pytest.mark.parametrize(("query_len", "kv_len"), ((3, 3), (1, 5)))
@torch.inference_mode()
def test_hcu_writer_to_vendor_varlen_matches_reference(query_len, kv_len):
    import vllm._C_stable_libtorch  # noqa: F401
    from flash_attn import flash_attn_varlen_func
    from hcu_vllm_das.attention.flash_attn import HcuFlashAttentionImpl
    from vllm import _custom_ops as ops

    torch.manual_seed(17 + query_len)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size, num_blocks = 64, 1
    num_query_heads, num_kv_heads, head_size = 4, 2, 64
    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        kv_len,
        num_kv_heads,
        head_size,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    packed_cache = torch.empty(
        num_blocks,
        num_kv_heads,
        block_size,
        2 * head_size,
        device=device,
        dtype=dtype,
    )
    impl = object.__new__(HcuFlashAttentionImpl)
    impl.head_size = head_size
    key_cache, value_cache = impl._get_kv_cache_views(packed_cache)
    write_key_cache, write_value_cache = impl._get_kv_cache_write_views(
        packed_cache
    )
    slot_mapping = torch.arange(kv_len, device=device, dtype=torch.long)
    scale = torch.ones((), device=device, dtype=torch.float32)
    ops.reshape_and_cache_flash(
        key,
        value,
        write_key_cache,
        write_value_cache,
        slot_mapping,
        "auto",
        scale,
        scale,
    )
    torch.testing.assert_close(
        key_cache[0, :, :kv_len].transpose(0, 1), key
    )
    torch.testing.assert_close(
        value_cache[0, :, :kv_len].transpose(0, 1), value
    )

    output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=torch.tensor(
            [0, query_len], device=device, dtype=torch.int32
        ),
        seqused_k=torch.tensor([kv_len], device=device, dtype=torch.int32),
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        softmax_scale=1.0 / math.sqrt(head_size),
        causal=True,
        block_table=torch.zeros((1, 1), device=device, dtype=torch.int32),
        layout="bhsd",
        fa_version=2,
    )
    expected = _reference_attention(query, key, value)

    torch.testing.assert_close(output, expected, atol=1.6e-2, rtol=1.6e-2)
