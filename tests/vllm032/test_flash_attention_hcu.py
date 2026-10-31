from __future__ import annotations

import math
import os
from types import SimpleNamespace

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


def _make_packed_cache(
    *,
    layout_name: str,
    num_blocks: int,
    num_heads: int,
    block_size: int,
    head_size: int,
    device: torch.device,
    dtype: torch.dtype,
    block_padding: int = 0,
) -> torch.Tensor:
    packed_head_size = 2 * head_size
    block_stride = (
        num_heads * block_size * packed_head_size + block_padding
    )
    if layout_name == "LBHNC":
        strides = (
            block_stride,
            block_size * packed_head_size,
            packed_head_size,
            1,
        )
    elif layout_name == "LBNHC":
        strides = (
            block_stride,
            packed_head_size,
            num_heads * packed_head_size,
            1,
        )
    else:
        raise AssertionError(f"unexpected test layout: {layout_name}")
    return torch.empty_strided(
        (num_blocks, num_heads, block_size, packed_head_size),
        strides,
        device=device,
        dtype=dtype,
    )


def _make_impl(layout_name: str, head_size: int):
    from hcu_vllm_das.attention.flash_attn import HcuFlashAttentionImpl
    from vllm.v1.kv_cache_layout import KVCacheLayout

    impl = object.__new__(HcuFlashAttentionImpl)
    impl.head_size = head_size
    impl.kv_cache_dtype = "auto"
    layout = getattr(KVCacheLayout, layout_name)
    impl.cache_config = SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: layout
    )
    return impl


def _e5m2_round_trip(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float8_e5m2).to(tensor.dtype)


@pytest.mark.parametrize(
    (
        "layout_name",
        "query_len",
        "kv_len",
        "num_query_heads",
        "num_kv_heads",
        "dtype",
    ),
    tuple(
        (layout, query_len, kv_len, num_query_heads, num_kv_heads, dtype)
        for layout in ("LBHNC", "LBNHC")
        for query_len, kv_len, num_query_heads, num_kv_heads, dtype in (
            (3, 3, 4, 2, torch.bfloat16),
            (1, 5, 4, 2, torch.bfloat16),
            (5, 5, 14, 2, torch.bfloat16),
            (3, 3, 2, 2, torch.float16),
            (3, 3, 4, 2, torch.float16),
        )
    ),
)
@torch.inference_mode()
def test_hcu_writer_to_vendor_varlen_matches_reference(
    layout_name,
    query_len,
    kv_len,
    num_query_heads,
    num_kv_heads,
    dtype,
):
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm import _custom_ops as ops

    torch.manual_seed(17 + query_len)
    device = torch.device("cuda")
    block_size, num_blocks, cache_block = 64, 2, 1
    head_size = 64
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
    packed_cache = _make_packed_cache(
        layout_name=layout_name,
        num_blocks=num_blocks,
        num_heads=num_kv_heads,
        block_size=block_size,
        head_size=head_size,
        device=device,
        dtype=dtype,
        block_padding=64,
    )
    impl = _make_impl(layout_name, head_size)
    key_cache, value_cache = impl._get_kv_cache_views(packed_cache)
    write_key_cache, write_value_cache = impl._get_kv_cache_write_views(
        packed_cache
    )
    slot_mapping = cache_block * block_size + torch.arange(
        kv_len, device=device, dtype=torch.long
    )
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
    cached_key = key_cache[cache_block, :, :kv_len].transpose(0, 1)
    cached_value = value_cache[cache_block, :, :kv_len].transpose(0, 1)
    if layout_name == "LBNHC":
        cached_key = key_cache[cache_block, :kv_len]
        cached_value = value_cache[cache_block, :kv_len]
    torch.testing.assert_close(cached_key, key)
    torch.testing.assert_close(cached_value, value)

    output = impl._run_flash_attention(
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
        block_table=torch.full(
            (1, 1), cache_block, device=device, dtype=torch.int32
        ),
        fa_version=2,
    )
    expected = _reference_attention(query, key, value)

    torch.testing.assert_close(output, expected, atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.parametrize("layout_name", ("LBHNC", "LBNHC"))
@torch.inference_mode()
def test_hcu_writer_and_vendor_varlen_replay_with_changed_graph_inputs(
    layout_name,
):
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm import _custom_ops as ops

    torch.manual_seed(29)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, kv_len = 1, 5
    block_size, num_query_heads, num_kv_heads, head_size = 64, 4, 2, 64
    query = torch.randn(
        query_len, num_query_heads, head_size, device=device, dtype=dtype
    )
    key = torch.randn(
        kv_len, num_kv_heads, head_size, device=device, dtype=dtype
    )
    value = torch.randn_like(key)
    cache_block = 1
    packed_cache = _make_packed_cache(
        layout_name=layout_name,
        num_blocks=2,
        num_heads=num_kv_heads,
        block_size=block_size,
        head_size=head_size,
        device=device,
        dtype=dtype,
        block_padding=64,
    )
    output = torch.empty_like(query)
    impl = _make_impl(layout_name, head_size)
    key_cache, value_cache = impl._get_kv_cache_views(packed_cache)
    write_key_cache, write_value_cache = impl._get_kv_cache_write_views(
        packed_cache
    )
    slot_mapping = cache_block * block_size + torch.arange(
        kv_len, device=device, dtype=torch.long
    )
    scale = torch.ones((), device=device, dtype=torch.float32)
    cu_seqlens_q = torch.tensor(
        [0, query_len], device=device, dtype=torch.int32
    )
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.full(
        (1, 1), cache_block, device=device, dtype=torch.int32
    )

    def run() -> None:
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
        impl._run_flash_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=1.0 / math.sqrt(head_size),
            causal=True,
            block_table=block_table,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()

    expected_first = _reference_attention(query, key, value)
    graph.replay()
    torch.cuda.synchronize()
    first_output = output.clone()
    torch.testing.assert_close(
        first_output, expected_first, atol=1.6e-2, rtol=1.6e-2
    )

    query.copy_(torch.randn_like(query))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    expected_second = _reference_attention(query, key, value)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output, expected_second, atol=1.6e-2, rtol=1.6e-2
    )
    assert not torch.equal(output, first_output)


@pytest.mark.parametrize(
    (
        "layout_name",
        "query_len",
        "kv_len",
        "num_query_heads",
        "num_kv_heads",
    ),
    tuple(
        (layout, query_len, kv_len, num_query_heads, num_kv_heads)
        for layout in ("LBHNC", "LBNHC")
        for query_len, kv_len, num_query_heads, num_kv_heads in (
            (3, 3, 2, 2),
            (1, 5, 4, 2),
        )
    ),
)
@torch.inference_mode()
def test_hcu_e5m2_writer_to_vendor_varlen_matches_quantized_reference(
    layout_name,
    query_len,
    kv_len,
    num_query_heads,
    num_kv_heads,
):
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm import _custom_ops as ops

    torch.manual_seed(71 + query_len)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size, num_blocks, cache_block, head_size = 64, 2, 1, 64
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
    packed_cache = _make_packed_cache(
        layout_name=layout_name,
        num_blocks=num_blocks,
        num_heads=num_kv_heads,
        block_size=block_size,
        head_size=head_size,
        device=device,
        dtype=torch.uint8,
        block_padding=64,
    )
    impl = _make_impl(layout_name, head_size)
    impl.kv_cache_dtype = "fp8_e5m2"
    key_cache, value_cache = impl._get_kv_cache_views(packed_cache)
    slot_mapping = cache_block * block_size + torch.arange(
        kv_len, device=device, dtype=torch.long
    )
    scale = torch.ones((), device=device, dtype=torch.float32)
    impl.do_kv_cache_update(
        SimpleNamespace(_k_scale=scale, _v_scale=scale),
        key,
        value,
        packed_cache,
        slot_mapping,
    )
    if layout_name == "LBHNC":
        cached_key = key_cache[cache_block, :, :kv_len].transpose(0, 1)
        cached_value = value_cache[cache_block, :, :kv_len].transpose(0, 1)
    else:
        cached_key = key_cache[cache_block, :kv_len]
        cached_value = value_cache[cache_block, :kv_len]
    quantized_key = cached_key.view(torch.float8_e5m2).to(dtype)
    quantized_value = cached_value.view(torch.float8_e5m2).to(dtype)
    torch.testing.assert_close(quantized_key, _e5m2_round_trip(key), rtol=0, atol=0)
    torch.testing.assert_close(
        quantized_value,
        _e5m2_round_trip(value),
        rtol=0,
        atol=0,
    )

    descale = scale.expand(1, num_kv_heads)
    output = impl._run_flash_attention(
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
        block_table=torch.full(
            (1, 1), cache_block, device=device, dtype=torch.int32
        ),
        k_descale=descale,
        v_descale=descale,
        fa_version=2,
    )
    expected = _reference_attention(query, quantized_key, quantized_value)

    torch.testing.assert_close(output, expected, atol=3.0e-2, rtol=3.0e-2)


@pytest.mark.parametrize("layout_name", ("LBHNC", "LBNHC"))
@torch.inference_mode()
def test_hcu_e5m2_writer_supports_per_head_scales(layout_name):
    torch.manual_seed(83)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, num_heads, head_size, block_size = 3, 2, 64, 64
    key = torch.randn(
        num_tokens, num_heads, head_size, device=device, dtype=dtype
    )
    value = torch.randn_like(key)
    packed_cache = _make_packed_cache(
        layout_name=layout_name,
        num_blocks=1,
        num_heads=num_heads,
        block_size=block_size,
        head_size=head_size,
        device=device,
        dtype=torch.uint8,
        block_padding=64,
    )
    impl = _make_impl(layout_name, head_size)
    impl.kv_cache_dtype = "fp8_e5m2"
    k_scale = torch.tensor([0.5, 1.25], device=device, dtype=torch.float32)
    v_scale = torch.tensor([1.5, 0.75], device=device, dtype=torch.float32)
    impl.do_kv_cache_update(
        SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale),
        key,
        value,
        packed_cache,
        torch.arange(num_tokens, device=device, dtype=torch.long),
    )

    key_cache, value_cache = impl._get_kv_cache_views(packed_cache)
    if layout_name == "LBHNC":
        cached_key = key_cache[0, :, :num_tokens].transpose(0, 1)
        cached_value = value_cache[0, :, :num_tokens].transpose(0, 1)
    else:
        cached_key = key_cache[0, :num_tokens]
        cached_value = value_cache[0, :num_tokens]
    expected_key = (key / k_scale[None, :, None]).to(torch.float8_e5m2)
    expected_value = (value / v_scale[None, :, None]).to(torch.float8_e5m2)
    torch.testing.assert_close(
        cached_key,
        expected_key.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        cached_value,
        expected_value.view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("layout_name", ("LBHNC", "LBNHC"))
@torch.inference_mode()
def test_hcu_e5m2_writer_and_vendor_replay_changed_graph_inputs(layout_name):
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(89)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, kv_len = 1, 5
    block_size, num_query_heads, num_kv_heads, head_size = 64, 4, 2, 64
    query = torch.randn(
        query_len, num_query_heads, head_size, device=device, dtype=dtype
    )
    key = torch.randn(
        kv_len, num_kv_heads, head_size, device=device, dtype=dtype
    )
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    cache_block = 1
    packed_cache = _make_packed_cache(
        layout_name=layout_name,
        num_blocks=2,
        num_heads=num_kv_heads,
        block_size=block_size,
        head_size=head_size,
        device=device,
        dtype=torch.uint8,
        block_padding=64,
    )
    impl = _make_impl(layout_name, head_size)
    impl.kv_cache_dtype = "fp8_e5m2"
    key_cache, value_cache = impl._get_kv_cache_views(packed_cache)
    slot_mapping = cache_block * block_size + torch.arange(
        kv_len, device=device, dtype=torch.long
    )
    scale = torch.ones((), device=device, dtype=torch.float32)
    descale = scale.expand(1, num_kv_heads)
    cu_seqlens_q = torch.tensor(
        [0, query_len], device=device, dtype=torch.int32
    )
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.full(
        (1, 1), cache_block, device=device, dtype=torch.int32
    )
    layer = SimpleNamespace(_k_scale=scale, _v_scale=scale)

    def run() -> None:
        impl.do_kv_cache_update(
            layer,
            key,
            value,
            packed_cache,
            slot_mapping,
        )
        impl._run_flash_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=1.0 / math.sqrt(head_size),
            causal=True,
            block_table=block_table,
            k_descale=descale,
            v_descale=descale,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()

    expected_first = _reference_attention(
        query,
        _e5m2_round_trip(key),
        _e5m2_round_trip(value),
    )
    graph.replay()
    torch.cuda.synchronize()
    first_output = output.clone()
    torch.testing.assert_close(
        first_output,
        expected_first,
        atol=3.0e-2,
        rtol=3.0e-2,
    )

    query.copy_(torch.randn_like(query))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    expected_second = _reference_attention(
        query,
        _e5m2_round_trip(key),
        _e5m2_round_trip(value),
    )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output,
        expected_second,
        atol=3.0e-2,
        rtol=3.0e-2,
    )
    assert not torch.equal(output, first_output)
