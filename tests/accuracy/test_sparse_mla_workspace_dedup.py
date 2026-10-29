# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import pytest
import torch

from vllm_hcu.v1.attention.backends.mla.prefill_workspace import plan_prefill_workspace

from vllm_hcu.v1.attention.backends.mla.workspace_dedup import gather_workspace_pages

pytestmark = pytest.mark.hcu

HAS_GPU = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.mark.skipif(
    not (HAS_GPU or INTERPRET), reason="requires GPU or Triton interpreter"
)
@pytest.mark.parametrize("compact", [False, True])
def test_dcp_remapping_unchanged(compact):
    from vllm_hcu.v1.attention.backends.mla.workspace_indices import (
        triton_filter_and_convert_dcp_index,
    )

    device = "cuda" if HAS_GPU else "cpu"
    result, counts = triton_filter_and_convert_dcp_index(
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.tensor([[5, 3]], device=device, dtype=torch.int32),
        torch.tensor([[-1, 0, 1, 2, 3, 4, 5, 999]], device=device, dtype=torch.int32),
        dcp_size=2,
        dcp_rank=1,
        BLOCK_SIZE=4,
        NUM_TOPK_TOKENS=8,
        BLOCK_N=4,
        return_valid_counts=True,
        compact_valid_to_front=compact,
    )
    expected = (
        [20, 21, 22, -1, -1, -1, -1, -1]
        if compact
        else [-1, -1, 20, -1, 21, -1, 22, -1]
    )
    if compact:
        assert sorted(result[0, :3].tolist()) == [20, 21, 22]
        assert result[0, 3:].tolist() == [-1] * 5
    else:
        assert result[0].tolist() == expected
    assert counts.tolist() == [3]


@pytest.mark.skipif(
    not (HAS_GPU or INTERPRET), reason="requires GPU or Triton interpreter"
)
def test_legacy_prefill_remapping_unchanged():
    from vllm_hcu.v1.attention.backends.mla.workspace_indices import (
        triton_convert_req_index_to_global_index,
    )

    device = "cuda" if HAS_GPU else "cpu"
    req = torch.tensor([0], device=device, dtype=torch.int32)
    result = triton_convert_req_index_to_global_index(
        req,
        torch.tensor([[5, 1]], device=device, dtype=torch.int32),
        torch.tensor([[0, 64, -1, 127]], device=device, dtype=torch.int32),
        NUM_TOPK_TOKENS=4,
        BLOCK_N=4,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=req,
        prefill_workspace_starts=torch.tensor([10], device=device, dtype=torch.int32),
    )
    assert result.tolist() == [[10, 74, -1, 137]]


@pytest.mark.skipif(
    not (HAS_GPU or INTERPRET), reason="requires GPU or Triton interpreter"
)
def test_dedup_index_remapping_preserves_decode_padding_and_multiplicity():
    from vllm_hcu.v1.attention.backends.mla.workspace_indices import (
        triton_convert_req_index_to_global_index,
    )

    device = "cuda" if HAS_GPU else "cpu"
    table_cpu = np.array([[6, 0, 0], [5, 1, 7], [5, 1, 8]], dtype=np.int32)
    plan = plan_prefill_workspace(table_cpu[1:], [130, 150], 64, 256)[0]
    table = torch.tensor(table_cpu, device=device)
    indices = torch.tensor(
        [
            [0, 63, -1, 999, 0, 1, 2, 3],
            [0, 64, 129, -1, 1, 1, 192, 999],
            [0, 64, 149, -1, 1, 1, 192, 999],
        ],
        dtype=torch.int32,
        device=device,
    )
    result, counts = triton_convert_req_index_to_global_index(
        torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        table,
        indices,
        BLOCK_SIZE=64,
        NUM_TOPK_TOKENS=8,
        BLOCK_N=4,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=torch.tensor(
            [-1, 0, 1], dtype=torch.int32, device=device
        ),
        prefill_workspace_starts=torch.zeros(2, dtype=torch.int32, device=device),
        return_valid_counts=True,
        prefill_workspace_block_starts=torch.tensor(
            plan.row_block_starts, device=device
        ),
    )
    expected = torch.tensor(
        [
            [384, 447, -1, -1, 384, 385, 386, 387],
            [0, 64, 129, -1, 1, 1, -1, -1],
            [0, 64, 151, -1, 1, 1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        counts, torch.tensor([6, 5, 5], device=device, dtype=torch.int32)
    )


@pytest.mark.skipif(not HAS_GPU, reason="requires compiled CUDA/HIP cache operations")
@pytest.mark.parametrize("capacity", [192, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_unique_page_gather_matches_existing_gather(capacity, strided):
    from vllm import _custom_ops as ops

    torch.manual_seed(42)
    block_size = 64
    cache = torch.empty(
        (9, block_size, 672 if strided else 656), dtype=torch.uint8, device="cuda"
    )[..., :656]
    fp8 = torch.randn((9, block_size, 512), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((9, block_size, 4), device="cuda") + 0.01
    rope = torch.randn((9, block_size, 64), device="cuda", dtype=torch.bfloat16)
    cache[..., :512] = fp8.view(torch.uint8)
    cache[..., 512:528] = scales.view(torch.uint8)
    cache[..., 528:] = rope.view(torch.uint8)
    table_cpu = np.array([[5, 1, 7], [5, 1, 8], [5, 1, 7]], dtype=np.int32)
    lengths = [130, 150, 190]
    starts = np.array([0, 130, 280], dtype=np.int32)
    reference = torch.empty((sum(lengths), 576), dtype=torch.bfloat16, device="cuda")
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        reference,
        torch.tensor(table_cpu, device="cuda"),
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        torch.tensor(starts, device="cuda"),
        len(lengths),
    )
    dequantized = fp8.float().view(9, block_size, 4, 128) * scales[..., None]
    expected_cache = torch.cat(
        (dequantized.flatten(-2).to(torch.bfloat16), rope), dim=-1
    )
    for row, length in enumerate(lengths):
        tokens = np.arange(length)
        physical = torch.tensor(table_cpu[row, tokens // block_size], device="cuda")
        offsets = torch.tensor(tokens % block_size, device="cuda")
        torch.testing.assert_close(
            reference[starts[row] : starts[row] + length],
            expected_cache[physical, offsets],
            rtol=0,
            atol=0,
        )
    for plan in plan_prefill_workspace(table_cpu, lengths, block_size, capacity):
        descriptors = torch.tensor(
            np.column_stack(
                (plan.physical_blocks, plan.valid_tokens, plan.workspace_starts)
            ),
            device="cuda",
        )
        backing = torch.full(
            (plan.num_tokens + 1, 584), -123, dtype=torch.bfloat16, device="cuda"
        )
        workspace = backing[: plan.num_tokens, :576]
        gather_workspace_pages(cache, workspace, descriptors)
        for local_row, row in enumerate(range(plan.req_start, plan.req_end)):
            tokens = np.arange(lengths[row])
            indices = (
                plan.row_block_starts[local_row, tokens // block_size]
                + tokens % block_size
            )
            torch.testing.assert_close(
                workspace[torch.tensor(indices, device="cuda")],
                reference[starts[row] : starts[row] + lengths[row]],
                rtol=0,
                atol=0,
            )
        assert (backing[:, 576:] == -123).all()
        assert (backing[-1] == -123).all()


@pytest.mark.skipif(not HAS_GPU, reason="requires compiled CUDA/HIP cache operations")
def test_page_gather_graph_replay_reads_current_descriptors():
    cache = torch.zeros((3, 64, 656), dtype=torch.uint8, device="cuda")
    scales = torch.ones((3, 64, 4), device="cuda")
    cache[..., 512:528] = scales.view(torch.uint8)
    cache[1, :, :512] = (
        torch.ones((64, 512), device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    )
    cache[2, :, :512] = (
        torch.full((64, 512), 2.0, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    descriptors = torch.tensor([[1, 63, 0]], dtype=torch.int32, device="cuda")
    dst = torch.empty((63, 576), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gather_workspace_pages(cache, dst, descriptors)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_workspace_pages(cache, dst, descriptors)
    graph.replay()
    assert (dst[:, :512] == 1).all()
    descriptors[0, 0] = 2
    graph.replay()
    assert (dst[:, :512] == 2).all()
    assert (dst[:, 512:] == 0).all()


@pytest.mark.skipif(
    not HAS_GPU, reason="requires compiled sparse FlashMLA and cache ops"
)
def test_sparse_attention_with_shared_workspace_rows():
    from vllm import _custom_ops as ops
    from vllm_hcu.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    torch.manual_seed(1)
    cache = torch.empty((5, 64, 656), dtype=torch.uint8, device="cuda")
    values = torch.randn((5, 64, 512), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((5, 64, 4), device="cuda") + 0.01
    rope = torch.randn((5, 64, 64), device="cuda", dtype=torch.bfloat16)
    cache[..., :512] = values.view(torch.uint8)
    cache[..., 512:528] = scales.view(torch.uint8)
    cache[..., 528:] = rope.view(torch.uint8)
    table_cpu = np.array([[1, 2, 3], [1, 2, 4]], dtype=np.int32)
    plan = plan_prefill_workspace(table_cpu, [130, 150], 64, 256)[0]
    legacy = torch.empty((280, 576), device="cuda", dtype=torch.bfloat16)
    compact = torch.empty((plan.num_tokens, 576), device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0, 130], dtype=torch.int32, device="cuda")
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        legacy,
        torch.tensor(table_cpu, device="cuda"),
        torch.tensor([130, 150], dtype=torch.int32, device="cuda"),
        starts,
        2,
    )
    descriptors = torch.tensor(
        np.column_stack(
            (plan.physical_blocks, plan.valid_tokens, plan.workspace_starts)
        ),
        device="cuda",
    )
    gather_workspace_pages(cache, compact, descriptors)
    logical = np.stack([np.arange(128), np.arange(22, 150)])
    # Exercise repeated selections and invalid padding without changing them.
    logical[:, 5] = logical[:, 4]
    compact_indices = (
        plan.row_block_starts[np.arange(2)[:, None], logical // 64] + logical % 64
    )
    legacy_indices = logical + np.array([0, 130])[:, None]
    compact_indices[:, -1] = legacy_indices[:, -1] = -1
    q = torch.randn((2, 128, 576), dtype=torch.bfloat16, device="cuda")

    def attend(kv, indices):
        return flash_mla_sparse_fwd(
            q,
            kv.view(-1, 1, 576),
            torch.tensor(indices, dtype=torch.int32, device="cuda").view(2, 1, 128),
            576**-0.5,
        )[0]

    torch.testing.assert_close(
        attend(compact, compact_indices),
        attend(legacy, legacy_indices),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not HAS_GPU, reason="requires compiled HCU gather and FlashMLA")
@pytest.mark.parametrize("decode", [False, True])
def test_hcu_backend_build_and_forward_with_unmodified_vllm(monkeypatch, decode):
    from types import SimpleNamespace

    from vllm.v1.attention import backend as backend_module
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla import flashmla_sparse as core
    from vllm_hcu.patch.worker.op_opt import (
        patch_attention_backend,
        patch_flashmla_sparse,
    )
    from vllm_hcu.v1.workspace_block_tables import HOST_TABLE_ATTRIBUTE
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseImpl,
        HcuFlashMLASparseMetadataBuilder,
    )

    patch_attention_backend.apply_to_module(backend_module)
    patch_flashmla_sparse.apply_to_module(core)
    monkeypatch.setenv("VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP", "1")
    table = np.array([[5, 1, 7], [5, 1, 8], [5, 1, 7]], dtype=np.int32)
    lengths = [130, 150, 190]
    widths = [2, 2, 2]
    if decode:
        table = np.vstack(([6, 2, 0], table)).astype(np.int32)
        lengths.insert(0, 70)
        widths.insert(0, 1)
    lengths = torch.tensor(lengths, dtype=torch.int32)
    query_starts = torch.tensor(np.cumsum([0] + widths), dtype=torch.int32)
    num_reqs = len(lengths)
    num_tokens = sum(widths)
    device_table = torch.tensor(table, device="cuda")
    setattr(device_table, HOST_TABLE_ATTRIBUTE, table)
    common = CommonAttentionMetadata(
        query_start_loc=query_starts.cuda(),
        query_start_loc_cpu=query_starts,
        seq_lens=lengths.cuda(),
        seq_lens_cpu_upper_bound=lengths,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=2,
        max_seq_len=190,
        block_table_tensor=device_table,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
        is_prefilling=torch.tensor(([False] if decode else []) + [True] * 3),
    )
    assert common.sparse_mla_workspace_block_table_cpu is table
    builder = HcuFlashMLASparseMetadataBuilder.__new__(HcuFlashMLASparseMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=190),
        attention_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=16,
            cp_kv_cache_interleave_size=1,
            decode_context_parallel_size=1,
        ),
    )
    builder.device = torch.device("cuda")
    builder.kv_cache_spec = SimpleNamespace(block_size=64)
    builder.reorder_batch_threshold = 1
    builder.num_heads = 128
    builder.use_fp8_kv_cache = True
    builder.topk_tokens = 2048
    builder.req_id_per_token_buffer = torch.zeros(
        num_tokens, dtype=torch.int32, device="cuda"
    )
    builder.dummy_block_table = torch.zeros(
        (num_reqs, 1), dtype=torch.int32, device="cuda"
    )
    builder.max_model_len_tensor = torch.full(
        (num_reqs,), 190, dtype=torch.int32, device="cuda"
    )
    metadata = builder.build(0, common)
    assert metadata.hcu_workspace_dedup.chunks[0].num_tokens == 212
    impl = HcuFlashMLASparseImpl.__new__(HcuFlashMLASparseImpl)
    impl.num_heads = 128
    impl.kv_lora_rank = 512
    impl.softmax_scale = 576**-0.5
    impl.prefill_bf16_workspace = torch.empty(
        (950, 576), dtype=torch.bfloat16, device="cuda"
    )
    torch.manual_seed(42)
    cache = torch.empty((9, 64, 656), dtype=torch.uint8, device="cuda")
    cache[..., :512] = (
        torch.randn((9, 64, 512), device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    cache[..., 512:528] = (torch.rand((9, 64, 4), device="cuda") + 0.01).view(
        torch.uint8
    )
    cache[..., 528:] = torch.randn(
        (9, 64, 64), dtype=torch.bfloat16, device="cuda"
    ).view(torch.uint8)
    q = torch.randn((num_tokens, 128, 576), dtype=torch.bfloat16, device="cuda")
    indices = (torch.arange(2048, dtype=torch.int32, device="cuda") % 128).repeat(
        num_tokens, 1
    )
    if decode:
        indices[0] %= 64
    indices[:, 5] = indices[:, 4]
    indices[:, -1] = -1
    expected = core.FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode(
        impl, q, cache, indices, metadata
    ).clone()
    actual = impl._forward_fp8_kv_separate_prefill_decode(q, cache, indices, metadata)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
