# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Query subchunks must share initialized K even on PCP ranks with no early Q."""

from types import SimpleNamespace

import pytest
import torch

from tests.runtime_patch.test_glm52_pcp_manager import _make_batch, _make_managers
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_pcp as patch
from vllm_hcu.v1.glm53_pcp import (
    _current_pcp_batch, bind_glm53_pcp_batch, glm53_pcp_step_scope,
)


@pytest.mark.parametrize("pcp_size,rank", [
    (size, rank) for size in (2, 4, 8) for rank in range(size)
])
@pytest.mark.parametrize("local_queries", [False, True])
@pytest.mark.parametrize("gfx938", [False, True])
def test_query_subchunks_gather_k_before_skipping_empty_pcp_rows(
    monkeypatch, pcp_size, rank, local_queries, gfx938,
):
    from vllm import _custom_ops as ops
    import vllm.utils.torch_utils as torch_utils
    from vllm.v1.attention.backends.mla import indexer
    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as sparse

    # Use the native splitter and metadata builder. Only its GPU metadata
    # kernel is replaced by a CPU implementation of single-request bounds.
    def metadata_kernel(query_start, seq_lens, cu_seq, local_cu_seq,
                        token_to_seq, starts, ends, q_start, q_end,
                        *args, **kwargs):
        token_to_seq.zero_()
        starts.zero_()
        context = seq_lens[0] - (query_start[-1] - query_start[0])
        ends.copy_((context + torch.arange(q_start, q_end) + 1)
                   // kwargs["COMPRESS_RATIO"])

    monkeypatch.setattr(indexer, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", metadata_kernel)
    seq_lens = torch.tensor([32, 32], dtype=torch.int32)
    compressed = seq_lens // 4
    query_lens = torch.tensor([16, 16], dtype=torch.int32)
    query_start = torch.tensor([0, 16, 32], dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    splits = indexer.split_indexer_prefill_chunks(
        compressed, query_lens, workspace_size=8, max_logits_bytes=8 * 4,
    )
    chunks = [indexer.build_prefill_chunk_metadata(
        req.start, req.stop, query_start, query_start, seq_lens,
        compressed, compressed, block_table, compress_ratio=4,
        query_slice=query,
    ) for req, query in splits]
    assert [(chunk.token_start, chunk.skip_kv_gather) for chunk in chunks] == [
        (start, start % 16 != 0) for start in range(32)
    ]
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(32), num_kv_actual_tokens=32,
        num_decodes=0, num_decode_tokens=0, num_prefills=2, num_prefill_tokens=32,
        prefill=SimpleNamespace(chunks=chunks),
    )
    monkeypatch.setattr(sparse, "DeepseekV32IndexerMetadata", SimpleNamespace)
    monkeypatch.setattr(sparse, "get_forward_context", lambda: SimpleNamespace(
        attn_metadata={"indexer": metadata},
    ))
    monkeypatch.setattr(sparse, "current_platform", SimpleNamespace(
        is_rocm=lambda: True, fp8_dtype=lambda: torch.float32,
    ))
    monkeypatch.setattr(sparse, "on_gfx938", lambda: gfx938)
    monkeypatch.setattr(sparse, "_indexer_cache_as_hipc_view", lambda cache: cache)
    monkeypatch.setattr(torch_utils, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(sparse, "_use_lightop_sparse_mla_topk", lambda: False)

    # Distinct request-level K workspaces expose reuse of stale values from
    # the preceding request, as well as an uninitialized first workspace.
    keys = torch.tensor([[1, 7, 2, 0, 0, 0, 0, 0],
                         [1, 2, 9, 0, 0, 0, 0, 0]], dtype=torch.float32)
    gathered = []

    def gather_k(cache, output, table, cu_seq):
        request = int(table[0, 0])
        output.copy_(keys[request, :, None])
        gathered.append(request)

    def gather_quant(cache, output, scales, table, cu_seq):
        gather_k(cache, output, table, cu_seq)
        scales.view(torch.float32).fill_(1)

    monkeypatch.setattr(sparse, "cp_gather_indexer_k_bf16_cache_triton", gather_k)
    monkeypatch.setattr(ops, "cp_gather_indexer_k_quant_cache", gather_quant)
    monkeypatch.setattr(sparse, "rocm_fp8_mqa_logits",
                        lambda q, kv, weights, starts, ends:
                        sparse.fp8_mqa_logits_torch(
                            q, (kv[0], kv[1] if gfx938 else None),
                            weights, starts, ends))

    original_empty = torch.empty

    def sentinel_empty(*args, **kwargs):
        output = original_empty(*args, **kwargs)
        if output.shape == (8, 1) and output.dtype == torch.float32:
            output.zero_()
            output[0] = 1000  # invalid K deterministically selects key 0
        elif output.shape == (8, 4) and output.dtype == torch.uint8:
            output.view(torch.float32).fill_(1)
        return output

    monkeypatch.setattr(torch, "empty", sentinel_empty)
    batch = _make_batch([
        ("first", list(range(1, 17)), 32, True),
        ("second", list(range(17, 33)), 32, True),
    ])
    managers, _ = _make_managers(pcp_size)
    manager = managers[rank]
    local_batch = manager.partition_batch(batch)
    with glm53_pcp_step_scope():
        bind_glm53_pcp_batch(manager, local_batch)
        pcp_batch = _current_pcp_batch()
    owned, _ = pcp_batch.local_query_rows(0, batch.num_tokens)
    if rank > 0:
        assert pcp_batch.local_query_rows(0, 1)[0].numel() == 0
        assert owned.numel() > 0  # later skip_kv_gather=True chunks have Q

    num_queries = local_batch.num_tokens_after_padding if local_queries else 32
    token = patch._ATTENTION_BATCH.set(pcp_batch)
    local = patch._LOCAL_QUERIES.set(pcp_batch if local_queries else None)
    try:
        actual = sparse.rocm_aiter_sparse_attn_indexer_native(
            hidden_states=torch.zeros(32, 1), k_cache_prefix="indexer",
            kv_cache=torch.zeros(2, 8, 1),
            q_fp8=torch.ones(num_queries, 1, 1), k=None,
            weights=torch.ones(num_queries, 1), quant_block_size=1,
            scale_fmt="e4m3", topk_tokens=1, head_dim=1,
            max_model_len=32, total_seq_lens=16,
            topk_indices_buffer=torch.empty(32, 1, dtype=torch.int32),
            skip_k_cache_insert=True,
        )
    finally:
        patch._LOCAL_QUERIES.reset(local)
        patch._ATTENTION_BATCH.reset(token)

    expected = torch.tensor([1] * 16 + [2] * 16, dtype=torch.int32)[:, None]
    torch.testing.assert_close(actual[owned], expected[owned])
    assert gathered == [0, 1]  # once per request, including empty first Q chunks
    unowned = torch.ones(32, dtype=torch.bool)
    unowned[owned] = False
    assert (actual[unowned] == -1).all()
