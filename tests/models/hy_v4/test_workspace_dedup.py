# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU integration checks; native gather and attention use Torch references."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.mla import flashmla_sparse as core
from vllm_hcu.models.hy_v4 import attention, hcu_sparse
from vllm_hcu.v1.attention.backends.mla import workspace_dedup, workspace_indices
from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import HcuFlashMLASparseBackend


@pytest.fixture
def sink_backend(monkeypatch):
    monkeypatch.setattr(
        attention, "get_attn_backend", lambda **kw: HcuFlashMLASparseBackend
    )
    monkeypatch.setattr(
        attention,
        "current_platform",
        SimpleNamespace(get_device_capability=lambda: (9, 0)),
    )
    monkeypatch.setattr(
        attention,
        "get_current_vllm_config",
        lambda: SimpleNamespace(cache_config=None),
    )
    # Backend selection runs on CPU; hardware validation is tested separately.
    monkeypatch.setattr(
        hcu_sparse.HYV4FlashMLASparseBackend, "validate_configuration", lambda **kw: []
    )
    model = SimpleNamespace(
        kv_lora_rank=512, qk_rope_head_dim=64, is_sparse=True, num_local_heads=4
    )
    backend = attention.HYV4MLAAttention._resolve_sink_backend(model, "fp8_ds_mla")
    assert backend is hcu_sparse.HYV4FlashMLASparseBackend
    return backend


def _inputs(decode, factor):
    table = [[2, 0], [2, 1]]
    lengths, widths = [3, 4], [2, 1]
    if decode:
        table.insert(0, [4, 3])
        lengths.insert(0, 3)
        widths.insert(0, 2)
    table = torch.tensor(table, dtype=torch.int32)
    starts = torch.tensor(np.cumsum([0] + widths), dtype=torch.int32)
    common = SimpleNamespace(
        block_table_tensor=table,
        sparse_mla_workspace_block_table_cpu=table.numpy(),
        seq_lens_cpu_upper_bound=torch.tensor(lengths, dtype=torch.int32),
        query_start_loc_cpu=starts,
        is_prefilling=torch.tensor(([False] if decode else []) + [True, True]),
    )
    extra_cls = core.FlashMLASparseMetadata.FP8SeparatePrefillDecode
    chunks = []
    for i, length in enumerate([3, 4]):
        row = i + int(decode)
        chunks.append(
            extra_cls.Prefill.Chunk(
                seq_lens=torch.tensor([length], dtype=torch.int32),
                tokens_slice=slice(int(starts[row]), int(starts[row + 1])),
                block_table=table[row : row + 1],
                req_start_idx=i,
                workspace_starts=torch.zeros(1, dtype=torch.int32),
                chunk_tot_seqlen=length,
            )
        )
    tokens = int(starts[-1])
    extra = extra_cls(
        num_decodes=int(decode),
        num_decode_tokens=2 * int(decode),
        num_prefills=2,
        num_prefill_tokens=3,
        decode=SimpleNamespace(kernel_metadata=None) if decode else None,
        prefill=extra_cls.Prefill(
            seq_lens=torch.tensor([3, 4], dtype=torch.int32),
            request_ids=torch.tensor(
                [-1] * (2 * int(decode)) + [0, 0, 1], dtype=torch.int32
            ),
            workspace_starts=torch.zeros(2, dtype=torch.int32),
            chunks=chunks,
        ),
    )
    metadata = SimpleNamespace(
        fp8_extra_metadata=extra,
        fp8_use_mixed_batch=False,
        num_actual_tokens=tokens,
        num_reqs=len(table),
        max_query_len=2,
        req_id_per_token=torch.repeat_interleave(
            torch.arange(len(table), dtype=torch.int32), torch.tensor(widths)
        ),
        block_table=table,
        block_size=2,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4),
        attention_config=SimpleNamespace(sparse_mla_prefill_workspace_factor=factor),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    return common, metadata, config


@pytest.mark.parametrize("enabled,dcp", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("prefilling", [False, True])
def test_selected_hyv4_builder_keeps_dedup_and_lightop_metadata(
    monkeypatch, sink_backend, enabled, dcp, prefilling
):
    common, metadata, config = _inputs(False, 2)
    config.parallel_config.decode_context_parallel_size = dcp
    common.is_prefilling.fill_(prefilling)
    metadata.num_actual_tokens, metadata.max_query_len = 8, 4
    if not prefilling:
        metadata.fp8_extra_metadata.prefill = None
    monkeypatch.setattr(
        core.FlashMLASparseMetadataBuilder, "build", lambda *a: metadata
    )
    monkeypatch.setenv("VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP", str(int(enabled)))
    cls = sink_backend.get_builder_cls()
    builder = cls.__new__(cls)
    builder.vllm_config = config
    builder.device = torch.device("cpu")
    builder.kv_cache_spec = SimpleNamespace(block_size=2)
    result = builder.build(0, common)
    assert result.lightop_kv_group_size == (1 if prefilling else 4)
    assert hasattr(result, "hcu_workspace_dedup") == (enabled and dcp == 1)
    assert (getattr(result, "hcu_workspace_dedup", None) is not None) == (
        enabled and dcp == 1 and prefilling
    )


def _map_indices(
    req_ids,
    table,
    indices,
    *,
    BLOCK_SIZE,
    HAS_PREFILL_WORKSPACE=False,
    prefill_workspace_request_ids=None,
    prefill_workspace_starts=None,
    prefill_workspace_block_starts=None,
    return_valid_counts=False,
    **kw,
):
    mapped = torch.full_like(indices, -1)
    for row, selected in enumerate(indices):
        for col, token in enumerate(selected.tolist()):
            if token < 0:
                continue
            block, offset = divmod(token, BLOCK_SIZE)
            prefill = (
                int(prefill_workspace_request_ids[row]) if HAS_PREFILL_WORKSPACE else -1
            )
            if prefill < 0:
                mapped[row, col] = table[req_ids[row], block] * BLOCK_SIZE + offset
            elif prefill_workspace_block_starts is None:
                mapped[row, col] = prefill_workspace_starts[prefill] + token
            else:
                mapped[row, col] = (
                    prefill_workspace_block_starts[prefill, block] + offset
                )
    counts = (mapped >= 0).sum(-1).to(torch.int32)
    return (mapped, counts) if return_valid_counts else mapped


def _attention_reference(q, kv, indices, scale, attn_sink=None, topk_length=None):
    indices = indices.view(q.shape[0], -1)
    selected = kv.view(-1, kv.shape[-1])[indices.clamp_min(0)]
    scores = torch.einsum("thd,tkd->thk", q, selected) * scale
    scores.masked_fill_((indices < 0).unsqueeze(1), float("-inf"))
    lse = scores.logsumexp(-1)
    if attn_sink is not None:
        scores = torch.cat(
            (scores, attn_sink.view(1, -1, 1).expand(q.shape[0], -1, -1)), -1
        )
    weights = scores.softmax(-1)[..., : indices.shape[1]]
    output = torch.einsum("thk,tkd->thd", weights, selected[..., :512])
    return output, torch.empty(0), lse


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("factor", [1, 2])
def test_selected_hyv4_dedup_forward_preserves_sinks_and_decode(
    monkeypatch, sink_backend, decode, factor
):
    monkeypatch.setattr(core, "triton_convert_req_index_to_global_index", _map_indices)
    monkeypatch.setattr(
        workspace_indices, "triton_convert_req_index_to_global_index", _map_indices
    )
    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", _attention_reference)
    monkeypatch.setenv("VLLM_HCU_HYV4_FP8_KV_DEQUANT", "1")
    gathers, decode_widths = [], []

    def legacy_gather(cache, workspace, table, lengths, starts, num_reqs):
        gathers.append("legacy")
        for row in range(num_reqs):
            for token in range(int(lengths[row])):
                block, offset = divmod(token, 2)
                workspace[int(starts[row]) + token] = cache[table[row, block], offset]

    def page_gather(cache, workspace, descriptors):
        gathers.append("dedup")
        for block, length, start in descriptors.tolist():
            workspace[start : start + length] = cache[block, :length]

    def decode_gather(cache, indices, rank, rope, tokens_per_request, **kw):
        decode_widths.append(tokens_per_request)
        return cache.view(-1, 576), indices

    monkeypatch.setattr(core.ops, "cp_gather_and_upconvert_fp8_kv_cache", legacy_gather)
    monkeypatch.setattr(workspace_dedup, "gather_workspace_pages", page_gather)
    monkeypatch.setattr(hcu_sparse, "gather_dequantize_fp8_ds_mla_cache", decode_gather)
    generator = torch.Generator().manual_seed(42)
    cache = torch.randn(5, 2, 576, generator=generator)
    indices = torch.tensor(
        [[0, 2, 2, -1], [1, 0, -1, -1], [3, 2, 0, -1]], dtype=torch.int32
    )
    if decode:
        indices = torch.cat(
            (torch.tensor([[0, 1, 0, -1], [2, -1, -1, -1]], dtype=torch.int32), indices)
        )
    q = torch.randn(len(indices), 4, 576, generator=generator)
    cls = sink_backend.get_impl_cls()
    impl = cls.__new__(cls)
    impl.num_heads, impl.head_size, impl.kv_lora_rank = 4, 576, 512
    impl.dcp_world_size, impl.dcp_rank = 1, 0
    impl.prefill_padding, impl.fp8_decode_padded_heads = 64, 64
    impl.kv_cache_dtype, impl.softmax_scale = "fp8_ds_mla", 576**-0.5
    impl.sinks = torch.tensor([0.0, 1.1, float("-inf"), -0.7])
    impl._dcp_sinks = None
    impl.topk_indices_buffer = indices
    impl.prefill_bf16_workspace = torch.empty(factor * 4, 576)
    outputs = []
    for enabled in [False, True]:
        common, metadata, config = _inputs(decode, factor)
        monkeypatch.setenv(
            "VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP", str(int(enabled))
        )
        monkeypatch.setattr(
            core.FlashMLASparseMetadataBuilder, "build", lambda *a: metadata
        )
        builder_cls = sink_backend.get_builder_cls()
        builder = builder_cls.__new__(builder_cls)
        builder.vllm_config, builder.device = config, torch.device("cpu")
        builder.kv_cache_spec = SimpleNamespace(block_size=2)
        metadata = builder.build(0, common)
        if enabled:
            plan = metadata.hcu_workspace_dedup
            assert len(plan.chunks) == (2 if factor == 1 else 1)
            assert sum(chunk.num_tokens for chunk in plan.chunks) == (
                7 if factor == 1 else 5
            )
        output, lse = impl.forward_mqa(q, cache, metadata, object())
        assert lse is None
        outputs.append(output)
    physical = _map_indices(
        metadata.req_id_per_token, metadata.block_table, indices, BLOCK_SIZE=2
    )
    expected = _attention_reference(q, cache, physical, impl.softmax_scale, impl.sinks)[
        0
    ]
    torch.testing.assert_close(outputs[0], expected)
    torch.testing.assert_close(outputs[1], expected)
    assert gathers == ["legacy", "legacy"] + ["dedup"] * (2 if factor == 1 else 1)
    assert decode_widths == ([2, 2] if decode else [])
