# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Unpruned causal equivalence, dispatch bounds, and live cache transition."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm_hcu.models.hy_v4_flash.nvidia import compressed_indexer as module
from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_cache import IndexPoolMetadata
from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_ops import (
    expand_groups, select_groups, unpruned_indices,
)


@pytest.mark.parametrize("ratio,budget", [(1, 0), (4, 1), (4, 7), (4, 511)])
def test_unpruned_matches_score_selection_exactly(ratio, budget):
    torch.manual_seed(9)
    width = (budget + 1) * ratio
    positions = torch.tensor([-1, 0, ratio - 1, width - 1])
    active = torch.tensor([True, True, False, True])
    logits = torch.randn(4, max(64, budget + 1))
    groups = select_groups(logits, positions.clamp_min(0) // ratio, budget)
    reference = expand_groups(groups, positions, ratio, active)
    torch.testing.assert_close(unpruned_indices(positions, width, active), reference,
                               rtol=0, atol=0)


class Projection(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x), None


def make_indexer(device):
    indexer = module.FlashCompressedIndexer.__new__(module.FlashCompressedIndexer)
    nn.Module.__init__(indexer)
    indexer.ratio, indexer.group_budget, indexer.num_heads = 4, 7, 4
    indexer.max_model_len, indexer.logits_budget, indexer.num_sms = 64, 1 << 20, 2
    indexer.rope_dim, indexer.rope_parameters, indexer.eps = 0, {}, 1e-6
    indexer.use_mx_scales = indexer.varlen = False
    indexer.topk_indices_buffer = torch.empty(64, 32, dtype=torch.int32, device=device)
    indexer.token_key_projection = Projection(32, 128).to(device)
    indexer.token_key_norm = nn.RMSNorm(128).to(device)
    indexer.query_projection = Projection(32, 4 * 128).to(device)
    indexer.head_weight_projection = Projection(32, 4).to(device)
    indexer.key_value_projection = nn.Linear(128, 128, bias=False).to(device)
    indexer.key_gate_projection = nn.Linear(128, 128, bias=False).to(device)
    indexer.key_norm = nn.RMSNorm(128).to(device)
    indexer.intra_group_ape = nn.Parameter(torch.randn(4, 128, device=device))
    indexer.k_cache = SimpleNamespace(prefix="pool", kv_cache=torch.zeros(
        2, 64, 132, dtype=torch.uint8, device=device))
    indexer.tail_cache = SimpleNamespace(prefix="tail", kv_cache=torch.zeros(
        2, 4, 131, dtype=torch.float32, device=device))
    from hcu_adapter.kernels import mla_attention_emu as emu
    indexer._paged_metadata = emu.get_paged_mqa_logits_metadata
    indexer._paged_logits = lambda q, cache, w, lens, table, meta, max_model_len, indices: (
        emu.fp8_fp4_paged_mqa_logits(q, cache, w, lens, table, meta, max_model_len, indices)
    )
    return indexer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU/CUDA runtime")
@pytest.mark.parametrize("backend", ["emu", "triton"])
def test_real_cache_writes_survive_short_to_pruned_transition(monkeypatch, backend):
    torch.manual_seed(19)
    baseline = make_indexer("cuda")
    candidate = deepcopy(baseline)
    if backend == "triton":
        from vllm.v1.worker import workspace
        from hcu_adapter.kernels.vllm_paged_mqa_triton import fp32_paged_mqa_logits
        monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(torch.device("cuda")))
        candidate._paged_metadata = lambda *args, **kwargs: None
        candidate._paged_logits = fp32_paged_mqa_logits
    ctx = SimpleNamespace(attn_metadata={}, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(module, "get_forward_context", lambda: ctx)
    context = 0
    with torch.inference_mode():
        for count in (31, 1, 1, 3):
            seq_len = context + count
            starts = torch.tensor([0, count], device="cuda", dtype=torch.int32)
            lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
            table = torch.tensor([[1]], device="cuda", dtype=torch.int32)
            hidden = torch.randn(count, 32, device="cuda")
            positions = torch.arange(context, seq_len, device="cuda")
            outputs = []
            for indexer, bound in ((baseline, 64), (candidate, seq_len)):
                meta = IndexPoolMetadata(starts, lens, table, count, bound)
                ctx.attn_metadata = {"pool": meta, "tail": meta}
                outputs.append(indexer(hidden, hidden, positions).clone())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            torch.testing.assert_close(baseline.k_cache.kv_cache,
                                       candidate.k_cache.kv_cache, rtol=0, atol=0)
            torch.testing.assert_close(baseline.tail_cache.kv_cache,
                                       candidate.tail_cache.kv_cache, rtol=0, atol=0)
            context = seq_len
    assert baseline.query_projection.calls == 4
    assert candidate.query_projection.calls == 2
    assert candidate.head_weight_projection.calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU/CUDA runtime")
@pytest.mark.parametrize("mode,bound,query_calls", [
    (CUDAGraphMode.NONE, 32, 0),
    (CUDAGraphMode.NONE, 33, 1),
    (CUDAGraphMode.FULL, 32, 1),
])
def test_runtime_bound_and_full_graph_static_bound(monkeypatch, mode, bound, query_calls):
    indexer = make_indexer("cuda")
    starts = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    lens = torch.tensor([2], dtype=torch.int32, device="cuda")
    table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    meta = IndexPoolMetadata(starts, lens, table, 2, bound)
    ctx = SimpleNamespace(attn_metadata={"pool": meta, "tail": meta},
                          cudagraph_runtime_mode=mode)
    monkeypatch.setattr(module, "get_forward_context", lambda: ctx)
    with torch.inference_mode():
        indexer(torch.randn(2, 32, device="cuda"), torch.randn(2, 32, device="cuda"),
                torch.arange(2, device="cuda"))
    assert indexer.query_projection.calls == query_calls
