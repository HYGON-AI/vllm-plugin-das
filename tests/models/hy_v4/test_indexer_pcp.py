# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exercise HYV4's constructed indexer through the real PCP cache-write seam."""

from types import SimpleNamespace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

CHILD = __name__ == "__main__" or os.environ.get("HYV4_PCP_CONTRACT_CHILD") == "1"
if CHILD:
    # The real whole-module exchange must be armed before the canonical
    # indexer is imported. Isolate its process-global custom-op registration
    # from other tests, which legitimately import the unexchanged target.
    os.environ["HYV4_PCP_CONTRACT_CHILD"] = "1"
    from vllm_hcu.patch.module_exchange import register_attention_exchanges
    from vllm_hcu.patch.import_coordinator import install_import_coordinator
    register_attention_exchanges()
    install_import_coordinator()
    from vllm.model_executor.layers import sparse_attn_indexer as indexer_ops
    from vllm_hcu.model_executor.layers.attention import pcp
    from vllm_hcu.models.hy_v4 import attention
    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as native
    from vllm_hcu.v1.pcp_manager import HcuPCPManager


@pytest.fixture(scope="module")
def isolated_results():
    if CHILD:
        return None
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = "HYV4_PCP_RESULTS "
    line = next((x for x in result.stdout.splitlines() if x.startswith(marker)), None)
    assert line is not None, result.stdout + result.stderr
    return json.loads(line[len(marker):])


def _assert_isolated(results, request):
    case = request.node.nodeid.split("::", 1)[1]
    assert case in results, results
    assert results[case]["passed"], results[case]["detail"]


def _constructed_indexer(monkeypatch, world_size, metadata, cache, inputs, group):
    """Keep model construction/forward and PCP gather real; double GPU leaves."""
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=world_size),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    hf = SimpleNamespace(index_topk=2048, index_n_heads=1, index_head_dim=128,
                         qk_rope_head_dim=64)
    for name in ("ReplicatedLinear", "MergedColumnParallelLinear", "LayerNorm"):
        monkeypatch.setattr(attention, name, lambda *a, **kw: torch.nn.Identity())
    cache_module = torch.nn.Module()
    cache_module.prefix = "model.layers.41.self_attn.indexer.k_cache"
    cache_module.kv_cache = cache
    monkeypatch.setattr(attention, "DeepseekV32IndexerCache", lambda **kw: cache_module)
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.indexer.get_max_prefill_buffer_size",
        lambda config: 4096,
    )

    def init_op(self, k_cache, quant_block_size, scale_fmt, topk_tokens,
                head_dim, max_model_len, max_total_seq_len, topk_indices_buffer,
                skip_k_cache_insert=False, use_fp4_cache=False):
        torch.nn.Module.__init__(self)
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.pcp_world_size = world_size
        self._forward_method = self.forward_hip

    # Do not replace the class chosen by HYV4: its real HIP dispatch is the
    # contract under test. Constructor doubles only avoid accelerator allocation.
    monkeypatch.setattr(indexer_ops.SparseAttnIndexer, "__init__", init_op)
    monkeypatch.setattr(indexer_ops, "on_gfx938", lambda: True)
    monkeypatch.setattr(indexer_ops.rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(indexer_ops, "get_forward_context", lambda: SimpleNamespace(
        attn_metadata={cache_module.prefix: metadata}))
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    monkeypatch.setattr(attention.Indexer, "prepare_inputs", lambda *a: inputs)
    calls = []

    def write(k, destination, slots, quant_block_size, scale_fmt):
        assert destination is cache
        calls.append(("write", k.clone(), slots.clone()))
        # GPU leaf: the native call consumes one slot per supplied K row.
        active_slots = slots[:len(k)]
        valid = active_slots >= 0
        cache[active_slots[valid]] = k[valid]

    monkeypatch.setattr(indexer_ops.ops, "indexer_k_quant_and_cache", write)

    def native_leaf(hidden, prefix, destination, q, k, weights, quant_block_size,
                    scale_fmt, topk_tokens, head_dim, max_model_len,
                    max_total_seq_len, topk_buffer, skip_k_cache_insert=False):
        calls.append(("native", skip_k_cache_insert))
        if not skip_k_cache_insert:
            write(k, destination, metadata.slot_mapping,
                  quant_block_size, scale_fmt)
        return topk_buffer

    def isolated_leaf(*args):
        calls.append(("isolated", args[-1]))
        assert args[3] is inputs[1]  # Q remains PCP-local.
        assert args[4] is inputs[2]
        return args[-2]

    monkeypatch.setattr(native, "rocm_aiter_sparse_attn_indexer_native", native_leaf)
    monkeypatch.setattr(torch.ops.vllm, "hcu_sparse_attn_indexer", isolated_leaf)
    topk = torch.full((len(inputs[0]), 2048), -1, dtype=torch.int32)
    model_indexer = attention.Indexer(
        cfg, hf, hidden_size=8, q_lora_rank=8, quant_config=None,
        cache_config=SimpleNamespace(), topk_indices_buffer=topk,
        prefix="model.layers.41.self_attn.indexer",
    )
    return model_indexer, calls, topk


@pytest.mark.parametrize("length", [34, 2310, 3145])
@pytest.mark.parametrize("rank", range(4))
def test_constructed_hyv4_indexer_writes_all_pcp_keys_before_local_topk(
    monkeypatch, length, rank, isolated_results, request,
):
    if not CHILD:
        return _assert_isolated(isolated_results, request)
    width = 2 * ((length + 7) // 8)
    rank_slots = []
    for r in range(4):
        early, late = HcuPCPManager.rank_segments(length, pcp_size=4, pcp_rank=r)
        slots = torch.cat((torch.arange(late.start, late.stop),
                           torch.arange(early.start, early.stop)))
        rank_slots.append(torch.cat((slots, torch.full((width-len(slots),), -1))))
    all_slots = torch.cat(rank_slots)
    keys = [(s.float()+1)[:, None].expand(-1, 128).clone() for s in rank_slots]
    all_keys = torch.cat(keys)
    gathers = []

    def gather(tensor, dim=0):
        assert dim == 0
        if tensor.dtype == torch.int64:
            torch.testing.assert_close(tensor, rank_slots[rank])
            gathers.append("slots")
            return all_slots
        torch.testing.assert_close(tensor, keys[rank])
        gathers.append("keys")
        return all_keys

    group = SimpleNamespace(world_size=4, rank_in_group=rank, all_gather=gather)
    metadata = SimpleNamespace(pcp_world_size=4, pcp_token_counts=(width,)*4,
                               num_decode_tokens=0, slot_mapping=all_slots)
    cache = torch.full((length, 128), -999.0)
    inputs = (torch.zeros(width, 8), torch.zeros(width, 1, 128),
              keys[rank], torch.ones(width, 1))
    model_indexer, calls, topk = _constructed_indexer(
        monkeypatch, 4, metadata, cache, inputs, group)

    result = model_indexer(inputs[0], torch.zeros(width, 8), rank_slots[rank], None)

    assert gathers == ["keys", "slots"]
    assert result is topk
    torch.testing.assert_close(cache, torch.arange(1, length+1).float()[:, None].expand(-1,128))
    assert [c[0] for c in calls] == ["write", "isolated"]
    torch.testing.assert_close(calls[0][1], all_keys)
    torch.testing.assert_close(calls[0][2], all_slots)
    assert calls[1][1] is True  # Do not insert rank-local K a second time.


@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("rank", range(4))
def test_constructed_hyv4_indexer_decode_is_replicated_without_gather(
    monkeypatch, world_size, rank, isolated_results, request,
):
    if not CHILD:
        return _assert_isolated(isolated_results, request)
    def forbidden_gather(*args, **kwargs):
        pytest.fail("replicated decode must not gather")

    group = SimpleNamespace(world_size=world_size, rank_in_group=rank % world_size,
                            all_gather=forbidden_gather)
    metadata = SimpleNamespace(pcp_world_size=world_size, num_decode_tokens=1,
                               slot_mapping=torch.tensor([3]))
    cache = torch.full((4,128), -999.0)
    inputs = (torch.zeros(1,8), torch.zeros(1,1,128), torch.full((1,128),7.0),
              torch.ones(1,1))
    model_indexer, calls, topk = _constructed_indexer(
        monkeypatch, world_size, metadata, cache, inputs, group)
    assert model_indexer(inputs[0], torch.zeros(1,8), torch.tensor([3145]), None) is topk
    torch.testing.assert_close(cache[3], inputs[2][0])
    if world_size == 1:
        assert type(model_indexer.indexer_op) is indexer_ops.SparseAttnIndexer
        assert [c[0] for c in calls] == ["native", "write"]
    else:
        assert [c[0] for c in calls] == ["write", "isolated"]


def test_constructed_hyv4_indexer_rejects_pcp_metadata_mismatch(
    monkeypatch, isolated_results, request,
):
    if not CHILD:
        return _assert_isolated(isolated_results, request)
    metadata = SimpleNamespace(pcp_world_size=1, num_decode_tokens=0,
                               slot_mapping=torch.tensor([0]))
    inputs = (torch.zeros(1,8), torch.zeros(1,1,128), torch.zeros(1,128),
              torch.ones(1,1))
    model_indexer, calls, _ = _constructed_indexer(
        monkeypatch, 4, metadata, torch.zeros(1,128), inputs, None)
    with pytest.raises(RuntimeError, match="PCP sparse-indexer metadata world size mismatch"):
        model_indexer(inputs[0], torch.zeros(1,8), torch.tensor([0]), None)
    assert calls == []


if __name__ == "__main__":
    class Results:
        values = {}

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed:
                self.values[report.nodeid.split("::", 1)[1]] = dict(
                    passed=report.passed, detail=str(report.longrepr))

    results = Results()
    pytest.main([str(Path(__file__).resolve()), "-q"], plugins=[results])
    print("HYV4_PCP_RESULTS " + json.dumps(results.values))
