# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU numerical and lifecycle contracts for native DeepSeek V4 Full KV PCP."""

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from vllm_hcu.model_executor.layers.attention.pcp import (
    effective_pcp_world_size, replicated_mtp_batch_scope,
)
from vllm_hcu.patch.platform.core_fix.patch_vllm_config import _validate_hcu_pcp_scope
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_pcp as patch
from vllm_hcu.v1.glm53_pcp import (
    _prepare_attn, bind_glm53_pcp_batch, glm53_pcp_step_scope,
)
from tests.runtime_patch.test_glm52_pcp_config import _make_pcp_config
from tests.runtime_patch.test_glm52_pcp_manager import _make_batch, _make_managers
from tests.runtime_patch.test_glm52_pcp_runner import _config, pcp_runner_module


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("draft_tokens", [0, 1, 2, 3])
def test_v4_pcp_config_accepts_full_kv_and_native_mtp(pcp, draft_tokens):
    config = _make_pcp_config(
        architecture="DeepseekV4ForCausalLM", pcp=pcp,
        speculative=bool(draft_tokens), num_speculative_tokens=draft_tokens,
    )
    assert _validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("overrides", [
    {"dcp": 2}, {"dp": 2}, {"pp": 2}, {"enforce_eager": False},
    {"enable_expert_parallel": False}, {"use_mla": False},
    {"speculative": True, "speculative_method": "dspark"},
])
def test_v4_rejects_unsupported_pcp_without_affecting_pcp_one(overrides):
    config = _make_pcp_config(architecture="DeepseekV4ForCausalLM", **overrides)
    with pytest.raises(ValueError):
        _validate_hcu_pcp_scope(config)
    config.parallel_config.prefill_context_parallel_size = 1
    assert _validate_hcu_pcp_scope(config) is False


def test_v4_interleaved_cache_is_rejected():
    config = _make_pcp_config(architecture="DeepseekV4ForCausalLM")
    config.parallel_config.cp_kv_cache_interleave_size = 2
    with pytest.raises(ValueError, match="Full KV"):
        _validate_hcu_pcp_scope(config)


def test_v4_runner_accepts_multiple_cache_groups_and_installs_global_metadata(
    pcp_runner_module, monkeypatch,
):
    from vllm_hcu.v1 import glm53_pcp

    module, _ = pcp_runner_module
    installed = []
    monkeypatch.setattr(glm53_pcp, "install_deepseek_v4_pcp", installed.append)
    config = _config(2)
    config.model_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    runner = module.HcuGPUModelRunnerV2(config, "cpu")
    runner.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[object(), object()]))
    assert installed == [runner]
    runner.pcp_manager = SimpleNamespace(
        prepare_attn=lambda batch: ("local-tables", "expanded-slots"),
        prepare_global_attn=lambda: ("global-tables", "global-slots"),
    )
    assert runner.prepare_attn(SimpleNamespace()) == ("global-tables", "global-slots")


def test_empty_v4_shards_use_global_deepep_phase(pcp_runner_module, monkeypatch):
    from vllm_hcu.forward_context_runtime import (
        deepep_auto_request_phase_scope, get_deepep_auto_request_phase,
    )
    from vllm_hcu.v1 import glm53_pcp
    import numpy as np

    module, _ = pcp_runner_module
    bound = []
    monkeypatch.setattr(glm53_pcp, "bind_glm53_pcp_batch",
                        lambda manager, batch: bound.append(batch))
    config = _config(8)
    config.model_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    runner = module.HcuGPUModelRunnerV2(config, "cpu")
    global_batch = SimpleNamespace(is_prefilling_np=np.array([True]))
    local_batch = SimpleNamespace(is_prefilling_np=np.array([False]))
    runner.global_batch = global_batch
    runner.pcp_manager = SimpleNamespace(
        _global_batch=global_batch,
        partition_batch=lambda *args, **kwargs: local_batch,
    )
    with deepep_auto_request_phase_scope():
        result = runner.prepare_inputs("scheduler-output", "batch-req-state",
                                       SimpleNamespace(num_tokens=1))
        assert result is local_batch
        assert get_deepep_auto_request_phase() is global_batch.is_prefilling_np
    assert bound == [local_batch]


def test_v4_metadata_uses_original_request_rows_for_every_group():
    from test_glm52_pcp_manager import _InMemoryBlockTables

    batch = _make_batch([("decode", [17], 40, False),
                         ("prefill", list(range(9)), 30, True)])
    tables = _InMemoryBlockTables()
    tables.allow_global_gather = True
    managers, _ = _make_managers(4, block_tables=tables)
    manager = managers[1]
    local = manager.partition_batch(batch)
    local_tables, local_slots = manager.prepare_attn(local)

    def group(name):
        return SimpleNamespace(kv_cache_spec=None,
                               backend=SimpleNamespace(get_name=lambda: name))

    groups = [[group("ROCM_AITER_MLA_SPARSE"), group("SPARSE_SWA")],
              [group("COMPRESSOR")]]
    captured = []

    def prepare(input_batch, mode, blocks, slots, attn_groups, config, capture, ubatch):
        captured.append(input_batch)
        return {g.backend.get_name(): SimpleNamespace(
            pcp_world_size=4, batch=input_batch, slots=slots)
            for subgroups in attn_groups for g in subgroups}

    state = SimpleNamespace(_hcu_glm53_original_prepare_attn=prepare,
                            _hcu_pcp_replicate_all_groups=True)
    with glm53_pcp_step_scope():
        bind_glm53_pcp_batch(manager, local)
        metadata = _prepare_attn(state, local, None, local_tables,
                                 local_slots, groups, None)
    assert set(metadata) == {"ROCM_AITER_MLA_SPARSE", "SPARSE_SWA", "COMPRESSOR"}
    for item in metadata.values():
        assert item.batch is batch
        assert item.pcp_world_size == 1
        assert item.slots.shape[1] == batch.num_tokens


def test_install_deepseek_v4_pcp_wraps_model_state_once():
    from vllm_hcu.v1.glm53_pcp import install_deepseek_v4_pcp

    original = object()
    state = SimpleNamespace(prepare_attn=original)
    runner = SimpleNamespace(model_state=state)
    install_deepseek_v4_pcp(runner)
    wrapped = state.prepare_attn
    install_deepseek_v4_pcp(runner)
    assert state.prepare_attn is wrapped
    assert state._hcu_glm53_original_prepare_attn is original
    assert state._hcu_pcp_replicate_all_groups


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
def test_owned_rows_follow_partition_layout(pcp_size):
    batch = _make_batch([
        ("d", [17, 18], 40, False),
        ("p", list(range(9)), 30, True),
        ("short", [11], 90, True),
    ])
    managers, _ = _make_managers(pcp_size)
    owners = torch.zeros(batch.num_tokens, dtype=torch.int64)
    for manager in managers:
        local = manager.partition_batch(batch)
        with glm53_pcp_step_scope():
            bind_glm53_pcp_batch(manager, local)
            pcp_batch = patch_batch()
        expected = pcp_batch.local_indices[~local.is_padding]
        torch.testing.assert_close(pcp_batch.owned_rows(0, batch.num_tokens), expected)
        rows = pcp_batch.owned_rows(2, batch.num_tokens)
        assert pcp_batch.owned_rows(2, batch.num_tokens) is rows
        torch.testing.assert_close(rows + 2, expected[expected >= 2])
        owners[expected[expected >= 2]] += 1
    # Prefill rows have exactly one owner; decode rows are replicated.
    assert (owners[2:] == 1).all()


def patch_batch():
    from vllm_hcu.v1.glm53_pcp import _current_pcp_batch
    return _current_pcp_batch()


@pytest.fixture
def native_attention(monkeypatch):
    """Behavioral native attention double; all arithmetic uses real tensors."""
    rocm = ModuleType(patch.TARGET_MODULE)
    events = []

    def kernel(q, kv, indices, topk_length, scale, head_dim,
               nope_head_dim, rope_head_dim, attn_sink, output,
               ragged_indices=None, ragged_indptr=None):
        events.append(("sparse", q.shape[0]))
        output.copy_(q + kv.index_select(0, indices[:, 0]))

    class Attention:
        def _run_parallel_input_projections(self, hidden_states):
            events.append(("projections", hidden_states.shape[0]))
            return tuple(hidden_states * factor for factor in (2, 3, 4, 5))

        def _wq_b_proj(self, qr, qr_scale=None):
            events.append(("query", qr.shape[0]))
            return qr + 7

        def _o_proj(self, o, positions):
            events.append(("output", o.shape[0]))
            return o.flatten(1) * 11

        def _forward_prefill(self, q, positions, compressed_k_cache,
                             swa_k_cache, output, attn_metadata, swa_metadata):
            # Three chunks exercise nonzero Q storage offsets and rank shards
            # with no queries in some of the native chunks.
            for start in range(0, q.shape[0], 3):
                end = min(start + 3, q.shape[0])
                rows = torch.arange(start, end) + swa_metadata.num_decode_tokens
                rocm.rocm_sparse_attn_prefill(
                    q[start:end], compressed_k_cache, rows[:, None],
                    torch.ones(end - start, dtype=torch.int32),
                    1.0, q.shape[-1], q.shape[-1], 0, None, output[start:end],
                )

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            qr, kv, _, _ = self._run_parallel_input_projections(hidden_states)
            q = self._wq_b_proj(qr).unsqueeze(1)
            kv = kv.unsqueeze(1)
            output = torch.empty_like(q)
            decode = self.num_decode_tokens
            output[:decode] = q[:decode] + kv[:decode]
            self._forward_prefill(
                q[decode:], positions[decode:], kv, kv, output[decode:], None,
                SimpleNamespace(num_decode_tokens=decode),
            )
            return self._o_proj(output, positions)

    class Indexer:
        def _wq_b_proj(self, qr, qr_scale=None):
            return qr + 7

    rocm.DeepseekV4ROCMAiterMLAAttention = Attention
    rocm.DeepseekV4ROCMAiterMLASparseBackend = type("MLABackend", (), {})
    rocm.rocm_sparse_attn_prefill = kernel
    for module_name, symbol, value in (
        ("vllm.models.deepseek_v4.attention", "DeepseekV4Indexer", Indexer),
        ("vllm.models.deepseek_v4.compressor", "CompressorBackend", type("Compressor", (), {})),
        ("vllm.v1.attention.backends.mla.sparse_swa", "DeepseekSparseSWABackend", type("SWA", (), {})),
    ):
        stub = ModuleType(module_name)
        setattr(stub, symbol, value)
        monkeypatch.setitem(sys.modules, module_name, stub)
    assert patch.apply_to_module(rocm)
    assert patch.apply_to_module(rocm) is False
    return Attention, events, rocm


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
@pytest.mark.parametrize("requests", [
    [("p", list(range(1, 10)), 9, True)],
    [("tiny", [3], 9, True)],
    [("d", [17, 18, 19], 40, False)],
    [("d", [17], 40, False), ("p", list(range(1, 10)), 30, True),
     ("short", [11, 12], 90, True)],
])
def test_native_pcp_matches_global_attention_and_keeps_local_projections(
    native_attention, pcp_size, requests,
):
    Attention, events, rocm = native_attention
    global_batch = _make_batch(requests)
    managers, groups = _make_managers(pcp_size)
    local_batches = [manager.partition_batch(global_batch) for manager in managers]
    full_hidden = global_batch.input_ids[:, None].float().repeat(1, 2)
    expected = (5 * full_hidden + 7) * 11
    decode_tokens = sum(len(ids) for _, ids, _, prefill in requests if not prefill)

    def select(manager, value):
        start = manager.pcp_rank * manager._padded_num_tokens
        return value[manager._padded_gather_idx[start:start + manager._padded_num_tokens]]

    for rank, manager in enumerate(managers):
        events.clear()
        layer = Attention()
        layer.num_decode_tokens = decode_tokens
        # Independent collective oracle: projections are known pointwise
        # operations. It also checks each rank actually projects local inputs.
        stages = [full_hidden, *[full_hidden * f for f in (2, 3, 4, 5)],
                  full_hidden * 2 + 7]
        calls = []

        def gather(local, dim=0):
            full = stages[len(calls)]
            torch.testing.assert_close(local, select(manager, full))
            calls.append(local.shape[0])
            return torch.cat([select(peer, full) for peer in managers])

        groups[rank].all_gather = gather
        with glm53_pcp_step_scope():
            bind_glm53_pcp_batch(manager, local_batches[rank])
            actual = layer.forward(local_batches[rank].positions,
                                   select(manager, full_hidden))
            owned = patch_batch().owned_rows(decode_tokens, global_batch.num_tokens)
        valid = ~local_batches[rank].is_padding
        torch.testing.assert_close(actual[valid], select(manager, expected)[valid])
        assert calls == [local_batches[rank].num_tokens_after_padding] * 6
        assert ("projections", actual.shape[0]) in events
        assert ("output", actual.shape[0]) in events
        assert sum(count for name, count in events if name == "sparse") == owned.numel()
        assert patch.current_pcp_batch() is None
        assert patch._PREFILL_ROWS.get() is None
        assert effective_pcp_world_size(pcp_size) == pcp_size


def test_non_pcp_and_replicated_mtp_call_original_attention(native_attention):
    Attention, events, _ = native_attention
    layer = Attention()
    layer.num_decode_tokens = 1
    hidden = torch.arange(8).float().view(4, 2)
    batch = _make_batch([("p", list(range(4)), 4, True)])
    managers, _ = _make_managers(2)
    local = managers[0].partition_batch(batch)
    with glm53_pcp_step_scope():
        bind_glm53_pcp_batch(managers[0], local)
        with replicated_mtp_batch_scope():
            actual = layer.forward(torch.arange(4), hidden)
    torch.testing.assert_close(actual, (5 * hidden + 7) * 11)
    torch.testing.assert_close(layer.forward(torch.arange(4), hidden), actual)
    assert patch.current_pcp_batch() is None


def test_pcp_patch_rejects_stale_wrappers(native_attention, monkeypatch):
    Attention, _, rocm = native_attention
    monkeypatch.setattr(Attention, "forward", lambda *args: None)
    with pytest.raises(patch.PatchCompatibilityError, match="stale"):
        patch.apply_to_module(rocm)


def test_pcp_patch_rejects_signature_drift_before_mutation(native_attention, monkeypatch):
    Attention, _, rocm = native_attention
    original_forward = Attention.forward
    monkeypatch.setattr(Attention, patch._MARKER, False)
    monkeypatch.setattr(Attention, "_o_proj", lambda self, o: o)
    with pytest.raises(patch.PatchCompatibilityError, match="signature"):
        patch.apply_to_module(rocm)
    assert Attention.forward is original_forward


def test_prefill_exception_restores_query_scope(native_attention, monkeypatch):
    Attention, _, rocm = native_attention
    layer = Attention()
    monkeypatch.setattr(patch, "_ATTENTION_BATCH", SimpleNamespace(
        get=lambda: SimpleNamespace(owned_rows=lambda start, end: torch.tensor([0])),
    ))
    monkeypatch.setattr(rocm, "rocm_sparse_attn_prefill",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("kernel failed")))
    q = torch.ones(1, 1, 2)
    with pytest.raises(RuntimeError, match="kernel failed"):
        layer._forward_prefill(q, torch.arange(1), q, q, torch.empty_like(q), None,
                               SimpleNamespace(num_decode_tokens=0))
    assert patch._PREFILL_ROWS.get() is None


@pytest.mark.parametrize("owned_rows", [[0, 2, 7], [], [4, 5, 6, 7]])
@pytest.mark.parametrize("use_lightop", [False, True])
@pytest.mark.parametrize("gfx938", [False, True])
def test_indexer_owned_queries_keep_full_kv_coordinates(
    monkeypatch, owned_rows, use_lightop, gfx938,
):
    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as sparse
    from vllm import _custom_ops as ops
    import vllm.utils.torch_utils as torch_utils

    monkeypatch.setattr(sparse, "DeepseekV32IndexerMetadata", SimpleNamespace)
    monkeypatch.setattr(sparse, "current_platform", SimpleNamespace(
        is_rocm=lambda: True, fp8_dtype=lambda: torch.float32,
    ))
    monkeypatch.setattr(sparse, "on_gfx938", lambda: gfx938)
    monkeypatch.setattr(sparse, "_indexer_cache_as_hipc_view", lambda cache: cache)
    monkeypatch.setattr(sparse, "cp_gather_indexer_k_bf16_cache_triton", lambda *a, **k: None)
    monkeypatch.setattr(ops, "cp_gather_indexer_k_quant_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch_utils, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(sparse, "_use_lightop_sparse_mla_topk", lambda: use_lightop)
    monkeypatch.setattr(sparse, "_lightop_topk_indices_prefill",
                        lambda logits, starts, ends, output, topk:
                        output.copy_(sparse._topk_indices_torch(logits, topk, starts, ends)))
    calls = []

    def logits(q, kv, weights, starts, ends):
        calls.append(q.shape[0])
        # Different signs make the best keys vary by row; different causal
        # intervals verify PCP selects bounds as well as queries and weights.
        return q[:, 0, 0, None] * torch.arange(kv[0].shape[0])[None, :] + weights[:, :1]

    monkeypatch.setattr(sparse, "rocm_fp8_mqa_logits", logits)
    chunks = [SimpleNamespace(
        token_start=start, token_end=start + 4, total_seq_lens=8,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlen_ks=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
    ) for start in (0, 4)]
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(8), num_kv_actual_tokens=8,
        num_decodes=0, num_decode_tokens=0, num_prefills=2, num_prefill_tokens=8,
        prefill=SimpleNamespace(chunks=chunks),
    )
    monkeypatch.setattr(sparse, "get_forward_context", lambda: SimpleNamespace(
        attn_metadata={"indexer": metadata},
    ))

    def run():
        q = torch.tensor([1, -1, 2, -2, 3, -3, 4, -4],
                         dtype=torch.float32).view(8, 1, 1)
        weights = torch.arange(8).view(8, 1).float()
        return sparse.rocm_aiter_sparse_attn_indexer_native(
            hidden_states=torch.zeros(8, 1), k_cache_prefix="indexer",
            kv_cache=torch.zeros(1, 8, 1), q_fp8=q, k=None, weights=weights,
            quant_block_size=1, scale_fmt="e4m3", topk_tokens=2, head_dim=1,
            max_model_len=8, total_seq_lens=16,
            topk_indices_buffer=torch.empty(8, 2, dtype=torch.int32),
            skip_k_cache_insert=True,
        )

    expected = run()
    calls.clear()
    owned = torch.tensor(owned_rows, dtype=torch.int64)

    def rows(start, end):
        return owned[(owned >= start) & (owned < end)] - start

    token = patch._ATTENTION_BATCH.set(SimpleNamespace(owned_rows=rows))
    try:
        actual = run()
    finally:
        patch._ATTENTION_BATCH.reset(token)
    torch.testing.assert_close(actual[owned], expected[owned])
    mask = torch.ones(8, dtype=torch.bool)
    mask[owned] = False
    assert (actual[mask] == -1).all()
    assert sum(calls) == len(owned_rows)
