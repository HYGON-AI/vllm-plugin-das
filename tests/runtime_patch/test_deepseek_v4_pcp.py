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


@pytest.mark.parametrize("architecture", [
    "DeepseekV4ForCausalLM", "DeepSeekV4MTPModel",
])
@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("draft_tokens", [0, 1, 2, 3])
def test_v4_pcp_config_accepts_full_kv_and_native_mtp(
    architecture, pcp, draft_tokens,
):
    config = _make_pcp_config(
        architecture=architecture, pcp=pcp,
        speculative=bool(draft_tokens), num_speculative_tokens=draft_tokens,
    )
    assert _validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("architecture", [
    "DeepseekV4ForCausalLM", "DeepSeekV4MTPModel",
])
@pytest.mark.parametrize("overrides", [
    {"dcp": 2}, {"dp": 2}, {"pp": 2}, {"enforce_eager": False},
    {"enable_expert_parallel": False}, {"use_mla": False},
    {"speculative": True, "speculative_method": "dspark"},
    {"speculative": True, "num_speculative_tokens": 0},
    {"speculative": True, "num_speculative_tokens": 4},
])
def test_v4_rejects_unsupported_pcp_without_affecting_pcp_one(
    architecture, overrides,
):
    config = _make_pcp_config(architecture=architecture, **overrides)
    with pytest.raises(ValueError):
        _validate_hcu_pcp_scope(config)
    config.parallel_config.prefill_context_parallel_size = 1
    assert _validate_hcu_pcp_scope(config) is False


@pytest.mark.parametrize("architecture", [
    "DeepseekV4ForCausalLM", "DeepSeekV4MTPModel",
])
def test_v4_interleaved_cache_is_rejected(architecture):
    config = _make_pcp_config(architecture=architecture)
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


@pytest.mark.parametrize("with_mtp", [False, True])
def test_v4_profile_keeps_global_budget(pcp_runner_module, monkeypatch, with_mtp):
    module, _ = pcp_runner_module
    seen = []
    monkeypatch.setattr(
        module.GPUModelRunner, "profile_run",
        lambda self: seen.append((self.max_num_tokens, self.speculator)),
        raising=False,
    )
    config = _config(8)
    config.model_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    runner = module.HcuGPUModelRunnerV2(config, "cpu")
    runner.max_num_tokens = 16384
    runner.speculator = object() if with_mtp else None

    runner.profile_run()

    assert seen == [(16384, runner.speculator)]
    assert runner.max_num_tokens == 16384


@pytest.mark.parametrize("architecture,pcp", [
    ("DeepseekV4ForCausalLM", 8),
    ("Glm5NextForCausalLM", 8),
    ("GlmMoeDsaForCausalLM", 8),
    ("DeepseekV4ForCausalLM", 1),
    ("Glm5NextForCausalLM", 1),
])
def test_prepare_inputs_selects_deepep_phase_batch(
    pcp_runner_module, monkeypatch, architecture, pcp,
):
    from vllm_hcu.forward_context_runtime import (
        deepep_auto_request_phase_scope, get_deepep_auto_request_phase,
    )
    from vllm_hcu.v1 import glm53_pcp
    import numpy as np

    module, _ = pcp_runner_module
    bound = []
    monkeypatch.setattr(glm53_pcp, "bind_glm53_pcp_batch",
                        lambda manager, batch: bound.append(batch))
    config = _config(pcp)
    config.model_config = SimpleNamespace(architectures=[architecture])
    runner = module.HcuGPUModelRunnerV2(config, "cpu")
    global_batch = SimpleNamespace(is_prefilling_np=np.array([True]))
    local_batch = SimpleNamespace(is_prefilling_np=np.array([False]))
    runner.global_batch = global_batch
    runner.pcp_manager = None if pcp == 1 else SimpleNamespace(
        _global_batch=global_batch,
        partition_batch=lambda *args, **kwargs: local_batch,
    )
    with deepep_auto_request_phase_scope():
        result = runner.prepare_inputs("scheduler-output", "batch-req-state",
                                       SimpleNamespace(num_tokens=1))
        expected_batch = global_batch if pcp == 1 else local_batch
        phase_batch = (global_batch if architecture == "DeepseekV4ForCausalLM"
                       else expected_batch)
        assert result is expected_batch
        assert get_deepep_auto_request_phase() is phase_batch.is_prefilling_np
    uses_global_batch = pcp > 1 and architecture in (
        "DeepseekV4ForCausalLM", "Glm5NextForCausalLM",
    )
    assert bound == ([local_batch] if uses_global_batch else [])


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


def _select(manager, value):
    start = manager.pcp_rank * manager._padded_num_tokens
    return value[manager._padded_gather_idx[start:start + manager._padded_num_tokens]]


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
        head_dim = 2  # not the 512-wide V4 head: exercise the compatibility path

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

        def forward_mqa(self, q, kv, positions, output):
            raise AssertionError("local-query MQA must not run")

        def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
            raise AssertionError("local-query cache insert must not run")

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
            torch.testing.assert_close(local, _select(manager, full))
            calls.append(local.shape[0])
            return torch.cat([_select(peer, full) for peer in managers])

        groups[rank].all_gather = gather
        with glm53_pcp_step_scope():
            bind_glm53_pcp_batch(manager, local_batches[rank])
            actual = layer.forward(local_batches[rank].positions,
                                   _select(manager, full_hidden))
            owned, _ = patch_batch().local_query_rows(decode_tokens, global_batch.num_tokens)
        valid = ~local_batches[rank].is_padding
        torch.testing.assert_close(actual[valid], _select(manager, expected)[valid])
        expected_calls = ([local_batches[rank].num_tokens_after_padding] * 6
                          if manager._global_has_prefill else [])
        assert calls == expected_calls
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


@pytest.mark.parametrize("local_queries", [False, True])
@pytest.mark.parametrize("owned_rows", [[0, 2, 7], [], [4, 5, 6, 7]])
@pytest.mark.parametrize("use_lightop", [False, True])
@pytest.mark.parametrize("gfx938", [False, True])
def test_indexer_owned_queries_keep_full_kv_coordinates(
    monkeypatch, owned_rows, use_lightop, gfx938, local_queries,
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

    def run(owned=None):
        q = torch.tensor([1, -1, 2, -2, 3, -3, 4, -4],
                         dtype=torch.float32).view(8, 1, 1)
        weights = torch.arange(8).view(8, 1).float()
        if owned is not None:
            q, weights = q[owned], weights[owned]
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
        mask = (owned >= start) & (owned < end)
        return owned[mask] - start, torch.arange(owned.numel())[mask]

    batch = SimpleNamespace(local_query_rows=rows)
    token = patch._ATTENTION_BATCH.set(batch)
    local = patch._LOCAL_QUERIES.set(batch if local_queries else None)
    try:
        actual = run(owned if local_queries else None)
    finally:
        patch._LOCAL_QUERIES.reset(local)
        patch._ATTENTION_BATCH.reset(token)
    torch.testing.assert_close(actual[owned], expected[owned])
    mask = torch.ones(8, dtype=torch.bool)
    mask[owned] = False
    assert (actual[mask] == -1).all()
    assert sum(calls) == len(owned_rows)


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
def test_local_query_rows_map_global_rows_to_local_positions(pcp_size):
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
        rows, positions = pcp_batch.local_query_rows(2, batch.num_tokens)
        torch.testing.assert_close(pcp_batch.local_indices[positions], rows + 2)
        assert not local.is_padding[positions].any()
        assert pcp_batch.local_query_rows(2, batch.num_tokens)[1] is positions
        owners[rows + 2] += 1
    # Prefill rows have exactly one owner; decode rows are replicated.
    assert (owners[2:] == 1).all()


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
@pytest.mark.parametrize("length", [1, 9, 32])
@pytest.mark.parametrize("fail_sparse", [False, True])
def test_local_query_path_only_gathers_kv(native_attention, monkeypatch,
                                          pcp_size, length, fail_sparse):
    Attention, events, _ = native_attention
    batch = _make_batch([("p", list(range(1, length + 1)), length, True)])
    managers, groups = _make_managers(pcp_size)
    local_batches = [manager.partition_batch(batch) for manager in managers]
    full = batch.input_ids[:, None].float().repeat(1, 512)

    # CPU numerical native-interface double: distinct Q and KV lengths,
    # chunk boundaries, padding and local output all use real tensors.
    def project(self, hidden_states):
        return torch.cat((hidden_states * 2, hidden_states * 3), -1), None, None, None

    def query(self, qr, qr_scale=None):
        return qr.repeat(1, 2)

    def insert(self, q, kv, positions, attn_metadata):
        raise AssertionError("full-query cache insert must not run")

    cache_store = {}

    def mqa(self, q, kv, positions, output):
        assert q.shape == output.shape
        kv = cache_store["kv"]
        self._forward_prefill(q, positions, kv.unsqueeze(1), kv.unsqueeze(1),
                              output, None, SimpleNamespace(num_decode_tokens=0))

    def prepare(self, hidden, qr, kv, qr_scale, kv_score, indexer_kv_score,
                weights, positions, output):
        assert hidden.shape[0] == batch.num_tokens
        q = self._wq_b_proj(qr, qr_scale).view(-1, 2, 512)
        q = self._fused_qnorm_rope_kv_insert(q, kv, positions, None)
        self.forward_mqa(q, kv, positions, output)

    def pcp_insert(q, kv, cache, slots, q_positions, kv_positions, rope, eps, block):
        assert q.shape[0] == q_positions.numel()
        assert kv.shape[0] == kv_positions.numel() == batch.num_tokens
        torch.testing.assert_close(kv, full * 6)  # learned KVNorm exactly once
        cache_store["kv"] = kv
        q.add_(7)

    # A fresh class/module isolates optional native methods from wrappers
    # already installed by the fixture.
    original_attention = Attention
    rocm = ModuleType(patch.TARGET_MODULE)

    class LocalAttention(original_attention):
        _vllm_hcu_deepseek_v4_pcp_applied = False
        forward = original_attention.forward.__wrapped__
        _run_parallel_input_projections = project
        _wq_b_proj = query
        _o_proj = original_attention._o_proj.__wrapped__
        _fused_qnorm_rope_kv_insert = insert
        forward_mqa = mqa
        _prepare_and_attn_fn = prepare

    rocm.DeepseekV4ROCMAiterMLAAttention = LocalAttention
    rocm.DeepseekV4ROCMAiterMLASparseBackend = type("LocalBackend", (), {})
    # Original numerical sparse provider from the fixture's wrapper closure.
    provider = native_attention[2].rocm_sparse_attn_prefill.__wrapped__

    def sparse_provider(q, kv, indices, topk_length, scale, head_dim,
                        nope_head_dim, rope_head_dim, attn_sink, output,
                        ragged_indices=None, ragged_indptr=None):
        if fail_sparse:
            raise RuntimeError("sparse kernel failed")
        return provider(q, kv, indices, topk_length, scale, head_dim,
                        nope_head_dim, rope_head_dim, attn_sink, output,
                        ragged_indices, ragged_indptr)
    rocm.rocm_sparse_attn_prefill = sparse_provider

    # The numerical prefill double resolves its provider through this module.
    def prefill(self, q, positions, compressed_k_cache, swa_k_cache, output,
                attn_metadata, swa_metadata):
        for start in range(0, q.shape[0], 3):
            end = min(start + 3, q.shape[0])
            rocm.rocm_sparse_attn_prefill(
                q[start:end], compressed_k_cache,
                torch.arange(start, end)[:, None],
                torch.ones(end - start, dtype=torch.int32), 1., 512, 512, 0,
                None, output[start:end],
            )
    LocalAttention._forward_prefill = prefill
    monkeypatch.setattr(patch, "_pcp_insert_op", lambda: pcp_insert)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES", True)
    assert patch.apply_to_module(rocm)
    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "get_forward_context", lambda: SimpleNamespace(
        attn_metadata={"swa": SimpleNamespace(
            slot_mapping=torch.arange(length), block_size=256,
            num_decodes=0, num_decode_tokens=0)},
    ))

    for rank, manager in enumerate(managers):
        layer = LocalAttention()
        layer.q_lora_rank = layer.head_dim = 512
        layer.n_local_heads = layer.padded_heads = 2
        layer.q_norm = lambda x: x
        layer.kv_norm = lambda x: x * 2
        layer.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(1))
        layer.eps = 1e-6
        layer.swa_cache_layer = SimpleNamespace(
            prefix="swa", kv_cache=torch.zeros(1, 1, dtype=torch.uint8))
        gathered = []

        def gather(local, dim=0):
            torch.testing.assert_close(local, _select(manager, full * 6))
            gathered.append(local.shape)
            return torch.cat([_select(peer, full * 6) for peer in managers])
        groups[rank].all_gather = gather
        valid = ~local_batches[rank].is_padding
        with glm53_pcp_step_scope():
            bind_glm53_pcp_batch(manager, local_batches[rank])
            if fail_sparse and valid.any():
                with pytest.raises(RuntimeError, match="sparse kernel failed"):
                    layer.forward(local_batches[rank].positions,
                                  _select(manager, full))
            else:
                actual = layer.forward(local_batches[rank].positions,
                                       _select(manager, full))
                expected = ((full * 8 + 7).repeat(1, 2)) * 11
                torch.testing.assert_close(actual[valid],
                                           _select(manager, expected)[valid])
        assert len(gathered) == 1  # KV only; never hidden, QR or Q
        assert patch._LOCAL_QUERIES.get() is None
        assert patch._LOCAL_MQA.get() is None
        assert patch.current_pcp_batch() is None
        assert patch._PREFILL_ROWS.get() is None


def test_indexer_quant_selects_local_positions(native_attention, monkeypatch):
    _, _, rocm = native_attention
    attention_module = sys.modules["vllm.models.deepseek_v4.attention"]
    recorded = []

    def quant(positions, q, weights):
        recorded.append(positions)
        return q, weights
    monkeypatch.setattr(attention_module, "fused_indexer_q_rope_quant", quant,
                        raising=False)
    # Reapply to a fresh native-interface class so the quant provider is
    # wrapped at the same lifecycle point as in the real import callback.
    old = rocm.DeepseekV4ROCMAiterMLAAttention

    class Fresh(old):
        _vllm_hcu_deepseek_v4_pcp_applied = False
        forward = old.forward.__wrapped__
        _run_parallel_input_projections = old._run_parallel_input_projections.__wrapped__
        _wq_b_proj = old._wq_b_proj.__wrapped__
        _o_proj = old._o_proj.__wrapped__
        _forward_prefill = old._forward_prefill.__wrapped__
        forward_mqa = old.forward_mqa.__wrapped__
        _fused_qnorm_rope_kv_insert = old._fused_qnorm_rope_kv_insert.__wrapped__
    monkeypatch.setattr(rocm, "DeepseekV4ROCMAiterMLAAttention", Fresh)
    monkeypatch.setattr(rocm, "rocm_sparse_attn_prefill",
                        rocm.rocm_sparse_attn_prefill.__wrapped__)
    patch.apply_to_module(rocm)
    batch = SimpleNamespace(localize=lambda x: x[torch.tensor([1, 3])])
    token = patch._ATTENTION_BATCH.set(batch)
    local = patch._LOCAL_QUERIES.set(batch)
    try:
        q, weights = torch.ones(2, 1), torch.ones(2, 1)
        result = attention_module.fused_indexer_q_rope_quant(torch.arange(4), q, weights)
        torch.testing.assert_close(recorded[-1], torch.tensor([1, 3]))
        assert result[0] is q and result[1] is weights
    finally:
        patch._LOCAL_QUERIES.reset(local)
        patch._ATTENTION_BATCH.reset(token)
    attention_module.fused_indexer_q_rope_quant(torch.arange(4), q, weights)
    torch.testing.assert_close(recorded[-1], torch.arange(4))


@pytest.mark.parametrize("pure_prefill", [False, True])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("metadata_decode", [False, True])
def test_local_query_capability_gate(monkeypatch, pure_prefill, available, enabled,
                                     metadata_decode):
    import numpy as np
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES", enabled)
    monkeypatch.setattr(patch, "_pcp_insert_op", lambda:
                        (lambda *args: None) if available else None)
    layer = SimpleNamespace(
        _prepare_and_attn_fn=object(), _fused_qnorm_rope_kv_insert=object(),
        forward_mqa=object(), head_dim=512, n_local_heads=64, padded_heads=64,
        swa_cache_layer=SimpleNamespace(prefix="swa",
                                        kv_cache=torch.empty(1, dtype=torch.uint8)),
    )
    batch = SimpleNamespace(global_batch=SimpleNamespace(
        is_prefilling_np=np.array([pure_prefill]),
    ))
    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "get_forward_context", lambda: SimpleNamespace(
        attn_metadata={"swa": SimpleNamespace(num_decodes=int(metadata_decode),
                                              num_decode_tokens=int(metadata_decode))},
    ))
    assert patch._can_use_local_queries(layer, batch) == (
        pure_prefill and available and enabled and not metadata_decode
    )
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert not patch._can_use_local_queries(layer, batch)


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
@pytest.mark.parametrize("num_requests", [1, 3, 9])
@pytest.mark.parametrize("with_indexer", [False, True])
@pytest.mark.parametrize("local_queries", [False, True])
def test_replicated_decode_raw_kv_forward_never_gathers(
    native_attention, monkeypatch, pcp_size, num_requests, with_indexer,
    local_queries,
):
    import ast
    from pathlib import Path

    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_attention
    from vllm_hcu.platforms import envs as henvs
    from vllm_hcu.model_executor.layers.attention.pcp import in_replicated_mtp_batch
    from vllm_hcu.v1.glm53_pcp import _current_pcp_batch

    # Retain the actual HCU raw-KV forward; only projection/norm/attention
    # arithmetic and collective device boundaries are replaced below.
    source = ast.parse(Path(patch_deepseek_v4_attention.__file__).read_text())
    forward = next(node for node in ast.walk(source)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "hcu_attention_forward")
    forward.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[forward], type_ignores=[]),
                 patch_deepseek_v4_attention.__file__, "exec"), namespace)

    Attention, _, rocm = native_attention
    attention_module = sys.modules["vllm.models.deepseek_v4.attention"]
    Indexer = attention_module.DeepseekV4Indexer
    monkeypatch.setattr(Indexer, "_wq_b_proj", Indexer._wq_b_proj.__wrapped__)

    class DecodeAttention(Attention):
        _vllm_hcu_deepseek_v4_pcp_applied = False
        forward = namespace["hcu_attention_forward"]
        _wq_b_proj = Attention._wq_b_proj.__wrapped__
        _o_proj = Attention._o_proj.__wrapped__
        _forward_prefill = Attention._forward_prefill.__wrapped__
        _fused_qnorm_rope_kv_insert = Attention._fused_qnorm_rope_kv_insert.__wrapped__
        forward_mqa = Attention.forward_mqa.__wrapped__
        head_dim = q_lora_rank = 2
        n_local_heads = padded_heads = 1
        q_norm = staticmethod(lambda qr: qr)

        def _run_parallel_input_projections(self, hidden_states):
            projected = torch.cat((hidden_states * 2, hidden_states * 3), -1)
            if with_indexer:
                return projected, hidden_states * 4, hidden_states * 5, hidden_states * 6
            return projected, None, None, None

        def _prepare_and_attn_fn(self, hidden, qr, kv, qr_scale, kv_score,
                                 indexer_kv_score, weights, positions, output):
            assert effective_pcp_world_size(pcp_size) == 1
            assert in_replicated_mtp_batch()
            assert _current_pcp_batch() is None
            assert hidden.shape[0] == positions.numel() == num_requests
            q = self._wq_b_proj(qr, qr_scale)
            value = q + kv
            if with_indexer:
                torch.testing.assert_close(kv_score, hidden * 4)
                torch.testing.assert_close(indexer_kv_score, hidden * 5)
                torch.testing.assert_close(weights, hidden * 6)
                value = value + Indexer()._wq_b_proj(qr, qr_scale)
            output.copy_(value[:, None, :])

    monkeypatch.setattr(rocm, "DeepseekV4ROCMAiterMLAAttention", DecodeAttention)
    monkeypatch.setattr(rocm, "rocm_sparse_attn_prefill",
                        rocm.rocm_sparse_attn_prefill.__wrapped__)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES", local_queries)
    assert patch.apply_to_module(rocm)

    def unexpected_gather(*args, **kwargs):
        pytest.fail("replicated decode must not issue attention PCP all-gathers")

    managers, groups = _make_managers(pcp_size)
    for group in groups:
        monkeypatch.setattr(group, "all_gather", unexpected_gather)
    # Reuse the managers across decode steps with new values to detect stale
    # inputs, and add model-input padding beyond the valid global token count.
    for step in range(2):
        batch = _make_batch([
            (f"d{idx}", [step * 100 + idx + 1], 30 + step, False)
            for idx in range(num_requests)
        ], pad_to=num_requests + 3)
        full_hidden = batch.input_ids[:num_requests, None].float().repeat(1, 2)
        expected = ((full_hidden * 7 + 14) if with_indexer
                    else (full_hidden * 5 + 7)) * 11
        layer = DecodeAttention()
        with replicated_mtp_batch_scope():
            reference = layer.forward(batch.positions[:num_requests], full_hidden)
        torch.testing.assert_close(reference, expected)
        for manager in managers:
            local = manager.partition_batch(batch, padded_num_tokens=num_requests + 3)
            assert not manager._global_has_prefill
            with glm53_pcp_step_scope():
                bind_glm53_pcp_batch(manager, local)
                actual = layer.forward(local.positions, _select(manager, full_hidden))
            assert actual.shape[0] == local.num_tokens_after_padding
            assert int(local.is_padding.sum()) == 3
            torch.testing.assert_close(actual, _select(manager, expected))
            torch.testing.assert_close(actual[~local.is_padding], reference)
            assert patch.current_pcp_batch() is None
            assert patch._LOCAL_QUERIES.get() is None
            assert not in_replicated_mtp_batch()
            assert effective_pcp_world_size(pcp_size) == pcp_size
