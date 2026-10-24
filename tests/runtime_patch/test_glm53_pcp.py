# SPDX-License-Identifier: Apache-2.0
"""Variable-width GLM5Next PCP ownership and configuration regressions."""

from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.v1 import kv_cache_interface

from test_glm52_pcp_config import _make_pcp_config
from test_glm52_pcp_manager import _make_batch, _make_managers
from vllm_hcu.patch.platform.core_fix.patch_vllm_config import _validate_hcu_pcp_scope
from vllm_hcu.patch.platform.framework_opt import patch_glm53_pcp_kv_cache
from vllm_hcu.v1.glm53_pcp import (
    _kda_forward,
    _indexer_forward,
    _prepare_attn,
    bind_glm53_pcp_batch,
    glm53_pcp_step_scope,
)


def _config(**overrides):
    values = dict(architecture="Glm5NextForConditionalGeneration", pcp=8,
                  tp=1, hybrid=True, multimodal=True)
    values.update(overrides)
    config = _make_pcp_config(**values)
    config.model_config.multimodal_config = SimpleNamespace(language_model_only=True)
    return config


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_glm53_variable_width_eager_no_mtp_contract(cp_size):
    assert _validate_hcu_pcp_scope(_config(pcp=cp_size)) is True


def test_glm53_internal_language_model_revalidation():
    config = _config(architecture="Glm5NextForCausalLM", multimodal=False)
    config.model_config.multimodal_config = None
    assert _validate_hcu_pcp_scope(config) is True


@pytest.mark.parametrize("tokens", [1, 2, 3])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_glm53_builtin_mtp_contract(tokens, cp_size):
    assert _validate_hcu_pcp_scope(
        _config(pcp=cp_size, speculative=True, num_speculative_tokens=tokens)
    ) is True


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_glm53_mtp_draft_config_is_revalidated_as_mla_pcp(cp_size):
    config = _config(
        architecture="Glm5NextMTPModel",
        pcp=cp_size,
        hybrid=False,
        multimodal=False,
        speculative=True,
        num_speculative_tokens=3,
    )
    config.model_config.multimodal_config = None
    assert _validate_hcu_pcp_scope(config) is True


def test_glm53_mtp_draft_rejects_non_mla_config():
    config = _config(
        architecture="Glm5NextMTPModel",
        hybrid=False,
        multimodal=False,
        speculative=True,
        use_mla=False,
    )
    config.model_config.multimodal_config = None
    with pytest.raises(ValueError, match="built-in MLA draft layer"):
        _validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("overrides", [
    {"speculative_method": "ngram"}, {"num_speculative_tokens": 4},
])
def test_glm53_rejects_unsupported_speculation(overrides):
    with pytest.raises(ValueError):
        _validate_hcu_pcp_scope(_config(speculative=True, **overrides))


@pytest.mark.parametrize("overrides", [
    {"tp": 2}, {"tp": 8}, {"pp": 2}, {"dp": 2}, {"dcp": 2},
    {"enforce_eager": False}, {"enable_expert_parallel": False},
])
@pytest.mark.parametrize("speculative", [False, True])
def test_glm53_relaxed_topology_admission(overrides, speculative):
    # Configuration admission does not qualify full-model execution support.
    assert _validate_hcu_pcp_scope(_config(
        speculative=speculative, **overrides,
    )) is True


@pytest.mark.parametrize("architecture", [
    "Glm5NextForConditionalGeneration", "Glm5NextForCausalLM", "Glm5NextMTPModel",
])
def test_glm53_combined_relaxed_admission_is_not_execution_qualification(architecture):
    assert _validate_hcu_pcp_scope(_config(
        architecture=architecture, tp=2, pp=2, dp=2, dcp=2,
        enforce_eager=False, enable_expert_parallel=False,
        multimodal=False, speculative=True,
    )) is True


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_glm53_pcp1_does_not_apply_pcp_restrictions(tp_size):
    assert _validate_hcu_pcp_scope(_config(
        tp=tp_size, pcp=1, enforce_eager=False, enable_expert_parallel=False,
    )) is False


def test_glm53_rejects_multimodal_inputs():
    config = _config()
    config.model_config.multimodal_config.language_model_only = False
    with pytest.raises(ValueError, match="language-model-only"):
        _validate_hcu_pcp_scope(config)


def test_glm53_replicated_mtp_skips_target_gather_and_localize():
    from vllm_hcu.model_executor.layers.attention.pcp import replicated_mtp_batch_scope
    from vllm_hcu.v1.glm53_pcp import _BATCH

    def unexpected(*args):
        raise AssertionError("replicated draft must not gather/localize target rows")

    batch = SimpleNamespace(gather=unexpected, localize=unexpected)
    hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
    positions = torch.arange(6)
    expected = hidden + 1
    kda = SimpleNamespace(_hcu_glm53_original_forward=lambda h, p: h + 1)
    indexer = SimpleNamespace(
        _hcu_glm53_original_forward=lambda h, q, p, r: h + 1
    )
    with glm53_pcp_step_scope():
        _BATCH.set(batch)
        with replicated_mtp_batch_scope():
            torch.testing.assert_close(_kda_forward(kda, hidden, positions), expected)
            torch.testing.assert_close(
                _indexer_forward(indexer, hidden, hidden, positions, None), expected
            )


@pytest.mark.parametrize("length", [1, 2, 7, 16, 35, 65])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_recurrent_state_preserves_causal_order_across_pcp_chunks(length, cp_size):
    # Cumsum stands in for a recurrent state update: running it on each PCP
    # chunk independently loses all preceding chunks, even with correct shapes.
    global_batch = _make_batch([
        ("prefill", list(range(1, length + 1)), length, True),
        ("decode", [500], 23, False),
        ("continued", list(range(101, 110)), 29, True),
    ])
    managers, groups = _make_managers(cp_size)
    local_batches = [m.partition_batch(global_batch) for m in managers]
    inputs = [b.input_ids.float().unsqueeze(1) for b in local_batches]
    expected_inputs = global_batch.input_ids[:global_batch.num_tokens].float().unsqueeze(1)
    boundaries = global_batch.query_start_loc_np

    def recurrent(value, positions):
        torch.testing.assert_close(value, expected_inputs)
        torch.testing.assert_close(positions, global_batch.positions[:value.shape[0]])
        result = torch.empty_like(value)
        for request, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            result[begin:end] = value[begin:end].cumsum(0) + request * 1000
        return result

    expected = recurrent(expected_inputs, global_batch.positions[:global_batch.num_tokens])
    outputs = []
    for manager, group, batch, value in zip(managers, groups, local_batches, inputs):
        group.gathered = inputs
        layer = SimpleNamespace(_hcu_glm53_original_forward=recurrent)
        with glm53_pcp_step_scope():
            bind_glm53_pcp_batch(manager, batch)
            outputs.append(_kda_forward(layer, value, batch.positions))
    for manager, group, output in zip(managers, groups, outputs):
        group.gathered = outputs
        torch.testing.assert_close(manager.restore_hidden_states(output), expected)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_global_recurrent_metadata_and_local_mla_metadata_do_not_alias(cp_size):
    from test_glm52_pcp_manager import _InMemoryBlockTables

    global_batch = _make_batch([("prefill", list(range(35)), 35, True)])
    tables = _InMemoryBlockTables()
    tables.allow_global_gather = True
    managers, _ = _make_managers(cp_size, block_tables=tables)
    manager = managers[min(3, cp_size - 1)]
    local_batch = manager.partition_batch(global_batch)
    local_tables, local_slots = manager.prepare_attn(local_batch)

    def group(name):
        return SimpleNamespace(kv_cache_spec=None,
                               backend=SimpleNamespace(get_name=lambda: name))

    groups = [[group("FLASHMLA_SPARSE"), group("DEEPSEEK_V32_INDEXER"),
               group("KPOOL_TAIL")]]

    def prepare(batch, mode, blocks, slots, attn_groups, config, capture, ubatch):
        return {g.backend.get_name(): SimpleNamespace(
            pcp_world_size=cp_size, batch=batch, table=blocks[0].clone(), slots=slots.clone())
            for subgroups in attn_groups for g in subgroups}

    state = SimpleNamespace(_hcu_glm53_original_prepare_attn=prepare)
    with glm53_pcp_step_scope():
        bind_glm53_pcp_batch(manager, local_batch)
        metadata = _prepare_attn(state, local_batch, None, local_tables,
                                 local_slots, groups, None)
    mla = metadata["FLASHMLA_SPARSE"]
    assert mla.batch is local_batch
    assert mla.pcp_has_global_prefill is True
    assert mla.pcp_world_size == cp_size
    for name in ("DEEPSEEK_V32_INDEXER", "KPOOL_TAIL"):
        recurrent = metadata[name]
        assert recurrent.batch is global_batch
        assert recurrent.pcp_has_global_prefill is False
        assert recurrent.pcp_world_size == 1
        assert recurrent.slots.shape[1] == 35
    torch.testing.assert_close(mla.table, local_tables[0])


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_kpool_shared_topk_buffer_is_remapped_to_local_query_order(cp_size):
    global_batch = _make_batch([
        ("prefill", list(range(1, 48)), 47, True),
        ("decode", [100], 50, False),
    ])
    managers, groups = _make_managers(cp_size)
    batches = [m.partition_batch(global_batch) for m in managers]
    inputs = [b.input_ids.float().unsqueeze(1) for b in batches]
    outputs = []
    expected = global_batch.input_ids[:global_batch.num_tokens].long().unsqueeze(1)
    for manager, group, batch, value in zip(managers, groups, batches, inputs):
        group.gathered = inputs
        buffer = torch.full((128, 1), -1, dtype=torch.int64)

        def indexer(hidden, qr, positions, rotary):
            torch.testing.assert_close(hidden.long(), expected)
            torch.testing.assert_close(qr, hidden)
            buffer[:hidden.shape[0]] = hidden.long()
            return buffer

        layer = SimpleNamespace(_hcu_glm53_original_forward=indexer,
                                topk_indices_buffer=buffer)
        with glm53_pcp_step_scope():
            bind_glm53_pcp_batch(manager, batch)
            result = _indexer_forward(layer, value, value, batch.positions, None)
            assert result is buffer
            outputs.append(result[:value.shape[0]].clone())
    for manager, group, output in zip(managers, groups, outputs):
        group.gathered = outputs
        torch.testing.assert_close(manager.restore_hidden_states(output), expected)


@pytest.fixture
def hybrid_cache_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MambaSpec:
        pass

    class KpoolTailSpec:
        pass

    monkeypatch.setattr(kv_cache_interface, "MambaSpec", MambaSpec)
    monkeypatch.setattr(kv_cache_interface, "KpoolTailSpec", KpoolTailSpec)

    module = ModuleType(patch_glm53_pcp_kv_cache.TARGET_MODULE)

    class HybridKVCacheCoordinator:
        def __init__(self, kv_cache_config, pcp_world_size):
            self.kv_cache_config = kv_cache_config
            self.pcp_world_size = pcp_world_size

    module.HybridKVCacheCoordinator = HybridKVCacheCoordinator
    assert patch_glm53_pcp_kv_cache.apply_to_module(module) is True
    assert patch_glm53_pcp_kv_cache.apply_to_module(module) is False

    hybrid_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=MambaSpec()),
            SimpleNamespace(kv_cache_spec=KpoolTailSpec()),
        ]
    )
    return HybridKVCacheCoordinator, hybrid_config, MambaSpec, KpoolTailSpec


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("head_switch", ["0", "1"])
def test_glm53_hybrid_cache_uses_explicit_ownership(hybrid_cache_patch, monkeypatch,
                                                  cp_size, head_switch):
    from vllm_hcu.v1.glm53_kda_cp_config import KDA_CP_HEAD_ENV

    monkeypatch.setenv(KDA_CP_HEAD_ENV, head_switch)
    coordinator, cache, _, _ = hybrid_cache_patch
    # Matching spec types alone must not change any model's cache policy.
    assert coordinator(cache, cp_size).pcp_world_size == cp_size
    patch_glm53_pcp_kv_cache.mark_glm53_cache_ownership(cache, _config(pcp=cp_size))
    assert coordinator(cache, cp_size).pcp_world_size == 1
    with pytest.raises(ValueError, match="width mismatch"):
        coordinator(cache, cp_size * 2)


@pytest.mark.parametrize("architecture", ["OtherHybridModel", "Glm5NextMTPModel"])
def test_unrelated_cache_is_not_marked(hybrid_cache_patch, architecture):
    coordinator, cache, _, _ = hybrid_cache_patch
    cfg = _config(pcp=4)
    patch_glm53_pcp_kv_cache.mark_glm53_cache_ownership(cache, cfg)
    assert coordinator(cache, 4).pcp_world_size == 1
    cfg.model_config.architectures = [architecture]
    patch_glm53_pcp_kv_cache.mark_glm53_cache_ownership(cache, cfg)
    assert coordinator(cache, 4).pcp_world_size == 4


def test_cache_pcp1_and_incomplete_groups_keep_original_policy(hybrid_cache_patch):
    coordinator, cache, mamba, tail = hybrid_cache_patch
    patch_glm53_pcp_kv_cache.mark_glm53_cache_ownership(cache, _config(pcp=1))
    assert coordinator(cache, 1).pcp_world_size == 1
    assert not hasattr(cache, patch_glm53_pcp_kv_cache._OWNERSHIP)
    for spec in [mamba(), tail()]:
        partial = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
        patch_glm53_pcp_kv_cache.mark_glm53_cache_ownership(partial, _config(pcp=4))
        assert coordinator(partial, 4).pcp_world_size == 4
    patch_glm53_pcp_kv_cache.mark_glm53_cache_ownership(cache, _config(pcp=4))
    cache.kv_cache_groups = []
    with pytest.raises(ValueError, match="state groups changed"):
        coordinator(cache, 4)


def test_block_size_resolution_stamps_ownership_before_coordinator(hybrid_cache_patch):
    from vllm_hcu.patch.platform.framework_opt import patch_pcp_kv_cache_utils

    coordinator, cache, _, _ = hybrid_cache_patch
    module = ModuleType(patch_pcp_kv_cache_utils.TARGET_MODULE)
    calls = []

    def resolve(kv_cache_config, vllm_config):
        calls.append(coordinator(kv_cache_config, 4).pcp_world_size)
        return (16, 64)

    module.resolve_kv_cache_block_sizes = resolve
    assert patch_pcp_kv_cache_utils.apply_to_module(module) is True
    assert patch_pcp_kv_cache_utils.apply_to_module(module) is False
    assert module.resolve_kv_cache_block_sizes(cache, _config(pcp=4)) == (16, 64)
    assert calls == [1]


def test_non_glm_block_size_resolution_does_not_require_model_fields(hybrid_cache_patch):
    coordinator, cache, _, _ = hybrid_cache_patch
    patch_glm53_pcp_kv_cache.mark_glm53_cache_ownership(cache, SimpleNamespace())
    assert coordinator(cache, 4).pcp_world_size == 4
