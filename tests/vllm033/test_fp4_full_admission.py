# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FP4 preallocation policy and one-owner publication, CPU only."""
from types import SimpleNamespace as NS

import pytest
import torch
from hcu_adapter import hcu_env
from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
from hcu_adapter.fp4_indexer import applicable, selected, validate_config
from hcu_vllm_das import v41_indexer_activation as activation


def attention_config(indexer_kv_dtype='auto'):
    """Stand-in for core's AttentionConfig on the one method admission reads.

    ``fp4_indexer.applicable()`` asks core's
    ``resolve_indexer_kv_dtype('fp8')`` whether this deployment actually uses an
    MXFP4 indexer cache. Core resolves ``auto`` to ``fp8``, so the fake has to
    reproduce that resolution instead of only carrying the raw attribute.
    """
    return NS(indexer_kv_dtype=indexer_kv_dtype, indexer_sparse_logits=False,
              resolve_indexer_kv_dtype=lambda default: (
                  default if indexer_kv_dtype == 'auto' else indexer_kv_dtype))


def config_with(**overrides):
    """Same fake config with one field replaced (attention config included)."""
    cfg = config()
    if 'indexer_kv_dtype' in overrides:
        cfg.attention_config = attention_config(overrides.pop('indexer_kv_dtype'))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def config(ratio=1):
    hf = NS(model_type='deepseek_v41', compress_ratios=[0, ratio], index_head_dim=128,
            index_n_heads=32, qk_rope_head_dim=64, kv_source_layer_ids=[1],
            index_source_layer_ids=[1])
    return NS(model_config=NS(hf_text_config=hf, hf_config=hf, enforce_eager=True,
                             max_model_len=4096),
              parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                                 prefill_context_parallel_size=1, decode_context_parallel_size=1,
                                 data_parallel_size=1, enable_dbo=False, enable_eplb=False),
              cache_config=NS(block_size=64*ratio), attention_config=attention_config(),
              kernel_config=NS(monokernel=False, enable_jit_warmup=False),
              speculative_config=None, lora_config=None, kv_transfer_config=None,
              use_v2_model_runner=True)


@pytest.fixture
def optin(monkeypatch):
    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'lightop')
    monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'triton')
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'stock')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    return config()


def test_off_preserves_default_fp8(monkeypatch):
    """`selected()` stays on by default; what keeps FP8 off the FP4 path is
    `applicable()`, which asks core for the actual indexer cache dtype.

    `validate_config` deliberately keeps validating the FP4 plan whenever the
    knob is on: it is the admission of the FP4 deployment itself, and the
    callers (`v41_indexer`, `models/v41_pcp`) are the ones that ask
    `applicable()` first.
    """
    monkeypatch.delenv('HCU_VLLM_FP4_INDEXER_IMPL', raising=False)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    assert selected()
    assert not applicable(config())
    assert not applicable(config_with(indexer_kv_dtype='fp8'))
    assert applicable(config_with(indexer_kv_dtype='mxfp4'))
    assert validate_config(config_with(indexer_kv_dtype='mxfp4')).ratios == (1,)


def test_config_file_optin_uses_real_resolver(monkeypatch):
    monkeypatch.delenv('HCU_VLLM_FP4_INDEXER_IMPL', raising=False)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    from hcu_adapter import hcu_config
    monkeypatch.setattr(hcu_config, 'lookup', lambda key: 'lightop' if key == 'HCU_VLLM_FP4_INDEXER_IMPL' else None)
    assert hcu_env.resolve('HCU_VLLM_FP4_INDEXER_IMPL')[1] == 'config'
    assert selected()


@pytest.mark.parametrize('ratio', [1, 2])
def test_optin_freezes_page64_plan(optin, ratio):
    cfg = config(ratio)
    plan = validate_config(cfg)
    assert plan.ratios == (ratio,)
    assert plan.page_records == 64 and plan.record_bytes == 68


@pytest.mark.parametrize('model_type', ['deepseek_v41', 'deepseek_v41_text'])
def test_exact_core_v41_identifiers_are_admitted(optin, model_type):
    optin.model_config.hf_text_config.model_type = model_type
    assert validate_config(optin).page_records == 64


@pytest.mark.parametrize('model_type', ['deepseek_v4', 'deepseek_v41_other', 'deepseek_v41_text_extra'])
def test_other_models_and_prefix_lookalikes_are_rejected(optin, model_type):
    optin.model_config.hf_text_config.model_type = model_type
    with pytest.raises(ValueError, match='DeepSeek-V4.1 required'):
        validate_config(optin)


def test_real_text_identity_allows_mixed_ratio_staging_domain(optin):
    optin.model_config.hf_text_config.model_type = 'deepseek_v41_text'
    optin.model_config.hf_text_config.compress_ratios = [0, 1, 2]
    plan = validate_config(optin)
    assert plan.ratios == (1, 2)
    assert 'DeepSeek-V4.1 required' not in str(plan)


def test_kill_wins_before_model_selection(optin, monkeypatch):
    monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_fp4_indexer')
    assert not selected()
    assert validate_config(optin) is None


@pytest.mark.parametrize('field,value', [
    ('tensor_parallel_size', 2), ('pipeline_parallel_size', 2),
    ('prefill_context_parallel_size', 8), ('decode_context_parallel_size', 2),
    ('data_parallel_size', 2), ('enable_dbo', True), ('enable_eplb', True)])
def test_topology_refused_before_allocation(optin, field, value):
    setattr(optin.parallel_config, field, value)
    with pytest.raises(ValueError, match='before allocation'):
        validate_config(optin)


@pytest.mark.parametrize('ratios,block', [([0, 4], 256), ([0], 64)])
def test_ratio_domain_refused(optin, ratios, block):
    optin.model_config.hf_text_config.compress_ratios = ratios
    optin.cache_config.block_size = block
    with pytest.raises(ValueError, match='before allocation'):
        validate_config(optin)


@pytest.mark.parametrize('ratios,block', [([0, 1, 2], 64), ([0, 1, 2], 128),
                                          ([1], 32)])
def test_mixed_ratio_and_variable_physical_pages_admitted(optin, ratios, block):
    optin.model_config.hf_text_config.compress_ratios = ratios
    optin.cache_config.block_size = block
    assert validate_config(optin).ratios == tuple(sorted(set(ratios) - {0}))


@pytest.mark.parametrize('field,value', [('lora_config', object()),
                                       ('use_v2_model_runner', False)])
def test_unsupported_deployment_refused(optin, field, value):
    setattr(optin, field, value)
    with pytest.raises(ValueError, match='before allocation'):
        validate_config(optin)


def test_pd_connector_is_admitted(optin):
    """PD/KV-transfer is no longer rejected: the FP4 page format travels as
    opaque block bytes, so an identically configured prefill/decode pair is
    expected to transfer unchanged. Still unmeasured, so this asserts admission
    only, not that the transfer works."""
    optin.kv_transfer_config = object()
    assert validate_config(optin).ratios == (1,)


def test_lora_still_refused(optin):
    optin.lora_config = object()
    with pytest.raises(ValueError, match='LoRA'):
        validate_config(optin)


@pytest.mark.parametrize('warmup', [True, False])
def test_jit_warmup_does_not_block_admission(optin, warmup):
    """`register_warmup()` only queues a compile key, so the owner kernels are
    still unhashed when the packer rebinds them; `_require_cold()` is the
    runtime judge. Admission must therefore accept either value."""
    optin.kernel_config.enable_jit_warmup = warmup
    assert validate_config(optin).ratios == (1,)


def test_monokernel_still_refused(optin):
    optin.kernel_config.monokernel = True
    with pytest.raises(ValueError, match='monokernel'):
        validate_config(optin)


def _compilation(mode='NONE', full=False, piecewise=True):
    return NS(mode=NS(name=mode), cudagraph_mode=NS(
        has_full_cudagraphs=lambda: full,
        has_piecewise_cudagraphs=lambda: piecewise))


def test_breakable_piecewise_graph_is_admitted(optin, monkeypatch):
    from hcu_vllm_das import v41_indexer_graph
    optin.model_config.enforce_eager = False
    optin.compilation_config = _compilation()
    monkeypatch.setattr(v41_indexer_graph, 'graph_capture_allowed', lambda cfg: True)
    assert validate_config(optin).page_records == 64


def test_full_cudagraph_is_refused(optin):
    optin.model_config.enforce_eager = False
    optin.compilation_config = _compilation(full=True)
    with pytest.raises(ValueError, match='Graph'):
        validate_config(optin)


def test_legacy_inductor_piecewise_is_refused(optin):
    optin.model_config.enforce_eager = False
    optin.compilation_config = _compilation(mode='VLLM_COMPILE')
    with pytest.raises(ValueError, match='Graph'):
        validate_config(optin)


def test_graph_and_sparse_logits_refused(optin):
    optin.model_config.enforce_eager = False
    optin.attention_config.indexer_sparse_logits = True
    with pytest.raises(ValueError, match='Graph'):
        validate_config(optin)


@pytest.mark.parametrize('method,n,adaptive,ok', [('dspark', 5, False, True),
    ('dspark', 5, True, False), ('eagle', 5, False, False), ('dspark', 3, False, False)])
def test_speculation_domain(optin, method, n, adaptive, ok):
    optin.speculative_config = NS(method=method, num_speculative_tokens=n, enable_adaptive_verification=adaptive)
    if ok:
        assert validate_config(optin)
    else:
        with pytest.raises(ValueError, match='DSpark'):
            validate_config(optin)


def test_c4_conflict_is_explicit(optin, monkeypatch):
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'native')
    with pytest.raises(ValueError, match='C4 Adapter OOT'):
        validate_config(optin)


@pytest.mark.parametrize('setting', ['stock', 'disabled'])
def test_missing_lowering_refused(optin, monkeypatch, setting):
    if setting == 'stock':
        monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'stock')
    else:
        monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_fp4_packer')
    with pytest.raises(ValueError, match='lowering required'):
        validate_config(optin)


@pytest.fixture
def isolated_owner(monkeypatch):
    from vllm import envs
    from vllm.model_executor import custom_op
    from hcu_vllm_das import v41_indexer as scorer
    monkeypatch.setattr(activation, '_state', None)
    monkeypatch.setattr(activation, '_failure', None)
    monkeypatch.setattr(activation, '_is_hcu', lambda: True)
    monkeypatch.setattr(activation, '_publishing', False)
    monkeypatch.setattr(envs, 'environment_variables', dict(envs.environment_variables))
    envs.environment_variables.pop(activation._FACTOR, None)
    monkeypatch.setattr(custom_op, 'op_registry_oot', {})
    monkeypatch.setattr(scorer, 'register_op', lambda: None)
    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'stock')
    return custom_op, scorer


def test_only_existing_oot_owner_registers_when_fp8_flag_off(isolated_owner, monkeypatch):
    custom_op, scorer = isolated_owner
    monkeypatch.setenv(activation._FLAG, '0')
    activation.activate_extension(ExtensionContext('0.1.0', '0.33.0'))
    assert custom_op.op_registry_oot['SparseAttnIndexer'] is scorer.HcuV41SparseAttnIndexer
    activation.check_health()


def test_third_party_oot_owner_is_not_overwritten(isolated_owner):
    custom_op, _ = isolated_owner
    foreign = type('Foreign', (), {})
    custom_op.op_registry_oot['SparseAttnIndexer'] = foreign
    with pytest.raises(RuntimeError, match='foreign OOT owner'):
        activation.activate_extension(ExtensionContext('0.1.0', '0.33.0'))
    assert custom_op.op_registry_oot['SparseAttnIndexer'] is foreign


def test_postpublication_policy_change_latches_failure(isolated_owner, monkeypatch):
    from vllm import config as core_config
    cfg = config_with(indexer_kv_dtype='mxfp4')
    monkeypatch.setattr(core_config, 'get_current_vllm_config_or_none', lambda: cfg)
    activation.activate_extension(ExtensionContext('0.1.0', '0.33.0'))
    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'lightop')
    with pytest.raises(ExtensionCommitError):
        activation.check_health()
    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'stock')
    with pytest.raises(ExtensionCommitError):
        activation.check_health()


def test_provider_missing_rejected_without_allocating(optin, monkeypatch):
    from hcu_vllm_das import fp4_indexer_lightop as reader
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(reader, 'import_module', missing)
    with pytest.raises(ModuleNotFoundError):
        reader.load_providers()


def test_provider_abi_drift_rejected(optin, monkeypatch):
    from hcu_vllm_das import fp4_indexer_lightop as reader
    monkeypatch.setattr(reader, 'import_module', lambda name: NS(dsv41_gather_index_k_fp4=lambda x: None))
    with pytest.raises(RuntimeError, match='ABI changed'):
        reader.load_providers()
