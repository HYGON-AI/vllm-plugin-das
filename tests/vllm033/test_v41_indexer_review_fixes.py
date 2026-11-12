# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for four review findings on the V4.1 indexer boundary.

Each case is written so it fails on the pre-fix tree:

* D3 -- the CPU candidate path must refuse instead of tripping the upstream
  ``assert logits.is_cuda`` inside the kernel.
* D8 -- the decode staging pool must be reserved during the startup profile and
  then reused, instead of being allocated fresh on the decode hot path.
* D9 -- ``check_health()`` must re-verify the pinned implementation instead of
  returning the digest captured at activation.
* D4 -- the three names a caller historically patched must still change what
  executes after the implementation moved into the adapter.

CPU only; no Triton JIT, no device.
"""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_adapter import v41_fp8_indexer as impl
from hcu_adapter.extensions import ExtensionCommitError
from hcu_vllm_das import fp4_indexer_lightop as reader
from hcu_vllm_das import v41_indexer as scorer
from hcu_vllm_das import v41_indexer_activation as activation

from test_fp4_full_admission import config


@pytest.fixture(autouse=True)
def isolate_module_state(monkeypatch):
    """Keep the module globals these cases touch from leaking between them."""
    monkeypatch.setattr(activation, '_state', None)
    monkeypatch.setattr(activation, '_failure', None)
    # Only reset the pool when it exists: on the pre-fix tree it does not, and
    # raising here would turn every case into a collection error instead of the
    # individual failure that proves the fix is what makes them pass.
    if hasattr(reader, '_reserved_pools'):
        monkeypatch.setattr(reader, '_reserved_pools', {})
    elif hasattr(reader, '_reserved_pool'):
        monkeypatch.setattr(reader, '_reserved_pool', None)
    original = dict(impl._resolvers)
    try:
        yield
    finally:
        impl._resolvers.clear()
        impl._resolvers.update(original)


def _bind(**resolvers):
    """Bind zero-argument resolvers, which is the contract `_resolve` expects."""
    for name, value in resolvers.items():
        impl._resolvers[name] = value


# ---------------------------------------------------------------- D3


def test_cpu_candidate_path_refuses_with_a_named_error():
    """Upstream candidate kernels are CUDA-only; CPU must refuse explicitly."""
    select = impl.build()['select']
    logits = torch.zeros((2, 8), dtype=torch.float32)
    starts = torch.zeros(2, dtype=torch.int32)
    ends = torch.full((2,), 8, dtype=torch.int32)
    out = torch.zeros((2, 4), dtype=torch.int32)
    candidates = torch.zeros((2, 2), dtype=torch.int32)

    with pytest.raises(impl.V41IndexerError) as refusal:
        select(logits, starts, ends, out, candidates, 4, False)
    assert 'CUDA' in str(refusal.value)


def test_cpu_candidate_path_still_works_when_the_kernel_is_substituted():
    """A doubled kernel must keep running: the refusal is about the real one."""
    select = impl.build()['select']
    calls = []

    def fake_mask(*args):
        calls.append('mask')

    def fake_select(*args):
        calls.append('select')

    _bind(candidate_mask=lambda: fake_mask, candidate_select=lambda: fake_select)
    logits = torch.zeros((2, 8), dtype=torch.float32)
    starts = torch.zeros(2, dtype=torch.int32)
    ends = torch.full((2,), 8, dtype=torch.int32)
    out = torch.zeros((2, 4), dtype=torch.int32)
    candidates = torch.zeros((2, 2), dtype=torch.int32)

    select(logits, starts, ends, out, candidates, 4, False)
    select(logits, starts, ends, out, candidates, 4, True)
    assert calls == ['mask', 'select']


def test_cpu_without_candidates_never_touches_the_candidate_kernels():
    """No candidate bank means no candidate kernel, on any device."""
    select = impl.build()['select']
    touched = []
    _bind(candidate_mask=lambda: touched.append('mask'),
          candidate_select=lambda: touched.append('select'))
    out = torch.zeros((1, 4), dtype=torch.int32)
    select(torch.zeros((1, 8), dtype=torch.float32),
           torch.zeros(1, dtype=torch.int32),
           torch.full((1,), 8, dtype=torch.int32), out, None, 0, False)
    assert touched == []


# ---------------------------------------------------------------- D8


def staged_config():
    """Config fixture plus the scheduler view the staging plan reads."""
    cfg = config(ratio=1)
    cfg.scheduler_config = NS(max_num_seqs=32)
    return cfg


def test_staging_plan_is_empty_for_64_record_pages():
    """64-record pages stage zero-copy, so nothing needs reserving."""
    assert reader.staging_plan(staged_config(), 4096, 1) is None


def test_staging_plan_sizes_the_pool_and_transient_scratch():
    """A 64-block cache at ratio 2 yields 32-record pages, which do stage."""
    cfg = staged_config()
    assert cfg.cache_config.block_size == 64
    plan = reader.staging_plan(cfg, 8192, 2)
    assert plan is not None
    assert plan['page_records'] == 32
    assert plan['pages_per_row'] == (8192 + 63) // 64
    assert plan['shape'] == (plan['requests'], plan['pages_per_row'],
                             reader.STAGING_PAGE_BYTES)
    assert plan['transient'] > 0


def test_profile_pass_reserves_the_staging_pool(monkeypatch):
    """The profile branch must reserve staging, not only the FP32 logits."""
    cfg = staged_config()
    calls = []

    class Workspace:
        def get_simultaneous(self, *shapes):
            calls.append(('simultaneous', shapes))
            return [torch.zeros(shapes[0][0], dtype=torch.float32)]

        def get_persistent(self, key, shape, dtype, **kwargs):
            calls.append(('persistent', key, shape))
            return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(reader, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setattr(reader, 'get_forward_context',
                        lambda: NS(attn_metadata=None))

    import vllm.v1.worker.workspace as workspace
    monkeypatch.setattr(workspace, 'is_workspace_manager_initialized',
                        lambda: True)
    monkeypatch.setattr(workspace, 'current_workspace_manager', lambda: Workspace())

    layer = NS(max_model_len=8192, compress_ratio=2, topk_indices_buffer='out')
    reader.execute(layer, torch.zeros((4, 128)), None, None)
    plan_records = reader.staging_plan(cfg, 8192, 2)['page_records']

    kinds = [entry[0] for entry in calls]
    assert 'persistent' in kinds, 'profile pass did not reserve the staging pool'
    pool = reader._reserved_pools[plan_records]
    assert pool.shape[2] == reader.STAGING_PAGE_BYTES


def test_staging_reuses_the_reserved_pool():
    """The decode step must write into the reserved pool, not allocate anew."""
    shape = (2, 2, reader.STAGING_PAGE_BYTES)
    reserved = torch.zeros(shape, dtype=torch.uint8)
    reader._reserved_pools[32] = reserved
    cache = NS(shape=(1, 32, 68), device=torch.device('cpu'))
    got = reader._staging_target(cache, 2, 2)
    assert got is reserved


def test_staging_falls_back_when_the_request_exceeds_the_reservation():
    """An oversized request stays correct instead of writing out of bounds."""
    reader._reserved_pools[32] = torch.zeros((1, 1, reader.STAGING_PAGE_BYTES),
                                             dtype=torch.uint8)
    cache = NS(shape=(1, 32, 68), device=torch.device('cpu'))
    got = reader._staging_target(cache, 4, 4)
    assert got.shape == (4, 4, reader.STAGING_PAGE_BYTES)
    assert got is not reader._reserved_pools[32]


def test_mixed_compress_ratios_get_separate_pools():
    """ratio-1 and ratio-2 layers coexist, so one fixed key cannot serve both.

    Measured on the real formula: at block_size 32 both ratios stage (32 and 16
    records); at 256 both stage again (256 and 128). A single fixed workspace key
    therefore made the second layer's request a shape mismatch, which the
    manager rejects with ValueError during the profile pass -- a startup failure,
    not a slow path.
    """
    cfg = staged_config()
    cfg.cache_config.block_size = 32
    ratio1 = reader.staging_plan(cfg, 8192, 1)
    ratio2 = reader.staging_plan(cfg, 8192, 2)
    assert ratio1 is not None and ratio2 is not None
    assert ratio1['page_records'] == 32 and ratio2['page_records'] == 16
    assert reader.staging_key(ratio1) != reader.staging_key(ratio2)

    first = torch.zeros(ratio1['shape'], dtype=torch.uint8)
    second = torch.zeros(ratio2['shape'], dtype=torch.uint8)
    reader._reserved_pools[ratio1['page_records']] = first
    reader._reserved_pools[ratio2['page_records']] = second

    cache1 = NS(shape=(1, 32, 68), device=torch.device('cpu'))
    cache2 = NS(shape=(1, 16, 68), device=torch.device('cpu'))
    # Ask for exactly the reserved geometry; the pools were sized from the plan.
    rows, per_row = ratio1['requests'], ratio1['pages_per_row']
    assert reader._staging_target(cache1, rows, per_row) is first
    rows, per_row = ratio2['requests'], ratio2['pages_per_row']
    assert reader._staging_target(cache2, rows, per_row) is second

def test_digest_covers_the_adapter_triton_implementation():
    """The GPU FP4 path runs the adapter Triton kernels; they must be pinned."""
    import hcu_adapter.v41_fp8_indexer_triton as triton

    paths = [Path(p).resolve() for p in activation._digest_paths()]
    assert Path(triton.__file__).resolve() in paths


def _healthy_state():
    """A published FP4 state whose only moving part is the digest signature."""
    digest = activation._compute_digest()
    native = NS(forward_hip=None, name='SparseAttnIndexer')
    state = dict(enabled=True, fp4=True, digest=digest,
                 signature=activation._digest_signature(),
                 registry={}, owner=impl.resolver_owner(), forward=None,
                 native=native, **{'class': NS(name=native.name)})

    class Registry(dict):
        pass

    registry = Registry({'SparseAttnIndexer': state['class']})
    state['registry'] = registry
    return state, registry


def _bind_health_host(monkeypatch, registry, state):
    """Stand up just enough of the host for `check_health` to run on CPU.

    `environment_variables` is patched item-wise on purpose: replacing the whole
    mapping makes every unrelated env lookup raise, which hides the real verdict.
    """
    import vllm.envs as envs
    from vllm.model_executor import custom_op

    from hcu_vllm_das import fp4_indexer, fp4_packer

    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'lightop')
    monkeypatch.setattr(activation, '_is_hcu', lambda: True)
    monkeypatch.setattr(custom_op, 'op_registry_oot', registry)
    monkeypatch.setitem(envs.environment_variables, activation._FACTOR,
                        activation.policy_factor)
    monkeypatch.setattr(impl, 'resolver_owner', lambda: state['owner'])
    # The FP4 leaf contracts are device work; they are covered by their own
    # suites and are not what these cases are about.
    monkeypatch.setattr(fp4_packer, 'check_health', lambda: None)
    monkeypatch.setattr(reader, 'load_providers', lambda: None)
    monkeypatch.setattr(fp4_indexer, 'check_native_contract', lambda: None)


def test_health_recomputes_when_the_source_changes(monkeypatch):
    """A changed implementation after activation must be refused, not reused."""
    state, registry = _healthy_state()
    activation._state = state
    _bind_health_host(monkeypatch, registry, state)
    monkeypatch.setattr(activation, '_digest_signature',
                        lambda: (('moved', 1, 2),))
    monkeypatch.setattr(activation, '_compute_digest', lambda: 'different')

    with pytest.raises(ExtensionCommitError):
        activation.check_health()


def test_health_is_quiet_when_the_signature_is_unchanged(monkeypatch):
    """The hot path must not re-hash: an unchanged signature short-circuits."""
    state, registry = _healthy_state()
    activation._state = state
    _bind_health_host(monkeypatch, registry, state)
    recomputed = []
    monkeypatch.setattr(activation, '_compute_digest',
                        lambda: recomputed.append(1) or state['digest'])

    activation.check_health()
    assert recomputed == [], 'unchanged signature must not trigger a re-hash'


def test_health_refuses_a_replaced_binding_module(monkeypatch):
    """A reloaded binding module leaves resolvers pointing at dead code."""
    state, registry = _healthy_state()
    activation._state = state
    _bind_health_host(monkeypatch, registry, state)
    monkeypatch.setattr(impl, 'resolver_owner', lambda: 'some.other.module')

    with pytest.raises(ExtensionCommitError):
        activation.check_health()


# ---------------------------------------------------------------- D4


def _run_profile_path(indexer):
    """Call the reference op through its profile branch with doubles bound."""
    return indexer(torch.zeros((1, 8)), 'cache', torch.zeros((1, 64)),
                   torch.zeros((1, 2, 128)), torch.zeros((1, 2)), 4, 128, 4096,
                   4, torch.zeros((1, 4), dtype=torch.int32), 1)


def test_patched_native_indexer_is_the_one_that_runs(monkeypatch):
    """Patching the plugin attribute must change the adapter's profile path."""
    calls = []
    monkeypatch.setattr(scorer, 'native_indexer',
                        lambda *a, **k: calls.append('patched') or 'patched')
    _bind(forward_context=lambda: NS(attn_metadata=None),
          layer_name=lambda: (lambda prefix: prefix),
          native_indexer=lambda: scorer.native_indexer)

    assert _run_profile_path(impl.build()['indexer']) == 'patched'
    assert calls == ['patched']


def test_patched_layer_name_is_the_one_that_runs(monkeypatch):
    """Layer-name resolution is read through the resolver at call time."""
    seen = []
    monkeypatch.setattr(scorer, '_resolve_layer_name',
                        lambda prefix: seen.append(prefix) or 'resolved')
    _bind(forward_context=lambda: NS(attn_metadata=None),
          layer_name=lambda: scorer._resolve_layer_name,
          native_indexer=lambda: (lambda *a, **k: 'native'))

    assert _run_profile_path(impl.build()['indexer']) == 'native'
    assert seen == ['cache']


def test_patched_mqa_logits_is_read_at_call_time(monkeypatch):
    """The prefill scorer must also come from the resolver, not a closure."""
    calls = []
    monkeypatch.setattr(scorer, 'fp8_mqa_logits_torch',
                        lambda *a, **k: calls.append('patched') or 'logits')
    _bind(mqa_logits=lambda: scorer.fp8_mqa_logits_torch)

    assert impl._resolve('mqa_logits') is scorer.fp8_mqa_logits_torch
    assert calls == []


# ---------------------------------------------------------------- review round 2
# The second review found that two fixes landed on functions the production path
# never calls, one guard was too weak, and one new startup failure was introduced.
# These cases target the real entry points.


def test_cpu_refusal_survives_a_wrapped_cuda_only_helper():
    """A decorated/partial upstream helper must still be refused on CPU.

    Object identity was too weak: a wrapper, decorator or functools.partial
    around the upstream kernel slipped past it and let the bare
    `assert logits.is_cuda` fire.
    """
    import functools

    select = impl.build()['select']
    upstream = impl.build()['select']
    from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb

    wrapped = functools.partial(cb.apply_candidate_mask)
    _bind(candidate_mask=lambda: wrapped)

    logits = torch.zeros((2, 8), dtype=torch.float32)
    starts = torch.zeros(2, dtype=torch.int32)
    ends = torch.full((2,), 8, dtype=torch.int32)
    out = torch.zeros((2, 4), dtype=torch.int32)
    candidates = torch.zeros((2, 2), dtype=torch.int32)
    with pytest.raises(impl.V41IndexerError):
        select(logits, starts, ends, out, candidates, 4, False)


def test_cpu_assertion_from_an_unidentified_helper_becomes_the_named_error():
    """A CUDA-only helper we cannot unwrap must not leak a bare AssertionError."""
    select = impl.build()['select']

    def disguised(*args):
        raise AssertionError

    _bind(candidate_mask=lambda: disguised)
    logits = torch.zeros((2, 8), dtype=torch.float32)
    with pytest.raises(impl.V41IndexerError):
        select(logits, torch.zeros(2, dtype=torch.int32),
               torch.full((2,), 8, dtype=torch.int32),
               torch.zeros((2, 4), dtype=torch.int32),
               torch.zeros((2, 2), dtype=torch.int32), 4, False)


def _pcp_config(width=8):
    """A PCP-looking config, shaped like the PCP replay's own fixture."""
    return NS(
        model_config=NS(enforce_eager=True,
                        hf_config=NS(model_type='deepseek_v41',
                                     engram_layer_ids=[], vision_n_layers=0)),
        parallel_config=NS(prefill_context_parallel_size=width,
                           decode_context_parallel_size=1,
                           pipeline_parallel_size=1, data_parallel_size=1,
                           tensor_parallel_size=1, enable_dbo=False,
                           enable_eplb=False),
        use_v2_model_runner=True, speculative_config=None, lora_config=None,
        kv_transfer_config=None, kernel_config=NS(monokernel=False),
        cache_config=NS(swa_bounded_replay=False),
        attention_config=NS(backend=None, indexer_sparse_logits=False,
                            resolve_indexer_kv_dtype=lambda _: 'fp8'))


def test_plugin_pcp_selector_applies_the_mode_gate(monkeypatch):
    """The decoder walks the plugin's local selector, not pcp_fork's.

    `_DECODER_INIT` substitutes this function into the reviewed native
    constructor, so a gate added only to `pcp_fork.select_attention()` never ran
    on the production path. This is the regression that mattered: with stock,
    the local selector must not return the owned class.
    """
    from hcu_vllm_das.models import v41_pcp as models
    from hcu_adapter.kernels import pcp_fork

    cfg = _pcp_config()
    assert models.validate_config(cfg), 'fixture must look like PCP'

    owned = []
    monkeypatch.setattr(pcp_fork, 'selected_mode', lambda: 'native')
    monkeypatch.setattr(models, 'attention_class',
                        lambda: owned.append('owned') or 'OWNED')
    assert models.select_attention(cfg) == 'OWNED'
    assert owned == ['owned']

    monkeypatch.setattr(pcp_fork, 'selected_mode', lambda: 'stock')
    owned.clear()
    # Under stock the selector delegates to core's native picker. On CPU that
    # picker raises NotImplementedError (PCP is CUDA-only upstream), which is
    # exactly the proof we want: it reached native code, not the owned class.
    try:
        result = models.select_attention(cfg)
    except NotImplementedError:
        result = None
    assert result != 'OWNED', 'stock must not construct the owned class'
    assert owned == [], 'stock must not even ask for the owned class'


def test_seam_check_reports_a_missing_adapter_module(monkeypatch):
    """A whole missing module must be a named contract error, not ModuleNotFound.

    On a mixed revision pair `dsv41_dspark_registration` does not exist yet; a
    module-scope import raised before the check could run, which is the very
    failure the check was added to replace.
    """
    from hcu_vllm_das import _seams
    import importlib

    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name == 'hcu_adapter.dsv41_dspark_registration':
            raise ModuleNotFoundError(
                "No module named 'hcu_adapter.dsv41_dspark_registration'",
                name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, 'import_module', fake)
    with pytest.raises(_seams.AdapterContractError) as refusal:
        _seams._seam_modules()
    assert 'dsv41_dspark_registration' in str(refusal.value)


def test_seam_table_covers_the_ct_activation_binder():
    """CT activation calls set_provider_binder at import; the table must name it."""
    from hcu_vllm_das import _seams

    modules = _seams._seam_modules()
    names = {(module.__name__, name) for module, name, _ in
             _seams._required_seams(modules)}
    assert ('hcu_adapter.dsv41_activation', 'set_provider_binder') in names
    purposes = [purpose for _, _, purpose in _seams._required_seams(modules)]
    assert any('CT W4A8' in purpose for purpose in purposes)
