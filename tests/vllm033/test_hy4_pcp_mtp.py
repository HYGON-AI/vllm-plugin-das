# SPDX-License-Identifier: Apache-2.0
"""Exercise native draft sampling before the next speculative cache update."""
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import hy4_pcp_mtp as owned
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def config(pcp=16, method='mtp', eager=True):
    return NS(model_config=NS(hf_config=NS(model_type='hy_v4'), enforce_eager=eager),
        use_v2_model_runner=True,
        parallel_config=NS(prefill_context_parallel_size=pcp,
            tensor_parallel_size=1, pipeline_parallel_size=1,
            data_parallel_size=1, decode_context_parallel_size=1),
        speculative_config=NS(method=method, draft_sample_method='greedy'))


class Group:
    world_size = 16
    rank_in_group = 0
    canonical = None
    calls = 0

    def broadcast(self, tensor, src=0):
        assert src == 0
        self.calls += 1
        if self.rank_in_group == 0:
            self.canonical = tensor.clone()
        else:
            tensor.copy_(self.canonical)
        return tensor


def speculator():
    obj = object.__new__(MTPSpeculator)
    obj.use_local_argmax_reduction = False
    obj.acceptance_estimator = None
    obj.draft_watermarker = None
    obj.compute_draft_logits = lambda hidden, step: hidden
    return obj


def draw(obj, logits):
    ids = torch.zeros(len(logits), dtype=torch.int32)
    return obj.sample_draft(logits, ids, ids, ids.float(), ids.long(), ids, None)


def test_each_native_draw_is_canonical_before_next_draft_state(monkeypatch):
    group = Group()
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: group)
    primary, replica = speculator(), speculator()
    # Native greedy draws can disagree; accepted-length bookkeeping must never
    # consume two different draft sequences for one replicated PCP request.
    a = torch.tensor([[0., 2., 1.]])
    b = torch.tensor([[0., 1., 2.]])
    assert draw(primary, a).item() == 1
    assert draw(replica, b).item() == 2
    owned.bind_speculator(primary, config())
    owned.bind_speculator(replica, config())
    histories = [[], []]
    for step in range(3):
        for rank, obj in enumerate((primary, replica)):
            group.rank_in_group = rank
            token = draw(obj, a if rank == 0 else b)
            # This models the next forward's input/cache write. Broadcasting
            # only the completed proposal would be too late for this state.
            histories[rank].append(token.item())
        assert histories[0] == histories[1]
    assert group.calls == 6
    assert DraftModelSpeculator.sample_draft is MTPSpeculator.sample_draft


def test_binding_is_idempotent_and_preserves_native_method(monkeypatch):
    group = Group()
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: group)
    obj = speculator()
    original = MTPSpeculator.sample_draft
    owned.bind_speculator(obj, config())
    bound = obj.sample_draft
    owned.bind_speculator(obj, config())
    assert obj.sample_draft is bound
    assert MTPSpeculator.sample_draft is original
    assert isinstance(obj, MTPSpeculator)
    assert draw(obj, torch.tensor([[1., 0., 0.]])).item() == 0
    assert group.calls == 1


def test_non_pcp_and_no_mtp_keep_native_instance_untouched(monkeypatch):
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: pytest.fail('queried group'))
    for cfg in (config(pcp=1), config(method='ngram')):
        obj = speculator()
        owned.bind_speculator(obj, cfg)
        assert 'sample_draft' not in obj.__dict__


@pytest.mark.parametrize('change', ['graph', 'tp', 'probabilistic'])
def test_unsupported_execution_refuses_before_binding(monkeypatch, change):
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: pytest.fail('queried group'))
    cfg = config()
    if change == 'graph':
        cfg.model_config.enforce_eager = False
    elif change == 'tp':
        cfg.parallel_config.tensor_parallel_size = 2
    else:
        cfg.speculative_config.draft_sample_method = 'probabilistic'
    obj = speculator()
    with pytest.raises(ValueError):
        owned.bind_speculator(obj, cfg)
    assert 'sample_draft' not in obj.__dict__


def test_public_post_load_callback_binds_only_loaded_speculator(monkeypatch):
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: Group())
    runner = NS(speculator=speculator())
    ctx = NS(worker=NS(model_runner=runner), vllm_config=config(), load_dummy_weights=False)
    owned.on_model_loaded(ctx)
    assert 'sample_draft' in runner.speculator.__dict__


def test_foreign_instance_method_is_not_overwritten(monkeypatch):
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: Group())
    obj = speculator()
    obj.sample_draft = MethodType(lambda self, *a, **k: None, obj)
    with pytest.raises(RuntimeError, match='owner'):
        owned.bind_speculator(obj, config())


def test_late_owner_drift_latches_even_after_callable_is_restored(monkeypatch):
    group = Group()
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: group)
    obj = speculator()
    owned.bind_speculator(obj, config())
    bound = obj.sample_draft
    obj.sample_draft = MethodType(lambda self, *a, **k: None, obj)
    with pytest.raises(RuntimeError, match='owner/process/group'):
        owned.bind_speculator(obj, config())
    obj.sample_draft = bound
    with pytest.raises(RuntimeError, match='restart'):
        draw(obj, torch.tensor([[0., 1.]]))


def test_method_defaults_and_named_arguments_reach_native_sampler(monkeypatch):
    group = Group()
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: group)
    obj = speculator()
    seen = []
    obj.compute_draft_logits = lambda hidden, step: seen.append(step) or hidden
    owned.bind_speculator(obj, config())
    values = torch.zeros(2, dtype=torch.int32)
    tokens = obj.sample_draft(hidden_states=torch.tensor([[1., 0.], [0., 1.]]),
        sample_src_positions=values, idx_mapping=values, temperature=values.float(),
        seeds=values.long(), draft_step=values, draft_logits=None,
        spec_step_idx=2, num_steps=3)
    assert seen == [2] and tokens.tolist() == [0, 1]
    assert group.calls == 1


def test_prepare_uses_actual_shared_post_load_dispatch(monkeypatch):
    from hcu_adapter import model_load as shared
    from hcu_adapter.bridges import vllm_model_load as bridge
    from vllm import envs
    monkeypatch.setattr(shared._callbacks, 'callbacks', {})
    monkeypatch.setattr(envs, '_is_envs_cache_enabled', lambda: False)
    group = Group()
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: group)
    cfg = config()
    cfg.parallel_config.worker_cls = bridge.POST_MODEL_LOAD_WORKER
    cfg.parallel_config.distributed_executor_backend = 'mp'
    cfg.parallel_config.enable_elastic_ep = False
    owned.prepare(cfg)
    owned.prepare(cfg)
    obj = speculator()
    worker = NS(vllm_config=cfg, model_runner=NS(speculator=obj))
    bridge.dispatch_post_model_load(worker, load_dummy_weights=False)
    assert draw(obj, torch.tensor([[1., 0.]])).item() == 0
    assert group.calls == 1


def test_prepare_rejects_foreign_worker_before_registering(monkeypatch):
    from hcu_adapter import model_load as shared
    from hcu_adapter.bridges import vllm_model_load as bridge
    monkeypatch.setattr(shared._callbacks, 'callbacks', {})
    cfg = config()
    cfg.parallel_config.worker_cls = 'foreign.Worker'
    with pytest.raises(bridge.PostModelLoadError, match='does not provide'):
        owned.prepare(cfg)
    assert not shared._callbacks.callbacks
