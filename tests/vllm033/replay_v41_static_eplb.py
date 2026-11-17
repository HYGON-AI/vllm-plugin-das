#!/usr/bin/env python3
"""Fixed before implementation: CPU source integration of HCU V41 static EPLB.

Distributed membership, Core routing kernels and GPU placement are explicit CPU
doubles. Plan parsing, scope admission, loader fan-out, replica ordering and the
published routing tensors execute as production source.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import torch

import importlib
try:
    plan_mod = importlib.import_module('hcu_vllm_das.static_eplb')
    dispatch = importlib.import_module('hcu_vllm_das.static_eplb_dispatch')
    runtime = importlib.import_module('hcu_vllm_das.static_eplb_runtime')
except ModuleNotFoundError as error:
    if not error.name.startswith('hcu_vllm_das.'): raise
    plan_mod = dispatch = runtime = None

RESULTS = []
LOGICAL, REDUNDANT, LAYERS, EP_SIZE = 8, 2, 2, 2
PHYSICAL = LOGICAL + REDUNDANT
MODEL_NAME = '/models/DeepSeek-V4.1-Flash'
MODEL_KEY = 'HcuV41EplbLLM'
# Logical experts 0 and 1 carry the two redundant replicas.
ROWS = [[0, 1, 2, 3, 4, 5, 6, 7, 0, 1], [1, 0, 2, 3, 4, 5, 6, 7, 1, 0]]


def require(): assert runtime is not None, 'static EPLB production modules are absent'


def case(name, fn):
    try:
        fn(); RESULTS.append(dict(case=name, status='passed'))
    except Exception as error:
        import traceback
        traceback.print_exc()
        RESULTS.append(dict(case=name, status='failed', error=f'{type(error).__name__}: {error}'))


def reject(fn, kind=(ValueError, RuntimeError, TypeError)):
    try:
        fn()
    except kind:
        return
    raise AssertionError('invalid static EPLB contract admitted')


def payload(**overrides):
    top_level = {name: overrides.pop(name) for name in ('version', 'format')
                 if name in overrides}
    entry = dict(record_kind='proposal', model_name=MODEL_NAME, model_class=MODEL_KEY,
                 num_moe_layers=LAYERS, num_logical_experts=LOGICAL,
                 num_physical_experts=PHYSICAL, num_redundant_experts=REDUNDANT,
                 physical_to_logical_map=[list(row) for row in ROWS])
    entry.update(overrides)
    document = dict(version=2, format='vllm_offline_eplb_physical_to_logical_by_model',
                    model_maps={MODEL_KEY: entry})
    document.update(top_level)
    return document


def write_map(directory, **overrides):
    path = Path(directory) / 'eplb_map.json'
    path.write_text(json.dumps(payload(**overrides)), encoding='utf-8')
    return str(path)


def load(path, **overrides):
    arguments = dict(model_key=MODEL_KEY, expected_model_name=MODEL_NAME,
                     expected_shape=(LAYERS, PHYSICAL), num_logical_experts=LOGICAL,
                     num_redundant_experts=REDUNDANT)
    arguments.update(overrides)
    return plan_mod.load_static_eplb_plan(path, **arguments)


def cfg(width=8, **options):
    sidecar = dict(expert_map_path='/tmp/eplb_map.json', num_redundant_experts=REDUNDANT)
    sidecar.update(options)
    if sidecar.get('expert_map_record_path') and 'expert_map_path' not in options:
        sidecar.pop('expert_map_path')
    return NS(
        model_config=NS(enforce_eager=True, model=MODEL_NAME,
                        hf_config=NS(model_type='deepseek_v41', engram_layer_ids=[],
                                     vision_n_layers=0, num_hidden_layers=4,
                                     n_routed_experts=6)),
        parallel_config=NS(prefill_context_parallel_size=width, decode_context_parallel_size=1,
                           pipeline_parallel_size=1, data_parallel_size=1, tensor_parallel_size=1,
                           enable_dbo=False, enable_eplb=False, enable_expert_parallel=True,
                           enable_elastic_ep=False, all2all_backend='deepep_high_throughput',
                           num_ubatches=0),
        use_v2_model_runner=True, speculative_config=None, lora_config=None, kv_transfer_config=None,
        kernel_config=NS(monokernel=False, moe_backend='deep_gemm'),
        cache_config=NS(swa_bounded_replay=False),
        attention_config=NS(backend=None, indexer_sparse_logits=False,
                            resolve_indexer_kv_dtype=lambda _: 'fp8'),
        additional_config={'hcu_static_eplb': sidecar})


def options_parsing():
    require()
    parsed = runtime.read_options(cfg())
    assert parsed.num_redundant_experts == REDUNDANT and parsed.static_dispatch_policy == 'nearest'
    assert not parsed.recording and parsed.record_step_interval == 200
    recording = runtime.read_options(cfg(expert_map_record_path='/tmp/out.json'))
    assert recording.recording and recording.expert_map_path is None
    assert runtime.read_options(NS(additional_config={})) is None
    assert runtime.read_options(NS(additional_config=None)) is None
    assert not runtime.enabled(NS(additional_config={}))
    reject(lambda: runtime.read_options(NS(additional_config={'hcu_static_eplb': {'oops': 1}})))
    reject(lambda: runtime.read_options(NS(additional_config={'hcu_static_eplb': {}})))
    reject(lambda: runtime.read_options(cfg(static_dispatch_policy='round_robin')))
    reject(lambda: runtime.read_options(cfg(num_redundant_experts=-1)))
    reject(lambda: runtime.read_options(cfg(expert_map_path='   ')))
    reject(lambda: runtime.read_options(NS(additional_config={'hcu_static_eplb': 3})))
    reject(lambda: runtime.read_options(cfg(record_step_interval=0)))
    # Deploying and calibrating a placement in one run is not a coherent request.
    both = cfg(); both.additional_config['hcu_static_eplb']['expert_map_record_path'] = '/tmp/o.json'
    reject(lambda: runtime.read_options(both))


def scope_admission():
    require()
    assert runtime.validate_config(cfg())
    assert not runtime.validate_config(NS(additional_config={}))
    for name, value in (('enable_expert_parallel', False), ('enable_eplb', True),
                        ('enable_elastic_ep', True), ('all2all_backend', 'deepep_low_latency')):
        x = cfg(); setattr(x.parallel_config, name, value)
        reject(lambda x=x: runtime.validate_config(x))
    x = cfg(); x.kernel_config.moe_backend = 'auto'
    reject(lambda: runtime.validate_config(x))
    # PCP1 is not the owned target ladder, so static placement has no owner.
    reject(lambda: runtime.validate_config(cfg(width=1)))
    # An incomplete DSpark configuration is rejected before construction.
    x = cfg(); x.speculative_config = NS(method='dspark')
    reject(lambda: runtime.validate_config(x))
    # 6 logical + 3 redundant = 9 physical experts, which EP width 8 cannot
    # divide evenly; this must fail at startup, not at the first proposal.
    reject(lambda: runtime.validate_config(cfg(num_redundant_experts=3)))
    assert runtime.validate_config(cfg(num_redundant_experts=10))
    # Calibration cannot attribute load across DBO ubatches.
    x = cfg(expert_map_record_path='/tmp/o.json'); x.parallel_config.enable_dbo = True
    reject(lambda: runtime.validate_config(x))


def moe_construction_keywords():
    require()
    backbone = runtime.moe_eplb_kwargs(cfg(), 'model.layers.2.ffn.experts')
    assert backbone == {'enable_eplb': True, 'num_redundant_experts': REDUNDANT}
    # DSpark draft layers sit past the backbone and keep native EP construction.
    assert runtime.moe_eplb_kwargs(cfg(), 'model.layers.4.ffn.experts') == {}
    assert runtime.moe_eplb_kwargs(NS(additional_config={}), 'model.layers.0.ffn.experts') == {}


def plan_loading():
    require()
    with tempfile.TemporaryDirectory() as directory:
        plan = load(write_map(directory))
        assert plan.num_physical_experts == PHYSICAL and plan.num_redundant_experts == REDUNDANT
        assert plan.layer_map(1) == tuple(ROWS[1])
        assert plan.physical_to_logical_map.shape == (LAYERS, PHYSICAL)
        assert len(plan.source_sha256) == 64 and plan.fingerprint()[0] == MODEL_KEY
        # Identical bytes must not be served for a different expected layout.
        reject(lambda: load(write_map(directory), expected_shape=(LAYERS + 1, PHYSICAL)))
        reject(lambda: load(write_map(directory), model_key='Other'))
        reject(lambda: load(write_map(directory), expected_model_name='/models/other'))
        reject(lambda: load(write_map(directory, record_kind='initial')))
        reject(lambda: load(write_map(directory, num_logical_experts=LOGICAL + 1)))
        reject(lambda: load(write_map(directory, version=1)))
        reject(lambda: load(write_map(directory, format='other')))
        # A row that drops a logical expert would leave it unroutable.
        broken = [list(ROWS[0]), [0] * PHYSICAL]
        reject(lambda: load(write_map(directory, physical_to_logical_map=broken)))
        # A row of the wrong width cannot index the physical slots.
        short = [list(ROWS[0])[:-1], list(ROWS[1])[:-1]]
        reject(lambda: load(write_map(directory, physical_to_logical_map=short)))


def plan_rejects_missing_file():
    require()
    reject(lambda: load('/nonexistent/eplb_map.json'), kind=(OSError, ValueError))


class LoaderDouble:
    """Minimal RoutedExperts surface the static loader is allowed to touch."""

    def __init__(self, row, fused_shared=0, use_global_sf=False):
        self.moe_config = NS(num_logical_experts=LOGICAL, num_experts=PHYSICAL)
        self.expert_map_manager = NS(num_fused_shared_experts=fused_shared)
        self.quant_method = NS(use_global_sf=use_global_sf)
        setattr(self, runtime._ROW_ATTR, row)
        self.calls = []

    def original(self, param, loaded_weight, weight_name, shard_id, expert_id,
                 return_success=False):
        self.calls.append(expert_id)
        return True if return_success else None


def loader_fans_out_replicas():
    require()
    double = LoaderDouble(tuple(ROWS[0]))
    load_fn = runtime._static_loader(LoaderDouble.original)
    # Logical expert 0 owns physical slots 0 and 8 in this layer.
    assert load_fn(double, None, torch.zeros(2), 'w13_weight', 'w1', 0, return_success=True)
    assert double.calls == [0, 8]
    double.calls.clear()
    # A non-replicated logical expert reaches exactly its one slot.
    load_fn(double, None, torch.zeros(2), 'w13_weight', 'w1', 5, return_success=True)
    assert double.calls == [5]
    # Out-of-range and fused tensors must not be placed by guesswork.
    reject(lambda: load_fn(double, None, torch.zeros(2), 'w13_weight', 'w1', LOGICAL))
    reject(lambda: load_fn(double, None, torch.zeros(2, 2, 2), 'w13_weight', 'w1', 0))


def loader_passes_through_tails():
    require()
    fused = LoaderDouble(tuple(ROWS[0]), fused_shared=1)
    load_fn = runtime._static_loader(LoaderDouble.original)
    # The fused shared expert keeps the physical tail slot, unreplicated.
    load_fn(fused, None, torch.zeros(2), 'w13_weight', 'w1', LOGICAL, return_success=True)
    assert fused.calls == [PHYSICAL]
    # A global scale is per-tensor, so it must be written once in logical space.
    scaled = LoaderDouble(tuple(ROWS[0]), use_global_sf=True)
    load_fn(scaled, None, torch.zeros(1), 'w13_input_scale', 'w1', 0, return_success=True)
    assert scaled.calls == [0]


def replica_map_from_rows(rows):
    counts = [[0] * LOGICAL for _ in rows]
    width = max(row.count(value) for row in rows for value in set(row))
    result = torch.full((len(rows), LOGICAL, width), -1, dtype=torch.long)
    for layer, row in enumerate(rows):
        for physical, logical in enumerate(row):
            result[layer, logical, counts[layer][logical]] = physical
            counts[layer][logical] += 1
    return result


def nearest_replica_order():
    require()
    base = replica_map_from_rows([list(ROWS[0])])
    # ep_size 2 over 10 physical slots: rank 1 owns slots 5..9.
    ordered = dispatch.build_nearest_replica_order(
        base, ep_rank=1, ep_size=EP_SIZE, num_nodes=1, num_physical_experts=PHYSICAL)
    assert ordered[0, 0, 0].item() == 8, 'rank 1 must prefer its local replica of expert 0'
    assert ordered[0, 1, 0].item() == 9
    assert ordered[0, 5, 0].item() == 5
    rank0 = dispatch.build_nearest_replica_order(
        base, ep_rank=0, ep_size=EP_SIZE, num_nodes=1, num_physical_experts=PHYSICAL)
    assert rank0[0, 0, 0].item() == 0, 'rank 0 must prefer its own replica of expert 0'
    # Ordering is a permutation: no replica may be invented or dropped.
    for rank_map in (ordered, rank0):
        for logical in range(LOGICAL):
            assert (set(int(v) for v in rank_map[0, logical].tolist() if v >= 0)
                    == set(int(v) for v in base[0, logical].tolist() if v >= 0))


def locality_fair_replica_order():
    require()
    base = replica_map_from_rows([list(ROWS[0])])
    primaries = set()
    for rank in range(EP_SIZE):
        ordered = dispatch.build_locality_fair_replica_order(
            base, ep_rank=rank, ep_size=EP_SIZE, num_nodes=1, num_physical_experts=PHYSICAL)
        primaries.add(int(ordered[0, 0, 0].item()))
        for logical in range(LOGICAL):
            assert (set(int(v) for v in ordered[0, logical].tolist() if v >= 0)
                    == set(int(v) for v in base[0, logical].tolist() if v >= 0))
    # Expert 0 has one replica per rank, so each rank keeps a same-GPU primary.
    assert primaries == {0, 8}
    reject(lambda: dispatch.build_nearest_replica_order(
        base, ep_rank=EP_SIZE, ep_size=EP_SIZE, num_nodes=1, num_physical_experts=PHYSICAL))
    reject(lambda: dispatch.build_nearest_replica_order(
        base, ep_rank=0, ep_size=3, num_nodes=1, num_physical_experts=PHYSICAL))


class LayerDouble:
    def __init__(self):
        self.eplb_state = NS(expert_load_view=None, logical_to_physical_map=None,
                             logical_replica_count=None, should_record_tensor=None,
                             num_unpadded_tokens_tensors=None)
        self.routed_experts = NS(get_expert_weights=lambda: [torch.zeros(1)])

    def set_eplb_state(self, moe_layer_idx, expert_load_view, logical_to_physical_map,
                       logical_replica_count):
        self.eplb_state.expert_load_view = expert_load_view[moe_layer_idx]
        self.eplb_state.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.eplb_state.logical_replica_count = logical_replica_count[moe_layer_idx]


class ModelDouble:
    """The MixtureOfExperts surface static routing publication is allowed to use."""

    def __init__(self, plan):
        self.layers = [LayerDouble() for _ in range(LAYERS)]
        self.moe_layers = self.layers
        self.moe_mlp_layers = [NS(experts=layer) for layer in self.layers]
        self.num_moe_layers = LAYERS
        self.num_logical_experts = LOGICAL
        self.num_physical_experts = PHYSICAL
        self.num_redundant_experts = REDUNDANT
        self.num_expert_groups = 1
        for index, layer in enumerate(self.layers):
            setattr(layer.routed_experts, runtime._ROW_ATTR, plan.layer_map(index))
        setattr(self, runtime._PLAN_ATTR, plan)

    def parameters(self):
        return iter([torch.zeros(1)])

    def set_eplb_state(self, expert_load_view, logical_to_physical_map, logical_replica_count):
        for index, layer in enumerate(self.layers):
            layer.set_eplb_state(moe_layer_idx=index, expert_load_view=expert_load_view,
                                 logical_to_physical_map=logical_to_physical_map,
                                 logical_replica_count=logical_replica_count)


def install_routing_publishes_primary_only():
    require()
    with tempfile.TemporaryDirectory() as directory:
        path = write_map(directory)
        plan = load(path)
        model = ModelDouble(plan)
        group = NS(device_group=NS(rank=lambda: 1, size=lambda: EP_SIZE))
        with (patch('vllm.distributed.get_ep_group', return_value=group),
              patch('vllm.distributed.get_node_count', return_value=1),
              patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                    return_value=model)):
            runtime.install_routing(model, cfg(expert_map_path=path))
        for index, layer in enumerate(model.layers):
            state = layer.eplb_state
            # Exactly one replica is routable, so no token can reach a stale slot.
            assert torch.equal(state.logical_replica_count,
                               torch.ones(LOGICAL, dtype=torch.long))
            assert state.expert_load_view.shape == (PHYSICAL,)
            assert bool(state.should_record_tensor.item()) is False
            assert len(state.num_unpadded_tokens_tensors) == 1
            row = ROWS[index]
            for logical in range(LOGICAL):
                primary = int(state.logical_to_physical_map[logical, 0].item())
                assert row[primary] == logical, 'primary replica must hold its logical expert'
            # Rank 1 owns the upper half of the physical slots.
            assert int(state.logical_to_physical_map[row[8], 0].item()) == 8


def install_routing_requires_binding():
    require()
    with tempfile.TemporaryDirectory() as directory:
        path = write_map(directory)
        plan = load(path)
        model = ModelDouble(plan)
        delattr(model, runtime._PLAN_ATTR)
        with patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                   return_value=model):
            reject(lambda: runtime.install_routing(model, cfg(expert_map_path=path)))
        assert runtime.install_routing(NS(additional_config={}), NS(additional_config={})) is None


def model_key_stays_native():
    """Offline maps are keyed by Core's class name, not the owned subclass."""
    require()
    models = importlib.import_module('hcu_vllm_das.models.v41_pcp')
    for owner in (models.HcuV41EplbLLM, models.HcuV41PcpLLM):
        instance = owner.__new__(owner)
        assert runtime.resolve_model_key(instance) == 'DeepseekV41LLMForCausalLM'
    reject(lambda: runtime.resolve_model_key(NS()))


def calibration_allocates_identity():
    """A calibration run must perturb nothing: identity placement, 0 redundancy."""
    require()
    recording = cfg(expert_map_record_path='/tmp/out.json')
    keywords = runtime.moe_eplb_kwargs(recording, 'model.layers.2.ffn.experts')
    # Router state is still required, but no redundant slot is allocated: the
    # configured redundancy describes the map to propose, not this run.
    assert keywords == {'enable_eplb': True, 'num_redundant_experts': 0}
    assert runtime.bind_placement(NS(), recording) is None


def recording_session(output, *, model=None):
    """Install (or re-install) calibration routing; returns (model, session)."""
    if model is None:
        identity = [tuple(range(LOGICAL))] * LAYERS
        model = ModelDouble(NS(layer_map=lambda index: identity[index],
                               physical_to_logical_map=identity,
                               num_logical_experts=LOGICAL,
                               num_physical_experts=LOGICAL,
                               num_redundant_experts=0))
        model.num_physical_experts = LOGICAL
        model.num_redundant_experts = 0
    with (patch.object(runtime, 'resolve_model_key',
                       return_value='DeepseekV41LLMForCausalLM'),
          patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                return_value=model)):
        session = runtime.install_routing(model, cfg(expert_map_record_path=output))
    return model, session


def calibration_records_and_proposes():
    require()
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory) / 'calibrated.json')
        model, session = recording_session(output)
        options = runtime.read_options(cfg(expert_map_record_path=output))
        with patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                   return_value=model):
            assert runtime.record_session(model) is session
        for index, layer in enumerate(model.layers):
            state = layer.eplb_state
            # Recording on, identity mapping: routing matches a plain run.
            assert bool(state.should_record_tensor.item()) is True
            assert torch.equal(state.logical_to_physical_map.reshape(-1),
                               torch.arange(LOGICAL, dtype=torch.long))
            assert state.expert_load_view.shape == (LOGICAL,)
        group = NS(device_group=NS(rank=lambda: 0, size=lambda: EP_SIZE))
        with (patch('vllm.distributed.get_ep_group', return_value=group),
              patch('vllm.distributed.get_node_count', return_value=1)):
            # No routing observed yet, so no proposal may be written.
            assert session.flush() is None
            assert not Path(output).exists()
            # Skew the load so the policy has something to balance.
            session.load[:, 0] = 4096
            session.load[:, 1:] = 16
            for _ in range(options.record_step_interval):
                session.step()
            assert session.writes == 1 and Path(output).exists()
        # The proposal must load back through the deployment path unchanged.
        plan = load(output, num_redundant_experts=REDUNDANT,
                    expected_shape=(LAYERS, LOGICAL + REDUNDANT),
                    model_key='DeepseekV41LLMForCausalLM')
        assert plan.num_physical_experts == LOGICAL + REDUNDANT
        assert plan.source_sha256 == session.last_sha256
        # A hot logical expert is the one worth replicating.
        assert plan.layer_map(0).count(0) > 1


def calibration_publishes_token_count():
    """Core's kernel only records rows below num_unpadded_tokens.

    Left at the zero it is allocated with, every recorded row is filtered out
    and a calibration run produces nothing at all, silently.
    """
    require()
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory) / 'calibrated.json')
        model, session = recording_session(output)
        unpadded = model.layers[0].eplb_state.num_unpadded_tokens_tensors
        assert int(unpadded[0].item()) == 0, 'allocated counters start at zero'

        # No padding mask: every row of the batch is a real token.
        with patch('vllm.forward_context.get_forward_context',
                   return_value=NS(attn_metadata={}, ubatch_slices=None, is_padding=None)):
            session.prepare(96)
        assert int(unpadded[0].item()) == 96

        # With padding, only the real rows may be counted, or a padded batch
        # would bias the load toward whatever the padding rows route to.
        padding = torch.zeros(128, dtype=torch.bool)
        padding[70:] = True
        with patch('vllm.forward_context.get_forward_context',
                   return_value=NS(attn_metadata={}, ubatch_slices=None, is_padding=padding)):
            session.prepare(128)
        assert int(unpadded[0].item()) == 70
        # Every layer shares the counter the routers read.
        for layer in model.layers:
            assert layer.eplb_state.num_unpadded_tokens_tensors is unpadded
        # DBO would need per-ubatch counts; refuse rather than mis-attribute.
        with patch('vllm.forward_context.get_forward_context',
                   return_value=NS(attn_metadata={}, ubatch_slices=[object()], is_padding=None)):
            reject(lambda: session.prepare(96))


def routing_install_is_idempotent():
    """The post-load hook can fire twice; re-publishing would drop recorded load."""
    require()
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory) / 'calibrated.json')
        model, session = recording_session(output)
        session.load[:, 3] = 11
        again = recording_session(output, model=model)[1]
        assert again is session, 'second install must reuse the bound session'
        assert int(session.load[:, 3].sum().item()) == 11 * LAYERS, \
            'accumulated load survived re-install'


def proposal_rejects_bad_shapes():
    require()
    record_mod = importlib.import_module('hcu_vllm_das.static_eplb_record')
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'bad.json'
        # A row that drops a logical expert must never reach the file.
        broken = (tuple([0] * (LOGICAL + REDUNDANT)),) * LAYERS
        reject(lambda: record_mod.write_offline_map(
            output, broken, model_key=MODEL_KEY, model_name=MODEL_NAME,
            num_logical=LOGICAL, num_redundant=REDUNDANT))
        assert not output.exists()
        reject(lambda: record_mod.propose_rows(
            torch.ones(LAYERS, LOGICAL), num_physical=LOGICAL + 1, num_groups=1,
            num_nodes=1, num_ranks=EP_SIZE))


def calibration_excludes_startup_forwards():
    """The actual model wrapper records real requests after dummy startup."""
    require()
    models = importlib.import_module('hcu_vllm_das.models.v41_pcp')
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory) / 'calibrated.json')
        _, session = recording_session(output)
        outer = models.HcuV41PcpForCausalLM.__new__(models.HcuV41PcpForCausalLM)
        torch.nn.Module.__init__(outer)
        outer._hcu_static_eplb_session = session

        def router_double(self, input_ids, positions, *args, **kwargs):
            # The GPU router records only rows below the published counter.
            session.load[:, 0] += int(session.unpadded[0].item())
            return positions

        group = NS(device_group=NS(rank=lambda: 0, size=lambda: EP_SIZE))
        with (patch.object(models.native_vl.DeepseekV41ForCausalLM,
                           'forward', router_double),
              patch('vllm.distributed.get_ep_group', return_value=group),
              patch('vllm.distributed.get_node_count', return_value=1)):
            inputs = torch.zeros(8, dtype=torch.long)
            positions = torch.arange(8)
            profile = NS(attn_metadata=None, ubatch_slices=None,
                         is_padding=torch.zeros(8, dtype=torch.bool))
            with patch('vllm.forward_context.get_forward_context', return_value=profile):
                outer.forward(inputs, positions)
            assert session.steps == 0 and session.load.sum().item() == 0
            assert session.flush() is None and not Path(output).exists()

            warmup = NS(attn_metadata={}, ubatch_slices=None,
                        is_padding=torch.ones(8, dtype=torch.bool))
            with patch('vllm.forward_context.get_forward_context', return_value=warmup):
                outer.forward(inputs, positions)
            assert session.load.sum().item() == 0
            assert session.flush() is None and not Path(output).exists()

            request = NS(attn_metadata={}, ubatch_slices=None,
                         is_padding=torch.tensor([False] * 6 + [True] * 2))
            with patch('vllm.forward_context.get_forward_context', return_value=request):
                outer.forward(inputs, positions)
            assert int(session.unpadded[0].item()) == 6
            assert int(session.load.sum().item()) == 6 * LAYERS
            # A later profile must preserve, and add nothing to, real load.
            steps = session.steps
            with patch('vllm.forward_context.get_forward_context', return_value=profile):
                outer.forward(inputs, positions)
            assert session.steps == steps
            assert int(session.load.sum().item()) == 6 * LAYERS
            assert session.flush() is not None and Path(output).exists()


def dummy_deployment_binds_before_routing():
    """Post-load must work even when model.load_weights was never called."""
    require()
    models = importlib.import_module('hcu_vllm_das.models.v41_pcp')
    with tempfile.TemporaryDirectory() as directory:
        path = write_map(directory)
        model = ModelDouble(load(path))
        delattr(model, runtime._PLAN_ATTR)
        for layer in model.layers:
            experts = layer.routed_experts
            delattr(experts, runtime._ROW_ATTR)
            experts.moe_config = NS(num_logical_experts=LOGICAL, num_experts=PHYSICAL)
            experts.local_num_experts = PHYSICAL // EP_SIZE
            # Quantization can replace parameters with packed tensors that
            # have no checkpoint loader. Binding layout must leave those alone.
            weight = torch.nn.Parameter(torch.zeros(PHYSICAL // EP_SIZE, 2))
            experts.get_expert_weights = lambda weight=weight: [weight]
            experts.named_parameters = lambda weight=weight: [('packed_weight', weight)]
        config = cfg(width=EP_SIZE, expert_map_path=path)
        config.model_config.hf_config.n_routed_experts = LOGICAL
        outer = models.HcuV41PcpForCausalLM.__new__(models.HcuV41PcpForCausalLM)
        torch.nn.Module.__init__(outer)
        outer._hcu_static_eplb_config = config
        group = NS(device_group=NS(rank=lambda: 1, size=lambda: EP_SIZE))
        with (patch.object(models.native_vl.DeepseekV41ForCausalLM,
                           'process_weights_after_loading', lambda self: None),
              patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                    return_value=model),
              patch('vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts', NS),
              patch.object(runtime, 'resolve_model_key', return_value=MODEL_KEY),
              patch('vllm.distributed.get_ep_group', return_value=group),
              patch('vllm.distributed.get_node_count', return_value=1)):
            outer.process_weights_after_loading()
            plan = getattr(model, runtime._PLAN_ATTR)
            outer.process_weights_after_loading()
            assert getattr(model, runtime._PLAN_ATTR) is plan
        assert outer._hcu_static_eplb_session is None
        for index, layer in enumerate(model.layers):
            assert getattr(layer.routed_experts, runtime._ROW_ATTR) == tuple(ROWS[index])
            assert layer.eplb_state.logical_to_physical_map is not None
            assert not hasattr(layer.routed_experts.get_expert_weights()[0], 'weight_loader')
        assert models.HcuV41EplbBackbone._decoder_replay_supported is (
            models.HcuV41PcpBackbone._decoder_replay_supported)


CASES = (
    ('options_parsing', options_parsing),
    ('scope_admission', scope_admission),
    ('moe_construction_keywords', moe_construction_keywords),
    ('plan_loading', plan_loading),
    ('plan_rejects_missing_file', plan_rejects_missing_file),
    ('model_key_stays_native', model_key_stays_native),
    ('loader_fans_out_replicas', loader_fans_out_replicas),
    ('loader_passes_through_tails', loader_passes_through_tails),
    ('nearest_replica_order', nearest_replica_order),
    ('locality_fair_replica_order', locality_fair_replica_order),
    ('install_routing_publishes_primary_only', install_routing_publishes_primary_only),
    ('install_routing_requires_binding', install_routing_requires_binding),
    ('calibration_allocates_identity', calibration_allocates_identity),
    ('calibration_records_and_proposes', calibration_records_and_proposes),
    ('calibration_publishes_token_count', calibration_publishes_token_count),
    ('routing_install_is_idempotent', routing_install_is_idempotent),
    ('proposal_rejects_bad_shapes', proposal_rejects_bad_shapes),
    ('calibration_excludes_startup_forwards', calibration_excludes_startup_forwards),
    ('dummy_deployment_binds_before_routing', dummy_deployment_binds_before_routing),
)


def main(report_path):
    for name, fn in CASES:
        case(name, fn)
    failed = [entry for entry in RESULTS if entry['status'] != 'passed']
    report = dict(total=len(RESULTS), passed=len(RESULTS) - len(failed), cases=RESULTS)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"static EPLB replay: {report['passed']}/{report['total']} passed")
    for entry in failed:
        print(f"  FAILED {entry['case']}: {entry['error']}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'report.json'))
