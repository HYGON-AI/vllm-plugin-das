"""CPU admission contracts for the public INT8 owner; no hardware acceptance."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adapter_extension/src"))

from hcu_adapter.extensions import ExtensionContext
from hcu_vllm_das import int8_activation
from hcu_vllm_das.moe.int8_experts import config_rejection
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
)


def config(**changes):
    values = dict(
        num_experts=8, experts_per_token=2, hidden_dim=256,
        intermediate_size=128, num_local_experts=8, num_logical_experts=8,
        activation=MoEActivation.SILU, device="cuda", in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.Default, moe_backend="aiter",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    result = FusedMoEConfig(**(values | changes))
    # Core resolves this derived field in __post_init__; emulate the actual
    # resolved config contract, rather than a discarded constructor argument.
    if "aiter_fmoe_shared_expert_enabled" in changes:
        result.aiter_fmoe_shared_expert_enabled = changes["aiter_fmoe_shared_expert_enabled"]
    return result


def test_int8_is_an_independent_explicit_capability():
    spec = int8_activation.extension_spec()
    assert spec.capabilities == frozenset({"hcu-aiter-int8-moe"})
    assert spec.check_health is int8_activation.check_health
    with pytest.raises(RuntimeError, match="explicit"):
        spec.activate(ExtensionContext(adapter_version="0.5.22", vllm_version="0.33.0"))
    with pytest.raises(RuntimeError, match="not active"):
        spec.check_health()


def test_extension_discovery_does_not_import_framework_or_vendor():
    # Portable tier supplies Adapter. Source-only packaging deliberately does
    # not import an extension that depends on that separately installed host.
    source = Path(__file__).resolve().parents[2] / "adapter_extension/src"
    code = f"""
import sys
sys.path[:] = {[str(source), *sys.path]!r}
blocked = ('torch', 'vllm', 'lightop', 'aiter', 'vllm_hcu')
assert not any(n.split('.')[0] in blocked for n in sys.modules)
from hcu_vllm_das.int8_activation import extension_spec
assert extension_spec().capabilities == frozenset({{'hcu-aiter-int8-moe'}})
assert not any(n.split('.')[0] in blocked for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("changes", [
    dict(has_bias=True), dict(is_lora_enabled=True),
    dict(apply_router_weight_on_input=True), dict(in_dtype=torch.float32),
    dict(activation=MoEActivation.GELU), dict(swiglu_alpha=1.2),
    dict(swiglu_limit=1.0), dict(hidden_dim_unpadded=255),
    dict(intermediate_size_per_partition_unpadded=127),
    dict(tp_shard_with_padding=True), dict(aiter_fmoe_shared_expert_enabled=True),
])
def test_unsupported_config_rejects_before_provider(changes):
    assert config_rejection(config()) is None
    assert config_rejection(config(**changes)) is not None


def test_production_dispatch_requests_asm_and_uses_whole_call_fallback(monkeypatch):
    from types import SimpleNamespace
    from hcu_vllm_das.moe import fused_moe_aiter as provider
    seen = {}
    result = torch.ones((2, 8), dtype=torch.bfloat16)

    def get_config(*args, **kwargs):
        seen.update(kwargs)
        return False, SimpleNamespace(solution_type=None, config=None)

    monkeypatch.setattr(provider, '_load_aiter_api', lambda: SimpleNamespace(
        quant_type_w8a8='int8_w8a8', get_config=get_config,
        aiter_moe=lambda **kwargs: pytest.fail('no native launch on config miss'),
    ))

    def fallback(**kwargs):
        # The complete raw-input call reaches Core, which owns scratch and quant.
        assert kwargs['hidden_states'].dtype is torch.bfloat16
        assert kwargs['use_int8_w8a8'] and kwargs['per_channel_quant']
        assert 'workspace13' not in kwargs and 'workspace2' not in kwargs
        return result

    monkeypatch.setattr(provider, '_load_triton_fallback', lambda: fallback)
    quant = SimpleNamespace(use_int8_w8a8=True)
    actual = provider.apply_aiter_int8_moe(
        hidden_states=torch.zeros((2, 8), dtype=torch.bfloat16),
        w1=torch.zeros((4, 12, 8), dtype=torch.int8),
        w2=torch.zeros((4, 8, 6), dtype=torch.int8),
        topk_weights=torch.ones((2, 2), dtype=torch.float32),
        topk_ids=torch.zeros((2, 2), dtype=torch.int32),
        moe_config=SimpleNamespace(num_experts=4), quant_config=quant,
        activation='silu', expert_map=None, output_dtype=torch.bfloat16,
        asm_only=True,
    )
    assert seen['spec_sol_type'] == 'asm'
    assert actual is result


def test_frozen_identity_hashes_content_and_refuses_drift(tmp_path):
    from hcu_vllm_das.moe.int8_identity import FrozenFiles
    a = tmp_path / 'source_a'
    b = tmp_path / 'source_b'
    a.write_bytes(b'first implementation')
    b.write_bytes(a.read_bytes())
    first = FrozenFiles.capture({'provider': a})
    equivalent = FrozenFiles.capture({'provider': b})
    assert first.digest == equivalent.digest
    first.validate()
    a.write_bytes(b'other implementation')
    with pytest.raises(RuntimeError, match='changed'):
        first.validate()
    second = FrozenFiles.capture({'provider': a})
    assert first.digest != second.digest


def test_late_activation_refuses_vendor_config_cache_before_device_work(monkeypatch):
    from types import ModuleType
    from vllm.model_executor.layers.fused_moe.oracle.int8_backend_registry import get_int8_moe_backend
    module = 'aiter.fused_moe_asm_wna16'
    monkeypatch.setitem(sys.modules, module, ModuleType(module))
    before = get_int8_moe_backend('aiter')
    context = ExtensionContext(adapter_version='0.5.22', vllm_version='0.33.0',
                               requested_capabilities={'hcu-aiter-int8-moe'})
    with pytest.raises(RuntimeError, match='before vendor MoE'):
        int8_activation.activate(context)
    assert get_int8_moe_backend('aiter') is before
    assert int8_activation._published is None and not int8_activation._failure


def ep_config(**changes):
    parallel = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=2,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1,
        use_ep=True, all2all_backend='allgather_reducescatter', enable_eplb=False,
    )
    return config(num_experts=256, experts_per_token=8, hidden_dim=2048,
                  intermediate_size=512, num_local_experts=128,
                  num_logical_experts=256, moe_parallel_config=parallel, **changes)


@pytest.fixture
def qualified_ep2_device(monkeypatch):
    from hcu_adapter.bridges import vllm as host
    monkeypatch.setattr(host, 'detect_hcu',
        lambda: host.HcuDetection(True, 'gfx936', 'CPU external-device fixture'))


def test_static_ep2_config_is_admitted_for_qualified_bf16_geometry(qualified_ep2_device):
    from hcu_vllm_das.moe.int8_experts import HcuAiterInt8Experts
    actual = ep_config()
    assert config_rejection(actual) is None
    assert HcuAiterInt8Experts._supports_parallel_config(actual.moe_parallel_config)


@pytest.mark.parametrize(('field', 'value'), [
    ('ep_size', 4), ('tp_size', 2), ('dp_size', 2), ('pcp_size', 2),
    ('sp_size', 2), ('enable_eplb', True),
])
def test_unqualified_ep_topologies_refuse_before_conversion(field, value):
    actual = ep_config()
    setattr(actual.moe_parallel_config, field, value)
    assert config_rejection(actual) is not None


@pytest.mark.parametrize(('field', 'value'), [
    ('num_experts', 128), ('num_local_experts', 64), ('hidden_dim', 1024),
    ('intermediate_size_per_partition', 256), ('experts_per_token', 4),
    ('in_dtype', torch.float16), ('elastic_ep_max_dp_size', 2),
])
def test_unqualified_ep_geometry_refuses_before_conversion(field, value):
    actual = ep_config()
    setattr(actual, field, value)
    assert config_rejection(actual) is not None


def test_public_ep_apply_forwards_canonical_map_and_compatibility_optin(monkeypatch,qualified_ep2_device):
    from hcu_vllm_das.moe import int8_experts, fused_moe_aiter
    from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
    monkeypatch.setattr(int8_experts, '_require_active', lambda: None)
    cfg = ep_config()
    s1 = torch.empty((128, 1024, 1), device='meta')
    s2 = torch.empty((128, 2048, 1), device='meta')
    quant = int8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2,
                                    a1_scale=None, a2_scale=None, per_act_token_quant=True)
    expert = int8_experts.HcuAiterInt8Experts(cfg, quant)
    x = torch.empty((1, 2048), dtype=torch.bfloat16, device='meta')
    w1 = torch.empty((128, 1024, 2048), dtype=torch.int8, device='meta')
    w2 = torch.empty((128, 2048, 512), dtype=torch.int8, device='meta')
    ids = torch.empty((1, 8), dtype=torch.int32, device='meta')
    weights = torch.empty((1, 8), dtype=torch.float32, device='meta')
    mapping = torch.empty(256, dtype=torch.int32, device='meta')
    empty = torch.empty(0, device='meta')
    seen = []
    monkeypatch.setattr(fused_moe_aiter, 'apply_aiter_int8_moe', lambda **kw: seen.append(kw) or x)
    output = torch.empty_like(x)
    expert.apply(output, x, w1, w2, weights, ids, MoEActivation.SILU,
                 256, mapping, None, None, empty, empty, None, False)
    assert seen[0]['expert_map'] is mapping and seen[0]['allow_ep_global_profile'] is True
    with pytest.raises(ValueError, match='routing'):
        expert.apply(output, x, w1, w2, weights, ids, MoEActivation.SILU,
                     256, None, None, None, empty, empty, None, False)


@pytest.mark.parametrize('ep', [False, True])
def test_passthrough_pf_cannot_claim_local_asm_output_is_globally_reduced(ep):
    from hcu_vllm_das.moe.int8_experts import HcuAiterInt8Experts
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
    actual = ep_config() if ep else config()
    actual.moe_parallel_config.all2all_backend = 'passthrough'
    pf = maybe_make_prepare_finalize(actual, None, allow_new_interface=True)
    assert type(pf).__name__ == 'PassThroughPrepareAndFinalize'
    assert pf.output_is_reduced()
    # ASM computes only local experts and delegates reduction to Core runner.
    assert config_rejection(actual) is not None
    assert not HcuAiterInt8Experts._supports_parallel_config(actual.moe_parallel_config)


def test_static_ep_uses_actual_parallel_config_normalized_dp_capacity(qualified_ep2_device):
    from vllm.config import ParallelConfig
    parallel = ParallelConfig(tensor_parallel_size=2, enable_expert_parallel=True,
                              disable_custom_all_reduce=True)
    assert not parallel.enable_elastic_ep and parallel.elastic_ep_max_dp_size == 1
    actual = ep_config(elastic_ep_max_dp_size=parallel.elastic_ep_max_dp_size)
    assert config_rejection(actual) is None
