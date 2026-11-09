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
