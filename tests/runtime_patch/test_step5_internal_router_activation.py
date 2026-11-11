"""The internal router must pass its explicit activation to the shared MLP."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import functools
import os
import subprocess
import sys

import pytest
import torch


def _isolated_worker_test(function):
    # Worker replacement modules must bootstrap before upstream MoE imports.
    # Keep this test's registration out of unrelated tests' collection process.
    @functools.wraps(function)
    def run(*args, **kwargs):
        node = kwargs['request'].node.nodeid
        if os.environ.get('VLLM_HCU_ROUTER_ACTIVATION_TEST_CHILD') == node:
            return function(*args, **kwargs)
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-xq', node],
            env=dict(os.environ, VLLM_HCU_ROUTER_ACTIVATION_TEST_CHILD=node),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return run


class Projection(torch.nn.Module):
    def __init__(self, gate=False):
        super().__init__()
        self.gate = gate
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return (torch.cat([x, x], dim=-1) if self.gate else x), None


class Router(torch.nn.Module):
    def __init__(self, declares_activation, returns_aux):
        super().__init__()
        self.returns_fused_shared_activation = declares_activation
        self.returns_aux = returns_aux

    def forward(self, x):
        return torch.zeros_like(x), x * 7 if self.returns_aux else None


@pytest.mark.parametrize('declares_activation,returns_aux', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('fused_stream', [False, True])
@_isolated_worker_test
def test_internal_router_handoff_consumes_fresh_activation_only(
    monkeypatch, declares_activation, returns_aux, fused_stream, request
):
    import vllm
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_hcu.patch.worker import apply_worker_patches

    load_general_plugins()
    apply_worker_patches()
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
    from vllm_hcu.models.step5_text import Step5MLP
    import vllm_hcu.model_executor.layers.fused_moe.shared_experts as shared_runtime

    events = []
    class Stream:
        def __init__(self, number): self.cuda_stream = number
        def wait_stream(self, other): events.append(('wait', self.cuda_stream, other.cuda_stream))
    main, auxiliary = Stream(77), Stream(88)
    active = [main]
    @contextmanager
    def use_stream(stream):
        previous = active[0]; active[0] = stream
        try: yield
        finally: active[0] = previous
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: active[0])
    monkeypatch.setattr(shared_runtime, 'current_stream', lambda: active[0])
    monkeypatch.setattr(torch.cuda, 'stream', use_stream)
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda x, s: events.append(('record', s.cuda_stream)))
    # Set after module import, matching executor-side worker environment setup.
    monkeypatch.setenv('HCU_STEP5_FUSED_ROUTER_STREAM', '1' if fused_stream else '0')
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setattr(shared_runtime.henvs, 'VLLM_HCU_USE_CUSTOM_OPS', False)
    shared = Step5MLP.__new__(Step5MLP)
    torch.nn.Module.__init__(shared)
    shared.fused_act = None
    shared.gate_up_proj = Projection(gate=True)
    shared.act_fn = lambda x: x[..., :x.shape[-1] // 2]
    shared.down_proj = Projection()
    shared._fused_router_shared_max_m = 64 if declares_activation else None

    # Exercise the real ordering and output-consumption wrapper. The gate runs
    # after maybe_sync, and NO_OVERLAP must consume the activation after it.
    wrapper = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(wrapper)
    wrapper.enable_dbo = False
    wrapper._hcu_step5_owner = True
    wrapper._layer = shared
    wrapper._stream = auxiliary if fused_stream else None
    wrapper._output = [None, None]
    wrapper._output_pending_on_stream = [False, False]
    wrapper._output_pending_late = [False, False]
    wrapper._output_pending_fused_router = [False, False]
    wrapper._moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(
        enable_eplb=False, use_fi_nvl_two_sided_kernels=False,
    ))
    wrapper._mk_can_overlap_shared_experts = lambda: False
    wrapper._is_multistream_safe = lambda: False

    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.gate = Router(declares_activation, returns_aux)
    runner._shared_experts = wrapper
    runner._fse_fuse_gate = False
    runner._ep_fused_combine = False
    runner.moe_config = SimpleNamespace(pcp_size=1)
    quant_method = SimpleNamespace(is_monolithic=False, topk_indices_dtype=torch.int32)
    runner.routed_experts = SimpleNamespace(
        _ensure_moe_quant_config_init=lambda: None,
        quant_method=quant_method,
        forward_modular=lambda **kwargs: torch.zeros_like(kwargs['x']),
    )
    runner.router = SimpleNamespace(select_experts=lambda **kwargs: (
        torch.ones((kwargs['hidden_states'].shape[0], 1)),
        torch.zeros((kwargs['hidden_states'].shape[0], 1), dtype=torch.int32),
    ))
    runner._sequence_parallel_context = lambda: nullcontext()
    runner._maybe_dispatch = lambda x, logits: (x, logits)
    runner._pcp_routing_inputs = lambda *args: nullcontext(None)
    runner._maybe_combine = lambda shared_output, routed_output: shared_output + routed_output

    for value in [1., 2.]:
        events.clear()
        x = torch.full((2, 4), value)
        out = runner._forward_impl(x, None, x)
        expected = x * 7 if declares_activation and returns_aux else x
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert shared.fused_act is None
        assert wrapper._output == [None, None]
        assert not any(wrapper._output_pending_on_stream)
        assert not any(wrapper._output_pending_fused_router)
        if fused_stream and declares_activation and returns_aux:
            assert events == [('wait', 88, 77), ('record', 88), ('wait', 77, 88)]
        else:
            assert not events
    assert shared.gate_up_proj.calls == (0 if declares_activation and returns_aux else 2)
