"""Compare OOT and genuine Core layers through the serving compiler passes."""
import hashlib
import json

import pytest

pytestmark = pytest.mark.hcu


@pytest.mark.parametrize('dtype_name', ['bfloat16', 'float16'])
@pytest.mark.parametrize('width', [128, 2048, 4096])
@pytest.mark.parametrize('residual_chain', [False, True])
def test_installed_rms_eager_compiled_and_composed_graph(
    monkeypatch, record_property, dtype_name, width, residual_chain,
):
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from vllm.compilation.backends import VllmBackend
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from hcu_vllm_das.rms_norm import HcuRMSNorm

    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_RMS_NORM', '1')
    # Independent layer closures should not exhaust the shared specialization cache.
    torch._dynamo.reset()
    assert not torch._inductor.config.emulate_precision_casts
    dtype = getattr(torch, dtype_name)
    observed = {'core': {}, 'oot': {}}
    active_owner = [None]
    original_run = CachingAutotuner.run

    def observe_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        owner = active_owner[0]
        if owner is not None:
            observed[owner][self.inductor_meta['kernel_name']] = {
                'source_sha256': hashlib.sha256(self.fn.src.encode()).hexdigest(),
                'launch_config': [str(launcher.config) for launcher in self.launchers],
            }
        return result

    # Observe the selected launchers without altering the available configurations.
    monkeypatch.setattr(CachingAutotuner, 'run', observe_run)

    def invoke(owner, fn, *inputs):
        active_owner[0] = owner
        try:
            return fn(*inputs)
        finally:
            active_owner[0] = None

    def compare(actual, expected, *, exact=False):
        if not residual_chain:
            actual, expected = (actual,), (expected,)
        for index, (left, right) in enumerate(zip(actual, expected)):
            if exact or index == 2:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif dtype == torch.float16:
                # Exact target IR explicitly sets this float16 reduction tolerance.
                torch.testing.assert_close(left, right, atol=1e-2, rtol=2e-3)
            else:
                torch.testing.assert_close(left, right)

    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        torch.manual_seed(20261117)
        layer = RMSNorm(width, eps=1e-6).to(device='cuda', dtype=dtype)
        assert isinstance(layer, HcuRMSNorm)
        assert layer._forward_method.__name__ == 'forward_hip'
        layer.weight.data.normal_(1, .2)
        # Bypass only the OOT factory, retaining the real Core init and methods.
        core_layer = object.__new__(RMSNorm)
        RMSNorm.__init__(core_layer, width, eps=1e-6)
        core_layer.weight = layer.weight
        assert type(core_layer) is RMSNorm
        assert core_layer._forward_method.__func__ is RMSNorm.forward_native

        def expected_call(first, second, residual):
            if not residual_chain:
                return core_layer(first.clone())
            y1, r1 = core_layer(first.clone(), residual.clone())
            y2, r2 = core_layer(second.clone(), r1)
            # Returning r1 as well would force an extra intermediate store.
            return y1, y2, r2

        def actual_call(first, second, residual):
            if not residual_chain:
                return layer(first.clone())
            # The IR permits inplace implementations, so callers supply clones.
            y1, r1 = layer(first.clone(), residual.clone())
            y2, r2 = layer(second.clone(), r1)
            return y1, y2, r2

        backends = {}

        def backend_for(name):
            def compile_graph(gm, inputs):
                # Core creates one backend for each compilation invocation.
                # Dynamo may restart analysis, so a backend cannot be reused.
                config = VllmConfig()
                assert config.compilation_config.ir_enable_torch_wrap
                # Force passes to execute so their selected providers are observable.
                # This changes compile caching, not arithmetic or precision policy.
                config.compilation_config.inductor_compile_config['force_disable_caches'] = True
                backend = VllmBackend(config, prefix=name, use_aot_compile=False)
                backends[name] = backend
                return backend(gm, inputs)
            return compile_graph

        compiled_expected = torch.compile(
            expected_call, backend=backend_for('core'), fullgraph=True, dynamic=False,
        )
        compiled_actual = torch.compile(
            actual_call, backend=backend_for('oot'), fullgraph=True, dynamic=False,
        )
        for count in (17, 33):
            inputs = [torch.randn(count, width, device='cuda', dtype=dtype) for _ in range(3)]
            for value in inputs:
                torch._dynamo.mark_dynamic(value, 0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    invoke('core', compiled_expected, *inputs)
                    invoke('oot', compiled_actual, *inputs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                expected = invoke('core', compiled_expected, *inputs)
                actual = invoke('oot', compiled_actual, *inputs)
            assert observed['core'] and observed['oot']
            assert {k: v['source_sha256'] for k, v in observed['core'].items()} == {
                k: v['source_sha256'] for k, v in observed['oot'].items()
            }
            same_launch_config = observed['core'] == observed['oot']
            addresses = [t.data_ptr() for t in inputs]
            for scale in (.25, -1.75, 3.5):
                for value in inputs:
                    value.copy_(torch.randn_like(value) * scale)
                unchanged = [value.clone() for value in inputs]
                compare(actual_call(*inputs), expected_call(*inputs))
                graph.replay()
                torch.cuda.synchronize()
                # Independent default autotuning can change reduction order even
                # between two genuine Core layers. Equal source/config must be
                # bitwise equal; differing launch configs retain the official
                # dtype tolerance above, with residual/input checks still exact.
                compare(actual, expected, exact=same_launch_config)
                assert addresses == [t.data_ptr() for t in inputs]
                for before, after in zip(unchanged, inputs):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
        selected = {
            name: {op: dict(nodes) for op, nodes in
                   backend.pass_manager.ir_lowering.selected_impls.items()}
            for name, backend in backends.items()
        }
        op = 'fused_add_rms_norm' if residual_chain else 'rms_norm'
        expected_count = 2 if residual_chain else 1
        for name in ('core', 'oot'):
            assert len(selected[name][op]) == expected_count
        assert selected['core'] == selected['oot']
        record_property('serving_ir_providers', json.dumps(selected, sort_keys=True))
        record_property('serving_kernel_launches', json.dumps(observed, sort_keys=True))
