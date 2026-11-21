# SPDX-License-Identifier: Apache-2.0
"""Real official projection constructors and fullgraph; CPU GEMM/transport doubles."""
import os
import signal
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from test_runtime_sp_attention import core, sp, values


def make_projection(kind):
    with sp.runtime_sp_model_construction():
        if kind == 'qkv':
            return sp.RuntimeSPQKVParallelLinear(16, 4, 8, 2, bias=False,
                prefix='model.layers.0.self_attn.qkv_proj'), 16
        if kind == 'row':
            return sp.RuntimeSPRowParallelLinear(32, 16, bias=False,
                prefix='model.layers.0.self_attn.o_proj'), 32
        if kind in ('dense_down', 'shared_down'):
            return sp.RuntimeSPRowParallelLinear(32, 16, bias=False,
                prefix='model.layers.0.mlp.' + ('shared_experts.' if kind == 'shared_down' else '') + 'down_proj'), 32
        return sp.RuntimeSPMergedColumnParallelLinear(16, [32, 32], bias=False,
            prefix='model.layers.0.mlp.' + ('shared_experts.' if kind == 'shared_up' else '') + 'gate_up_proj'), 16


@pytest.mark.parametrize('kind', ['qkv', 'row', 'merged'])
def test_real_projection_fullgraph_changed_input(core, monkeypatch, kind):
    # One-rank isolates compilation from native transport; other suites retain
    # the original independent four-rank layout and weight-loader assertions.
    core.world_size = 1
    layer, width = make_projection(kind)
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    with torch.no_grad():
        layer.weight.copy_(values(*layer.weight.shape))
    x = values(4, width, offset=1)
    graphs = []
    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward
    compiled = torch.compile(layer, backend=backend, fullgraph=True)
    try:
        for offset in [1, 3, 7]:
            x.copy_(values(4, width, offset=offset))
            actual = compiled(x)
            reference = layer(x)
            torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
            assert actual[1] is reference[1] is None
        assert len(graphs) == 1
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize('kind', ['qkv', 'row', 'merged', 'dense_down', 'shared_down', 'shared_up'])
@pytest.mark.parametrize('backend', ['eager', 'aot_eager'])
def test_four_rank_fullgraph_keeps_changed_device_mask(core, monkeypatch, kind, backend):
    from hcu_vllm_das.parallel.runtime_sp_batch import RuntimeSPBatch, _BATCH_KEY
    from vllm.forward_context import ForwardContext, override_forward_context
    core.device_communicator = SimpleNamespace(all_to_all_single=lambda out, value: out.copy_(value))
    core.all_gather = lambda value, dim: torch.cat([value + rank for rank in range(4)], dim=dim)
    core.reduce_scatter = lambda value, dim: value.reshape(4, 2, -1).sum(dim=0)
    layer, width = make_projection(kind)
    if kind in ('row', 'dense_down'):
        width //= 4
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    with torch.no_grad():
        layer.weight.copy_(values(*layer.weight.shape))
    batch = RuntimeSPBatch.create(4, 0, 8, torch.zeros(8, dtype=torch.bool), torch.device('cpu'))
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
        is_padding=batch.local_padding, additional_kwargs={_BATCH_KEY: batch})
    x = values(8 if kind in ('row', 'dense_down') else 2, width, offset=1)
    compiled = torch.compile(layer, backend=backend, fullgraph=True)
    try:
        with override_forward_context(ctx):
            for epoch in (0, 1, 2):
                x.copy_(values(*x.shape, offset=epoch + 1))
                batch.global_padding[1::2] = bool(epoch % 2)
                # The same mask storage is read in eager and cached compiled
                # execution. Padded rows contain NaN to reject multiply-by-zero.
                mask = batch.global_padding if x.shape[0] == 8 else batch.local_padding
                x.masked_fill_(mask[:, None], float('nan'))
                actual = compiled(x)
                reference = layer(x)
                torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
                assert torch.isfinite(actual[0]).all()
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize('kind', ['qkv', 'row', 'merged'])
def test_cached_fullgraph_checks_group_before_gemm(core, monkeypatch, kind):
    from torch.utils._python_dispatch import TorchDispatchMode
    core.world_size = 1
    layer, width = make_projection(kind)
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    compiled = torch.compile(layer, backend='eager', fullgraph=True)
    x = torch.zeros(4, width)
    compiled(x)
    monkeypatch.setattr(sp, 'get_tp_group', lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    class NoGemm(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func not in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default), 'GEMM ran after owner failure'
            return func(*args, **(kwargs or {}))
    try:
        with NoGemm(), pytest.raises(RuntimeError, match='ownership changed'):
            compiled(x)
    finally:
        torch._dynamo.reset()


def test_cached_projection_refuses_actual_fork_before_gemm(core):
    core.world_size = 1
    layer, width = make_projection('qkv')
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    compiled = torch.compile(layer, backend='eager', fullgraph=True)
    x = torch.zeros(4, width)
    compiled(x)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        signal.alarm(10)
        try:
            compiled(x)
        except RuntimeError as error:
            os.write(write_fd, str(error).encode())
            os._exit(0)
        os._exit(1)
    os.close(write_fd)
    try:
        message = os.read(read_fd, 4096)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert b'inherited across fork' in message
    finally:
        os.close(read_fd)
        torch._dynamo.reset()


@pytest.mark.parametrize('kind', ['qkv', 'row', 'merged'])
def test_real_projection_default_inductor_fullgraph(core, kind):
    core.world_size = 1
    layer, width = make_projection(kind)
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    with torch.no_grad():
        layer.weight.copy_(values(*layer.weight.shape))
    compiled = torch.compile(layer, fullgraph=True)
    x = values(4, width, offset=1)
    try:
        for offset in (1, 3):
            x.copy_(values(4, width, offset=offset))
            actual, bias = compiled(x)
            reference, reference_bias = layer(x)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            assert bias is reference_bias is None
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize('kind', ['qkv', 'row', 'merged', 'dense_down'])
def test_captured_fx_reads_mask_tensor_inputs_without_python_context(core, kind):
    from hcu_vllm_das.parallel.runtime_sp_batch import RuntimeSPBatch, _BATCH_KEY, current_sp_batch
    from vllm.forward_context import ForwardContext, override_forward_context
    core.world_size = 1
    layer, width = make_projection(kind)
    if kind == 'dense_down':
        width = layer.input_size_per_partition
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    with torch.no_grad():
        layer.weight.copy_(values(*layer.weight.shape))
    x = values(4, width, offset=1)
    batch = RuntimeSPBatch.create(1, 0, 4, torch.zeros(4, dtype=torch.bool), torch.device('cpu'))
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
        is_padding=batch.local_padding, additional_kwargs={_BATCH_KEY: batch})
    captured = []
    def backend(gm, arguments):
        captured.append((gm, list(arguments)))
        return gm.forward
    compiled = torch.compile(layer, backend=backend, fullgraph=True)
    try:
        with override_forward_context(ctx):
            compiled(x)
        assert current_sp_batch() is None
        assert len(captured) == 1
        graph, arguments = captured[0]
        masks = [arg for arg in arguments if isinstance(arg, torch.Tensor) and arg.dtype == torch.bool]
        assert masks, 'padding must be an explicit graph tensor input'
        for mask in masks:
            mask.fill_(True)
        # A captured graph has tensor arguments, but no Python SP batch. This
        # rejects an opaque custom op that silently re-reads capture-time state.
        actual = graph.forward(*arguments)[0]
        with override_forward_context(ctx):
            reference = layer(x)[0]
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        replacement = RuntimeSPBatch.create(1, 0, 4,
            torch.tensor([False, True, False, True]), torch.device('cpu'))
        assert replacement.global_padding.data_ptr() != batch.global_padding.data_ptr()
        ctx.additional_kwargs[_BATCH_KEY] = replacement
        ctx.is_padding = replacement.local_padding
        with override_forward_context(ctx):
            actual = compiled(x)[0]
            reference = layer(x)[0]
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        assert len(captured) == 1
    finally:
        torch._dynamo.reset()
