# SPDX-License-Identifier: Apache-2.0
"""Real Core constructors/loaders; CPU transport/GEMM stand-ins are explicit."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'adapter_extension/src'))
from hcu_vllm_das.parallel import runtime_sp_attention as sp
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers import linear
from vllm.model_executor import parameter


@pytest.fixture
def core(monkeypatch):
    monkeypatch.setattr(sp, '_publication_lifecycle', sp.ProcessLocalLifecycle(
        runtime_action=sp._PUBLICATION_ACTION, error_type=sp.ExtensionCommitError))
    group = SimpleNamespace(world_size=4, rank_in_group=0)
    monkeypatch.setattr(sp, 'get_tp_group', lambda: group)
    monkeypatch.setattr(linear, 'get_tensor_model_parallel_world_size', lambda: 4)
    monkeypatch.setattr(linear, 'get_tensor_model_parallel_rank', lambda: group.rank_in_group)
    monkeypatch.setattr(parameter, 'get_tensor_model_parallel_world_size', lambda: 4)
    monkeypatch.setattr(parameter, 'get_tensor_model_parallel_rank', lambda: group.rank_in_group)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig('cpu'))):
        yield group


def values(rows, cols, offset=0):
    return (((torch.arange(rows * cols).reshape(rows, cols) + offset) % 13) - 6).float() / 8


def local_qkv(qkv, rank, size, heads, kv_heads, hd, vd):
    q, k, v = qkv.split([heads * hd, kv_heads * hd, kv_heads * vd], -1)
    q = q[:, rank * (heads // size) * hd:(rank + 1) * (heads // size) * hd]
    count = max(1, kv_heads // size)
    start = rank * count if kv_heads >= size else rank // (size // kv_heads)
    return torch.cat((q, k[:, start * hd:(start + count) * hd],
                      v[:, start * vd:(start + count) * vd]), -1)


@pytest.mark.parametrize('rank', range(4))
@pytest.mark.parametrize('kv_heads', [1, 2, 8])
@pytest.mark.parametrize('return_bias,skip_bias', [(True, False), (True, True), (False, False)])
def test_real_qkv_loading_exchange_and_reload(core, rank, kv_heads, return_bias, skip_bias):
    core.rank_in_group = rank
    heads, hd, vd, hidden, rows = 8, 4, 2, 16, 3
    weight = values((heads + kv_heads) * hd + kv_heads * vd, hidden)
    bias = values(1, weight.shape[0])[0]
    x = values(rows * 4, hidden, offset=1)
    with sp.runtime_sp_attention_construction():
        layer = sp.RuntimeSPQKVParallelLinear(
            hidden, hd, heads, kv_heads, prefix='model.layers.0.self_attn.qkv_proj',
            v_head_size=vd, bias=True, skip_bias_add=skip_bias, return_bias=return_bias,
        )
    assert layer.tp_size == 1 and layer.tp_rank == 0
    assert layer.weight_loader_v2.__func__ is linear.QKVParallelLinear.weight_loader_v2
    assert layer.weight.shape == weight.shape
    # CPU contracts isolate the layout from the platform-specific GEMM dispatcher.
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    for epoch in [0, 2]:
        w, b = weight + epoch / 8, bias + epoch / 8
        if epoch == 0:
            offset = 0
            for name, width in [('q', heads * hd), ('k', kv_heads * hd), ('v', kv_heads * vd)]:
                layer.weight.weight_loader(layer.weight, w[offset:offset + width], name)
                layer.bias.weight_loader(layer.bias, b[offset:offset + width], name)
                offset += width
        else:
            layer.weight.weight_loader(layer.weight, w)
            layer.bias.weight_loader(layer.bias, b)
        torch.testing.assert_close(layer.weight, w, rtol=0, atol=0)
        torch.testing.assert_close(layer.bias, b, rtol=0, atol=0)
        projected = F.linear(x, w, None if skip_bias else b)
        expected = local_qkv(projected, rank, 4, heads, kv_heads, hd, vd)
        called = []
        def exchange(out, send):
            for dest in range(4):
                target = local_qkv(projected[rank * rows:(rank + 1) * rows],
                                   dest, 4, heads, kv_heads, hd, vd)
                torch.testing.assert_close(send[dest], target, rtol=0, atol=0)
            out.copy_(expected.reshape_as(out))
            called.append(True)
        core.device_communicator = SimpleNamespace(all_to_all_single=exchange)
        result = layer(x[rank * rows:(rank + 1) * rows])
        out, actual_bias = result if return_bias else (result, None)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        if skip_bias:
            torch.testing.assert_close(actual_bias, local_qkv(b[None], rank, 4,
                heads, kv_heads, hd, vd)[0], rtol=0, atol=0)
        else:
            assert actual_bias is None
        assert called == [True]


@pytest.mark.parametrize('rank', range(4))
@pytest.mark.parametrize('return_bias,skip_bias', [(True, False), (True, True), (False, False)])
def test_real_row_loading_inverse_exchange_and_reload(core, rank, return_bias, skip_bias):
    core.rank_in_group = rank
    width, hidden, rows = 32, 16, 3
    all_heads = values(rows * 4, width)
    weight = values(hidden, width, offset=3)
    bias = values(1, hidden)[0]
    with sp.runtime_sp_attention_construction():
        layer = sp.RuntimeSPRowParallelLinear(width, hidden, bias=True,
            skip_bias_add=skip_bias, return_bias=return_bias,
            prefix='model.layers.0.self_attn.o_proj')
    assert layer.weight.shape == weight.shape and layer.tp_size == 1
    assert layer.weight_loader_v2.__func__ is linear.RowParallelLinear.weight_loader_v2
    layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    local_heads = all_heads[:, rank * (width // 4):(rank + 1) * (width // 4)].contiguous()
    called = []
    def exchange(out, send):
        for dest in range(4):
            torch.testing.assert_close(send[dest], local_heads[dest * rows:(dest + 1) * rows], rtol=0, atol=0)
        for source in range(4):
            out[source].copy_(all_heads[rank * rows:(rank + 1) * rows,
                source * (width // 4):(source + 1) * (width // 4)])
        called.append(True)
    core.device_communicator = SimpleNamespace(all_to_all_single=exchange)
    for epoch in [0, 2]:
        w, b = weight + epoch / 8, bias + epoch / 8
        layer.weight.weight_loader(layer.weight, w)
        layer.bias.weight_loader(layer.bias, b)
        torch.testing.assert_close(layer.weight, w, rtol=0, atol=0)
        result = layer(local_heads)
        out, actual_bias = result if return_bias else (result, None)
        expected = F.linear(all_heads[rank * rows:(rank + 1) * rows], w, None if skip_bias else b)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        if skip_bias:torch.testing.assert_close(actual_bias, b, rtol=0, atol=0)
        else:assert actual_bias is None
    assert called == [True, True]
    with pytest.raises(ValueError, match='padded'):
        layer(local_heads[:5])
    assert len(called) == 2


def test_construction_scope_and_unrelated_prefix_retain_official_tp(core):
    for cls, args, name in [(sp.RuntimeSPQKVParallelLinear, (16, 4, 8, 2), 'qkv_proj'),
                            (sp.RuntimeSPRowParallelLinear, (32, 16), 'o_proj')]:
        layer = cls(*args, bias=False, prefix='model.layers.0.self_attn.' + name)
        assert not layer._runtime_sp and layer.tp_size == 4
        with pytest.raises(RuntimeError, match='scope unwind'):
            with sp.runtime_sp_attention_construction():
                assert cls(*args, bias=False, prefix='model.layers.0.self_attn.' + name)._runtime_sp
                other = cls(*args, bias=False, prefix='other.' + name)
                assert not other._runtime_sp and other.tp_size == 4
                raise RuntimeError('scope unwind')
        assert not cls(*args, bias=False, prefix='model.layers.0.self_attn.' + name)._runtime_sp


def test_tp_owner_change_refuses_before_any_gemm(core, monkeypatch):
    with sp.runtime_sp_attention_construction():
        layer = sp.RuntimeSPQKVParallelLinear(16, 4, 8, 2, bias=False,
            prefix='model.layers.0.self_attn.qkv_proj')
    layer.quant_method.apply = lambda *a: pytest.fail('GEMM must not run')
    monkeypatch.setattr(sp, 'get_tp_group', lambda: SimpleNamespace(world_size=4, rank_in_group=0))
    with pytest.raises(RuntimeError, match='ownership'):
        layer(torch.zeros(3, 16))


def test_public_oot_registration_covers_concrete_classes(core, monkeypatch):
    for cls, _ in sp._BINDINGS:monkeypatch.delitem(op_registry_oot, cls.__name__, raising=False)
    sp.install_attention_layers()
    try:
        sp.install_attention_layers()
        with sp.runtime_sp_attention_construction():
            qkv = linear.QKVParallelLinear(16, 4, 8, 2, bias=False,
                prefix='model.layers.0.self_attn.qkv_proj')
            row = linear.RowParallelLinear(32, 16, bias=False,
                prefix='model.layers.0.self_attn.o_proj')
        assert type(qkv) is sp.RuntimeSPQKVParallelLinear and qkv._runtime_sp
        assert type(row) is sp.RuntimeSPRowParallelLinear and row._runtime_sp
        monkeypatch.setitem(op_registry_oot, 'RowParallelLinear', object())
        with pytest.raises(RuntimeError, match='ownership'):sp.install_attention_layers()
        with pytest.raises(RuntimeError, match='ownership'):sp.check_attention_layers()
    finally:
        for cls, _ in sp._BINDINGS:op_registry_oot.pop(cls.__name__, None)


def test_registration_preflight_does_not_publish_before_rejecting_foreign(core, monkeypatch):
    for cls, _ in sp._BINDINGS:monkeypatch.delitem(op_registry_oot, cls.__name__, raising=False)
    monkeypatch.setitem(op_registry_oot, 'RowParallelLinear', object())
    with pytest.raises(RuntimeError, match='foreign'):sp.install_attention_layers()
    assert 'QKVParallelLinear' not in op_registry_oot


def test_partial_publication_stays_failed_after_registry_repair(core, monkeypatch):
    for cls, _ in sp._BINDINGS:
        monkeypatch.delitem(op_registry_oot, cls.__name__, raising=False)
    original = linear.RowParallelLinear.register_oot
    calls = []
    def fail(candidate):
        calls.append(candidate)
        raise RuntimeError('injected second publication failure')
    monkeypatch.setattr(linear.RowParallelLinear, 'register_oot', fail)
    with pytest.raises(sp.ExtensionCommitError, match='publication incomplete'):
        sp.install_attention_layers()
    assert op_registry_oot['QKVParallelLinear'] is sp.RuntimeSPQKVParallelLinear
    monkeypatch.setattr(linear.RowParallelLinear, 'register_oot', original)
    monkeypatch.delitem(op_registry_oot, 'QKVParallelLinear')
    for action in [sp.install_attention_layers, sp.check_attention_layers]:
        with pytest.raises(sp.ExtensionCommitError, match='publication incomplete'):
            action()
    assert calls == [sp.RuntimeSPRowParallelLinear]
    assert all(cls.__name__ not in op_registry_oot for cls, _ in sp._BINDINGS)


def test_concurrent_publication_publishes_each_real_owner_once(core, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    start, lock, calls = Barrier(4), Lock(), []
    for cls, _ in sp._BINDINGS:
        monkeypatch.delitem(op_registry_oot, cls.__name__, raising=False)
        original = cls.register_oot
        def register(candidate, original=original, cls=cls):
            with lock:
                calls.append(cls.__name__)
            return original(candidate)
        monkeypatch.setattr(cls, 'register_oot', register)
    def install(_):
        start.wait(timeout=10)
        sp.install_attention_layers()
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(install, range(4)))
        assert sorted(calls) == ['MergedColumnParallelLinear', 'QKVParallelLinear', 'RowParallelLinear']
        sp.check_attention_layers()
    finally:
        for cls, _ in sp._BINDINGS:
            op_registry_oot.pop(cls.__name__, None)


@pytest.mark.parametrize('rank', range(4))
def test_real_fp8_weight_and_scale_loaders_use_replicated_geometry(core, rank, monkeypatch):
    from vllm.model_executor.layers.quantization import fp8
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    # Allocation/loader contract only: a hidden-device CPU process cannot
    # admit a real GPU ScaledMM provider. Do not report this as FP8 execution.
    monkeypatch.setattr(fp8, 'init_fp8_linear_kernel', lambda **kwargs: object())
    core.rank_in_group = rank
    get_current_vllm_config().model_config = SimpleNamespace(dtype=torch.float32)
    with sp.runtime_sp_attention_construction():
        layer = sp.RuntimeSPQKVParallelLinear(16, 4, 8, 2, bias=False,
            quant_config=Fp8Config(True, 'static'), prefix='model.layers.0.self_attn.qkv_proj')
    assert layer.tp_size == 1 and layer.tp_rank == 0
    assert layer.weight.shape == (48, 16)
    assert layer.weight_scale.shape == (3,) and layer.input_scale.shape == (3,)
    for epoch in [0, 1]:
        offset = 0
        for index, (name, width) in enumerate([('q', 32), ('k', 8), ('v', 8)]):
            w = values(width, 16, offset=epoch).to(layer.weight.dtype)
            layer.weight.weight_loader(layer.weight, w, name)
            layer.weight_scale.weight_loader(layer.weight_scale, torch.tensor(index + epoch + 1.0), name)
            layer.input_scale.weight_loader(layer.input_scale, torch.tensor(index + epoch + 4.0), name)
            assert torch.equal(layer.weight[offset:offset + width].float(), w.float())
            offset += width
        torch.testing.assert_close(layer.weight_scale, torch.tensor([1., 2., 3.]) + epoch, rtol=0, atol=0)
        torch.testing.assert_close(layer.input_scale, torch.tensor([4., 5., 6.]) + epoch, rtol=0, atol=0)


@pytest.mark.parametrize('geometry', [(4, 4, 8, 2, 4, 4), (4, 0, 6, 2, 4, 4),
    (4, 0, 8, 3, 4, 4), (4, 0, 8, 2, 0, 4), (True, 0, 8, 2, 4, 4)])
def test_invalid_geometry_fails_before_transport(geometry):
    with pytest.raises(ValueError):sp.AttentionLayout(*geometry)


def test_inherited_pid_refuses_before_projection(core):
    with sp.runtime_sp_attention_construction():
        layer = sp.RuntimeSPQKVParallelLinear(16, 4, 8, 2, bias=False,
            prefix='model.layers.0.self_attn.qkv_proj')
    layer._runtime_sp_pid -= 1
    layer.quant_method.apply = lambda *a: pytest.fail('GEMM must not run')
    with pytest.raises(RuntimeError, match='inherited across fork'):
        layer(torch.zeros(3, 16))
