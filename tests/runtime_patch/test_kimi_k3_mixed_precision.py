# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; native vLLM loaders, no accelerator/kernel initialization."""
import inspect
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KimiGDNMergedColumnParallelLinear,
)
from vllm_hcu.model_executor.layers.kimi_k3_mixed_precision import (
    KimiK3MixedKDAProjection,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8Config, SlimQuantW4A8Int8LinearMethod,
)
from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_mixed_linear as patch
from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_mixed_kda as kda_patch


def config():
    return SlimQuantW4A8Int8Config.from_config({
        "ignore": [r"re:.*self_attn\.(g_proj|b_proj|f_a_proj|f_b_proj)(?:\.|$)"],
        "mixed_precision": {"w8a8_include": {"modules": [
            "language_model.model.layers.*.self_attn.q_proj",
            "language_model.model.layers.*.self_attn.k_proj",
            "language_model.model.layers.*.self_attn.v_proj",
            "language_model.model.layers.*.self_attn.q_a_proj",
            "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa",
            "language_model.model.layers.*.mlp.gate_proj",
            "language_model.model.layers.*.mlp.up_proj",
            "language_model.model.layers.*.block_sparse_moe.routed_expert_*_proj",
        ]}}})


def _parameter(shape, dtype, loader):
    value = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
    value.output_dim = 0
    value.weight_loader = loader
    return value


def _cpu_factories(tp_size, rank):
    class CPUColumn(nn.Module):
        # Reuse the installed loader without constructing a GPU linear kernel.
        weight_loader = ColumnParallelLinear.weight_loader

        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.tp_rank = rank
            self.weight = _parameter((output_size // tp_size, input_size),
                                     torch.int8, self.weight_loader)
            self.weight_scale = _parameter((output_size // tp_size, 1),
                                           torch.float32, self.weight_loader)

        def forward(self, x):
            # Numerical reference only; this is not the HCU W8A8 kernel.
            scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
            activation = (x / scale).round().clamp(-127, 127) * scale
            return activation @ (self.weight.float() * self.weight_scale).T, None

    # Keep the native replicated-shard method's real MRO while bypassing
    # the constructor and GPU kernel selection.
    class NativeCPUMerged(_KimiGDNMergedColumnParallelLinear):
        def __init__(self, input_size, output_sizes, replicated_shard_id,
                     tp_size, **kwargs):
            nn.Module.__init__(self)
            self.tp_rank, self.tp_size = rank, tp_size
            self.replicated_shard_id = replicated_shard_id
            self.output_sizes = list(output_sizes)
            self.output_sizes[replicated_shard_id] *= tp_size
            self.weight = _parameter((sum(self.output_sizes) // tp_size, input_size),
                                     torch.float32, self.weight_loader)

        def forward(self, x):
            return x @ self.weight.T, None

    return CPUColumn, NativeCPUMerged


def _target(native_load=None, column_linear=None):
    target = ModuleType(patch.TARGET_MODULE)

    class Linear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.config = kwargs.get("quant_config")

    class MoE(nn.Module):
        def __init__(self, config, quant_config=None, prefix="", fail=False, sharded=False):
            super().__init__()
            self.down = target.ReplicatedLinear(
                quant_config=None, prefix=prefix + ".routed_expert_down_proj")
            up_linear = target.ColumnParallelLinear if sharded else target.ReplicatedLinear
            self.up = up_linear(
                quant_config=None, prefix=prefix + ".routed_expert_up_proj",
                **({"gather_output": True} if sharded else {}))
            self.other = target.ReplicatedLinear(quant_config=None, prefix=prefix + ".other")
            # The runner must retain the same quantized module objects.
            self.runner_refs = (self.down, self.up)
            if fail:
                raise RuntimeError("construction failed")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.other_entries = []

        def load_weights(self, weights):
            self.other_entries.extend(weights)
            return {entry[0] for entry in self.other_entries}

    class CausalLM:
        def __init__(self, *, vllm_config, prefix=""):
            self.quant_config = vllm_config.quant_config

    if native_load is not None:
        Model.load_weights = native_load
    target.ReplicatedLinear = Linear
    target.ColumnParallelLinear = column_linear or Linear
    target.KimiMoE, target.KimiLinearModel = MoE, Model
    target.KimiLinearForCausalLM = CausalLM
    patch.apply_to_module(target)
    return target


@pytest.mark.parametrize("sharded", [False, True])
def test_fused_w8a8_selection_and_latent_constructor_scope(sharded):
    target = _target()
    cfg = config()
    assert not patch.apply_to_module(target)
    target.KimiLinearForCausalLM(vllm_config=SimpleNamespace(quant_config=cfg))
    for prefix in ("language_model.model.layers.4.self_attn.fused_qkv_a_proj",
                   "language_model.model.layers.0.mlp.gate_up_proj"):
        layer = ReplicatedLinear.__new__(ReplicatedLinear)
        nn.Module.__init__(layer)
        assert isinstance(cfg.get_quant_method(layer, prefix), SlimQuantW4A8Int8LinearMethod)
    moe = target.KimiMoE(None, cfg, "language_model.model.layers.1.block_sparse_moe", sharded=sharded)
    assert moe.down.config is cfg and moe.up.config is cfg
    assert moe.runner_refs == (moe.down, moe.up)
    assert moe.other.config is None
    assert target.KimiMoE(None).down.config is None
    with pytest.raises(RuntimeError, match="construction failed"):
        target.KimiMoE(None, cfg, "x", fail=True)
    assert target.ReplicatedLinear(prefix="x.routed_expert_up_proj").config is None
    assert target.ColumnParallelLinear(prefix="x.routed_expert_up_proj").config is None


@pytest.mark.parametrize("tp_size", [1, 16])
def test_sharded_latent_up_loads_int8_weights_and_scales(tp_size):
    cfg = config()
    weights = torch.arange(32 * 4).remainder(17).to(torch.int8).view(32, 4)
    scales = torch.linspace(.02, .12, 32).view(-1, 1)
    x = torch.tensor([[1., 2., 3., 4.]])
    sx = x.abs().amax(-1, keepdim=True) / 127
    xq = (x / sx).round().clamp(-127, 127) * sx
    for rank in range(tp_size):
        column, _ = _cpu_factories(tp_size, rank)

        class CheckedColumn(column):
            def __init__(self, **kwargs):
                assert kwargs['quant_config'] is cfg
                assert kwargs.pop('gather_output') is True
                probe = ColumnParallelLinear.__new__(ColumnParallelLinear)
                nn.Module.__init__(probe)
                assert isinstance(cfg.get_quant_method(probe, kwargs['prefix']),
                                  SlimQuantW4A8Int8LinearMethod)
                super().__init__(4, 32, **kwargs)

        target = _target(column_linear=CheckedColumn)
        moe = target.KimiMoE(None, cfg, 'language_model.model.layers.1.block_sparse_moe',
                            sharded=True)
        up = moe.up
        up.weight.weight_loader(up.weight, weights)
        up.weight_scale.weight_loader(up.weight_scale, scales)
        expected_w = weights.chunk(tp_size)[rank]
        expected_s = scales.chunk(tp_size)[rank]
        torch.testing.assert_close(up.weight, expected_w)
        torch.testing.assert_close(up.weight_scale, expected_s)
        torch.testing.assert_close(up(x)[0], xq @ (expected_w.float() * expected_s).T)
        assert moe.runner_refs[1] is up


@pytest.mark.parametrize("tp_size", [1, 16])
def test_mixed_kda_loading_tp_rows_scales_replication_padding_and_output(tp_size):
    cfg = config()
    target = _target()
    torch.manual_seed(7)
    width, head_dim, heads, hidden, padding = 32, 8, 16, 4, 3
    weights = {}
    for branch in ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"):
        rows = head_dim if branch == "f_a_proj" else heads if branch == "b_proj" else width
        quant = branch in ("q_proj", "k_proj", "v_proj")
        weights[branch, "weight"] = (torch.randint(-12, 12, (rows, hidden), dtype=torch.int8)
                                      if quant else torch.randn(rows, hidden))
        if quant:
            weights[branch, "weight_scale"] = torch.linspace(.02, .13, rows).view(-1, 1)
    x = torch.randn(3, hidden)
    for rank in range(tp_size):
        column, merged = _cpu_factories(tp_size, rank)
        projection = KimiK3MixedKDAProjection(
            hidden, [width] * 4 + [head_dim, heads, padding * tp_size],
            4, tp_size, column_linear=column, merged_linear=merged, quant_config=cfg,
            prefix="language_model.model.layers.1.self_attn.in_proj_qkvgfab", bias=False)
        model = target.KimiLinearModel()
        model.attn = nn.Module()
        model.attn.in_proj_qkvgfab = projection
        entries = [(f"attn.{branch}.{suffix}", value)
                   for (branch, suffix), value in reversed(list(weights.items()))]
        entries.append(("other.weight", torch.ones(1)))
        loaded = model.load_weights(iter(entries))
        assert model.other_entries[0][0] == "other.weight"
        assert len(model.other_entries) == 1
        assert len(loaded) == 8  # 6 QKV params + floating gate + other
        assert torch.count_nonzero(projection.weight[-padding:]) == 0
        expected = []
        sx = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
        xq = (x / sx).round().clamp(-127, 127) * sx
        for branch in ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"):
            w = weights[branch, "weight"].float()
            quant = (branch, "weight_scale") in weights
            if quant:
                w *= weights[branch, "weight_scale"]
            if branch != "f_a_proj":
                w = w.chunk(tp_size, dim=0)[rank]
            expected.append((xq if quant else x) @ w.T)
        expected.append(torch.zeros(x.shape[0], padding))
        torch.testing.assert_close(projection(x)[0], torch.cat(expected, dim=-1))
        assert projection.fp_gate.tp_rank == rank
        with pytest.raises(TypeError, match="INT8"):
            projection.load_branch("q_proj", "weight", weights["q_proj", "weight"].float())
        with pytest.raises(ValueError, match="floating gate"):
            projection.load_branch("g_proj", "weight_scale", torch.ones(width, 1))


def test_kda_callback_keeps_legacy_constructor():
    target = ModuleType(kda_patch.TARGET_MODULE)
    calls = []
    target._KimiGDNMergedColumnParallelLinear = lambda *a, **k: calls.append((a, k))
    target.ColumnParallelLinear = object
    assert kda_patch.apply_to_module(target)
    assert not kda_patch.apply_to_module(target)
    target._KimiGDNMergedColumnParallelLinear(4, [8] * 6, quant_config=None)
    assert len(calls) == 1


@pytest.mark.parametrize("selection", ["empty", "mlp", "other_layer", "ignored"])
def test_kda_callback_preserves_unquantized_qkv(selection):
    cfg = config()
    parent = "language_model.model.layers.1.self_attn"
    if selection == "empty":
        cfg.w8a8_include = []
    elif selection == "mlp":
        cfg.w8a8_include = ["language_model.model.layers.*.mlp.*"]
    elif selection == "other_layer":
        cfg.w8a8_include = ["language_model.model.layers.0.self_attn.*"]
    else:
        cfg.ignore.append(parent + ".*")
    # Verify these configurations really select floating Q/K/V in production.
    for branch in ("q_proj", "k_proj", "v_proj"):
        layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
        nn.Module.__init__(layer)
        assert isinstance(cfg.get_quant_method(layer, f"{parent}.{branch}"),
                          UnquantizedLinearMethod)

    target = ModuleType(kda_patch.TARGET_MODULE)
    calls = []
    sentinel = object()

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    def forbidden_column(*args, **kwargs):
        pytest.fail("Floating Q/K/V must retain the native merged constructor")

    target._KimiGDNMergedColumnParallelLinear = native
    target.ColumnParallelLinear = forbidden_column
    kda_patch.apply_to_module(target)
    args = (4, [32] * 4 + [8, 16], 4, 1)
    kwargs = dict(quant_config=cfg, prefix=f"{parent}.in_proj_qkvgfab", bias=False)
    assert target._KimiGDNMergedColumnParallelLinear(*args, **kwargs) is sentinel
    assert calls == [(args, kwargs)]


def test_kda_callback_keeps_selected_qkv_quantized():
    target = ModuleType(kda_patch.TARGET_MODULE)
    column, merged = _cpu_factories(1, 0)
    target.ColumnParallelLinear = column
    target._KimiGDNMergedColumnParallelLinear = merged
    kda_patch.apply_to_module(target)
    projection = target._KimiGDNMergedColumnParallelLinear(
        4, [32] * 4 + [8, 16], 4, 1, quant_config=config(),
        prefix="language_model.model.layers.1.self_attn.in_proj_qkvgfab", bias=False)
    assert isinstance(projection, KimiK3MixedKDAProjection)
    assert all(getattr(projection, branch).weight.dtype == torch.int8
               for branch in ("q_proj", "k_proj", "v_proj"))
    assert projection.fp_gate.weight.is_floating_point()


def test_callbacks_registered():
    from vllm_hcu.patch.worker import worker_callback_names
    callbacks = dict(worker_callback_names())
    assert callbacks[patch.PATCH_ID] == patch.TARGET_MODULE
    assert callbacks[kda_patch.PATCH_ID] == kda_patch.TARGET_MODULE


@pytest.mark.parametrize("replicated", [False, True])
def test_fused_w8a8_weights_and_scales_use_same_native_tp_rows(replicated):
    # Dense/shared gate-up uses TP16; MLA Q-A/KV-A is explicitly replicated.
    for rank in range(16):
        layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
        nn.Module.__init__(layer)
        layer.tp_size = 1 if replicated else 16
        layer.tp_rank = 0 if replicated else rank
        layer.output_sizes = [32, 16]
        local_widths = [size // layer.tp_size for size in layer.output_sizes]
        weight = _parameter((sum(local_widths), 4), torch.int8, layer.weight_loader)
        scale = _parameter((sum(local_widths), 1), torch.float32, layer.weight_loader)
        expected_weights, expected_scales = [], []
        for shard, size in enumerate(layer.output_sizes):
            w = torch.arange(size * 4).remainder(127).to(torch.int8).view(size, 4)
            s = torch.linspace(.01, .09, size).view(size, 1)
            layer.weight_loader(weight, w, shard)
            layer.weight_loader(scale, s, shard)
            expected_weights.append(w.chunk(layer.tp_size)[layer.tp_rank])
            expected_scales.append(s.chunk(layer.tp_size)[layer.tp_rank])
        torch.testing.assert_close(weight, torch.cat(expected_weights))
        torch.testing.assert_close(scale, torch.cat(expected_scales))


def test_mixed_projection_is_not_used_by_preload_bf16_warmup(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_kda_gemm_warmup as warmup
    monkeypatch.setattr(warmup, "_warmup_done", False)
    target = ModuleType(warmup.TARGET_MODULE)

    class Projection:
        _hcu_mixed_kda = True

        @property
        def weight(self):
            raise AssertionError("preload warmup must not access mixed INT8 branches")

    class Attention:
        def __init__(self):
            self.in_proj_qkvgfab = Projection()

    target.KimiK3DeltaAttention = Attention
    assert warmup.apply_to_module(target)
    target.KimiK3DeltaAttention()
    assert warmup._warmup_done is False


def test_mixed_and_fused_branches_compose_with_installed_kimi_model_loader():
    from vllm.models.kimi_k3.amd.linear import KimiLinearModel
    target = _target(native_load=KimiLinearModel.load_weights)
    cfg = config()
    for rank in range(16):
        column, merged = _cpu_factories(16, rank)
        model = target.KimiLinearModel()
        model.config = SimpleNamespace(linear_attn_config={'use_full_rank_gate': True},
                                       q_lora_rank=32, is_moe=False,
                                       num_hidden_layers=2, num_nextn_predict_layers=0,
                                       is_linear_attn=True)
        model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        model.layers[0].self_attn = nn.Module()
        proj = KimiK3MixedKDAProjection(
            4, [32] * 4 + [8, 16, 16], 4, 16, column_linear=column,
            merged_linear=merged, quant_config=cfg,
            prefix='language_model.model.layers.0.self_attn.in_proj_qkvgfab', bias=False)
        model.layers[0].self_attn.in_proj_qkvgfab = proj

        def fused(sizes, replicated=False):
            layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
            nn.Module.__init__(layer)
            layer.tp_size, layer.tp_rank = (1, 0) if replicated else (16, rank)
            layer.output_sizes = sizes
            # Hermetic vs the installed linear class: __new__ skips the
            # constructor, so materialize what a body OR plugin-fork
            # weight_loader reads off self. The stub stands in for a W8A8
            # layer (int8 weight + weight_scale), i.e. a QUANTIZED one --
            # the fork's loader takes the NN-layout path under
            # VLLM_USE_NN when is_quantization is False (param narrowed on
            # dim 1 + transposed shard), which no realistic deployment
            # binding would produce for this layer.
            layer.is_quantization = True
            local = sum(sizes) // layer.tp_size
            layer.weight = _parameter((local, 4), torch.int8, layer.weight_loader)
            layer.weight_scale = _parameter((local, 1), torch.float32, layer.weight_loader)
            return layer

        model.layers[0].mlp = nn.Module()
        dense = fused([32, 32])
        model.layers[0].mlp.gate_up_proj = dense
        model.layers[1].self_attn = nn.Module()
        mla = fused([32, 16], replicated=True)
        model.layers[1].self_attn.fused_qkv_a_proj = mla
        model.layers[1].block_sparse_moe = nn.Module()
        shared = fused([32, 32])
        model.layers[1].block_sparse_moe.shared_experts = nn.Module()
        model.layers[1].block_sparse_moe.shared_experts.gate_up_proj = shared
        model.direct = nn.Module()
        model.direct.weight = _parameter((1, 4), torch.float32, lambda p, t: p.data.copy_(t))
        entries = []
        def branch_tensors(rows, seed, quant=True):
            values = torch.arange(rows * 4).view(rows, 4) + seed * 19
            weight = (values.remainder(127).to(torch.int8) if quant
                      else values.float() / 128)
            scale = (torch.arange(rows).float() + 1 + seed * 64).view(rows, 1) / 1024
            return weight, scale

        for seed, (branch, rows) in enumerate([
            ('q_proj',32),('k_proj',32),('v_proj',32),
            ('g_proj',32),('f_a_proj',8),('b_proj',16),
        ]):
            quant = branch in ('q_proj','k_proj','v_proj')
            weight, scale = branch_tensors(rows, seed, quant)
            entries.append((f'layers.0.self_attn.{branch}.weight', weight))
            if quant:
                entries.append((f'layers.0.self_attn.{branch}.weight_scale', scale))
        for group, (parent, names, sizes) in enumerate([
            ('layers.0.mlp', ['gate_proj','up_proj'], [32,32]),
            ('layers.1.self_attn', ['q_a_proj','kv_a_proj_with_mqa'], [32,16]),
            ('layers.1.block_sparse_moe.shared_experts', ['gate_proj','up_proj'], [32,32]),
        ]):
            for branch, (n, rows) in enumerate(zip(names, sizes)):
                weight, scale = branch_tensors(rows, 6 + group * 2 + branch)
                entries.append((parent+'.'+n+'.weight', weight))
                entries.append((parent+'.'+n+'.weight_scale', scale))
        entries.append(('direct.weight',torch.ones(1,4)))
        expected = {name: tensor.clone() for name, tensor in entries}
        loaded = model.load_weights(iter(reversed(entries)))
        # Environment fingerprint: a CI-only loader delta (py version, vLLM
        # build, loader module resolution) must be visible in the failure
        # log without a local repro. No repr of the bound loader -- it
        # reaches extra_repr on a __new__-built layer and would raise
        # AttributeError inside the diagnostic itself.
        import vllm as _vllm_mod

        dense_loader = dense.weight.weight_loader
        loader_func = getattr(dense_loader, "__func__", dense_loader)
        loader_id = (
            f"{getattr(loader_func, '__module__', '?')}."
            f"{getattr(loader_func, '__qualname__', type(loader_func).__name__)}"
        )
        loaded_set, param_names = (
            set(loaded), {name for name, _ in model.named_parameters()}
        )
        # Self-diagnosing: pin WHICH side diverged, which loader ran, and
        # which vLLM build served the native K3 loader.
        assert loaded_set == param_names, (
            f"loader skipped {sorted(param_names - loaded_set)} and invented "
            f"{sorted(loaded_set - param_names)}; dense loader is "
            f"{loader_id}; K3 native load_weights sourcefile "
            f"{inspect.getsourcefile(type(model).load_weights)}; "
            f"vllm {_vllm_mod.__version__} from {_vllm_mod.__file__}"
        )

        def local_rows(tensor, replicated=False):
            if replicated:
                return tensor
            width = tensor.shape[0] // 16
            return tensor.narrow(0, rank * width, width)

        for layer, parent, names, replicated in [
            (dense, 'layers.0.mlp', ['gate_proj', 'up_proj'], False),
            (mla, 'layers.1.self_attn', ['q_a_proj', 'kv_a_proj_with_mqa'], True),
            (shared, 'layers.1.block_sparse_moe.shared_experts',
             ['gate_proj', 'up_proj'], False),
        ]:
            for suffix in ('weight', 'weight_scale'):
                rows = [local_rows(expected[f'{parent}.{name}.{suffix}'], replicated)
                        for name in names]
                torch.testing.assert_close(getattr(layer, suffix), torch.cat(rows),
                                           rtol=0, atol=0)
        for branch in ('q_proj', 'k_proj', 'v_proj'):
            for suffix in ('weight', 'weight_scale'):
                tensor = expected[f'layers.0.self_attn.{branch}.{suffix}']
                torch.testing.assert_close(getattr(getattr(proj, branch), suffix),
                                           local_rows(tensor), rtol=0, atol=0)
        gate_rows = [local_rows(expected[f'layers.0.self_attn.{name}.weight'],
                                replicated=name == 'f_a_proj')
                     for name in ('g_proj', 'f_a_proj', 'b_proj')]
        gate_rows.append(torch.zeros(1, 4))  # One local output-padding row.
        torch.testing.assert_close(proj.fp_gate.weight, torch.cat(gate_rows),
                                   rtol=0, atol=0)
        assert torch.all(model.direct.weight == 1)
