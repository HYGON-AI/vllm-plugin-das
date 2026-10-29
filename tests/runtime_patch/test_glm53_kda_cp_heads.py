# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for CP-head loading, sizing and output ownership.

These tests do not establish fused-kernel accuracy or multi-HCU performance.
"""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.v1.glm53_kda_cp_config import (
    KDA_CP_HEAD_ENV,
    KdaHeadParallelPlan,
    kda_cp_head_size,
    resolve_kda_head_parallel,
    validate_glm53_pcp_topology,
)


def _config(**overrides):
    parallel = dict(
        tensor_parallel_size=1, prefill_context_parallel_size=8,
        pipeline_parallel_size=1, data_parallel_size=1,
        decode_context_parallel_size=1, enable_expert_parallel=True,
    )
    parallel.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_config=SimpleNamespace(
                linear_num_heads=64, linear_head_dim=2, linear_conv_kernel_dim=4,
            ),
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )


def test_head_path_is_opt_in(monkeypatch):
    monkeypatch.delenv(KDA_CP_HEAD_ENV, raising=False)
    assert kda_cp_head_size(_config()) is None
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    for size in (2, 4, 8):
        assert kda_cp_head_size(_config(prefill_context_parallel_size=size)) == size
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        kda_cp_head_size(_config())


@pytest.mark.parametrize("override", [
    {"prefill_context_parallel_size": 0},
    {"prefill_context_parallel_size": 3},
    {"tensor_parallel_size": 0}, {"pipeline_parallel_size": 0},
    {"data_parallel_size": -1}, {"decode_context_parallel_size": 0},
])
def test_reject_invalid_parallel_sizes(monkeypatch, override):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    with pytest.raises(ValueError):
        kda_cp_head_size(_config(**override))


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_pcp1_keeps_native_tp_even_when_switch_is_on(monkeypatch, tp_size):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    cfg = _config(tensor_parallel_size=tp_size, prefill_context_parallel_size=1,
                  enable_expert_parallel=False)
    cfg.model_config.enforce_eager = False
    assert kda_cp_head_size(cfg) is None


@pytest.mark.parametrize("heads", [0, -64, 63])
def test_reject_invalid_head_count(monkeypatch, heads):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    cfg = _config(prefill_context_parallel_size=4)
    cfg.model_config.hf_config.linear_num_heads = heads
    with pytest.raises(ValueError, match="head count.*divisible"):
        kda_cp_head_size(cfg)


def test_head_config_uses_text_config_without_requiring_eager(monkeypatch):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    cfg = _config(prefill_context_parallel_size=4)
    cfg.model_config.hf_text_config = SimpleNamespace(linear_num_heads=64)
    cfg.model_config.hf_config.linear_num_heads = 63
    assert kda_cp_head_size(cfg) == 4
    cfg.model_config.enforce_eager = False
    assert kda_cp_head_size(cfg) == 4


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_cp_plan_selects_one_head_group_not_tp_times_cp(monkeypatch, tp_size, cp_size):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    plan = resolve_kda_head_parallel(_config(
        tensor_parallel_size=tp_size, prefill_context_parallel_size=cp_size,
    ))
    assert plan == KdaHeadParallelPlan("pcp", cp_size)
    assert plan.local_num_heads(64) == 64 // cp_size


@pytest.mark.parametrize("override", [
    {"tensor_parallel_size": 2}, {"pipeline_parallel_size": 2},
    {"data_parallel_size": 2}, {"decode_context_parallel_size": 2},
    {"enable_expert_parallel": False},
])
def test_relaxed_topology_admission_keeps_pcp_head_plan(monkeypatch, override):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    cfg = _config(**override)
    assert resolve_kda_head_parallel(cfg) == KdaHeadParallelPlan("pcp", 8)
    # Admission alone is not proof of full-model execution support.
    assert validate_glm53_pcp_topology(cfg) == 8
    assert kda_cp_head_size(cfg) == 8


def test_head_plan_does_not_depend_on_graph_admission(monkeypatch):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    cfg = _config()
    cfg.model_config.enforce_eager = False
    assert resolve_kda_head_parallel(cfg) == KdaHeadParallelPlan("pcp", 8)
    assert kda_cp_head_size(cfg) == 8


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_native_tp_plan_does_not_inherit_pcp_only_restrictions(monkeypatch, tp_size, enabled):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, enabled)
    cfg = _config(tensor_parallel_size=tp_size, prefill_context_parallel_size=1,
                  enable_expert_parallel=False, data_parallel_size=2,
                  pipeline_parallel_size=2, decode_context_parallel_size=2)
    cfg.model_config.enforce_eager = False
    plan = resolve_kda_head_parallel(cfg)
    assert plan == KdaHeadParallelPlan("tp", tp_size)
    assert plan.local_num_heads(64) == 64 // tp_size
    assert kda_cp_head_size(cfg) is None


def test_switch_off_preserves_replicated_pcp_baseline(monkeypatch):
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "0")
    cfg = _config()
    assert resolve_kda_head_parallel(cfg) == KdaHeadParallelPlan("tp", 1)
    assert kda_cp_head_size(cfg) is None


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_runtime_head_group_must_match_cpu_plan(cp_size):
    plan = KdaHeadParallelPlan("pcp", cp_size)
    for rank in range(cp_size):
        assert plan.rank_in_group(SimpleNamespace(
            world_size=cp_size, rank_in_group=rank,
        )) == rank
    with pytest.raises(ValueError, match="runtime group size"):
        plan.rank_in_group(SimpleNamespace(world_size=cp_size * 2, rank_in_group=0))
    for rank in (-1, cp_size):
        with pytest.raises(ValueError, match="outside"):
            plan.rank_in_group(SimpleNamespace(world_size=cp_size, rank_in_group=rank))


@pytest.mark.parametrize("size", [0, -1, True, 2.0, "2"])
def test_head_plan_rejects_invalid_group_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        KdaHeadParallelPlan("pcp", size)


def test_head_plan_is_immutable_and_group_name_is_checked():
    from dataclasses import FrozenInstanceError

    plan = KdaHeadParallelPlan("pcp", 4)
    with pytest.raises(FrozenInstanceError):
        plan.size = 8
    with pytest.raises(ValueError, match="invalid KDA head group"):
        KdaHeadParallelPlan("world", 4)


@pytest.fixture
def cp_linear(monkeypatch):
    # Tensor allocation and loader tests only. No distributed init or HCU kernels.
    from vllm.model_executor.layers import linear
    from vllm.model_executor import parameter
    from vllm_hcu.model_executor.layers import glm53_kda_cp

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    return glm53_kda_cp


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("mode", ["separate", "fused", "tuple"])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_merged_checkpoint_loading_covers_all_heads(cp_linear, v2, mode, cp_size):
    sizes = [128, 128, 128, 64, 2, 2]
    full = [torch.arange(n * 3).reshape(n, 3).float() + i * 10000
            for i, n in enumerate(sizes)]
    assembled = [[] for _ in sizes]
    for rank in range(cp_size):
        with torch.device("cpu"):
            layer = cp_linear.KdaCpMergedLinear(
                3, sizes, head_size=cp_size, head_rank=rank,
                bias=False, params_dtype=torch.float32,
            )
        load = layer.weight_loader_v2 if v2 else layer.weight_loader
        if mode == "separate":
            for i, weight in enumerate(full):
                load(layer.weight, weight, i)
        elif mode == "fused":
            load(layer.weight, torch.cat(full), None)
        else:
            load(layer.weight, torch.cat(full[:3]), (0, 1, 2))
            load(layer.weight, torch.cat(full[3:]), (3, 4, 5))
        # Both native and HCU NN physical layouts are accepted.
        weight = layer.weight.detach()
        logical = weight if weight.shape[1] == 3 else weight.t()
        parts = logical.split(layer.output_sizes, dim=0)
        for i, part in enumerate(parts):
            expected = full[i] if i in (4, 5) else full[i].chunk(cp_size)[rank]
            torch.testing.assert_close(part, expected)
            assembled[i].append(part.clone())
    for i in range(4):
        torch.testing.assert_close(torch.cat(assembled[i]), full[i])


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_column_row_partial_sums_match_full_projection(cp_linear, v2, cp_size):
    torch.manual_seed(20)
    x = torch.randn(7, 3)
    up = torch.randn(128, 3)
    down = torch.randn(5, 128)
    partials = []
    for rank in range(cp_size):
        with torch.device("cpu"):
            col = cp_linear.KdaCpColumnLinear(
                3, 128, head_size=cp_size, head_rank=rank, bias=False,
                params_dtype=torch.float32,
            )
            row = cp_linear.KdaCpRowLinear(
                128, 5, head_size=cp_size, head_rank=rank, bias=False,
                params_dtype=torch.float32,
            )
        for layer, checkpoint in [(col, up), (row, down)]:
            load = layer.weight_loader_v2 if v2 else layer.weight_loader
            load(layer.weight, checkpoint)
            assert layer.tp_size == 1  # Global TP is never rewritten to PCP.
        width = 128 // cp_size
        cw = col.weight if col.weight.shape == (width, 3) else col.weight.t()
        rw = row.weight if row.weight.shape == (5, width) else row.weight.t()
        assert row.reduce_results is False
        local = torch.nn.functional.linear(x, cw).tanh()
        partials.append(torch.nn.functional.linear(local, rw))
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(x, up).tanh(), down
    )
    torch.testing.assert_close(sum(partials), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_parameter_loaders_cover_a_log_and_dt_bias(cp_linear, cp_size):
    a = torch.arange(64).float()
    dt = torch.arange(128).float()
    heads, channels = 64 // cp_size, 128 // cp_size
    for rank in range(cp_size):
        pa = torch.nn.Parameter(torch.empty(1, 1, heads, 1))
        pd = torch.nn.Parameter(torch.empty(channels))
        cp_linear.head_weight_loader(2, cp_size, rank, a_log=True)(pa, a)
        cp_linear.head_weight_loader(0, cp_size, rank)(pd, dt)
        torch.testing.assert_close(pa.flatten(), a[rank * heads:(rank + 1) * heads])
        torch.testing.assert_close(pd, dt[rank * channels:(rank + 1) * channels])
    with pytest.raises(ValueError, match="not divisible"):
        cp_linear.head_slice(torch.empty(7, 2), 0, 8, 0)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_cp_constructor_keeps_tp_and_quantization_scope(cp_linear, monkeypatch, cp_size):
    import vllm.distributed as distributed

    class Base(torch.nn.Module):
        def __init__(self, config, vllm_config, prefix):
            super().__init__()
            self.tp_size, self.tp_rank = 1, 0
            self.hidden_size, self.num_spec = 3, 3
            self.quant_config = vllm_config.quant_config

    fake_kda = SimpleNamespace(
        GatedDeltaNetAttention=Base,
        set_weight_attrs=lambda p, attrs: [setattr(p, k, v) for k, v in attrs.items()],
        FusedRMSNormGated=lambda *args, **kwargs: torch.nn.Identity(),
        is_conv_state_dim_first=lambda: True,
    )
    config = SimpleNamespace(
        linear_num_heads=64, linear_head_dim=2,
        linear_conv_kernel_dim=4, linear_lower_bound=-5.0,
    )
    quantization = object()
    cfg = SimpleNamespace(
        quant_config=quantization,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    heads, channels = 64 // cp_size, 128 // cp_size
    for rank in range(cp_size):
        monkeypatch.setattr(distributed, "get_pcp_group", lambda r=rank: SimpleNamespace(
            world_size=cp_size, rank_in_group=r,
        ))
        layer = Base.__new__(Base)
        with torch.device("cpu"):
            cp_linear.initialize_cp_kda(layer, config, cfg, f"layer.{rank}", fake_kda, cp_size)
        assert cfg.quant_config is quantization
        assert layer.quant_config is None
        assert layer.tp_size == 1 and layer.head_parallel_size == cp_size
        assert layer._hcu_kda_head_parallel_plan == KdaHeadParallelPlan("pcp", cp_size)
        assert layer.head_parallel_rank == rank and layer.local_num_heads == heads
        assert layer.local_projection_size == channels and layer.dt_bias.shape == (channels,)
        assert layer.A_log.shape == (1, 1, heads, 1)
        assert layer.q_conv1d.weight.numel() == channels * 4
        assert layer.in_proj_qkvbfg_a.output_sizes == [channels] * 3 + [heads, 2, 2]
        assert layer.o_proj.reduce_results is False


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_engine_state_shape_uses_cp_without_initialized_groups(monkeypatch, cp_size):
    from vllm_hcu.patch.platform.framework_opt import patch_glm53_kda_cp_state as patch
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    module = ModuleType(patch.TARGET_MODULE)

    class Model:
        @classmethod
        def get_mamba_state_shape_from_config(cls, vllm_config):
            return "original"

    module.Glm5NextForCausalLM = Model
    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "0")
    cfg = _config(prefill_context_parallel_size=cp_size)
    assert Model.get_mamba_state_shape_from_config(cfg) == "original"
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    expected = MambaStateShapeCalculator.kda_state_shape(
        cp_size, 64, 2, conv_kernel_size=4, num_spec=3,
    )
    assert Model.get_mamba_state_shape_from_config(cfg) == expected
    # Engine sizing must read the same text config as head selection.
    cfg.model_config.hf_text_config = SimpleNamespace(
        linear_num_heads=32, linear_head_dim=4, linear_conv_kernel_dim=3,
    )
    assert Model.get_mamba_state_shape_from_config(cfg) == (
        MambaStateShapeCalculator.kda_state_shape(
            cp_size, 32, 4, conv_kernel_size=3, num_spec=3,
        )
    )
    assert Model.get_mamba_state_shape_from_config(_config(
        tensor_parallel_size=8, prefill_context_parallel_size=1,
    )) == "original"


@pytest.mark.parametrize("override", [
    {"tensor_parallel_size": 2}, {"pipeline_parallel_size": 2},
    {"data_parallel_size": 2}, {"decode_context_parallel_size": 2},
    {"enable_expert_parallel": False}, {"enforce_eager": False},
])
def test_engine_and_worker_use_relaxed_admission(monkeypatch, override):
    from vllm_hcu.patch.platform.framework_opt import patch_glm53_kda_cp_state as engine
    from vllm_hcu.patch.worker.core_fix import patch_glm53_kda_cp as worker
    from vllm_hcu.model_executor.layers import glm53_kda_cp as cp

    def initialize(layer, config, vllm_config, prefix, kda, size):
        layer.test_head_size = size

    monkeypatch.setattr(cp, "initialize_cp_kda", initialize)

    class Model:
        @classmethod
        def get_mamba_state_shape_from_config(cls, vllm_config):
            raise AssertionError("CP must not use native state sizing")

    class Layer:
        def __init__(self, config, vllm_config, prefix=""):
            raise AssertionError("CP must not construct native weights")

        def get_state_shape(self):
            return "original"

    engine_module = ModuleType(engine.TARGET_MODULE)
    engine_module.Glm5NextForCausalLM = Model
    engine.apply_to_module(engine_module)
    worker_module = ModuleType(worker.TARGET_MODULE)
    worker_module.Glm5NextLinearAttention = Layer
    worker.apply_to_module(worker_module)
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    cfg = _config(**{k: v for k, v in override.items() if k != "enforce_eager"})
    cfg.model_config.enforce_eager = override.get("enforce_eager", True)
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    expected = MambaStateShapeCalculator.kda_state_shape(
        8, 64, 2, conv_kernel_size=4, num_spec=3,
    )
    assert Model.get_mamba_state_shape_from_config(cfg) == expected
    assert Layer(None, cfg).test_head_size == 8


def test_worker_shape_and_disabled_initializer(monkeypatch):
    from vllm_hcu.patch.worker.core_fix import patch_glm53_kda_cp as patch

    module = ModuleType(patch.TARGET_MODULE)

    class Layer:
        def __init__(self, config, vllm_config, prefix=""):
            self.original = prefix

        def get_state_shape(self):
            return "original"

    module.Glm5NextLinearAttention = Layer
    module.MambaStateShapeCalculator = SimpleNamespace(
        kda_state_shape=lambda *args, **kwargs: (args, kwargs)
    )
    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "0")
    layer = Layer(None, _config(), "layer.0")
    assert layer.original == "layer.0"
    assert layer.get_state_shape() == "original"
    monkeypatch.setenv(KDA_CP_HEAD_ENV, "1")
    native = Layer(None, _config(tensor_parallel_size=8,
                                prefill_context_parallel_size=1), "native.tp")
    assert native.original == "native.tp"
    assert native.get_state_shape() == "original"
    layer._hcu_kda_cp_head_sharded = True
    layer.head_parallel_size, layer.num_heads = 8, 64
    layer.head_dim, layer.conv_size, layer.num_spec = 2, 4, 3
    assert layer.get_state_shape() == ((8, 64, 2), dict(conv_kernel_size=4, num_spec=3))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_reduction_precedes_localize_and_rebinds_output(monkeypatch, cp_size):
    from vllm_hcu.v1 import glm53_pcp as pcp

    events = []
    hidden = torch.arange(10).reshape(5, 2).float()
    positions = torch.arange(5)
    batch = SimpleNamespace(
        gather=lambda x: hidden,
        global_batch=SimpleNamespace(positions=positions),
        localize=lambda x: events.append("localize") or x[[4, 0]],
    )
    monkeypatch.setattr(pcp, "_current_pcp_batch", lambda: batch)
    layer = SimpleNamespace(
        _hcu_kda_cp_head_sharded=True,
        _hcu_glm53_original_forward=lambda h, p: h + 1,
        _hcu_kda_cp_group=SimpleNamespace(
            all_reduce=lambda x: events.append("reduce") or x * cp_size
        ),
    )
    out = pcp._kda_forward(layer, hidden[:2], positions[:2])
    assert events == ["reduce", "localize"]
    torch.testing.assert_close(out, ((hidden + 1) * cp_size)[[4, 0]])
    # No target PCP batch (decode/verify/dummy) must not skip the head sum.
    monkeypatch.setattr(pcp, "_current_pcp_batch", lambda: None)
    torch.testing.assert_close(pcp._kda_forward(layer, hidden, positions), (hidden + 1) * cp_size)
