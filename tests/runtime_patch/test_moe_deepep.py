# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import linecache
import os
import subprocess
import sys
import textwrap
from enum import Enum, IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt.moe import (
    patch_all2all_utils,
    patch_base_router,
    patch_config,
    patch_deepep_ht,
    patch_deepep_ll,
    patch_fp8_oracle,
    patch_fused_moe,
    patch_fused_moe_modular_method,
    patch_fused_topk_bias_router,
    patch_layer,
    patch_moe_align_block_size,
    patch_moe_runner,
    patch_rocm_aiter_moe,
    patch_router_factory,
    patch_shared_experts,
    patch_triton_moe,
    patch_utils,
)
from vllm_hcu.patch.worker.op_opt.moe._common import PatchCompatibilityError


ADAPTERS = (
    patch_all2all_utils,
    patch_config,
    patch_rocm_aiter_moe,
    patch_triton_moe,
    patch_fused_moe,
    patch_fused_moe_modular_method,
    patch_layer,
    patch_moe_align_block_size,
    patch_fp8_oracle,
    patch_deepep_ht,
    patch_deepep_ll,
    patch_base_router,
    patch_fused_topk_bias_router,
    patch_router_factory,
    patch_moe_runner,
    patch_shared_experts,
    patch_utils,
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_all2all_dispatch_selection_contract():
    class DeepEPLLPrepareAndFinalize:
        pass

    prepare_finalize = DeepEPLLPrepareAndFinalize()

    def maybe_make_prepare_finalize(
        moe,
        quant_config,
        routing_tables=None,
        allow_new_interface=False,
        use_monolithic=False,
        eep_stage=False,
    ):
        del (
            moe,
            quant_config,
            routing_tables,
            allow_new_interface,
            use_monolithic,
            eep_stage,
        )
        return prepare_finalize

    def maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    ):
        del act_dtype, moe_parallel_config
        return hidden_size

    fp8_dtype = torch.float8_e4m3fn
    module = _module(
        patch_all2all_utils.TARGET_MODULE,
        torch=torch,
        current_platform=SimpleNamespace(fp8_dtype=lambda: fp8_dtype),
        DeepEPLLPrepareAndFinalize=DeepEPLLPrepareAndFinalize,
        maybe_make_prepare_finalize=maybe_make_prepare_finalize,
        maybe_roundup_layer_hidden_size=maybe_roundup_layer_hidden_size,
    )
    assert patch_all2all_utils.apply_to_module(module) is True
    assert patch_all2all_utils.apply_to_module(module) is False

    fp8_config = SimpleNamespace(quant_dtype=fp8_dtype)
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_deepep_auto_kernels=False)
    )
    result = module.maybe_make_prepare_finalize(moe, fp8_config)
    assert result is prepare_finalize
    assert result.use_fp8_dispatch is True
    assert result.use_int8_dispatch is False

    int8_config = SimpleNamespace(quant_dtype=torch.int8)
    result = module.maybe_make_prepare_finalize(moe, int8_config)
    assert result.use_fp8_dispatch is False
    assert result.use_int8_dispatch is True


def test_all2all_auto_builds_ht_and_ll_around_one_manager_handle():
    calls: dict[str, object] = {}

    class Manager:
        is_deepep_auto_manager = True
        dp_world_size = 2
        world_size = 2
        rank = 1

        def get_handle(self, kwargs):
            calls["handle_kwargs"] = kwargs
            return "shared-handle"

    class DeepEPHTPrepareAndFinalize:
        def __init__(self, handle, **kwargs):
            self.handle = handle
            self.kwargs = kwargs

        @staticmethod
        def maybe_roundup_layer_hidden_size(hidden_size, act_dtype):
            del act_dtype
            return hidden_size + 1

    class DeepEPLLPrepareAndFinalize:
        def __init__(self, handle, **kwargs):
            self.handle = handle
            self.kwargs = kwargs

        @staticmethod
        def maybe_roundup_layer_hidden_size(hidden_size):
            return hidden_size + 2

    def maybe_make_prepare_finalize(
        moe,
        quant_config,
        routing_tables=None,
        allow_new_interface=False,
        use_monolithic=False,
        eep_stage=False,
    ):
        del (
            moe,
            quant_config,
            routing_tables,
            allow_new_interface,
            use_monolithic,
            eep_stage,
        )
        return "official"

    def maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    ):
        del act_dtype, moe_parallel_config
        return hidden_size

    manager = Manager()
    module = _module(
        patch_all2all_utils.TARGET_MODULE,
        torch=torch,
        current_platform=SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn),
        get_ep_all2all_manager=lambda eep_stage=False: manager,
        get_current_vllm_config=lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            speculative_config=SimpleNamespace(num_speculative_tokens=2),
        ),
        DeepEPHTPrepareAndFinalize=DeepEPHTPrepareAndFinalize,
        DeepEPLLPrepareAndFinalize=DeepEPLLPrepareAndFinalize,
        maybe_make_prepare_finalize=maybe_make_prepare_finalize,
        maybe_roundup_layer_hidden_size=maybe_roundup_layer_hidden_size,
    )
    assert patch_all2all_utils.apply_to_module(module)
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_deepep_auto_kernels=True),
        dp_size=2,
        hidden_dim=7168,
        num_experts=8,
        num_local_experts=4,
    )
    routing_tables = ("global-to-physical", "physical-to-global", "local-ids")
    result = module.maybe_make_prepare_finalize(
        moe,
        SimpleNamespace(quant_dtype=torch.float8_e4m3fn),
        routing_tables,
    )
    assert calls["handle_kwargs"] == {
        "max_num_tokens_per_dp_rank": 12,
        "token_hidden_size": 7168,
        "num_ep_ranks": 2,
        "num_global_experts": 8,
        "num_local_experts": 4,
    }
    assert result.ht_prepare_finalize.handle == "shared-handle"
    assert result.ht_prepare_finalize.kwargs == {
        "num_dispatchers": 2,
        "dp_size": 2,
        "rank_expert_offset": 4,
    }
    assert result.ll_prepare_finalize.handle == "shared-handle"
    assert result.ll_prepare_finalize.kwargs == {
        "max_tokens_per_rank": 12,
        "num_dispatchers": 2,
        "use_fp8_dispatch": True,
        "global_to_physical": "global-to-physical",
        "physical_to_global": "physical-to-global",
        "local_expert_global_ids": "local-ids",
    }
    assert module.maybe_roundup_layer_hidden_size(
        10, torch.float16, moe.moe_parallel_config
    ) == 13


def test_deepep_auto_prepare_snapshots_mode_for_matching_finalize(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_hcu.model_executor.layers.fused_moe.prepare_finalize.deepep_auto as auto_module

    class Delegate:
        def __init__(self, name):
            self.name = name
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def post_init_setup(self, experts):
            self.calls.append(("post_init_setup", (experts,)))

        def prepare(self, *args):
            self.calls.append(("prepare", args))
            return self.name

        def finalize(self, *args):
            self.calls.append(("finalize", args))
            return None

    ht = Delegate("ht")
    ll = Delegate("ll")
    prepare_finalize = auto_module.DeepEPAutoPrepareAndFinalize(ht, ll)
    mode = {"low_latency": True}
    monkeypatch.setattr(
        auto_module,
        "_forward_uses_low_latency",
        lambda: mode["low_latency"],
    )

    class Experts:
        ht_experts = "ht-experts"
        ll_experts = "ll-experts"

        def set_deepep_auto_use_low_latency(self, value):
            self.low_latency = value

    experts = Experts()
    prepare_finalize.post_init_setup(experts)
    assert ht.calls == [("post_init_setup", ("ht-experts",))]
    assert ll.calls == [("post_init_setup", ("ll-experts",))]

    assert prepare_finalize.prepare(
        "a1", "weights", "ids", 8, None, False, "quant"
    ) == "ll"
    assert experts.low_latency is True
    mode["low_latency"] = False
    prepare_finalize.finalize(
        "output", "experts", "weights", "ids", False, "reduce"
    )
    assert [name for name, _ in ll.calls[-2:]] == ["prepare", "finalize"]

    assert prepare_finalize.prepare(
        "a1", "weights", "ids", 8, None, False, "quant"
    ) == "ht"
    assert experts.low_latency is False
    prepare_finalize.finalize(
        "output", "experts", "weights", "ids", False, "reduce"
    )
    assert [name for name, _ in ht.calls[-2:]] == ["prepare", "finalize"]


class _GroupShape:
    PER_TENSOR = object()
    PER_TOKEN = object()

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col


def _fake_config_module() -> ModuleType:
    def flags(quant_dtype, per_act_token_quant, per_out_ch_quant, block_shape):
        del quant_dtype, per_act_token_quant, per_out_ch_quant, block_shape
        return "official-a", "official-w"

    class FusedMoEQuantConfig:
        def __post_init__(self):
            return "official-post-init"

        @staticmethod
        def make(
            quant_dtype=None,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
            w1_scale=None,
            w2_scale=None,
            a1_scale=None,
            a2_scale=None,
            g1_alphas=None,
            g2_alphas=None,
            a1_gscale=None,
            a2_gscale=None,
            w1_bias=None,
            w2_bias=None,
            w1_zp=None,
            w2_zp=None,
            weight_dtype=None,
            is_scale_swizzled=True,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
        ):
            return SimpleNamespace(
                quant_dtype=quant_dtype,
                per_act_token_quant=per_act_token_quant,
                per_out_ch_quant=per_out_ch_quant,
                block_shape=block_shape,
            )

    def int8_config(
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
    ):
        del w1_scale, w2_scale, a1_scale, a2_scale, w1_bias, w2_bias
        return ("official", per_act_token_quant)

    class FusedMoEParallelConfig:
        @property
        def use_all2all_kernels(self):
            return self.dp_size > 1 and self.use_ep

        @property
        def use_batched_activation_format(self):
            return False

        @property
        def needs_round_robin_routing_tables(self):
            return False

        @staticmethod
        def make(
            tp_size_, pcp_size_, dp_size_, sp_size_, vllm_parallel_config
        ):
            result = FusedMoEParallelConfig()
            result.tp_size = tp_size_
            result.pcp_size = pcp_size_
            result.dp_size = dp_size_
            result.sp_size = sp_size_
            result.use_ep = vllm_parallel_config.enable_expert_parallel
            result.all2all_backend = vllm_parallel_config.all2all_backend
            return result

    class FusedMoEConfig:
        pass

    return _module(
        patch_config.TARGET_MODULE,
        torch=torch,
        current_platform=SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn),
        GroupShape=_GroupShape,
        _quant_flags_to_group_shape=flags,
        FusedMoEQuantConfig=FusedMoEQuantConfig,
        int8_w8a8_moe_quant_config=int8_config,
        FusedMoEParallelConfig=FusedMoEParallelConfig,
        FusedMoEConfig=FusedMoEConfig,
    )


def test_hcu_block_quant_group_shapes_and_sequence_parallel_contract():
    module = _fake_config_module()
    assert patch_config.apply_to_module(module) is True
    assert patch_config.apply_to_module(module) is False
    a_shape, w_shape = module._quant_flags_to_group_shape(
        torch.int8,
        True,
        False,
        [128, 128],
    )
    assert (a_shape.row, a_shape.col) == (128, 128)
    assert (w_shape.row, w_shape.col) == (128, 128)
    int8_config = module.int8_w8a8_moe_quant_config(
        torch.ones(1),
        torch.ones(1),
        None,
        None,
        per_act_token_quant=True,
        block_shape=[128, 128],
    )
    assert int8_config.quant_dtype == torch.int8
    assert int8_config.per_act_token_quant is False
    assert int8_config.per_out_ch_quant is False
    assert int8_config.block_shape == [128, 128]
    special = SimpleNamespace(
        quant_dtype=torch.int8,
        block_shape=[128, 128],
    )
    assert module.FusedMoEQuantConfig.__post_init__(special) is None
    assert module.int8_w8a8_moe_quant_config(
        None,
        None,
        None,
        None,
    ) == ("official", False)
    parallel = module.FusedMoEParallelConfig()
    parallel.dp_size = 1
    parallel.use_ep = True
    parallel.is_sequence_parallel = True
    assert parallel.use_all2all_kernels is True
    upstream = SimpleNamespace(
        all2all_backend="deepep_low_latency",
        enable_expert_parallel=True,
        _vllm_hcu_deepep_auto=True,
    )
    auto = module.FusedMoEParallelConfig.make(1, 1, 2, 1, upstream)
    assert auto.all2all_backend == "deepep_auto"
    assert auto.use_deepep_auto_kernels is True
    assert auto.use_batched_activation_format is True
    assert auto.needs_round_robin_routing_tables is True
    moe = module.FusedMoEConfig()
    moe.moe_parallel_config = auto
    assert moe.use_deepep_auto_kernels is True


def test_config_signature_drift_fails_before_mutation():
    module = _fake_config_module()
    module._quant_flags_to_group_shape = lambda quant_dtype: quant_dtype
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_config.apply_to_module(module)
    assert not hasattr(module, "_vllm_hcu_moe_config_applied")


def test_aiter_and_triton_expert_capability_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoEActivation(Enum):
        SILU = "silu"
        GELU = "gelu"
        GELU_TANH = "gelu_tanh"
        SWIGLUOAI = "swigluoai"
        SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"

    class ActivationMethod(IntEnum):
        SILU = 0
        GELU = 1

    namespace = {
        "ActivationMethod": ActivationMethod,
        "MoEActivation": MoEActivation,
    }
    exec(
        textwrap.dedent(
            """
            def rocm_aiter_fused_experts(
                hidden_states, w1, w2, topk_weights, topk_ids, moe_config,
                activation, apply_router_weight_on_input, expert_map,
                quant_config, a1q_scale, num_local_tokens, output_dtype,
                moe_sorting_dispatch_policy=0,
            ):
                del hidden_states, w1, w2, topk_weights, topk_ids, moe_config
                del apply_router_weight_on_input, expert_map, quant_config
                del a1q_scale, num_local_tokens, output_dtype
                del moe_sorting_dispatch_policy
                if activation == MoEActivation.SILU:
                    return ActivationMethod.SILU
                if activation == MoEActivation.GELU:
                    return ActivationMethod.GELU
                if activation == MoEActivation.SWIGLUOAI:
                    return 2
                raise ValueError(activation)
            """
        ),
        namespace,
    )

    class AiterExperts:
        @staticmethod
        def _supports_activation(activation):
            return activation in (MoEActivation.SILU, MoEActivation.GELU)

        @staticmethod
        def _supports_current_device():
            return False

        @staticmethod
        def is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        ):
            del cls, moe_config, weight_key, activation_key, activation_format
            return AiterExperts._supports_current_device(), None

    aiter_module = _module(
        patch_rocm_aiter_moe.TARGET_MODULE,
        IntEnum=IntEnum,
        ActivationMethod=ActivationMethod,
        MoEActivation=MoEActivation,
        kMxfp4Static=object(),
        rocm_aiter_fused_experts=namespace["rocm_aiter_fused_experts"],
        AiterExperts=AiterExperts,
    )
    assert patch_rocm_aiter_moe.apply_to_module(aiter_module) is True
    assert aiter_module.ActivationMethod.GELU_TANH.value == 3
    activation = aiter_module.rocm_aiter_fused_experts(
        None,
        None,
        None,
        None,
        None,
        None,
        MoEActivation.GELU_TANH,
        False,
        None,
        None,
        None,
        None,
        None,
    )
    assert activation == aiter_module.ActivationMethod.GELU_TANH
    assert AiterExperts._supports_activation(MoEActivation.GELU_TANH) is True
    assert (
        AiterExperts._supports_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE
        )
        is False
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm._aiter_ops",
        _module(
            "vllm._aiter_ops",
            is_aiter_found_and_supported=lambda: True,
        ),
    )
    assert AiterExperts.is_supported_config(
        AiterExperts,
        SimpleNamespace(moe_backend="auto"),
        None,
        None,
        None,
    )[0] is False
    assert AiterExperts.is_supported_config(
        AiterExperts,
        SimpleNamespace(moe_backend="aiter"),
        None,
        None,
        None,
    )[0] is True
    assert AiterExperts.is_supported_config(
        AiterExperts,
        SimpleNamespace(moe_backend="aiter"),
        aiter_module.kMxfp4Static,
        None,
        None,
    )[0] is False

    weight_key = object()
    activation_key = object()

    class TritonExperts:
        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key):
            del weight_key, activation_key
            return False

    triton_module = _module(
        patch_triton_moe.TARGET_MODULE,
        current_platform=SimpleNamespace(is_rocm=lambda: True),
        kInt8StaticChannelSym=weight_key,
        kInt8DynamicTokenSym=activation_key,
        TritonExperts=TritonExperts,
    )
    assert patch_triton_moe.apply_to_module(triton_module) is True
    assert TritonExperts._supports_quant_scheme(weight_key, activation_key) is True
    assert TritonExperts._supports_quant_scheme(object(), object()) is False


def test_aiter_expert_wrapper_removes_flydsl_import(
    monkeypatch: pytest.MonkeyPatch,
):
    source = textwrap.dedent(
        """
        def example(use_interleave):
            from aiter.ops.flydsl.moe_common import GateMode
            if use_interleave:
                return GateMode.INTERLEAVE.value
            return GateMode.SEPARATED.value
        """
    )
    filename = "<workspace-aiter-gate-mode-test>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {}
    exec(compile(source, filename, "exec"), namespace)

    monkeypatch.delitem(sys.modules, "aiter.ops.flydsl", raising=False)
    monkeypatch.delitem(sys.modules, "aiter.ops.flydsl.moe_common", raising=False)
    rebuilt = patch_rocm_aiter_moe._build_workspace_aiter_fused_experts(
        namespace["example"]
    )

    assert rebuilt(True) == "interleave"
    assert rebuilt(False) == "separated"
    assert "aiter.ops.flydsl.moe_common" not in rebuilt.__code__.co_names
    assert "aiter.ops.flydsl" not in sys.modules
    assert "aiter.ops.flydsl.moe_common" not in sys.modules


def test_fused_moe_aiter_feature_gate_and_obsolete_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    parameter_names = (
        "hidden_states", "w1", "w2", "topk_weights", "topk_ids",
        "activation", "apply_router_weight_on_input", "use_fp8_w8a8",
        "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16",
        "ocp_mx_scheme", "per_channel_quant", "global_num_experts",
        "expert_map", "w1_scale", "w2_scale", "w1_zp", "w2_zp",
        "a1_scale", "a2_scale", "block_shape", "w1_bias", "w2_bias",
    )
    namespace: dict[str, object] = {}
    exec(
        "def fused_experts_impl("
        + ", ".join(parameter_names)
        + "):\n    return 'official'\n",
        namespace,
    )
    module = _module(
        patch_fused_moe.TARGET_MODULE,
        torch=torch,
        fused_experts_impl=namespace["fused_experts_impl"],
    )
    assert patch_fused_moe.apply_to_module(module) is True
    from vllm_hcu.platforms import envs as henvs

    hidden = torch.zeros((1, 2), dtype=torch.bfloat16)
    w1 = torch.zeros((1, 4, 2), dtype=torch.int8)
    w2 = torch.zeros((1, 2, 2), dtype=torch.int8)
    weights = torch.ones((1, 1))
    ids = torch.zeros((1, 1), dtype=torch.int32)
    arguments = (
        hidden, w1, w2, weights, ids, "silu", False, False, False,
        False, True, None, False, 1, None, None, None, None, None, None,
        None, [128, 128], None, None,
    )

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    assert module.fused_experts_impl(*arguments) == "official"

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W4A16_MOE", True)
    monkeypatch.setitem(sys.modules, "aiter", ModuleType("aiter"))
    monkeypatch.delitem(sys.modules, "aiter.moe", raising=False)
    with pytest.raises(RuntimeError, match="aiter.moe API is unavailable"):
        module.fused_experts_impl(*arguments)


def test_fused_moe_w4a16_uses_workspace_aiter_public_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    parameter_names = (
        "hidden_states", "w1", "w2", "topk_weights", "topk_ids",
        "activation", "apply_router_weight_on_input", "use_fp8_w8a8",
        "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16",
        "ocp_mx_scheme", "per_channel_quant", "global_num_experts",
        "expert_map", "w1_scale", "w2_scale", "w1_zp", "w2_zp",
        "a1_scale", "a2_scale", "block_shape", "w1_bias", "w2_bias",
    )
    namespace: dict[str, object] = {}
    exec(
        "def fused_experts_impl("
        + ", ".join(parameter_names)
        + "):\n    return 'official'\n",
        namespace,
    )
    module = _module(
        patch_fused_moe.TARGET_MODULE,
        torch=torch,
        fused_experts_impl=namespace["fused_experts_impl"],
    )
    assert patch_fused_moe.apply_to_module(module) is True

    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W4A16_MOE", True)
    calls: dict[str, object] = {}
    moe_config = SimpleNamespace(
        quant_type="w4a16",
        solution_type="moe_c",
        need_shuffle=False,
        need_shuffle_scale=True,
    )

    class MoeQuantType:
        W4A16 = "w4a16"

    def get_config(**kwargs):
        calls["config"] = kwargs
        return True, moe_config

    def shuffle_weight(*unused):
        pytest.fail("need_shuffle=False must preserve the loaded weights")

    def shuffle_scale(scale1, scale2, config):
        assert config is moe_config
        calls["scale"] = (scale1, scale2)
        return scale1 + 1, scale2 + 2

    def aiter_moe(**kwargs):
        calls["moe"] = kwargs
        return "workspace-aiter"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
            aiter_moe_shfl_weight=shuffle_weight,
            aiter_moe_shfl_scale=shuffle_scale,
        ),
    )

    hidden = torch.zeros((1, 2), dtype=torch.bfloat16)
    w1 = torch.zeros((1, 4, 2), dtype=torch.int8)
    w2 = torch.zeros((1, 2, 2), dtype=torch.int8)
    topk_weights = torch.ones((1, 1))
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    w1_scale = torch.ones((1, 4, 1))
    w2_scale = torch.ones((1, 2, 1))
    result = module.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, "silu", False, False,
        False, False, True, None, False, 1, None, w1_scale, w2_scale,
        None, None, None, None, [128, 128], None, None,
    )

    assert result == "workspace-aiter"
    assert calls["config"]["N2"] == w2.shape[1]
    assert calls["moe"]["moe_config"] is moe_config
    assert calls["moe"]["w1"] is w1
    torch.testing.assert_close(calls["moe"]["w1_scale"], w1_scale + 1)
    torch.testing.assert_close(calls["moe"]["w2_scale"], w2_scale + 2)
    assert calls["moe"]["use_weight_shuffle"] is False


def test_modular_method_dimensions_and_prequant_contract():
    class FusedMoEKernel:
        last = None

        def __init__(
            self,
            prepare_finalize,
            experts,
            shared_experts=None,
            inplace=False,
            N=-1,
            K=-1,
        ):
            self.arguments = (
                prepare_finalize,
                experts,
                shared_experts,
                inplace,
                N,
                K,
            )
            self.applied = None
            FusedMoEKernel.last = self

        def apply(self, **kwargs):
            self.applied = kwargs
            return kwargs["hidden_states"]

    class FusedMoEModularMethod:
        def __init__(self, old_quant_method, moe_kernel):
            del old_quant_method
            self.moe_kernel = moe_kernel
            self.disable_expert_map = False

        @staticmethod
        def make(
            routed_experts,
            old_quant_method,
            prepare_finalize,
        ):
            del routed_experts, old_quant_method, prepare_finalize
            return None

        def apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        ):
            del layer, topk_weights, topk_ids, shared_experts, shared_experts_input
            return x

    module = _module(
        patch_fused_moe_modular_method.TARGET_MODULE,
        FusedMoEKernel=FusedMoEKernel,
        FusedMoEModularMethod=FusedMoEModularMethod,
    )
    assert patch_fused_moe_modular_method.apply_to_module(module) is True
    old_method = SimpleNamespace(
        N=32,
        K=64,
        select_gemm_impl=lambda prepare, layer: (prepare, layer),
    )
    method = FusedMoEModularMethod.make("layer", old_method, "prepare")
    assert FusedMoEKernel.last.arguments == (
        "prepare", ("prepare", "layer"), None, False, 32, 64,
    )
    layer = SimpleNamespace(
        w13_weight="w1",
        w2_weight="w2",
        activation="silu",
        global_num_experts=8,
        apply_router_weight_on_input=False,
        expert_map="map",
    )
    x = torch.ones((2, 3))
    i_q = torch.ones((2, 3), dtype=torch.int8)
    i_s = torch.ones((2, 1))
    assert method.apply(
        layer, x, "weights", "ids", "shared", "shared-input", False, i_q, i_s
    ) is x
    assert FusedMoEKernel.last.applied["quanted_hidden_states"] is i_q
    assert FusedMoEKernel.last.applied["scale"] is i_s
    with pytest.raises(ValueError, match="i_q and i_s together"):
        method.apply(
            layer, x, "weights", "ids", "shared", "shared-input", False, i_q, None
        )
    with pytest.raises(RuntimeError, match="use_nn_moe"):
        method.apply(layer, x, "weights", "ids", "shared", "shared-input", True)


def test_eplb_torch_map_and_record_numeric_contract(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def official(
        topk_ids,
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
        record_enabled,
        num_unpadded_tokens=None,
    ):
        del num_unpadded_tokens
        calls.append(True)
        return topk_ids + 100

    module = _module(
        patch_base_router.TARGET_MODULE,
        torch=torch,
        eplb_map_to_physical_and_record=official,
    )
    patch_base_router.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD", False)
    ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    loads = torch.zeros(3, dtype=torch.int64)
    mapping = torch.tensor([[0, 1], [2, 2]], dtype=torch.int64)
    replicas = torch.tensor([2, 1], dtype=torch.int64)
    enabled = torch.tensor(True)
    assert torch.equal(
        module.eplb_map_to_physical_and_record(
            ids,
            loads,
            mapping,
            replicas,
            enabled,
        ),
        ids + 100,
    )
    assert calls == [True]

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD", True)
    loads.zero_()
    result = module.eplb_map_to_physical_and_record(
        ids,
        loads,
        mapping,
        replicas,
        enabled,
    )
    assert torch.equal(result, torch.tensor([[0, 2], [1, 2]], dtype=torch.int32))
    assert torch.equal(loads, torch.tensor([1, 1, 2]))

    loads.zero_()
    result = module.eplb_map_to_physical_and_record(
        ids,
        loads,
        mapping,
        replicas,
        enabled,
        num_unpadded_tokens=torch.tensor(1),
    )
    assert torch.equal(result, torch.tensor([[0, 2], [1, 2]], dtype=torch.int32))
    assert torch.equal(loads, torch.tensor([1, 0, 1]))


def test_hash_router_normalizes_index_dtypes():
    captured = {}

    def original(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
    ):
        del topk_weights, token_expert_indices, gating_output, renormalize
        del e_score_correction_bias, routed_scaling_factor
        captured["input"] = input_tokens.dtype
        captured["hash"] = hash_indices_table.dtype
        return topk_indices, topk_indices

    module = _module(
        patch_fused_topk_bias_router.TARGET_MODULE,
        vllm_topk_softplus_sqrt=original,
    )
    patch_fused_topk_bias_router.apply_to_module(module)
    indices = torch.empty((1, 1), dtype=torch.int32)
    module.vllm_topk_softplus_sqrt(
        torch.empty(1, 1),
        indices,
        indices,
        torch.empty(1, 1),
        input_tokens=torch.tensor([1], dtype=torch.int64),
        hash_indices_table=torch.tensor([[1]], dtype=torch.int64),
    )
    assert captured == {"input": torch.int32, "hash": torch.int32}


def test_moe_layer_forward_and_repacked_weight_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    factory_names = (
        "num_experts", "top_k", "hidden_size", "intermediate_size",
        "intermediate_pad", "params_dtype", "renormalize", "use_grouped_topk",
        "num_expert_group", "topk_group", "quant_config", "tp_size", "dp_size",
        "pcp_size", "prefix", "custom_routing_function", "router", "scoring_func",
        "routed_scaling_factor", "swiglu_limit", "swiglu_alpha", "swiglu_beta",
        "e_score_correction_bias", "apply_router_weight_on_input", "activation",
        "enable_eplb", "num_redundant_experts", "has_bias",
        "is_sequence_parallel", "reduce_results", "ckpt_names", "n_shared_experts",
        "router_logits_dtype", "gate", "shared_experts", "shared_expert_gate",
        "routed_input_transform", "routed_output_transform",
        "apply_routed_scale_to_output", "zero_expert_type", "hash_indices_table",
        "runner_cls", "runner_args", "routed_experts_cls", "routed_experts_args",
    )

    class UnquantizedFusedMoEMethod:
        def __init__(self):
            self.moe_quant_config = "official-config"

    class RoutedExperts:
        def __init__(self):
            self.moe_config = "moe-config"
            self.quant_method = UnquantizedFusedMoEMethod()
            self.local_num_experts = 2
            self._dsv4_channel_fp8_deepgemm_repacked = False

        def _replace_quant_method(self, method):
            self.quant_method = method

        def get_expert_weights(self):
            return "official-weights"

    class Runner:
        def __init__(self):
            self.routed_experts = RoutedExperts()
            self.replaced = None

        def _replace_quant_method(self, method):
            self.replaced = method

    module = _module(
        patch_layer.TARGET_MODULE,
        UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod,
        RoutedExperts=RoutedExperts,
        Runner=Runner,
    )
    source = (
        "def FusedMoE("
        + ", ".join(f"{name}=None" for name in factory_names)
        + "):\n    return Runner()\n"
    )
    exec(source, module.__dict__)

    class HcuUnquantizedFusedMoEMethod:
        def __init__(self, moe_config):
            self.moe_config = moe_config
            self.moe_quant_config = None

    hcu_module_name = (
        "vllm_hcu.model_executor.layers.fused_moe."
        "unquantized_fused_moe_method"
    )
    hcu_module = _module(
        hcu_module_name,
        HcuUnquantizedFusedMoEMethod=HcuUnquantizedFusedMoEMethod,
    )
    monkeypatch.setitem(sys.modules, hcu_module_name, hcu_module)

    assert patch_layer.apply_to_module(module) is True
    runner = module.FusedMoE()
    experts = runner.routed_experts
    assert isinstance(experts.quant_method, HcuUnquantizedFusedMoEMethod)
    assert experts.quant_method.moe_quant_config == "official-config"
    assert runner.replaced is experts.quant_method

    experts._dsv4_channel_fp8_deepgemm_repacked = True
    experts.w13_weight = torch.arange(24).reshape(2, 3, 4)
    experts.w2_weight = torch.arange(16).reshape(2, 2, 4)
    experts.w13_weight_scale = torch.arange(6).reshape(2, 3)
    experts.w2_weight_scale = torch.arange(4).reshape(2, 2)
    weights = experts.get_expert_weights()
    assert [tuple(weight.shape) for weight in weights] == [
        (2, 12),
        (2, 8),
        (2, 3),
        (2, 2),
    ]


def test_moe_align_feature_off_and_lightop_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    def official(
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
    ):
        del (
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )
        return "official"

    module = _module(
        patch_moe_align_block_size.TARGET_MODULE,
        torch=torch,
        triton=SimpleNamespace(
            cdiv=lambda value, block: (value + block - 1) // block
        ),
        round_up=lambda value, block: (value + block - 1) // block * block,
        moe_align_block_size=official,
    )
    assert patch_moe_align_block_size.apply_to_module(module) is True
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    ids = torch.tensor([[0], [1]], dtype=torch.int32)
    assert module.moe_align_block_size(ids, 2, 2) == "official"

    calls = []

    def lightop_align(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map=None,
        expert_mask=None,
        num_local_tokens=None,
        is_ep=False,
        is_fuse_fill=True,
    ):
        calls.append(
            (
                topk_ids,
                num_experts,
                block_size,
                expert_map,
                expert_mask,
                num_local_tokens,
                is_ep,
                is_fuse_fill,
            )
        )
        assert torch.all(sorted_ids == topk_ids.numel())
        # Model one token per expert in a four-entry padded valid range.
        sorted_ids[0] = 0
        sorted_ids[2] = 1
        expert_ids.copy_(torch.arange(expert_ids.numel(), dtype=torch.int32))
        num_tokens_post_pad.fill_(sorted_ids.numel())

    lightop_package = ModuleType("lightop")
    lightop_package.op = SimpleNamespace(moe_align_block_size=lightop_align)
    monkeypatch.setitem(sys.modules, "lightop", lightop_package)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_LIGHTOP_MOE_ALIGN", True)
    sorted_ids, expert_ids, count = module.moe_align_block_size(ids, 2, 2)
    assert calls[0][1:] == (2, 2, None, None, None, False, False)
    assert count.item() == 4
    assert torch.equal(
        sorted_ids[: count.item()],
        torch.tensor([0, 2, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(expert_ids, torch.tensor([0, 1], dtype=torch.int32))

    expert_map = torch.tensor([1, 0], dtype=torch.int32)
    _, expert_ids, _ = module.moe_align_block_size(
        ids,
        2,
        2,
        expert_map,
        ignore_invalid_experts=True,
    )
    assert calls[-1][3] is expert_map
    assert torch.equal(expert_ids, torch.tensor([0, 1], dtype=torch.int32))

    _, expert_ids, _ = module.moe_align_block_size(
        ids,
        2,
        2,
        expert_map,
        ignore_invalid_experts=False,
    )
    assert calls[-1][3] is None
    assert torch.equal(expert_ids, torch.tensor([1, 0], dtype=torch.int32))


def test_moe_align_ep_remap_rejects_uninitialized_buffer_ids(
    monkeypatch: pytest.MonkeyPatch,
):
    def official(
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
    ):
        del (
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )
        raise AssertionError("EP remapping must use the guarded HCU path")

    def native_align(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
    ):
        del topk_ids, num_experts, block_size, expert_map
        sorted_ids.zero_()
        expert_ids.copy_(torch.tensor([0, 2], dtype=torch.int32))
        num_tokens_post_pad.fill_(2)

    module = _module(
        patch_moe_align_block_size.TARGET_MODULE,
        torch=torch,
        ops=SimpleNamespace(moe_align_block_size=native_align),
        triton=SimpleNamespace(cdiv=lambda value, block: (value + block - 1) // block),
        round_up=lambda value, block: (value + block - 1) // block * block,
        moe_align_block_size=official,
    )
    assert patch_moe_align_block_size.apply_to_module(module) is True
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    ids = torch.tensor([[0], [1]], dtype=torch.int32)
    expert_map = torch.tensor([1, 0], dtype=torch.int32)

    _, expert_ids, count = module.moe_align_block_size(
        ids,
        2,
        2,
        expert_map,
    )

    assert torch.equal(expert_ids, torch.tensor([1, -1], dtype=torch.int32))
    assert count.item() == 2


def test_fp8_oracle_sidecar_selection_and_format_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    auto_kernel_calls: list[tuple[object, object, object]] = []

    class Fp8MoeBackend(Enum):
        DEEPGEMM = "DEEPGEMM"
        BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
        TRITON = "TRITON"

    def backend_to_kernel_cls(backend):
        return [backend]

    def map_fp8_backend(runner_backend):
        return "official-map", runner_backend

    def select_fp8_moe_backend(
        config,
        weight_key,
        activation_key,
        allow_vllm_cutlass=False,
    ):
        del config, weight_key, activation_key, allow_vllm_cutlass
        return "official-select"

    def convert_to_fp8_moe_kernel_format(
        fp8_backend,
        layer,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_input_scale,
        w2_input_scale,
    ):
        del fp8_backend, layer, w13_input_scale, w2_input_scale
        return "converted", w2, w13_scale, w2_scale

    def make_fp8_moe_kernel(
        moe_quant_config,
        moe_config,
        experts_cls,
        fp8_backend,
        routing_tables=None,
        layer=None,
    ):
        del (
            moe_quant_config,
            moe_config,
            experts_cls,
            fp8_backend,
            routing_tables,
            layer,
        )
        return "official-kernel"

    module = _module(
        patch_fp8_oracle.TARGET_MODULE,
        Enum=Enum,
        Fp8MoeBackend=Fp8MoeBackend,
        backend_to_kernel_cls=backend_to_kernel_cls,
        map_fp8_backend=map_fp8_backend,
        select_fp8_moe_backend=select_fp8_moe_backend,
        convert_to_fp8_moe_kernel_format=convert_to_fp8_moe_kernel_format,
        make_fp8_moe_kernel=make_fp8_moe_kernel,
        mk=SimpleNamespace(
            FusedMoEActivationFormat=SimpleNamespace(
                Standard="standard",
                BatchedExperts="batched",
            )
        ),
    )
    assert patch_fp8_oracle.apply_to_module(module) is True
    assert module.Fp8MoeBackend.DPSK_DEEPGEMM.value == "DPSK_DEEPGEMM"

    class SupportedExperts:
        @staticmethod
        def is_supported_config(
            cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        ):
            del cls, config, weight_key, activation_key, activation_format
            return True, None

    class UnsupportedExperts:
        @staticmethod
        def is_supported_config(
            cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        ):
            del cls, config, weight_key, activation_key, activation_format
            return False, "unsupported"

    def make_deepep_auto_deepgemm_fp8_moe_kernel(
        *, moe_quant_config, moe_config, routing_tables
    ):
        auto_kernel_calls.append(
            (moe_quant_config, moe_config, routing_tables)
        )
        return "deepep-auto-kernel"

    experts_name = (
        "vllm_hcu.model_executor.layers.fused_moe.experts."
        "dpsk_v4_deep_gemm_moe"
    )
    experts_module = _module(
        experts_name,
        DeepEPDeepGemmContiguousExperts=UnsupportedExperts,
        DeepEPDeepGemmMaskedExperts=SupportedExperts,
        make_deepep_auto_deepgemm_fp8_moe_kernel=(
            make_deepep_auto_deepgemm_fp8_moe_kernel
        ),
    )
    monkeypatch.setitem(sys.modules, experts_name, experts_module)
    monkeypatch.setattr(
        patch_fp8_oracle,
        "_sidecar_config",
        lambda config: SimpleNamespace(
            deepep_auto=False, moe_backend="dpsk_deep_gemm"
        ),
    )
    config = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )
    backend, experts = module.select_fp8_moe_backend(config, "w", "a")
    assert backend is module.Fp8MoeBackend.DPSK_DEEPGEMM
    assert experts is SupportedExperts
    assert module.map_fp8_backend("dpsk_deep_gemm") is backend

    tensors = tuple(object() for _ in range(4))
    assert module.convert_to_fp8_moe_kernel_format(
        backend,
        SimpleNamespace(weight_block_size=None),
        *tensors,
        None,
        None,
    ) == tensors
    assert module.convert_to_fp8_moe_kernel_format(
        module.Fp8MoeBackend.DEEPGEMM,
        SimpleNamespace(weight_block_size=None),
        *tensors,
        None,
        None,
    ) == tensors

    assert module.make_fp8_moe_kernel(
        "quant",
        config,
        SupportedExperts,
        backend,
        "routing",
        "layer",
    ) == "official-kernel"

    monkeypatch.setattr(
        patch_fp8_oracle,
        "_sidecar_config",
        lambda config: SimpleNamespace(deepep_auto=True, moe_backend="auto"),
    )
    auto_backend, auto_experts = module.select_fp8_moe_backend(
        config, "w", "a"
    )
    assert auto_backend is backend
    assert auto_experts is UnsupportedExperts
    auto_config = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=False,
            use_deepep_auto_kernels=True,
        ),
    )
    assert module.make_fp8_moe_kernel(
        "quant",
        auto_config,
        auto_experts,
        auto_backend,
        "routing",
        "layer",
    ) == "deepep-auto-kernel"
    assert auto_kernel_calls == [("quant", auto_config, "routing")]
    with pytest.raises(ValueError, match="only the HCU DPSK_DEEPGEMM"):
        module.make_fp8_moe_kernel(
            "quant",
            auto_config,
            auto_experts,
            module.Fp8MoeBackend.TRITON,
        )

    monkeypatch.setattr(
        patch_fp8_oracle,
        "_sidecar_config",
        lambda config: SimpleNamespace(deepep_auto=False, moe_backend="auto"),
    )
    assert module.select_fp8_moe_backend(config, "w", "a") == "official-select"


def test_deepep_ht_quant_and_alignment_contract():
    class DeepEPHTPrepareAndFinalize:
        def _do_dispatch(
            self,
            tokens,
            token_scales,
            rank_topk_ids,
            rank_topk_weights,
            num_experts,
            a1_scale,
            quant_config,
            defer_input_quant,
        ):
            del (
                tokens, token_scales, rank_topk_ids, rank_topk_weights,
                num_experts, a1_scale, quant_config, defer_input_quant,
            )

        def _receiver(
            self,
            event,
            has_scales,
            token_data,
            expert_topk_ids,
            num_experts,
            expert_num_tokens_per_expert_list,
            expert_topk_weights,
            a1_scale,
            quant_config,
            defer_input_quant,
        ):
            del (
                event, has_scales, token_data, expert_topk_ids, num_experts,
                expert_num_tokens_per_expert_list, expert_topk_weights,
                a1_scale, quant_config, defer_input_quant,
            )

        def prepare_async(
            self,
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        ):
            del (
                a1, topk_weights, topk_ids, num_experts, expert_map,
                apply_router_weight_on_input, quant_config, defer_input_quant,
            )

    class ExpertTokensMetadata:
        @staticmethod
        def make_from_list(values, device=None):
            return values, device

    dispatched = {}

    class Buffer:
        capture = False

        def get_dispatch_layout(self, **kwargs):
            del kwargs
            return None, None, None, None, SimpleNamespace(event=None)

        def dispatch(self, **kwargs):
            dispatched.update(kwargs)
            return (
                kwargs["x"],
                torch.zeros((1, 1), dtype=torch.int32),
                torch.ones((1, 1)),
                [1],
                "handle",
                SimpleNamespace(event=None),
            )

    module = _module(
        patch_deepep_ht.TARGET_MODULE,
        torch=torch,
        DeepEPHTPrepareAndFinalize=DeepEPHTPrepareAndFinalize,
        dbo_get_previous_event=lambda capture: None,
        dbo_yield_and_switch_from_compute_to_comm=lambda: None,
        dbo_switch_to_compute_sync=lambda: None,
        dbo_enabled=lambda: False,
        dbo_current_ubatch_id=lambda: 0,
        mk=SimpleNamespace(ExpertTokensMetadata=ExpertTokensMetadata),
        moe_kernel_quantize_input=lambda value, scale, **kwargs: (
            value.to(torch.int8),
            torch.ones((value.shape[0], 1)),
        ),
    )
    assert patch_deepep_ht.apply_to_module(module) is True
    instance = object.__new__(DeepEPHTPrepareAndFinalize)
    instance.buffer = Buffer()
    instance.async_prepare = False
    instance.handles = [None, None]
    instance.rank_expert_offset = 0
    instance._get_dispatch_config = lambda: "config"
    quant_config = SimpleNamespace(
        is_block_quantized=False,
        is_per_act_token=True,
        per_act_token_quant=True,
        use_int8_w8a8=True,
        use_fp8_w8a8=False,
        quant_dtype=torch.int8,
        block_shape=None,
        is_scale_swizzled=True,
        a1_scale=None,
        a1_gscale=None,
    )
    captured = {}
    real_dispatch = instance._do_dispatch

    def capture_dispatch(**kwargs):
        captured.update(kwargs)
        return "prepared"

    instance._do_dispatch = capture_dispatch
    hidden = torch.ones((2, 4))
    topk_weights = torch.ones((2, 1))
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)
    assert instance.prepare_async(
        hidden,
        topk_weights,
        topk_ids,
        1,
        None,
        False,
        quant_config,
    ) == "prepared"
    assert captured["tokens"].dtype == torch.int8
    assert captured["token_scales"].shape == (2, 1)

    instance._do_dispatch = real_dispatch
    receiver = instance._do_dispatch(
        captured["tokens"],
        captured["token_scales"],
        topk_ids,
        topk_weights,
        1,
        None,
        quant_config,
        False,
    )
    assert dispatched["expert_alignment"] == 256
    expert_x, expert_scale, metadata, _, _ = receiver()
    assert expert_x is captured["tokens"]
    assert expert_scale is captured["token_scales"]
    assert metadata[0] == [1]


def test_router_factory_feature_gated_hcu_subclass_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    class GroupedTopKRouter:
        def _compute_routing(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            del hidden_states, router_logits, indices_type, input_ids
            return "official"

    module = _module(
        patch_router_factory.TARGET_MODULE,
        GroupedTopKRouter=GroupedTopKRouter,
    )
    factory_names = (
        "top_k", "global_num_experts", "renormalize",
        "use_grouped_topk", "num_expert_group", "topk_group", "scoring_func",
        "num_fused_shared_experts", "shared_expert_weight",
        "routed_scaling_factor", "e_score_correction_bias",
        "custom_routing_function",
        "eplb_state", "zero_expert_type", "num_logical_experts",
        "hash_indices_table",
    )
    exec(
        "def create_fused_moe_router("
        + ", ".join(factory_names)
        + "):\n    return GroupedTopKRouter()\n",
        module.__dict__,
    )
    assert patch_router_factory.apply_to_module(module) is True
    router = module.create_fused_moe_router(*([None] * len(factory_names)))
    assert type(router).__name__ == "HcuGroupedTopKRouter"
    router.num_expert_group = 2
    router.topk_group = 1
    router.top_k = 1
    router.e_score_correction_bias = torch.ones(4)
    router.routed_scaling_factor = 1.0

    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    logits = torch.ones((1, 4))
    assert router._compute_routing(None, logits, torch.int32) == "official"

    lightop_package = ModuleType("lightop")
    lightop_package.op = SimpleNamespace(
        moe_fused_gate=lambda *args: (
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int64),
        )
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop_package)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSE_MOE_GATE", True)
    weights, ids = router._compute_routing(None, logits, torch.int32)
    assert weights.shape == (1, 1)
    assert ids.dtype == torch.int32


def _fake_deepep_ll_module() -> ModuleType:
    class DeepEPLLPrepareAndFinalize:
        def __init__(
            self,
            buffer,
            max_tokens_per_rank,
            num_dispatchers,
            use_fp8_dispatch=False,
            global_to_physical=None,
            physical_to_global=None,
            local_expert_global_ids=None,
        ):
            self.buffer = buffer
            self.max_tokens_per_rank = max_tokens_per_rank
            self.use_fp8_dispatch = use_fp8_dispatch

        def _do_quant(self, x, a1_dtype, quant_config):
            return x, a1_dtype

        def prepare_async(
            self,
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=False,
        ):
            return "official-feature-off"

        def _receiver(self, expert_x, expert_num_tokens, a1_scale, a1_dtype, quant_config):
            return expert_x

    return _module(
        patch_deepep_ll.TARGET_MODULE,
        DeepEPLLPrepareAndFinalize=DeepEPLLPrepareAndFinalize,
    )


def test_deepep_ll_feature_off_delegates_and_expanded_signatures(monkeypatch: pytest.MonkeyPatch):
    module = _fake_deepep_ll_module()
    assert patch_deepep_ll.apply_to_module(module) is True
    assert patch_deepep_ll.apply_to_module(module) is False
    cls = module.DeepEPLLPrepareAndFinalize
    instance = object.__new__(cls)
    instance.use_int8_dispatch = False
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_DPSK_V4_DEEPEP_LL_USE_HCU_DISPATCH_API",
        False,
    )
    assert (
        instance.prepare_async(None, None, None, 1, None, False, None)
        == "official-feature-off"
    )
    inspect = importlib.import_module("inspect")
    assert "use_int8_dispatch" in inspect.signature(cls.__init__).parameters
    assert "expert_num_tokens" in inspect.signature(cls._do_quant).parameters


def test_deepep_ll_hcu_int8_dispatch_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_deepep_ll_module()

    class ExpertTokensMetadata:
        def __init__(self, expert_num_tokens, expert_num_tokens_cpu):
            self.expert_num_tokens = expert_num_tokens
            self.expert_num_tokens_cpu = expert_num_tokens_cpu

    module.torch = torch
    module.mk = SimpleNamespace(ExpertTokensMetadata=ExpertTokensMetadata)
    module.dbo_current_ubatch_id = lambda: 0
    module.DEEPEP_QUANT_BLOCK_SIZE = 128
    module.envs = SimpleNamespace(VLLM_DEEPEPLL_NVFP4_DISPATCH=False)
    module.normalize_batched_scales_shape = (
        lambda scales, experts: scales.reshape(experts, -1, 1)
    )
    module.dequant_fp8 = lambda values, scales: values.float() * scales.reshape(
        values.shape[0], -1, 1
    )
    module.moe_kernel_quantize_input = lambda *args, **kwargs: (args[0], None)
    assert patch_deepep_ll.apply_to_module(module) is True
    cls = module.DeepEPLLPrepareAndFinalize
    cls.SUPPORTED_HIDDEN_SIZES = [2048]
    cls._map_global_to_physical_ids = lambda self, ids: ids

    calls = {}
    expert_x = (
        torch.ones((1, 1, 2048), dtype=torch.int8),
        torch.ones((1, 1, 1)),
    )
    expert_counts = torch.tensor([1], dtype=torch.int32)

    class Buffer:
        def low_latency_dispatch(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return expert_x, expert_counts, "handle", None, lambda: None

    instance = cls(Buffer(), 8, 1, use_int8_dispatch=True)
    instance.handles = [None, None]
    instance.use_ue8m0_dispatch = False
    quant_config = SimpleNamespace(
        quant_dtype=torch.int8,
        block_shape=None,
        per_act_token_quant=True,
        a1_scale=None,
        a2_scale=None,
        a1_gscale=None,
    )
    hidden = torch.ones((1, 2048), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1))
    topk_ids = torch.zeros((1, 1), dtype=torch.int64)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_DPSK_V4_DEEPEP_LL_USE_HCU_DISPATCH_API",
        False,
    )
    hook, receiver = instance.prepare_async(
        hidden,
        topk_weights,
        topk_ids,
        1,
        None,
        False,
        quant_config,
    )
    assert callable(hook)
    assert calls["args"][2] is topk_weights
    assert calls["kwargs"]["quant_type"] == 1
    assert instance.handles[0] == "handle"
    quanted, scales, metadata, routed_ids, routed_weights = receiver()
    assert quanted is expert_x[0]
    assert scales is expert_x[1]
    assert metadata.expert_num_tokens is expert_counts
    assert routed_ids is None and routed_weights is None

    fp8_instance = cls(None, 8, 1, use_fp8_dispatch=True)
    fp8_instance.use_int8_dispatch = False
    fp8_values = torch.ones((1, 2, 4), dtype=torch.float8_e4m3fn)
    fp8_scales = torch.ones((2, 1))
    fp8_config = SimpleNamespace(
        quant_dtype=torch.float8_e4m3fn,
        block_shape=None,
        per_act_token_quant=True,
    )
    values, normalized_scales = fp8_instance._do_quant(
        (fp8_values, fp8_scales),
        torch.bfloat16,
        fp8_config,
    )
    assert values is fp8_values
    assert normalized_scales.shape == (1, 2, 1)


def test_custom_op_runner_rejects_post_import_callback():
    official = ModuleType(patch_moe_runner.TARGET_MODULE)
    with pytest.raises(PatchCompatibilityError, match="must be replaced before import"):
        patch_moe_runner.apply_to_module(official)


def test_moe_runner_and_shared_experts_cold_replacement_contract():
    repository = Path(__file__).resolve().parents[2]
    target_vllm = Path(
        os.environ.get("VLLM_V0251_SOURCE_ROOT", repository.parent / "vllm_0251")
    ).resolve()
    if not (target_vllm / "vllm" / "__init__.py").is_file():
        raise RuntimeError(
            f"VLLM_V0251_SOURCE_ROOT does not contain vllm: {target_vllm}"
        )
    python_path = [str(target_vllm), str(repository)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_path.append(existing)

    script = textwrap.dedent(
        """
        import importlib
        import inspect
        import os
        from pathlib import Path
        from types import SimpleNamespace

        import torch
        import vllm

        target_root = Path(os.environ["VLLM_V0251_SOURCE_ROOT"]).resolve()
        target_file = Path(vllm.__file__).resolve()
        assert target_file.is_relative_to(target_root), (
            f"vllm resolved outside target root: {target_file} not under {target_root}"
        )

        from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
        from vllm_hcu.patch.runtime_state import PatchRegistry
        from vllm_hcu.patch.worker.op_opt.moe import (
            patch_moe_runner,
            patch_shared_experts,
        )

        coordinator = ExactImportCoordinator(registry=PatchRegistry())
        coordinator.install()
        coordinator.register_replacement(
            patch_shared_experts.PATCH_ID,
            patch_shared_experts.TARGET_MODULE,
            patch_shared_experts.REPLACEMENT_MODULE,
            targets=patch_shared_experts.TARGETS,
            late_policy="fail",
        )
        coordinator.register_replacement(
            patch_moe_runner.PATCH_ID,
            patch_moe_runner.TARGET_MODULE,
            patch_moe_runner.REPLACEMENT_MODULE,
            targets=patch_moe_runner.TARGETS,
            late_policy="fail",
        )
        runner_module = importlib.import_module(patch_moe_runner.TARGET_MODULE)
        shared_module = importlib.import_module(patch_shared_experts.TARGET_MODULE)
        assert runner_module.__name__ == patch_moe_runner.REPLACEMENT_MODULE
        assert shared_module.__name__ == patch_shared_experts.REPLACEMENT_MODULE
        assert runner_module.SharedExperts is shared_module.SharedExperts
        assert patch_moe_runner.apply_to_module(runner_module) is True
        assert patch_moe_runner.apply_to_module(runner_module) is False
        assert patch_shared_experts.apply_to_module(shared_module) is True
        assert patch_shared_experts.apply_to_module(shared_module) is False

        schema = tuple(inspect.signature(runner_module._moe_forward).parameters)
        assert schema == (
            "hidden_states", "router_logits", "shared_experts_input", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
            "layer_name", "hidden_dim_unpadded",
        )
        assert tuple(inspect.signature(runner_module.MoERunner.forward).parameters) == (
            "self", "hidden_states", "router_logits", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
        )

        class QuantMethod:
            is_monolithic = False

            def apply(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
                i_q=None,
                i_s=None,
            ):
                del (
                    layer,
                    topk_weights,
                    topk_ids,
                    shared_experts,
                    shared_experts_input,
                )
                assert i_q is quanted and i_s is scale
                return x + 1

        class Router:
            def select_experts(self, **kwargs):
                del kwargs
                raise AssertionError("preselected routing must bypass router")

        runner = object.__new__(runner_module.MoERunner)
        runner.routed_experts = SimpleNamespace(quant_method=QuantMethod())
        runner._shared_experts = None
        runner.router = Router()
        hidden = torch.ones((2, 3))
        quanted = torch.ones((2, 3), dtype=torch.int8)
        scale = torch.ones((2, 1))
        topk_weights = torch.ones((2, 1))
        topk_ids = torch.zeros((2, 1), dtype=torch.int32)
        shared, output = runner._apply_quant_method(
            hidden,
            None,
            None,
            quanted_hidden_states=quanted,
            scale=scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        assert shared is None
        assert torch.equal(output, hidden + 1)

        class UnsupportedQuantMethod:
            is_monolithic = False

            def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
                del layer, topk_weights, topk_ids, shared_experts_input
                return x

        runner.routed_experts.quant_method = UnsupportedQuantMethod()
        runner.__dict__.pop("_supports_quanted_inputs", None)
        try:
            runner._apply_quant_method(
                hidden,
                None,
                None,
                quanted_hidden_states=quanted,
                scale=scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
        except RuntimeError as error:
            assert "does not accept i_q/i_s" in str(error)
        else:
            raise AssertionError("unsupported prequantized input did not fail")

        class SharedLayer:
            def forward(self, value, x_and_scale_quanted=None):
                assert x_and_scale_quanted == (quanted, scale)
                return value + 2

            __call__ = forward

        shared_experts = object.__new__(shared_module.SharedExperts)
        shared_experts._layer = SharedLayer()
        value = shared_experts._run_layer(hidden, (quanted, scale))
        assert torch.equal(value, hidden + 2)
        shared_source = inspect.getsource(shared_module.SharedExperts)
        assert "_output_pending_on_stream" in shared_source
        assert "VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH" in shared_source
        assert "current_platform.is_cuda_alike()" in shared_source
        coordinator.reset_for_tests()
        """
    )
    environment = os.environ.copy()
    environment["VLLM_PLUGINS"] = "__disabled__"
    environment["VLLM_V0251_SOURCE_ROOT"] = str(target_vllm)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_int8_expert_quant_adapter_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import int8_quant_runtime

    calls = []

    def official(A, A_scale, per_act_token, block_shape):
        calls.append((A, A_scale, per_act_token, block_shape))
        return "official"

    module = _module(
        patch_utils.TARGET_MODULE,
        _int8_quantize=official,
    )
    assert patch_utils.apply_to_module(module) is True
    tensor = torch.ones((1, 2, 2))
    assert module._int8_quantize(tensor, None, True, None) == "official"
    assert calls == [(tensor, None, True, None)]

    facade_calls = []
    runtime_name = (
        "vllm_hcu.model_executor.layers.fused_moe.int8_quant_runtime"
    )
    facade = _module(
        runtime_name,
        per_token_quant_int8=lambda values, counts: (
            facade_calls.append((values, counts))
            or (values.to(torch.int8), torch.ones(values.shape[:-1] + (1,)))
        ),
    )
    monkeypatch.setitem(sys.modules, runtime_name, facade)
    counts = torch.tensor([1], dtype=torch.int32)
    quanted, scales = module._int8_quantize(
        tensor,
        None,
        True,
        None,
        expert_num_tokens=counts,
    )
    assert facade_calls == [(tensor, counts)]
    assert quanted.dtype == torch.int8
    assert scales.shape == (1, 2, 1)
    with pytest.raises(ValueError, match="without block_shape"):
        module._int8_quantize(
            tensor,
            None,
            True,
            [128, 128],
            expert_num_tokens=counts,
        )

    class CpuLauncher:
        def __getitem__(self, grid):
            del grid

            def launch(
                values,
                output,
                output_scales,
                stride_x,
                stride_xq,
                hidden,
                tokens_per_expert,
                max_tokens,
                **kwargs,
            ):
                del stride_x, stride_xq, hidden, max_tokens, kwargs
                valid = int(tokens_per_expert[0])
                for row in range(valid):
                    source = values[0, row].float()
                    scale_value = source.abs().max().clamp_min(1e-10) / 127.0
                    output[0, row].copy_(torch.round(source / scale_value).to(torch.int8))
                    output_scales[0, row, 0] = scale_value

            return launch

    monkeypatch.setattr(
        int8_quant_runtime,
        "_per_token_quant_int8_one_kernel",
        CpuLauncher(),
    )
    monkeypatch.setattr(
        int8_quant_runtime.triton,
        "next_power_of_2",
        lambda value: 1 << (value - 1).bit_length(),
        raising=False,
    )
    values = torch.tensor([[[1.0, -2.0], [5.0, 9.0]]])
    quanted, scales = int8_quant_runtime.per_token_quant_int8(values, counts)
    expected_scale = torch.tensor(2.0 / 127.0)
    assert torch.isclose(scales[0, 0, 0], expected_scale)
    assert torch.equal(quanted[0, 0], torch.tensor([64, -127], dtype=torch.int8))


def test_importing_adapters_does_not_eager_import_optional_moe_stacks():
    optional = ("deep_ep", "deepgemm", "lightop")
    before = {name: sys.modules.get(name) for name in optional}
    for adapter in ADAPTERS:
        importlib.reload(adapter)
    for name in optional:
        assert sys.modules.get(name) is before[name]
