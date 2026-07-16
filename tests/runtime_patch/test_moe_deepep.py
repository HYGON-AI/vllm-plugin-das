# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
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
    ):
        del (
            moe,
            quant_config,
            routing_tables,
            allow_new_interface,
            use_monolithic,
        )
        return prepare_finalize

    fp8_dtype = torch.float8_e4m3fn
    module = _module(
        patch_all2all_utils.TARGET_MODULE,
        torch=torch,
        current_platform=SimpleNamespace(fp8_dtype=lambda: fp8_dtype),
        DeepEPLLPrepareAndFinalize=DeepEPLLPrepareAndFinalize,
        maybe_make_prepare_finalize=maybe_make_prepare_finalize,
    )
    assert patch_all2all_utils.apply_to_module(module) is True
    assert patch_all2all_utils.apply_to_module(module) is False

    fp8_config = SimpleNamespace(quant_dtype=fp8_dtype)
    result = module.maybe_make_prepare_finalize(None, fp8_config)
    assert result is prepare_finalize
    assert result.use_fp8_dispatch is True
    assert result.use_int8_dispatch is False

    int8_config = SimpleNamespace(quant_dtype=torch.int8)
    result = module.maybe_make_prepare_finalize(None, int8_config)
    assert result.use_fp8_dispatch is False
    assert result.use_int8_dispatch is True


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

    return _module(
        patch_config.TARGET_MODULE,
        torch=torch,
        current_platform=SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn),
        GroupShape=_GroupShape,
        _quant_flags_to_group_shape=flags,
        FusedMoEQuantConfig=FusedMoEQuantConfig,
        int8_w8a8_moe_quant_config=int8_config,
        FusedMoEParallelConfig=FusedMoEParallelConfig,
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


def test_config_signature_drift_fails_before_mutation():
    module = _fake_config_module()
    module._quant_flags_to_group_shape = lambda quant_dtype: quant_dtype
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_config.apply_to_module(module)
    assert not hasattr(module, "_vllm_hcu_moe_config_applied")


def test_aiter_and_triton_expert_capability_contract():
    class MoEActivation(Enum):
        SILU = "silu"
        GELU = "gelu"
        GELU_TANH = "gelu_tanh"
        SWIGLUOAI = "swigluoai"

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
            ):
                del hidden_states, w1, w2, topk_weights, topk_ids, moe_config
                del apply_router_weight_on_input, expert_map, quant_config
                del a1q_scale, num_local_tokens, output_dtype
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

    aiter_module = _module(
        patch_rocm_aiter_moe.TARGET_MODULE,
        IntEnum=IntEnum,
        ActivationMethod=ActivationMethod,
        MoEActivation=MoEActivation,
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


def test_fused_moe_aiter_feature_gate_and_obsolete_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    parameter_names = (
        "hidden_states", "w1", "w2", "topk_weights", "topk_ids", "inplace",
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
        disable_inplace=lambda: False,
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
        hidden, w1, w2, weights, ids, False, "silu", False, False, False,
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
            moe_layer,
            old_quant_method,
            prepare_finalize,
            shared_experts,
            inplace,
        ):
            del moe_layer, old_quant_method, prepare_finalize, shared_experts, inplace
            return None

        def apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts_input,
        ):
            del layer, topk_weights, topk_ids, shared_experts_input
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
    method = FusedMoEModularMethod.make(
        "layer", old_method, "prepare", "shared", True
    )
    assert FusedMoEKernel.last.arguments == (
        "prepare", ("prepare", "layer"), "shared", True, 32, 64,
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
    assert method.apply(layer, x, "weights", "ids", "shared", False, i_q, i_s) is x
    assert FusedMoEKernel.last.applied["quanted_hidden_states"] is i_q
    assert FusedMoEKernel.last.applied["scale"] is i_s
    with pytest.raises(ValueError, match="i_q and i_s together"):
        method.apply(layer, x, "weights", "ids", "shared", False, i_q, None)
    with pytest.raises(RuntimeError, match="use_nn_moe"):
        method.apply(layer, x, "weights", "ids", "shared", True)


def test_eplb_torch_map_and_record_numeric_contract(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def official(
        topk_ids,
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
        record_enabled,
    ):
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
    init_names = (
        "num_experts", "top_k", "hidden_size", "intermediate_size",
        "params_dtype", "renormalize", "use_grouped_topk", "num_expert_group",
        "topk_group", "quant_config", "tp_size", "ep_size", "dp_size",
        "pcp_size", "prefix", "custom_routing_function", "scoring_func",
        "routed_scaling_factor", "swiglu_limit", "e_score_correction_bias",
        "apply_router_weight_on_input", "activation", "is_act_and_mul",
        "enable_eplb", "num_redundant_experts", "has_bias",
        "is_sequence_parallel", "expert_mapping", "n_shared_experts",
        "router_logits_dtype", "gate", "shared_experts", "shared_expert_gate",
        "routed_input_transform", "routed_output_transform",
        "apply_routed_scale_to_output", "zero_expert_type", "hash_indices_table",
    )

    class UnquantizedFusedMoEMethod:
        def __init__(self):
            self.moe_quant_config = "official-config"

    class Runner:
        def __init__(self):
            self.replaced = None
            self.forwarded = None

        def _replace_quant_method(self, method):
            self.replaced = method

        def forward(self, *args, **kwargs):
            self.forwarded = (args, kwargs)
            return "forwarded"

    module = _module(
        patch_layer.TARGET_MODULE,
        UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod,
        Runner=Runner,
    )
    source = "class FusedMoE:\n"
    source += "    def __init__(self, " + ", ".join(
        f"{name}=None" for name in init_names
    ) + "):\n"
    source += textwrap.indent(
        textwrap.dedent(
            """
            self.moe_config = "moe-config"
            self.quant_method = UnquantizedFusedMoEMethod()
            self.base_quant_method = self.quant_method
            self.runner = Runner()
            self.local_num_experts = 2
            self._dsv4_channel_fp8_deepgemm_repacked = False
            """
        ),
        "        ",
    )
    source += (
        "    def forward(self, hidden_states, router_logits, input_ids):\n"
        "        return 'official-forward'\n\n"
        "    def get_expert_weights(self):\n"
        "        return 'official-weights'\n"
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
    layer = module.FusedMoE()
    assert isinstance(layer.quant_method, HcuUnquantizedFusedMoEMethod)
    assert layer.quant_method.moe_quant_config == "official-config"
    assert layer.base_quant_method is layer.quant_method
    assert layer.runner.replaced is layer.quant_method

    hidden = torch.ones((1, 2))
    quanted = torch.ones((1, 2), dtype=torch.int8)
    scale = torch.ones((1, 1))
    topk_weights = torch.ones((1, 1))
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    assert (
        layer.forward(
            hidden,
            None,
            None,
            quanted,
            scale,
            topk_weights,
            topk_ids,
        )
        == "forwarded"
    )
    assert layer.runner.forwarded[1] == {
        "quanted_hidden_states": quanted,
        "scale": scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }

    layer._dsv4_channel_fp8_deepgemm_repacked = True
    layer.w13_weight = torch.arange(24).reshape(2, 3, 4)
    layer.w2_weight = torch.arange(16).reshape(2, 2, 4)
    layer.w13_weight_scale = torch.arange(6).reshape(2, 3)
    layer.w2_weight_scale = torch.arange(4).reshape(2, 2)
    weights = layer.get_expert_weights()
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
        triton=SimpleNamespace(cdiv=lambda value, block: (value + block - 1) // block),
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
    ):
        calls.append((topk_ids, num_experts, block_size, expert_map))
        sorted_ids.fill_(topk_ids.numel())
        sorted_ids[: topk_ids.numel()].copy_(
            torch.arange(topk_ids.numel(), dtype=torch.int32)
        )
        expert_ids.copy_(torch.arange(expert_ids.numel(), dtype=torch.int32))
        num_tokens_post_pad.fill_(topk_ids.numel())

    lightop_package = ModuleType("lightop")
    lightop_package.op = SimpleNamespace(moe_align_block_size=lightop_align)
    monkeypatch.setitem(sys.modules, "lightop", lightop_package)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_LIGHTOP_MOE_ALIGN", True)
    sorted_ids, expert_ids, count = module.moe_align_block_size(ids, 2, 2)
    assert calls[0][1:] == (2, 2, None)
    assert torch.equal(sorted_ids[:2], torch.tensor([0, 1], dtype=torch.int32))
    assert torch.equal(expert_ids, torch.tensor([0, 1], dtype=torch.int32))
    assert count.item() == 2


def test_fp8_oracle_sidecar_selection_and_format_contract(
    monkeypatch: pytest.MonkeyPatch,
):
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

    module = _module(
        patch_fp8_oracle.TARGET_MODULE,
        Enum=Enum,
        Fp8MoeBackend=Fp8MoeBackend,
        backend_to_kernel_cls=backend_to_kernel_cls,
        map_fp8_backend=map_fp8_backend,
        select_fp8_moe_backend=select_fp8_moe_backend,
        convert_to_fp8_moe_kernel_format=convert_to_fp8_moe_kernel_format,
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

    experts_name = (
        "vllm_hcu.model_executor.layers.fused_moe.experts."
        "dpsk_v4_deep_gemm_moe"
    )
    experts_module = _module(
        experts_name,
        DeepEPDeepGemmContiguousExperts=UnsupportedExperts,
        DeepEPDeepGemmMaskedExperts=SupportedExperts,
    )
    monkeypatch.setitem(sys.modules, experts_name, experts_module)
    monkeypatch.setattr(
        patch_fp8_oracle,
        "_sidecar_backend",
        lambda config: "dpsk_deep_gemm",
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

    monkeypatch.setattr(patch_fp8_oracle, "_sidecar_backend", lambda config: "auto")
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
        "top_k", "global_num_experts", "renormalize", "indices_type_getter",
        "use_grouped_topk", "num_expert_group", "topk_group", "scoring_func",
        "num_fused_shared_experts", "routed_scaling_factor",
        "e_score_correction_bias", "custom_routing_function", "enable_eplb",
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
    clean_vllm = repository.parent / "vllm_dcu_v0.21"
    python_path = [str(repository)]
    if clean_vllm.is_dir():
        python_path.append(str(clean_vllm))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_path.append(existing)

    script = textwrap.dedent(
        """
        import importlib
        import inspect
        from types import SimpleNamespace

        import torch

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
                shared_experts_input,
                i_q=None,
                i_s=None,
            ):
                del layer, topk_weights, topk_ids, shared_experts_input
                assert i_q is quanted and i_s is scale
                return x + 1

        class Router:
            def select_experts(self, **kwargs):
                del kwargs
                raise AssertionError("preselected routing must bypass router")

        runner = object.__new__(runner_module.MoERunner)
        runner._quant_method = QuantMethod()
        runner._shared_experts = None
        runner.router = Router()
        hidden = torch.ones((2, 3))
        quanted = torch.ones((2, 3), dtype=torch.int8)
        scale = torch.ones((2, 1))
        topk_weights = torch.ones((2, 1))
        topk_ids = torch.zeros((2, 1), dtype=torch.int32)
        shared, output = runner._apply_quant_method(
            SimpleNamespace(_routing_replay_out=None),
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

        runner._quant_method = UnsupportedQuantMethod()
        runner.__dict__.pop("_supports_quanted_inputs", None)
        try:
            runner._apply_quant_method(
                SimpleNamespace(_routing_replay_out=None),
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
