# SPDX-License-Identifier: Apache-2.0
"""SWIGLUSTEP capability must match the channel/token AITER implementation."""

from enum import Enum, IntEnum
from types import ModuleType, SimpleNamespace

import pytest

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym, kFp8DynamicTensorSym, kFp8DynamicTokenSym,
    kFp8Static128BlockSym, kFp8StaticChannelSym, kFp8StaticTensorSym,
    kInt8DynamicTokenSym, kInt8StaticChannelSym,
)
from vllm_hcu.patch.worker.op_opt.moe import patch_rocm_aiter_moe as patch


@pytest.fixture
def expert_class():
    class Activation(Enum):
        SILU = "silu"
        GELU = "gelu"
        GELU_TANH = "gelu_tanh"
        SWIGLUSTEP = "swiglustep"
        SWIGLUOAI = "swigluoai"
        SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"

    class ActivationMethod(IntEnum):
        SILU = 0
        GELU = 1

    # No device implementation is called; this target exercises the installed
    # adapter including its existing signature/lifecycle validation.
    namespace = {"MoEActivation": Activation}
    exec("""def rocm_aiter_fused_experts(
    hidden_states, w1, w2, topk_weights, topk_ids, moe_config,
    activation=MoEActivation.SILU, apply_router_weight_on_input=False,
    expert_mask=None, quant_config=None, a1q_scale=None,
    num_local_tokens=None, output_dtype=None, moe_sorting_dispatch_policy=0,
    shared_w1=None, shared_w2=None, shared_w1_scale=None,
    shared_w2_scale=None, shared_expert_id=-1,
):
    raise AssertionError('GPU expert body must not run in capability tests')
""", namespace)

    class Experts:
        @staticmethod
        def _supports_activation(activation):
            return activation == Activation.SILU

        @staticmethod
        def _supports_current_device():
            return True

        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key):
            return True

        @staticmethod
        def is_supported_config(cls, moe_config, weight_key, activation_key, activation_format):
            if getattr(moe_config, "reject_upstream", False):
                return False, "upstream rejection"
            return cls._supports_activation(moe_config.activation), "upstream"

    target = ModuleType(patch.TARGET_MODULE)
    target.IntEnum = IntEnum
    target.ActivationMethod = ActivationMethod
    target.MoEActivation = Activation
    target.AiterExperts = Experts
    target.rocm_aiter_fused_experts = namespace["rocm_aiter_fused_experts"]
    assert patch.apply_to_module(target)
    assert not patch.apply_to_module(target)
    return Experts, Activation


@pytest.mark.parametrize("weight,activation,expected", [
    (kFp8StaticChannelSym, kFp8DynamicTokenSym, True),
    (kInt8StaticChannelSym, kInt8DynamicTokenSym, True),
    (kFp8Static128BlockSym, kFp8Dynamic128Sym, False),
    (kFp8StaticTensorSym, kFp8DynamicTensorSym, False),
    (kFp8StaticTensorSym, kFp8StaticTensorSym, False),
    (None, None, False),
    (kFp8StaticChannelSym, kInt8DynamicTokenSym, False),
    (kInt8StaticChannelSym, kFp8DynamicTokenSym, False),
    (kFp8StaticChannelSym, kFp8DynamicTensorSym, False),
])
def test_swiglustep_accepts_only_implemented_quantization_pairs(
    expert_class, weight, activation, expected
):
    experts, enum = expert_class
    config = SimpleNamespace(moe_backend="auto", activation=enum.SWIGLUSTEP)
    supported, reason = experts.is_supported_config(experts, config, weight, activation, "standard")
    assert supported is expected
    if not expected:
        assert "channel weights with dynamic per-token activations" in reason


@pytest.mark.parametrize("weight,activation", [
    (kFp8Static128BlockSym, kFp8Dynamic128Sym),
    (None, None),
])
def test_other_activations_preserve_original_capability_decision(expert_class, weight, activation):
    experts, enum = expert_class
    config = SimpleNamespace(moe_backend="auto", activation=enum.SILU)
    assert experts.is_supported_config(experts, config, weight, activation, "standard") == (True, "upstream")


def test_upstream_failure_is_not_overridden_for_supported_pair(expert_class):
    experts, enum = expert_class
    config = SimpleNamespace(moe_backend="auto", activation=enum.SWIGLUSTEP, reject_upstream=True)
    assert experts.is_supported_config(
        experts, config, kFp8StaticChannelSym, kFp8DynamicTokenSym, "standard"
    ) == (False, "upstream rejection")
