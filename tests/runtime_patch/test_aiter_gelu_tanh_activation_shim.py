# SPDX-License-Identifier: Apache-2.0
"""The GELU_TANH clone must see every upstream MoEActivation member."""

import contextlib
import sys
from enum import Enum, IntEnum
from types import ModuleType

import pytest

from vllm_hcu.patch.worker.op_opt.moe import patch_rocm_aiter_moe as patch


class Activation(Enum):
    SILU = "silu"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    SWIGLUOAI = "swigluoai"
    SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
    SITU = "situ"


class ActivationMethod(IntEnum):
    SILU = 0
    GELU = 1


@pytest.fixture
def fused_experts(monkeypatch):
    runtime = ModuleType("vllm_hcu.model_executor.layers.fused_moe.aiter_runtime")
    runtime.aiter_moe_request_context = lambda moe_config: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    # Mirrors the upstream body: SITU is read unconditionally, after the
    # activation has already been mapped.
    namespace = {"MoEActivation": Activation, "ActivationMethod": ActivationMethod}
    exec("""def rocm_aiter_fused_experts(
    hidden_states, w1, w2, topk_weights, topk_ids, moe_config,
    activation=MoEActivation.SILU, apply_router_weight_on_input=False,
    expert_mask=None, quant_config=None, a1q_scale=None,
    num_local_tokens=None, output_dtype=None, moe_sorting_dispatch_policy=0,
    shared_w1=None, shared_w2=None, shared_w1_scale=None,
    shared_w2_scale=None, shared_expert_id=-1,
):
    if activation == MoEActivation.SILU:
        method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        method = ActivationMethod.GELU
    else:
        method = None
    uses_situ = activation == MoEActivation.SITU
    return method, uses_situ
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
            return True, "upstream"

    target = ModuleType(patch.TARGET_MODULE)
    target.IntEnum = IntEnum
    target.ActivationMethod = ActivationMethod
    target.MoEActivation = Activation
    target.AiterExperts = Experts
    target.rocm_aiter_fused_experts = namespace["rocm_aiter_fused_experts"]
    assert patch.apply_to_module(target)
    return target.rocm_aiter_fused_experts


def _call(fn, activation):
    return fn(None, None, None, None, None, None, activation=activation)


def test_gelu_tanh_clone_resolves_unconditional_situ_check(fused_experts):
    method, uses_situ = _call(fused_experts, Activation.GELU_TANH)
    assert method == 3  # HCU ActivationMethod.GELU_TANH
    assert uses_situ is False


def test_non_tanh_activations_keep_upstream_body(fused_experts):
    assert _call(fused_experts, Activation.SILU) == (ActivationMethod.SILU, False)
    assert _call(fused_experts, Activation.SITU) == (None, True)
