# SPDX-License-Identifier: Apache-2.0

"""
HCU version of UnquantizedFusedMoEMethod.
Inherits from vllm's version and overrides process_weights_after_loading
to use AITER ASM weight shuffling + moe_kernel initialization.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    make_unquantized_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as _Original,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


def _copy_parameter_attrs(src: torch.nn.Parameter, dst: torch.nn.Parameter) -> None:
    if hasattr(src, "__dict__"):
        for key, value in src.__dict__.items():
            setattr(dst, key, value)


def _is_hcu_aiter_moe_asm_requested() -> bool:
    return envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MOE


def _activation_name(layer: torch.nn.Module) -> str | None:
    activation = getattr(layer, "activation", None)
    if activation is None:
        return None
    return getattr(activation, "value", activation)


def _raise_if_aiter_moe_asm_blocked(method: object, layer: torch.nn.Module) -> None:
    # Fail before ASM weight shuffle if the layer is not a W16A16 AITER MoE
    # layout supported by fused_experts_asm_impl.
    blockers: list[str] = []
    if not current_platform.is_rocm():
        blockers.append("current platform is not ROCm")
    if not rocm_aiter_ops.is_fused_moe_enabled():
        blockers.append("rocm_aiter_ops.is_fused_moe_enabled() is False")
    if getattr(method, "unquantized_backend", None) != UnquantizedMoeBackend.AITER:
        blockers.append("unquantized backend is not UnquantizedMoeBackend.AITER")
    if getattr(layer, "apply_router_weight_on_input", False):
        blockers.append("apply_router_weight_on_input=True is unsupported")
    if getattr(layer, "w13_bias", None) is not None:
        blockers.append("w13_bias is not None")
    if getattr(layer, "w2_bias", None) is not None:
        blockers.append("w2_bias is not None")
    if _activation_name(layer) not in ("silu", "gelu_tanh"):
        blockers.append("activation is not silu or gelu_tanh")

    w1 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    # Expected W16A16 MoE layout: w13=[E, 2N, K], w2=[E, K, N].
    if (
        not isinstance(w1, torch.Tensor)
        or not isinstance(w2, torch.Tensor)
        or w1.dim() != 3
        or w2.dim() != 3
        or w1.size(0) != w2.size(0)
    ):
        blockers.append("w13_weight / w2_weight shape rank or expert dim mismatch")
    else:
        if not w1.is_cuda or not w2.is_cuda:
            blockers.append("w13_weight or w2_weight is not on CUDA/ROCm device")
        if w1.dtype not in (torch.float16, torch.bfloat16):
            blockers.append(f"unsupported w13_weight dtype: {w1.dtype}")
        if w2.dtype not in (torch.float16, torch.bfloat16):
            blockers.append(f"unsupported w2_weight dtype: {w2.dtype}")
        if w2.size(1) != w1.size(2) or w1.size(1) != 2 * w2.size(2):
            blockers.append("w13_weight and w2_weight shapes are incompatible")
        if w1.size(2) % 32 != 0 or w2.size(2) % 16 != 0:
            blockers.append("shape alignment requires K % 32 == 0 and N % 16 == 0")
    if blockers:
        raise RuntimeError(
            "VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 but "
            "ASM MoE is blocked: " + "; ".join(blockers)
        )




class HcuUnquantizedFusedMoEMethod(_Original):
    """HCU version of UnquantizedFusedMoEMethod.

    Overrides process_weights_after_loading to use AITER ASM weight shuffling
    and initialize moe_kernel for the new vllm runner architecture.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _is_hcu_aiter_moe_asm_requested():
            return super().process_weights_after_loading(layer)

        if getattr(layer, "_hcu_aiter_moe_asm_packed", False):
            return

        _raise_if_aiter_moe_asm_blocked(self, layer)

        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        w1 = layer.w13_weight
        w2 = layer.w2_weight

        try:
            from aiter.ops.shuffle import asm_shuffle_weight_b8

            with torch.no_grad():
                replace_parameter(
                    layer, "w13_weight", asm_shuffle_weight_b8(w1, stage=1)
                )
                replace_parameter(
                    layer, "w2_weight", asm_shuffle_weight_b8(w2, stage=2)
                )

                new_w1 = layer.w13_weight
                new_w2 = layer.w2_weight
                _copy_parameter_attrs(w1, new_w1)
                _copy_parameter_attrs(w2, new_w2)
                setattr(new_w1, "aiter_moe_shuffled", True)
                setattr(new_w2, "aiter_moe_shuffled", True)

            self.moe_quant_config = self.get_fused_moe_quant_config(layer)

            # Initialize moe_kernel (required by new vllm runner)
            self.moe_kernel = make_unquantized_moe_kernel(
                quant_config=self.moe_quant_config,
                moe_config=self.moe,
                backend=self.unquantized_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

            layer._hcu_aiter_moe_asm_packed = True
        except Exception as exc:
            try:
                replace_parameter(layer, "w13_weight", w1)
                replace_parameter(layer, "w2_weight", w2)
            except Exception:
                pass
            layer_name = getattr(layer, "layer_name", "unknown")
            raise RuntimeError(
                "VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 failed "
                "while preparing ASM-shuffled weights for layer "
                f"'{layer_name}'."
            ) from exc
