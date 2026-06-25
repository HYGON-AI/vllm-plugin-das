# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


def _copy_parameter_attrs(src: torch.nn.Parameter, dst: torch.nn.Parameter) -> None:
    if hasattr(src, "__dict__"):
        for key, value in src.__dict__.items():
            setattr(dst, key, value)


def _activation_name(layer: torch.nn.Module) -> str | None:
    activation = getattr(layer, "activation", None)
    if activation is None:
        return None
    return getattr(activation, "value", activation)


def _is_hcu_aiter_moe_asm_requested() -> bool:
    return envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MOE


def _get_aiter_moe_asm_blockers(
    method: object,
    layer: torch.nn.Module,
) -> list[str]:
    blockers: list[str] = []
    if not current_platform.is_rocm():
        blockers.append("current platform is not ROCm")
    if not rocm_aiter_ops.is_fused_moe_enabled():
        blockers.append("rocm_aiter_ops.is_fused_moe_enabled() is False")
    if getattr(method, "unquantized_backend", None) != UnquantizedMoeBackend.AITER:
        blockers.append(
            "unquantized backend is not UnquantizedMoeBackend.AITER"
        )
    if getattr(layer, "apply_router_weight_on_input", False):
        blockers.append("apply_router_weight_on_input=True is unsupported")
    if getattr(layer, "w13_bias", None) is not None:
        blockers.append("w13_bias is unsupported")
    if getattr(layer, "w2_bias", None) is not None:
        blockers.append("w2_bias is unsupported")
    if _activation_name(layer) not in("silu","gelu_tanh"): 
    #if _activation_name(layer) != "silu":
        blockers.append("activation is not silu")

    w1 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    if w1 is None or w2 is None:
        blockers.append("w13_weight or w2_weight is missing")
        return blockers
    if not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor):
        blockers.append("w13_weight or w2_weight is not a tensor")
        return blockers
    if not w1.is_cuda or not w2.is_cuda:
        blockers.append("w13_weight or w2_weight is not on CUDA/ROCm device")
    if w1.dtype not in (torch.float16, torch.bfloat16):
        blockers.append(f"unsupported w13_weight dtype: {w1.dtype}")
    if w2.dtype not in (torch.float16, torch.bfloat16):
        blockers.append(f"unsupported w2_weight dtype: {w2.dtype}")
    if w1.dim() != 3 or w2.dim() != 3 or w1.size(0) != w2.size(0):
        blockers.append("w13_weight / w2_weight shape rank or expert dim mismatch")
        return blockers

    two_n, k_dim = w1.size(1), w1.size(2)
    if w2.size(1) != k_dim:
        blockers.append("w2_weight.shape[1] != w13_weight.shape[2]")
        return blockers
    n_dim = w2.size(2)
    if two_n != 2 * n_dim:
        blockers.append("w13_weight.shape[1] != 2 * w2_weight.shape[2]")
    if k_dim % 32 != 0 or n_dim % 16 != 0:
        blockers.append("shape alignment requires K % 32 == 0 and N % 16 == 0")
    return blockers


def _raise_if_aiter_moe_asm_blocked(
    method: object,
    layer: torch.nn.Module,
) -> None:
    blockers = _get_aiter_moe_asm_blockers(method, layer)
    if not blockers:
        return

    layer_name = getattr(layer, "layer_name", "unknown")
    raise RuntimeError(
        "VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 require every "
        f"patched MoE layer to use the HCU AITER ASM path, but layer "
        f"'{layer_name}' is unsupported: "
        + "; ".join(blockers)
    )


def _asm_shuffle_weight_b8(x: torch.Tensor, stage: int = 1) -> torch.Tensor:
    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    supported_dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8]
    if float8_dtype is not None:
        supported_dtypes.append(float8_dtype)
    assert x.dtype in supported_dtypes, f"unsupported dtype: {x.dtype}"

    is_int8_or_fp8 = x.dtype == torch.int8 or (
        float8_dtype is not None and x.dtype == float8_dtype
    )

    if is_int8_or_fp8:
        n_tile = 16
        k_tile = 16
        inner_k = 64
        inner_n = 64
        block_k = 256
        block_n = 128
        if stage == 1 and x.shape[-2] % 128 != 0 and x.shape[-2] % 64 == 0:
            block_n = 64
        if stage == 2:
            if x.shape[-1] % 128 == 0:
                block_k = 128
            elif x.shape[-1] % 128 == 96:
                block_n = 64
                block_k = 64
                assert x.shape[-2] % block_n == 0
                multiple = x.shape[-1] // block_k * block_k
                part1 = x[:, :, :multiple]
                part1 = part1.view(
                    -1,
                    part1.shape[-2] // block_n,
                    block_n // inner_n,
                    inner_n // n_tile,
                    n_tile,
                    part1.shape[-1] // block_k,
                    block_k // inner_k,
                    inner_k // k_tile,
                    k_tile,
                )
                part1 = part1.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
                part1 = part1.flatten(start_dim=1)

                part2 = x[:, :, multiple:]
                inner_k = 32
                block_k = 32
                part2 = part2.view(
                    -1,
                    part2.shape[-2] // block_n,
                    block_n // inner_n,
                    inner_n // n_tile,
                    n_tile,
                    part2.shape[-1] // block_k,
                    block_k // inner_k,
                    inner_k // k_tile,
                    k_tile,
                )
                part2 = part2.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
                part2 = part2.flatten(start_dim=1)
                shuffled = torch.cat((part1, part2), dim=1)
                return shuffled.view(*x.shape)
    elif x.dtype in (torch.float16, torch.bfloat16):
        n_tile = 16
        k_tile = 8
        inner_k = 32
        inner_n = 64
        block_k = 128
        block_n = 64
        if stage == 2:
            block_k = 32
    else:
        raise AssertionError(f"unsupported dtype: {x.dtype}")

    assert x.shape[-2] % block_n == 0, f"{x.shape[-2]} % {block_n} != 0"
    assert x.shape[-1] % block_k == 0, f"{x.shape[-1]} % {block_k} != 0"

    shuffled = x.view(
        -1,
        x.shape[-2] // block_n,
        block_n // inner_n,
        inner_n // n_tile,
        n_tile,
        x.shape[-1] // block_k,
        block_k // inner_k,
        inner_k // k_tile,
        k_tile,
    )
    shuffled = shuffled.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
    return shuffled.view(*x.shape)


def patch_aiter_moe_asm() -> None:
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    if getattr(UnquantizedFusedMoEMethod, "_hcu_aiter_moe_asm_patch_applied", False):
        return

    original_process_weights_after_loading = (
        UnquantizedFusedMoEMethod.process_weights_after_loading
    )
    original_forward_cuda = UnquantizedFusedMoEMethod.forward_cuda

    def patched_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _is_hcu_aiter_moe_asm_requested():
            return original_process_weights_after_loading(self, layer)

        if getattr(layer, "_hcu_aiter_moe_asm_packed", False):
            return

        _raise_if_aiter_moe_asm_blocked(self, layer)

        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        w1 = layer.w13_weight
        w2 = layer.w2_weight

        try:
            with torch.no_grad():
                replace_parameter(
                    layer, "w13_weight", _asm_shuffle_weight_b8(w1, stage=1)
                )
                replace_parameter(
                    layer, "w2_weight", _asm_shuffle_weight_b8(w2, stage=2)
                )

                new_w1 = layer.w13_weight
                new_w2 = layer.w2_weight
                _copy_parameter_attrs(w1, new_w1)
                _copy_parameter_attrs(w2, new_w2)
                setattr(new_w1, "aiter_moe_shuffled", True)
                setattr(new_w2, "aiter_moe_shuffled", True)
                self.moe_quant_config = self.get_fused_moe_quant_config(layer)
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

    def patched_forward_cuda(
        self,
        layer: "FusedMoE",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not _is_hcu_aiter_moe_asm_requested():
            return original_forward_cuda(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts_input,
            )

        _raise_if_aiter_moe_asm_blocked(
            self,
            layer,
        )
        if not getattr(layer, "_hcu_aiter_moe_asm_packed", False):
            layer_name = getattr(layer, "layer_name", "unknown")
            raise RuntimeError(
                "VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 expect "
                "ASM-shuffled weights to be prepared before forward, but layer "
                f"'{layer_name}' is not packed."
            )

        try:
            from aiter.fused_moe_asm_wna16 import fused_experts_asm_impl
        except Exception as exc:
            raise RuntimeError(
                "HCU AITER ASM MoE is enabled but aiter.fused_moe_asm_wna16 "
                "is unavailable."
            ) from exc

        quant_config = self.get_fused_moe_quant_config(layer)
        return fused_experts_asm_impl(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights.to(torch.float32),
            topk_ids=topk_ids.to(torch.int32),
            dtype=x.dtype,
            inplace=not layer.moe_config.disable_inplace,
            activation=_activation_name(layer),
            per_channel_quant=quant_config.per_act_token_quant,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            w1_zp=quant_config.w1_zp,
            w2_zp=quant_config.w2_zp,
            a1_scale=quant_config.a1_scale,
            a2_scale=quant_config.a2_scale,
            block_shape=quant_config.block_shape,
            use_shuffle=1,
        )

    UnquantizedFusedMoEMethod.process_weights_after_loading = (
        patched_process_weights_after_loading
    )
    UnquantizedFusedMoEMethod.forward_cuda = patched_forward_cuda
    UnquantizedFusedMoEMethod._hcu_aiter_moe_asm_patch_applied = True
