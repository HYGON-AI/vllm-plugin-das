# SPDX-License-Identifier: Apache-2.0
"""HCU-owned INT8 linear implementation backed by LMSlim hipBLASLt."""

from __future__ import annotations

import torch


class HcuInt8LinearError(RuntimeError):
    """The explicitly enabled HCU INT8 kernel cannot execute safely."""


def apply_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor | None = None,
    input_zero_point: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply symmetric W8A8 linear and preserve arbitrary token dimensions."""

    del input_scale  # Dynamic and fused paths supply their actual per-token scale.
    if input_zero_point is not None or azp_adj is not None:
        raise HcuInt8LinearError(
            "HCU hipBLASLt W8A8 linear supports symmetric activations only; "
            "input_zero_point/azp_adj must be None"
        )
    if input.ndim < 2 or weight.ndim != 2:
        raise HcuInt8LinearError(
            f"HCU W8A8 linear expects input rank >=2 and weight rank 2, got "
            f"{input.ndim} and {weight.ndim}"
        )

    try:
        from vllm_hcu.platforms import envs as henvs
    except Exception as exc:
        raise HcuInt8LinearError("HCU environment flags are unavailable") from exc

    use_fused_input = (
        bool(henvs.VLLM_HCU_USE_CUSTOM_OPS)
        and (
            bool(henvs.VLLM_HCU_USE_FUSED_SILU_MUL_QUANT)
            or bool(henvs.VLLM_HCU_USE_FUSED_RMS_QUANT)
        )
        and x_and_scale_quanted is not None
    )
    if use_fused_input:
        if (
            not isinstance(x_and_scale_quanted, tuple)
            or len(x_and_scale_quanted) != 2
            or not all(isinstance(item, torch.Tensor) for item in x_and_scale_quanted)
        ):
            raise HcuInt8LinearError(
                "fused HCU W8A8 input must be a (quantized_tensor, scale) tuple"
            )
        x_q, x_scale = x_and_scale_quanted
    else:
        try:
            from lmslim.layers.gemm.int8_utils import per_token_quant_int8
        except Exception as exc:
            raise HcuInt8LinearError(
                "HCU W8A8 linear is enabled, but LMSlim per-token INT8 "
                "quantization is unavailable"
            ) from exc
        x_q, x_scale = per_token_quant_int8(input)

    if x_q.shape != input.shape or x_scale.shape != (*input.shape[:-1], 1):
        raise HcuInt8LinearError(
            "HCU W8A8 quantized input/scale shapes do not match the activation"
        )
    if x_q.dtype is not torch.int8 or x_scale.dtype is not torch.float32:
        raise HcuInt8LinearError(
            "HCU W8A8 requires an int8 activation and float32 per-token scale"
        )
    if weight.dtype is not torch.int8 or weight_scale.dtype is not torch.float32:
        raise HcuInt8LinearError(
            "HCU W8A8 requires int8 weights and float32 weight scales"
        )

    m = input.numel() // input.shape[-1]
    n, k = weight.shape
    if x_q.shape[-1] != k:
        raise HcuInt8LinearError(
            f"HCU W8A8 K mismatch: activation K={x_q.shape[-1]}, weight K={k}"
        )
    out_dtype = (
        params_dtype
        if params_dtype in (torch.bfloat16, torch.float16)
        else torch.bfloat16
    )
    if bias is not None and bias.dtype is not out_dtype:
        raise HcuInt8LinearError(
            f"HCU W8A8 bias dtype {bias.dtype} does not match output {out_dtype}"
        )

    try:
        from lmslim import quant_ops
    except Exception as exc:
        raise HcuInt8LinearError(
            "HCU W8A8 linear is enabled, but LMSlim quant_ops is unavailable"
        ) from exc

    x_q_2d = x_q.reshape(m, k).contiguous()
    x_scale_2d = x_scale.reshape(m, 1).contiguous()
    weight = weight.contiguous()
    weight_scale = weight_scale.contiguous()
    try:
        status, output = quant_ops.hipblaslt_w8a8_gemm(
            x_q_2d,
            weight,
            x_scale_2d,
            weight_scale,
            m,
            n,
            k,
            "NT",
            out_dtype,
        )
    except Exception as exc:
        raise HcuInt8LinearError(
            f"LMSlim hipBLASLt W8A8 GEMM failed for M={m}, N={n}, K={k}"
        ) from exc
    if status is not True or output.shape != (m, n):
        raise HcuInt8LinearError(
            "LMSlim hipBLASLt W8A8 GEMM returned an invalid status or shape"
        )
    if bias is not None:
        output = output + bias
    return output.view(*input.shape[:-1], n)


__all__ = ["HcuInt8LinearError", "apply_int8_linear"]
