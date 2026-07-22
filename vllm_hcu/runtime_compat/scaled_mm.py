# SPDX-License-Identifier: Apache-2.0
"""Target-native Triton compatibility for channel-wise FP8 scaled MM."""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from math import prod
from threading import RLock
from types import ModuleType

import torch


_TARGET_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.pytorch"
_TRITON_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "triton_scaled_mm"
)
_CUSTOM_OP_NAME = "hcu_channel_fp8_target_triton_scaled_mm"
_CUSTOM_OP_LOCK = RLock()
_CUSTOM_OP: Callable[..., torch.Tensor] | None = None
_CUSTOM_OP_BACKEND: Callable[..., torch.Tensor] | None = None
_CUSTOM_OP_REGISTRATION_ERROR: str | None = None


def _eager_shape_check(condition, message: str) -> None:
    """Validate a concrete shape relation outside compiler capture."""

    if not condition:
        raise ValueError(message)


def _activation_scale_shape_ok(scale: torch.Tensor, tokens):
    if scale.ndim == 0:
        return True
    if scale.ndim == 1:
        return (scale.shape[0] == 1) | (scale.shape[0] == tokens)
    if scale.ndim == 2:
        return (
            ((scale.shape[0] == 1) | (scale.shape[0] == tokens))
            & (scale.shape[1] == 1)
        )
    return False


def _weight_scale_shape_ok(scale: torch.Tensor, channels):
    if scale.ndim == 0:
        return True
    if scale.ndim == 1:
        return (scale.shape[0] == 1) | (scale.shape[0] == channels)
    if scale.ndim == 2:
        return (
            ((scale.shape[0] == 1) | (scale.shape[0] == channels))
            & (scale.shape[1] == 1)
        )
    return False


def _validate_scale_shapes(
    As: torch.Tensor,
    Bs: torch.Tensor,
    tokens,
    channels,
) -> None:
    _eager_shape_check(
        _activation_scale_shape_ok(As, tokens),
        "channelwise FP8 activation scale must be scalar or per-token",
    )
    _eager_shape_check(
        _weight_scale_shape_ok(Bs, channels),
        "channelwise FP8 weight scale must be scalar or contain one value per "
        "output channel",
    )


def _validate_target_output(
    output,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("target triton_scaled_mm did not return a tensor")
    if (
        output.ndim != 2
        or tuple(output.shape) != (A.shape[0], B.shape[1])
        or output.device != A.device
        or output.dtype != out_dtype
    ):
        raise RuntimeError("target triton_scaled_mm returned an incompatible output")
    return output


def _channel_fp8_target_triton_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    backend = _CUSTOM_OP_BACKEND
    if backend is None:
        raise RuntimeError("Channel-FP8 target Triton backend is not initialized")
    # The dispatcher calls this real implementation outside Dynamo capture, so
    # these relations use concrete runtime dimensions and cannot become
    # piecewise graph inputs.  Repeat them here because the fake implementation
    # is the only custom-op body observed during compilation.
    _validate_scale_shapes(As, Bs, A.shape[0], B.shape[1])
    output = backend(
        A,
        B,
        scale_a=As,
        scale_b=Bs,
        out_dtype=out_dtype,
        bias=bias,
    )
    return _validate_target_output(output, A, B, out_dtype)


def _channel_fp8_target_triton_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del As, Bs, bias
    return torch.empty(
        (A.shape[0], B.shape[1]),
        dtype=out_dtype,
        device=A.device,
    )


def _resolve_channel_fp8_custom_op() -> Callable[..., torch.Tensor]:
    return getattr(torch.ops.vllm, _CUSTOM_OP_NAME)


def _register_channel_fp8_custom_op() -> Callable[..., torch.Tensor]:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_CUSTOM_OP_NAME,
        op_func=_channel_fp8_target_triton_impl,
        mutates_args=[],
        fake_impl=_channel_fp8_target_triton_fake,
    )
    custom_op = _resolve_channel_fp8_custom_op()
    if not callable(custom_op):
        raise RuntimeError(
            f"registered vllm::{_CUSTOM_OP_NAME} operator is not callable"
        )
    return custom_op


def _ensure_channel_fp8_custom_op(
    triton_scaled_mm: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    global _CUSTOM_OP, _CUSTOM_OP_BACKEND, _CUSTOM_OP_REGISTRATION_ERROR

    if not callable(triton_scaled_mm):
        raise TypeError("target triton_scaled_mm backend must be callable")
    with _CUSTOM_OP_LOCK:
        if _CUSTOM_OP_REGISTRATION_ERROR is not None:
            raise RuntimeError(
                "Channel-FP8 target Triton custom-op registration previously "
                f"failed: {_CUSTOM_OP_REGISTRATION_ERROR}"
            )
        if _CUSTOM_OP is not None:
            if _CUSTOM_OP_BACKEND is not triton_scaled_mm:
                raise RuntimeError(
                    "Channel-FP8 target Triton custom op is already bound to a "
                    "different backend"
                )
            return _CUSTOM_OP

        _CUSTOM_OP_BACKEND = triton_scaled_mm
        try:
            custom_op = _register_channel_fp8_custom_op()
        except Exception as exc:
            _CUSTOM_OP_BACKEND = None
            _CUSTOM_OP_REGISTRATION_ERROR = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                "failed to register the HCU-owned Channel-FP8 target Triton "
                "custom op"
            ) from exc
        _CUSTOM_OP = custom_op
        return custom_op


def _reset_channel_fp8_custom_op_for_tests() -> None:
    """Reset Python registration state; tests must not register a real op first."""

    global _CUSTOM_OP, _CUSTOM_OP_BACKEND, _CUSTOM_OP_REGISTRATION_ERROR
    with _CUSTOM_OP_LOCK:
        _CUSTOM_OP = None
        _CUSTOM_OP_BACKEND = None
        _CUSTOM_OP_REGISTRATION_ERROR = None


def install_fp8_scaled_mm_compat(module: ModuleType | None = None) -> None:
    if module is None:
        from vllm.model_executor.kernels.linear.scaled_mm import (
            pytorch as module,
        )
    if not isinstance(module, ModuleType) or module.__name__ != _TARGET_MODULE:
        raise RuntimeError(
            "channelwise FP8 scaled-mm adapter requires the exact vLLM "
            f"module {_TARGET_MODULE}"
        )

    ChannelWiseTorchFP8ScaledMMLinearKernel = getattr(
        module, "ChannelWiseTorchFP8ScaledMMLinearKernel", None
    )
    if not isinstance(ChannelWiseTorchFP8ScaledMMLinearKernel, type):
        raise RuntimeError(
            "vLLM ChannelWiseTorchFP8ScaledMMLinearKernel is unavailable"
        )

    already_installed = getattr(
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        "_hcu_fp8_patch_applied",
        False,
    )
    if already_installed:
        if (
            getattr(
                ChannelWiseTorchFP8ScaledMMLinearKernel,
                "_hcu_fp8_backend",
                None,
            )
            == "target-triton"
            and getattr(
                ChannelWiseTorchFP8ScaledMMLinearKernel.get_output_padding,
                "_hcu_fp8_target_triton_wrapper",
                False,
            )
            and getattr(
                ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm,
                "_hcu_fp8_target_triton_wrapper",
                False,
            )
        ):
            return
        raise RuntimeError(
            "channelwise FP8 scaled-mm adapter marker exists without the "
            "reviewed target Triton wrapper ownership"
        )

    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    triton_parameters = tuple(signature(triton_scaled_mm).parameters)
    if triton_parameters[:6] != (
        "input",
        "weight",
        "scale_a",
        "scale_b",
        "out_dtype",
        "bias",
    ):
        raise RuntimeError(
            "vLLM target triton_scaled_mm signature drifted from the reviewed "
            "v0.25 contract"
        )

    channel_fp8_custom_op = _ensure_channel_fp8_custom_op(triton_scaled_mm)

    original_get_output_padding = (
        ChannelWiseTorchFP8ScaledMMLinearKernel.get_output_padding
    )
    original_apply_scaled_mm = ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm

    def new_get_output_padding(self):
        return None

    new_get_output_padding._hcu_fp8_target_triton_wrapper = True

    ChannelWiseTorchFP8ScaledMMLinearKernel.get_output_padding = (
        new_get_output_padding
    )
    ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_rocm_no_output_padding = True

    def new_apply_scaled_mm(
        self,
        *,
        A,
        B,
        As,
        Bs,
        out_dtype,
        bias,
        output_shape,
    ):
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            raise TypeError("channelwise FP8 scaled-mm A and B must be tensors")
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("channelwise FP8 scaled-mm requires 2D A and B")

        m, k = A.shape
        if m <= 0 or k <= 0 or B.shape[0] != k or B.shape[1] <= 0:
            raise ValueError(
                "channelwise FP8 scaled-mm requires A=[M,K], B=[K,N] with "
                "positive dimensions"
            )
        n = B.shape[1]
        if not A.is_contiguous():
            raise ValueError("channelwise FP8 scaled-mm A must be contiguous")
        if B.stride() != (1, k):
            raise ValueError(
                "channelwise FP8 scaled-mm B must retain the target v0.25 "
                "column-major [K,N] post-load view"
            )
        if A.device != B.device or A.dtype != B.dtype:
            raise ValueError(
                "channelwise FP8 scaled-mm A and B must share device and dtype"
            )
        if not A.is_floating_point() or A.element_size() != 1:
            raise ValueError("channelwise FP8 scaled-mm requires FP8 A and B")

        if not isinstance(As, torch.Tensor) or not isinstance(Bs, torch.Tensor):
            raise TypeError("channelwise FP8 scaled-mm scales must be tensors")
        # Constructing a symbolic equality here and passing it to
        # ``torch._check`` makes vLLM's piecewise splitter thread a SymBool
        # across subgraphs.  The target Torch Inductor does not accept a
        # ``sympy.Equality`` graph input.  Keep the friendly validation for
        # concrete eager calls; the custom-op implementation delegates to the
        # target v0.25 Triton backend and repeats the HCU/target scale contract
        # with concrete runtime dimensions.
        if not torch.compiler.is_compiling():
            _validate_scale_shapes(As, Bs, m, n)
        if (
            As.device != A.device
            or Bs.device != A.device
            or As.dtype != Bs.dtype
            or not As.is_floating_point()
        ):
            raise ValueError(
                "channelwise FP8 scales must share a floating dtype and the "
                "operand device"
            )

        if not isinstance(output_shape, (list, tuple)) or len(output_shape) < 2:
            raise ValueError("channelwise FP8 output_shape must have at least 2 dims")
        if not torch.compiler.is_compiling():
            _eager_shape_check(
                (prod(output_shape[:-1]) == m) & (output_shape[-1] == n),
                "channelwise FP8 output_shape does not match A tokens and B "
                "output channels",
            )
        if not isinstance(out_dtype, torch.dtype) or not out_dtype.is_floating_point:
            raise TypeError("channelwise FP8 output dtype must be floating point")
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("channelwise FP8 bias must be a tensor or None")
            if (
                tuple(bias.shape) != (n,)
                or bias.device != A.device
                or not bias.is_floating_point()
            ):
                raise ValueError(
                    "channelwise FP8 bias must be floating [N] on the operand "
                    "device"
                )

        output = channel_fp8_custom_op(
            A,
            B,
            As,
            Bs,
            out_dtype,
            bias,
        )
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("target triton_scaled_mm did not return a tensor")
        if (
            output.ndim != 2
            or output.device != A.device
            or output.dtype != out_dtype
        ):
            raise RuntimeError(
                "target triton_scaled_mm returned an incompatible output"
            )
        if not torch.compiler.is_compiling():
            try:
                _eager_shape_check(
                    (output.shape[0] == m) & (output.shape[1] == n),
                    "target triton_scaled_mm returned an incompatible output",
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
        return output.view(*output_shape)

    new_apply_scaled_mm._hcu_fp8_target_triton_wrapper = True

    ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm = new_apply_scaled_mm
    ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_fp8_patch_applied = True
    ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_fp8_backend = "target-triton"
    ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_original_get_output_padding = (
        original_get_output_padding
    )
    ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_original_apply_scaled_mm = (
        original_apply_scaled_mm
    )
