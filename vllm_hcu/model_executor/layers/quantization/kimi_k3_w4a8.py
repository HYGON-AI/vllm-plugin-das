# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 routed-MoE INT4 W4A8 checkpoint format and gfx938 kernel.

The checkpoint format uses slimquant's packed two's-complement INT4 values:
``(even_k << 4) | odd_k`` with one FP32 scale per output channel.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
from vllm.logger import init_logger

from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

KIMI_K3_W4A8_QUANT_METHOD = "kimi_k3_w4a8"
SLIMQUANT_W4A8_QUANT_METHOD = "slimquant_w4a8"
KIMI_K3_W4A8_FORMAT = "kimi-k3-int4-w4a8-v1"
KIMI_K3_W4A8_GROUP_SIZE = 32
KIMI_K3_W4A8_PACKING = "twos-complement-high-even-low-odd"
KIMI_K3_W4A8_DEFAULT_METADATA = {
    "model_version": "slimquant_w4a8",
    "num_experts": 896,
    "top_k": 16,
    "hidden_size": 3584,
    "intermediate_size": 3072,
}
W4A8_MOE_BACKENDS = ("aiter", "triton", "lightop")
logger = init_logger(__name__)


def kimi_w4a8_requested_backend() -> str:
    """Read the W4A8-only selector without changing MXFP4 behavior."""

    requested = os.environ.get("VLLM_KIMI_W4A8_MOE_BACKEND", "aiter").strip().lower()
    assert requested in (*W4A8_MOE_BACKENDS, "auto"), (
        "VLLM_KIMI_W4A8_MOE_BACKEND must be triton|aiter|lightop|auto, "
        f"got {requested!r}"
    )
    assert requested != "lightop", "lightop is unsupported for INT4 W4A8 MoE"
    return requested


def _aiter_w4a8_api() -> dict[str, Any]:
    """Load exactly the AITER API required by the INT4 W4A8 route."""

    try:
        from aiter.moe import MoeQuantType, aiter_moe, get_aiter_moe_config
    except (ImportError, ModuleNotFoundError) as error:
        raise AssertionError("AITER is required for Kimi INT4 W4A8 MoE") from error
    assert hasattr(MoeQuantType, "W4A8"), (
        "installed AITER does not expose MoeQuantType.W4A8"
    )
    assert callable(aiter_moe) and callable(get_aiter_moe_config), (
        "installed AITER does not provide the required MoE API"
    )
    # The W4A8 layout conversion runs once in process_weights_after_loading
    # via repack_and_shuffle_w4a8 (aiter.ops.shuffle); aiter_moe itself needs
    # no per-forward weight shuffling.
    return {
        "MoeQuantType": MoeQuantType,
        "aiter_moe": aiter_moe,
        "get_aiter_moe_config": get_aiter_moe_config,
    }


def _w4a8_backend_available(backend: str) -> bool:
    """Compatibility probe used by tests; lightop is deliberately absent."""

    if backend == "triton":
        return True
    try:
        if backend == "aiter":
            _aiter_w4a8_api()
            return True
    except (ImportError, ModuleNotFoundError):
        return False
    except AssertionError:
        return False
    return False


def resolve_kimi_w4a8_backend(requested: str | None = None) -> str:
    """Resolve one backend; explicit requests never silently fall back."""

    requested = kimi_w4a8_requested_backend() if requested is None else requested.strip().lower()
    assert requested in (*W4A8_MOE_BACKENDS, "auto"), f"invalid W4A8 backend {requested!r}"
    assert requested != "lightop", "lightop is unsupported for INT4 W4A8 MoE"
    resolved = "aiter" if requested == "auto" else requested
    assert _w4a8_backend_available(resolved), (
        f"requested Kimi W4A8 backend {resolved!r} is unavailable"
    )
    return resolved


@dataclass(frozen=True)
class KimiK3W4A8Metadata:
    model_version: str
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int

    def __post_init__(self) -> None:
        """Keep every construction path at the same fail-closed boundary."""

        if not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError(
                "Kimi-K3 W4A8 metadata requires a non-empty model_version."
            )

        dimensions = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        }
        for key, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"Kimi-K3 W4A8 metadata requires positive integer {key}."
                )
        if self.top_k > self.num_experts:
            raise ValueError(
                "Kimi-K3 W4A8 metadata requires top_k <= num_experts."
            )
        if self.hidden_size % KIMI_K3_W4A8_GROUP_SIZE or (
            self.intermediate_size % KIMI_K3_W4A8_GROUP_SIZE
        ):
            raise ValueError("Kimi-K3 W4A8 dimensions must be divisible by 32.")


def validate_kimi_k3_w4a8_metadata(
    config: dict[str, Any], text_config: Any | None = None
) -> KimiK3W4A8Metadata:
    """Reject malformed or incompatible INT4 checkpoints before allocation."""

    quant_method = config.get("quant_method")
    is_slimquant = quant_method == SLIMQUANT_W4A8_QUANT_METHOD or (
        quant_method == KIMI_K3_W4A8_QUANT_METHOD
        and config.get("format") == SLIMQUANT_W4A8_QUANT_METHOD
    )
    if is_slimquant:
        if text_config is None:
            dimensions = {
                key: config.get(key)
                for key in (
                    "num_experts",
                    "top_k",
                    "hidden_size",
                    "intermediate_size",
                )
            }
            if all(value is None for value in dimensions.values()):
                dimensions = {
                    "num_experts": 896,
                    "top_k": 16,
                    "hidden_size": 3584,
                    "intermediate_size": 3072,
                }
        else:
            dimensions = {
                "num_experts": getattr(text_config, "num_experts", None),
                "top_k": getattr(text_config, "num_experts_per_token", None),
                "hidden_size": getattr(text_config, "routed_expert_hidden_size", None),
                "intermediate_size": getattr(text_config, "moe_intermediate_size", None),
            }
        if any(
            not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        ):
            raise ValueError("slimquant_w4a8 checkpoint has incomplete Kimi-K3 dimensions.")
        return KimiK3W4A8Metadata(
            model_version=SLIMQUANT_W4A8_QUANT_METHOD,
            **dimensions,
        )

    required = {
        "quant_method": KIMI_K3_W4A8_QUANT_METHOD,
        "format": KIMI_K3_W4A8_FORMAT,
        "weight_bits": 4,
        "activation_bits": 8,
        "group_size": KIMI_K3_W4A8_GROUP_SIZE,
        "symmetric": True,
        "scale_dtype": "float32",
        "packing": KIMI_K3_W4A8_PACKING,
    }
    for key, expected in required.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(
                "Invalid Kimi-K3 W4A8 checkpoint metadata: "
                f"{key}={actual!r}, expected {expected!r}."
            )

    model_version = config.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("Kimi-K3 W4A8 metadata requires a non-empty model_version.")

    dimensions: dict[str, int] = {}
    for key in ("num_experts", "top_k", "hidden_size", "intermediate_size"):
        value = config.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Kimi-K3 W4A8 metadata requires positive integer {key}.")
        dimensions[key] = value

    if dimensions["hidden_size"] % 32 or dimensions["intermediate_size"] % 32:
        raise ValueError("Kimi-K3 W4A8 dimensions must be divisible by 32.")

    if text_config is not None:
        expected = {
            "num_experts": getattr(text_config, "num_experts", None),
            "top_k": getattr(text_config, "num_experts_per_token", None),
            "hidden_size": getattr(text_config, "routed_expert_hidden_size", None),
            "intermediate_size": getattr(text_config, "moe_intermediate_size", None),
        }
        for key, value in expected.items():
            if value is not None and dimensions[key] != value:
                raise ValueError(
                    "Kimi-K3 W4A8 metadata does not match text_config: "
                    f"{key}={dimensions[key]}, model={value}."
                )

    return KimiK3W4A8Metadata(model_version=model_version, **dimensions)


def pack_int4_twos_complement(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 ``[..., K]`` values using the checkpoint byte order."""

    if values.dtype != torch.int8 or values.shape[-1] % 2:
        raise ValueError("INT4 values must be int8 with an even final dimension.")
    if torch.any(values < -8) or torch.any(values > 7):
        raise ValueError("INT4 values must be in [-8, 7].")
    unsigned = values.to(torch.int16) & 0xF
    return ((unsigned[..., 0::2] << 4) | unsigned[..., 1::2]).to(torch.uint8)


def unpack_int4_twos_complement(packed: torch.Tensor) -> torch.Tensor:
    """Invert :func:`pack_int4_twos_complement` to signed INT8 values."""

    if packed.dtype not in (torch.uint8, torch.int8):
        raise ValueError("Packed INT4 weights must have uint8 or int8 dtype.")
    unsigned = packed.to(torch.uint8)
    high = (unsigned >> 4).to(torch.int16)
    low = (unsigned & 0xF).to(torch.int16)
    values = torch.stack((high, low), dim=-1).reshape(*packed.shape[:-1], -1)
    return torch.where(values >= 8, values - 16, values).to(torch.int8)


def repack_and_shuffle_w4a8(
    weight: torch.Tensor, shuffle_fn: Any | None = None
) -> torch.Tensor:
    """Reformat checkpoint packed INT4 into the AITER W4A8 moe_c layout.

    Mirrors sglang's ``repack_and_shuffle_w4a8`` exactly: each expert's packed
    bytes are unpacked into two's-complement nibbles, regrouped eight at a time
    so the first/last four nibbles become the low/high nibbles of the new
    bytes, then every expert is permuted by AITER's
    ``w4a8_moe_layout_shuffle_gemm2``.  The two's-complement nibble encoding
    itself is untouched (unlike the Triton debug path, which swaps nibbles and
    xors ``0x88``).  The result keeps the input dtype, is a one-shot conversion
    and must be cached at weight-load time, never per forward.
    """

    if weight.ndim != 3 or weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError("W4A8 repack expects [E, N, K/2] int8/uint8 weights.")
    if shuffle_fn is None:
        from aiter.ops.shuffle import (
            w4a8_moe_layout_shuffle_gemm2 as shuffle_fn,
        )

    experts, rows, k_half = weight.shape
    packed_u8 = weight.to(torch.uint8)
    # One byte -> [even_k, odd_k], then regroup eight nibbles per output byte:
    # new byte_j = (nibble_{2j} << 4) | nibble_{2j+4}.
    nibbles = torch.stack(
        ((packed_u8 >> 4) & 0x0F, packed_u8 & 0x0F), dim=-1
    ).view(experts, rows, -1)
    blocks = nibbles.view(experts, rows, -1, 8)
    repacked = ((blocks[..., :4] << 4) | blocks[..., 4:]).view(
        experts, rows, k_half
    )
    shuffled = torch.stack(
        [shuffle_fn(repacked[i]) for i in range(experts)]
    ).reshape(experts, rows, k_half)
    return shuffled.contiguous().to(weight.dtype)


def quantize_int4_groupwise(
    weight: torch.Tensor, group_size: int = KIMI_K3_W4A8_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., N, K]`` weights to packed INT4 plus FP32 group scales."""

    if weight.ndim < 2 or weight.shape[-1] % group_size:
        raise ValueError(f"Weight K dimension must be divisible by group_size={group_size}.")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, group_size)
    scale = grouped.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).eps) / 7.0
    quantized = torch.round(grouped / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    return pack_int4_twos_complement(quantized.reshape_as(weight)), scale.float()


def dequantize_int4_groupwise(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = KIMI_K3_W4A8_GROUP_SIZE,
) -> torch.Tensor:
    values = unpack_int4_twos_complement(packed).float()
    if values.shape[-1] % group_size:
        raise ValueError("Packed INT4 K dimension is not aligned to group_size.")
    expected = (*values.shape[:-1], values.shape[-1] // group_size)
    if tuple(scales.shape) != expected:
        raise ValueError(f"INT4 scale shape {tuple(scales.shape)} does not match {expected}.")
    logical_shape = values.shape
    values = values.reshape(*values.shape[:-1], -1, group_size)
    return (values * scales.float().unsqueeze(-1)).reshape(logical_shape)


def quantize_int4_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., N, K]`` weights to slimquant's per-channel layout."""

    if weight.ndim < 2 or weight.shape[-1] % 2:
        raise ValueError("Per-channel INT4 weights must have an even K dimension.")
    scale = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).eps
    ) / 7.0
    quantized = torch.round(weight.float() / scale).clamp(-8, 7).to(torch.int8)
    return pack_int4_twos_complement(quantized), scale.float()


def dequantize_int4_per_channel(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize slimquant's packed INT4 weights and per-channel scales."""

    values = unpack_int4_twos_complement(packed).float()
    if tuple(scales.shape) != (*values.shape[:-1], 1):
        raise ValueError(
            "Per-channel INT4 scale shape must match the weight output channels."
        )
    return values * scales.float()


def quantize_activations_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric dynamic INT8 quantization with a scale for each token."""

    scale = x.float().abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).eps) / 127.0
    return torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8), scale


def w4a8_reference_gemm(
    x: torch.Tensor, packed_weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """Numerical reference for one expert GEMM; used only by unit tests."""

    if weight_scale.shape[-1] == 1:
        weight = dequantize_int4_per_channel(packed_weight, weight_scale)
    else:
        weight = dequantize_int4_groupwise(packed_weight, weight_scale)
    return x.float() @ weight.t()


def _require_gfx938() -> None:
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx938

    if not current_platform.is_rocm() or not on_gfx938():
        raise RuntimeError(
            "Kimi-K3 W4A8 requires the gfx938 Triton INT8 kernel; "
            "no BF16/reference fallback is enabled."
        )


_rank_local_cache_configured = False


def _use_rank_local_triton_cache() -> None:
    """Point Triton's cache at a per-rank subdir, exactly once per process.

    Triton's knob setter (``env_base.__set__``) writes the assigned value
    back into the ``TRITON_CACHE_DIR`` environment variable, so re-reading
    the env and re-assigning on every GEMM call keeps appending ``rank_<r>``
    and eventually blows past PATH_MAX (observed as nested
    ``rank_15/rank_15/...`` + ``OSError: [Errno 36] File name too long``).
    Setting it once keeps the per-rank isolation (no cross-rank HIP binary
    races during first-use JIT) without the runaway path growth.
    """

    global _rank_local_cache_configured
    if _rank_local_cache_configured:
        return
    base = os.environ.get("TRITON_CACHE_DIR")
    if not base:
        return
    rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if rank < 0 and torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank < 0:
        return
    import triton.runtime.cache as triton_cache

    triton_cache.knobs.cache.dir = os.path.join(base, f"rank_{rank}")
    _rank_local_cache_configured = True


_w4a8_gemm_kernel: Any | None = None


def _get_w4a8_gemm_kernel():
    """Define the HIP Triton kernel once and return the cached object.

    Re-defining the ``@triton.jit`` function on every expert GEMM call would
    create a fresh JIT wrapper each time; Triton dedups by source hash, but
    caching the function object mirrors how the production aiter/
    ``matmul_ogs`` W4A8 kernels are structured and avoids needless work.
    """

    global _w4a8_gemm_kernel
    if _w4a8_gemm_kernel is not None:
        return _w4a8_gemm_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        xq, x_scale, weight, weight_scale, out,
        stride_xm: tl.constexpr, stride_wn: tl.constexpr,
        stride_ws: tl.constexpr, stride_om: tl.constexpr,
        stride_on: tl.constexpr, M, N,
        K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for group in range(0, K // 32):
            offs_k = group * 32 + tl.arange(0, 32)
            x = tl.load(
                xq + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=offs_m[:, None] < M,
                other=0,
            ).to(tl.int8)
            byte = tl.load(
                weight
                + offs_n[None, :] * stride_wn
                + (offs_k // 2)[:, None],
                mask=offs_n[None, :] < N,
                other=0,
            ).to(tl.uint8)
            nibble = tl.where((offs_k[:, None] & 1) == 0, byte >> 4, byte & 0xF)
            qweight = tl.where(
                nibble >= 8, nibble.to(tl.int16) - 16, nibble.to(tl.int16)
            ).to(tl.int8)
            dot = tl.dot(x, qweight)
            ws = tl.load(weight_scale + offs_n * stride_ws, mask=offs_n < N, other=0.0)
            xs = tl.load(x_scale + offs_m, mask=offs_m < M, other=0.0)
            acc += dot.to(tl.float32) * xs[:, None] * ws[None, :]
        tl.store(
            out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    _w4a8_gemm_kernel = _kernel
    return _kernel


def w4a8_gfx938_gemm(
    x: torch.Tensor, packed_weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """Run a gfx938 INT8 x INT4 GEMM with FP32 accumulation and rescaling."""

    _require_gfx938()
    if x.ndim != 2 or packed_weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("W4A8 GEMM expects x[M,K], weight[N,K/2], scale[N,1].")
    m, k = x.shape
    n, packed_k = packed_weight.shape
    if k % 32 or packed_k * 2 != k or tuple(weight_scale.shape) != (n, 1):
        raise ValueError("Invalid W4A8 GEMM shapes or per-channel scale layout.")
    if not (x.is_cuda and packed_weight.is_cuda and weight_scale.is_cuda):
        raise ValueError("W4A8 GEMM tensors must be device tensors.")
    xq, x_scale = quantize_activations_per_token(x)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _use_rank_local_triton_cache()
    kernel = _get_w4a8_gemm_kernel()
    kernel[((m + 15) // 16, (n + 31) // 32)](
        xq, x_scale, packed_weight, weight_scale, out,
        xq.stride(0), packed_weight.stride(0), weight_scale.stride(0),
        out.stride(0), out.stride(1), M=m, N=n, K=k,
        BLOCK_M=16, BLOCK_N=32, num_warps=4,
    )
    return out


def _situ(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def run_kimi_k3_w4a8_moe(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
    w13_scale: torch.Tensor, w2_scale: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, *,
    situ_beta: float, situ_linear_beta: float | None,
    backend: str | None = None,
    backend_context: Any | None = None,
) -> torch.Tensor:
    """Direct routed-MoE call used by M2 tests and the initial module path."""

    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise AssertionError("topk_ids and topk_weights must have matching [tokens, top_k] shapes.")
    resolved = resolve_kimi_w4a8_backend(backend)
    if resolved == "aiter":
        return run_aiter_w4a8_moe(
            x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids,
            situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
            backend_context=backend_context,
        )
    if topk_ids.numel() and (topk_ids.min() < 0 or topk_ids.max() >= w13.shape[0]):
        raise AssertionError("Triton W4A8 route contains a non-local expert id.")
    out = torch.zeros((x.shape[0], w2.shape[1]), device=x.device, dtype=torch.float32)
    for expert_id in torch.unique(topk_ids).tolist():
        token_ids, route_slots = (topk_ids == expert_id).nonzero(as_tuple=True)
        h = w4a8_gfx938_gemm(x[token_ids], w13[expert_id], w13_scale[expert_id])
        h = _situ(h, situ_beta, situ_linear_beta)
        y = w4a8_gfx938_gemm(h, w2[expert_id], w2_scale[expert_id])
        out.index_add_(0, token_ids, y * topk_weights[token_ids, route_slots].float().unsqueeze(1))
    return out.to(x.dtype)


def _aiter_w4a8_config(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    backend_context: dict[str, Any],
) -> Any:
    """Get a checked AITER W4A8 configuration, cached by logical MoE shape."""

    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert w13.ndim == w2.ndim == 3 and w13.is_contiguous() and w2.is_contiguous()
    m, k = x.shape
    experts, n1, packed_k = w13.shape
    assert w2.shape[0] == experts and w2.shape[1] == k
    assert packed_k * 2 == k and w2.shape[2] * 2 == n1 // 2, (
        "invalid packed INT4 Kimi W4A8 expert weight layout"
    )
    top_k = topk_ids.shape[1]
    key = (m, experts, n1, w2.shape[1], k, top_k, x.dtype, activation)
    cache = backend_context.setdefault("config_cache", {})
    if key in cache:
        return cache[key]

    api = backend_context.get("api") or _aiter_w4a8_api()
    backend_context["api"] = api
    kwargs = dict(
        M=m, E=experts, N1=n1, N2=w2.shape[1], K=k, top_k=top_k,
        # slimquant's K3 W4A8 scales are per output channel, not block scales.
        block_size=0, dtype=x.dtype,
        quant_type=api["MoeQuantType"].W4A8, activation=activation,
    )
    try:
        status, config = api["get_aiter_moe_config"](**kwargs)
    except TypeError as error:
        raise AssertionError(
            "installed AITER get_aiter_moe_config does not support W4A8 activation"
        ) from error
    assert status and config is not None, (
        "AITER has no supported W4A8 MoE config for "
        f"M={m}, E={experts}, N1={n1}, N2={w2.shape[1]}, K={k}, top_k={top_k}"
    )
    assert getattr(config, "solution_type", None) is not None, (
        "AITER returned a successful W4A8 config without a solution type"
    )
    assert getattr(config, "quant_type", api["MoeQuantType"].W4A8) == api["MoeQuantType"].W4A8, (
        "AITER returned a non-W4A8 MoE config"
    )
    cache[key] = config
    return config


def run_aiter_w4a8_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    backend_context: Any | None,
) -> torch.Tensor:
    """Execute Kimi routed experts through AITER's native W4A8 MoE API."""

    context = backend_context if isinstance(backend_context, dict) else {}
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.is_cuda or context.get("allow_cpu_mock", False), (
        "AITER W4A8 requires CUDA/ROCm device tensors"
    )
    assert w13.dtype == w2.dtype == torch.int8
    assert w13_scale.dtype == w2_scale.dtype == torch.float32
    assert w13_scale.shape == (*w13.shape[:2], 1)
    assert w2_scale.shape == (*w2.shape[:2], 1)
    assert topk_ids.shape == topk_weights.shape and topk_ids.ndim == 2
    assert not context.get("no_combine", False), "AITER W4A8 requires no_combine=False"
    layer = context.get("layer")
    assert not getattr(layer, "apply_router_weight_on_input", False), (
        "AITER W4A8 does not support apply_router_weight_on_input=True"
    )
    assert layer is None or getattr(layer, "_kimi_w4a8_aiter_repacked", False), (
        "AITER W4A8 requires routed weights repacked by "
        "process_weights_after_loading before the first forward"
    )
    # The weights passed here are already in the AITER moe_c layout:
    # process_weights_after_loading() applied the one-shot 8-nibble regroup +
    # gemm2 shuffle when aiter was resolved, so no per-forward shuffle runs.
    config = _aiter_w4a8_config(x, w13, w2, topk_ids, activation="situ", backend_context=context)
    api = context["api"]
    expert_map = getattr(layer, "expert_map", None) if layer is not None else context.get("expert_map")
    global_num_experts = (
        getattr(layer, "global_num_experts", w13.shape[0]) if expert_map is not None else w13.shape[0]
    )
    output = api["aiter_moe"](
        hidden_states=x,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int32),
        moe_config=config,
        inplace=False,
        activation="situ",
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        routed_scaling_factor=float(getattr(layer, "routed_scaling_factor", 1.0) or 1.0),
        output_dtype=x.dtype,
        gemm1_alpha=float(situ_beta),
        gemm1_limit=situ_linear_beta,
    )
    assert isinstance(output, torch.Tensor)
    assert output.shape == (x.shape[0], w2.shape[1]) and output.dtype == x.dtype
    return output


class KimiK3W4A8Config(QuantizationConfig):
    """Explicit Kimi-K3 W4A8 method; only routed experts are quantized."""

    def __init__(self, metadata: KimiK3W4A8Metadata | None = None):
        super().__init__()
        self.metadata = metadata or KimiK3W4A8Metadata(
            **KIMI_K3_W4A8_DEFAULT_METADATA
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return KIMI_K3_W4A8_QUANT_METHOD  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KimiK3W4A8Config":
        return cls(validate_kimi_k3_w4a8_metadata(config))

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if not isinstance(hf_quant_cfg, dict):
            return None
        if hf_quant_cfg.get("quant_method") not in (
            KIMI_K3_W4A8_QUANT_METHOD,
            SLIMQUANT_W4A8_QUANT_METHOD,
        ):
            return None
        if getattr(hf_config, "model_type", None) != "kimi_k3":
            return None
        validate_kimi_k3_w4a8_metadata(hf_quant_cfg, getattr(hf_config, "text_config", None))
        return KIMI_K3_W4A8_QUANT_METHOD

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return KimiK3W4A8MoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


class KimiK3W4A8MoEMethod(FusedMoEMethodBase):
    """Routed-expert storage and direct gfx938 W4A8 dispatch for Kimi-K3."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        # vLLM 0.25.x has no ``situ`` enum; the HCU legacy-MoE adapter carries
        # the K3 beta through ``swiglu_beta`` and restores the semantic value
        # on the completed config immediately after factory construction.
        if moe.activation.value != "situ" and moe.swiglu_beta is None:
            raise ValueError("Kimi-K3 W4A8 only supports SiTU routed experts.")
        parallel = getattr(moe, "moe_parallel_config", None)
        self.use_deepep_ht = bool(getattr(parallel, "use_deepep_ht_kernels", False))
        self.use_deepep_ll = bool(getattr(parallel, "use_deepep_ll_kernels", False))
        if self.use_deepep_ht and self.use_deepep_ll:
            raise ValueError("Kimi-K3 cannot enable DeepEP HT and LL simultaneously")
        self._hcu_kimi_ll_graph_boundary = self.use_deepep_ll

    def create_weights(self, layer: RoutedExperts, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if hidden_size % 32 or intermediate_size_per_partition % 32:
            raise ValueError("Kimi-K3 W4A8 requires TP-local dimensions divisible by 32.")
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        specs = (
            (
                "w13_weight",
                (num_experts, 2 * intermediate_size_per_partition, hidden_size // 2),
                torch.int8,
            ),
            (
                "w2_weight",
                (num_experts, hidden_size, intermediate_size_per_partition // 2),
                torch.int8,
            ),
            (
                "w13_weight_scale",
                (num_experts, 2 * intermediate_size_per_partition, 1),
                torch.float32,
            ),
            (
                "w2_weight_scale",
                (num_experts, hidden_size, 1),
                torch.float32,
            ),
        )
        for name, shape, dtype in specs:
            parameter = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Cache AITER state; repack routed weights once for the AITER layout.

        AITER W4A8 consumes the checkpoint packed bytes only after the
        8-nibble regroup + per-expert gemm2 shuffle (mirrors sglang's
        ``repack_and_shuffle_w4a8``).  The conversion is one-shot and
        destructive, so it happens here, only when aiter is the resolved
        backend.  The checkpoint's native scales are left untouched: AITER
        applies the missing x16 internally, and the Triton debug path rescales
        with x16 at its own call site.
        """

        if self.use_deepep_ht or self.use_deepep_ll:
            # Preserve canonical packed parameters for the loader. The HT
            # expert owns the one-shot HIPC cache, bound by select_gemm_impl.
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            return

        if hasattr(layer, "w13_weight") and not getattr(
            layer, "_kimi_w4a8_aiter_repacked", False
        ):
            resolved = resolve_kimi_w4a8_backend()
            if resolved == "aiter":
                layer.w13_weight.data.copy_(
                    repack_and_shuffle_w4a8(layer.w13_weight.data)
                )
                layer.w2_weight.data.copy_(
                    repack_and_shuffle_w4a8(layer.w2_weight.data)
                )
                layer._kimi_w4a8_aiter_repacked = True
        layer._kimi_w4a8_backend_context = {
            "layer": layer,
            "config_cache": {},
        }

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig | None:
        if self.use_deepep_ht or self.use_deepep_ll:
            return FusedMoEQuantConfig.make(
                torch.int8, weight_dtype="int4", per_act_token_quant=True,
                per_out_ch_quant=False,
                block_shape=[256, 256] if self.use_deepep_ll else None,
                w1_scale=(
                    layer.w13_weight_scale
                    if self.use_deepep_ll
                    else layer.w13_weight_scale * 16.0
                ),
                w2_scale=(
                    layer.w2_weight_scale
                    if self.use_deepep_ll
                    else layer.w2_weight_scale * 16.0
                ),
                a1_scale=None, a2_scale=None,
            )
        return None

    def select_gemm_impl(self, prepare_finalize, layer: RoutedExperts):
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEActivationFormat,
        )
        expected_format = (
            FusedMoEActivationFormat.BatchedExperts
            if self.use_deepep_ll
            else FusedMoEActivationFormat.Standard
        )
        if not (self.use_deepep_ht or self.use_deepep_ll) or (
            prepare_finalize.activation_format != expected_format
        ):
            raise ValueError("Kimi-K3 DeepEP selected an incompatible expert format")
        if self.moe_quant_config is None:
            raise RuntimeError("Kimi-K3 HT requires quant config before expert selection")
        if self.use_deepep_ll:
            from .kimi_k3_ll_runtime import KimiK3LLExperts

            experts = KimiK3LLExperts(
                self.moe,
                self.moe_quant_config,
                prepare_finalize.max_num_tokens_per_rank(),
                prepare_finalize.num_dispatchers(),
            )
            experts.process_weights_after_loading(layer)
            # Explicit LL selection owns a fixed-layout, LL-only buffer: no HT
            # dispatch can dirty it between calls. Keep initial cleanup but
            # avoid resetting it before every layer on every decode step.
            prepare_finalize._vllm_hcu_clean_low_latency_buffer = False
            prepare_finalize._hcu_ll_cleaned_buffer_layout = None
            return experts

        from .kimi_k3_ht_runtime import KimiK3HTExperts

        experts = KimiK3HTExperts(self.moe, self.moe_quant_config)
        experts.process_weights_after_loading(layer)
        return experts

    def apply(self, layer: RoutedExperts, x: torch.Tensor,
              topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts, shared_experts_input: torch.Tensor | None) -> torch.Tensor:
        if self.use_deepep_ht or self.use_deepep_ll:
            raise RuntimeError("Kimi-K3 DeepEP requires the modular DeepGEMM expert path")
        beta = layer.moe_config.activation_situ_beta
        if beta is None:
            raise ValueError("Kimi-K3 W4A8 requires activation_situ_beta.")
        requested = kimi_w4a8_requested_backend()
        resolved = resolve_kimi_w4a8_backend(requested)
        logger.info_once(
            "[kimi_k3_w4a8] requested_backend=%s resolved_backend=%s kernel=%s",
            requested, resolved,
            "w4a8_gfx938_gemm" if resolved == "triton" else "aiter.moe.aiter_moe",
        )
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        if resolved == "triton":
            w13_scale = w13_scale * 16.0
            w2_scale = w2_scale * 16.0
        return run_kimi_k3_w4a8_moe(
            x, layer.w13_weight, layer.w2_weight, w13_scale,
            w2_scale, topk_weights, topk_ids, situ_beta=beta,
            situ_linear_beta=layer.moe_config.activation_situ_linear_beta,
            backend=resolved,
            backend_context=getattr(layer, "_kimi_w4a8_backend_context", None),
        )
