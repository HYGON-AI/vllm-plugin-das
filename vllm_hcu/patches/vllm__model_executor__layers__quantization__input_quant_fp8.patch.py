# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm/model_executor/layers/quantization/input_quant_fp8.py.
"""

PATCHES = [
(
"""
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
""",
"""
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
from vllm.utils.torch_utils import direct_register_custom_op
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


# --8<-- [start:quant_fp8]
""",
"""
_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


def _lightop_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from lightop import op

    out = torch.empty_like(x, dtype=_FP8_DTYPE)
    scale = torch.empty(
        (*x.shape[:-1], 1),
        device=x.device,
        dtype=torch.float32,
    )
    op.per_token_quant_fp8(out, x, scale)
    return out, scale


def _lightop_per_token_quant_fp8_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x, dtype=_FP8_DTYPE)
    scale = torch.empty(
        (*x.shape[:-1], 1),
        device=x.device,
        dtype=torch.float32,
    )
    return out, scale


direct_register_custom_op(
    op_name="lightop_per_token_quant_fp8",
    op_func=_lightop_per_token_quant_fp8,
    mutates_args=[],
    fake_impl=_lightop_per_token_quant_fp8_fake,
)


# --8<-- [start:quant_fp8]
""",
),

(
"""
        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            group_shape=(self.group_shape.row, self.group_shape.col)
            if self.static
            else None,
        )
""",
"""
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8
            and scale is None
            and scale_ub is None
            and self.group_shape == GroupShape.PER_TOKEN
            and self.num_token_padding is None
            and x.is_contiguous()
        ):
            return torch.ops.vllm.lightop_per_token_quant_fp8(x)

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            group_shape=(self.group_shape.row, self.group_shape.col)
            if self.static
            else None,
        )
""",
),

(
"""
        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
""",
"""
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8
            and scale is None
            and scale_ub is None
            and self.group_shape == GroupShape.PER_TOKEN
            and self.num_token_padding is None
            and x.is_contiguous()
        ):
            return torch.ops.vllm.lightop_per_token_quant_fp8(x)

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
""",
),
]
