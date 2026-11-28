# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.utils.torch_utils import direct_register_custom_op
import vllm_hcu.platforms.envs as henvs


def _hcu_gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from lightop.norm import gemma_rmsnorm

    out = x.clone()
    gemma_rmsnorm(x, weight, variance_epsilon, out=out)
    return out


def _hcu_gemma_rms_norm_fake(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    return torch.empty_like(x)


def _hcu_gemma_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> None:
    from lightop.norm import gemma_fused_add_rmsnorm

    gemma_fused_add_rmsnorm(x, residual, weight, variance_epsilon)


def _hcu_gemma_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> None:
    return None


# Keep the LightOp bindings opaque to Dynamo. Declare both writes so
# functionalization can preserve the fused residual update.
direct_register_custom_op(
    op_name="hcu_gemma_rms_norm",
    op_func=_hcu_gemma_rms_norm,
    fake_impl=_hcu_gemma_rms_norm_fake,
)
direct_register_custom_op(
    op_name="hcu_gemma_fused_add_rms_norm",
    op_func=_hcu_gemma_fused_add_rms_norm,
    mutates_args=["x", "residual"],
    fake_impl=_hcu_gemma_fused_add_rms_norm_fake,
)


def _step5_gemma_rms_norm_impl(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float,
) -> torch.Tensor:
    flat = x.view(-1, x.shape[-1])
    if x.dtype == weight.dtype:
        from lightop.norm import gemma_rmsnorm

        out = torch.empty_like(flat)
        gemma_rmsnorm(flat, weight, variance_epsilon, out=out)
    else:
        # LightOp's Gemma kernel requires matching input/weight dtypes.
        # Keep gamma in FP32 for the mixed Q/K and MTP cases. AITER's
        # ordinary RMSNorm takes the effective Gemma weight (1 + gamma).
        from aiter.ops.triton.rmsnorm import rms_norm

        out = rms_norm(flat, weight.float() + 1.0, variance_epsilon)
    return out.view_as(x)


def _step5_gemma_rms_norm_fake(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="hcu_step5_gemma_rms_norm",
    op_func=_step5_gemma_rms_norm_impl,
    fake_impl=_step5_gemma_rms_norm_fake,
)


@GemmaRMSNorm.register_oot
class HcuGemmaRMSNorm(GemmaRMSNorm):
    def __init__(self, *args, **kwargs):
        from vllm_hcu.runtime_compat.step5 import is_step5_owner

        self._hcu_step5_owner = is_step5_owner()
        super().__init__(*args, **kwargs)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if henvs.optional_custom_op_enabled(
            henvs.VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM
        ):
            if getattr(self, "_hcu_step5_owner", False):
                if (residual is None and x.is_cuda and x.is_contiguous()
                        and x.shape[-1] in (192, 4096)
                        and x.dtype in (torch.bfloat16, torch.float32)
                        and self.weight.dtype in (torch.bfloat16, torch.float32)):
                    return torch.ops.vllm.hcu_step5_gemma_rms_norm(
                        x, self.weight, self.variance_epsilon)
                return self.forward_native(x, residual)
            if residual is None:
                return torch.ops.vllm.hcu_gemma_rms_norm(
                    x, self.weight, self.variance_epsilon
                )
            torch.ops.vllm.hcu_gemma_fused_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon
            )
            return x, residual
        else:
            # vLLM 0.28.1 implements the portable fallback through IR ops in
            # ``forward_native``.  The private ``_forward_static_*`` helpers
            # used by older releases no longer exist, so delegating is both
            # the current ABI and the master-switch fallback path.
            return self.forward_native(x, residual)
