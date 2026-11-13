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


@GemmaRMSNorm.register_oot
class HcuGemmaRMSNorm(GemmaRMSNorm):
    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if henvs.optional_custom_op_enabled(
            henvs.VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM
        ):
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
