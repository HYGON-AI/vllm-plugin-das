# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
import vllm_hcu.platforms.envs as henvs

logger = init_logger(__name__)

try:
    from lightop.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm
except (ImportError, AttributeError):
    from lightop.op import gemma_fused_add_rmsnorm

    gemma_rmsnorm = None
    logger.warning_once(
        "Using deprecated lightop.op gemma_fused_add_rmsnorm API because "
        "lightop.norm is unavailable; upgrade LightOp."
    )


@GemmaRMSNorm.register_oot
class HcuGemmaRMSNorm(GemmaRMSNorm):
    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM:
            if residual is None:
                out = x.clone()
                if gemma_rmsnorm is None:
                    raise RuntimeError(
                        "lightop.norm.gemma_rmsnorm is required because its ABI differs "
                        "from lightop.op.gemma_rmsnorm; upgrade LightOp"
                    )
                gemma_rmsnorm(x, self.weight, self.variance_epsilon, out=out)
                return out
            else:
                out = x.clone()
                gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
                return x, residual
        else:
            if torch.compiler.is_compiling():
                return self.forward_native(x, residual)
            if not getattr(self, "_is_compiled", False):
                self._forward_static_no_residual = torch.compile(  # type: ignore
                    self._forward_static_no_residual
                )
                self._forward_static_with_residual = torch.compile(  # type: ignore
                    self._forward_static_with_residual
                )
                self._is_compiled = True
            return self.forward_native(x, residual)
