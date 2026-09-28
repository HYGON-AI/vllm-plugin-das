# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
import vllm_hcu.platforms.envs as henvs


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
            from lightop.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

            if residual is None:
                out = x.clone()
                gemma_rmsnorm(x, self.weight, self.variance_epsilon, out=out)
                return out
            else:
                gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
                return x, residual
        else:
            # vLLM 0.28.1 implements the portable fallback through IR ops in
            # ``forward_native``.  The private ``_forward_static_*`` helpers
            # used by older releases no longer exist, so delegating is both
            # the current ABI and the master-switch fallback path.
            return self.forward_native(x, residual)
