import os
import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
import lightop.op as op
import vllm_hcu.platforms.envs as henvs
    
@GemmaRMSNorm.register_oot
class HcuGemmaRMSNorm(GemmaRMSNorm):
    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if  henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM:
            if residual is None:
                out = x.clone()
                op.gemma_rmsnorm(out, x, self.weight, self.variance_epsilon)
                return out
            else:
                out = x.clone()
                op.gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
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
