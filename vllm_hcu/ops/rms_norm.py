import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from lightop import fused_add_rms_norm
from lightop.op import rmsnorm_forward_autograd
import vllm_hcu.platforms.envs as henvs
    
@RMSNorm.register_oot
class HcuRMSNorm(RMSNorm):
    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if  henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_CUSTOM_RMS_NORM:
            if residual is None:
                out = rmsnorm_forward_autograd(x, self.weight,self.variance_epsilon, self.training)
                return out
            else:
                fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
                return x, residual
        else:
            return self.forward_cuda(x, residual)
