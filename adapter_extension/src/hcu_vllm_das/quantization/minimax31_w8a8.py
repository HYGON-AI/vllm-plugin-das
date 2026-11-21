"""MiniMax-only W8A8 hipBLASLt kernel over the existing KN IPC layout."""
import json
import os
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel


class MiniMax31HipBlasLtInt8Kernel(TritonInt8ScaledMMLinearKernel):
    def __init__(self,c,layer_param_names,*,fallback="error"):
        super().__init__(c,layer_param_names)
        if fallback not in ("error","triton"):
            raise ValueError("unknown MiniMax W8A8 fallback")
        self.fallback=fallback
        self._fallback_kernel=TritonInt8ScaledMMLinearKernel(c,layer_param_names)
        self._logged=set()

    @classmethod
    def is_supported(cls,compute_capability=None):
        return (True,None) if torch.version.hip is not None else (False,"MiniMax hipBLASLt requires ROCm")

    @classmethod
    def can_implement(cls,c):
        valid=(not c.is_static_input_scheme and c.is_channelwise and c.input_symmetric)
        return valid,None if valid else "requires dynamic symmetric per-token/per-channel INT8"

    def _log(self,event,**details):
        if event not in self._logged:
            print("MINIMAX_W8A8_BACKEND "+json.dumps({"event":event,"backend":"hipblaslt_v1",**details}),flush=True)
            self._logged.add(event)

    def _unsupported(self,layer,x,bias,reason):
        if self.fallback=="triton":
            self._log("fallback",selected="triton",reason=reason,shape=list(x.shape))
            return self._fallback_kernel.apply_weights(layer,x,bias)
        raise ValueError("MiniMax hipBLASLt unsupported input: "+reason)

    def apply_weights(self,layer,x,bias=None):
        w,ws,ins,izp,azp=self._get_layer_params(layer)
        if ins is not None or izp is not None or azp is not None:
            return self._unsupported(layer,x,bias,"only dynamic symmetric inputs")
        if x.ndim<2 or x.dtype!=torch.bfloat16 or not x.is_cuda:
            return self._unsupported(layer,x,bias,"requires BF16 HIP [...,K]")
        k,n=w.shape if w.ndim==2 else (0,0)
        if (w.dtype!=torch.int8 or w.stride()!=(1,k) or x.shape[-1]!=k
                or k%16 or n%16 or k<=0 or n<=0):
            return self._unsupported(layer,x,bias,"requires KN stride(1,K), aligned K/N")
        if (ws.dtype!=torch.float32 or ws.shape!=(n,1) or not ws.is_contiguous()
                or w.device!=x.device or ws.device!=x.device):
            return self._unsupported(layer,x,bias,"requires FP32 channel [N,1] on one device")
        if bias is not None and (bias.shape!=(n,) or bias.dtype!=x.dtype
                                  or bias.device!=x.device or not bias.is_contiguous()):
            return self._unsupported(layer,x,bias,"bias must be contiguous output dtype [N]")
        flat=x.reshape(-1,k).contiguous()
        m=flat.shape[0]
        if m==0:
            return torch.empty((*x.shape[:-1],n),dtype=x.dtype,device=x.device)
        if max(m,k,n,m*k,m*n,k*n)>=2**31:
            return self._unsupported(layer,x,bias,"native INT32 dimension/address bound")
        # Retain the stock vLLM quantizer, including its rounding and zero-row
        # semantics; only the scaled INT8 matrix multiplication changes.
        q,qs,zp=ops.scaled_int8_quant(flat,None,None,symmetric=True)
        if (q.shape!=(m,k) or q.dtype!=torch.int8 or not q.is_contiguous()
                or qs.shape!=(m,1) or qs.dtype!=torch.float32 or not qs.is_contiguous()
                or zp is not None):
            raise RuntimeError("unexpected vLLM dynamic symmetric INT8 quant ABI")
        # Existing processed weight is KN stride(1,K). Its transpose is the
        # original contiguous NK storage, a view with no copy/repack.
        b=w.t()
        if not b.is_contiguous() or b.data_ptr()!=w.data_ptr():
            raise RuntimeError("hipBLASLt weight view unexpectedly copied")
        from lightop._lmslim_native.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm
        ok,out=hipblaslt_w8a8_channelwise_gemm(q,b,qs,ws,m,n,k,"NT",x.dtype,bias)
        # Target native channelwise API retains a batch=1 dimension.
        # Accept only its exact 2D/one-batch contiguous forms, never transpose.
        if (not ok or out.shape not in ((m,n),(1,m,n)) or not out.is_contiguous()
                or out.dtype!=x.dtype or out.device!=x.device):
            raise RuntimeError("unexpected hipBLASLt W8A8 return ABI: "+str(
                (ok,out.shape,out.dtype,out.device,out.stride())))
        out=out.view(m,n)
        self._log("native",api="torch.ops.lmslim.hipblaslt_gemm_channelwise",
                  source="lightop._lmslim_native.quantize.quant_ops",
                  m=m,n=n,k=k,weight_stride=list(w.stride()),zero_copy_weight_view=True,
                  quantizer="vllm.scaled_int8_quant",bias=bias is not None,
                  tuning_override=os.environ.get("HIPBLASLT_TUNING_OVERRIDE_FILE"))
        return out.view(*x.shape[:-1],n)
