"""Opt-in benchmark operator binding. The checkpoint retains its original INT8 layout."""
import json

import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm


@triton.jit
def _rescale(raw, scale_a, scale_b, out, rows:tl.constexpr, cols:tl.constexpr, width:tl.constexpr):
    index=tl.program_id(0)*width+tl.arange(0,width)
    mask=index<rows*cols
    v=tl.load(raw+index,mask=mask,other=0)
    a=tl.load(scale_a+index//cols,mask=mask,other=0)
    b=tl.load(scale_b+index%cols)
    tl.store(out+index, (v.to(tl.float32)*a)*b,mask=mask)

class DraftInt8Tiles(TritonInt8ScaledMMLinearKernel):
    def __init__(self, original, tile):
        super().__init__(original.config, original.layer_param_names)
        self.original = original
        self.tile = tile

    def apply_weights(self, layer, x, bias=None):
        if x.ndim != 2 or x.shape[0] != 112 or x.dtype != torch.bfloat16 or bias is not None:
            return self.original.apply_weights(layer,x,bias)
        weight, scale, input_scale, zero, correction = self._get_layer_params(layer)
        if any(t is not None for t in (input_scale,zero,correction)):
            return self.original.apply_weights(layer,x,bias)
        q, row_scale, zp = ops.scaled_int8_quant(x.contiguous(),None,None,symmetric=True)
        if zp is not None:
            raise RuntimeError("unexpected D-Spark INT8 zero point")
        if self.tile == "raw":
            accum = torch._int_mm(q, weight)
            rows, cols = accum.shape
            result = torch.empty_like(accum, dtype=x.dtype)
            _rescale[(triton.cdiv(rows * cols, 1024),)](
                accum, row_scale, scale, result, rows, cols, 1024
            )
            return result
        bm,bn,bk,desc = self.tile
        return triton_scaled_mm(q,weight,scale_a=row_scale,scale_b=scale,out_dtype=x.dtype,
                               block_size_m=bm,block_size_n=bn,block_size_k=bk,
                               use_heuristic=False,use_td=bool(desc))


def install(draft, config):
    config=json.loads(config)
    touched=[]
    for name,module in draft.named_modules():
        scheme=getattr(module,"scheme",None)
        kernel=getattr(scheme,"kernel",None)
        if type(kernel) is not TritonInt8ScaledMMLinearKernel:
            continue
        weight=module.weight
        tile=config.get(str(weight.size(0)))
        if (tile and weight.shape in ((8192,6144),(12288,6144))
                and weight.dtype==torch.int8 and weight.stride()==(1,weight.size(0))):
            scheme.kernel=DraftInt8Tiles(kernel,tile)
            touched.append(name)
    if not touched:
        raise RuntimeError("no D-Spark INT8 module matched the selected tiles")
    print("DRAFT_INT8_TILES",config,touched,flush=True)
