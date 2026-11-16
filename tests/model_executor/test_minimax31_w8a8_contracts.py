"""CPU contracts for MiniMax-only W8A8 backend/IPC behavior."""
import ast
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[2]
QUANT=ROOT/"adapter_extension/src/hcu_vllm_das/quantization/minimax31.py"
KERNEL=ROOT/"adapter_extension/src/hcu_vllm_das/quantization/minimax31_w8a8.py"


def method(cls,name):
    return next(f for c in ast.parse(QUANT.read_text()).body
                if isinstance(c,ast.ClassDef) and c.name==cls
                for f in c.body if isinstance(f,ast.FunctionDef) and f.name==name)


class W8A8Contracts(unittest.TestCase):
    def test_ipc_restore_changes_only_kernel_object(self):
        node=method("MiniMax31W8A8LinearMethod","process_weights_after_loading")
        old_type=type("TritonInt8ScaledMMLinearKernel",(),{})
        def init(self,c,names,**kwargs):
            self.config=c;self.layer_param_names=names
        selected=type("MiniMax31HipBlasLtInt8Kernel",(),{"__init__":init})
        fake=types.ModuleType("hcu_vllm_das.quantization.minimax31_w8a8")
        fake.MiniMax31HipBlasLtInt8Kernel=selected
        old=old_type();old.config=object();old.layer_param_names=["weight","weight_scale","input_scale","input_zero_point","azp_adj"]
        layer=types.SimpleNamespace(scheme=types.SimpleNamespace(kernel=old),
            logical_widths=[512],input_size_per_partition=256,
            weight=torch.zeros((512,256),dtype=torch.int8).t(),
            weight_scale=torch.ones((512,1)),input_scale=None,input_zero_point=None,azp_adj=None)
        self_method=types.SimpleNamespace(quantization_config=types.SimpleNamespace(
            w8a8_backend="hipblaslt_v1",w8a8_fallback="error"))
        ns={"torch":torch,"is_weights_pre_processed":lambda:True,
            "__package__":"hcu_vllm_das.quantization"}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(QUANT),"exec"),ns)
        ptrs=(layer.weight.data_ptr(),layer.weight_scale.data_ptr())
        with patch.dict(sys.modules,{fake.__name__:fake}):
            ns[node.name](self_method,layer)
            ns[node.name](self_method,layer)
        self.assertIsInstance(layer.scheme.kernel,selected)
        self.assertEqual(ptrs,(layer.weight.data_ptr(),layer.weight_scale.data_ptr()))
        self.assertEqual(layer.weight.stride(),(1,256))
        self.assertTrue(layer.weight.t().is_contiguous())
        self.assertEqual(layer.weight.t().data_ptr(),layer.weight.data_ptr())
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_native_single_batch_output_view_is_zero_copy(self):
        cls=next(n for n in ast.parse(KERNEL.read_text()).body if isinstance(n,ast.ClassDef))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="apply_weights")
        start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If)
                   and "unexpected hipBLASLt W8A8 return ABI" in ast.unparse(n))
        code=compile(ast.Module(body=fn.body[start:start+2],type_ignores=[]),str(KERNEL),"exec")
        x=torch.empty((17,256),dtype=torch.bfloat16)
        for shape in ((17,512),(1,17,512)):
            out=torch.empty(shape,dtype=x.dtype)
            ptr=out.data_ptr()
            ns={"ok":True,"out":out,"m":17,"n":512,"x":x}
            exec(code,ns)
            self.assertEqual(ns["out"].shape,(17,512))
            self.assertEqual(ns["out"].data_ptr(),ptr)
        with self.assertRaises(RuntimeError):
            exec(code,{"ok":True,"out":torch.empty((17,512,1),dtype=x.dtype),
                       "m":17,"n":512,"x":x})

    def test_config_selects_w8_without_changing_routed_backend(self):
        body=method("MiniMax31SlimQuantConfig","from_config").body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Import) and n.names[0].name=="os")
        stop=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Attribute) and t.attr=="expert_modules" for t in n.targets))
        code=compile(ast.Module(body=body[start:stop],type_ignores=[]),str(QUANT),"exec")
        cfg={"hcu_minimax_routed_backend":"deepgemm_hipc_v1",
             "hcu_minimax_w8a8_backend":"hipblaslt_v1"}
        with patch.dict(os.environ,{"HCU_MINIMAX_ROUTED_BACKEND":"deepgemm_hipc_v1",
                                   "HCU_MINIMAX_W8A8_BACKEND":"hipblaslt_v1"}):
            result=types.SimpleNamespace()
            exec(code,{"config":cfg,"result":result})
            self.assertEqual(result.routed_backend,"deepgemm_hipc_v1")
            self.assertEqual(result.w8a8_backend,"hipblaslt_v1")
            self.assertEqual(result.w8a8_fallback,"error")
            with self.assertRaises(ValueError):
                exec(code,{"config":{"hcu_minimax_routed_backend":"deepgemm_hipc_v1"},
                           "result":types.SimpleNamespace()})

    def test_unsupported_input_is_error_or_explicit_logged_fallback(self):
        class Stock:
            def __init__(self,c,names):
                self.config=c;self.layer_param_names=names
            def apply_weights(self,layer,x,bias):return "explicit-triton-fallback"
            def _get_layer_params(self,layer):
                return layer.weight,layer.weight_scale,None,None,None
        cls=next(n for n in ast.parse(KERNEL.read_text()).body if isinstance(n,ast.ClassDef))
        ns={"torch":torch,"json":json,"os":os,"TritonInt8ScaledMMLinearKernel":Stock}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(KERNEL),"exec"),ns)
        kernel=ns[cls.name](object(),[],fallback="error")
        layer=types.SimpleNamespace(weight=torch.zeros((256,512),dtype=torch.int8),
                                    weight_scale=torch.ones((512,1)))
        x=torch.zeros((1,256),dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            kernel.apply_weights(layer,x,None)
        kernel=ns[cls.name](object(),[],fallback="triton")
        self.assertEqual(kernel.apply_weights(layer,x,None),"explicit-triton-fallback")
        self.assertIn("fallback",kernel._logged)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":
    unittest.main()
