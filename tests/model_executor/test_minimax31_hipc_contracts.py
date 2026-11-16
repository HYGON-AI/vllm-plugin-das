"""CPU contracts for versioned MiniMax HIPC weight loading; no GPU imports."""
import ast
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT/"adapter_extension/src/hcu_vllm_das/quantization/minimax31.py"


def method_ast(class_name, method):
    tree=ast.parse(QUANT.read_text())
    return next(f for c in tree.body if isinstance(c,ast.ClassDef) and c.name==class_name
                for f in c.body if isinstance(f,ast.FunctionDef) and f.name==method)


class HIPCContracts(unittest.TestCase):
    def test_processed_ipc_does_not_repack(self):
        node=method_ast("MiniMax31W4A8MoEMethod","process_weights_after_loading")
        calls=[]
        fake=types.ModuleType("triton_tmp_dir.minimax31_hipc_moe")
        fake.LAYOUT="signed-high-nibble-first-hipc-split-v1"
        def pack(a,b):
            calls.append("pack")
            return a.clone(),b.clone()
        fake.prepare_hipc_weights=pack
        processed=False
        ns={"torch":torch,"is_weights_pre_processed":lambda:processed,
            "replace_parameter":lambda layer,name,value:setattr(layer,name,value)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(QUANT),"exec"),ns)
        method=types.SimpleNamespace(
            moe=types.SimpleNamespace(activation="swigluoai_uninterleave",hidden_dim=512,
                                      intermediate_size_per_partition=256),
            quant_config=types.SimpleNamespace(routed_backend="deepgemm_hipc_v1"),
            get_fused_moe_quant_config=lambda layer:"scales")
        layer=types.SimpleNamespace(
            w13_weight=torch.zeros((2,512,256),dtype=torch.int8),
            w2_weight=torch.zeros((2,512,128),dtype=torch.int8),
            w13_weight_scale=torch.ones((2,512,1)),
            w2_weight_scale=torch.ones((2,512,1)))
        scales=(layer.w13_weight_scale.data_ptr(),layer.w2_weight_scale.data_ptr())
        with patch.dict(sys.modules,{fake.__name__:fake}):
            ns[node.name](method,layer)
            self.assertEqual(calls,["pack"])
            self.assertEqual(layer._minimax31_expert_layout,fake.LAYOUT)
            storage=(layer.w13_weight.data_ptr(),layer.w2_weight.data_ptr())
            processed=True
            ns[node.name](method,layer)
        self.assertEqual(calls,["pack"])
        self.assertEqual(storage,(layer.w13_weight.data_ptr(),layer.w2_weight.data_ptr()))
        self.assertEqual(scales,(layer.w13_weight_scale.data_ptr(),layer.w2_weight_scale.data_ptr()))
        self.assertEqual(method.moe_quant_config,"scales")
        self.assertFalse(torch.cuda.is_initialized())

    def test_only_fresh_hipc_requires_device_loading(self):
        node=method_ast("MiniMax31W4A8MoEMethod","requires_device_loading")
        node.decorator_list=[]
        state={"processed":False}
        ns={"is_weights_pre_processed":lambda:state["processed"]}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(QUANT),"exec"),ns)
        method=types.SimpleNamespace(quant_config=types.SimpleNamespace(routed_backend="deepgemm_hipc_v1"))
        self.assertTrue(ns[node.name](method))
        state["processed"]=True
        self.assertFalse(ns[node.name](method))
        state["processed"]=False
        method.quant_config.routed_backend="bolt"
        self.assertFalse(ns[node.name](method))

    def test_environment_cannot_override_versioned_backend(self):
        body=method_ast("MiniMax31SlimQuantConfig","from_config").body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Import) and n.names[0].name=="os")
        stop=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Attribute) and t.attr=="expert_modules" for t in n.targets))
        code=compile(ast.Module(body=body[start:stop],type_ignores=[]),str(QUANT),"exec")
        for backend in ("bolt","deepgemm_hipc_v1"):
            with patch.dict(os.environ,{"HCU_MINIMAX_ROUTED_BACKEND":backend}):
                result=types.SimpleNamespace()
                exec(code,{"config":{"hcu_minimax_routed_backend":backend},"result":result})
                self.assertEqual(result.routed_backend,backend)
        with patch.dict(os.environ,{"HCU_MINIMAX_ROUTED_BACKEND":"deepgemm_hipc_v1"}):
            with self.assertRaises(ValueError):
                exec(code,{"config":{},"result":types.SimpleNamespace()})

    def test_padding_contract(self):
        path=ROOT/"triton_tmp_dir/minimax31_hipc_moe.py"
        node=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=="padded_layout")
        ns={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),ns)
        actual,padded=ns["padded_layout"]([1,128,129]+[0]*13,100,4)
        self.assertEqual(actual[-1],258)
        self.assertEqual(padded[-1],512)
        self.assertEqual(padded[:4],[0,128,256,512])
        with self.assertRaises(ValueError):
            ns["padded_layout"]([401]+[0]*15,100,4)


if __name__=="__main__":
    unittest.main()
