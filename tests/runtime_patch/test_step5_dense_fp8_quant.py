"""Dispatch, fake and fallback contracts for Step5 dense FP8 quantization."""
from contextlib import contextmanager
import __future__
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]

class TensorMetadata:
    def __init__(self,shape,dtype='bf16',device='cuda:0'):
        self.shape=tuple(shape);self.dtype=dtype;self.device=device
        self.ndim=len(shape);self.is_cuda=str(device).startswith('cuda')
    def stride(self,index):return math.prod(self.shape[index+1:])
    def reshape(self,*shape):
        shape=list(shape)
        if -1 in shape:shape[shape.index(-1)]=math.prod(self.shape)//math.prod(x for x in shape if x!=-1)
        return TensorMetadata(shape,self.dtype,self.device)
    def view(self,*shape):return self.reshape(*shape)
    def is_contiguous(self):return True

@contextmanager
def runtime(child='1',master='1'):
    torch=ModuleType('torch');torch.Tensor=TensorMetadata
    torch.bfloat16='bf16';torch.float32='f32';torch.float16='f16';torch.float8_e4m3fn='fp8'
    torch.no_grad=lambda:lambda fn:fn
    torch.empty=lambda shape,device,dtype:TensorMetadata(shape,dtype,device)
    torch.ops=SimpleNamespace(vllm=SimpleNamespace())
    launches=[];registrations={};imports=[]
    class Kernel:
        def __init__(self,fn):self.fn=fn
        def __getitem__(self,grid):
            def launch(*args,**kwargs):launches.append(dict(grid=grid,args=args,kwargs=kwargs))
            return launch
    def jit(fn):imports.append('triton-kernel');return Kernel(fn)
    triton=SimpleNamespace(jit=jit,next_power_of_2=lambda k:1<<(k-1).bit_length())
    triton_utils=ModuleType('vllm.triton_utils');triton_utils.triton=triton;triton_utils.tl=SimpleNamespace(constexpr=object)
    torch_utils=ModuleType('vllm.utils.torch_utils')
    def register(**kwargs):
        registrations[kwargs['op_name']]=kwargs
        setattr(torch.ops.vllm,kwargs['op_name'],kwargs['op_func'])
    torch_utils.direct_register_custom_op=register
    env_spec=importlib.util.spec_from_file_location('dense_quant_policy',ROOT/'vllm_hcu/platforms/envs.py')
    envs=importlib.util.module_from_spec(env_spec);env_spec.loader.exec_module(envs)
    platforms=ModuleType('vllm_hcu.platforms');platforms.envs=envs
    modules={'torch':torch,'vllm':ModuleType('vllm'),'vllm.utils':ModuleType('vllm.utils'),
        'vllm.utils.torch_utils':torch_utils,'vllm.triton_utils':triton_utils,
        'vllm_hcu':ModuleType('vllm_hcu'),'vllm_hcu.platforms':platforms}
    with patch.dict(os.environ,{'VLLM_HCU_USE_CUSTOM_OPS':master,'VLLM_HCU_TRITON_TOKEN_FP8_QUANT':child}),patch.dict(sys.modules,modules):
        spec=importlib.util.spec_from_file_location('dense_quant_cpu_runtime',ROOT/'vllm_hcu/ops/step5/fp8_dense.py')
        module=importlib.util.module_from_spec(spec)
        # The plugin targets Python3.12; postponed annotations also let this
        # dependency-free dispatch test run on older CPU-only developer hosts.
        exec(compile(Path(spec.origin).read_text(encoding='utf-8'),spec.origin,'exec',
                     flags=__future__.annotations.compiler_flag),module.__dict__)
        yield module,torch,launches,registrations,imports

class DenseDecodeQuantContracts(unittest.TestCase):
    def test_supported_decode_dispatches_registered_op(self):
        with runtime() as (module,torch,launches,registered,imports):
            seen=[];torch.ops.vllm.step5_dense_fp8_quant=lambda x:seen.append(x) or ('q','s')
            for m in (0,1,16,64):
                for dtype in (torch.bfloat16,torch.float32):
                    x=TensorMetadata((m,4096),dtype)
                    self.assertEqual(module.quant_per_token_fp8(x),('q','s'));self.assertIs(seen[-1],x)
            self.assertEqual(registered['step5_dense_fp8_quant']['mutates_args'],[])
    def test_unsupported_inputs_preserve_original_torch_path(self):
        with runtime() as (module,torch,launches,registered,imports):
            seen=[];module._quant_per_token_fp8_torch=lambda x:seen.append(x) or ('original-q','original-s')
            torch.ops.vllm.step5_dense_fp8_quant=lambda x:self.fail('Unexpected decode custom-op dispatch')
            inputs=[TensorMetadata((65,4096)),TensorMetadata((16,8193)),TensorMetadata((16,0)),
                    TensorMetadata((16,4096),device='cpu'),TensorMetadata((16,4096),dtype=torch.float16),
                    TensorMetadata((4096,)),TensorMetadata((1,16,4096))]
            for x in inputs:self.assertEqual(module.quant_per_token_fp8(x),('original-q','original-s'));self.assertIs(seen[-1],x)
    def test_dynamic_row_policy_dispatches_op_without_inspecting_symbolic_m(self):
        class SymbolicM:
            def __le__(self,other):raise AssertionError('M guard escaped the opaque real implementation')
        with runtime() as (module,torch,launches,registered,imports):
            seen=[];torch.ops.vllm.step5_dense_fp8_quant=lambda x:seen.append(x) or ('q','s')
            x=TensorMetadata((SymbolicM(),4096),torch.float32)
            self.assertEqual(module.quant_per_token_fp8(x,dynamic_small_m=True),('q','s'))
            self.assertEqual(seen,[x])
    def test_dynamic_row_policy_preserves_large_batch_math_in_real_op(self):
        with runtime() as (module,torch,launches,registered,imports):
            seen=[];module._quant_per_token_fp8_torch=lambda x:seen.append(x) or ('original-q','original-s')
            x=TensorMetadata((512,4096),torch.float32)
            self.assertEqual(module.quant_per_token_fp8(x,dynamic_small_m=True),('original-q','original-s'))
            self.assertEqual(seen,[x]);self.assertEqual(launches,[])
    def test_dynamic_row_policy_still_obeys_master_and_structural_guards(self):
        with runtime() as (module,torch,launches,registered,imports):
            module._quant_per_token_fp8_torch=lambda x:('original-q','original-s')
            torch.ops.vllm.step5_dense_fp8_quant=lambda x:self.fail('Unexpected custom-op dispatch')
            for x in [TensorMetadata((512,8193)),TensorMetadata((64,4096),device='cpu'),
                      TensorMetadata((64,4096),dtype=torch.float16)]:
                self.assertEqual(module.quant_per_token_fp8(x,dynamic_small_m=True),('original-q','original-s'))
            os.environ['VLLM_HCU_USE_CUSTOM_OPS']='0'
            self.assertEqual(module.quant_per_token_fp8(TensorMetadata((64,4096)),dynamic_small_m=True),('original-q','original-s'))
    def test_linear_policy_is_latched_from_actual_construction_config(self):
        source=ROOT/'vllm_hcu/models/step5_text.py'
        cls=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                 if isinstance(n,ast.ClassDef) and n.name=='Step5FP8LinearMethod')
        namespace={'LinearMethodBase':object}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec',
                     flags=__future__.annotations.compiler_flag),namespace)
        policy=namespace['Step5FP8LinearMethod'];config_module=ModuleType('vllm.config')
        for backend,expected in [('eager',True),('inductor',False),(None,False)]:
            config=SimpleNamespace(compilation_config=SimpleNamespace(backend=backend))
            config_module.get_current_vllm_config_or_none=lambda:config
            with patch.dict(sys.modules,{'vllm.config':config_module}):self.assertIs(policy().dynamic_small_m,expected)
        config_module.get_current_vllm_config_or_none=lambda:None
        with patch.dict(sys.modules,{'vllm.config':config_module}):self.assertFalse(policy().dynamic_small_m)
    def test_runtime_master_off_cannot_be_bypassed_by_cached_opt_in(self):
        with runtime() as (module,torch,launches,registered,imports):
            module._quant_per_token_fp8_torch=lambda x:('original-q','original-s')
            os.environ['VLLM_HCU_USE_CUSTOM_OPS']='0';x=TensorMetadata((64,4096))
            self.assertEqual(module.quant_per_token_fp8(x),('original-q','original-s'))
            self.assertEqual(module._step5_dense_fp8_quant(x),('original-q','original-s'))
            self.assertEqual(launches,[])
    def test_disabled_child_or_master_does_not_import_kernel(self):
        for child,master in [('0','1'),('1','0')]:
            with runtime(child,master) as (module,torch,launches,registered,imports):
                self.assertFalse(module._can_use_decode_fp8_quant(TensorMetadata((64,4096))))
                self.assertEqual(imports,[])
    def test_empty_decode_allocates_metadata_without_launch(self):
        with runtime() as (module,torch,launches,registered,imports):
            q,s=module._step5_dense_fp8_quant(TensorMetadata((0,4096),torch.float32))
            self.assertEqual((q.shape,q.dtype),((0,4096),torch.float8_e4m3fn))
            self.assertEqual((s.shape,s.dtype),((0,1),torch.float32));self.assertEqual(launches,[])
    def test_fake_matches_return_contract_without_launch(self):
        with runtime() as (module,torch,launches,registered,imports):
            q,s=registered['step5_dense_fp8_quant']['fake_impl'](TensorMetadata((64,3072),torch.float32))
            self.assertEqual((q.shape,q.dtype),((64,3072),torch.float8_e4m3fn))
            self.assertEqual((s.shape,s.dtype),((64,1),torch.float32));self.assertEqual(launches,[])
    def test_dense_projection_keeps_weight_and_gemm_path(self):
        with runtime() as (module,torch,launches,registered,imports):
            x=TensorMetadata((16,4096));w=TensorMetadata((256,4096),'fp8');s=TensorMetadata((256,1),'f32');seen=[]
            def gemm(q,qs,wq,ws):
                seen.append((wq,ws));return TensorMetadata((16,256))
            torch.ops.vllm.step5_fp8_gemm=gemm
            out=module.fp8_linear(x,w,s)
            self.assertEqual(out.shape,(16,256));self.assertEqual(seen,[(w,s)])

if __name__=='__main__':unittest.main()
