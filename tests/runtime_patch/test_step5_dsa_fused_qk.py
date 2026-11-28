"""CPU guard/fake contracts for the opt-in Step5 DSA Q/K adapter."""
from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class TensorMetadata:
    def __init__(self, shape, dtype="bf16", device="cuda:0", strides=None):
        self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device
        self.is_cuda=device.startswith("cuda")
        self.strides=strides or tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
    def stride(self, axis):return self.strides[axis]
    def __getitem__(self, slices):
        shape=list(self.shape)
        for axis, value in enumerate(slices):
            if isinstance(value,slice):shape[axis]=len(range(*value.indices(shape[axis])))
        return TensorMetadata(shape,self.dtype,self.device,self.strides)
    def view(self,*shape):
        if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
        strides=[math.prod(shape[i+1:]) for i in range(len(shape))]
        if shape[0]==self.shape[0]:strides[0]=self.stride(0)
        return TensorMetadata(shape,self.dtype,self.device,tuple(strides))
    def contiguous(self):return TensorMetadata(self.shape,self.dtype,self.device)
    def split(self,sections,dim=-1):
        if sum(sections)!=self.shape[dim]:raise ValueError("split sizes must sum to input width")
        result=[];start=0
        for size in sections:
            slices=[slice(None)]*self.ndim;slices[dim]=slice(start,start+size)
            result.append(self[tuple(slices)]);start+=size
        return tuple(result)
    def to(self,device=None,dtype=None):
        return TensorMetadata(self.shape,dtype or self.dtype,device or self.device,self.strides)
    def float(self):return self.to(dtype="f32")
    def __add__(self,value):return TensorMetadata(self.shape,self.dtype,self.device,self.strides)


def attention():
    def forward_native():pass
    def forward_hip():pass
    def norm():return SimpleNamespace(weight=TensorMetadata((192,),"f32"),
        variance_epsilon=1.e-6,_hcu_step5_owner=True,_forward_method=forward_hip)
    return SimpleNamespace(_hcu_step5_owner=True,is_sparse_layer=True,head_dim=192,use_rope=True,
        num_heads=12,num_kv_heads=1,
        q_norm=norm(),k_norm=norm(),
        rotary_emb=SimpleNamespace(is_neox_style=True,rotary_dim=64,head_size=192,_forward_method=forward_native,
                                  cos_sin_cache=TensorMetadata((128,64))))


@contextmanager
def runtime(child="1", master="1", gemma=True):
    launches=[];allocated=[];registered={}
    torch=ModuleType("torch");torch.Tensor=TensorMetadata
    torch.bfloat16="bf16";torch.float32="f32";torch.float16="f16";torch.int32="i32";torch.int64="i64"
    def empty(shape,device,dtype):
        value=TensorMetadata(shape,dtype,device);allocated.append(value);return value
    torch.empty=empty;torch.ops=SimpleNamespace(vllm=SimpleNamespace())
    torch.norm_calls=[];torch.native_norm_calls=[];torch.rope_calls=[]
    def norm(x,weight,eps):
        torch.norm_calls.append((x,weight,eps))
        return empty(x.shape,x.device,x.dtype)
    torch.ops.vllm.hcu_step5_gemma_rms_norm=norm
    def native_norm(x,effective_weight,eps):
        torch.native_norm_calls.append((x,effective_weight,eps))
        return empty(x.shape,x.device,x.dtype)
    def rope(positions,q,k,head_size,rotary_dim,cache,neox):
        torch.rope_calls.append((positions,q,k,head_size,rotary_dim,cache,neox))
        return empty(q.shape,q.device,q.dtype),empty(k.shape,k.device,k.dtype)
    rotary_base=ModuleType("vllm.model_executor.layers.rotary_embedding.base")
    rotary_base.RotaryEmbedding=SimpleNamespace(forward_static=rope)
    utils=ModuleType("vllm.utils.torch_utils")
    def register(**kwargs):
        registered.update(kwargs)
        setattr(torch.ops.vllm,kwargs["op_name"],kwargs["op_func"])
    utils.direct_register_custom_op=register
    platforms=ModuleType("vllm_hcu.platforms")
    platforms.envs=SimpleNamespace(VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=gemma,
        optional_custom_op_enabled=lambda value=True:
        os.environ.get("VLLM_HCU_USE_CUSTOM_OPS")=="1" and bool(value))
    class Kernel:
        def __getitem__(self,grid):
            return lambda *args,**kwargs:launches.append((args,kwargs))
    triton_utils=ModuleType("vllm.triton_utils")
    triton_utils.triton=SimpleNamespace(jit=lambda function:Kernel())
    triton_utils.tl=SimpleNamespace(constexpr=object)
    modules={"torch":torch,"vllm":ModuleType("vllm"),"vllm.utils":ModuleType("vllm.utils"),
        "vllm.utils.torch_utils":utils,"vllm_hcu":ModuleType("vllm_hcu"),
        "vllm_hcu.platforms":platforms,"vllm.triton_utils":triton_utils,
        "vllm.model_executor":ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers":ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.rotary_embedding":ModuleType("vllm.model_executor.layers.rotary_embedding"),
        "vllm.model_executor.layers.rotary_embedding.base":rotary_base}
    modules["vllm"].ir=SimpleNamespace(ops=SimpleNamespace(rms_norm=native_norm))
    with patch.dict(os.environ,{"VLLM_HCU_STEP5_DSA_FUSED_QK":child,"VLLM_HCU_USE_CUSTOM_OPS":master}),patch.dict(sys.modules,modules):
        spec=importlib.util.spec_from_file_location("step5_dsa_qk_cpu",ROOT/"vllm_hcu/models/step5_dsa_qk_fused.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        yield module,torch,launches,allocated,registered


class Step5DSAQKContracts(unittest.TestCase):
    def test_opt_in_master_and_sparse_owner_guards(self):
        for child,master in (("0","1"),("1","0")):
            with runtime(child,master) as (module,*_):
                self.assertFalse(module.can_dispatch(attention()))
                self.assertIsNone(module._qk_kernel)
        with runtime() as (module,*_):
            self.assertTrue(module.can_dispatch(attention()))
            for field,value in (("_hcu_step5_owner",False),("is_sparse_layer",False),("head_dim",128),("use_rope",False)):
                owner=attention();setattr(owner,field,value);self.assertFalse(module.can_dispatch(owner))
            for field,value in (("rotary_dim",192),("is_neox_style",False),("head_size",128)):
                owner=attention();setattr(owner.rotary_emb,field,value);self.assertFalse(module.can_dispatch(owner))
            owner=attention();owner.rotary_emb._forward_method=lambda:None
            self.assertFalse(module.can_dispatch(owner))
            for name in ("q_norm","k_norm"):
                owner=attention();getattr(owner,name)._forward_method=lambda:None
                self.assertFalse(module.can_dispatch(owner))
                owner=attention();getattr(owner,name)._hcu_step5_owner=False
                self.assertFalse(module.can_dispatch(owner))
            owner=attention();owner.k_norm.variance_epsilon=2.e-6
            self.assertFalse(module.can_dispatch(owner))
            os.environ["VLLM_HCU_USE_CUSTOM_OPS"]="0"
            self.assertFalse(module.can_dispatch(attention()))
        with runtime(gemma=False) as (module,*_):
            self.assertFalse(module.can_dispatch(attention()))

    def test_incompatible_tensor_metadata_falls_back(self):
        with runtime() as (module,torch,launches,_,registered):
            # Static selection must not inspect any weight/cache Tensor metadata.
            owner=attention();owner.q_norm.weight=object();owner.k_norm.weight=object()
            owner.rotary_emb.cos_sin_cache=object()
            self.assertTrue(module.can_dispatch(owner))
            for dtype,device,strides,cache_dtype in (
                ("f32","cuda:0",None,"bf16"),("bf16","cpu",None,"f32"),
                ("bf16","cuda:0",(5376,2),"bf16"),("f16","cuda:0",None,"bf16"),
                ("bf16","cuda:0",None,"f32")):
                owner=attention();x=TensorMetadata((16,2688),dtype,device,strides)
                p=TensorMetadata((16,),"i64",device)
                owner.q_norm.weight=TensorMetadata((192,),"f32",device)
                owner.k_norm.weight=TensorMetadata((192,),"f32",device)
                cache_device="cpu" if cache_dtype!=dtype else device
                owner.rotary_emb.cos_sin_cache=TensorMetadata((128,64),cache_dtype,cache_device)
                self.assertTrue(module.can_dispatch(owner))
                before=len(launches)
                q,k=module.forward(owner,p,x)
                self.assertEqual(len(launches),before)
                self.assertEqual((q.dtype,k.dtype),(dtype,dtype))
                self.assertEqual((q.device,k.device),(device,device))
                cache=torch.rope_calls[-1][5]
                self.assertEqual((cache.dtype,cache.device),(dtype,device))
                calls=torch.native_norm_calls if device=="cpu" or dtype=="f16" else torch.norm_calls
                self.assertEqual((calls[-2][0].shape,calls[-1][0].shape),((16,12,192),(16,1,192)))
                fake=registered["fake_impl"](x,p,owner.q_norm.weight,owner.k_norm.weight,
                    owner.rotary_emb.cos_sin_cache,12,1,1.e-6)
                self.assertEqual(tuple(t.dtype for t in fake),(dtype,dtype))
            self.assertEqual(len(torch.native_norm_calls),4)

    def test_runtime_master_disable_overrides_cached_dispatch(self):
        with runtime() as (module,torch,launches,_,registered):
            owner=attention()
            self.assertTrue(module.can_dispatch(owner))
            self.assertIsNotNone(module._qk_kernel)
            # The model retains this startup selection, so exercise forward
            # directly instead of calling can_dispatch again after disabling.
            os.environ["VLLM_HCU_USE_CUSTOM_OPS"]="0"
            for rows in (0,16,64,512):
                x=TensorMetadata((rows,2688),strides=(4096,1))
                p=TensorMetadata((rows,),"i64")
                before=(len(torch.native_norm_calls),len(torch.rope_calls))
                q,k=module.forward(owner,p,x)
                self.assertEqual((q.shape,k.shape),((rows,2304),(rows,192)))
                self.assertEqual((q.dtype,k.dtype),(torch.bfloat16,torch.bfloat16))
                self.assertEqual(len(torch.native_norm_calls)-before[0],2 if rows else 0)
                self.assertEqual(len(torch.rope_calls)-before[1],1 if rows else 0)
            self.assertEqual(launches,[])
            self.assertEqual(torch.norm_calls,[])
            os.environ["VLLM_HCU_USE_CUSTOM_OPS"]="1"
            module.forward(owner,TensorMetadata((16,),"i64"),TensorMetadata((16,2688)))
            self.assertEqual(len(launches),1)

    def test_real_entry_and_fake_only_allocate_qk(self):
        with runtime() as (module,torch,launches,allocated,registered):
            owner=attention()
            for rows in (0,16,64):
                allocated.clear();x=TensorMetadata((rows,2688),strides=(4096,1));p=TensorMetadata((rows,),"i64")
                self.assertTrue(module.can_dispatch(owner))
                previous=len(launches);q,k=module.forward(owner,p,x)
                self.assertEqual((q.shape,k.shape),((rows,2304),(rows,192)))
                self.assertEqual(len(allocated),2)
                self.assertEqual(len(launches)-previous,1 if rows else 0)
                if rows:self.assertIs(launches[-1][0][0],x)
                before=len(launches);allocated.clear()
                fq,fk=registered["fake_impl"](x,p,owner.q_norm.weight,owner.k_norm.weight,
                    owner.rotary_emb.cos_sin_cache,12,1,1.e-6)
                self.assertEqual((fq.shape,fk.shape),(q.shape,k.shape))
                self.assertEqual((fq.dtype,fk.dtype),(torch.bfloat16,torch.bfloat16))
                self.assertEqual((len(allocated),len(launches)),(2,before))
            self.assertEqual(registered["mutates_args"],[])

    def test_prefill_then_decode_dispatch_remains_inside_opaque_op(self):
        with runtime() as (module,torch,launches,allocated,registered):
            owner=attention()
            # A large first call must not exclude the op from the public graph.
            for rows in (512,65,64,16):
                x=TensorMetadata((rows,2688),strides=(4096,1));p=TensorMetadata((rows,),"i64")
                self.assertTrue(module.can_dispatch(owner))
                before=(len(launches),len(torch.norm_calls),len(torch.rope_calls))
                q,k=module.forward(owner,p,x)
                self.assertEqual((q.shape,k.shape),((rows,2304),(rows,192)))
                if rows>64:
                    self.assertEqual(len(launches),before[0])
                    self.assertEqual(len(torch.norm_calls)-before[1],2)
                    qnorm,knorm=torch.norm_calls[-2:]
                    self.assertEqual((qnorm[0].shape,knorm[0].shape),((rows,12,192),(rows,1,192)))
                    self.assertEqual((qnorm[0].stride(0),knorm[0].stride(0)),(2304,192))
                    self.assertIs(qnorm[1],owner.q_norm.weight)
                    self.assertIs(knorm[1],owner.k_norm.weight)
                    self.assertEqual((qnorm[2],knorm[2]),(1.e-6,1.e-6))
                    self.assertEqual(len(torch.rope_calls)-before[2],1)
                    call=torch.rope_calls[-1]
                    self.assertIs(call[0],p)
                    self.assertEqual(call[3:5],(192,64))
                    self.assertIs(call[5],owner.rotary_emb.cos_sin_cache)
                    self.assertIs(call[6],True)
                else:
                    self.assertEqual(len(launches)-before[0],1)
                    self.assertEqual((len(torch.norm_calls),len(torch.rope_calls)),before[1:])
                fake=registered["fake_impl"](x,p,owner.q_norm.weight,owner.k_norm.weight,
                    owner.rotary_emb.cos_sin_cache,12,1,1.e-6)
                self.assertEqual(tuple(t.shape for t in fake),(q.shape,k.shape))
                self.assertEqual(tuple(t.dtype for t in fake),(torch.bfloat16,torch.bfloat16))


if __name__=="__main__":unittest.main()
