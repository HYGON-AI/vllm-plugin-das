# SPDX-License-Identifier: Apache-2.0
"""Explicit verify8 headwise backend for a fresh offline MiniMax decode run.

This backend owns its complete cache lifetime. It cannot reinterpret a live
Core KV cache, and binding is per model instance. Fake-PD history initialization
is explicit; real requests must supply their actual history before execution.
"""
from __future__ import annotations
from types import MethodType
import torch
from .minimax31_hisparse import MappedHostStorage,HeadwisePageState


class Verify8Runtime:
    def __init__(self,*,index_native,decode_native,page_native,requests=16,
                 history_stride=148608,host_slots=2378880,main_slots=493696,
                 layers=60,fake_zero_history=False,registration_lock=None,address_hint=None,
                 registration_chunk_bytes=4*1024**3,host_backing=None,
                 qkv_gemma_int8=False, qk_fp8=False, msa_selector="stock"):
        if msa_selector not in ("stock", "fused", "tuned"):
            raise ValueError("unknown offline MiniMax Verify8 MSA selector")
        if not fake_zero_history:
            raise ValueError('offline fake-PD backend requires explicit fake_zero_history; no implicit request history')
        if requests!=16 or layers!=60 or history_stride!=148608:
            raise ValueError('only the qualified BS128/147K/60-layer geometry is implemented')
        if host_slots<128+requests*history_stride or main_slots<128+requests*30848:
            raise ValueError('cache capacity does not contain the complete active workload')
        self.index_native=index_native;self.decode_native=decode_native;self.page_native=page_native
        self.requests=requests;self.layers=layers;self.history_stride=history_stride
        self.main_slots=main_slots;self.host_slots=host_slots
        dev=torch.device('cuda',torch.cuda.current_device())
        self.fuse_qkv_input = qkv_gemma_int8
        if self.fuse_qkv_input:
            arch = getattr(torch.cuda.get_device_properties(dev), "gcnArchName", "").split(":")[0]
            if arch != "gfx938":
                raise RuntimeError("MiniMax Gemma INT8 QKV fusion was qualified on gfx938")
        self.host_map=(128+torch.arange(requests,device=dev)[:,None]*history_stride+
                       torch.arange(history_stride,device=dev)[None,:]).long()
        self.device_map=(128+torch.arange(requests,device=dev)[:,None]*30848+
                         torch.arange(30848,device=dev)[None,:]).long()
        self.scratch=self.device_map[:,30720:30728].contiguous()
        self.mapping=torch.zeros((host_slots,),device=dev,dtype=torch.int32)
        self.reqs=torch.arange(requests,device=dev,dtype=torch.int64)
        self.real=torch.tensor([requests],device=dev,dtype=torch.int32)
        self.positions=torch.empty((requests*8,),device=dev,dtype=torch.int64)
        self.ends=torch.empty_like(self.positions,dtype=torch.int32)
        self.locations=torch.empty_like(self.positions)
        self.host_locations=torch.empty((requests,8),device=dev,dtype=torch.int32)
        self.prefix=torch.full((requests,),147456,device=dev,dtype=torch.int32)
        # Stage1's bounded path accepts device metadata directly. CPU tensors
        # trigger a blocking H2D copy in every layer of every eager forward.
        self.q_lens=torch.full((requests,),8,device=dev,dtype=torch.int32)
        self.kv_lens=torch.full((requests,),history_stride,device=dev,dtype=torch.int32)
        self.packed_pages=(self.host_map[:,::128]//128).int().flatten().contiguous()
        self.unit=torch.ones((1,),device=dev,dtype=torch.float32)
        self.qk_fp8_cache = None
        if qk_fp8:
            from lightop import op
            from lightop.attention import minimax_qk_norm_rope_fp8_cache

            props = torch.cuda.get_device_properties(dev)
            arch = getattr(props, "gcnArchName", "").split(":")[0]
            if arch != "gfx938" or not hasattr(op, "minimax_qk_norm_rope_fp8_cache"):
                raise RuntimeError("MiniMax QK FP8 requires the gfx938 LightOP build")
            self.qk_fp8_cache = minimax_qk_norm_rope_fp8_cache

        self.msa_page_selector = None
        msa_tuned = msa_selector == "tuned"
        if msa_selector != "stock":
            from hcu_minimax_msa_verify8 import Verify8PageSelector

            args = (self.ends, self.packed_pages, self.unit, history_stride // 128)
            if msa_tuned:
                # gfx938 full-visible pages and four-query score groups.
                self.msa_page_selector = Verify8PageSelector(*args, mode=3)
            else:
                self.msa_page_selector = Verify8PageSelector(*args)
        self.states=[HeadwisePageState(native=page_native,host_map=self.host_map,
            device_map=self.device_map,reqs=self.reqs,real=self.real) for _ in range(layers)]
        # Pin the full host history while the device still has registration
        # headroom. On the 96 GiB node, allocating all GPU caches first leaves
        # only ~1.8 GB for the driver's registration temporaries.
        if host_backing is None:
            self.backing=MappedHostStorage((2,layers,host_slots,4,128),
                layer_bytes=host_slots*512,zero=True,
                registration_lock=registration_lock,zero_threads=8,address_hint=address_hint,
                chunk_bytes=registration_chunk_bytes)
        else:
            storage=host_backing.storage
            zero_alias=(getattr(host_backing,'zero_history_layer_alias',False) and
                        getattr(host_backing,'readonly_zero_history',False) and storage.stride()[:2]==(0,0) and
                        storage[0,0].is_contiguous())
            if (storage.shape!=(2,layers,host_slots,4,128) or storage.is_cuda or
                    storage.dtype!=torch.uint8 or (not storage.is_contiguous() and not zero_alias) or
                    not host_backing.zero_initialized):
                raise ValueError('fake-PD requires a registered, zero-initialized full host pool')
            self.backing=host_backing
        self.main=[];self.index=[]
        for _ in range(layers):
            main=torch.zeros((2,main_slots,4,128),device=dev,dtype=torch.uint8)
            index=torch.zeros((host_slots//128,1,128,128),device=dev,dtype=torch.uint8)
            self.main.append(main.view(torch.float8_e4m3fn))
            self.index.append(index.view(torch.float8_e4m3fn))
        self.k_ptrs=torch.tensor([cache[0].data_ptr() for cache in self.main],device=dev,dtype=torch.int64)
        self.v_ptrs=torch.tensor([cache[1].data_ptr() for cache in self.main],device=dev,dtype=torch.int64)
        self.scheduler=[(None,None) for _ in range(layers)]
        self.bound=[];self.rope=None
        self.qkv_decoder_bindings = []
        self.prepare()

    def prepare(self):
        from triton_tmp_dir.minimax31_hisparse_triton import prepare_verify8
        prepare_verify8(self.prefix,self.positions,self.ends,self.locations,
                        self.host_locations,self.mapping,self.scratch,self.history_stride)

    def bind(self,model):
        if self.bound:raise RuntimeError('verify8 runtime is already bound')
        layers=[m for m in model.modules() if type(m).__name__=='MiniMaxM3SparseAttention']
        if len(layers)!=self.layers:raise ValueError('expected all 60 actual sparse attention layers')
        for layer in layers:
            if type(layer).__module__!='vllm.models.minimax_m3.amd.model':
                raise ValueError('unexpected sparse attention implementation')
            if (layer.num_heads,layer.num_kv_heads,layer.num_idx_heads,layer.head_dim)!=(64,4,4,128):
                raise ValueError('verify8 attention geometry differs from reference')
            if layer.skip_index_topk or hasattr(layer,'_hcu_verify8_runtime'):
                raise ValueError('unsupported index sharing or existing backend owner')
            if layer.rotary_emb.rotary_dim!=64:
                raise ValueError('expected partial RoPE dimension 64')
            for field in ('_k_scale','_v_scale'):
                scale=getattr(layer,field,None)
                if not isinstance(scale,torch.Tensor) or scale.numel()!=1 or scale.item()!=1:
                    raise ValueError('reference FP8 backend requires explicit unit '+field)
        # Preserve the actual checkpoint's rotary semantics and storage dtype.
        # The cache is read-only, may be IPC-owned, and must never be rewritten.
        self.rope=layers[0].rotary_emb.cos_sin_cache
        qkv_bindings = self._prepare_qkv_bindings(model, layers) if self.fuse_qkv_input else []
        if self.fuse_qkv_input and not qkv_bindings:
            raise RuntimeError("VERIFY has no eligible MiniMax Gemma W8A8 QKV projections")
        for index,layer in enumerate(layers):
            original=layer.forward
            def forward(module,positions,hidden_states,_index=index):
                return self.attend(module,_index,positions,hidden_states)
            layer.forward=MethodType(forward,layer)
            layer._hcu_verify8_runtime=self
            self.bound.append((layer,original))
        for decoder, index, kernel in qkv_bindings:
            original = decoder.forward
            decoder.forward = MethodType(self._qkv_forward(original, index, kernel), decoder)
            self.qkv_decoder_bindings.append((decoder, original))

    def _prepare_qkv_bindings(self, model, layers):
        from vllm.models.minimax_m3.amd.model import MiniMaxM3DecoderLayer, MiniMAXGemmaRMSNorm
        from ..quantization.minimax31_w8a8 import MiniMax31HipBlasLtInt8Kernel

        indices = {id(layer): index for index, layer in enumerate(layers)}
        bindings = []
        for decoder in model.modules():
            if type(decoder) is not MiniMaxM3DecoderLayer:
                continue
            index = indices.get(id(decoder.self_attn))
            if index is None or type(decoder.input_layernorm) is not MiniMAXGemmaRMSNorm:
                continue
            proj = decoder.self_attn.qkv_proj
            kernel = getattr(getattr(proj, "scheme", None), "kernel", None)
            if (type(kernel) is MiniMax31HipBlasLtInt8Kernel
                    and proj.tp_size == 1 and not proj.gather_output and proj.bias is None
                    and kernel.supports_prequantized_qkv(proj, self.positions.device)):
                bindings.append((decoder, index, kernel))
        return bindings

    def _qkv_forward(self, original, index, kernel):
        from vllm.models.minimax_m3.amd.model import fused_allreduce_gemma_rms_norm
        from .minimax31_attention_w8a8 import gemma_rmsnorm_quant, supports_gemma_quant

        def forward(decoder, positions, hidden_states, residual):
            norm = decoder.input_layernorm
            if not supports_gemma_quant(hidden_states, residual, norm.weight):
                return original(positions, hidden_states, residual)
            progress = getattr(self, "gpu_progress", None)
            if progress:
                progress.attention(index, 0)
            quant, scale, residual = gemma_rmsnorm_quant(
                hidden_states, residual, norm.weight, norm.variance_epsilon,
            )
            attention = decoder.self_attn
            qkv = kernel.apply_prequantized_qkv(attention.qkv_proj, quant, scale)
            hidden_states = self.attend(attention, index, positions, hidden_states, qkv=qkv)
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, decoder.post_attention_layernorm,
            )
            ffn = decoder.block_sparse_moe if decoder.is_moe_layer else decoder.mlp
            return ffn(hidden_states), residual

        return forward

    def unbind(self):
        for decoder, original in self.qkv_decoder_bindings:
            decoder.forward = original
        self.qkv_decoder_bindings = []
        for layer,original in self.bound:
            if getattr(layer,'_hcu_verify8_runtime',None) is not self:
                raise RuntimeError('verify8 backend ownership changed')
            layer.forward=original;del layer._hcu_verify8_runtime
        self.bound=[]

    def attend(self,layer,index,positions,hidden,*,qkv=None):
        from triton_tmp_dir.minimax31_decode_qk_triton import sparse_qk_index_gemma_rmsnorm_rope_cache
        from triton_tmp_dir.minimax31_prefill_triton import force_local_pages
        if hidden.shape!=(self.requests*8,6144) or positions.data_ptr()!=self.positions.data_ptr():
            raise ValueError('verify8 requires its own stable request metadata')
        progress=getattr(self,'gpu_progress',None)
        if qkv is None:
            if progress:progress.attention(index,0)
            qkv, _ = layer.qkv_proj(hidden)
        if progress:progress.attention(index,1)
        cache=self.main[index];index_cache=self.index[index]
        index_slots = index_cache.view(-1, 1, 128)
        weights = (
            layer.q_norm.weight, layer.k_norm.weight,
            layer.index_q_norm.weight, layer.index_k_norm.weight,
        )
        eps = layer.q_norm.variance_epsilon
        if self.qk_fp8_cache is None:
            q, k, v, iq, ik = qkv.split([8192, 512, 512, 512, 128], dim=-1)
            q, _, iq, _ = sparse_qk_index_gemma_rmsnorm_rope_cache(
                q, k, v, iq, ik, cache[0], cache[1], index_slots, self.locations,
                *weights, positions, self.rope, eps, 128, 64, True,
                fp8_cache=True, main_cache_mapping=self.mapping,
            )
        else:
            q, iq = self.qk_fp8_cache(
                qkv, weights, positions, self.rope, eps, self.locations, self.mapping,
                cache[0], cache[1], index_slots,
            )
        if progress:progress.attention(index,2)
        if self.msa_page_selector is None:
            score=self.index_native.msa128_stage1_page_max(iq.view(-1,4,128),index_cache,
                self.ends,True,-1,self.q_lens,self.kv_lens,None,self.packed_pages,
                self.history_stride,self.unit,self.unit,False,False,True)
            if progress:progress.attention(index,3)
            force_local_pages(score,self.ends,1)
            selected=self.index_native.msa128_stage2_topk_select(
                score,16,self.history_stride//128,0,0,None,None)
        else:
            selected = self.msa_page_selector.select(iq.view(-1, 4, 128), index_cache)
        if progress:progress.attention(index,4)
        state=self.states[index]
        indices=state.plan(selected.permute(1,0,2),self.ends)
        if progress:progress.attention(index,5)
        state.swap(self.backing.storage[0,index],self.backing.storage[1,index],cache[0],cache[1])
        if progress:progress.attention(index,6)
        meta,splits=self.scheduler[index]
        out,lse,meta,splits=self.decode_native.msa128_decode_fp8(
            q.view(self.requests,8,64,128),cache[0].view(-1,128,4,128),indices,
            None,None,meta,splits,None,None,None,128,128**-.5,
            cache[1].view(-1,128,4,128),self.unit,self.unit,self.unit,self.unit)
        self.scheduler[index]=(meta,splits)
        if progress:progress.attention(index,7)
        output=layer.o_proj(out.reshape(-1,8192))[0]
        if progress:progress.attention(index,8)
        return output

    def commit(self,accepted):
        if getattr(self.backing,'readonly_zero_history',False):
            raise RuntimeError('reused zero history is qualified only for isolated VERIFY without commit')
        if accepted.shape!=(self.requests,) or accepted.dtype!=torch.int32:
            raise ValueError('accepted prefix must be a device int32 vector')
        torch._assert_async(((accepted>=0)&(accepted<=8)).all(),'invalid accepted verify8 prefix')
        # Each native call receives host layer views covered by one complete
        # registration; a device pointer cannot span separate USERPTR mappings.
        for begin,end in self.backing.complete_layer_ranges(self.layers):
            self.page_native.commit_all(self.scratch,self.host_locations,accepted,
                self.k_ptrs[begin:end],self.v_ptrs[begin:end],
                self.backing.storage[0,begin:end],self.backing.storage[1,begin:end])
        self.prefix.add_(accepted)
        self.prepare()
