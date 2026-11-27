# SPDX-License-Identifier: Apache-2.0
"""Fresh DSpark SWA runtime and zero-copy context KV-only INT8 projection.

The model continues to execute Core's real five decoder layers. This explicit
offline backend owns its FP8 context pool and per-instance attention bindings.
"""
from types import MethodType
import torch


class _KVProjectionView:
    def __init__(self,layer,q_size):
        self.layer=layer
        self.weight=layer.weight[:,q_size:]
        self.weight_scale=layer.weight_scale[q_size:]
        if self.weight.stride()!=(1,self.weight.shape[0]) or not self.weight_scale.is_contiguous():
            raise ValueError('KV-only projection requires processed contiguous channel INT8 weights')

    def __getattr__(self,name):return getattr(self.layer,name)


class DSparkDecodeRuntime:
    def __init__(self,draft,*,fake_zero_history=False):
        if not fake_zero_history:raise ValueError('supply real context history before using this backend')
        if type(draft).__name__!='DSparkMiniMaxDraftModel' or len(draft.model.layers)!=5:
            raise ValueError('expected the actual five-layer MiniMax DSpark draft')
        from triton_tmp_dir.minimax31_dspark_attention import VerifyMLA,block_config
        self.draft=draft;self.requests=16;self.width=7;self.window=4096
        self.positions=torch.empty((112,),device='cuda',dtype=torch.int64)
        self.kv_indices=torch.empty((16*4096,),device='cuda',dtype=torch.int32)
        self.q_indptr=torch.arange(17,device='cuda',dtype=torch.int32)*7
        self.kv_indptr=torch.arange(17,device='cuda',dtype=torch.int32)*4096
        self.cache=torch.zeros((5,2,16*4096//64,64,4,128),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)
        bh,bn,warps=block_config(128,16)
        self.attention=VerifyMLA(16,64,128,128,7,block_h=bh,block_n=bn,num_warps=warps,kv_group_num=16)
        self.projections=[];self.bound=[];pending=[]
        for index,layer in enumerate(draft.model.layers):
            a=layer.self_attn
            if (a.num_heads,a.num_kv_heads,a.head_dim)!=(64,4,128) or a.v_scale is not None:
                raise ValueError('unsupported DSpark attention geometry/value scaling')
            if a.sliding_window!=4096 or a.attention_sink_bias is not None:
                raise ValueError('expected trained SWA4096 without sinks')
            if a.causal is not True:
                raise ValueError('only independently qualified causal DSpark attention is enabled')
            if hasattr(a,'_hcu_dspark_decode_runtime'):
                raise ValueError('DSpark attention already has a runtime owner')
            self.projections.append(_KVProjectionView(a.qkv_proj,a.q_size))
            pending.append((a,index))
        for a,index in pending:
            original=a.forward
            def forward(module,positions,hidden_states,_index=index):
                return self.attend(module,_index,positions,hidden_states)
            a.forward=MethodType(forward,a);a._hcu_dspark_decode_runtime=self
            self.bound.append((a,original))

    def prepare(self,prefix):
        from triton_tmp_dir.minimax31_hisparse_triton import prepare_dspark
        prepare_dspark(prefix,self.positions,self.kv_indices)

    def attend(self,a,index,positions,hidden):
        if hidden.shape!=(112,6144) or positions.data_ptr()!=self.positions.data_ptr():
            raise ValueError('DSpark requires stable B16/gamma7 metadata')
        qkv,_=a.qkv_proj(hidden)
        q,k,v=qkv.split([8192,512,512],dim=-1)
        q=a.q_norm(q.reshape(-1,64,128)).reshape(-1,8192)
        k=a.k_norm(k.reshape(-1,4,128)).reshape(-1,512)
        q,k=a.rotary_emb(positions,q,k)
        out=self.attention(q.view(-1,64,128),k.view(-1,4,128),v.reshape(-1,4,128).contiguous(),
            self.cache[index,0],self.cache[index,1],self.q_indptr,self.kv_indptr,self.kv_indices,
            128**-.5,page_size=64,sliding_window_size=4096,is_causal=a.causal)
        return a.o_proj(out.reshape(-1,8192))[0]

    def update_context(self,aux_hidden,positions,accepted):
        from triton_tmp_dir.minimax31_hisparse_triton import store_dspark_context
        if len(aux_hidden)!=5 or any(t.shape!=(128,6144) for t in aux_hidden):
            raise ValueError('all five trained target auxiliary states are required')
        context=self.draft.model.fc(torch.cat(aux_hidden,dim=-1))
        normed=self.draft.model.hidden_norm(context)
        for index,layer in enumerate(self.draft.model.layers):
            a=layer.self_attn
            # Same vLLM quantizer and hipBLASLt kernel as the full projection;
            # slicing KN channels and per-channel scales creates no weight copy.
            kv=a.qkv_proj.scheme.kernel.apply_weights(self.projections[index],normed)
            k,v=kv.split(512,dim=-1)
            k=a.k_norm(k.reshape(-1,4,128)).reshape(-1,512)
            k,_=a.rotary_emb(positions,k,None)
            store_dspark_context(k,v,positions,accepted,self.cache[index,0],self.cache[index,1])

    def unbind(self):
        for a,original in self.bound:
            if getattr(a,'_hcu_dspark_decode_runtime',None) is not self:
                raise RuntimeError('DSpark attention ownership changed')
            a.forward=original;del a._hcu_dspark_decode_runtime
        self.bound=[]
