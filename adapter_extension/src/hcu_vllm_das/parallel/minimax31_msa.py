"""Experimental opt-in PCP prefill adapter for a SAME-ABI private teacher119 native build.

No native import/registration. Caller supplies native extension and forward-local
CPU positions/request IDs from PCP planning. Never read GPU positions on host.
The KV storage reinterpretation is legal only before this engine writes any KV;
all cache writers/readers must select this layout for the engine lifetime.
"""
from dataclasses import dataclass
import torch

LAYOUT='pcp_msa128_tokenmajor_v1'

@dataclass(frozen=True)
class NativeCache:
    storage: torch.Tensor
    key_writer: torch.Tensor
    value_writer: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    layout: str=LAYOUT


def bind_fresh_main_cache(storage, *, fresh, previous_layout=None):
    if not fresh or previous_layout not in (None,LAYOUT):
        raise ValueError('MSA cache layout requires a fresh engine-private cache; no mid-cache migration')
    if (storage.ndim!=4 or tuple(storage.shape[1:])!=(4,128,256)
            or storage.dtype not in (torch.uint8,torch.float8_e4m3fn)):
        raise ValueError('expected fresh contiguous uint8/E4M3 main storage [pages,4,128,256]; got shape=%s dtype=%s stride=%s' % (tuple(storage.shape),storage.dtype,storage.stride()))
    # Reinterpret the SAME bytes; never transpose+contiguous the live history.
    if storage.stride() == (131072,256,1024,1):
        # Actual vLLM LBNHC physical order is already B,N,H,D.
        token_major=storage.transpose(1,2)
    elif storage.is_contiguous():
        # Fresh synthetic/explicit head-major allocation uses scoped layout.
        token_major=storage.view(storage.shape[0],128,4,256)
    else:
        raise ValueError('unsupported main cache stride %s' % (storage.stride(),))
    if not token_major.is_contiguous():
        raise RuntimeError('native MSA token-major view is not contiguous')
    k=token_major[...,:128];v=token_major[...,128:]
    return NativeCache(storage,k,v,k.view(torch.float8_e4m3fn).view(-1,4,128),v.view(torch.float8_e4m3fn).view(-1,4,128))


@dataclass(frozen=True)
class ForwardMetadata:
    segments: tuple
    q_lens_cpu: torch.Tensor
    kv_lens_cpu: torch.Tensor
    cu_q: torch.Tensor
    q_offsets: torch.Tensor
    seq_lens: torch.Tensor
    k_end: torch.Tensor
    page_table: torch.Tensor
    packed_kv_pages: torch.Tensor
    max_kv_len: int


def prepare_pcp_forward(page_table, request_ids, positions, request_seq_lens):
    """Split by actual request/contiguous position, including zigzag fragments.

    Inputs except page_table are host sequences supplied once by PCP planner.
    Repeated segment page-table rows intentionally preserve shared prefixes.
    """
    if isinstance(request_ids,torch.Tensor) or isinstance(positions,torch.Tensor):
        raise TypeError('pass planner CPU sequences, not tensor materialization from GPU')
    if page_table.ndim!=2 or page_table.dtype!=torch.int32:
        raise ValueError('expected int32 request page table')
    if len(request_ids)!=len(positions) or not request_ids:
        raise ValueError('expected nonempty matching real-query owner/position lists')
    if len(request_seq_lens)!=page_table.shape[0] or any(type(n) is not int or n<=0 for n in request_seq_lens):
        raise ValueError('invalid host request KV lengths')
    seg=[]
    for row,(req,pos) in enumerate(zip(request_ids,positions)):
        if type(req) is not int or type(pos) is not int or not 0<=req<len(request_seq_lens) or not 0<=pos<request_seq_lens[req]:
            raise ValueError('invalid real query position/request; padding must be excluded before native call')
        if seg and req==seg[-1][0] and pos==seg[-1][2]+seg[-1][3]:
            a,b,c,n=seg[-1];seg[-1]=(a,b,c,n+1)
        else:seg.append((req,row,pos,1))
    dev=page_table.device
    owners=[x[0] for x in seg];lens=[x[3] for x in seg];kv_lens=[request_seq_lens[r] for r in owners]
    counts=[(n+127)//128 for n in kv_lens]
    if max(counts)>page_table.shape[1]:raise ValueError('page table too short for declared KV extent')
    selected_table=page_table.index_select(0,torch.tensor(owners,device=dev,dtype=torch.long)).contiguous()
    row_ids=[r for r,n in enumerate(counts) for _ in range(n)]
    cols=[c for n in counts for c in range(n)]
    packed=selected_table[torch.tensor(row_ids,device=dev),torch.tensor(cols,device=dev)].contiguous()
    torch._assert_async((packed>=0).all(),'active native page table contains invalid page')
    cu=[0]
    for n in lens:cu.append(cu[-1]+n)
    return ForwardMetadata(tuple(seg),torch.tensor(lens,dtype=torch.int32),torch.tensor(kv_lens,dtype=torch.int32),
        torch.tensor(cu,device=dev,dtype=torch.int32),torch.tensor([x[2] for x in seg],device=dev,dtype=torch.int32),
        torch.tensor(kv_lens,device=dev,dtype=torch.int32),torch.tensor([p+1 for p in positions],device=dev,dtype=torch.int32),
        selected_table,packed,max(kv_lens))


def _scalar(scale,device,name):
    if not isinstance(scale,torch.Tensor) or scale.dtype!=torch.float32 or scale.numel()!=1 or scale.device!=device or not scale.is_contiguous():
        raise ValueError(name+' must be caller-owned device FP32 scalar')
    # Existing layer scale stability/finite/positive guard remains the caller's
    # responsibility; this adapter does not silently remove or memoize it.
    return scale


class TeacherMSAPrefill:
    def __init__(self,native):
        required=('msa128_stage1_page_max','msa128_stage2_topk_select','msa128_expand_block_indexes','sparse_prefill_fwd')
        if not all(callable(getattr(native,k,None)) for k in required):
            raise ValueError('same-ABI MSA128 native extension with all four entrypoints is required')
        self.native=native

    def write_gathered_main(self,k,v,slots,cache,*,k_scale,v_scale,writer):
        """Use the existing post-PCP-gather writer on the new physical view.

        Caller passes stock triton_reshape_and_cache_flash; there is no second
        gather and no historical-cache transpose/copy. -1 slots retain stock
        writer padding behavior. Layout must be selected before prefix creation.
        """
        if (cache.layout!=LAYOUT or k.shape!=v.shape or k.ndim!=3 or tuple(k.shape[1:])!=(4,128)
                or slots.ndim!=1 or slots.numel()!=k.shape[0] or k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16
                or slots.dtype not in (torch.int32,torch.int64)
                or any(t.device!=cache.storage.device for t in (k,v,slots))):
            raise ValueError('invalid gathered K/V and physical slot mapping')
        _scalar(k_scale,k.device,'K scale');_scalar(v_scale,k.device,'V scale')
        writer(k,v,cache.key_writer,cache.value_writer,slots,'fp8',k_scale,v_scale)

    def index(self,index_q,index_cache,meta,*,init_blocks,local_blocks,index_q_scale=None,index_k_scale=None):
        if index_q.ndim!=3 or tuple(index_q.shape[1:])!=(4,128) or index_cache.ndim!=3 or tuple(index_cache.shape[1:])!=(128,128):
            raise ValueError('expected index Q [Q,4,128], index cache [pages,128,128]')
        if index_q.dtype!=index_cache.dtype or index_q.dtype not in (torch.bfloat16,torch.float8_e4m3fn):
            raise ValueError('index Q/K must both be BF16 or E4M3')
        if index_q.device != index_cache.device or index_cache.stride(-1) != 1:
            raise ValueError('index Q/K device or cache head stride mismatch')
        if index_q.shape[0]!=meta.k_end.numel():raise ValueError('query metadata rows disagree')
        if index_q.dtype==torch.float8_e4m3fn:
            _scalar(index_q_scale,index_q.device,'index Q scale');_scalar(index_k_scale,index_q.device,'index K scale')
        elif index_q_scale is not None or index_k_scale is not None:
            raise ValueError('BF16 index mode does not consume FP8 scales')
        if (type(init_blocks) is not int or type(local_blocks) is not int or init_blocks<0 or local_blocks<0 or init_blocks+local_blocks>16):
            raise ValueError('forced prefix/local pages must be nonnegative and fit Top16')
        score=self.native.msa128_stage1_page_max(index_q,index_cache.unsqueeze(1),meta.k_end,True,-1,
            meta.q_lens_cpu,meta.kv_lens_cpu,None,meta.packed_kv_pages,meta.max_kv_len,index_q_scale,index_k_scale,True,False,False)
        if local_blocks:
            last=(meta.k_end-1)//128
            pages=torch.arange(score.shape[1],device=score.device).view(1,-1,1)
            mask=(pages<=last.view(1,1,-1))&(pages>=(last-local_blocks+1).clamp_min(0).view(1,1,-1))
            score.masked_fill_(mask,1.e30) # same FP32 sentinel as reference 1.e30-offset
        # Return native [Q,4,16]; the stock vLLM representation is permute(1,0,2).
        return self.native.msa128_stage2_topk_select(score,16,(meta.max_kv_len+127)//128,
                                                     init_blocks,0,None,None)

    def attend(self,q,cache,selected,meta,*,q_scale,k_scale,v_scale,p_scale):
        if cache.layout!=LAYOUT or q.ndim!=3 or tuple(q.shape[1:])!=(64,128):
            raise ValueError('native FP8 main requires bound token-major cache and Q[Q,64,128]')
        if q.dtype!=torch.bfloat16 or selected.shape!=(q.shape[0],4,16) or selected.dtype!=torch.int32:
            raise ValueError('expected BF16 Q and native selected int32[Q,4,16]')
        for s,n in ((q_scale,'Q'),(k_scale,'K'),(v_scale,'V'),(p_scale,'P')):_scalar(s,q.device,n+' scale')
        if q.shape[0]!=meta.k_end.numel():raise ValueError('query metadata rows disagree')
        indices=self.native.msa128_expand_block_indexes(selected.contiguous(),True,meta.q_offsets,meta.seq_lens,
            meta.page_table,None,None,meta.cu_q,None,None,0)
        q8=(q.float()/q_scale).to(torch.float8_e4m3fn)
        out,_,_=self.native.sparse_prefill_fwd(q8,cache.key,indices,128**-.5,128,None,None,cache.value,
                                             q_scale,k_scale,v_scale,p_scale)
        # Native applies all descales: no second scale fold or V multiply.
        return out

# Scoped production bridge: explicit startup opt-in in minimax31_pcp only.
_HOST = None
_META = {}
_ADAPTER = None
_ADAPTER_ROOT = None
_UNIT = {}
_UNIT_VERSION = {}


def set_forward_layout(batch):
    """Publish planner-owned CPU metadata once; local routing uses local mask."""
    global _HOST, _META
    lens=tuple(int(x) for x in batch.num_scheduled_tokens.tolist())
    starts=tuple(int(x) for x in batch.num_computed_tokens_np[:len(lens)].tolist())
    if len(starts)!=len(lens) or sum(lens)!=int(batch.num_tokens):
        raise RuntimeError('native MSA planner start/lens contract mismatch')
    _HOST=(lens,starts,int(batch.num_tokens))
    _META={}


def ensure_native_cache(layer):
    binding=getattr(layer,'_hcu_native_msa_cache',None)
    if binding is None:
        from vllm.model_executor.layers.quantization.utils.quant_utils import FP8_DTYPE
        if FP8_DTYPE != torch.float8_e4m3fn:
            raise RuntimeError('native MSA requires E4M3FN cache bytes, got %s' % FP8_DTYPE)
        binding=bind_fresh_main_cache(layer.kv_cache,fresh=True)
        layer._hcu_native_msa_cache=binding
    elif binding.storage.data_ptr()!=layer.kv_cache.data_ptr():
        raise RuntimeError('native MSA cache storage changed; restart with a fresh layout binding')
    return binding


def _get_adapter(extension_dir):
    global _ADAPTER, _ADAPTER_ROOT
    import hashlib,importlib,importlib.metadata,json,sys
    from pathlib import Path
    root=Path(extension_dir).resolve()
    if _ADAPTER is not None:
        if root != _ADAPTER_ROOT:
            raise RuntimeError('native MSA artifact changed; restart the engine')
        return _ADAPTER
    proof=json.loads((root.parent/'native_import_result.json').read_text())
    plan=json.loads((root.parent/'build_plan.json').read_text())
    so=root/'hcu_minimax_msa'/Path(proof['path']).name
    if (sys.version_info[:2]!=(3,12)
            or importlib.metadata.version('torch')!=plan['target_identity']['torch_distribution']
            or hashlib.sha256(so.read_bytes()).hexdigest()!=proof['sha256']
            or bool(torch._C._GLIBCXX_USE_CXX11_ABI)!=proof['cxx11_abi']):
        raise RuntimeError('native MSA target ABI/build identity mismatch')
    sys.path.insert(0,str(root))
    native=importlib.import_module('hcu_minimax_msa._native')
    if Path(native.__file__).resolve()!=so:
        raise RuntimeError('native MSA loaded from a foreign path')
    _ADAPTER=TeacherMSAPrefill(native)
    _ADAPTER_ROOT=root
    return _ADAPTER


def _forward_meta(sub,req_start,req_count):
    if _HOST is None:raise RuntimeError('native MSA has no PCP planner metadata')
    key=(id(sub),req_start,req_count)
    if key in _META:return _META[key]
    lens,starts,_=_HOST
    lengths=lens[req_start:req_start+req_count];offsets=starts[req_start:req_start+req_count]
    if len(lengths)!=req_count or sub.block_table.shape[0]!=req_count or any(n<=0 for n in lengths):
        raise RuntimeError('native MSA split metadata differs from planner rows')
    owners=[];positions=[]
    for row,(n,start) in enumerate(zip(lengths,offsets)):
        owners.extend([row]*n);positions.extend(range(start,start+n))
    meta=prepare_pcp_forward(sub.block_table,owners,positions,[s+n for s,n in zip(offsets,lengths)])
    torch._assert_async((meta.seq_lens==sub.seq_lens).all(),'native MSA planner seq_lens mismatch')
    if hasattr(sub,'cu_seqlens_q'):
        torch._assert_async((meta.cu_q==sub.cu_seqlens_q).all(),'native MSA planner cu_seqlens mismatch')
    _META[key]=meta
    return meta


def run_native_attention(layer,query,index_query):
    from vllm.forward_context import get_forward_context
    context=get_forward_context();metadata=context.attn_metadata
    if not isinstance(metadata,dict) or _HOST is None or _HOST[2]==0:
        return torch.zeros_like(query)
    main=metadata.get(layer.layer_name);side=metadata.get(layer.indexer.index_cache.prefix)
    if main is None or side is None:
        raise RuntimeError('native MSA requires both main and index cache metadata')
    for field in ('num_decodes','num_decode_tokens','num_prefills','num_prefill_tokens'):
        if getattr(main,field)!=getattr(side,field):raise RuntimeError('native MSA main/index query split mismatch')
    if layer.skip_index_topk or index_query is None:
        raise RuntimeError('native MSA index sharing is outside current opt-in contract')
    adapter=_get_adapter(layer._hcu_msa_extension_dir);cache=ensure_native_cache(layer)
    n=query.shape[0];q=query.view(n,64,128);iq=index_query.view(n,4,128)
    unit=_UNIT.get(query.device)
    if unit is None:
        # Inference tensors do not expose a version counter. Allocate this
        # immutable software-contract scalar outside inference mode.
        with torch.inference_mode(False):
            unit=torch.ones(1,device=query.device,dtype=torch.float32)
        _UNIT[query.device]=unit
        # Unit Q/K descales are fixed for the whole engine/cache lifetime.
        # The private tensor is only read by native calls, never updated.
        _UNIT_VERSION[query.device] = unit._version
    if iq.dtype == torch.float8_e4m3fn:
        if unit._version != _UNIT_VERSION[query.device]:
            raise RuntimeError('FP8 index unit scale was modified while cache is bound')
    parts=[]
    split=[(0,main.num_decode_tokens,0,main.num_decodes,main.decode,side.decode),
           (main.num_decode_tokens,main.num_prefill_tokens,main.num_decodes,main.num_prefills,main.prefill,side.prefill)]
    for begin,count,req_start,req_count,main_sub,index_sub in split:
        if count==0:continue
        mm=_forward_meta(main_sub,req_start,req_count)
        im=_forward_meta(index_sub,req_start,req_count)
        if mm.k_end.numel()!=count or im.k_end.numel()!=count:
            raise RuntimeError('native MSA query token count mismatch')
        selected=adapter.index(iq[begin:begin+count],layer.indexer.index_cache.kv_cache,im,
            init_blocks=layer.indexer.impl.init_blocks,local_blocks=layer.indexer.impl.local_blocks,
            index_q_scale=unit if iq.dtype == torch.float8_e4m3fn else None,
            index_k_scale=unit if iq.dtype == torch.float8_e4m3fn else None)
        layer.topk_indices_buffer[:,begin:begin+count,:].copy_(selected.permute(1,0,2))
        out=adapter.attend(q[begin:begin+count],cache,selected,mm,
            q_scale=unit,k_scale=layer._k_scale,v_scale=layer._v_scale,p_scale=unit)
        parts.append((begin,count,out))
    if len(parts)==1 and parts[0][0]==0 and parts[0][1]==n:
        return parts[0][2].view(n,-1)
    result=torch.zeros_like(query)
    for begin,count,out in parts:result[begin:begin+count].copy_(out.reshape(count,-1))
    return result
