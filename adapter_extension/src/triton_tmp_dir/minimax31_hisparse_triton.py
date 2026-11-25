"""Direct verify indices, derived from the trace reference with its launch shape."""
import triton
import triton.language as tl


@triton.jit
def _prepare_verify8(Prefix,Positions,Ends,Locations,HostLocations,Mapping,Scratch,
                     STRIDE:tl.constexpr):
    row=tl.program_id(0)
    j=tl.arange(0,8)
    pos=tl.load(Prefix+row)+j
    loc=128+row*STRIDE+pos
    scratch=tl.load(Scratch+row*8+j)
    tl.store(Positions+row*8+j,pos)
    tl.store(Ends+row*8+j,pos+1)
    tl.store(Locations+row*8+j,loc)
    tl.store(HostLocations+row*8+j,loc)
    tl.store(Mapping+loc,scratch)


def prepare_verify8(prefix,positions,ends,locations,host_locations,mapping,scratch,stride):
    _prepare_verify8[(prefix.numel(),)](prefix,positions,ends,locations,host_locations,
                                     mapping,scratch,stride,num_warps=1)


@triton.jit
def _prepare_dspark(Prefix,Positions,Indices):
    row=tl.program_id(0)
    start=tl.load(Prefix+row)
    j=tl.arange(0,4096)
    tl.store(Indices+row*4096+j,row*4096+(start-4096+j)%4096)
    q=tl.arange(0,8)
    tl.store(Positions+row*7+q,start+q,mask=q<7)


def prepare_dspark(prefix,positions,indices):
    _prepare_dspark[(16,)](prefix,positions,indices,num_warps=4)


@triton.jit
def _store_dspark_context(K,V,Positions,Accepted,KC,VC,KS:tl.constexpr,VS:tl.constexpr):
    token=tl.program_id(0)
    row=token//8
    active=token%8<tl.load(Accepted+row)
    position=tl.load(Positions+token)
    col=tl.arange(0,512)
    k=tl.load(K+token*KS+col,mask=active,other=0).to(tl.float32)
    v=tl.load(V+token*VS+col,mask=active,other=0).to(tl.float32)
    slot=row*4096+position%4096
    tl.store(KC+slot*512+col,k.to(KC.dtype.element_ty),mask=active)
    tl.store(VC+slot*512+col,v.to(VC.dtype.element_ty),mask=active)


def store_dspark_context(k,v,positions,accepted,kc,vc):
    _store_dspark_context[(128,)](k,v,positions,accepted,kc,vc,k.stride(0),v.stride(0),num_warps=4)

@triton.jit
def _expand_indices(
    Reqs,
    TopK,
    Ends,
    Host,
    Device,
    HotForSelection,
    Indices,
    B: tl.constexpr,
    REQ_SLOTS: tl.constexpr,
    HOST_STRIDE: tl.constexpr,
    DEVICE_STRIDE: tl.constexpr,
    BUFFER_SIZE: tl.constexpr,
    TOP_HEAD_STRIDE: tl.constexpr,
    TOP_QUERY_STRIDE: tl.constexpr,
    TOP_CHOICE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    TOP: tl.constexpr,
    COUNT: tl.constexpr,
    SELECTION_TILE: tl.constexpr,
    DIRECT_VERIFY_INDICES: tl.constexpr,
):
    row = tl.program_id(1)
    selection = tl.program_id(0) * SELECTION_TILE + tl.arange(0, SELECTION_TILE)
    offset = tl.arange(0, PAGE)
    query = selection // (HEADS * TOP)
    head = selection // TOP % HEADS
    choice = selection % TOP
    hot = tl.load(HotForSelection + row * COUNT + selection)
    logical = tl.load(
        TopK
        + head * TOP_HEAD_STRIDE
        + (row * WIDTH + query) * TOP_QUERY_STRIDE
        + choice * TOP_CHOICE_STRIDE
    )
    position = logical[:, None] * PAGE + offset[None, :]
    end = tl.load(Ends + row * WIDTH + query)
    prefix = tl.load(Ends + row * WIDTH + WIDTH - 1) - WIDTH
    request = tl.load(Reqs + row).to(tl.int32)
    request = tl.where((request >= 0) & (request < REQ_SLOTS), request, 0)
    mask = (hot[:, None] > 0) & (position < end[:, None])
    host = tl.load(
        Host + request * HOST_STRIDE + logical * PAGE,
        mask=(hot > 0) & (logical * PAGE < prefix),
        other=-1,
    )
    step = tl.minimum(tl.maximum(position - prefix, 0), WIDTH - 1)
    scratch = tl.load(
        Device + request * DEVICE_STRIDE + BUFFER_SIZE + step,
        mask=mask & (position >= prefix),
        other=0,
    )
    valid = mask & tl.where(position < prefix, host[:, None] >= 0, scratch > 0)
    indices = (row * COUNT + selection[:, None]) * PAGE + offset[None, :]
    physical = hot[:, None] * PAGE + offset[None, :]
    if DIRECT_VERIFY_INDICES:
        # The current verify tokens already live in per-request scratch.
        physical = tl.where(position < prefix, physical, scratch)
    tl.store(Indices + indices, tl.where(valid, physical, -1))


def expand_indices(state,topk,ends):
    b=state.reqs.numel()
    _expand_indices[(32,b)](state.reqs,topk,ends,state.host_map,state.device_map,
        state.hot,state.indices,b,state.host_map.shape[0],state.host_map.stride(0),
        state.device_map.stride(0),30720,*topk.stride(),128,8,4,16,512,16,True,num_warps=4)
