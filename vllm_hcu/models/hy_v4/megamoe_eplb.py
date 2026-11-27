from vllm.model_executor.layers.quantization.utils.int8_utils import round_int8
from vllm.triton_utils import tl, triton


@triton.jit
def _stage_logical(hidden, ids, weights, quant, scales, dest_ids, dest_weights,
                   mapping, counts, load, enabled, unpadded,
                   hidden_sr, id_sr, id_sc, weight_sr, weight_sc, quant_sr,
                   map_sr, map_sc, count_stride, load_stride, offset):
    row = tl.program_id(0)
    cols = tl.arange(0, 8192)
    values = tl.load(hidden + row * hidden_sr + cols, cols < 6144, 0.).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values)), 1e-10)
    tl.store(quant + row * quant_sr + cols, round_int8(values * (127 / amax)), cols < 6144)
    tl.store(scales + row, amax / 127)
    lanes = tl.arange(0, 8)
    logical = tl.load(ids + row * id_sr + lanes * id_sc)
    valid = (logical >= 0) & (logical < 256)
    replicas = tl.maximum(1, tl.load(counts + logical * count_stride, mask=valid, other=1))
    token = (row + offset).to(tl.int64)
    column = ((token * 2654435769) & 0xffffffff) % replicas
    physical = tl.load(mapping + logical * map_sr + column * map_sc, mask=valid, other=-1)
    tl.store(dest_ids + row * 8 + lanes, physical)
    tl.store(dest_weights + row * 8 + lanes, tl.load(weights + row * weight_sr + lanes * weight_sc))
    if tl.load(enabled) != 0:
        active = (physical >= 0) & (physical < 272)
        if unpadded is not None:
            active = active & (token < tl.load(unpadded))
        tl.atomic_add(load + tl.maximum(physical, 0) * load_stride, 1, mask=active)


def stage_logical(hidden, ids, weights, buffer, state, offset):
    hidden = hidden.contiguous()
    mapping = state.logical_to_physical_map
    counts = state.logical_replica_count
    load = state.expert_load_view
    if mapping.ndim != 2 or mapping.shape[0] != 256 or counts.shape != (256,) or load.shape != (272,):
        raise ValueError("Unexpected HY4 static R16 routing state")
    _stage_logical[(hidden.shape[0],)](
        hidden, ids, weights, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
        mapping, counts, load, state.should_record_tensor, state.num_unpadded_tokens_tensors[0],
        hidden.stride(0), *ids.stride(), *weights.stride(), buffer.x.stride(0),
        *mapping.stride(), counts.stride(0), load.stride(0), offset, num_warps=8, num_stages=1,
    )
