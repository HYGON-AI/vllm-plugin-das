"""EP16 MegaMoE for HY4-VL channel FP8 and HY4 channel INT4 experts."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.distributed as dist


# One EP group and buffer capacity serve all routed layers in this backbone.
_forward_rounds = ContextVar("hy4_mega_forward_rounds", default=None)


@contextmanager
def mega_forward_batch(num_rows):
    # Populate on first use so a forward with no MegaMoE needs no collective.
    token = _forward_rounds.set((num_rows, {}))
    try:
        yield
    finally:
        _forward_rounds.reset(token)


def _chunk_rounds(ep, capacity, local_rows):
    batch = _forward_rounds.get()
    rows, cache = (local_rows, {}) if batch is None else batch
    if local_rows > rows:
        raise ValueError("HY4 routed input exceeds the current backbone batch")
    key = (id(ep), capacity)
    if key not in cache:
        rounds = torch.tensor([max(1, (rows + capacity - 1) // capacity)],
                              device="cpu", dtype=torch.int64)
        dist.all_reduce(rounds, op=dist.ReduceOp.MAX, group=ep.cpu_group)
        cache[key] = rounds.item()
    return cache[key]


def pack_hy4_mega_weights(provider, gate_up, down, gate_up_scale, down_scale):
    count = gate_up.size(0)
    if count not in (16, 17) or gate_up.shape != (count, 4096, 6144) or down.shape != (count, 6144, 2048):
        raise ValueError("HY4 EP16 requires 16 experts per rank, or 17 with R16")
    pairs = []
    for weight, scale in ((gate_up, gate_up_scale), (down, down_scale)):
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError("HY4 MegaMoE requires channel FP8 E4M3 weights")
        if scale.shape not in (weight.shape[:2], (*weight.shape[:2], 1)):
            raise ValueError("HY4 MegaMoE requires one scale per expert output channel")
        pairs.append((weight, scale.reshape(weight.shape[:2]).float().contiguous()))
    return provider.transform_weights_for_mega_moe(*pairs)


def _stage_w4a8(hidden, physical_ids, weights, buffer):
    from vllm.model_executor.layers.quantization.utils.int8_utils import _per_token_quant_int8

    rows, width = hidden.shape
    hidden = hidden.contiguous()
    _per_token_quant_int8[(rows,)](
        hidden, buffer.x, buffer.x_sf, hidden.stride(0), buffer.x.stride(0), width,
        BLOCK=8192, num_warps=8, num_stages=1,
    )
    buffer.topk_idx[:rows].copy_(physical_ids)
    buffer.topk_weights[:rows].copy_(weights)


def run_hy4_mega_experts(provider, hidden, physical_ids, weights, packed, buffer, clamp):
    """Consume the physical top-8 IDs selected by the static EPLB router."""
    rows = hidden.shape[0]
    if hidden.ndim != 2 or hidden.shape[1] != 6144 or hidden.dtype != torch.bfloat16:
        raise ValueError("HY4 MegaMoE expects BF16 hidden states with width 6144")
    if physical_ids.shape != (rows, 8) or weights.shape != physical_ids.shape:
        raise ValueError("HY4 MegaMoE routing must supply eight IDs and weights per token")
    if rows > buffer.num_max_tokens_per_rank:
        raise ValueError("HY4 MegaMoE per-rank token capacity exceeded")
    w4a8 = getattr(buffer, "mode", None) == "w4a8"
    if rows and w4a8:
        _stage_w4a8(hidden, physical_ids, weights, buffer)
    elif rows:
        provider.mega_moe_preprocess(hidden.contiguous(), physical_ids.long().contiguous(),
                                     weights.float().contiguous(), buffer, rows)
    # Empty ranks must also participate in dispatch and combine.
    output = torch.empty_like(hidden)
    if w4a8:
        (gate_up, gate_up_div16), (down, down_div16) = packed
        provider._C.w4a8_mega_moe(
            output, gate_up, down, gate_up_div16, down_div16,
            buffer.buffer, buffer.local_buffer, buffer.handle.buffer_ptrs,
            buffer.handle.signal_ptrs, buffer.group.rank(), buffer.num_max_tokens_per_rank,
            buffer.num_experts, buffer.num_topk, clamp, True,
            l1_weights_sf_div16=gate_up_div16, l2_weights_sf_div16=down_div16,
        )
    else:
        provider.fp8_mega_moe(output, *packed, buffer, activation_clamp=clamp, is_int8=False)
    if w4a8 and not getattr(buffer, "_hy4_w4a8_launched", False):
        buffer._hy4_w4a8_launched = True
        if buffer.group.rank() == 0:
            from vllm.logger import init_logger

            init_logger("vllm.hy4.megamoe").info(
                "Launched HY4 _C.w4a8_mega_moe via %s, rows=%d, clamp=%g",
                provider._C.__file__, rows, clamp,
            )
    return output


class _RoutingFormat:
    @staticmethod
    def topk_indices_dtype():
        return torch.int64


class Hy4MegaKernel:
    # Shared experts execute in the existing runner, outside the routed kernel.
    can_overlap_shared_experts = False
    is_monolithic = False
    prepare_finalize = _RoutingFormat()

    def __init__(self, provider, packed, buffer, ep, clamp, *, decode=False):
        self.decode = decode
        self.provider = provider
        self.packed = packed
        self.buffer = buffer
        self.ep = ep
        self.clamp = clamp

    @staticmethod
    def output_is_reduced():
        return True

    def apply(self, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, apply_router_weight_on_input,
              shared_experts=None, shared_experts_input=None, shared_experts_x_and_scale_quanted=None):
        if self.decode:
            if global_num_experts != 256 or apply_router_weight_on_input:
                raise ValueError("HY4 decode MegaMoE requires 256 output-weighted experts")
            # The scheduler and capture buckets fit the buffer on every DP rank.
            # Launch exactly once even for idle ranks; no CPU collective in a graph.
            return run_hy4_mega_experts(
                self.provider, hidden_states, topk_ids, topk_weights, self.packed,
                self.buffer, self.clamp,
            )
        if global_num_experts != 272 or apply_router_weight_on_input:
            raise ValueError("HY4 MegaMoE requires output-weighted physical R16 routing")
        # Profiling and replicated MTP can supply a much larger batch than a
        # normal PCP shard. All ranks must issue the same number of IPC calls.
        capacity = self.buffer.num_max_tokens_per_rank
        rounds = _chunk_rounds(self.ep, capacity, hidden_states.shape[0])
        if rounds == 1:
            return run_hy4_mega_experts(
                self.provider, hidden_states, topk_ids, topk_weights, self.packed,
                self.buffer, self.clamp,
            )
        outputs = []
        for step in range(rounds):
            span = slice(step * capacity, (step + 1) * capacity)
            outputs.append(run_hy4_mega_experts(
                self.provider, hidden_states[span], topk_ids[span], topk_weights[span],
                self.packed, self.buffer, self.clamp,
            ))
        return torch.cat(outputs)


def _group_buffer(provider, ep, experts, capacity, *, w4a8=False):
    existing = getattr(ep, "_hy4_mega_buffer", None)
    if existing is not None:
        if (existing.num_experts, existing.num_max_tokens_per_rank) != (experts, capacity):
            raise ValueError("HY4 MegaMoE group buffer geometry changed")
        if (getattr(existing, "mode", None) == "w4a8") != w4a8:
            raise ValueError("HY4 MegaMoE group buffer quantization changed")
        return existing
    # Keep the IPC allocation bounded independently of the global 40960
    # scheduler budget. A 2560-token shard fits a normal PCP16 prefill step.
    buf = provider.get_symm_buffer_for_mega_moe(
        ep.device_group, experts, capacity, 8, 6144, 2048, use_fp8_dispatch=True,
        activation="swiglu", **({"mode": "w4a8"} if w4a8 else {}),
    )
    destroy_group = ep.destroy

    def destroy():
        try:
            ep._hy4_mega_buffer = None
            buf.destroy()
        finally:
            ep.destroy = destroy_group
            destroy_group()

    ep._hy4_mega_buffer = buf
    ep.destroy = destroy
    return buf


def install_hy4_mega_kernel(method, layer):
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_ep_group

    config = get_current_vllm_config()
    # Only replace HY4-VL target experts. Native MTP may have its own config
    # and remains supported by its existing grouped-GEMM execution path.
    if config.model_config.architectures != ["HYV4VLForConditionalGeneration"]:
        return False
    p = config.parallel_config
    if p.enable_dbo:
        raise ValueError("HY4 FP8 MegaMoE does not support DBO: overlapping microbatches "
                         "would share the EP group's mutable symmetric buffer")
    if config.lora_config is not None:
        raise ValueError("HY4 FP8 MegaMoE does not support LoRA: its packed expert "
                         "weights cannot apply routed-expert adapters")
    if p.expert_placement_strategy != "linear":
        raise ValueError("HY4 FP8 MegaMoE requires linear expert placement: "
                         "DeepGEMM expects contiguous physical expert IDs on each rank")
    if config.model_config.dtype != torch.bfloat16:
        raise ValueError("HY4 FP8 MegaMoE requires BF16 activations; "
                         f"got model dtype {config.model_config.dtype}")
    ep = get_ep_group()
    decode = (
        p.data_parallel_size == 16 and p.prefill_context_parallel_size == 1
        and p.all2all_backend == "deepep_low_latency" and not p.enable_eplb
        and layer.global_num_experts == 256 and p.decode_context_parallel_size == 1
        and p.pipeline_parallel_size == 1
    )
    prefill = (
        config.model_config.enforce_eager and p.prefill_context_parallel_size == 16
        and p.data_parallel_size == 1 and p.enable_eplb
        and p.eplb_config.num_redundant_experts == 16 and layer.global_num_experts == 272
        and getattr(layer, "_vllm_hcu_static_eplb_row", None) is not None
    )
    if not (
        (decode or prefill) and p.tensor_parallel_size == 1 and p.enable_expert_parallel
        and ep.world_size == 16
        and method.fp8_backend.value == "HCU_DEEPGEMM"
        and method.weight_quant.strategy == "channel" and method.input_quant.strategy == "token"
        and method.input_quant.dynamic and not layer.apply_router_weight_on_input
    ):
        raise ValueError("HY4 FP8 MegaMoE requires PCP16/R16 or low-latency DP16/EP16 and channel/token FP8")
    import deepgemm.mega as mega
    from deepgemm.version import git_hash

    if git_hash == "fcff57a":
        raise RuntimeError("DeepGEMM fcff57a has no HY4 (8,6144,2048) fused kernel; "
                           "install a HY4-capable build or select --moe-backend deep_gemm")
    if isinstance(method.moe_kernel, Hy4MegaKernel):
        raise RuntimeError("Restart HY4 MegaMoE to reload canonical checkpoint weights")
    w1, w2 = layer.w13_weight, layer.w2_weight
    s1, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    packed = pack_hy4_mega_weights(mega, w1, w2, s1, s2)
    capacity = 2560
    if decode:
        capacity = max(config.scheduler_config.max_num_batched_tokens,
                       config.compilation_config.max_cudagraph_capture_size or 0)
    buffer = _group_buffer(mega, ep, layer.global_num_experts, capacity)
    for param, tensor in zip((w1, s1, w2, s2), (*packed[0], *packed[1])):
        if param.dtype != tensor.dtype or param.numel() != tensor.numel():
            raise ValueError("MegaMoE repacking changed checkpoint element count or dtype")
    for param, tensor in zip((w1, s1, w2, s2), (*packed[0], *packed[1])):
        param.data = tensor.view(param.shape)
    method.moe_quant_config = method.get_fused_moe_quant_config(layer)
    method.moe_kernel = Hy4MegaKernel(mega, packed, buffer, ep, layer.swiglu_limit, decode=decode)
    if decode and ep.rank_in_group == 0:
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "HY4 FP8 decode MegaMoE enabled: DP16/EP16, 256 experts, capacity=%d",
            capacity,
        )
    return True


def install_hy4_w4a8_mega_kernel(method, layer):
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_ep_group
    from vllm.logger import init_logger
    import deepgemm.mega as mega
    from deepgemm import pack_w4a8_moe_hipc_weight
    from deepgemm.version import git_hash

    config = get_current_vllm_config()
    parallel = config.parallel_config
    if parallel.enable_dbo or config.lora_config is not None:
        raise ValueError("HY4 W4A8 MegaMoE does not support DBO or LoRA")
    ep = get_ep_group()
    if not (config.model_config.enforce_eager and config.model_config.dtype == torch.bfloat16
            and parallel.enable_expert_parallel and parallel.prefill_context_parallel_size == ep.world_size == 16
            and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
            and parallel.expert_placement_strategy == "linear" and method.supports_eplb
            and getattr(layer, "_vllm_hcu_static_eplb_row", None) is not None
            and layer.global_num_experts == 272 and not layer.apply_router_weight_on_input
            and layer.swiglu_limit == 10):
        raise ValueError("HY4 W4A8 MegaMoE requires eager BF16 PCP16/EP16 static R16 and clamp=10")
    if not hasattr(getattr(mega, "_C", None), "w4a8_mega_moe"):
        raise RuntimeError("HY4 W4A8 MegaMoE requires the DeepGEMM MegaMoe_w4a8 extension")
    if isinstance(method.moe_kernel, Hy4MegaKernel):
        raise RuntimeError("Restart HY4 MegaMoE before reloading routed expert weights")
    w1, w2 = layer.w13_weight, layer.w2_weight
    s1, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    if (w1.shape != (17, 4096, 3072) or w2.shape != (17, 6144, 1024)
            or w1.dtype != torch.int8 or w2.dtype != torch.int8):
        raise ValueError("Unexpected HY4 W4A8 packed INT4 layout")
    if any(scale.dtype != torch.float32 or scale.shape not in ((17, width), (17, width, 1))
           for scale, width in ((s1, 4096), (s2, 6144))):
        raise ValueError("HY4 W4A8 MegaMoE requires FP32 output-channel scales")
    w1.data = pack_w4a8_moe_hipc_weight(mega.tile256_gate_up(w1))
    w2.data = pack_w4a8_moe_hipc_weight(w2)
    s1.data = mega.tile256_gate_up(s1)
    packed = ((w1.data, s1.reshape(17, 4096)), (w2.data, s2.reshape(17, 6144)))
    shard_rows = (config.scheduler_config.max_num_batched_tokens + ep.world_size - 1) // ep.world_size
    capacity = min(4096, ((shard_rows + 63) // 64) * 64)
    buffer = _group_buffer(mega, ep, 272, capacity, w4a8=True)
    method.moe_quant_config = method.get_fused_moe_quant_config(layer)
    method.moe_kernel = Hy4MegaKernel(mega, packed, buffer, ep, layer.swiglu_limit)
    if ep.rank_in_group == 0:
        init_logger("vllm.hy4.megamoe").info(
            "Installed HY4 W4A8 MegaMoE: layer=%s revision=%s capacity=%d",
            method.layer_name, git_hash, buffer.num_max_tokens_per_rank,
        )
