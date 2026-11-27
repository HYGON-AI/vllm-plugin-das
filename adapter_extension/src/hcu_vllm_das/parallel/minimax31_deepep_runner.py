"""Experimental opt-in DeepEP normal routed-expert runner.

The caller owns shared-expert scheduling and outer routed scaling. `expert_fn`
accepts received hidden, FP32 route weights, local INT32 expert IDs and returns
per-received-token weighted local-expert sums. Never invoke without first
validating this wheel's received ID representation on all eight GPUs.
"""
from functools import lru_cache
import json
import threading

import torch

from .minimax31_prefill import enabled as prefill_optimized


def dispatch_routed(hidden, logits, select_experts, expert_fn, buffer,
                    *, ep_rank, recv_id_abi, physical_experts=128):
    if recv_id_abi != 'local_minus_one_v1':
        raise RuntimeError('unverified DeepEP received expert ID ABI')
    if not 0 <= ep_rank < 8 or hidden.ndim != 2 or logits.shape != (hidden.shape[0],128):
        raise ValueError('invalid MiniMax PCP/EP router shape or rank')
    if hidden.dtype != torch.bfloat16:
        raise ValueError('DeepEP normal path requires BF16 activations')
    if physical_experts not in (128,136):raise ValueError('unsupported physical expert capacity')
    local_experts=physical_experts//8
    weights, ids = select_experts(hidden,logits)
    if ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != hidden.shape[0]:
        raise ValueError('router result shape mismatch')
    if weights.dtype != torch.float32 or ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('unexpected topk dtypes')
    if not bool(((ids>=-1)&(ids<physical_experts)).all()):
        raise ValueError('global expert ID out of range')
    ids=ids.to(torch.int64).contiguous()
    weights=weights.masked_fill(ids<0,0).contiguous()
    nr,nrdma,ne,mask,event=buffer.get_dispatch_layout(ids,physical_experts,async_finish=False)
    result=buffer.dispatch(hidden.contiguous(),topk_idx=ids,topk_weights=weights,
        num_tokens_per_rank=nr,num_tokens_per_rdma_rank=nrdma,is_token_in_rank=mask,
        num_tokens_per_expert=ne,expert_alignment=1,async_finish=False)
    if len(result)!=6:raise RuntimeError('unsupported DeepEP dispatch return ABI')
    recv_x,recv_ids,recv_weights,counts,handle,event=result
    if (not isinstance(recv_x,torch.Tensor) or recv_x.ndim!=2 or recv_x.dtype!=torch.bfloat16
            or recv_ids is None or recv_weights is None or recv_ids.shape!=recv_weights.shape
            or recv_ids.shape[0]!=recv_x.shape[0]):
        raise RuntimeError('unexpected DeepEP receive shape/dtype')
    if recv_ids.dtype not in (torch.int32,torch.int64) or recv_weights.dtype!=torch.float32:
        raise RuntimeError('unexpected DeepEP receive route dtype')
    if not bool(((recv_ids>=-1)&(recv_ids<local_experts)).all()):
        raise RuntimeError('unexpected DeepEP local route ID')
    safe_weights=recv_weights.masked_fill(recv_ids<0,0).contiguous()
    # alignment=1 counts are exact valid-route counts, already returned by DeepEP.
    # Keep received IDs local: the expert adapter must not remap and compact again.
    local=expert_fn(recv_x.contiguous(),safe_weights,
                    recv_ids.to(torch.int32).contiguous(),counts)
    if not isinstance(local,torch.Tensor) or local.shape!=recv_x.shape or local.dtype!=torch.bfloat16:
        raise RuntimeError('unexpected local expert output')
    combined,returned_weights,event=buffer.combine(local,handle,async_finish=False)
    if not isinstance(combined,torch.Tensor) or combined.shape!=hidden.shape or combined.dtype!=hidden.dtype:
        raise RuntimeError('unexpected DeepEP combine output')
    return combined

# Model-scoped factory seam; fixed-count routing is the default.
from .minimax31_ep_runner import MiniMax31PcpEpMoERunner
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder

_buffer = None
_buffer_identity = None
_buffer_inflight = threading.Lock()

def _get_deepep_buffer(group, hidden):
    global _buffer, _buffer_identity
    import deep_ep
    # A single shared buffer is safe only for calls ordered on one stream.
    identity = (id(group.device_group), hidden.device, hidden.shape[1], hidden.dtype,
                torch.cuda.current_stream(hidden.device).cuda_stream)
    if _buffer is not None:
        if identity != _buffer_identity:
            raise RuntimeError('DeepEP buffer group/device/hidden/stream identity changed')
        return _buffer
    dispatch = deep_ep.Buffer.get_dispatch_config(8)
    combine = deep_ep.Buffer.get_combine_config(8)
    hidden_bytes = hidden.shape[1] * hidden.element_size()
    nvl = max(dispatch.get_nvl_buffer_size_hint(hidden_bytes,8),
              combine.get_nvl_buffer_size_hint(hidden_bytes,8))
    rdma = max(dispatch.get_rdma_buffer_size_hint(hidden_bytes,8),
               combine.get_rdma_buffer_size_hint(hidden_bytes,8))
    if rdma != 0:
        raise RuntimeError('unexpected multi-node DeepEP buffer requirement')
    _buffer = deep_ep.Buffer(group.device_group, num_nvl_bytes=nvl, num_rdma_bytes=rdma)
    _buffer_identity = identity
    return _buffer


_aligned_moe = None
_aligned_identity = None


def _get_aligned_moe(layer, hidden):
    """One buffer/workspace per rank, while weights remain owned by each layer."""
    global _aligned_moe, _aligned_identity
    from triton_tmp_dir.minimax31_moe_sglang_v1 import AlignedMoE, PackedExperts
    from triton_tmp_dir.minimax31_hipc_moe import LAYOUT
    if (layer.quant_method.quant_config.routed_backend != "deepgemm_hipc_v1"
            or getattr(layer, "_minimax31_expert_layout", None) != LAYOUT):
        raise RuntimeError("aligned MoE requires verified HIPC packed weights")
    m = layer.quant_method.moe
    if (m.swiglu_alpha, m.swiglu_beta, m.swiglu_limit) != (1.702, 1.0, 7.0):
        raise RuntimeError("aligned MoE requires MiniMax31 split OAI parameters")
    group = get_ep_group()
    identity = (id(group.device_group), hidden.device, hidden.shape[1],
                torch.cuda.current_stream(hidden.device).cuda_stream,
                layer.global_num_experts, layer.w13_weight.shape)
    weights = getattr(layer, "_minimax31_aligned_weights", None)
    if weights is None:
        weights = PackedExperts(layer.w13_weight, layer.w2_weight,
                                layer.w13_weight_scale, layer.w2_weight_scale)
        weights.validate()
        layer._minimax31_aligned_weights = weights
    if _aligned_moe is None:
        _aligned_moe = AlignedMoE(group.device_group, weights, num_sms=64,
                                 routed_scale=1.0, shared_fn=None)
        _aligned_identity = identity
    elif identity != _aligned_identity:
        raise RuntimeError("shared aligned MoE group/device/shape/stream identity changed")
    if not getattr(layer, "_minimax31_aligned_logged", False):
        print("MINIMAX_ALIGNED_MOE " + json.dumps({
            "layer": layer.layer_name, "backend": "sglang_aligned_v1",
            "transport": "int8+fp32_scale", "alignment": 256, "num_sms": 64,
            "workspace_policy": "shared-across-serial-layers", "executor_id": id(_aligned_moe),
            "weight_shape": list(layer.w13_weight.shape), "weight_before_bf16_store": False,
            "weight_pack_repeated": False}), flush=True)
        layer._minimax31_aligned_logged = True
    return _aligned_moe, weights


@lru_cache(maxsize=None)
def _supports_shared_down(device):
    gpu = torch.cuda.get_device_properties(device)
    return (getattr(gpu, "gcnArchName", "").split(":", 1)[0] == "gfx938"
            and gpu.multi_processor_count == 64)


class MiniMax31DeepEpRunner(MiniMax31PcpEpMoERunner):
    """Synchronous rank-local routing with existing modular expert math.

    Fixed-count local-ID expert adapter bypasses duplicate compaction and EP map.
    Shared expert retains the stock launch/wait lifecycle; inherited
    forward retains routed scaling and final output semantics. This experiment
    assumes the local ID ABI checked by the dedicated eight-rank probe.
    """
    def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        # One global DeepEP buffer and one auxiliary event slot: reject re-entry
        # before launching shared work, and hold the lock through the join.
        if not _buffer_inflight.acquire(blocking=False):
            raise RuntimeError("DeepEP experiment does not support overlapping calls")
        self._hcu_shared_pending = False
        try:
            if getattr(self, "_hcu_shared_poisoned", False):
                raise RuntimeError("shared overlap event state is invalid; restart the engine")
            result = self._forward_locked(hidden_states, router_logits, shared_experts_input, input_ids)
            if hasattr(self.routed_experts,"_static_placement"):
                from .minimax31_static_runtime import complete_layer
                complete_layer(self.routed_experts._static_layer_id)
            return result
        finally:
            try:
                if self._hcu_shared_pending:
                    shared = self._shared_experts
                    try:
                        shared._output_ready_event[0].wait(self._hcu_shared_main_stream)
                    except BaseException:
                        self._hcu_shared_poisoned = True
                        raise
                    finally:
                        shared._output[0] = None
                        self._hcu_shared_pending = False
            finally:
                _buffer_inflight.release()

    def _launch_long_shared(self, shared_input, *, defer_down=False):
        shared = self._shared_experts
        if (shared is None or shared.enable_dbo or shared._stream is None
                or shared._disable_shared_experts_overlap
                or self.routed_experts.quant_method.quant_config.routed_backend != "deepgemm_hipc_v1"):
            return False
        # Only the audited MiniMax TP1 shared MLP is allowed on the side
        # stream: RowParallelLinear is configured without a reduction, so it
        # cannot issue a collective that races DeepEP's main-stream sequence.
        from vllm.models.minimax_m3.amd.model import MiniMaxM3MLP
        layer = shared._layer
        if (type(layer) is not MiniMaxM3MLP
                or getattr(layer.down_proj, "reduce_results", None) is not False
                or any(getattr(proj, "tp_size", None) != 1
                       for proj in (layer.gate_up_proj, layer.down_proj))
                or any(type(getattr(getattr(proj, "scheme", None), "kernel", None)).__name__
                       != "MiniMax31HipBlasLtInt8Kernel"
                       for proj in (layer.gate_up_proj, layer.down_proj))
                or shared_input.ndim != 2 or shared_input.dtype != torch.bfloat16):
            return False
        if shared._output[0] is not None:
            raise RuntimeError("shared output slot is already in use")
        main = torch.cuda.current_stream(shared_input.device)
        self._hcu_shared_main_stream = main
        shared._input_ready_event[0].record(main)
        # The input was allocated on the caller's stream and is read asynchronously.
        shared_input.record_stream(shared._stream)
        intermediate = None
        with torch.cuda.stream(shared._stream):
            shared._input_ready_event[0].wait(shared._stream)
            try:
                if defer_down:
                    from vllm.models.minimax_m3.amd.model import swiglu_oai_split

                    gate_up, _ = layer.gate_up_proj(shared_input)
                    intermediate = swiglu_oai_split(
                        gate_up, alpha=layer.swiglu_alpha,
                        beta=layer.swiglu_beta, limit=layer.swiglu_limit,
                    )
                else:
                    shared._output[0] = layer(shared_input)
            finally:
                # Join even if a partially enqueued shared forward raises.
                try:
                    shared._output_ready_event[0].record(shared._stream)
                    self._hcu_shared_pending = True
                except BaseException:
                    self._hcu_shared_poisoned = True
                    raise
        return intermediate if defer_down else True

    def _launch_shared_down(self, intermediate):
        shared = self._shared_experts
        shared._stream.wait_stream(self._hcu_shared_main_stream)
        with torch.cuda.stream(shared._stream):
            try:
                shared._output[0], _ = shared._layer.down_proj(intermediate)
            finally:
                try:
                    shared._output_ready_event[0].record(shared._stream)
                except BaseException:
                    self._hcu_shared_poisoned = True
                    raise

    def _forward_locked(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        group = self._pcp_ep_group()
        if self.gate is not None or input_ids is not None or self.routed_experts.quant_method.is_monolithic:
            raise RuntimeError('unsupported DeepEP MiniMax runner configuration')
        self.routed_experts._ensure_moe_quant_config_init()
        # Match stock lifecycle: short queries may choose the auxiliary stream.
        shared_overlapping = False
        shared_intermediate = None
        if (self._shared_experts is not None
                and shared_experts_input is not None
                and shared_experts_input.shape[0] >= 2048):
            defer_down = (
                shared_experts_input.shape[0] == 6144
                and self.routed_experts.quant_method.quant_config.routed_backend == "deepgemm_hipc_v1"
                and not torch.cuda.is_current_stream_capturing()
                and prefill_optimized()
                and _supports_shared_down(shared_experts_input.device)
            )
            if defer_down:
                intermediate = self._launch_long_shared(shared_experts_input, defer_down=True)
                if intermediate is not False:
                    shared_intermediate = intermediate
                    shared_overlapping = True
            else:
                shared_overlapping = self._launch_long_shared(shared_experts_input)
        if not shared_overlapping:
            if self._shared_experts is not None:
                shared_overlapping = self._shared_experts.maybe_forward_async(shared_experts_input)
            self._maybe_apply_shared_experts(shared_experts_input, SharedExpertsOrder.NO_OVERLAP)
        def select(hidden, logits):
            weights, ids = self.router.select_experts(hidden_states=hidden, router_logits=logits,
                    topk_indices_dtype=self._quant_method.topk_indices_dtype, input_ids=None)
            if hasattr(self.routed_experts,"_static_placement"):
                from .minimax31_static_selector import select_physical
                ids = select_physical(self.routed_experts, ids)
            return weights, ids
        def experts(hidden, weights, ids, counts):
            from triton_tmp_dir.minimax31_deepep_pack import run_received_local_routes
            from triton_tmp_dir.minimax31_w4a8_moe import apply_minimax31_w4a8
            layer = self.routed_experts
            backend = layer.quant_method.quant_config.routed_backend
            if backend != "bolt":
                raise RuntimeError("unknown MiniMax routed backend")
            if getattr(layer, '_minimax31_expert_layout', None) != 'signed-high-nibble-first-interleaved-v1':
                raise RuntimeError('fixed-count expert layout is not prepared')
            moe = layer.quant_method.moe
            def local_expert(px, pw, pi):
                if px.shape[0] > 262144:
                    print('MINIMAX_LOCAL_EXPERT_MEMORY ' + json.dumps({
                        'packed_rows':px.shape[0], 'hidden':px.shape[1],
                        'packed_bytes':px.numel()*px.element_size(),
                        'reuse_input':True, 'bolt_chunk_size':32768}), flush=True)
                result = apply_minimax31_w4a8(
                    px, layer.w13_weight, layer.w2_weight,
                    layer.w13_weight_scale, layer.w2_weight_scale, pw, pi,
                    alpha=moe.swiglu_alpha, beta=moe.swiglu_beta,
                    limit=moe.swiglu_limit, expert_map=None, w13_layout='interleaved', inplace=True)
                if result.data_ptr() != px.data_ptr():
                    raise RuntimeError('Bolt did not reuse owned packed input')
                return result
            return run_received_local_routes(hidden, ids, weights, counts, local_expert,
                                             alignment=1)
        if self.routed_experts.quant_method.quant_config.routed_backend == "deepgemm_hipc_v1":
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("aligned MiniMax MoE currently requires eager execution")
            weights, ids = select(hidden_states, router_logits)
            executor, packed = _get_aligned_moe(self.routed_experts, hidden_states)
            # Static PCP padding is represented by physical ID -1. Router
            # weights on these rows are not necessarily zero; restore the same
            # invalid-route contract as dispatch_routed before the strict guard.
            executor_options = {"expert_weights": packed}
            if shared_intermediate is not None:
                executor_options["before_combine"] = lambda: self._launch_shared_down(shared_intermediate)
            result = executor(hidden_states, ids.to(torch.int64).contiguous(),
                              weights.masked_fill(ids < 0, 0).contiguous(),
                              **executor_options)
        else:
            result = dispatch_routed(hidden_states, router_logits, select, experts,
                                     _get_deepep_buffer(get_ep_group(), hidden_states),
                                     ep_rank=group.rank_in_group, recv_id_abi='local_minus_one_v1',
                                     physical_experts=self.routed_experts.global_num_experts)
        if self._shared_experts is not None:
            if shared_overlapping:
                try:
                    self._shared_experts.wait()
                except BaseException:
                    if self._hcu_shared_pending:
                        self._hcu_shared_poisoned = True
                    raise
            shared_result = self._shared_experts.output
            if self._hcu_shared_pending:
                shared_result.record_stream(torch.cuda.current_stream(shared_result.device))
                self._hcu_shared_pending = False
            return shared_result, result
        return result
