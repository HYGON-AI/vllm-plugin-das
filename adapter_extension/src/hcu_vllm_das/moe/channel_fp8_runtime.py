# SPDX-License-Identifier: Apache-2.0
"""Owned Channel-FP8 DeepGEMM HT/LL experts and physical-layout contract."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_scatter, ep_gather
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceDelegate, TopKWeightAndReduceNoOP
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
from vllm.platforms import current_platform


class ChannelFP8RuntimeError(RuntimeError):
    """An owned Channel-FP8 ABI or layout contract failed."""


def _witness(t):
    return (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes(), t.data_ptr(),
            t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.device, t.dtype)


@dataclass(frozen=True)
class ChannelFP8PackedWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor
    layout: str
    logical_shape: tuple[int, int, int, int]  # E, 2N, N, K, before packing
    witnesses: tuple[tuple, ...]

    def validate(self, w1, w2, s1, s2):
        if tuple(_witness(t) for t in (w1, w2, s1, s2)) != self.witnesses:
            raise ChannelFP8RuntimeError("Channel-FP8 installed storage/shape/stride/device/dtype witness mismatch")


def _provider_pack(weight, layout):
    import deepgemm
    name = f"marlin_fp8_{layout}_weight"
    provider = getattr(deepgemm, name, None)
    if not callable(provider):
        raise ChannelFP8RuntimeError(f"deepgemm provider lacks {name}")
    return provider(weight)


def _physical_shape(e, n, k, rank):
    if k % 64 or n % (32 if rank == 7 else 16):
        raise ChannelFP8RuntimeError("Channel-FP8 logical N/K do not fit provider tile layout")
    if rank == 6:
        return (e, k // 64, n // 16, 4, 16, 16)
    return (e, k // 64, n // 32, 2, 4, 16, 16)


def pack_channel_weights(w1, w2, s1, s2, *, layout, prior=None, max_tokens=None, num_dispatchers=None):
    if layout not in ("contiguous", "masked"):
        raise ChannelFP8RuntimeError("unknown Channel-FP8 DeepGEMM layout")
    if prior is not None:
        if prior.layout != layout:
            raise ChannelFP8RuntimeError("Channel-FP8 installed weight layout mismatch")
        prior.validate(w1, w2, s1, s2)
        return prior
    if not all(isinstance(t, torch.Tensor) for t in (w1, w2, s1, s2)):
        raise ChannelFP8RuntimeError("Channel-FP8 weights/scales must be tensors")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ChannelFP8RuntimeError("Channel-FP8 raw weights must be rank-3 [E,N,K]")
    e, two_n, k = map(int, w1.shape)
    n = int(w2.shape[-1])
    if e <= 0 or n <= 0 or k <= 0 or tuple(w2.shape) != (e, k, n) or two_n != 2 * n:
        raise ChannelFP8RuntimeError("Channel-FP8 logical expert dimensions disagree")
    if any(t.dtype != torch.float8_e4m3fn for t in (w1, w2)):
        raise ChannelFP8RuntimeError("Channel-FP8 weights must be float8_e4m3fn")
    for scale, width in ((s1, two_n), (s2, k)):
        if tuple(scale.shape) not in ((e, width, 1), (e, width)) or scale.dtype != torch.float32:
            raise ChannelFP8RuntimeError("Channel-FP8 scales must be FP32 [E,N,1] or [E,N]")
    if any(t.device != w1.device or not t.is_contiguous() for t in (w1, w2, s1, s2)):
        raise ChannelFP8RuntimeError("Channel-FP8 raw weights/scales must be contiguous on one device")
    if layout == "masked" and (not isinstance(max_tokens, int) or max_tokens <= 0 or not isinstance(num_dispatchers, int) or num_dispatchers <= 0):
        raise ChannelFP8RuntimeError("masked Channel-FP8 requires positive token/dispatcher capacities")
    _physical_shape(e, two_n, k, 6)
    _physical_shape(e, k, n, 6)
    p1, p2 = _provider_pack(w1, layout), _provider_pack(w2, layout)
    if not all(isinstance(t, torch.Tensor) for t in (p1, p2)):
        raise ChannelFP8RuntimeError("Channel-FP8 packer must return tensors")
    rank = p1.ndim
    if rank not in (6, 7) or p2.ndim != rank:
        raise ChannelFP8RuntimeError("Channel-FP8 provider requires matching rank-6 or rank-7 layouts")
    for packed, width, inner in ((p1, two_n, k), (p2, k, n)):
        if (tuple(packed.shape) != _physical_shape(e, width, inner, rank) or packed.dtype != w1.dtype or packed.device != w1.device or not packed.is_contiguous()):
            raise ChannelFP8RuntimeError("Channel-FP8 provider physical shape/stride/device/dtype mismatch")
    s1, s2 = s1.view(e, two_n), s2.view(e, k)
    tensors = (p1, p2, s1, s2)
    return ChannelFP8PackedWeights(*tensors, layout, (e, two_n, n, k), tuple(_witness(t) for t in tensors))


def validate_packed_layout(weight, packed):
    if _witness(weight) not in packed.witnesses[:2]:
        raise ChannelFP8RuntimeError("Channel-FP8 packed weight storage/layout witness mismatch")


def select_channel_backend(parallel, *, requested=None):
    if getattr(parallel, "all2all_backend", None) == "allgather_reducescatter":
        # Owned DSpark draft: AG-RS gather + contiguous DeepGEMM, no RocSHMEM.
        if requested not in (None, "auto", "deep_gemm"):
            raise ChannelFP8RuntimeError("Channel-FP8 requires the owned deep_gemm backend")
        return "agrs"
    ht, ll = bool(parallel.use_deepep_ht_kernels), bool(parallel.use_deepep_ll_kernels)
    if ht and ll:
        raise ChannelFP8RuntimeError("DeepEP HT and LL Channel-FP8 routes overlap")
    if requested not in (None, "auto", "deep_gemm"):
        raise ChannelFP8RuntimeError("Channel-FP8 requires the owned deep_gemm backend")
    if ht:
        return "ht"
    if ll:
        return "ll"
    raise ChannelFP8RuntimeError("Channel-FP8 DeepGEMM requires DeepEP HT/LL or owned AG-RS")


def deepgemm_contiguous(*args, **kwargs):
    import deepgemm
    return deepgemm.m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs)


def deepgemm_masked(*args, **kwargs):
    import deepgemm
    return deepgemm.m_grouped_fp8_gemm_nt_masked(*args, **kwargs)


def fuse_silu_mul_fp8_quant(*args, **kwargs):
    from lightop.activation import fuse_silu_mul_fp8_quant as provider
    return provider(*args, **kwargs)


def fuse_silu_mul_fp8_quant_ep(*args, **kwargs):
    from lightop.activation import fuse_silu_mul_fp8_quant_ep as provider
    return provider(*args, **kwargs)


def _token_scale(scale, shape):
    if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32 or not scale.is_contiguous() or tuple(scale.shape) not in (tuple(shape), (*shape, 1))):
        raise ChannelFP8RuntimeError("Channel-FP8 token scale requires contiguous FP32 with one scale per token")
    return scale.view(shape)


def compute_aligned_M(M, num_topk, local_num_experts, alignment, expert_tokens_meta):
    counts_cpu = getattr(expert_tokens_meta, "expert_num_tokens_cpu", None)
    if counts_cpu is not None:
        return int((((counts_cpu.to(torch.int64) + alignment - 1) // alignment) * alignment).sum())
    bound = M * num_topk + min(M * num_topk, local_num_experts) * (alignment - 1)
    return (bound + alignment - 1) // alignment * alignment


class ChannelFP8ContiguousExperts(mk.FusedMoEExpertsModular):
    ALIGNMENT = 256
    WEIGHT_LAYOUT = "contiguous"

    def __init__(self, moe_config, quant_config, max_num_tokens=None, num_dispatchers=None):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        self.packed = None

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        return current_platform.is_rocm() and current_platform.device_name == "hcu"

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return (weight_key, activation_key) == (kFp8StaticChannelSym, kFp8DynamicTokenSym)

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(parallel):
        # DeepEP HT target, or owned DSpark AG-RS draft (neither HT nor LL flags).
        if getattr(parallel, "all2all_backend", None) == "allgather_reducescatter":
            return bool(getattr(parallel, "use_ep", True))
        return parallel.use_deepep_ht_kernels and not parallel.use_deepep_ll_kernels

    def supports_expert_map(self):
        return True

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        if self.packed is None:
            raise ChannelFP8RuntimeError("Channel-FP8 weights are not installed")
        self.packed.validate(w1, w2, self.w1_scale, self.w2_scale)
        e, two_n, n, k = self.packed.logical_shape
        if a1.shape[-1] != k or topk_ids.ndim != 2:
            raise ChannelFP8RuntimeError("Channel-FP8 activation/routing logical shape mismatch")
        return e, a1.shape[-2], two_n, k, topk_ids.shape[1]

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts, expert_tokens_meta, activation):
        aligned = compute_aligned_M(M=M, num_topk=topk, local_num_experts=local_num_experts, alignment=self.ALIGNMENT, expert_tokens_meta=expert_tokens_meta)
        return (aligned, K), (aligned, max(N, K)), (M, K)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        e, m, two_n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if activation != MoEActivation.SILU or hidden_states.dtype != torch.float8_e4m3fn:
            raise ChannelFP8RuntimeError("Channel-FP8 contiguous experts require SiLU and FP8 input")
        scale = _token_scale(a1q_scale, (m, 1))
        if m == 0:
            return
        # DeepEP HT supplies counts; owned AG-RS draft does not — mirror CT and
        # derive local expert occupancy from topk_ids + expert_map.
        counts = getattr(expert_tokens_meta, "expert_num_tokens", None)
        if counts is None:
            from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens

            counts = count_expert_num_tokens(topk_ids, e, expert_map)
        if counts.dtype != torch.int32:
            counts = counts.to(torch.int32)
        if not counts.is_contiguous():
            counts = counts.contiguous()
        aligned = compute_aligned_M(M=m, num_topk=topk, local_num_experts=e, alignment=self.ALIGNMENT, expert_tokens_meta=expert_tokens_meta)
        q = _resize_cache(workspace13.view(dtype=hidden_states.dtype), (aligned, k))
        scales = torch.zeros((aligned, 1), dtype=torch.float32, device=q.device)
        ids = torch.full((aligned,), -1, dtype=torch.int32, device=q.device)
        inverse = torch.empty(topk_ids.shape, dtype=torch.int32, device=q.device)
        starts = torch.empty((e,), dtype=torch.int32, device=q.device)
        ep_scatter(recv_x=hidden_states, recv_x_scale=scale, recv_topk=topk_ids,
                   num_recv_tokens_per_expert=counts, expert_start_loc=starts,
                   expert_map=expert_map, output_tensor=q, output_tensor_scale=scales,
                   m_indices=ids, output_index=inverse, align_m=256, block_size=k,
                   pack_ue8m0=False, use_psum_layout=False)
        gateup = _resize_cache(workspace2, (aligned, two_n))
        deepgemm_contiguous((q, scales), (w1, self.w1_scale), gateup, ids)
        q2, s2 = fuse_silu_mul_fp8_quant(gateup, fp8type=0, expert_ids=ids, limit=self.quant_config.gemm1_clamp_limit)
        s2 = _token_scale(s2, (aligned, 1))
        down = _resize_cache(workspace2, (aligned, k))
        deepgemm_contiguous((q2, s2), (w2, self.w2_scale), down, ids)
        reduction_weights = torch.ones_like(topk_weights) if apply_router_weight_on_input else topk_weights
        ep_gather(input_tensor=down, recv_topk_ids=topk_ids, recv_topk_weight=reduction_weights, input_index=inverse, expert_map=expert_map, output_tensor=output)


class ChannelFP8MaskedExperts(ChannelFP8ContiguousExperts):
    WEIGHT_LAYOUT = "masked"

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_parallel_config(parallel):
        return parallel.use_deepep_ll_kernels and not parallel.use_deepep_ht_kernels

    def supports_expert_map(self):
        return False

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts, expert_tokens_meta, activation):
        capacity = self.max_num_tokens * self.num_dispatchers
        return (0,), (local_num_experts, capacity, max(N, K)), (local_num_experts, capacity, K)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        e, capacity, two_n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if hidden_states.ndim != 3 or hidden_states.shape[0] != e or activation != MoEActivation.SILU:
            raise ChannelFP8RuntimeError("Channel-FP8 LL requires SiLU and [E,T,K] input")
        # LL prepare owns global-to-physical routing. The canonical map from
        # RoutedExperts is intentionally unused by batched expert kernels.
        if hidden_states.dtype != torch.float8_e4m3fn:
            raise ChannelFP8RuntimeError("Channel-FP8 LL requires FP8 input")
        counts = getattr(expert_tokens_meta, "expert_num_tokens", None)
        if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int32 or tuple(counts.shape) != (e,) or not counts.is_contiguous() or counts.device != hidden_states.device):
            raise ChannelFP8RuntimeError("Channel-FP8 LL metadata requires contiguous int32 counts [E]")
        scale = _token_scale(a1q_scale, (e, capacity))
        if capacity == 0:
            return
        expected_m = min(capacity, (topk_ids.shape[0] * self.num_dispatchers * topk + global_num_experts) // global_num_experts)
        gateup = _resize_cache(workspace2, (e, capacity, two_n))
        deepgemm_masked((hidden_states, scale), (w1, self.w1_scale), gateup, counts, expected_m)
        q2, s2 = fuse_silu_mul_fp8_quant_ep(gateup, fp8type=0, tokens_per_expert=counts, limit=self.quant_config.gemm1_clamp_limit)
        s2 = _token_scale(s2, (e, capacity))
        deepgemm_masked((q2.view(e, capacity, two_n // 2), s2), (w2, self.w2_scale), output, counts, expected_m)
