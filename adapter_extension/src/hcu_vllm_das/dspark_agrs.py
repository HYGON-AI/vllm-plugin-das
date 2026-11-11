# SPDX-License-Identifier: Apache-2.0
"""Instance-owned V41 CT draft AG-RS; no target/group-global replacements.

Native DSpark constructs from a shallow private config. Native AG-RS uses the
actual draft forward DP sizes; reviewed native PF functions resolve an owned
manager only within their private instance namespaces.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
from types import FunctionType, MethodType
import torch
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, get_pcp_group
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import MoEPrepareAndFinalizeNaiveDPEPModular
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

_BACKEND = 'allgather_reducescatter'
_OWNER = 'hcu_dsv41_dspark_agrs_v1'
_META_KEY = '_hcu_dsv41_dspark_agrs'


def _metadata(cfg):
    extra = getattr(cfg, 'additional_config', None)
    return extra.get(_META_KEY) if isinstance(extra, dict) else None


def _capacity_tuple(capacity):
    return [
        capacity.query_per_request,
        capacity.query_rows,
        capacity.verify_rows,
        capacity.budget,
    ]


def _capacity_from_metadata(cfg):
    metadata = _metadata(cfg)
    values = metadata.get('capacity') if isinstance(metadata, dict) else None
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    if any(type(value) is not int for value in values):
        return None
    return TokenCapacity(*values)


@dataclass(frozen=True)
class TokenCapacity:
    query_per_request: int
    query_rows: int
    verify_rows: int
    budget: int


def token_capacity(cfg):
    spec, scheduler = cfg.speculative_config, cfg.scheduler_config
    n, reqs, budget = spec.num_speculative_tokens, scheduler.max_num_seqs, scheduler.max_num_batched_tokens
    if any(type(x) is not int or x <= 0 for x in (n, reqs, budget)):
        raise ValueError('DSpark requires positive integer n/requests/token budget')
    anchor = getattr(spec.draft_model_config.hf_config, 'sample_from_anchor', True)
    if type(anchor) is not bool:
        raise ValueError('DSpark sample_from_anchor must be bool')
    query = n if anchor else n + 1
    result = TokenCapacity(query, reqs * query, reqs * (n + 1), budget)
    if max(result.query_rows, result.verify_rows) > budget:
        raise ValueError(f'DSpark capacity exceeds original budget: query={result.query_rows}, verify={result.verify_rows}, budget={budget}')
    return result


def is_owned_draft(cfg):
    metadata = _metadata(cfg)
    return isinstance(metadata, dict) and metadata.get('owner') == _OWNER


def _owned_quant_kind(cfg):
    from .quantization.ct_w4a8 import HcuDSV41CTW4A8Config
    from .quantization.dsv41_channel_fp8 import HcuDSV41ChannelFP8Config

    if isinstance(cfg.quant_config, HcuDSV41CTW4A8Config):
        return "ct_w4a8"
    if isinstance(cfg.quant_config, HcuDSV41ChannelFP8Config):
        return "channel_fp8"
    return None


def draft_config(cfg):
    """Copy only construction policy, retaining native live registration tables."""
    # Graph loader may call draft_config, then re-enter model construction with
    # the already-owned AG-RS copy. Stay idempotent instead of re-checking the
    # target's DeepEP LL topology against the private backend.
    if is_owned_draft(cfg):
        if token_capacity(cfg) != _capacity_from_metadata(cfg):
            raise RuntimeError("DSpark token capacity changed after construction")
        return cfg
    p, spec = cfg.parallel_config, cfg.speculative_config
    kind = _owned_quant_kind(cfg)
    if kind is None:
        raise ValueError(
            "owned draft AG-RS requires HCU CT W4A8 or Channel-FP8 quantization"
        )
    if spec is None or spec.method != 'dspark' or spec.draft_model_config.hf_config.model_type not in ('deepseek_v41', 'deepseek_v41_text'):
        raise ValueError('owned draft AG-RS requires native V41 DSpark')
    if any(getattr(p, name) != 1 for name in ('tensor_parallel_size', 'pipeline_parallel_size', 'prefill_context_parallel_size', 'decode_context_parallel_size')):
        raise ValueError('DSpark AG-RS requires TP1/PP1/PCP1/DCP1')
    if type(p.data_parallel_size) is not int or p.data_parallel_size <= 1 or not p.enable_expert_parallel:
        raise ValueError('DSpark AG-RS requires actual DP+EP membership')
    if (cfg.lora_config is not None or p.enable_eplb or p.enable_elastic_ep or p.enable_dbo
            or cfg.kernel_config.monokernel):
        raise ValueError('DSpark AG-RS does not support LoRA/EPLB/ElasticEP/DBO/monokernel')
    if not cfg.use_v2_model_runner:
        raise ValueError('DSpark AG-RS requires the V2 model runner')
    if not cfg.model_config.enforce_eager or not spec.draft_model_config.enforce_eager:
        from .dspark_graph import native_decode_dspark, validate_graph_capacity

        if not native_decode_dspark(cfg):
            raise ValueError(
                'DSpark AG-RS Graph requires the native V41 DP8/EP8 target; '
                'unsupported Graph topology'
            )
        validate_graph_capacity(cfg)
        token_capacity(cfg)
    if cfg.kernel_config.moe_backend != 'deep_gemm':
        raise ValueError('DSpark AG-RS requires the owned DeepGEMM expert path')
    capacity = token_capacity(cfg)
    result = copy.copy(cfg)
    result.parallel_config = copy.copy(p)
    result.parallel_config.all2all_backend = _BACKEND
    extra = getattr(cfg, "additional_config", None)
    if extra is None:
        extra = {}
    elif not isinstance(extra, dict):
        raise ValueError("DSpark AG-RS requires a dict additional_config owner store")
    result.additional_config = dict(extra)
    result.additional_config[_META_KEY] = {
        'owner': _OWNER,
        'capacity': _capacity_tuple(capacity),
        'quant': kind,
    }
    return result


class HcuDSparkAgRsManager(AgRsAll2AllManager):
    """Native variable AG-RS over borrowed, ordered DP/EP groups; no ownership."""
    def __init__(self, capacity):
        self._dp, self._ep, self._tp, self._pcp = get_dp_group(), get_ep_group(), get_tp_group(), get_pcp_group()
        self._capacity = capacity
        self._ranks = tuple(self._dp.ranks)
        self.check_health()
        super().__init__(self._ep.cpu_group)
        if self.world_size != self._dp.world_size or self.rank != self._dp.rank_in_group or not self.use_ep:
            raise RuntimeError('DSpark AG-RS manager EP/DP coordinates disagree')

    def check_health(self):
        if (get_dp_group() is not self._dp or get_ep_group() is not self._ep
                or get_tp_group() is not self._tp or get_pcp_group() is not self._pcp
                or self._tp.world_size != 1 or self._pcp.world_size != 1
                or tuple(self._dp.ranks) != self._ranks or tuple(self._ep.ranks) != self._ranks
                or self._dp.world_size != len(self._ranks) or self._ep.world_size != len(self._ranks)
                or self._dp.rank_in_group != self._ep.rank_in_group):
            raise RuntimeError('DSpark AG-RS borrowed group identity/order changed')

    def _get_comm_group(self, is_sequence_parallel):
        self.check_health()
        if is_sequence_parallel:
            raise ValueError('DSpark AG-RS requires non-SP local query rows')
        return self._dp

    def _get_sizes(self, num_local_tokens, comm_group):
        sizes = super()._get_sizes(num_local_tokens, comm_group)
        if len(sizes) != self.world_size or any(type(n) is not int or n < 0 or n > self._capacity for n in sizes):
            raise ValueError('DSpark current query DP sizes exceed checked local capacity')
        return sizes

    def destroy(self):
        raise RuntimeError('DSpark AG-RS does not own its borrowed process groups')


_REVIEWED = {
    'prepare': 'f4069809c8fe534dd02304aad504d2bf795da5da1823fed424cb8acf89bdd8ca',
    'finalize': '9cddd21280de8991d002d51941ab1e12337949867752631737e8d9b2d99b2bde',
    '_allocate_combine_input': 'f01d608b957578f2c95c64c82a2c08890b8d94a15c1d3e08a676617fe6d6b1c2',
}


def _bind(instance, name, manager):
    function = getattr(MoEPrepareAndFinalizeNaiveDPEPModular, name)
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if actual != _REVIEWED[name]:
        raise RuntimeError(f'DSpark native modular PF source changed: {name}: {actual}')
    namespace = dict(function.__globals__, get_ep_group=lambda: manager)
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__, clone.__annotations__ = function.__kwdefaults__, function.__annotations__
    return MethodType(clone, instance)


class HcuCTAgRsPrepareAndFinalize(MoEPrepareAndFinalizeNaiveDPEPModular):
    def __init__(self, manager):
        super().__init__(is_sequence_parallel=False, num_dispatchers=manager.world_size)
        self.manager = manager
        self._native_prepare = _bind(self, 'prepare', manager)
        self.finalize = _bind(self, 'finalize', manager)
        self._allocate_combine_input = _bind(self, '_allocate_combine_input', manager)

    def prepare(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                apply_router_weight_on_input, quant_config, defer_input_quant=False):
        self.manager.check_health()
        if (defer_input_quant or quant_config.quant_dtype != torch.int8
                or not quant_config.per_act_token_quant or quant_config.block_shape is not None):
            raise ValueError('DSpark CT AG-RS requires owned dynamic token INT8')
        if (a1.ndim != 2 or a1.dtype != torch.bfloat16 or a1.shape[0] > self.manager._capacity
                or topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2
                or topk_ids.shape[0] != a1.shape[0]
                # v0.33 core routers emit int32 topk_ids; the legacy v0.29
                # plugin path used int64. Accept both, reject anything else.
                or topk_ids.dtype not in (torch.int32, torch.int64)
                or topk_weights.dtype != torch.float32):
            import logging
            logging.getLogger(__name__).error(
                'DSpark AG-RS contract violation: a1.ndim=%s a1.dtype=%s a1.shape=%s capacity=%s '
                'topk_ids.shape=%s topk_ids.ndim=%s topk_ids.dtype=%s '
                'topk_weights.shape=%s topk_weights.dtype=%s',
                a1.ndim, a1.dtype, tuple(a1.shape), self.manager._capacity,
                tuple(topk_ids.shape), topk_ids.ndim, topk_ids.dtype,
                tuple(topk_weights.shape), topk_weights.dtype)
            raise ValueError('DSpark AG-RS local input/routes violate the checked wire contract')
        # Gather BF16 first so even a zero-query rank reaches native collectives;
        # INT8 token quantization is then independent of unequal DP row counts.
        x, _, meta, ids, weights = self._native_prepare(a1, topk_weights, topk_ids,
            num_experts, expert_map, apply_router_weight_on_input, quant_config, defer_input_quant=True)
        if not x.shape[0]:
            q, scales = x.to(torch.int8), x.new_empty((0, 1), dtype=torch.float32)
        else:
            q, scales = moe_kernel_quantize_input(x, None, quant_dtype=torch.int8,
                per_act_token_quant=True, block_shape=None)
        if scales.ndim == 1:
            scales = scales.unsqueeze(-1)
        if q.dtype != torch.int8 or scales.dtype != torch.float32 or scales.device != q.device or scales.shape != (q.shape[0], 1):
            raise RuntimeError('DSpark AG-RS did not produce per-token INT8 scales')
        return q, scales, meta, ids, weights


def _make_owned_manager(moe, cfg):
    if not is_owned_draft(cfg):
        raise ValueError("AG-RS preparation requires owned DSpark construction")
    if token_capacity(cfg) != _capacity_from_metadata(cfg):
        raise RuntimeError("DSpark token capacity changed after construction")
    manager = HcuDSparkAgRsManager(moe.max_num_tokens)
    actual = moe.moe_parallel_config
    if (
        actual.ep_size != manager.world_size
        or actual.dp_size != manager.dp_world_size
        or actual.ep_rank != manager.rank
    ):
        raise RuntimeError("DSpark MoE membership differs from borrowed AG-RS groups")
    return manager


def make_draft_prepare(moe, cfg):
    return HcuCTAgRsPrepareAndFinalize(_make_owned_manager(moe, cfg))


class HcuChannelFP8AgRsPrepareAndFinalize(MoEPrepareAndFinalizeNaiveDPEPModular):
    """Channel-FP8 draft AG-RS: gather BF16, then owned per-token FP8 quant."""

    def __init__(self, manager):
        super().__init__(is_sequence_parallel=False, num_dispatchers=manager.world_size)
        self.manager = manager
        self._native_prepare = _bind(self, "prepare", manager)
        self.finalize = _bind(self, "finalize", manager)
        self._allocate_combine_input = _bind(self, "_allocate_combine_input", manager)

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        self.manager.check_health()
        if (
            defer_input_quant
            or quant_config.quant_dtype != torch.float8_e4m3fn
            or not quant_config.per_act_token_quant
            or quant_config.block_shape is not None
            or quant_config.is_scale_swizzled
        ):
            raise ValueError(
                "DSpark Channel-FP8 AG-RS requires owned dynamic token FP8"
            )
        if (
            a1.ndim != 2
            or a1.dtype != torch.bfloat16
            or a1.shape[0] > self.manager._capacity
            or topk_ids.shape != topk_weights.shape
            or topk_ids.ndim != 2
            or topk_ids.shape[0] != a1.shape[0]
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_weights.dtype != torch.float32
        ):
            raise ValueError(
                "DSpark Channel-FP8 AG-RS local input/routes violate the wire contract"
            )
        x, _, meta, ids, weights = self._native_prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=True,
        )
        if not x.shape[0]:
            q = x.to(torch.float8_e4m3fn)
            scales = x.new_empty((0, 1), dtype=torch.float32)
        else:
            q, scales = moe_kernel_quantize_input(
                x,
                None,
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=True,
                block_shape=None,
            )
        if scales.ndim == 1:
            scales = scales.unsqueeze(-1)
        if (
            q.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float32
            or scales.device != q.device
            or scales.shape != (q.shape[0], 1)
        ):
            raise RuntimeError(
                "DSpark Channel-FP8 AG-RS did not produce per-token FP8 scales"
            )
        return q, scales, meta, ids, weights


def make_channel_draft_prepare(moe, cfg):
    return HcuChannelFP8AgRsPrepareAndFinalize(_make_owned_manager(moe, cfg))
