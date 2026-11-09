# SPDX-License-Identifier: Apache-2.0
"""GLM5Next-only CP-head weights; never change the global TP process group.

The underlying vLLM linears are deliberately TP-disabled and allocate their
local dimensions directly. Their ordinary loaders and HCU NN-layout lifecycle
still run, after a CP-aware loader has selected the checkpoint slice.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm_hcu.v1.glm53_kda_cp_config import KdaHeadParallelPlan


def head_slice(weight, axis: int, size: int, rank: int):
    if size <= 0 or not 0 <= rank < size:
        raise ValueError(f"invalid KDA head group size/rank: {size}/{rank}")
    if weight.shape[axis] % size:
        raise ValueError(
            f"KDA checkpoint axis {axis} of {tuple(weight.shape)} "
            f"is not divisible by head group size {size}"
        )
    width = weight.shape[axis] // size
    return weight.narrow(axis, rank * width, width)


def head_weight_loader(axis: int, size: int, rank: int, *, a_log=False):
    def load(param, loaded_weight):
        if a_log and loaded_weight.ndim == 1:
            loaded_weight = loaded_weight.view(1, 1, -1, 1)
        selected = head_slice(loaded_weight, axis, size, rank)
        if selected.shape != param.shape:
            raise ValueError(
                "KDA CP-head parameter shape mismatch: "
                f"selected={tuple(selected.shape)}, target={tuple(param.shape)}"
            )
        default_weight_loader(param, selected)

    return load


def merged_head_slice(weight, shard_id, full_sizes, size, rank):
    """Q/K/V/b are sharded; f_a/g_a remain replicated, including fused loads."""
    ids = (
        tuple(range(len(full_sizes))) if shard_id is None
        else (shard_id,) if isinstance(shard_id, int)
        else tuple(shard_id)
    )
    if not ids or len(set(ids)) != len(ids) or any(i not in range(6) for i in ids):
        raise ValueError(f"invalid GLM5Next merged checkpoint shard {shard_id!r}")
    sizes = [full_sizes[i] for i in ids]
    if weight.shape[0] != sum(sizes):
        raise ValueError(
            f"KDA merged checkpoint rows {weight.shape[0]} != {sum(sizes)} "
            f"for shards {ids}"
        )
    selected = [
        part if i in (4, 5) else head_slice(part, 0, size, rank)
        for i, part in zip(ids, weight.split(sizes, dim=0))
    ]
    return selected[0] if len(selected) == 1 else torch.cat(selected, dim=0)


class KdaCpColumnLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_size, *, head_size, head_rank, **kwargs):
        if output_size % head_size:
            raise ValueError("KDA column output must divide the head group")
        self.head_size, self.head_rank = head_size, head_rank
        super().__init__(
            input_size, output_size // head_size, disable_tp=True, **kwargs
        )

    def weight_loader(self, param, loaded_weight):
        return super().weight_loader(
            param, head_slice(loaded_weight, 0, self.head_size, self.head_rank)
        )

    def weight_loader_v2(self, param, loaded_weight):
        return super().weight_loader_v2(
            param, head_slice(loaded_weight, 0, self.head_size, self.head_rank)
        )


class KdaCpMergedLinear(MergedColumnParallelLinear):
    def __init__(self, input_size, output_sizes, *, head_size, head_rank, **kwargs):
        if len(output_sizes) != 6:
            raise ValueError("GLM5Next merged projection requires six shards")
        if any(output_sizes[i] % head_size for i in range(4)):
            raise ValueError("KDA merged output must divide the head group")
        self.head_size, self.head_rank = head_size, head_rank
        self.full_output_sizes = tuple(output_sizes)
        local_sizes = [
            n if i in (4, 5) else n // head_size
            for i, n in enumerate(output_sizes)
        ]
        super().__init__(input_size, local_sizes, disable_tp=True, **kwargs)

    def _select(self, loaded_weight, shard_id):
        return merged_head_slice(
            loaded_weight, shard_id, self.full_output_sizes,
            self.head_size, self.head_rank,
        )

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        # Dispatch full/tuple checkpoint loads ourselves. The parent recurses
        # through self.weight_loader, which would CP-slice an already sliced
        # tensor a second time (and rebase tuple shard IDs incorrectly).
        return self._load(param, loaded_weight, loaded_shard_id, v2=False)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        return self._load(param, loaded_weight, loaded_shard_id, v2=True)

    def _load(self, param, weight, shard_id, *, v2):
        ids = (
            tuple(range(6)) if shard_id is None
            else (shard_id,) if isinstance(shard_id, int) else tuple(shard_id)
        )
        self.validate_shard_id(shard_id)
        selected = self._select(weight, shard_id)
        sizes = [self.output_sizes[i] for i in ids]
        loader = super().weight_loader_v2 if v2 else super().weight_loader
        for i, part in zip(ids, selected.split(sizes, dim=0)):
            loader(param, part, i)


class KdaCpRowLinear(RowParallelLinear):
    def __init__(self, input_size, output_size, *, head_size, head_rank, **kwargs):
        if input_size % head_size:
            raise ValueError("KDA row input must divide the head group")
        self.head_size, self.head_rank = head_size, head_rank
        super().__init__(
            input_size // head_size, output_size, disable_tp=True,
            reduce_results=False, **kwargs,
        )

    def weight_loader(self, param, loaded_weight):
        return super().weight_loader(
            param, head_slice(loaded_weight, 1, self.head_size, self.head_rank)
        )

    def weight_loader_v2(self, param, loaded_weight):
        return super().weight_loader_v2(
            param, head_slice(loaded_weight, 1, self.head_size, self.head_rank)
        )


def initialize_cp_kda(layer, config, vllm_config, prefix, kda, head_size):
    """Construct the audited native KDA fields, keeping its forward/kernels."""
    from vllm.distributed import get_pcp_group

    group = get_pcp_group()
    plan = KdaHeadParallelPlan("pcp", head_size)
    rank = plan.rank_in_group(group)
    local_num_heads = plan.local_num_heads(config.linear_num_heads)
    saved_quant_config = vllm_config.quant_config
    try:
        # Native KDA intentionally remains unquantized. Do not use the MoE
        # channel-FP8 loader for these projections.
        vllm_config.quant_config = None
        kda.GatedDeltaNetAttention.__init__(layer, config, vllm_config, prefix)
    finally:
        vllm_config.quant_config = saved_quant_config

    layer.head_parallel_size = head_size
    layer.head_parallel_rank = rank
    layer._hcu_kda_head_parallel_plan = plan
    layer._hcu_kda_cp_head_sharded = True
    layer._hcu_kda_cp_group = group
    layer.head_dim = config.linear_head_dim
    layer.num_heads = config.linear_num_heads
    layer.conv_size = config.linear_conv_kernel_dim
    layer.local_num_heads = local_num_heads
    projection_size = layer.head_dim * layer.num_heads
    layer.local_projection_size = projection_size // head_size
    common = dict(head_size=head_size, head_rank=rank, bias=False)
    linear = dict(common, quant_config=layer.quant_config)
    layer.in_proj_qkvbfg_a = KdaCpMergedLinear(
        layer.hidden_size,
        [projection_size] * 3 + [layer.num_heads, layer.head_dim, layer.head_dim],
        prefix=f"{prefix}.in_proj_qkvbfg_a", **linear,
    )
    layer.f_b_proj = KdaCpColumnLinear(
        layer.head_dim, projection_size, prefix=f"{prefix}.f_b_proj", **linear
    )
    layer.g_b_proj = KdaCpColumnLinear(
        layer.head_dim, projection_size, prefix=f"{prefix}.g_b_proj", **linear
    )
    layer.dt_bias = nn.Parameter(
        torch.empty(layer.local_projection_size, dtype=torch.float32)
    )
    kda.set_weight_attrs(
        layer.dt_bias, {"weight_loader": head_weight_loader(0, head_size, rank)}
    )
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = KdaCpColumnLinear(
            layer.conv_size, projection_size, params_dtype=torch.float32,
            prefix=f"{prefix}.{name}", **common,
        )
        conv.weight.data = conv.weight.data.unsqueeze(1)
        setattr(layer, name, conv)
    layer._merged_conv_weight = None
    layer.A_log = nn.Parameter(
        torch.empty(1, 1, layer.local_num_heads, 1, dtype=torch.float32)
    )
    kda.set_weight_attrs(
        layer.A_log,
        {"weight_loader": head_weight_loader(2, head_size, rank, a_log=True)},
    )
    layer.o_norm = kda.FusedRMSNormGated(layer.head_dim, activation="sigmoid")
    layer.o_proj = KdaCpRowLinear(
        projection_size, layer.hidden_size, prefix=f"{prefix}.o_proj", **linear
    )
    compilation = vllm_config.compilation_config
    if prefix in compilation.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation.static_forward_context[prefix] = layer
    layer.kda_safe_gate = True
    layer.kda_lower_bound = config.linear_lower_bound
    layer._conv_state_dim_first = kda.is_conv_state_dim_first()
