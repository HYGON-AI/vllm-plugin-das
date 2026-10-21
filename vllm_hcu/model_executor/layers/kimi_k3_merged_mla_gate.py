# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Plugin-owned mixed replicated/sharded QKV-A plus MLA gate projection."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from torch.nn import Parameter

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter

ENV_NAME = "VLLM_HCU_KIMI_FUSED_MLA_GATE"


def enabled() -> bool:
    return os.environ.get(ENV_NAME, "1").lower() in ("1", "true")


class KimiK3MergedQKVGateLinear(MergedColumnParallelLinear):
    """Replicated Q/KV latent shards followed by a TP-local output gate."""

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        total_num_heads: int,
        v_head_dim: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        super().__init__(
            input_size=hidden_size,
            output_sizes=[
                q_lora_rank * tp_size,
                (kv_lora_rank + qk_rope_head_dim) * tp_size,
                total_num_heads * v_head_dim,
            ],
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _load_with_replicated_latents(
        self,
        loader: Callable[..., None],
        param: Parameter | BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None,
    ) -> None:
        if not isinstance(loaded_shard_id, int) or loaded_shard_id >= 2:
            loader(param, loaded_weight, loaded_shard_id)
            return
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        self.tp_rank = 0
        if param_tp_rank is not None:
            param.tp_rank = 0
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        self._load_with_replicated_latents(
            super().weight_loader, param, loaded_weight, loaded_shard_id
        )

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        self._load_with_replicated_latents(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )
