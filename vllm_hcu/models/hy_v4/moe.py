# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Dense feed-forward and fused MoE blocks for HY V4 on HCU."""

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


def _require_supported_moe_backend(
    vllm_config: VllmConfig,
    model_config: PretrainedConfig,
) -> None:
    backend = vllm_config.kernel_config.moe_backend
    if backend not in ("triton", "aiter"):
        raise RuntimeError(
            "HY V4 FP8 W8A8 requires an explicit --moe-backend triton or "
            "--moe-backend aiter route; "
            f"got {backend!r}."
        )
    swiglu_limit = float(getattr(model_config, "swiglu_limit", 0) or 0)
    if backend == "aiter" and swiglu_limit > 0:
        raise RuntimeError(
            "HY V4 cannot use the installed AITER MoE-C backend because its "
            f"SILU path does not preserve swiglu_limit={swiglu_limit:g}; "
            "use --moe-backend triton for accuracy."
        )


class HYV4FeedForward(nn.Module):
    """Dense SwiGLU block used by dense layers and shared experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        activated = self.act_fn(gate_up)
        output, _ = self.down_proj(activated)
        return output


class HYV4MoEFused(nn.Module):
    """HY V4 sigmoid-routed MoE backed by an explicit HCU expert backend."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
        vllm_config: VllmConfig | None = None,
    ) -> None:
        super().__init__()
        if vllm_config is None:
            vllm_config = get_current_vllm_config()
        _require_supported_moe_backend(vllm_config, config)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        if vllm_config.parallel_config.enable_expert_parallel:
            self.ep_rank = get_ep_group().rank_in_group
            self.ep_size = self.ep_group.size()
        else:
            # vLLM creates an EP communication group for every MoE model even
            # when expert parallelism is disabled.  In pure TP that group spans
            # all TP ranks, but every rank still owns all experts and shards
            # only the intermediate dimension.
            self.ep_rank = 0
            self.ep_size = 1
        self.n_routed_experts = config.num_experts
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = (
            self.n_logical_experts + self.n_redundant_experts
        )
        if self.n_physical_experts % self.ep_size != 0:
            raise ValueError(
                f"Physical experts {self.n_physical_experts} must be divisible "
                f"by expert parallel size {self.ep_size}."
            )
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: HYV4FeedForward | None
        if config.num_shared_experts > 0:
            self.shared_experts = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.expert_hidden_dim * config.num_shared_experts
                ),
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )
        else:
            self.shared_experts = None

        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )
        raw_swiglu_limit = getattr(config, "swiglu_limit", 0)
        swiglu_limit = (
            float(raw_swiglu_limit)
            if raw_swiglu_limit and float(raw_swiglu_limit) > 0
            else None
        )
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_hidden_dim,
            renormalize=config.route_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            scoring_func="sigmoid",
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=getattr(config, "router_scaling_factor", 1.0),
            e_score_correction_bias=self.expert_bias,
            n_shared_experts=config.num_shared_experts,
            shared_experts=self.shared_experts,
            swiglu_limit=swiglu_limit,
            router_logits_dtype=torch.float32,
        )
        self.prefix = prefix
        if swiglu_limit is not None:
            logger.debug_once(
                "HYV4MoEFused: routed expert swiglu_limit=%.1f",
                swiglu_limit,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return final_hidden_states.view(original_shape)


__all__ = ["HYV4FeedForward", "HYV4MoEFused"]
