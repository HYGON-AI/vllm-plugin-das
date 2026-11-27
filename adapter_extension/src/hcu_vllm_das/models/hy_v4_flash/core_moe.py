# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core033 Flash FFN and unfused, reduction-safe latent MoE.

The routed factory owns its latent-space TP reduction. Shared experts own a
separate hidden-space TP reduction. Neither replicated up nor the final sum
performs another collective. No legacy runner or transform hooks are used.
"""

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward as CoreHYV4FeedForward
from vllm.models.hy_v4.nvidia.moe import HYV4MoEFused as CoreHYV4MoEFused


def _check_quantization(config, quant_config, *, for_moe=False):
    from hcu_vllm_das.models.hy_v4_flash.quantization import (
        prepare_flash_quant_config,
    )

    return prepare_flash_quant_config(config, quant_config, for_moe=for_moe)


class HYV4FlashFeedForward(CoreHYV4FeedForward):
    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        quant_config = _check_quantization(config, quant_config)
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        return self.post_fc2_norm(result) if self.post_fc2_norm is not None else result


class HYV4FlashMoE(CoreHYV4MoEFused):
    """Keep Core loader identity/names, but compose latent experts explicitly.

    Latent execution currently admits ordinary TP only. EP/DP/PCP/SP and EPLB
    require an independently verified shared/routed ownership contract and are
    rejected rather than silently mixing reduction domains.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        quant_config = _check_quantization(config, quant_config, for_moe=True)
        if quant_config is not None and enable_eplb:
            raise NotImplementedError("Flash quantized MoE does not admit EPLB")
        latent = getattr(config, "routed_expert_hidden_size", None)
        if latent is None:
            super().__init__(
                config,
                quant_config=quant_config,
                prefix=prefix,
                enable_eplb=enable_eplb,
            )
            self.routed_expert_down_proj = None
            self.routed_expert_up_proj = None
        else:
            nn.Module.__init__(self)
            if type(latent) is not int or latent <= 0:
                raise ValueError("routed_expert_hidden_size must be a positive integer")
            parallel = get_current_vllm_config().parallel_config
            if (
                parallel.enable_expert_parallel
                or parallel.enable_elastic_ep
                or parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
                or parallel.use_sequence_parallel_moe
                or parallel.enable_eplb
                or enable_eplb
                or parallel.eplb_config.num_redundant_experts != 0
            ):
                raise NotImplementedError(
                    "Core033 Flash latent MoE currently requires ordinary TP "
                    "(no EP/DP/PCP/DCP/SP/EPLB)"
                )
            if config.hidden_act != "silu":
                raise ValueError("Core033 Flash latent experts require silu")
            self.tp_size = get_tensor_model_parallel_world_size()
            if self.tp_size > config.num_experts:
                raise ValueError("Tensor parallel size exceeds the number of experts")
            ep = get_ep_group()
            self.ep_group = ep.device_group
            self.ep_rank = ep.rank_in_group
            self.ep_size = self.ep_group.size()
            # Match Core HYV4's loader/EPLB-facing metadata, even though latent
            # EPLB execution itself is not admitted yet.
            self.enable_eplb = False
            self.n_routed_experts = config.num_experts
            self.n_logical_experts = self.n_routed_experts
            self.n_redundant_experts = 0
            self.n_physical_experts = self.n_logical_experts
            self.n_local_physical_experts = self.n_physical_experts // self.ep_size
            self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
            self.physical_expert_end = (
                self.physical_expert_start + self.n_local_physical_experts
            )
            self.prefix = prefix
            self.gate = GateLinear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                out_dtype=torch.float32,
                params_dtype=torch.float32,
                prefix=f"{prefix}.gate",
            )
            self.expert_bias = nn.Parameter(
                torch.empty(config.num_experts, dtype=torch.float32)
            )
            self.routed_expert_down_proj = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_up_proj = ReplicatedLinear(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            self.shared_experts = (
                CoreHYV4FeedForward(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.expert_hidden_dim
                    * config.num_shared_experts,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=True,
                    prefix=f"{prefix}.shared_experts",
                )
                if config.num_shared_experts > 0
                else None
            )
            limit = getattr(config, "swiglu_limit", 0)
            self.experts = FusedMoEFactory(
                num_experts=self.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=latent,
                intermediate_size=config.expert_hidden_dim,
                renormalize=config.route_norm,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                enable_eplb=False,
                num_redundant_experts=0,
                scoring_func="sigmoid",
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
                routed_scaling_factor=getattr(config, "router_scaling_factor", 1.0),
                e_score_correction_bias=self.expert_bias,
                shared_experts=None,
                reduce_results=True,
                swiglu_limit=float(limit) if limit and float(limit) > 0 else None,
            )
        # W4A8's eager reference combines router slots in deterministic score
        # order. This is an instance-local Core router, not a global patch.
        if getattr(config, "flash_w4a8_backend", "disabled") == "adapter_reference":
            from hcu_vllm_das.models.hy_v4_flash.router import HYV4FlashStableRouter

            self.experts.router = HYV4FlashStableRouter.from_router(self.experts.router)
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.routed_expert_down_proj is None:
            result = super().forward(hidden_states)
        else:
            orig_shape = hidden_states.shape
            flat = hidden_states.reshape(-1, orig_shape[-1])
            router_logits, _ = self.gate(flat)
            latent, _ = self.routed_expert_down_proj(flat)
            # Core returns globally reduced latent states, not TP partials.
            routed = self.experts(hidden_states=latent, router_logits=router_logits)
            result, _ = self.routed_expert_up_proj(routed)
            if self.shared_experts is not None:
                result = result + self.shared_experts(flat)
            result = result.reshape(orig_shape)
        return (
            self.combine_output_norm(result)
            if self.combine_output_norm is not None
            else result
        )
