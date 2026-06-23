# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fused_moe.router.base_router.

Allow HCU to opt into a torch EPLB map+record fallback. Keep the original fused
Triton kernel as the default path.
"""

PATCHES = [
    (
        """from vllm.platforms import current_platform
""",
        """from vllm.platforms import current_platform
from vllm_hcu.platforms import envs as henvs
""",
    ),
    (
        """    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
    ) -> torch.Tensor:
        # Fused triton implementation: mapping + optional recording in one kernel.
        return _eplb_map_and_record_triton(
            topk_ids=topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_view=expert_load_view,
            record_enabled=record_enabled,
        )
""",
        """    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
    ) -> torch.Tensor:
        if not henvs.VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD:
            return _eplb_map_and_record_triton(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                expert_load_view=expert_load_view,
                record_enabled=record_enabled,
            )

        # HCU: optionally use torch ops instead of the fused Triton EPLB kernel.
        topk_shape = topk_ids.shape
        topk_ids_flat = topk_ids.reshape(-1)
        numel = topk_ids_flat.numel()
        if numel == 0:
            return topk_ids

        num_active_experts = topk_shape[-1]
        valid_expert = (topk_ids_flat >= 0) & (
            topk_ids_flat < logical_replica_count.shape[0]
        )
        safe_expert_id = torch.where(
            valid_expert,
            topk_ids_flat,
            torch.zeros_like(topk_ids_flat),
        ).to(torch.long)

        replica_count = logical_replica_count[safe_expert_id].clamp_min(1).to(torch.long)
        token_idx = (
            torch.arange(numel, device=topk_ids.device, dtype=torch.long)
            // num_active_experts
        )
        hashed = (token_idx * 2654435769) % (1 << 32)
        replica_idx = hashed % replica_count

        physical_flat = torch.full_like(topk_ids_flat, -1)
        mapped = logical_to_physical_map[safe_expert_id, replica_idx].to(topk_ids.dtype)
        physical_flat = torch.where(valid_expert, mapped, physical_flat)

        valid_physical = (physical_flat >= 0) & (
            physical_flat < expert_load_view.shape[0]
        )
        safe_physical_id = torch.where(
            valid_physical,
            physical_flat,
            torch.zeros_like(physical_flat),
        ).to(torch.long)
        increments = valid_physical.to(expert_load_view.dtype) * record_enabled.to(
            expert_load_view.dtype
        )
        expert_load_view.scatter_add_(0, safe_physical_id, increments)

        return physical_flat.reshape(topk_shape)
""",
    ),
]
