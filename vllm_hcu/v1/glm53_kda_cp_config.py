# SPDX-License-Identifier: Apache-2.0
"""CPU-safe topology rules shared by GLM5Next PCP and KDA state sizing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

KDA_CP_HEAD_ENV = "VLLM_HCU_GLM53_KDA_CP_HEAD_SHARD"


def glm53_kda_cp_head_shard_enabled(vllm_config: object) -> bool:
    """Read the frozen policy, or resolve the initial launch environment.

    Platform validation persists this decision before compilation hashing.
    Keep the environment fallback for unvalidated CPU-only config adapters;
    serialized engine/worker configs must not change modes with local env.
    """
    from vllm_hcu.patch.config import get_hcu_config

    feature = get_hcu_config(getattr(vllm_config, "additional_config", None))
    if feature.glm53_kda_cp_head_shard is not None:
        return feature.glm53_kda_cp_head_shard
    value = os.environ.get(KDA_CP_HEAD_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{KDA_CP_HEAD_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def _positive_size(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class KdaHeadParallelPlan:
    """Head ownership only; not permission to run an entire PCP topology.

    Like SGLang, select one group for heads, never implicitly TP * PCP.
    This object is CPU-safe and does not create or replace process groups.
    """

    group: Literal["tp", "pcp"]
    size: int

    def __post_init__(self):
        if self.group not in ("tp", "pcp"):
            raise ValueError(f"invalid KDA head group {self.group!r}")
        _positive_size(self.size, "KDA head group size")

    def local_num_heads(self, num_heads: int) -> int:
        if type(num_heads) is not int or num_heads <= 0 or num_heads % self.size:
            raise ValueError(
                f"GLM5Next KDA head count {num_heads} must be positive and "
                f"divisible by {self.group.upper()} size {self.size}"
            )
        return num_heads // self.size

    def rank_in_group(self, group: object) -> int:
        if group.world_size != self.size:
            raise ValueError(
                f"KDA configured {self.group.upper()} head size {self.size} "
                f"does not match the runtime group size {group.world_size}"
            )
        rank = group.rank_in_group
        if type(rank) is not int or not 0 <= rank < self.size:
            raise ValueError(f"KDA {self.group.upper()} rank is outside the head group")
        return rank


def glm53_text_config(vllm_config: object) -> object:
    model = vllm_config.model_config
    config = getattr(model, "hf_text_config", None)
    return model.hf_config if config is None else config


def resolve_kda_head_parallel(vllm_config: object) -> KdaHeadParallelPlan:
    """Select native TP or opt-in PCP head ownership, independent of admission.

    A valid head plan alone must never be interpreted as support for mixed
    TP/PCP. Execution admission separately enforces the supported PP/DCP
    layout and the model, metadata, communication and graph contracts.
    """
    enabled = glm53_kda_cp_head_shard_enabled(vllm_config)
    parallel = vllm_config.parallel_config
    tp_size = _positive_size(parallel.tensor_parallel_size, "TP size")
    if not enabled:
        return KdaHeadParallelPlan("tp", tp_size)
    pcp_size = _positive_size(parallel.prefill_context_parallel_size, "PCP size")
    if pcp_size == 1:
        return KdaHeadParallelPlan("tp", tp_size)
    plan = KdaHeadParallelPlan("pcp", pcp_size)
    plan.local_num_heads(glm53_text_config(vllm_config).linear_num_heads)
    return plan


def validate_glm53_pcp_topology(vllm_config: object) -> int:
    """Validate PCP sizes and reject unimplemented PP/DCP execution paths.

    Every current worker needs both KDA and K-pool layers, and the replicated
    hybrid-cache adapter does not provide DCP ownership for KpoolTailSpec.
    These constraints apply with either Head switch value. Platform scope
    validation separately checks EP, eager execution and model capabilities.
    """
    parallel = vllm_config.parallel_config
    size = _positive_size(parallel.prefill_context_parallel_size, "PCP size")
    if size <= 1:
        raise ValueError("GLM5Next PCP requires prefill context parallel size > 1")
    for field, label in (
        ("tensor_parallel_size", "TP"),
        ("pipeline_parallel_size", "PP"),
        ("data_parallel_size", "DP"),
        ("decode_context_parallel_size", "DCP"),
    ):
        _positive_size(getattr(parallel, field), f"{label} size")
    if parallel.pipeline_parallel_size != 1:
        raise ValueError(
            "GLM5Next PCP requires PP1: per-stage KDA/K-pool ownership "
            "is not implemented"
        )
    if parallel.decode_context_parallel_size != 1:
        raise ValueError(
            "GLM5Next PCP requires DCP1: the KDA/K-pool hybrid cache "
            "does not support DCP ownership for KpoolTailSpec"
        )
    return size


def validate_glm53_mtp_sync_contract(vllm_config: object, enabled: bool) -> None:
    """Engine preflight and worker defense for the same decision-sync contract."""
    if not enabled:
        return
    if not vllm_config.model_config.enforce_eager:
        raise ValueError("GLM53 PCP MTP sync v1 requires enforce_eager=True")
    spec = vllm_config.speculative_config
    if getattr(spec, "draft_sample_method", "greedy") != "greedy":
        raise ValueError("GLM53 PCP MTP sync v1 supports greedy draft sampling only")
    if getattr(spec, "enable_adaptive_verification", False):
        raise ValueError("GLM53 PCP MTP sync v1 does not cover adaptive verification")


def kda_cp_head_size(vllm_config: object) -> int | None:
    """Legacy accessor using the shared PCP topology validation.

    Engine-side state sizing and worker-side construction must use the same
    decision. Set the environment variable before launching any processes.
    PCP1 always retains native TP construction/state sizing, even when the
    experimental switch is enabled. No global TP group is rewritten.
    """
    plan = resolve_kda_head_parallel(vllm_config)
    if plan.group == "tp":
        return None
    validate_glm53_pcp_topology(vllm_config)
    return plan.size
