# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned behavior formerly embedded in vLLM communicator source patches."""

from __future__ import annotations

import sys
from typing import Any


_HT_NVL_BUFFER_BYTES = 1_000_000_000
_HT_RDMA_BUFFER_BYTES = 500_000_000


def initialize_deep_ep_manager(manager: object) -> None:
    """Use the HCU-tuned initial DeepEP communication budget."""

    manager.num_sms = 30


def make_deep_ep_ht_kwargs(manager: object, envs: object) -> dict[str, Any]:
    """Build the audited HCU DeepEP high-throughput buffer configuration."""

    internode = bool(getattr(manager, "internode"))
    force_intra = bool(
        getattr(envs, "VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE")
    )
    if internode and not force_intra:
        num_rdma_bytes = _HT_RDMA_BUFFER_BYTES
        num_qps_per_rank = 30
        manager.num_sms = 30
    else:
        num_rdma_bytes = 0
        num_qps_per_rank = 1
        manager.num_sms = 60

    return {
        "group": manager.cpu_group,
        "num_nvl_bytes": _HT_NVL_BUFFER_BYTES,
        "num_rdma_bytes": num_rdma_bytes,
        "low_latency_mode": False,
        "num_qps_per_rank": num_qps_per_rank,
        "explicitly_destroy": True,
    }


def requested_deep_ep_sms(num_sms: int) -> int:
    from vllm_hcu.platforms import envs as henvs

    override = henvs.VLLM_HCU_DEEPEP_NUM_SMS
    return int(num_sms if override is None else override)


def validated_low_latency_rdma_size_hint(
    max_num_tokens_per_dp_rank: int,
    token_hidden_size: int,
    num_ep_ranks: int,
    num_global_experts: int,
) -> int:
    """Return a model-specific DeepEP hint after detecting integer wraparound."""

    import deep_ep

    model_dimensions = {
        "max_num_tokens_per_dp_rank": max_num_tokens_per_dp_rank,
        "token_hidden_size": token_hidden_size,
        "num_ep_ranks": num_ep_ranks,
        "num_global_experts": num_global_experts,
    }
    if any(
        not isinstance(value, int) or value <= 0
        for value in model_dimensions.values()
    ):
        dimensions = ", ".join(
            f"{name}={value!r}" for name, value in model_dimensions.items()
        )
        raise ValueError(
            "DeepEP low-latency buffer dimensions must be positive integers: "
            f"{dimensions}"
        )

    previous_hint = 0
    for token_capacity in range(1, max_num_tokens_per_dp_rank + 1):
        try:
            hint = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                num_max_dispatch_tokens_per_rank=token_capacity,
                hidden=token_hidden_size,
                num_ranks=num_ep_ranks,
                num_experts=num_global_experts,
            )
        except Exception as exc:
            raise ValueError(
                "DeepEP low-latency RDMA size hint failed at token capacity "
                f"{token_capacity} for hidden_size={token_hidden_size}, "
                f"num_global_experts={num_global_experts}, ep_size={num_ep_ranks}. "
                "Reduce --max-num-batched-tokens or verify the DeepEP ABI."
            ) from exc
        if (
            not isinstance(hint, int)
            or hint <= 0
            or hint > sys.maxsize
            or hint < previous_hint
        ):
            raise ValueError(
                "DeepEP low-latency RDMA size hint overflowed at token capacity "
                f"{token_capacity} for hidden_size={token_hidden_size}, "
                f"num_global_experts={num_global_experts}, ep_size={num_ep_ranks}; "
                f"previous_hint={previous_hint}, current_hint={hint!r}. Reduce "
                "--max-num-batched-tokens or use deepep_high_throughput."
            )
        previous_hint = hint

    return previous_hint


def _manager_low_latency_rdma_size_hint(
    manager: object,
    max_num_tokens_per_dp_rank: int,
    token_hidden_size: int,
    num_ep_ranks: int,
    num_global_experts: int,
) -> int:
    """Cache successful validation because every MoE layer reuses the manager."""

    cache_key = (
        max_num_tokens_per_dp_rank,
        token_hidden_size,
        num_ep_ranks,
        num_global_experts,
    )
    cache = getattr(manager, "_vllm_hcu_ll_rdma_hint_cache", None)
    if cache is None:
        cache = {}
        setattr(manager, "_vllm_hcu_ll_rdma_hint_cache", cache)
    if cache_key not in cache:
        cache[cache_key] = validated_low_latency_rdma_size_hint(*cache_key)
    return cache[cache_key]


def make_deep_ep_ll_kwargs(
    manager: object,
    envs: object,
    max_num_tokens_per_dp_rank: int,
    token_hidden_size: int,
    num_ep_ranks: int,
    num_global_experts: int,
    num_local_experts: int,
) -> dict[str, Any]:
    """Build a dynamically validated DeepEP low-latency configuration."""

    num_rdma_bytes = _manager_low_latency_rdma_size_hint(
        manager,
        max_num_tokens_per_dp_rank=max_num_tokens_per_dp_rank,
        token_hidden_size=token_hidden_size,
        num_ep_ranks=num_ep_ranks,
        num_global_experts=num_global_experts,
    )
    return {
        "group": manager.cpu_group,
        "num_nvl_bytes": envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024,
        "num_rdma_bytes": num_rdma_bytes,
        "low_latency_mode": True,
        "num_qps_per_rank": num_local_experts,
        "allow_nvlink_for_low_latency_mode": True,
        "allow_mnnvl": envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
        "explicitly_destroy": True,
        "enable_shrink": manager.support_fault_tolerance,
    }


def install_deep_ep_auto_manager(module: object) -> type:
    """Install an HCU-owned manager using one DeepEP buffer for HT and LL."""

    ll_manager = module.DeepEPLLAll2AllManager
    if hasattr(module, "DeepEPAutoAll2AllManager"):
        raise RuntimeError("DeepEPAutoAll2AllManager already exists outside HCU")

    class DeepEPAutoAll2AllManager(ll_manager):
        is_deepep_auto_manager = True

        def __init__(self, cpu_group, tcp_store_group=None):
            super().__init__(cpu_group, tcp_store_group)
            self.num_sms = 48

        def _make_all2all_kwargs(
            self,
            max_num_tokens_per_dp_rank: int,
            token_hidden_size: int,
            num_ep_ranks: int,
            num_global_experts: int,
            num_local_experts: int,
        ) -> dict[str, Any]:
            ht_rdma_bytes = (
                _HT_RDMA_BUFFER_BYTES
                if self.internode
                and not module.envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE
                else 0
            )
            ht_qps_per_rank = 30 if ht_rdma_bytes else 1
            ll_nvl_bytes = module.envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            ll_rdma_bytes = _manager_low_latency_rdma_size_hint(
                self,
                max_num_tokens_per_dp_rank=max_num_tokens_per_dp_rank,
                token_hidden_size=token_hidden_size,
                num_ep_ranks=num_ep_ranks,
                num_global_experts=num_global_experts,
            )
            return {
                "group": self.cpu_group,
                "num_nvl_bytes": max(_HT_NVL_BUFFER_BYTES, ll_nvl_bytes),
                "num_rdma_bytes": max(ht_rdma_bytes, ll_rdma_bytes),
                "low_latency_mode": True,
                "num_qps_per_rank": max(ht_qps_per_rank, num_local_experts),
                "allow_nvlink_for_low_latency_mode": True,
                "allow_mnnvl": module.envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
                "explicitly_destroy": True,
                "enable_shrink": self.support_fault_tolerance,
            }

        def max_sms_used(self) -> int | None:
            return self.num_sms

    DeepEPAutoAll2AllManager.__name__ = "DeepEPAutoAll2AllManager"
    DeepEPAutoAll2AllManager.__qualname__ = "DeepEPAutoAll2AllManager"
    DeepEPAutoAll2AllManager.__module__ = module.__name__
    module.DeepEPAutoAll2AllManager = DeepEPAutoAll2AllManager
    return DeepEPAutoAll2AllManager


def all_to_all_single(communicator: object, output: object, input_: object):
    """Portable custom-SP collective; CUDA/HIP libraries remain optional."""

    import torch

    torch.distributed.all_to_all_single(
        output, input_, group=communicator.device_group
    )
    return output


__all__ = [
    "all_to_all_single",
    "initialize_deep_ep_manager",
    "install_deep_ep_auto_manager",
    "make_deep_ep_ht_kwargs",
    "make_deep_ep_ll_kwargs",
    "requested_deep_ep_sms",
    "validated_low_latency_rdma_size_hint",
]
