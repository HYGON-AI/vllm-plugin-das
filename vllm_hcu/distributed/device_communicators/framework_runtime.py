# SPDX-License-Identifier: Apache-2.0
"""HCU-owned behavior formerly embedded in vLLM communicator source patches."""

from __future__ import annotations

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
    "make_deep_ep_ht_kwargs",
    "requested_deep_ep_sms",
]
