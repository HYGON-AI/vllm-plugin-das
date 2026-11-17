# SPDX-License-Identifier: Apache-2.0
"""PCP8/EP8 producer identity and strict topology preflight.

This module does not install a loader or start any process. The native vLLM
key still has to be independently checked before any tensor handles are sent.
"""
from dataclasses import asdict, dataclass

PROTOCOL = "minimax31-pcp8-ep8-identity-v1"
WORLD_SIZE = 8
EXPERTS = 128


@dataclass(frozen=True)
class PartitionIdentity:
    protocol: str
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    pcp_size: int
    pcp_rank: int
    ep_size: int
    ep_rank: int
    expert_placement: str
    expert_begin: int
    expert_end: int
    gpu_uuid: str
    weight_contract_sha256: str


@dataclass(frozen=True)
class StaticPartitionIdentity(PartitionIdentity):
    logical_experts: int
    physical_experts: int
    slots_per_rank: int
    map_sha256: str
    selector_version: str
    loader_abi: str


def validate_parallel_config(config):
    required = {
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        "data_parallel_size": 1, "prefill_context_parallel_size": 8,
        "decode_context_parallel_size": 1, "enable_expert_parallel": True,
        "enable_eplb": False, "expert_placement_strategy": "linear",
        "all2all_backend": "allgather_reducescatter",
    }
    for name, expected in required.items():
        actual = getattr(config, name, None)
        if type(expected) is bool:
            match = actual is expected
        else:
            match = actual == expected and not isinstance(actual, bool)
        if not match:
            raise ValueError("unsupported PCP8/EP8 %s=%r; expected %r" % (name, actual, expected))
    if getattr(config, "nnodes", 1) != 1:
        raise ValueError("PCP8/EP8 preload supports exactly one node")
    if getattr(config, "world_size", WORLD_SIZE) != WORLD_SIZE:
        raise ValueError("PCP8/EP8 requires world size 8")
    return True


def plan_ranks(config, static_placement=None):
    validate_parallel_config(config)
    local=17 if static_placement is not None else 16
    return tuple({"global_rank": rank, "local_rank": rank, "tp_rank": 0,
                  "pp_rank": 0, "dp_rank": 0, "pcp_rank": rank,
                  "ep_rank": rank, "expert_begin": local * rank,
                  "expert_end": local * (rank + 1)} for rank in range(WORLD_SIZE))


def partition_identity(rank, gpu_uuid, weight_contract_sha256, static_placement=None):
    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise ValueError("PCP/EP rank must be an integer in [0, 8)")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("actual GPU UUID is required")
    if (not isinstance(weight_contract_sha256, str)
            or len(weight_contract_sha256) != 64
            or any(c not in "0123456789abcdef" for c in weight_contract_sha256)):
        raise ValueError("canonical weight-contract SHA256 is required")
    if static_placement is not None:
        from .minimax31_static_contract import StaticPlacement, SELECTOR, ABI
        if not isinstance(static_placement, StaticPlacement):raise ValueError("invalid immutable static partition")
        return StaticPartitionIdentity("minimax31-pcp8-ep8-static136-identity-v2",1,0,1,0,1,0,8,rank,8,rank,
                "static136-v1",rank*17,(rank+1)*17,gpu_uuid,weight_contract_sha256,
                128,136,17,static_placement.map_sha256,SELECTOR["version"],ABI)
    return PartitionIdentity(PROTOCOL, 1, 0, 1, 0, 1, 0, 8, rank, 8, rank,
                             "linear", rank * 16, (rank + 1) * 16,
                             gpu_uuid, weight_contract_sha256)


def mismatched_partition_fields(expected, received):
    """Compare all identity fields; never relax the native vLLM key check."""
    if not isinstance(expected, PartitionIdentity) or not isinstance(received, dict):
        raise ValueError("complete native and PCP/EP partition identities are required")
    fields = asdict(expected)
    if set(received) != set(fields):
        raise ValueError("missing or unrecognized PCP/EP identity fields")
    return tuple(key for key in fields if type(fields[key]) is not type(received[key])
                 or fields[key] != received[key])


def require_extensions():
    """Require both explicit extensions and check their current ownership."""
    from hcu_adapter.bridges.vllm import extension_reports, require_runtime_healthy
    from hcu_adapter.extensions import ExtensionStatus
    reports = extension_reports()
    for name in ("minimax31_slimquant", "minimax31_pcp_ep"):
        matches = [report for report in reports if report.entry_point_name == name]
        if len(matches) != 1 or matches[0].status is not ExtensionStatus.ACTIVE:
            raise RuntimeError("required MiniMax extension is not active: " + name)
    require_runtime_healthy()
