# SPDX-License-Identifier: Apache-2.0
"""Device-free MiniMax DP8 preload identities; independent of PCP publications."""
from dataclasses import asdict, dataclass
import hashlib
import json

PROTOCOL = "minimax31-dp8-target-draft-ipc-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def validate_parallel(config):
    for field, expected in {
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        "data_parallel_size": 8, "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1, "enable_eplb": False,
        "expert_placement_strategy": "linear", "nnodes": 1,
    }.items():
        value = getattr(config, field, None)
        if type(value) is not type(expected) or value != expected:
            raise ValueError("MiniMax decode requires %s=%r, got %r" % (field, expected, value))
    return True


@dataclass(frozen=True)
class DecodePartition:
    protocol: str
    role: str
    dp_size: int
    dp_rank: int
    tp_size: int
    pp_size: int
    pcp_size: int
    ep_size: int
    ep_rank: int
    logical_experts: int
    physical_experts: int
    expert_begin: int
    expert_end: int
    gpu_uuid: str
    weight_identity: str

    def to_dict(self):
        return asdict(self)


def partition(role, rank, gpu_uuid, weight_identity, physical_experts=128):
    if role not in ("target", "draft") or type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("invalid decode role or DP rank")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("actual GPU UUID is required")
    if (not isinstance(weight_identity, str) or len(weight_identity) != 64
            or any(c not in "0123456789abcdef" for c in weight_identity)):
        raise ValueError("complete weight identity SHA256 is required")
    if type(physical_experts) is not int or physical_experts not in (128, 136):
        raise ValueError("unsupported physical expert capacity")
    local = physical_experts // 8
    return DecodePartition(PROTOCOL, role, 8, rank, 1, 1, 1,
                           8 if role == "target" else 1,
                           rank if role == "target" else 0,
                           128 if role == "target" else 0,
                           physical_experts if role == "target" else 0,
                           rank * local if role == "target" else 0,
                           (rank + 1) * local if role == "target" else 0,
                           gpu_uuid, weight_identity)


def require_partition(expected, received):
    wanted = expected.to_dict()
    if not isinstance(received, dict) or set(received) != set(wanted):
        raise ValueError("incomplete or foreign decode partition")
    bad = [k for k, v in wanted.items() if type(v) is not type(received[k]) or v != received[k]]
    if bad:
        raise ValueError("decode partition mismatch: " + ",".join(bad))


def ready_roles(rows, *, with_draft):
    expected = {(role, rank) for role in (("target", "draft") if with_draft else ("target",))
                for rank in range(8)}
    actual = [(r["role"], r["dp_rank"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("missing, duplicate or foreign preload role/rank")
    for rank in range(8):
        devices = {r["gpu_uuid"] for r in rows if r["dp_rank"] == rank}
        if len(devices) != 1:
            raise ValueError("target/draft GPU mismatch")
    if len({r["gpu_uuid"] for r in rows if r["role"] == "target"}) != 8:
        raise ValueError("DP ranks must own distinct GPUs")
    return True
