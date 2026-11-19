# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Use the HCU rotary kernel for eager PCP prefill queries."""
from functools import lru_cache, partial

import torch

from hcu_adapter import registry

from . import inverse_rope_hcu


@lru_cache(maxsize=1)
def load_operator():
    # Forward Q retains separately rounded products; inverse RoPE must match
    # Core's cosine-product FMA. The native implementation specializes both.
    operator = partial(inverse_rope_hcu.load_operator(), fuse_cosine=False)
    registry.register(
        "vllm_pcp_q_rope", "hcu", "torch.ops.hcu_vllm_das.deepseek_v4_inv_rope",
        registry.MECH_SEAM, detail="Eager PCP prefill Q RoPE using a negated sine table",
    )
    return operator


def forward_table(cos_sin_cache):
    """Copy the static Core table and reverse its sine half."""
    with torch.no_grad():
        table = cos_sin_cache.clone(memory_format=torch.contiguous_format)
        table[:, 32:].neg_()
    return table


@lru_cache(maxsize=None)
def _supported_device(device):
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0] == "gfx938"


def _supports_table(cache):
    return (
        torch.version.hip is not None
        and cache.is_cuda and cache.dtype == torch.float32
        and cache.ndim == 2 and cache.shape[1] == 64
        and _supported_device(cache.device)
    )


@lru_cache(maxsize=4)
def initialize(cos_sin_cache):
    """Reserve the shared FP32 table and load the extension during profiling."""
    if not _supports_table(cos_sin_cache):
        return None

    operator = load_operator()
    # Layer closures retain the table even if this sharing cache evicts it.
    table = forward_table(cos_sin_cache)

    def prepare(q, positions, current_table):
        if (current_table is not cos_sin_cache or torch.is_grad_enabled()
                or q.device != table.device or q.dtype != torch.bfloat16 or q.ndim != 3
                or q.shape[-1] != 512 or not q.is_contiguous()
                or not 0 < q.shape[0] * q.shape[1] <= 2**31 - 9
                or positions.device != q.device or positions.dtype != torch.int64
                or positions.shape != (q.shape[0],) or not positions.is_contiguous()):
            return None
        return partial(operator, q[..., -64:], positions, table)

    return prepare
