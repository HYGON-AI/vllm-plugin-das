# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional legacy provider for the owned HY4 PCP16 HT expert output."""
from .environment import _flag, custom_ops_enabled


def _requested():
    return custom_ops_enabled() and _flag('HCU_VLLM_HY4_EXPERT_GATHER', default=False)


def configure(experts, cfg, moe, prepare):
    if not _requested():
        return
    from .moe import ct_w4a8_runtime as runtime
    from .hy4_ht_int8 import _supported

    if (type(experts) is not runtime.HcuCTContiguousExperts
            or experts._gather is not runtime.deepgemm_unpermute_and_reduce
            or not _supported(cfg, moe, prepare)):
        return
    from hcu_adapter.bridges.vllm_expert_gather import unpermute_and_reduce

    native = experts._gather

    def gather(*args, **kwargs):
        if _requested():
            return unpermute_and_reduce(*args, **kwargs)
        return native(*args, **kwargs)

    experts._gather = gather
