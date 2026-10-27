# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select the HCU v0.25.1 PCP MTP adapter through the audited factory."""

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode"
PATCH_ID = "worker.framework_opt.spec_decode.pcp_mtp"
TARGETS = (f"{TARGET_MODULE}.init_speculator",)
_MARKER = "_vllm_hcu_pcp_mtp_factory_applied"
_WRAPPER = "_vllm_hcu_pcp_mtp_factory_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        target, _MARKER, ((target, "init_speculator", TARGETS[0], _WRAPPER),)
    ):
        return False
    original = require_callable(target, "init_speculator", TARGETS[0])
    require_exact_signature(original, TARGETS[0], positional=("vllm_config", "device"))

    @functools.wraps(original)
    def init_speculator(vllm_config, device):
        spec = vllm_config.speculative_config
        if (
            vllm_config.parallel_config.prefill_context_parallel_size > 1
            and spec.method == "mtp"
            and not spec.use_gemma4_mtp()
        ):
            from vllm_hcu.v1.spec_decode.pcp_mtp import HcuPCPMTPSpeculator

            return HcuPCPMTPSpeculator(vllm_config, device)
        return original(vllm_config, device)

    setattr(init_speculator, _WRAPPER, True)
    target.init_speculator = init_speculator
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
