# SPDX-License-Identifier: Apache-2.0
"""Mooncake decoder-first-token trace at the OutputProcessor boundary."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.platforms import envs as henvs

from ._common import PatchCompatibilityError, load_exact_module, require_callable, require_class, require_signature_prefix

TARGET_MODULE = "vllm.v1.engine.output_processor"
PATCH_ID = "platform.framework_opt.output_processor_ttft"
TARGETS = (f"{TARGET_MODULE}.OutputProcessor.process_outputs",)
_MARKER = "_vllm_hcu_output_processor_ttft_applied"
_WRAPPER = "_vllm_hcu_output_processor_ttft_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    output_processor = require_class(target, "OutputProcessor", f"{TARGET_MODULE}.OutputProcessor")
    process_outputs = require_callable(output_processor, "process_outputs", TARGETS[0])
    if getattr(output_processor, _MARKER, False):
        if not getattr(process_outputs, _WRAPPER, False):
            raise PatchCompatibilityError(
                "HCU OutputProcessor TTFT marker is stale; restart the process"
            )
        return False
    require_signature_prefix(
        process_outputs,
        TARGETS[0],
        ("self", "engine_core_outputs", "engine_core_timestamp", "iteration_stats"),
    )

    @functools.wraps(process_outputs)
    def hcu_process_outputs(
        self,
        engine_core_outputs,
        engine_core_timestamp=None,
        iteration_stats=None,
    ):
        if henvs.VLLM_HCU_MOONCAKE_TTFT_TRACE:
            from vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
                log_ttft_event,
            )

            for output in engine_core_outputs:
                req_state = self.request_states.get(output.request_id)
                if req_state is not None and req_state.is_prefilling:
                    log_ttft_event(
                        "d_first_token",
                        req_id=output.request_id,
                        kv_params=output.kv_transfer_params,
                    )
        return process_outputs(
            self,
            engine_core_outputs,
            engine_core_timestamp,
            iteration_stats,
        )

    setattr(hcu_process_outputs, _WRAPPER, True)
    output_processor._vllm_hcu_original_process_outputs = process_outputs
    output_processor.process_outputs = hcu_process_outputs
    setattr(output_processor, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
