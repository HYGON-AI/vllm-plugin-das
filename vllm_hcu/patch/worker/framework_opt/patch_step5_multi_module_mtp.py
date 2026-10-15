# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator"
PATCH_ID = "worker.framework_opt.step5_multi_module_mtp"
TARGETS = (
    "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator.MultiModuleMTPSpeculator._generate_drafts",
)
_MARKER = "_vllm_hcu_step5_multi_module_mtp_applied"

SIGNATURES = (
    (
        "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator.MultiModuleMTPSpeculator._generate_drafts",
        (
            "self",
            "num_reqs",
            "num_tokens",
            "attn_metadata",
            "slot_mappings",
            "num_tokens_across_dp",
            "cudagraph_runtime_mode",
        ),
    ),
)
REVIEWED_SOURCES = (
    (
        "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator.MultiModuleMTPSpeculator._generate_drafts",
        (
            "88f8e6d38c898e68992d42e986a07bc7bd3277915224566399d00e93780abb78",
            "bdc143884da197f298dd2ba15c34427bda111ba12f8377e26752a03631f2a689",
        ),
    ),
)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools

    original_cls = getattr(target, "MultiModuleMTPSpeculator", None)
    generate = require_callable(original_cls, "_generate_drafts", TARGETS[0])

    class HcuMultiModuleMTPSpeculator(original_cls):
        @functools.wraps(generate)
        def _generate_drafts(self, *args, **kwargs):
            if getattr(self.model, "has_own_mtp_heads", False):
                from vllm_hcu.v1.step5_mtp_runtime import generate_step5_drafts

                return generate_step5_drafts(self, *args, **kwargs)
            return super()._generate_drafts(*args, **kwargs)

        def sample_draft(
            self,
            hidden_states,
            sample_src_positions,
            idx_mapping,
            temperature,
            seeds,
            draft_step,
            draft_logits,
            spec_step_idx=0,
        ):
            if getattr(self.model, "has_own_mtp_heads", False):
                from vllm_hcu.v1.step5_mtp_sampling import sample_step5_draft

                return sample_step5_draft(
                    self,
                    hidden_states,
                    sample_src_positions,
                    idx_mapping,
                    temperature,
                    seeds,
                    draft_step,
                    draft_logits,
                    spec_step_idx,
                )
            return super().sample_draft(
                hidden_states,
                sample_src_positions,
                idx_mapping,
                temperature,
                seeds,
                draft_step,
                draft_logits,
            )

        def _greedy_sample_draft(self, hidden_states, spec_step_idx=0):
            if getattr(self.model, "has_own_mtp_heads", False):
                from vllm_hcu.v1.step5_mtp_sampling import greedy_step5_draft

                return greedy_step5_draft(self, hidden_states, spec_step_idx)
            return super()._greedy_sample_draft(hidden_states)

    target.MultiModuleMTPSpeculator = HcuMultiModuleMTPSpeculator
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
