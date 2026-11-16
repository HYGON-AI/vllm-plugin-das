# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from vllm.v1.worker.gpu.spec_decode.speculator import (
    gumbel_sample,
    torch,
)


def sample_step5_draft(
    self,
    hidden_states: torch.Tensor,
    sample_src_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    draft_step: torch.Tensor,
    draft_logits: torch.Tensor | None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    # spec_step_idx selects the MTP module's own shared_head (norm + lm
    # head); it must be a Python int so it stays graph-capture safe.
    if draft_logits is not None:
        logits = self.model.compute_logits(hidden_states, spec_step_idx)
        return gumbel_sample(
            logits,
            idx_mapping,
            temperature,
            seeds,
            sample_src_positions,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=draft_logits,
            logits_cache_col=draft_step,
            use_fp64=self.use_fp64_gumbel,
        )
    return self._greedy_sample_draft(hidden_states, spec_step_idx)


def greedy_step5_draft(
    self, hidden_states: torch.Tensor, spec_step_idx: int = 0
) -> torch.Tensor:
    if self.use_local_argmax_reduction:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)
    logits = self.model.compute_logits(hidden_states, spec_step_idx)
    return logits.argmax(dim=-1)
