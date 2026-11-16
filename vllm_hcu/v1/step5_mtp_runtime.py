# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    Any,
    CUDAGraphMode,
    cache_inputs,
    torch,
    update_draft_inputs,
)


def generate_step5_drafts(
    self,
    num_reqs: int,
    num_tokens: int,
    attn_metadata: dict[str, Any] | None,
    slot_mappings: dict[str, torch.Tensor] | None,
    num_tokens_across_dp: torch.Tensor | None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> None:
    last_token_indices = self.last_token_indices[:num_reqs]
    positions = self.input_buffers.positions[last_token_indices]
    # The output hidden state at position P (= positions) and the token id
    # at P+1 are used to draft the token at P+2. Sampling keys a draw by the
    # position before the sampled token, so the net adjustment is +1.
    # After each draft, advance this sampling position with the draft inputs.
    # spec_step_idx selects the MTP layer; it does not advance the sampling key.
    sample_src_positions = positions + 1
    idx_mapping = self.idx_mapping[:num_reqs]

    # Cache the trailing token's ids, hidden states (and embeddings for
    # MM models), which are needed if the trailing tokens are re-prefilled
    # during the next decode step.
    cache_inputs(
        self.input_buffers,
        self.inputs_embeds,
        self.hidden_states,
        self.cached_draft_input_ids,
        self.cached_draft_input_embeds,
        self.cached_target_hidden_states,
        last_token_indices,
        idx_mapping,
        num_reqs,
        self.num_speculative_steps,
        use_input_embeds=self.inputs_embeds is not None,
    )

    for step in range(self.num_speculative_steps):
        # Update the current draft step.
        self.current_draft_step.fill_(step)

        # Run the model forward pass.
        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            spec_module_idx=step,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        )

        # Sample draft tokens for the current step.
        sample_hidden_states = last_hidden_states[last_token_indices]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
            spec_step_idx=step,
        )

        self.draft_tokens[:num_reqs, step] = draft_tokens
        if step < self.num_speculative_steps - 1:
            self.hidden_states[:num_tokens] = hidden_states
            # Mid-prefill requests append the known future prefill token
            # instead of the sampled draft.
            overrides = self.draft_input_id_overrides[:num_reqs, step]
            next_input_tokens = torch.where(overrides >= 0, overrides, draft_tokens)
            # Shift the draft inputs left by one and append the next
            # input token id/embeddings.
            draft_embeds = (
                self.model.embed_input_ids(next_input_tokens)
                if self.inputs_embeds is not None
                else None
            )
            update_draft_inputs(
                next_input_tokens,
                draft_embeds,
                self.input_buffers,
                self.inputs_embeds,
                last_token_indices,
                idx_mapping,
                num_reqs,
            )
            # Advance the draft sampling key.
            sample_src_positions += 1
