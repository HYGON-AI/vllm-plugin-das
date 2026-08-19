# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU integration boundary for vLLM's Model Runner V2."""

import functools

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


_FIXED_WIDTH_PP_BROADCAST_MARKER = "_vllm_hcu_fixed_width_pp_broadcast"


def install_fixed_width_pp_sample_broadcast(model_runner: object) -> bool:
    """Match the last PP rank's first sample to the fixed receive width.

    Before draft tokens exist, MRV2's sampler returns one token per request,
    while ``PPHandler.receive`` always allocates ``num_speculative_steps + 1``
    columns.  NCCL requires identical element counts on both sides.  Pad only
    the broadcast payload; ``num_sampled`` continues to describe valid tokens.
    """

    pp_handler = getattr(model_runner, "pp_handler", None)
    if pp_handler is None or not getattr(pp_handler, "is_last_rank", False):
        return False
    broadcast = getattr(pp_handler, "broadcast")
    if getattr(broadcast, _FIXED_WIDTH_PP_BROADCAST_MARKER, False):
        return False

    max_sample_len = int(getattr(pp_handler, "max_sample_len"))

    @functools.wraps(broadcast)
    def fixed_width_broadcast(sampled_token_ids, *args, **kwargs):
        sample_len = int(sampled_token_ids.shape[-1])
        if sample_len > max_sample_len:
            raise ValueError(
                "PP sampled-token width exceeds the receiver allocation: "
                f"{sample_len} > {max_sample_len}"
            )
        if sample_len < max_sample_len:
            padded = sampled_token_ids.new_zeros(
                (*sampled_token_ids.shape[:-1], max_sample_len)
            )
            padded[..., :sample_len].copy_(sampled_token_ids)
            sampled_token_ids = padded
        return broadcast(sampled_token_ids, *args, **kwargs)

    setattr(fixed_width_broadcast, _FIXED_WIDTH_PP_BROADCAST_MARKER, True)
    setattr(pp_handler, "broadcast", fixed_width_broadcast)
    return True


def synchronize_pp_spec_draft_tokens(
    model_runner: object, input_batch: object
) -> bool:
    """Copy last-rank MTP drafts to every earlier PP stage.

    Async scheduling sends only placeholder draft IDs through the scheduler.
    MRV2 keeps the real IDs in worker-local GPU state, but under PP only the
    last rank owns the speculator.  Without this transfer, earlier stages run
    the target model on zero draft IDs while the last stage rejection-samples
    against the real drafts.

    Use the sampled-token broadcast stream and group so collective ordering is
    identical on every rank.  The explicit stream completion is conservative
    but necessary before a non-last rank publishes the received tensor into
    its request state.
    """

    if getattr(model_runner, "_vllm_hcu_suppress_pp_spec_draft_sync", False):
        return False
    pp_handler = getattr(model_runner, "pp_handler", None)
    num_speculative_steps = int(
        getattr(model_runner, "num_speculative_steps", 0)
    )
    if pp_handler is None or num_speculative_steps == 0:
        return False

    req_states = getattr(model_runner, "req_states")
    idx_mapping = getattr(input_batch, "idx_mapping")
    num_reqs = int(getattr(input_batch, "num_reqs"))
    broadcast_stream = getattr(pp_handler, "broadcast_stream")
    main_stream = getattr(pp_handler, "main_stream")

    with torch.cuda.stream(broadcast_stream):
        broadcast_stream.wait_stream(main_stream)
        if getattr(pp_handler, "is_last_rank"):
            draft_tokens = req_states.draft_tokens[idx_mapping].contiguous()
        else:
            draft_tokens = torch.empty(
                (num_reqs, num_speculative_steps),
                dtype=req_states.draft_tokens.dtype,
                device=getattr(model_runner, "device"),
            )
        torch.distributed.broadcast(
            draft_tokens,
            src=getattr(pp_handler, "last_rank"),
            group=getattr(pp_handler, "broadcast_group"),
        )
    broadcast_stream.synchronize()

    if not getattr(pp_handler, "is_last_rank"):
        req_states.draft_tokens[idx_mapping] = draft_tokens
    return True


class HcuGPUModelRunnerV2(GPUModelRunner):
    """HCU compatibility adapter around upstream v0.25.1 Model Runner V2."""

    def sample_tokens(self, grammar_output):
        execute_model_state = self.execute_model_state
        input_batch = (
            None
            if execute_model_state is None
            else execute_model_state.input_batch
        )
        output = super().sample_tokens(grammar_output)
        if input_batch is not None:
            synchronize_pp_spec_draft_tokens(self, input_batch)
        return output


__all__ = [
    "HcuGPUModelRunnerV2",
    "install_fixed_width_pp_sample_broadcast",
    "synchronize_pp_spec_draft_tokens",
]
