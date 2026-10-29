# SPDX-License-Identifier: Apache-2.0
"""Candidate fix: PCP-consistent greedy MTP decisions for eager GLM5Next.

This does not fix the numerical source of different local draft predictions.
GPU correctness and end-to-end acceptance must be tested before release.
"""
from __future__ import annotations

import functools
import os

import torch


def _in_replicated_mtp_batch() -> bool:
    from vllm_hcu.model_executor.layers.attention.pcp import in_replicated_mtp_batch

    return in_replicated_mtp_batch()


def _broadcast_draft_tokens(tokens, group):
    if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError("greedy MTP must return a 1-D integer token tensor")
    if tokens.numel() == 0:
        return tokens
    # src is group-local, not the global distributed rank. Always consume the
    # returned tensor, including implementations returning a different buffer.
    return group.broadcast(tokens.contiguous(), src=0)


def _install_draft_sync(speculator, group):
    previous = getattr(speculator, "_hcu_glm53_pcp_mtp_sync_group", None)
    if previous is not None:
        if previous is not group:
            raise RuntimeError("MTP PCP group changed after sync installation")
        return
    original = speculator.sample_draft

    @functools.wraps(original)
    def sample_draft(*args, **kwargs):
        tokens = original(*args, **kwargs)
        if _in_replicated_mtp_batch():
            tokens = _broadcast_draft_tokens(tokens, group)
        # The caller writes this value to draft_tokens, then uses it as the
        # next MTP substep's input. Synchronizing only propose() is too late.
        return tokens

    speculator.sample_draft = sample_draft
    speculator._hcu_glm53_pcp_mtp_sync_group = group


def install_glm53_pcp_mtp_sync(runner) -> bool:
    from vllm_hcu.patch.platform.core_fix.patch_vllm_config import is_glm53_pcp

    config = runner.vllm_config
    speculator = getattr(runner, "speculator", None)
    spec = config.speculative_config
    if (not is_glm53_pcp(config) or speculator is None or spec is None
            or spec.method != "mtp"):
        return False
    if config.parallel_config.prefill_context_parallel_size <= 1:
        return False
    flag = os.environ.get("VLLM_HCU_GLM53_PCP_MTP_SYNC", "1")
    if flag not in ("0", "1"):
        raise ValueError("VLLM_HCU_GLM53_PCP_MTP_SYNC must be 0 or 1")
    if flag == "0":
        return False
    # Keep this first fix within the diagnosed execution contract. In
    # particular, greedy-token broadcast alone is insufficient when each rank
    # also caches a potentially different probabilistic draft distribution.
    if not config.model_config.enforce_eager:
        raise ValueError("GLM53 PCP MTP sync v1 requires enforce_eager=True")
    if getattr(spec, "draft_sample_method", "greedy") != "greedy":
        raise ValueError("GLM53 PCP MTP sync v1 supports greedy draft sampling only")
    if getattr(spec, "enable_adaptive_verification", False):
        raise ValueError("GLM53 PCP MTP sync v1 does not cover adaptive verification")

    from vllm.distributed import get_pcp_group

    group = get_pcp_group()
    if group.world_size != config.parallel_config.prefill_context_parallel_size:
        raise RuntimeError("configured PCP size and process group disagree")
    _install_draft_sync(speculator, group)
    # sample_tokens() temporarily hides runner.pcp_manager after restoring the
    # global batch. Do NOT use the manager as the runtime on/off condition.
    runner._hcu_glm53_pcp_mtp_sync_group = group
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "GLM53 PCP MTP decision sync enabled: PCP=%d, source=PCP rank0, "
        "per-draft-step tokens and pre-commit sampler decisions.",
        group.world_size,
    )
    return True


def sync_glm53_pcp_sampler_output(runner, out):
    group = getattr(runner, "_hcu_glm53_pcp_mtp_sync_group", None)
    if group is None or not _in_replicated_mtp_batch():
        return out
    output, num_sampled, num_rejected = out
    sampled = output.sampled_token_ids
    if num_sampled is None or num_rejected is None:
        raise RuntimeError("MTP sampler must provide sampled and rejected counts")
    if sampled.ndim != 2:
        raise ValueError("expected [num_reqs, max_sampled_tokens] sampler output")
    n, width = sampled.shape
    if num_sampled.shape != (n,) or num_rejected.shape != (n,):
        raise ValueError("sampled/rejected counts do not match the global batch")
    if sampled.device != num_sampled.device or sampled.device != num_rejected.device:
        raise ValueError("sampler decisions must be on the same device")
    if n == 0:
        return out

    # One collective commits the token list AND its counts atomically at this
    # program point. Unused output columns may be uninitialized: mask them in
    # the payload rather than reading/sending arbitrary padding as decisions.
    columns = torch.arange(width, device=sampled.device)
    valid_tokens = sampled.masked_fill(columns[None, :] >= num_sampled[:, None], -1)
    payload = torch.cat((valid_tokens.reshape(-1).to(torch.int64),
                         num_sampled.to(torch.int64), num_rejected.to(torch.int64)))
    payload = group.broadcast(payload, src=0)
    end = n * width
    # Preserve the original buffers: the tuple and SamplerOutput may alias
    # these tensors, and the subsequent post_update() must see the same data.
    sampled.copy_(payload[:end].view(n, width))
    num_sampled.copy_(payload[end:end+n])
    num_rejected.copy_(payload[end+n:end+2*n])
    output.num_sampled = num_sampled
    output.num_rejected = num_rejected
    return out
