"""Fuse the greedy DSv4.1 Markov loop on gfx938."""

from __future__ import annotations

import functools
import logging
from types import ModuleType

import torch

from ..core_fix._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
PATCH_ID = "worker.op_opt.dspark.greedy_markov"
_MARKER = "_vllm_hcu_greedy_markov_applied"
logger = logging.getLogger(__name__)


@functools.cache
def _log_selected():
    logger.info("DSv4.1 DSpark: using fused greedy Markov sampling on gfx938")


def _eligible(speculator, num_reqs, head_hidden):
    model = speculator.model
    if (
        type(model).__module__ != "vllm.models.deepseek_v41.amd.dspark"
        or type(model).__name__ != "DSparkDeepseekV4ForCausalLM"
        or speculator.draft_logits is not None
        or speculator._draft_topk is not None
        or speculator.use_confidence_head
        or speculator.acceptance_estimator is not None
        or speculator.draft_watermarker is not None
        or model.draft_id_to_target_id is not None
        or not 1 <= num_reqs <= 16
        or not 1 <= speculator.num_speculative_steps <= 8
        or head_hidden.device.type != "cuda"
        or head_hidden.dtype != torch.bfloat16
        or torch.version.hip is None
    ):
        return False
    processor = model.logits_processor
    if (
        processor.scale != 1.0 or processor.soft_cap is not None
        or processor.logits_as_input
        or processor.head_dtype not in (None, torch.bfloat16)
    ):
        return False
    markov = model.model.markov_head
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    projection = markov.markov_w2
    if (
        not isinstance(projection.quant_method, UnquantizedEmbeddingMethod)
        or not projection.disable_tp
        or getattr(projection, "bias", None) is not None
    ):
        return False
    weights = (markov.markov_w1.weight, projection.weight)
    if any(
        weight.ndim != 2 or weight.shape[1] != 256
        or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
        or weight.device != head_hidden.device
        for weight in weights
    ):
        return False
    if weights[0].shape[0] != model.config.vocab_size or weights[1].shape[0] < model.config.vocab_size:
        return False
    return torch.cuda.get_device_properties(head_hidden.device).gcnArchName.split(":")[0] == "gfx938"


def _try_sample(speculator, num_reqs, head_hidden):
    if not _eligible(speculator, num_reqs, head_hidden):
        return False
    from vllm_hcu.model_executor.layers.dsv41_greedy_markov import greedy_markov

    steps = speculator.num_speculative_steps
    sample_hidden = head_hidden[speculator.sample_indices[:num_reqs * steps]]
    base_logits = speculator.model.compute_draft_logits(sample_hidden)
    if base_logits.dtype != torch.bfloat16 or base_logits.shape != (num_reqs * steps, speculator.model.config.vocab_size):
        return False
    markov = speculator.model.model.markov_head
    greedy_markov(
        base_logits.view(num_reqs, steps, -1),
        markov.markov_w1.weight, markov.markov_w2.weight,
        speculator.input_buffers.input_ids, speculator._anchor_idx,
        speculator.draft_tokens,
    )
    _log_selected()
    return True


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "DSparkSpeculator", f"{TARGET_MODULE}.DSparkSpeculator")
    original = require_callable(cls, "_sample_sequential", f"{TARGET_MODULE}._sample_sequential")
    if getattr(cls, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError("stale DSpark greedy Markov patch")
        return False
    require_exact_signature(
        original, f"{TARGET_MODULE}.DSparkSpeculator._sample_sequential",
        positional=("self", "num_reqs", "head_hidden"),
    )

    @functools.wraps(original)
    def sample_sequential(self, num_reqs, head_hidden):
        if not _try_sample(self, num_reqs, head_hidden):
            return original(self, num_reqs, head_hidden)

    setattr(sample_sequential, _MARKER, True)
    cls._sample_sequential = sample_sequential
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
