# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Real image peak and resident encoder-cache profiling for HYV4-VL MRV2."""

from contextlib import contextmanager
from weakref import ref

import torch


def account_hyv4_vl_state_memory(runner):
    """Add retained MM-state torch bytes before worker profiling snapshots.

    Native load_model records model_memory_usage *before* constructing the
    model state. The worker subtracts the already-resident torch allocation
    baseline during profiling, so merely preallocating MM buffers loses them
    from automatic KV sizing. The HYV4-VL state records its actual retained
    allocation delta; count it as persistent memory exactly once per state.
    """
    if tuple(runner.model_config.architectures) != (
        "HYV4VLForConditionalGeneration",
    ):
        return
    state = runner.model_state
    memory_bytes = getattr(state, "_hcu_vl_persistent_memory_bytes", None)
    if type(memory_bytes) is not int or memory_bytes < 0:
        raise ValueError("HYV4-VL state requires a nonnegative persistent memory ledger")
    previous = getattr(runner, "_hcu_vl_accounted_state", None)
    if previous is not None and previous() is state:
        return
    # A strong reference would retain the previous model state on model reload.
    state_ref = ref(state)
    runner.model_memory_usage += memory_bytes
    runner._hcu_vl_accounted_state = state_ref


@contextmanager
def profile_hyv4_vl_vision(runner):
    """Hold budgeted image outputs/cache storage through decoder profiling.

    Enter *before* applying the PCP decoder-token reduction. Vision is replicated
    at TP1, so neither its inputs nor its cache budget is divided by PCP/DP.
    Native model-state buffers must already be allocated and included in the
    runner's persistent-memory accounting; this context never duplicates them.

    The native registry selects the configured worst-case resolution, and
    MultiModalBudget selects the scheduler's maximum image batch. Reserve the
    cache remainder before encoding: reserve + expected outputs equals
    encoder_cache_size tokens. An already-live cache is preserved but is part
    of the worker's pre-profile baseline, so it cannot replace the reserve.
    In that unusual case physical profiling overhead is conservatively bounded
    by its validated live token count (at most cache_size - new_output_tokens).
    Startup normally has no live cache and therefore no such extra overhead.
    No temporary entry is inserted into the live cache, whose MRV2
    reset_mm_cache implementation is currently a no-op.
    """
    if tuple(runner.model_config.architectures) != (
        "HYV4VLForConditionalGeneration",
    ):
        yield
        return

    mm_config = runner.model_config.multimodal_config
    if mm_config.skip_mm_profiling:
        raise ValueError("HYV4-VL MRV2 requires skip_mm_profiling=False")
    if mm_config.enable_mm_embeds:
        raise ValueError("HYV4-VL MRV2 profiling supports image inputs only")
    speculative = runner.vllm_config.speculative_config
    if speculative is not None and (
        speculative.method != "mtp"
        or runner.vllm_config.parallel_config.prefill_context_parallel_size != 16
    ):
        raise ValueError("HYV4-VL MRV2 profiling supports only native PCP16 MTP")
    accounted = getattr(runner, "_hcu_vl_accounted_state", None)
    if accounted is None or accounted() is not runner.model_state:
        raise ValueError("HYV4-VL state memory must be accounted before worker profiling")

    from vllm.multimodal.encoder_budget import MultiModalBudget

    budget = MultiModalBudget(runner.vllm_config, runner.mm_registry)
    reserve = outputs = None
    try:
        if not budget.mm_max_toks_per_item:
            # Explicitly disabled image inputs have no encoder/cache budget.
            if budget.encoder_cache_size:
                raise ValueError("HYV4-VL encoder cache budget has no image bound")
            yield
            return
        if set(budget.mm_max_toks_per_item) != {"image"}:
            raise ValueError("HYV4-VL MRV2 profiling supports image inputs only")

        count = budget.mm_max_items_per_batch["image"]
        max_tokens = budget.mm_max_toks_per_item["image"]
        if count <= 0 or max_tokens <= 0:
            raise ValueError("HYV4-VL image profiling requires positive bounds")
        dummy = runner.mm_registry.get_dummy_mm_inputs(
            runner.model_config, mm_counts={"image": 1}, cache=budget.cache
        )
        items = dummy["mm_kwargs"]["image"]
        placeholders = dummy["mm_placeholders"]["image"]
        if len(items) != 1 or items[0] is None or len(placeholders) != 1:
            raise ValueError("HYV4-VL registry must provide one uncached dummy image")
        tokens = placeholders[0].get_num_embeds()
        if tokens != max_tokens:
            raise ValueError(
                f"HYV4-VL dummy image tokens {tokens} differ from bound {max_tokens}"
            )
        new_tokens = count * tokens
        if new_tokens > budget.get_encoder_budget():
            raise ValueError("HYV4-VL dummy images exceed encoder/cache budget")

        hidden_size = runner.model_config.get_hidden_size()
        live_tokens = 0
        for value in runner.encoder_cache.encoder_outputs.values():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[1] != hidden_size
                or value.dtype != runner.dtype
                or value.device != runner.device
                or value.untyped_storage().nbytes()
                != value.numel() * value.element_size()
            ):
                raise ValueError("HYV4-VL live encoder output has an invalid layout")
            live_tokens += value.shape[0]
        reserve_tokens = budget.encoder_cache_size - new_tokens
        if live_tokens > reserve_tokens:
            raise ValueError("HYV4-VL live outputs and profile exceed cache budget")

        with torch.inference_mode():
            reserve = torch.empty(
                (reserve_tokens, hidden_size), dtype=runner.dtype, device=runner.device
            )
            outputs = runner.model_state.encoder_runner.execute_mm_encoder(
                [("image", items[0])] * count
            )
            if (
                not isinstance(outputs, (list, tuple, torch.Tensor))
                or len(outputs) != count
            ):
                raise ValueError(
                    "HYV4-VL encoder output cardinality differs from dummy batch"
                )
            if any(
                not isinstance(value, torch.Tensor)
                or value.shape != (tokens, hidden_size)
                or value.dtype != runner.dtype
                or value.device != runner.device
                or value.untyped_storage().nbytes()
                != value.numel() * value.element_size()
                for value in outputs
            ):
                # HYV4 currently projects each image separately: its transposed
                # result owns exactly one image's storage. A future batched-view
                # encoder needs explicit shared-storage budget accounting first.
                raise ValueError(
                    "HYV4-VL encoder output differs from budgeted shape/dtype/device"
                )
            # Include any encoder side-stream work before decoder profiling.
            torch.accelerator.synchronize()
            yield
    finally:
        try:
            # Also drain failed encoding/decoding before releasing owned storage.
            torch.accelerator.synchronize()
        finally:
            outputs = None
            del reserve
            budget.reset_cache()
