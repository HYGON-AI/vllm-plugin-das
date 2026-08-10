# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lightly-CP and multi-layer-MTP behavior for the v0.25.1 base proposer."""

from __future__ import annotations

from typing import Any

from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config


def initialize_proposer(
    module: object,
    proposer: object,
    vllm_config: object,
    device: object,
    runner: object | None,
) -> HcuFeatureConfig:
    """Attach HCU state after the complete official (including ROCm) init."""

    config = get_hcu_config(vllm_config)
    proposer._hcu_feature_config = config
    proposer.runner = runner

    # HCU FlashAttention and FlashMLA sparse both support speculative tokens as
    # decode queries, but target vLLM v0.25.1 does not include their metadata
    # classes in the ROCm multi-step drafting allowlist. Keep the target
    # proposer and extend only the HCU-owned backend capability registration.
    from vllm_hcu.v1.attention.backends.flash_attn import FlashAttentionMetadata

    hcu_attn_types = [FlashAttentionMetadata]
    try:
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseMetadata,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "vllm.v1.attention.backends.mla.flashmla_sparse":
            raise
    else:
        hcu_attn_types.append(FlashMLASparseMetadata)

    allowed_attn_types = getattr(proposer, "allowed_attn_types", None)
    if allowed_attn_types is not None:
        proposer.allowed_attn_types += tuple(
            metadata_type
            for metadata_type in hcu_attn_types
            if metadata_type not in proposer.allowed_attn_types
        )

    proposer.enable_multi_layers_mtp = config.enable_multi_layers_mtp
    proposer.enable_lightly_cp = config.enable_lightly_cp
    proposer.enable_lightly_cplb = (
        config.enable_lightly_cp and config.enable_lightly_cplb
    )
    proposer.scatter_indexes_tensor = None
    proposer.gather_indexes_tensor = None

    if config.enable_lightly_cp and runner is None:
        raise RuntimeError("enable_lightly_cp requires a model-runner-backed proposer")

    if config.enable_lightly_cplb:
        # Legacy used the undefined local max_batch_size here.  Grow every
        # proposer buffer that is actually indexed by request count using the
        # initialized self.max_batch_size instead.
        proposer.max_batch_size = vllm_config.scheduler_config.max_num_seqs * 2
        proposer.backup_next_token_ids = module.CpuGpuBuffer(
            proposer.max_batch_size,
            dtype=module.torch.int32,
            pin_memory=module.is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        required_arange = max(proposer.max_batch_size + 1, proposer.max_num_tokens)
        if proposer.arange.numel() < required_arange:
            proposer.arange = module.torch.arange(
                required_arange, device=device, dtype=module.torch.int32
            )

    if config.enable_lightly_cp:
        proposer.query_start_loc = module.CpuGpuBuffer(
            proposer.max_batch_size + 1,
            dtype=module.torch.int32,
            pin_memory=module.is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        proposer.seq_lens = module.CpuGpuBuffer(
            proposer.max_batch_size,
            dtype=module.torch.int32,
            pin_memory=module.is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
    return config


def _lightly_cp_active(proposer: object, num_tokens: int) -> bool:
    threshold = getattr(proposer.runner, "lightly_cp_threshold", None)
    if threshold is None:
        raise RuntimeError(
            "enable_lightly_cp requires runner.lightly_cp_threshold at runtime"
        )
    return bool(proposer.enable_lightly_cp and num_tokens > int(threshold))


def propose(
    module: object,
    proposer: object,
    num_speculative_tokens: int,
    target_token_ids: object,
    target_positions: object,
    target_hidden_states: object,
    next_token_ids: object,
    token_indices_to_sample: object | None,
    common_attn_metadata: object,
    sampling_metadata: object,
    mm_embed_inputs: object | None = None,
    num_rejected_tokens_gpu: object | None = None,
    slot_mappings: object | None = None,
):
    """v0.25.1 proposer algorithm with the Lightly-CP chain kept atomic."""

    torch = module.torch
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer._last_draft_probs = None
    batch_size = common_attn_metadata.batch_size()

    if proposer.method in ("eagle3", "dflash"):
        assert isinstance(
            proposer.model,
            (
                module.Eagle3LlamaForCausalLM,
                module.Eagle3DeepseekV2ForCausalLM,
                module.DFlashQwen3ForCausalLM,
            ),
        )
        target_hidden_states = proposer.model.combine_hidden_states(
            target_hidden_states
        )
        assert target_hidden_states.shape[-1] == proposer.hidden_size

    num_tokens, token_indices_to_sample, common_attn_metadata = (
        proposer.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
    )

    enable_lightly_cp = _lightly_cp_active(proposer, num_tokens)
    if enable_lightly_cp:
        from vllm_hcu.v1.attention.lightly_cp_utils import (
            pad_for_mla_cp,
            prepare_cp_metadata,
        )

        actual_num_tokens = num_tokens
        num_tokens = pad_for_mla_cp(num_tokens)
        common_attn_metadata = prepare_cp_metadata(
            num_reqs_padded=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            num_tokens=actual_num_tokens,
            block_table_gid_0=common_attn_metadata.block_table_tensor,
            slot_mapping_gid_0=common_attn_metadata.slot_mapping,
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            query_start_loc_buf=proposer.query_start_loc,
            seq_lens_buf=proposer.seq_lens,
            enable_lightly_cplb=proposer.enable_lightly_cplb,
        )
        proposer.scatter_indexes_tensor = common_attn_metadata.scatter_indexes_tensor
        proposer.gather_indexes_tensor = common_attn_metadata.gather_indexes_tensor
    else:
        proposer.scatter_indexes_tensor = None
        proposer.gather_indexes_tensor = None

    per_group_attn_metadata, per_layer_attn_metadata = (
        proposer.build_per_group_and_layer_attn_metadata(common_attn_metadata)
    )
    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
        proposer._determine_batch_execution_and_padding(num_tokens)
    )
    model_kwargs, slot_mapping_size = proposer.build_model_inputs_first_pass(
        num_tokens, num_input_tokens, mm_embed_inputs
    )

    with module.set_forward_context(
        per_layer_attn_metadata,
        proposer.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=proposer._get_slot_mapping(
            slot_mapping_size, common_attn_metadata.slot_mapping
        ),
        scatter_indexes_tensor=proposer.scatter_indexes_tensor,
        gather_indexes_tensor=proposer.gather_indexes_tensor,
        enable_lightly_cp=enable_lightly_cp,
        enable_lightly_cplb=(
            enable_lightly_cp and proposer.enable_lightly_cplb
        ),
    ):
        ret_hidden_states = proposer.model(**model_kwargs)
        if not proposer.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    if proposer.num_speculative_tokens == 1 or proposer.parallel_drafting:
        draft_token_ids = proposer._greedy_sample(sample_hidden_states)
        return draft_token_ids.view(-1, proposer.num_speculative_tokens)

    if proposer.uses_mrope:
        positions = proposer.mrope_positions[:, token_indices_to_sample]
    else:
        positions = proposer.positions[token_indices_to_sample]
    hidden_states = hidden_states[token_indices_to_sample]
    if proposer.constant_draft_positions:
        proposer.positions[:batch_size] = positions
    draft_token_ids = proposer._greedy_sample(sample_hidden_states)

    if proposer.allowed_attn_types is not None:
        for group_metadata in per_group_attn_metadata:
            if not isinstance(group_metadata, proposer.allowed_attn_types):
                raise ValueError(
                    "Unsupported attention metadata type for speculative decoding "
                    f"with num_speculative_tokens > 1: {type(group_metadata)}. "
                    f"Supported types are: {proposer.allowed_attn_types}"
                )

    draft_token_ids_list = [draft_token_ids]
    cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
        proposer._determine_batch_execution_and_padding(batch_size)
    )
    if enable_lightly_cp:
        common_attn_metadata = common_attn_metadata.cp_common_metadata
        if common_attn_metadata is None:
            raise RuntimeError("Lightly-CP metadata lost its canonical CP view")

    common_attn_metadata.num_actual_tokens = batch_size
    common_attn_metadata.num_kv_actual_tokens = batch_size
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.query_start_loc = proposer.arange[: batch_size + 1]
    common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
        proposer.token_arange_np[: batch_size + 1]
    ).clone()

    if proposer.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata._num_computed_tokens_cpu = None

    block_size = proposer.block_size
    assert block_size > 0, "block_size has not been initialized."
    for token_index in range(proposer.num_speculative_tokens - 1):
        input_ids = draft_token_ids_list[-1].int()
        if not proposer.constant_draft_positions:
            positions = proposer._update_positions_dependent_metadata(
                positions,
                common_attn_metadata,
                batch_size,
                input_batch_size,
                block_size,
            )
        if not proposer.constant_draft_positions or token_index == 0:
            _, per_layer_attn_metadata = (
                proposer.build_per_group_and_layer_attn_metadata(
                    common_attn_metadata, draft_index=token_index + 1
                )
            )

        proposer.input_ids[:batch_size] = input_ids
        proposer.hidden_states[:batch_size] = hidden_states
        if proposer.supports_mm_inputs:
            proposer.inputs_embeds[:batch_size] = proposer.model.embed_input_ids(
                input_ids
            )
            input_ids = None
            inputs_embeds = proposer.inputs_embeds[:input_batch_size]
        else:
            input_ids = proposer.input_ids[:input_batch_size]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": proposer._get_positions(input_batch_size),
            "inputs_embeds": inputs_embeds,
        }
        if proposer.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = proposer.hidden_states[:input_batch_size]

        with module.set_forward_context(
            per_layer_attn_metadata,
            proposer.vllm_config,
            num_tokens=input_batch_size,
            num_tokens_across_dp=batch_size_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=proposer._get_slot_mapping(input_batch_size),
            enable_lightly_cp=False,
            enable_lightly_cplb=False,
        ):
            ret_hidden_states = proposer.model(**model_kwargs)
            if not proposer.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        hidden_states = hidden_states[:batch_size]
        draft_token_ids = proposer._greedy_sample(last_hidden_states[:batch_size])
        draft_token_ids_list.append(draft_token_ids)

    return torch.stack(draft_token_ids_list, dim=1)


def preserve_multi_layer_mtp_heads(
    proposer: object,
    target_language_model: object,
    original_method: Any,
) -> Any:
    """Keep independently trained MTP heads; share missing/duplicate heads."""

    torch = __import__("torch")
    inner = getattr(proposer.model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    saved: list[tuple[object, object]] = []
    if layers is not None:
        items = layers.values() if hasattr(layers, "values") else layers
        for layer in items:
            shared_head = getattr(layer, "shared_head", None)
            if shared_head is not None and hasattr(shared_head, "head"):
                saved.append((shared_head, shared_head.head))

    result = original_method(proposer, target_language_model)
    target_head = getattr(target_language_model, "lm_head", None)
    target_weight = getattr(target_head, "weight", None)
    for shared_head, original_head in saved:
        original_weight = getattr(original_head, "weight", None)
        has_own_trained_weights = (
            isinstance(original_weight, torch.Tensor)
            and isinstance(target_weight, torch.Tensor)
            and not torch.isnan(original_weight).any()
            and not torch.equal(original_weight.cpu(), target_weight.cpu())
        )
        if has_own_trained_weights:
            shared_head.head = original_head
    return result


def pad_for_sequence_parallelism(proposer: object, num_tokens: int) -> int:
    config = getattr(proposer, "_hcu_feature_config", None)
    if config is None:
        config = get_hcu_config(proposer.vllm_config)
    enable_sp = bool(proposer.compilation_config.pass_config.enable_sp)
    if not enable_sp and not config.enable_custom_sp:
        return num_tokens
    tp_size = proposer.vllm_config.parallel_config.tensor_parallel_size
    if tp_size <= 1:
        return num_tokens
    from vllm.utils.math_utils import round_up

    return round_up(num_tokens, tp_size)


__all__ = [
    "initialize_proposer",
    "pad_for_sequence_parallelism",
    "preserve_multi_layer_mtp_heads",
    "propose",
]
