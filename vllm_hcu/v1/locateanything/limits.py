# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared physical/probe/logical context contract for LocateAnything Fast."""

PROBE_TOKENS = 6


def reject_speculative_decoding(speculative_config) -> None:
    if speculative_config is not None:
        raise ValueError("LocateAnything Fast uses six-position block generation and does not support vLLM speculative decoding")


def reject_pipeline_parallelism(pipeline_parallel_size: int) -> None:
    if pipeline_parallel_size != 1:
        raise ValueError("LocateAnything Fast requires pipeline_parallel_size=1; pipeline parallelism is not supported")


def validate_fast_config(config) -> None:
    """Fail before device setup on branches outside the qualified Fast profile."""
    reject_speculative_decoding(config.speculative_config)
    parallel = config.parallel_config
    reject_pipeline_parallelism(parallel.pipeline_parallel_size)
    for name in ('tensor_parallel_size', 'data_parallel_size',
                 'prefill_context_parallel_size', 'decode_context_parallel_size'):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f'LocateAnything Fast requires {name}=1; only the single-device profile is qualified')
    if getattr(parallel, 'enable_dbo', False) or getattr(parallel, 'ubatch_size', 0) > 1:
        raise ValueError('LocateAnything Fast does not support DBO or microbatching')
    model = config.model_config
    if getattr(model.hf_config, 'model_type', None) != 'locateanything':
        raise ValueError('LocateAnything Fast requires a LocateAnything checkpoint')
    if getattr(model, 'enforce_eager', False):
        raise ValueError('LocateAnything Fast requires graph execution; enforce_eager must be False')
    if str(model.dtype) not in ('torch.bfloat16', 'bfloat16'):
        raise ValueError('LocateAnything Fast requires bfloat16 weights')
    if getattr(model, 'runner_type', 'generate') != 'generate':
        raise ValueError('LocateAnything Fast supports the generate runner only')
    for name in ('logits_processors', 'return_sampling_mask', 'enable_trace_replay',
                 'enable_return_routed_experts', 'enable_prompt_embeds'):
        if getattr(model, name, None):
            raise ValueError(f'LocateAnything Fast does not support model_config.{name}')
    if config.quant_config is not None or config.lora_config is not None:
        raise ValueError('LocateAnything Fast does not support quantization or LoRA')
    if config.kv_transfer_config is not None or config.ec_transfer_config is not None:
        raise ValueError('LocateAnything Fast does not support KV or encoder transfer')
    cache = config.cache_config
    if cache.enable_prefix_caching:
        raise ValueError('LocateAnything Fast requires enable_prefix_caching=False')
    if cache.cache_dtype not in ('auto', 'bfloat16'):
        raise ValueError('LocateAnything Fast requires an unquantized bfloat16 KV cache')
    if not config.use_v2_model_runner:
        raise ValueError('LocateAnything Fast requires Model Runner V2')
    backend = config.attention_config.backend
    if getattr(backend, 'name', backend) != 'TRITON_ATTN':
        raise ValueError('LocateAnything Fast requires attention backend TRITON_ATTN for its probe mask')
    mode = config.compilation_config.cudagraph_mode
    if mode is not None and 'FULL' not in getattr(mode, 'name', str(mode)):
        raise ValueError('LocateAnything Fast requires FULL decode graphs')
    scheduling = config.scheduler_config
    if scheduling.async_scheduling or scheduling.enable_chunked_prefill:
        raise ValueError('LocateAnything Fast requires synchronous scheduling without chunked prefill')
    if scheduling.long_prefill_token_threshold != 0:
        raise ValueError('LocateAnything Fast requires long_prefill_token_threshold=0')
    limit = logical_context_limit(model.max_model_len)
    budgets = [scheduling.max_num_batched_tokens]
    if scheduling.max_num_scheduled_tokens is not None:
        budgets.append(scheduling.max_num_scheduled_tokens)
    if min(budgets) < limit + PROBE_TOKENS - 1:
        raise ValueError('LocateAnything Fast token budgets must fit a complete prompt plus six probes')


def logical_context_limit(max_model_len: int) -> int:
    """Reserve probes and leave room for a nonempty prompt and output."""
    limit = max_model_len - PROBE_TOKENS
    if limit < 2:
        raise ValueError("LocateAnything Fast max-model-len must be at least 8 (including six probe slots)")
    return limit


def context_error(max_model_len: int, prompt_tokens: int, max_new_tokens: int | None) -> str | None:
    limit = logical_context_limit(max_model_len)
    if prompt_tokens >= limit or (max_new_tokens is not None and prompt_tokens + max_new_tokens > limit):
        return (f"Fast logical context is {limit} tokens for max-model-len={max_model_len}; "
                "six physical probe slots are reserved separately")
    return None
