# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Replace only V4.1's decode method, retaining the original fallback.

The model owns KV caches and index-source selection. This boundary additionally
exposes the per-step, per-layer-type scheduler that the shared decode op lacks.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_class,
    require_exact_signature,
    require_unpatched,
)
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime

TARGET_MODULE = "vllm.models.deepseek_v41.amd.rocm"
PATCH_ID = "worker.op_opt.deepseek_v41.flashmla_sparse_decode"
_CLASS_TARGET = f"{TARGET_MODULE}.DeepseekV41ROCMAiterMLAAttention"
_INIT_TARGET = f"{_CLASS_TARGET}.__init__"
_DECODE_TARGET = f"{_CLASS_TARGET}._forward_decode"
TARGETS = (_INIT_TARGET, _DECODE_TARGET)
_MARKER = "_vllm_hcu_dsv41_flashmla_decode_applied"
_WRAPPER = "_vllm_hcu_dsv41_flashmla_decode_wrapper"
_INIT_WRAPPER = "_vllm_hcu_dsv41_flashmla_decode_init_wrapper"
_CONTEXT_FLAG = "_vllm_hcu_flashmla_decode_context_supported"


def _config_supported(config) -> bool:
    parallel = config.parallel_config
    return (
        parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and config.speculative_config is None
    )


def _context_supported(attention) -> bool:
    # MRV2 executes warmup and regular forwards without a process-global
    # set_current_vllm_config context. Topology/speculation are init invariants.
    return getattr(attention, _CONTEXT_FLAG, False)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "DeepseekV41ROCMAiterMLAAttention", _CLASS_TARGET)
    wrapped = (
        (cls, "__init__", _INIT_TARGET, _INIT_WRAPPER),
        (cls, "_forward_decode", _DECODE_TARGET, _WRAPPER),
    )
    if already_applied(cls, _MARKER, wrapped):
        return False
    original_init = require_unpatched(cls, "__init__", _INIT_TARGET, _INIT_WRAPPER)
    require_exact_signature(
        original_init, _INIT_TARGET, positional=("self",),
        var_positional="args", var_keyword="kwargs",
    )
    original = require_unpatched(cls, "_forward_decode", _DECODE_TARGET, _WRAPPER)
    require_exact_signature(
        original, _DECODE_TARGET,
        positional=("self", "q", "kv_cache", "swa_metadata", "attn_metadata", "swa_only", "output"),
    )

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        config = args[0] if args else kwargs["vllm_config"]
        original_init(self, *args, **kwargs)
        setattr(self, _CONTEXT_FLAG, _config_supported(config))

    @functools.wraps(original)
    def decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
        if henvs.dsv41_flashmla_decode_enabled():
            if _context_supported(self) and self.compress_ratio in (0, 1, 2):
                scheduler_name = "tile_sched_swaonly" if swa_only else {1: "tile_sched_c1a", 2: "tile_sched_c2a"}.get(self.compress_ratio)
                num_tokens = swa_metadata.num_decode_tokens
                source = None if swa_only else self.topk_indices_buffer
                if swa_only or (
                    source is not None and attn_metadata is not None and kv_cache is not None
                    and self.compress_ratio in (1, 2)
                    and kv_cache.shape[1] == attn_metadata.block_size // self.compress_ratio
                    and swa_metadata.token_to_req_indices is not None
                    and swa_metadata.is_valid_token is not None
                ):
                    def native_arguments(output_buffer):
                        return dict(
                            q=q,
                            swa_cache=self.swa_cache_layer.kv_cache,
                            kv_cache=None if swa_only else kv_cache,
                            swa_indices=swa_metadata.decode_swa_indices,
                            swa_lens=swa_metadata.decode_swa_lens,
                            scheduler=getattr(swa_metadata, scheduler_name, None),
                            scale=self.scale,
                            attn_sink=self.attn_sink,
                            output=output_buffer,
                            head_dim=self.head_dim,
                            nope_head_dim=self.nope_head_dim,
                            rope_head_dim=self.rope_head_dim,
                            max_decode_query_len=swa_metadata.max_decode_query_len,
                            kv_cache_dtype=self.kv_cache_dtype,
                            topk_indices=None if swa_only else source[:num_tokens],
                            block_table=(
                                None
                                if swa_only
                                else attn_metadata.block_table[:swa_metadata.num_decodes]
                            ),
                            token_to_req_indices=(
                                None
                                if swa_only
                                else swa_metadata.token_to_req_indices[:num_tokens]
                            ),
                            is_valid_token=(
                                None
                                if swa_only
                                else swa_metadata.is_valid_token[:num_tokens]
                            ),
                        )

                    native_args = native_arguments(output)
                    if runtime.try_flashmla_decode(**native_args):
                        return None
            else:
                runtime._log_fallback("speculative decoding or context parallelism is not validated")
        return original(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output)

    setattr(decode, _WRAPPER, True)
    setattr(init, _INIT_WRAPPER, True)
    cls._vllm_hcu_original_flashmla_decode_init = original_init
    cls.__init__ = init
    cls._vllm_hcu_original_forward_decode = original
    cls._forward_decode = decode
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
