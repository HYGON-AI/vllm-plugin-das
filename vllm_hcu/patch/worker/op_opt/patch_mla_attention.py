# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Strict v0.25.1 MLAAttention runtime adapter."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

from ._common import PatchCompatibilityError, already_applied, load_exact_module, require_callable, require_class, require_exact_signature

TARGET_MODULE = "vllm.model_executor.layers.attention.mla_attention"
PATCH_ID = "worker.op_opt.mla.attention_runtime"
TARGETS = (
    f"{TARGET_MODULE}.MLAAttention.__init__",
    f"{TARGET_MODULE}.MLAAttention.forward",
    f"{TARGET_MODULE}.MLAAttention.forward_impl",
    f"{TARGET_MODULE}.MLAAttention.process_weights_after_loading",
    f"{TARGET_MODULE}.MLACommonMetadata.__init__",
    f"{TARGET_MODULE}.MLACommonMetadataBuilder.build",
    f"{TARGET_MODULE}.split_decodes_and_prefills",
)
_MARKER = "_vllm_hcu_mla_attention_applied"
_WRAPPER = "_vllm_hcu_mla_attention_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    mla = load_exact_module(TARGET_MODULE, module)
    cls = require_class(mla, "MLAAttention", f"{TARGET_MODULE}.MLAAttention")
    metadata_cls = require_class(mla, "MLACommonMetadata", f"{TARGET_MODULE}.MLACommonMetadata")
    builder_cls = require_class(mla, "MLACommonMetadataBuilder", f"{TARGET_MODULE}.MLACommonMetadataBuilder")
    wrapped = (
        (cls, "__init__", TARGETS[0], _WRAPPER),
        (cls, "forward", TARGETS[1], _WRAPPER),
        (cls, "forward_impl", TARGETS[2], _WRAPPER),
        (cls, "process_weights_after_loading", TARGETS[3], _WRAPPER),
        (metadata_cls, "__init__", TARGETS[4], _WRAPPER),
        (builder_cls, "build", TARGETS[5], _WRAPPER),
        (mla, "split_decodes_and_prefills", TARGETS[6], _WRAPPER),
    )
    if already_applied(mla, _MARKER, wrapped):
        return False
    original_init = require_callable(cls, "__init__", TARGETS[0])
    require_exact_signature(
        original_init, TARGETS[0],
        positional=("self", "num_heads", "scale", "qk_nope_head_dim", "qk_rope_head_dim",
                    "v_head_dim", "q_lora_rank", "kv_lora_rank", "kv_b_proj",
                    "cache_config", "quant_config", "prefix", "attn_backend",
                    "use_sparse", "indexer", "topk_indices_buffer"),
        defaults={"cache_config": None, "quant_config": None, "prefix": "",
                  "attn_backend": None, "use_sparse": False, "indexer": None,
                  "topk_indices_buffer": None},
        var_keyword="extra_impl_args",
    )
    original_full_forward = require_callable(cls, "forward", TARGETS[1])
    require_exact_signature(
        original_full_forward,
        TARGETS[1],
        positional=("self", "q", "kv_c_normed", "k_pe", "output_shape"),
        defaults={"output_shape": None},
    )
    original_forward = require_callable(cls, "forward_impl", TARGETS[2])
    require_exact_signature(
        original_forward, TARGETS[2],
        positional=("self", "q", "k_c_normed", "k_pe", "kv_cache", "attn_metadata",
                    "output", "output_scale", "output_block_scale", "quant_group_size",
                    "quant_scale_ue8m0", "quant_col_major", "quant_tma_aligned"),
        defaults={"output_scale": None, "output_block_scale": None,
                  "quant_group_size": None, "quant_scale_ue8m0": None,
                  "quant_col_major": None, "quant_tma_aligned": None},
    )
    process = require_callable(cls, "process_weights_after_loading", TARGETS[3])
    require_exact_signature(process, TARGETS[3], positional=("self", "act_dtype"))
    metadata_init = require_callable(metadata_cls, "__init__", TARGETS[4])
    if "num_actual_tokens" not in inspect.signature(metadata_init).parameters:
        raise PatchCompatibilityError(
            f"required target {TARGETS[4]} has incompatible fields"
        )
    builder = require_callable(builder_cls, "build", TARGETS[5])
    require_exact_signature(
        builder, TARGETS[5],
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
        defaults={"fast_build": False},
    )
    split_batch = require_callable(mla, "split_decodes_and_prefills", TARGETS[6])
    require_exact_signature(
        split_batch,
        TARGETS[6],
        positional=(
            "common_attn_metadata",
            "decode_threshold",
            "require_uniform",
            "treat_short_extends_as_decodes",
        ),
        defaults={
            "decode_threshold": 1,
            "require_uniform": False,
            "treat_short_extends_as_decodes": True,
        },
    )
    get_forward_context = require_callable(
        mla,
        "get_forward_context",
        f"{TARGET_MODULE}.get_forward_context",
    )
    encode_layer_name = require_callable(
        mla,
        "_encode_layer_name",
        f"{TARGET_MODULE}._encode_layer_name",
    )
    torch_module = getattr(mla, "torch", None)
    if torch_module is None:
        raise PatchCompatibilityError(
            f"required HCU patch dependency {TARGET_MODULE}.torch is missing"
        )

    def configured_pcp_world_size(vllm_config) -> int:
        if vllm_config is None:
            return 1
        parallel_config = getattr(vllm_config, "parallel_config", None)
        value = getattr(parallel_config, "prefill_context_parallel_size", None)
        if value is None:
            raise PatchCompatibilityError(
                "required vLLM 0.25.1 prefill_context_parallel_size is missing"
            )
        world_size = int(value)
        if world_size < 1:
            raise PatchCompatibilityError(
                f"invalid prefill_context_parallel_size={world_size}"
            )
        return world_size

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        self._hcu_feature_config = get_hcu_config(vllm_config)
        self._hcu_pcp_world_size = configured_pcp_world_size(vllm_config)
        self._hcu_use_pcp = self._hcu_pcp_world_size > 1
        if self._hcu_use_pcp and self._hcu_feature_config.enable_lightly_cp:
            raise RuntimeError("HCU PCP and lightly-CP cannot be enabled together")
        if self._hcu_use_pcp:
            # PCP is eager-only in this backport.  Keep the registered opaque
            # custom-op graph untouched and take the direct Python path.
            self.use_direct_call = True

    @functools.wraps(original_full_forward)
    def hcu_full_forward(
        self,
        q,
        kv_c_normed,
        k_pe,
        output_shape=None,
    ):
        if not getattr(self, "_hcu_use_pcp", False):
            return original_full_forward(
                self,
                q,
                kv_c_normed,
                k_pe,
                output_shape,
            )

        if self.calculate_kv_scales:
            torch_module.ops.vllm.maybe_calc_kv_scales(
                q,
                kv_c_normed,
                k_pe,
                encode_layer_name(self.layer_name),
            )

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw[self.layer_name]
        elif isinstance(attn_metadata_raw, list):
            attn_metadata = attn_metadata_raw[0][self.layer_name]
        else:
            attn_metadata = attn_metadata_raw
        if attn_metadata is not None:
            slot_mapping = forward_context.slot_mapping
            if not isinstance(slot_mapping, dict):
                raise RuntimeError(
                    "PCP MLA direct forward requires per-layer slot mappings"
                )
            layer_slot_mapping = slot_mapping.get(self.layer_name)
            if layer_slot_mapping is None:
                raise RuntimeError(
                    "PCP MLA slot mapping is missing for layer "
                    f"{self.layer_name!r}"
                )
            metadata_world_size = int(
                getattr(attn_metadata, "pcp_world_size", 1)
            )
            if metadata_world_size != self._hcu_pcp_world_size:
                raise RuntimeError(
                    "PCP MLA metadata world size mismatch: "
                    f"layer={self._hcu_pcp_world_size}, "
                    f"metadata={metadata_world_size}"
                )

            from vllm_hcu.model_executor.layers.attention.pcp import (
                maybe_gather_mla_latent_cache_inputs,
            )

            kv_for_cache, kpe_for_cache, layer_slot_mapping = (
                maybe_gather_mla_latent_cache_inputs(
                    kv_c_normed,
                    k_pe,
                    layer_slot_mapping,
                    attn_metadata,
                )
            )
            self.impl.do_kv_cache_update(
                kv_for_cache,
                kpe_for_cache,
                self.kv_cache,
                layer_slot_mapping,
                self.kv_cache_dtype,
                self._k_scale,
            )
        output = torch_module.empty(output_shape, dtype=q.dtype, device=q.device)
        self.forward_impl(
            q,
            kv_c_normed,
            k_pe,
            self.kv_cache,
            attn_metadata,
            output=output,
        )
        return output

    @functools.wraps(original_forward)
    def hcu_forward(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output,
                    output_scale=None, output_block_scale=None, quant_group_size=None,
                    quant_scale_ue8m0=None, quant_col_major=None, quant_tma_aligned=None):
        config = getattr(self, "_hcu_feature_config", None)
        if config is None:
            raise RuntimeError("HCU MLA feature config was not initialized")
        if not config.enable_lightly_cp:
            return original_forward(
                self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output,
                output_scale, output_block_scale, quant_group_size,
                quant_scale_ue8m0, quant_col_major, quant_tma_aligned,
            )
        from vllm_hcu.model_executor.layers.mla_runtime import mla_forward_impl

        return mla_forward_impl(
            mla, self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output,
            output_scale, output_block_scale, quant_group_size,
            quant_scale_ue8m0, quant_col_major, quant_tma_aligned,
        )

    @functools.wraps(process)
    def hcu_process(self, act_dtype):
        from vllm_hcu.platforms import envs as henvs

        if not henvs.VLLM_USE_NN:
            return process(self, act_dtype)
        from vllm_hcu.model_executor.layers.mla_runtime import mla_process_weights_nn

        return mla_process_weights_nn(mla, self, act_dtype)

    @functools.wraps(metadata_init)
    def hcu_metadata_init(self, *args, **kwargs):
        num_kv = kwargs.pop("num_kv_actual_tokens", None)
        metadata_init(self, *args, **kwargs)
        self.num_kv_actual_tokens = (
            self.num_actual_tokens if num_kv is None else num_kv
        )

    @functools.wraps(builder)
    def hcu_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = builder(self, common_prefix_len, common_attn_metadata, fast_build)
        result.num_kv_actual_tokens = getattr(
            common_attn_metadata,
            "num_kv_actual_tokens",
            common_attn_metadata.num_actual_tokens,
        )
        result.pcp_world_size = configured_pcp_world_size(
            getattr(self, "vllm_config", None)
        )
        return result

    @functools.wraps(split_batch)
    def hcu_split_batch(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        # Warmup/capture placeholders do not carry request phase metadata.
        # Preserve the target vLLM decode classification in that case, while
        # retaining HCU's prefix-hit short-extend policy for real requests.
        del treat_short_extends_as_decodes
        hcu_treat_short_extends_as_decodes = (
            getattr(common_attn_metadata, "is_prefilling", None) is None
        )
        return split_batch(
            common_attn_metadata,
            decode_threshold,
            require_uniform,
            hcu_treat_short_extends_as_decodes,
        )

    for function in (
        hcu_init,
        hcu_full_forward,
        hcu_forward,
        hcu_process,
        hcu_metadata_init,
        hcu_build,
        hcu_split_batch,
    ):
        setattr(function, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_init", original_init)
    setattr(cls, "_vllm_hcu_original_forward", original_full_forward)
    setattr(cls, "_vllm_hcu_original_forward_impl", original_forward)
    setattr(cls, "_vllm_hcu_original_process_weights", process)
    setattr(metadata_cls, "_vllm_hcu_original_init", metadata_init)
    setattr(builder_cls, "_vllm_hcu_original_build", builder)
    setattr(mla, "_vllm_hcu_original_split_decodes_and_prefills", split_batch)
    setattr(cls, "__init__", hcu_init)
    setattr(cls, "forward", hcu_full_forward)
    setattr(cls, "forward_impl", hcu_forward)
    setattr(cls, "process_weights_after_loading", hcu_process)
    setattr(metadata_cls, "__init__", hcu_metadata_init)
    setattr(builder_cls, "build", hcu_build)
    setattr(mla, "split_decodes_and_prefills", hcu_split_batch)
    setattr(mla, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
