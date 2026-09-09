# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt Channel-FP8 loading and correctness fallbacks for GLM5Next."""

from __future__ import annotations

import functools
import sys
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.glm5next.nvidia.model"
ATTENTION_MODULE = "vllm.models.glm5next.nvidia.attention"
KPOOL_MODULE = "vllm.model_executor.layers.sparse_attn_indexer_kpool"
MTP_MODULE = "vllm.models.glm5next.nvidia.mtp"
PATCH_ID = "worker.core_fix.glm5next.channel_fp8_attn_projection_loader"
TARGET_SYMBOL = f"{TARGET_MODULE}._try_load_fp8_attn_proj"
_PATCH_MARKER = "_vllm_hcu_glm5next_channel_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_glm5next_channel_fp8_wrapper"
_INDEXER_PATCH_MARKER = "_vllm_hcu_glm5next_indexer_nn_layout_applied"
_INDEXER_WRAPPER_MARKER = "_vllm_hcu_glm5next_indexer_nn_layout_wrapper"
_KPOOL_PATCH_MARKER = "_vllm_hcu_sparse_indexer_kpool_triton_applied"
_KPOOL_WRAPPER_MARKER = "_vllm_hcu_sparse_indexer_kpool_triton_wrapper"
_INDEXER_CACHE_PATCH_MARKER = "_vllm_hcu_glm5next_indexer_cache_applied"
_INDEXER_CACHE_WRAPPER_MARKER = "_vllm_hcu_glm5next_indexer_cache_wrapper"
_QUANT_IGNORE_PATCH_MARKER = "_vllm_hcu_glm5next_quant_ignore_applied"
_QUANT_IGNORE_WRAPPER_MARKER = "_vllm_hcu_glm5next_quant_ignore_wrapper"
_MHC_PATCH_MARKER = "_vllm_hcu_glm5next_boltops_mhc_applied"
_MHC_WRAPPER_MARKER = "_vllm_hcu_glm5next_boltops_mhc_wrapper"


def _native_mhc_pre(
    mhc,
    op,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    post_mix, comb_mix, layer_input = op.forward_native(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        norm_weight,
        norm_eps,
    )
    return (
        post_mix,
        comb_mix,
        mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps),
    )


def _native_mhc_post(mhc, op, x, residual, post_layer_mix, comb_res_mix):
    del mhc
    return op.forward_native(x, residual, post_layer_mix, comb_res_mix)


def _native_mhc_fused_post_pre(
    mhc,
    op,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    residual_cur, post_mix, comb_mix, layer_input = op.forward_native(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )
    return (
        residual_cur,
        post_mix,
        comb_mix,
        mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps),
    )


def _bind_glm5next_native_mhc(layer, mhc) -> None:
    """Keep GLM5Next on the official native mHC equations on HCU."""

    layer.mhc_pre_op._forward_method = functools.partial(
        _native_mhc_pre, mhc, layer.mhc_pre_op
    )
    layer.mhc_post_op._forward_method = functools.partial(
        _native_mhc_post, mhc, layer.mhc_post_op
    )
    layer.mhc_fused_post_pre_op._forward_method = functools.partial(
        _native_mhc_fused_post_pre,
        mhc,
        layer.mhc_fused_post_pre_op,
    )


def _boltops_mhc_pre(
    backend,
    mhc,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    post_mix, comb_mix, layer_input = backend.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    return (
        post_mix,
        comb_mix,
        mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps),
    )


def _boltops_mhc_post(
    backend,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
):
    return backend.mhc_post(x, residual, post_layer_mix, comb_res_mix)


def _boltops_mhc_fused_post_pre(
    backend,
    mhc,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    residual_cur, post_mix, comb_mix, layer_input = backend.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
    )
    return (
        residual_cur,
        post_mix,
        comb_mix,
        mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps),
    )


def _bind_glm5next_boltops_mhc(layer, mhc, backend) -> None:
    """Bind GLM5Next to BoltOPs while preserving the official main ABI."""

    layer.mhc_pre_op._forward_method = functools.partial(
        _boltops_mhc_pre, backend, mhc
    )
    layer.mhc_post_op._forward_method = functools.partial(
        _boltops_mhc_post, backend
    )
    layer.mhc_fused_post_pre_op._forward_method = functools.partial(
        _boltops_mhc_fused_post_pre,
        backend,
        mhc,
    )


def _patch_glm5next_boltops_mhc(glm_model: ModuleType) -> bool:
    decoder_cls = vars(glm_model).get("Glm5NextDecoderLayer")
    if not isinstance(decoder_cls, type):
        raise PatchCompatibilityError(
            f"required class {TARGET_MODULE}.Glm5NextDecoderLayer is missing"
        )
    original = require_callable(
        decoder_cls,
        "__init__",
        f"{TARGET_MODULE}.Glm5NextDecoderLayer.__init__",
    )
    if getattr(decoder_cls, _MHC_PATCH_MARKER, False):
        if not getattr(original, _MHC_WRAPPER_MARKER, False):
            raise PatchCompatibilityError("GLM5Next BoltOPs-mHC patch marker is stale")
        return False

    @functools.wraps(original)
    def hcu_decoder_init(
        self,
        vllm_config,
        config,
        layer_idx,
        prefix="",
        topk_indices_buffer=None,
        is_mtp_layer=False,
        **kwargs,
    ):
        original(
            self,
            vllm_config,
            config,
            layer_idx,
            prefix,
            topk_indices_buffer,
            is_mtp_layer,
            **kwargs,
        )
        if getattr(self, "mhc", False) and not getattr(
            self, "is_mtp_layer", False
        ):
            from vllm.model_executor.layers import mhc
            from vllm_hcu.model_executor.layers import mhc as boltops_mhc

            _bind_glm5next_boltops_mhc(self, mhc, boltops_mhc)

    setattr(hcu_decoder_init, _MHC_WRAPPER_MARKER, True)
    setattr(decoder_cls, "_vllm_hcu_original_init", original)
    setattr(decoder_cls, "__init__", hcu_decoder_init)
    setattr(decoder_cls, _MHC_PATCH_MARKER, True)
    return True


def _expand_glm5next_multimodal_ignore_aliases(ignore: list[str]) -> list[str]:
    """Add runtime aliases for checkpoint-rooted GLM5Next regex ignores."""
    expanded = list(ignore)
    checkpoint_prefix = "re:^model\\.layers\\."
    runtime_prefix = "re:^language_model\\.model\\.layers\\."
    for pattern in ignore:
        if pattern.startswith(checkpoint_prefix):
            alias = runtime_prefix + pattern[len(checkpoint_prefix) :]
            if alias not in expanded:
                expanded.append(alias)
    return expanded


def _patch_multimodal_quant_ignore(glm_model: ModuleType) -> bool:
    model_cls = vars(glm_model).get("Glm5NextForConditionalGeneration")
    if not isinstance(model_cls, type):
        raise PatchCompatibilityError(
            f"required class {TARGET_MODULE}.Glm5NextForConditionalGeneration "
            "is missing"
        )
    original = require_callable(
        model_cls,
        "__init__",
        f"{TARGET_MODULE}.Glm5NextForConditionalGeneration.__init__",
    )
    if getattr(model_cls, _QUANT_IGNORE_PATCH_MARKER, False):
        if not getattr(original, _QUANT_IGNORE_WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                "GLM5Next quant-ignore patch marker is stale"
            )
        return False
    require_exact_signature(
        original,
        f"{TARGET_MODULE}.Glm5NextForConditionalGeneration.__init__",
        positional=("self",),
        keyword_only=("vllm_config", "prefix"),
        defaults={"prefix": ""},
    )

    @functools.wraps(original)
    def hcu_init(self, *, vllm_config, prefix=""):
        quant_config = getattr(vllm_config, "quant_config", None)
        if (
            quant_config is not None
            and callable(getattr(quant_config, "get_name", None))
            and quant_config.get_name() == "compressed-tensors"
            and isinstance(getattr(quant_config, "ignore", None), list)
        ):
            quant_config.ignore = _expand_glm5next_multimodal_ignore_aliases(
                quant_config.ignore
            )
        return original(self, vllm_config=vllm_config, prefix=prefix)

    setattr(hcu_init, _QUANT_IGNORE_WRAPPER_MARKER, True)
    setattr(model_cls, "_vllm_hcu_original_init", original)
    setattr(model_cls, "__init__", hcu_init)
    setattr(model_cls, _QUANT_IGNORE_PATCH_MARKER, True)
    return True


def _patch_glm5next_indexer_cache(attention: ModuleType) -> bool:
    cache_cls = vars(attention).get("Glm5NextIndexerCache")
    if not isinstance(cache_cls, type):
        raise PatchCompatibilityError(
            f"required class {ATTENTION_MODULE}.Glm5NextIndexerCache is missing"
        )
    original = require_callable(
        cache_cls,
        "get_kv_cache_spec",
        f"{ATTENTION_MODULE}.Glm5NextIndexerCache.get_kv_cache_spec",
    )
    if getattr(cache_cls, _INDEXER_CACHE_PATCH_MARKER, False):
        if not getattr(original, _INDEXER_CACHE_WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                "GLM5Next indexer cache patch marker is stale"
            )
        return False
    require_exact_signature(
        original,
        f"{ATTENTION_MODULE}.Glm5NextIndexerCache.get_kv_cache_spec",
        positional=("self", "vllm_config"),
    )
    base_get_kv_cache_spec = require_callable(
        cache_cls.__mro__[1],
        "get_kv_cache_spec",
        f"{cache_cls.__mro__[1].__name__}.get_kv_cache_spec",
    )
    require_exact_signature(
        base_get_kv_cache_spec,
        f"{cache_cls.__mro__[1].__name__}.get_kv_cache_spec",
        positional=("self", "vllm_config"),
    )

    @functools.wraps(original)
    def hcu_get_kv_cache_spec(self, vllm_config):
        from dataclasses import replace

        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        spec = base_get_kv_cache_spec(self, vllm_config)
        if not isinstance(spec, MLAAttentionSpec):
            raise PatchCompatibilityError(
                "GLM5Next indexer cache parent returned a non-MLA spec"
            )
        # The official override adds a DeepGEMM-only 32/64-state page
        # constraint. HCU uses the ROCm generic indexer path, whose cache page
        # width is the natural block_size // index_kpool value.
        return replace(spec, tokens_per_state=self._index_kpool)

    setattr(hcu_get_kv_cache_spec, _INDEXER_CACHE_WRAPPER_MARKER, True)
    setattr(cache_cls, "_vllm_hcu_original_get_kv_cache_spec", original)
    setattr(cache_cls, "get_kv_cache_spec", hcu_get_kv_cache_spec)
    setattr(cache_cls, _INDEXER_CACHE_PATCH_MARKER, True)
    return True


def _patch_sparse_indexer_kpool(kpool: ModuleType) -> bool:
    indexer = vars(kpool).get("SparseAttnIndexerKpool")
    if not isinstance(indexer, type):
        raise PatchCompatibilityError(
            f"required class {KPOOL_MODULE}.SparseAttnIndexerKpool is missing"
        )
    original = require_callable(
        indexer,
        "forward_hip",
        f"{KPOOL_MODULE}.SparseAttnIndexerKpool.forward_hip",
    )
    if getattr(indexer, _KPOOL_PATCH_MARKER, False):
        if not getattr(original, _KPOOL_WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                "sparse indexer kpool Triton fallback patch marker is stale"
            )
        return False
    require_exact_signature(
        original,
        f"{KPOOL_MODULE}.SparseAttnIndexerKpool.forward_hip",
        positional=("self", "hidden_states", "q_quant", "k", "weights"),
        keyword_only=("gate_score", "compress_ape", "index_kpool", "positions"),
        defaults={
            "gate_score": None,
            "compress_ape": None,
            "index_kpool": 1,
            "positions": None,
        },
    )

    @functools.wraps(original)
    def hcu_forward_hip(
        self,
        hidden_states,
        q_quant,
        k,
        weights,
        *,
        gate_score=None,
        compress_ape=None,
        index_kpool=1,
        positions=None,
    ):
        if index_kpool > 1 and not kpool.rocm_aiter_ops.is_enabled():
            # Upstream already delegates kpool>1 to this implementation when
            # AITER is enabled.  The implementation is Triton on ROCm and does
            # not require the upstream AITER sparse-indexer extension.
            return self.forward_cuda(
                hidden_states,
                q_quant,
                k,
                weights,
                gate_score=gate_score,
                compress_ape=compress_ape,
                index_kpool=index_kpool,
                positions=positions,
            )
        return original(
            self,
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    setattr(hcu_forward_hip, _KPOOL_WRAPPER_MARKER, True)
    setattr(indexer, "_vllm_hcu_original_forward_hip", original)
    setattr(indexer, "forward_hip", hcu_forward_hip)
    setattr(indexer, _KPOOL_PATCH_MARKER, True)
    return True


def _patch_indexer_nn_layout(attention: ModuleType) -> bool:
    indexer = vars(attention).get("Indexer")
    if not isinstance(indexer, type):
        raise PatchCompatibilityError(
            f"required class {ATTENTION_MODULE}.Indexer is missing"
        )
    original = require_callable(
        indexer,
        "forward",
        f"{ATTENTION_MODULE}.Indexer.forward",
    )
    if getattr(indexer, _INDEXER_PATCH_MARKER, False):
        if not getattr(original, _INDEXER_WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                "GLM5 Indexer NN-layout patch marker is stale"
            )
        return False
    require_exact_signature(
        original,
        f"{ATTENTION_MODULE}.Indexer.forward",
        positional=("self", "hidden_states", "qr", "positions", "rotary_emb"),
    )

    @functools.wraps(original)
    def hcu_indexer_forward(self, hidden_states, qr, positions, rotary_emb):
        if self._wp_fp32 is None:
            weight = self.wk_weights_proj.weight.data
            hidden_size = hidden_states.shape[-1]
            output_size = self.head_dim + self.n_head
            if tuple(weight.shape) == (hidden_size, output_size):
                # HCU NN linear storage is [in, out].  Official Indexer.forward
                # directly reads the parameter as [out, in] to cache the gate
                # projection, bypassing the layout-aware linear implementation.
                self._wp_fp32 = (
                    weight[:, self.head_dim :].contiguous().float()
                )
        return original(self, hidden_states, qr, positions, rotary_emb)

    setattr(hcu_indexer_forward, _INDEXER_WRAPPER_MARKER, True)
    setattr(indexer, "_vllm_hcu_original_forward", original)
    setattr(indexer, "forward", hcu_indexer_forward)
    setattr(indexer, _INDEXER_PATCH_MARKER, True)
    return True


def _projection(module: ModuleType, name: str):
    for suffix, info in module._FP8_ATTN_PROJS.items():
        if suffix in name:
            layer_prefix = name.rsplit(suffix, 1)[0]
            key, target_base, shard_id, is_kva = info
            target = f"{layer_prefix}.{target_base}"
            return layer_prefix, key, target, shard_id, is_kva
    return None


def _dequantize_channel_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(
            f"Channel-FP8 weight must be 2-D, got {tuple(weight.shape)}"
        )
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    if scale.shape != (weight.shape[0], 1):
        raise ValueError(
            "Channel-FP8 scale must have shape (out_features, 1), got "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )
    return (weight.float() * scale.float()).to(torch.bfloat16).contiguous()


def apply_to_module(module: ModuleType) -> bool:
    glm_model = load_exact_module(TARGET_MODULE, module)
    changed = _patch_multimodal_quant_ignore(glm_model)
    changed = _patch_glm5next_boltops_mhc(glm_model) or changed
    kpool = sys.modules.get(KPOOL_MODULE)
    if isinstance(kpool, ModuleType):
        changed = _patch_sparse_indexer_kpool(kpool)
    attention = sys.modules.get(ATTENTION_MODULE)
    if isinstance(attention, ModuleType):
        changed = _patch_glm5next_indexer_cache(attention) or changed
        changed = _patch_indexer_nn_layout(attention) or changed
    original = require_callable(
        glm_model,
        "_try_load_fp8_attn_proj",
        TARGET_SYMBOL,
    )
    if getattr(glm_model, _PATCH_MARKER, False):
        current = vars(glm_model).get("_try_load_fp8_attn_proj")
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return changed
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=(
            "name",
            "tensor",
            "buf",
            "params_dict",
            "loaded_params",
            "kv_a_pad_size",
        ),
    )

    @functools.wraps(original)
    def hcu_try_load_fp8_attn_proj(
        name,
        tensor,
        buf,
        params_dict,
        loaded_params,
        kv_a_pad_size,
    ):
        projection = _projection(glm_model, name)
        if projection is None:
            return original(
                name,
                tensor,
                buf,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            )

        layer_prefix, key, target, shard_id, is_kva = projection
        target_weight = f"{target}.weight"
        target_scale = f"{target}.weight_scale"
        target_scale_inv = f"{target}.weight_scale_inv"
        is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
        is_channel_scale = name.endswith(".weight_scale")

        # A target that remains quantized owns both tensors and must use the
        # normal official stacked/direct loading path.
        if target_scale in params_dict or target_scale_inv in params_dict:
            if is_weight or is_channel_scale:
                return False

        if not is_weight and not is_channel_scale:
            return original(
                name,
                tensor,
                buf,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            )

        entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
        entry["weight" if is_weight else "scale"] = tensor
        if "weight" not in entry or "scale" not in entry:
            return True

        weight_bf16 = _dequantize_channel_fp8(entry["weight"], entry["scale"])
        buf[layer_prefix].pop(key, None)
        if not buf[layer_prefix]:
            buf.pop(layer_prefix, None)

        if is_kva and kv_a_pad_size > 0:
            weight_bf16 = torch.nn.functional.pad(
                weight_bf16,
                (0, 0, 0, kv_a_pad_size),
            )

        param = params_dict[target_weight]
        if shard_id is None:
            param.weight_loader(param, weight_bf16)
        else:
            param.weight_loader(param, weight_bf16, shard_id)
        loaded_params.add(target_weight)
        return True

    setattr(hcu_try_load_fp8_attn_proj, _WRAPPER_MARKER, True)
    setattr(glm_model, "_vllm_hcu_original_try_load_fp8_attn_proj", original)
    setattr(glm_model, "_try_load_fp8_attn_proj", hcu_try_load_fp8_attn_proj)
    setattr(glm_model, _PATCH_MARKER, True)

    # MTP imports the helper by value. Refresh an already-imported module;
    # later imports naturally observe the patched model-module attribute.
    mtp = sys.modules.get(MTP_MODULE)
    if isinstance(mtp, ModuleType) and getattr(
        mtp, "_try_load_fp8_attn_proj", None
    ) is original:
        mtp._try_load_fp8_attn_proj = hcu_try_load_fp8_attn_proj
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "ATTENTION_MODULE",
    "KPOOL_MODULE",
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
]
