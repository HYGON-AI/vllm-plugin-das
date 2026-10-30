# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Full-vocabulary W8A8 head made from the checkpoint's original BF16 head.

Only the head is converted. The vocabulary shard, its padding, the checkpoint
loader and LogitsProcessor retain their vLLM contracts. H256 rotates the K
dimension; output channels are never pruned or remapped.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm_hcu.model_executor.layers.quantization.int8_runtime import apply_int8_linear

logger = init_logger(__name__)
BLOCK_SIZE = 256
_WEIGHT_CHUNK_ROWS = 4096


class ConvRotLMHeadError(RuntimeError):
    """An explicitly selected ConvRot head cannot preserve its load contract."""


def _require_hcu_backend(weight: torch.Tensor) -> None:
    if not weight.is_cuda or torch.version.hip is None:
        raise ConvRotLMHeadError("ConvRot INT8 lm_head requires an HCU/HIP device")
    from vllm_hcu.platforms import envs as henvs

    if not henvs.custom_quantization_gemm_enabled():
        raise ConvRotLMHeadError(
            "lm_head_convrot_int8 requires VLLM_HCU_USE_CUSTOM_OPS and "
            "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM"
        )


def _hadamard(device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < BLOCK_SIZE:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return (matrix / math.sqrt(BLOCK_SIZE)).to(device=device, dtype=torch.bfloat16)


def _rotate(value: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    return torch.matmul(value.reshape(-1, BLOCK_SIZE), hadamard).reshape(value.shape)


class ConvRotINT8LMHeadMethod(QuantizeMethodBase):
    """Load BF16 through the original method, then convert exactly once.

    No weight Tensor is cached on the method: MTP sharing and device movement
    continue to use the layer's registered storage. Pre-processed/IPC weight
    caches are deliberately not advertised as supported by this online format.
    """

    # Lets other modules detect this method without importing this module.
    hcu_convrot_lm_head = True

    def __init__(self, original_method: QuantizeMethodBase):
        self.original_method = original_method

    def create_weights(self, layer: nn.Module, *args, **kwargs):
        self.original_method.create_weights(layer, *args, **kwargs)

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module):
        raise ConvRotLMHeadError(
            "ConvRot INT8 lm_head does not support tied input embeddings"
        )

    @torch.no_grad()
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight = layer.weight
        if getattr(layer, "_hcu_convrot_ready", False) and weight.dtype == torch.int8:
            return
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ConvRotLMHeadError("ConvRot INT8 expects the original 2D BF16 lm_head")
        n, k = weight.shape
        if n == 0 or k == 0 or k % BLOCK_SIZE:
            raise ConvRotLMHeadError("ConvRot INT8 requires nonempty weights and K divisible by 256")
        if torch.isnan(weight).all().item():
            # A head the checkpoint omitted is left as an all-NaN placeholder
            # that the MTP loader later replaces with the (converted) target
            # head. Converting it would only fail; leave it for the sharing.
            logger.warning(
                "lm_head_convrot_int8: head is an all-NaN placeholder, "
                "skipping conversion (expected to be replaced by head sharing)"
            )
            return
        _require_hcu_backend(weight)
        hadamard = _hadamard(weight.device)
        quantized = torch.empty_like(weight, dtype=torch.int8)
        scales = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
        # Bound conversion transients without making a full FP32 copy of W.
        for start in range(0, n, _WEIGHT_CHUNK_ROWS):
            end = min(start + _WEIGHT_CHUNK_ROWS, n)
            rotated = _rotate(weight[start:end], hadamard).float()
            if not torch.isfinite(rotated).all().item():
                raise ConvRotLMHeadError("BF16 lm_head contains non-finite values")
            scale = (rotated.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min_(1e-8)
            quantized[start:end] = (rotated / scale).round().clamp_(-127, 127).to(torch.int8)
            scales[start:end] = scale
        # Publish only after all chunks succeeded. Retain the original Parameter
        # and its weight_loader/TP metadata; vLLM owns loading/reloading it.
        weight.requires_grad_(False)
        weight.data = quantized
        layer.register_parameter("weight_scale", nn.Parameter(scales, requires_grad=False))
        layer.register_buffer("_hcu_convrot_hadamard", hadamard)
        layer._hcu_convrot_ready = True
        # The BF16 shard was just freed; return its cached blocks to the device
        # so the memory profiler (and thus the KV cache budget) sees the saving.
        torch.cuda.empty_cache()
        logger.info(
            "ConvRot INT8 lm_head: converted BF16 shard [%d, %d], block=256, "
            "full vocabulary retained", n, k,
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        if not getattr(layer, "_hcu_convrot_ready", False):
            raise ConvRotLMHeadError("ConvRot INT8 lm_head was not finalized after loading")
        if x.ndim < 2 or x.shape[-1] != layer.weight.shape[1] or x.dtype != torch.bfloat16:
            raise ConvRotLMHeadError("ConvRot INT8 requires BF16 hidden states matching head K")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], layer.weight.shape[0]))
        rotated = _rotate(x, layer._hcu_convrot_hadamard)
        return apply_int8_linear(
            input=rotated, weight=layer.weight, weight_scale=layer.weight_scale,
            params_dtype=torch.bfloat16, bias=bias,
        )


def _unsupported_reason(layer: nn.Module, vllm_config: object) -> str | None:
    """Why this head cannot be converted, or None when it can."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    if type(layer.quant_method) not in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod):
        return "the head is not an unquantized BF16 head"
    if type(layer.quant_method) is UnquantizedLinearMethod:
        from vllm_hcu.platforms import envs as henvs

        if henvs.VLLM_USE_NN:
            return "UnquantizedLinearMethod with VLLM_USE_NN uses the NN weight layout"
    if layer.weight.dtype != torch.bfloat16:
        return f"head weight dtype is {layer.weight.dtype}, expected torch.bfloat16"
    if layer.embedding_dim % BLOCK_SIZE:
        return f"hidden size {layer.embedding_dim} is not divisible by {BLOCK_SIZE}"
    expected_shape = (layer.num_embeddings_per_partition, layer.embedding_dim)
    if tuple(layer.weight.shape) != expected_shape:
        return (
            f"weight shape {tuple(layer.weight.shape)} is not "
            f"[vocab_shard, hidden_size] = {expected_shape}"
        )
    if getattr(layer, "weight_scale", None) is not None:
        return "the head already has weight scales"
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if (
        getattr(text_config, "tie_word_embeddings", False)
        or getattr(model_config, "word_embeddings_untied_by_checkpoint", False)
    ):
        return "the head is tied to the input embeddings"
    if getattr(model_config, "head_dtype", None) not in (None, torch.bfloat16):
        return "head_dtype is not BF16"
    if getattr(vllm_config, "lora_config", None) is not None:
        return "LoRA is enabled"
    return None


def install_convrot_lm_head(layer: nn.Module, vllm_config: object) -> bool:
    """Switch one head to ConvRot INT8; True when the head is (now) converted.

    A head this format cannot serve keeps its original BF16 method and a warning
    is logged, so one unsupported head (tied, quantized, odd hidden size, ...)
    does not abort model construction. Errors in the post-load conversion itself
    still raise, because the head has been selected by then.
    """
    if isinstance(layer.quant_method, ConvRotINT8LMHeadMethod):
        return True
    reason = _unsupported_reason(layer, vllm_config)
    if reason is not None:
        logger.warning(
            "lm_head_convrot_int8 is not applied to this head, keeping BF16: %s",
            reason,
        )
        return False
    layer.quant_method = ConvRotINT8LMHeadMethod(layer.quant_method)
    return True
