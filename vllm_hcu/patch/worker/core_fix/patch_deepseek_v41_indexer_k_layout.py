# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Write V4.1 indexer keys in the row-major layout used by LightOp paged MQA."""

from __future__ import annotations

import functools
import importlib
import os
from types import ModuleType

import torch

from ._common import load_exact_module, require_callable, require_exact_signature
from vllm_hcu.model_executor.layers.fp4_indexer import (
    Fp4QTuple,
    fp4_physical_page_size,
    fp4_indexer_requested,
    rotate_indexer_query_bf16,
)

TARGET_MODULE = "vllm.models.deepseek_v41.attention"
PATCH_ID = "worker.core_fix.deepseek_v41.indexer_k_lightop_layout"
_MARKER = "_vllm_hcu_indexer_k_lightop_layout_applied"
_FLAG = "VLLM_HCU_DSV41_LIGHTOP_PAGED_NORMAL"
_Q_MARKER = "_vllm_hcu_v41_fp4_q_adapter_applied"


def use_normal_indexer_k_layout() -> bool:
    return os.environ.get(_FLAG, "0").lower() in ("1", "true")


def _fp4_domain_blockers(module: ModuleType, vllm_config=None) -> tuple[str, ...]:
    """Experimental V4.1 domain, selected once before cache allocation.

    DP ranks own independent request/cache metadata. DSpark(5) uses the
    upstream per-token flattened decode metadata on HIP; no CP/PP/PD or
    padded-native multi-token ABI is admitted here.
    """
    reasons = []
    if not fp4_indexer_requested():
        reasons.append("FP4 indexer opt-in is OFF")
    if vllm_config is None:
        vllm_config = getattr(module, "get_current_vllm_config", lambda: None)()
    config = getattr(vllm_config, "model_config", None)
    hf = getattr(config, "hf_text_config", None)
    if hf is None:
        hf = getattr(config, "hf_config", None)
        hf = getattr(hf, "text_config", hf)
    ratios = tuple(int(x) for x in (getattr(hf, "compress_ratios", ()) or ()))
    if not ratios or any(r not in (0, 1, 2) for r in ratios):
        reasons.append("only V4.1 compress_ratios in {0,1,2} are supported")
    if not getattr(hf, "kv_source_layer_ids", None) or not getattr(
        hf, "index_source_layer_ids", None
    ):
        reasons.append("shared-source layer IDs are missing")
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is not None and any(
        int(getattr(parallel, name, 1)) != 1
        for name in ("prefill_context_parallel_size", "decode_context_parallel_size",
                      "pipeline_parallel_size", "tensor_parallel_size")
    ):
        reasons.append("FP4 requires TP1/PP1/PCP1/DCP1")
    if int(getattr(parallel, "data_parallel_size", 1)) not in (1, 8):
        reasons.append("only DP1 or experimental DP8 is admitted")
    speculative = getattr(vllm_config, "speculative_config", None)
    method = getattr(speculative, "method", None)
    if method and not (
        method == "dspark"
        and getattr(speculative, "num_speculative_tokens", 0) == 5
        and not getattr(speculative, "enable_adaptive_verification", False)
    ):
        reasons.append("only non-adaptive DSpark(5) speculative decode is admitted")
    attention = getattr(vllm_config, "attention_config", None)
    if getattr(attention, "indexer_sparse_logits", False):
        reasons.append("sparse-logits backend is not supported by this dense FP4 adapter")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        # 真实 PD 仍不支持：FP4 indexer cache 是 68B/记录的独立格式，没有 P→D
        # 传输契约，P 端写 FP4、D 端读 FP8（或反之）会静默错位。
        # DecodeBenchConnector 例外：它不跨节点传输，只在本机用 dummy 值填充
        # KV cache 来模拟“已完成 prefill”，因此不涉及该契约。它仅用于 decode
        # 性能测试，不能用于正确性验证（dummy KV 的 logits/top-k 无语义）。
        kv_connector = getattr(
            getattr(vllm_config, "kv_transfer_config", None), "kv_connector", None
        )
        if kv_connector != "DecodeBenchConnector":
            reasons.append("PD KV transfer has no FP4 indexer cache contract")
    return tuple(reasons)


def _fp4_domain_requested(module: ModuleType, vllm_config=None) -> bool:
    return not _fp4_domain_blockers(module, vllm_config)


def _is_v41(vllm_config) -> bool:
    model = getattr(vllm_config, "model_config", None)
    hf = getattr(model, "hf_config", None)
    return getattr(hf, "model_type", None) == "deepseek_v41"


def _install_fp4_forward(target: ModuleType) -> None:
    """Keep both Q representations across V4.1's three-value return boundary.

    The upstream forward unpacks (packed, scales) and its attention caller
    reconstructs a *plain* tuple, discarding Fp4QTuple.bf16. Return the adapter
    itself as index_q with index_q_scale=None so that caller passes it intact.
    OFF delegates to the original method, including its short-context path.
    """
    cls = target.DeepseekV4Indexer
    original = require_callable(cls, "forward", f"{TARGET_MODULE}.DeepseekV4Indexer.forward")
    require_exact_signature(
        original, f"{TARGET_MODULE}.DeepseekV4Indexer.forward",
        positional=("self", "qr", "latent", "indexer_weights", "positions", "rotary_emb", "qr_scale"),
        defaults={"qr_scale": None},
    )
    if getattr(original, "_vllm_hcu_fp4_forward", False):
        return

    @functools.wraps(original)
    def forward(self, qr, latent, indexer_weights, positions, rotary_emb, qr_scale=None):
        if not (self.use_fp4_kv and fp4_indexer_requested()):
            return original(self, qr, latent, indexer_weights, positions, rotary_emb, qr_scale)
        metadata = target.get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            layer = metadata[self.k_cache.prefix]
            if (layer.max_seq_len // self.compress_ratio <= self.topk_tokens
                    and not torch.cuda.is_current_stream_capturing()):
                if self.owns_k:
                    self._produce_k(latent, positions, rotary_emb)
                assert self.topk_indices_buffer is not None
                num_tokens = layer.num_decode_tokens + layer.num_prefill_tokens
                if num_tokens > 0:
                    target._fill_short_context_topk_indices[(num_tokens,)](
                        self.topk_indices_buffer, positions,
                        TOP_K=self.topk_tokens, COMPRESS_RATIO=self.compress_ratio,
                        PADDED_TOP_K=target.triton.next_power_of_2(self.topk_tokens),
                        num_warps=8,
                    )
                return None, None, None
        if self.owns_k:
            self._produce_k(latent, positions, rotary_emb)
        q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_head, self.head_dim)
        query, weights = target.fused_indexer_q_rope_quant(
            positions, q, rotary_emb.cos_sin_cache, indexer_weights,
            self.softmax_scale, self.n_head**-0.5,
            use_fp4=True, weights_out_dtype=self.indexer_weights_dtype,
        )
        if not isinstance(query, Fp4QTuple):
            raise RuntimeError("V4.1 FP4 forward lost the BF16/packed Q adapter")
        return query, None, weights

    forward._vllm_hcu_fp4_forward = True
    cls.forward = forward


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    ops_module = importlib.import_module("vllm.models.deepseek_v41.common.ops")
    store_module = importlib.import_module(
        "vllm.models.deepseek_v41.common.ops.indexer_k_store"
    )
    original = require_callable(
        target,
        "indexer_k_norm_rope_store",
        f"{TARGET_MODULE}.indexer_k_norm_rope_store",
    )
    original_fp4_selector = getattr(target, "dsa_indexer_uses_fp4", None)
    if getattr(target, _MARKER, False):
        return False
    if fp4_indexer_requested():
        _install_fp4_forward(target)

    @functools.wraps(original)
    def store(k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
              k_cache, kv_slot_mapping, compress_ratio, use_fp4_cache):
        if not use_normal_indexer_k_layout() and not (
            use_fp4_cache and fp4_indexer_requested()
        ):
            return original(
                k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
                k_cache, kv_slot_mapping, compress_ratio, use_fp4_cache,
            )
        if use_fp4_cache:
            if not fp4_indexer_requested():
                return original(
                    k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
                    k_cache, kv_slot_mapping, compress_ratio, use_fp4_cache,
                )
            # Write the allocator's actual page planes. The reader stages
            # non-64-row pages, rather than reinterpreting their scale offset.
            fp4_physical_page_size(k_cache)
            num_tokens = kv_slot_mapping.numel()
            assert k_pre.ndim == 2 and k_pre.shape[1] == 128
            assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
            assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
            assert compress_ratio in (1, 2)
            if not num_tokens:
                return
            kernel = store_module._indexer_k_norm_rope_quant_store_kernel
            kernel[(num_tokens,)](
                k_pre, k_pre.stride(0), positions, rms_norm_weight, rms_norm_eps,
                cos_sin_cache, cos_sin_cache.stride(0), k_cache, kv_slot_mapping,
                k_cache.shape[1], HEAD_SIZE=128, ROPE_HEAD_DIM=64,
                COMPRESS_RATIO=compress_ratio, TOKEN_STRIDE=64, SCALE_DIM=4,
                KV_BLOCK_STRIDE=k_cache.stride(0), FP8_MAX=448.0,
                USE_FP4=True, SHUFFLE=False, BLOCK_TILE_SIZE=1,
                HEAD_TILE_SIZE=1, num_warps=1,
            )
            return

        # Reuse the upstream fused norm/RoPE/FP8 store kernel. Only its existing
        # SHUFFLE constexpr changes; the page and scale addressing stay identical.
        num_tokens = kv_slot_mapping.numel()
        assert k_pre.ndim == 2 and k_pre.shape[1] == 128
        assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
        assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
        assert compress_ratio in (1, 2)
        if not num_tokens:
            return
        kernel = store_module._indexer_k_norm_rope_quant_store_kernel
        kernel[(num_tokens,)](
            k_pre, k_pre.stride(0), positions, rms_norm_weight, rms_norm_eps,
            cos_sin_cache, cos_sin_cache.stride(0), k_cache, kv_slot_mapping,
            k_cache.shape[1], HEAD_SIZE=128, ROPE_HEAD_DIM=64,
            COMPRESS_RATIO=compress_ratio, TOKEN_STRIDE=128, SCALE_DIM=4,
            KV_BLOCK_STRIDE=k_cache.stride(0), FP8_MAX=448.0,
            USE_FP4=False, SHUFFLE=False, BLOCK_TILE_SIZE=16,
            HEAD_TILE_SIZE=16, num_warps=1,
        )

    target.indexer_k_norm_rope_store = store
    ops_module.indexer_k_norm_rope_store = store
    store_module.indexer_k_norm_rope_store = store
    setattr(target, _MARKER, True)

    # Patch the constructor's exact FP4 selector. This is the capability gate
    # before DeepseekV4Indexer allocates its K cache.
    if callable(original_fp4_selector) and not getattr(target, "_vllm_hcu_fp4_selector", False):
        @functools.wraps(original_fp4_selector)
        def select_fp4(vllm_config):
            if not fp4_indexer_requested():
                return False
            blockers = _fp4_domain_blockers(target, vllm_config)
            if blockers:
                raise RuntimeError("HCU FP4 indexer requested outside supported domain: "
                                   + "; ".join(blockers))
            from .patch_fp4_packer_lowering import require_fp4_packer_lowering
            require_fp4_packer_lowering()
            return True

        target.dsa_indexer_uses_fp4 = select_fp4
        setattr(target, "_vllm_hcu_fp4_selector", True)

        # Keep metadata and allocation format selection consistent. Other
        # model families retain the original selector, including when OFF.
        backend = importlib.import_module("vllm.v1.attention.backends.mla.indexer")
        backend_original = backend.dsa_indexer_uses_fp4

        @functools.wraps(backend_original)
        def metadata_select(vllm_config):
            if fp4_indexer_requested() and _is_v41(vllm_config):
                return select_fp4(vllm_config)
            return backend_original(vllm_config)

        backend.dsa_indexer_uses_fp4 = metadata_select

        # The inherited ROCm backend advertises 128 *uncompressed* tokens.
        # With the requested manager block 64 that cannot be selected. The
        # consumer now handles both 32- and 64-record physical pages, so V4.1
        # can honestly advertise a 64-token kernel block when opted in.
        backend_cls = target.DeepseekV41IndexerBackend
        original_sizes = backend_cls.get_supported_kernel_block_sizes

        def supported_sizes():
            return [64] if fp4_indexer_requested() else original_sizes()

        backend_cls.get_supported_kernel_block_sizes = staticmethod(supported_sizes)

    # Patch the exact V4.1 common.ops Q alias. The tuple ABI remains unchanged;
    # HCU readers can additionally access the retained BF16 prefill Q.
    q_original = ops_module.fused_indexer_q_rope_quant
    if not getattr(ops_module, _Q_MARKER, False):
        @functools.wraps(q_original)
        def fused_q(*args, **kwargs):
            use_fp4 = bool(kwargs.get("use_fp4", False))
            if len(args) >= 7:
                use_fp4 = use_fp4 or bool(args[6])
            result = q_original(*args, **kwargs)
            if not (use_fp4 and fp4_indexer_requested()):
                return result
            packed, weights = result
            if not isinstance(packed, tuple) or len(packed) != 2:
                raise RuntimeError("V4.1 FP4 Q producer returned invalid packed Q")
            bf16 = rotate_indexer_query_bf16(
                args[1], args[0], args[2], rope_dim=64
            )
            return Fp4QTuple(packed[0], packed[1], bf16, weights), weights

        ops_module.fused_indexer_q_rope_quant = fused_q
        target.fused_indexer_q_rope_quant = fused_q
        setattr(ops_module, _Q_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module",
           "use_normal_indexer_k_layout"]
