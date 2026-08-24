# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned MLA execution helpers."""

from __future__ import annotations

import torch

from vllm_hcu.patch.config import HcuFeatureConfig


def lightly_cp_mla_wrapper_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    feature_config: HcuFeatureConfig,
) -> torch.Tensor:
    """v0.25.1 target MLA wrapper forward plus the HCU Lightly-CP KV gather."""

    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.forward_context import get_forward_context

    q_c = None
    kv_lora = None
    if self.q_lora_rank is not None:
        if self.fused_qkv_a_proj is None or self.q_a_layernorm is None or self.q_b_proj is None:
            raise RuntimeError("Lightly-CP MLA requires fused q/kv projection modules")
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
    else:
        if self.kv_a_proj_with_mqa is None or self.q_proj is None:
            raise RuntimeError("Lightly-CP MLA requires q and kv projection modules")
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]

    kv_c, k_pe = kv_lora.split(
        [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    kv_c_normed = self.kv_a_layernorm(kv_c)
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    if self.rotary_emb is not None:
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )
    if self.indexer and self.is_sparse and not self.skip_topk:
        self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
    if llama_4_scaling is not None:
        q *= llama_4_scaling

    context = get_forward_context()
    if not bool(getattr(context, "enable_lightly_cp", False)):
        raise RuntimeError(
            "HcuFeatureConfig enables Lightly-CP but ForwardContext was not "
            "initialized with enable_lightly_cp"
        )
    kv_c_normed = tensor_model_parallel_all_gather(kv_c_normed.contiguous(), 0)
    k_pe = tensor_model_parallel_all_gather(k_pe.contiguous(), 0)
    if feature_config.enable_lightly_cplb:
        gather_indexes = getattr(context, "gather_indexes_tensor", None)
        if gather_indexes is None:
            raise RuntimeError(
                "HcuFeatureConfig enables Lightly-CPLB but gather_indexes_tensor is missing"
            )
        kv_c_normed = torch.index_select(kv_c_normed, 0, gather_indexes)
        k_pe = torch.index_select(k_pe, 0, gather_indexes)

    attn_out = self.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
    )
    return self.o_proj(attn_out)[0]


def mla_forward_impl(
    upstream,
    self,
    q,
    k_c_normed,
    k_pe,
    kv_cache,
    attn_metadata,
    output,
    output_scale=None,
    output_block_scale=None,
    quant_group_size=None,
    quant_scale_ue8m0=None,
    quant_col_major=None,
    quant_tma_aligned=None,
):
    """v0.25.1 target MLA forward plus HCU Lightly-CP KV length and CAT support."""

    quant_key = upstream._detect_output_quant_key(
        output, output_scale, output_block_scale, self.num_heads * self.v_head_dim
    )
    if quant_key is not None:
        quant_output = output
        output = torch.empty(
            output.shape[0], self.num_heads * self.v_head_dim,
            dtype=q.dtype, device=output.device,
        )
    if attn_metadata is None:
        _ = torch.empty(
            (self.chunked_prefill_workspace_size, self.num_heads,
             self.qk_nope_head_dim + self.v_head_dim),
            device=k_c_normed.device, dtype=k_c_normed.dtype,
        )
        return quant_output.fill_(0) if quant_key is not None else output.fill_(0)
    if self.impl.dcp_world_size == -1:
        self.impl.dcp_world_size = upstream.get_dcp_group().world_size
    fp8_attention = upstream.is_quantized_kv_cache(self.kv_cache_dtype)
    num_actual_toks = attn_metadata.num_actual_tokens
    num_kv_actual_toks = getattr(
        attn_metadata, "num_kv_actual_tokens", num_actual_toks
    )
    if num_kv_actual_toks is None:
        num_kv_actual_toks = num_actual_toks
    output_padded = output
    output = output[:num_actual_toks, ...]
    q = q[:num_actual_toks, ...]
    k_c_normed = k_c_normed[:num_kv_actual_toks, ...]
    k_pe = k_pe[:num_kv_actual_toks, ...]
    if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(upstream.current_platform.fp8_dtype())
    is_sparse_impl = isinstance(self.impl, upstream.SparseMLAAttentionImpl)
    if is_sparse_impl:
        num_mqa_tokens, num_mha_tokens = q.size(0), 0
    else:
        if any(
            getattr(attn_metadata, name, None) is None
            for name in ("num_decodes", "num_prefills", "num_decode_tokens")
        ):
            raise RuntimeError("HCU MLA metadata is missing decode/prefill counts")
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens
    if num_mha_tokens > 0:
        self.impl.forward_mha(
            q[num_mqa_tokens:], k_c_normed[num_mqa_tokens:],
            k_pe[num_mqa_tokens:], kv_cache, attn_metadata, self._k_scale,
            output=output[num_mqa_tokens:],
        )
    if num_mqa_tokens > 0:
        mqa_q = q[:num_mqa_tokens]
        mqa_output_slice = output[:num_mqa_tokens]
        mqa_q_nope, mqa_q_pe = mqa_q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        mqa_q_nope = mqa_q_nope.transpose(0, 1)
        if self.q_pad_num_heads is not None:
            B, N, L = mqa_q_pe.shape
            padded = mqa_q_pe.new_empty((B, self.q_pad_num_heads, L))
            padded.resize_((B, N, L))
            padded.copy_(mqa_q_pe)
            mqa_q_pe = padded
        if self.is_aiter_triton_fp4_bmm_enabled:
            from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

            mqa_ql_nope = batched_gemm_a16wfp4(
                mqa_q_nope, self.W_K, self.W_K_scale, transpose_bm=True,
                prequant=True, y_scale=self._q_scale if fp8_attention else None,
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
            mqa_ql_nope = upstream.rocm_aiter_ops.triton_fp8_bmm(
                mqa_q_nope, self.W_K, self.W_K_scale,
                group_size=128, transpose_bm=True,
            )
        else:
            N, B, _ = mqa_q_nope.shape
            L = self.W_UK_T.shape[-1]
            if self.q_pad_num_heads is not None:
                mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))
                mqa_ql_nope.resize_((N, B, L))
            else:
                mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))
            torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
            mqa_ql_nope = mqa_ql_nope.transpose(0, 1)
        from vllm_hcu.platforms.hcu import on_gfx938

        if (
            fp8_attention
            and self.impl.supports_quant_query_input
            and on_gfx938()
        ):
            if mqa_ql_nope.shape[:2] != mqa_q_pe.shape[:2]:
                raise RuntimeError("HCU MLA query components have incompatible shapes")
            from vllm_hcu.platforms import envs as henvs

            if henvs.VLLM_HCU_USE_CAT_MLA:
                mqa_q = (mqa_ql_nope, mqa_q_pe)
            else:
                mqa_q = self._decode_concat_quant_fp8_op(
                    mqa_ql_nope, mqa_q_pe, self._q_scale
                )
        else:
            mqa_q = (mqa_ql_nope, mqa_q_pe)
        if self.impl.dcp_world_size > 1:
            if fp8_attention:
                raise RuntimeError("HCU MLA DCP does not support FP8 KV cache")
            mqa_q = torch.cat(mqa_q, dim=-1)
            mqa_q = upstream.get_dcp_group().all_gather(mqa_q, dim=1)
        if not is_sparse_impl and attn_metadata.decode is None:
            raise RuntimeError("HCU MLA decode metadata is missing")
        attn_out, lse = self.impl.forward_mqa(
            mqa_q, kv_cache, attn_metadata, self
        )
        if self.impl.dcp_world_size > 1:
            if self.dcp_a2a:
                attn_out = upstream.dcp_a2a_lse_reduce(
                    attn_out, lse, upstream.get_dcp_group(),
                    is_lse_base_on_e=not getattr(self, "_use_fi_prefill", False),
                )
            else:
                attn_out = upstream.cp_lse_ag_out_rs(
                    attn_out, lse, upstream.get_dcp_group(),
                    is_lse_base_on_e=not getattr(self, "_use_fi_prefill", False),
                )
        self._v_up_proj(attn_out, out=mqa_output_slice)
    if quant_key is not None:
        actual = output[:num_actual_toks]
        if quant_key == upstream.kNvfp4Dynamic:
            if output_block_scale is None:
                raise RuntimeError("NVFP4 output requires output_block_scale")
            fp4_data, fp4_scales = upstream.ops.scaled_fp4_quant(actual, output_scale)
            quant_output[:num_actual_toks].copy_(fp4_data)
            output_block_scale[:fp4_scales.shape[0]].copy_(fp4_scales)
        elif quant_key in (upstream.kFp8Dynamic128Sym, upstream.kFp8Dynamic64Sym):
            if output_block_scale is None or quant_group_size is None:
                raise RuntimeError("group FP8 output requires block scale and group size")
            finfo = torch.finfo(upstream._FP8_DTYPE)
            torch.ops._C.per_token_group_fp8_quant(
                actual, quant_output[:num_actual_toks],
                output_block_scale[:num_actual_toks], quant_group_size, 1e-10,
                finfo.min, finfo.max, quant_scale_ue8m0, quant_col_major,
                quant_tma_aligned,
            )
        elif quant_key == upstream.kFp8StaticTensorSym:
            fp8_data, _ = self._quant_fp8_op(actual, output_scale)
            quant_output[:num_actual_toks].copy_(fp8_data)
        else:
            raise ValueError(f"Unsupported quant_key: {quant_key}")
        return quant_output
    return output_padded


def _get_mla_kv_b_proj_weight(upstream, layer, out_dtype):
    """Return logical ``[K, N]`` weight without executing an FP8 linear.

    Channel-wise CompressedTensors weights have already been normalized to the
    HCU column-major ``[K, N]`` layout before MLA post-processing.  Calling the
    upstream generic fallback would build an identity matrix and dynamically
    quantize it through NVIDIA-only operators.  The stored per-output scale is
    sufficient to dequantize the weight directly and exactly.
    """
    weight = getattr(layer, "weight", None)
    fp8_dtypes = {
        torch.float8_e4m3fn,
        getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
    }
    is_hcu_channel_layout = (
        isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and weight.dtype in fp8_dtypes
        and getattr(weight, "input_dim", None) == 0
        and getattr(weight, "output_dim", None) == 1
    )
    if not is_hcu_channel_layout:
        return upstream.get_and_maybe_dequant_weights(
            layer,
            out_dtype=out_dtype,
        )

    scale = getattr(layer, "weight_scale", None)
    if not isinstance(scale, torch.Tensor) or scale.numel() != weight.shape[1]:
        scale_shape = None if scale is None else tuple(scale.shape)
        raise ValueError(
            "HCU MLA Channel-FP8 scale must contain one value per output "
            f"channel: weight={tuple(weight.shape)}, scale={scale_shape}."
        )
    scale = scale.reshape(1, weight.shape[1]).float()
    return (weight.float() * scale).to(out_dtype)


def mla_process_weights_nn(upstream, self, act_dtype):
    """Normalize either HCU NN or upstream weight layout before MLA BMM setup."""

    kv_b_proj_weight = _get_mla_kv_b_proj_weight(
        upstream,
        self.kv_b_proj,
        act_dtype,
    )
    expected = (
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
    )
    if tuple(kv_b_proj_weight.shape) != expected:
        if tuple(kv_b_proj_weight.T.shape) == expected:
            kv_b_proj_weight = kv_b_proj_weight.T.contiguous()
        else:
            raise ValueError(
                f"kv_b_proj_weight.shape={tuple(kv_b_proj_weight.shape)}, "
                f"expected={expected}"
            )
    kv_b_proj_weight = kv_b_proj_weight.view(
        self.kv_lora_rank, self.num_heads,
        self.qk_nope_head_dim + self.v_head_dim,
    )
    W_UK, W_UV = kv_b_proj_weight.split(
        [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    if self.is_aiter_triton_fp4_bmm_enabled:
        from vllm.model_executor.layers.quantization.quark.utils import (
            quark_quantize_weight_to_mxfp4,
        )
        self.W_K, self.W_K_scale = quark_quantize_weight_to_mxfp4(W_UK)
        self.W_K = self.W_K.transpose(0, 1)
        self.W_K_scale = self.W_K_scale.transpose(0, 1)
        self.W_V, self.W_V_scale = quark_quantize_weight_to_mxfp4(
            W_UV.permute(1, 2, 0)
        )
    elif self.is_aiter_triton_fp8_bmm_enabled:
        W_K, W_V = W_UK.transpose(0, 1), W_UV.permute(1, 2, 0)
        self.W_K, self.W_K_scale = upstream.dynamic_per_batched_tensor_quant(
            W_K, dtype=upstream.current_platform.fp8_dtype()
        )
        self.W_V, self.W_V_scale = upstream.dynamic_per_batched_tensor_quant(
            W_V, dtype=upstream.current_platform.fp8_dtype()
        )
        precompile = list(range(1, 1025))
        if upstream.is_global_first_rank():
            precompile = upstream.tqdm(
                precompile,
                desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                total=1024,
            )
        for m in precompile:
            x = torch.empty(
                (self.W_K.shape[0], m, self.W_K.shape[2]),
                dtype=torch.bfloat16, device=self.W_K.device,
            )
            upstream.rocm_aiter_ops.triton_fp8_bmm(
                x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
            )
            x = torch.empty(
                (self.W_V.shape[0], m, self.W_V.shape[2]),
                dtype=torch.bfloat16, device=self.W_V.device,
            )
            upstream.rocm_aiter_ops.triton_fp8_bmm(
                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
            )
    else:
        self.W_UV = W_UV.transpose(0, 1)
        self.W_UK_T = W_UK.permute(1, 2, 0)
    quant_method = (
        self.quant_config.get_quant_method(self, prefix=self.layer_name)
        if self.quant_config else None
    )
    if not upstream.should_load_quant_weights(quant_method):
        upstream.set_default_quant_scales(self, register_buffer=False)


__all__ = [
    "lightly_cp_mla_wrapper_forward",
    "mla_forward_impl",
    "mla_process_weights_nn",
]
