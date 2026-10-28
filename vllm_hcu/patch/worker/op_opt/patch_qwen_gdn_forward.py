# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Skip the per-layer zero fill of the Qwen GDN core-attention buffer.

Upstream allocates ``core_attn_out`` with ``torch.zeros`` (vLLM PR 28182)
because rows past ``num_actual_tokens`` may never be written.  Every row below
``num_actual_tokens`` is always written by the core op (the HCU packed decode
kernel writes zeros for null-slot padding rows, and FULL-graph decode
metadata is token-padded), so only the tail beyond ``num_actual_tokens`` needs
clearing.  On decode steps that tail is empty, which removes one fill kernel
per GDN layer (30 per step for Qwen3.5-35B).
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.qwen_forward_no_zero_fill"
TARGETS = (
    f"{TARGET_MODULE}.QwenGatedDeltaNetAttention.forward_cuda",
    f"{TARGET_MODULE}.QwenGatedDeltaNetAttention._warmup_prefill_kernels",
    f"{TARGET_MODULE}.QwenGatedDeltaNetAttention._forward_core_decode_non_spec",
)
_MARKER = "_vllm_hcu_qwen_gdn_no_zero_fill_applied"
_WRAPPER = "_vllm_hcu_qwen_gdn_no_zero_fill_wrapper"

# Triton specializes integer arguments on ==1 / %16, so the stock warmup
# (one 64-token sequence) leaves the multi-sequence and unaligned-length
# variants of the chunked prefill kernels to be compiled - and autotuned -
# by the first real batch, a multi-second stall.  Cover them up front.
_EXTRA_WARMUP_SEQLENS = ((37,), (32, 32), (37, 27), (20, 21, 23))


def _extra_prefill_warmup(layer, qkv_or_qkvz: torch.Tensor, v_dim: int, qwen) -> None:
    device, dtype = qkv_or_qkvz.device, qkv_or_qkvz.dtype
    num_k_heads = layer.num_k_heads // layer.tp_size
    num_v_heads = layer.num_v_heads // layer.tp_size
    _, state_dtype = layer.get_state_dtype()
    qkv_dim = qkv_or_qkvz.shape[-1] - v_dim
    for seqlens in _EXTRA_WARMUP_SEQLENS:
        total = sum(seqlens)
        try:
            mixed = torch.randn(total, qkv_dim, device=device, dtype=dtype)
            a = torch.randn(total, num_v_heads, device=device, dtype=dtype)
            b = torch.randn(total, num_v_heads, device=device, dtype=dtype)
            layer.rearrange_mixed_qkv(mixed)
            # Real prefill feeds the transposed causal-conv output (row
            # stride 1), which Triton specializes separately.
            layer.rearrange_mixed_qkv(
                torch.randn(qkv_dim, total, device=device, dtype=dtype).t())
            q, k, v, g, beta = qwen.fused_post_conv_prep(
                conv_output=mixed, a=a, b=b, A_log=layer.A_log,
                dt_bias=layer.dt_bias, num_k_heads=num_k_heads,
                head_k_dim=layer.head_k_dim, head_v_dim=layer.head_v_dim,
                apply_l2norm=True, output_g_exp=False,
            )
            state = torch.zeros(len(seqlens), num_v_heads, layer.head_v_dim,
                                layer.head_k_dim, device=device,
                                dtype=state_dtype)
            cu = [0]
            for n in seqlens:
                cu.append(cu[-1] + n)
            cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
            if layer.gdn_prefill_backend == "cutedsl":
                return
            # The metadata builder always supplies chunk indices/offsets for
            # prefill; passing tensors (not None) matches its specialization.
            from vllm.third_party.flash_linear_attention.ops.index import (
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )

            cu_cpu = torch.tensor(cu, dtype=torch.int32)
            chunk_indices = prepare_chunk_indices(
                cu_cpu, qwen.FLA_CHUNK_SIZE).to(device)
            chunk_offsets = prepare_chunk_offsets(
                cu_cpu, qwen.FLA_CHUNK_SIZE).to(device)
            layer.chunk_gated_delta_rule(
                q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
                g=g.unsqueeze(0), beta=beta.unsqueeze(0),
                initial_state=state, output_final_state=True,
                cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets, use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            qwen.logger.warning(
                "HCU extra GDN prefill warmup %s failed for %s",
                seqlens, layer.prefix, exc_info=True)
            return


def _layer_metadata(prefix: str):
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    return metadata.get(prefix)


def _is_packed_decode(layer, metadata) -> bool:
    """True when ``_forward_core`` takes the packed non-spec decode path,
    whose kernels accept row-strided ``a``/``b``."""
    return (
        metadata is not None
        and getattr(layer, "enable_packed_recurrent_decode", False)
        and getattr(metadata, "spec_sequence_masks", 0) is None
        and getattr(metadata, "num_prefills", 1) == 0
        and getattr(metadata, "num_decodes", 0) > 0
    )


def _num_actual_tokens(prefix: str) -> int | None:
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    layer_metadata = metadata.get(prefix)
    return getattr(layer_metadata, "num_actual_tokens", None)


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    gdn_cls = require_class(qwen, "QwenGatedDeltaNetAttention", TARGETS[0])
    if getattr(qwen, _MARKER, False):
        if not getattr(gdn_cls.forward_cuda, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_unpatched(gdn_cls, "forward_cuda", TARGETS[0], _WRAPPER)
    encode_layer_name = getattr(qwen, "_encode_layer_name", None)
    if encode_layer_name is None:
        raise PatchCompatibilityError(
            f"{TARGET_MODULE}._encode_layer_name is required by {TARGETS[0]}"
        )

    @functools.wraps(original)
    def hcu_forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import vllm_hcu.platforms.envs as henvs

        use_fused_gdn_decode = (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )
        num_tokens = hidden_states.size(0)
        metadata = _layer_metadata(self.prefix)
        num_actual = getattr(metadata, "num_actual_tokens", None)
        if (
            not henvs.VLLM_HCU_USE_CUSTOM_OPS
            or use_fused_gdn_decode
            or self.gqa_interleaved_layout
            or num_actual is None
            or num_actual > num_tokens
        ):
            return original(self, hidden_states)

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, a = self.split_ba(ba)

        core_attn_out = torch.empty(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if num_actual < num_tokens:
            core_attn_out[num_actual:].zero_()
        # The packed decode kernel reads a/b through their row stride, so
        # the two per-layer gather copies are only needed by prefill paths.
        if not _is_packed_decode(self, metadata):
            b, a = b.contiguous(), a.contiguous()
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            layer_name=encode_layer_name(self.prefix),
        )
        return self._output_projection(core_attn_out, z)

    setattr(hcu_forward_cuda, _WRAPPER, True)
    gdn_cls._vllm_hcu_original_forward_cuda = original
    gdn_cls.forward_cuda = hcu_forward_cuda

    original_warmup = require_unpatched(
        gdn_cls, "_warmup_prefill_kernels", TARGETS[1], _WRAPPER)

    @functools.wraps(original_warmup)
    def hcu_warmup_prefill_kernels(self, qkv_or_qkvz, v_dim):
        already = getattr(self, "_prefill_kernels_warmed_up", False)
        original_warmup(self, qkv_or_qkvz, v_dim)
        if not already:
            _extra_prefill_warmup(self, qkv_or_qkvz, v_dim, qwen)
            torch.accelerator.empty_cache()

    setattr(hcu_warmup_prefill_kernels, _WRAPPER, True)
    gdn_cls._vllm_hcu_original_warmup_prefill_kernels = original_warmup
    gdn_cls._warmup_prefill_kernels = hcu_warmup_prefill_kernels
    original_decode = require_unpatched(
        gdn_cls, "_forward_core_decode_non_spec", TARGETS[2], _WRAPPER)

    @functools.wraps(original_decode)
    def hcu_forward_core_decode_non_spec(
        self, mixed_qkv, b, a, core_attn_out, attn_metadata
    ):
        """Upstream packed decode, but the causal-conv update writes into a
        contiguous buffer.  In-place, its output is a row-strided slice of
        ``in_proj_qkvz`` that the AITER packed decode host code re-copies
        with ``.contiguous()`` - one extra kernel per GDN layer."""
        import vllm_hcu.platforms.envs as henvs

        if not henvs.VLLM_HCU_USE_CUSTOM_OPS or mixed_qkv.is_contiguous():
            return original_decode(
                self, mixed_qkv, b, a, core_attn_out, attn_metadata)
        indices = attn_metadata.non_spec_state_indices_tensor
        conv_state = (
            self.kv_cache[0]
            if qwen.is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        n = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:n]
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        conv_out = qwen.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=indices[:n],
            validate_data=False,
            out=torch.empty_like(mixed_qkv, memory_format=torch.contiguous_format),
        )
        qwen.fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=conv_out,
            a=a[:n],
            b=b[:n],
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=self.kv_cache[1],
            out=core_attn_out[:n].unsqueeze(1),
            ssm_state_indices=indices[:n],
            use_qk_l2norm_in_kernel=True,
        )

    setattr(hcu_forward_core_decode_non_spec, _WRAPPER, True)
    gdn_cls._vllm_hcu_original_forward_core_decode_non_spec = original_decode
    gdn_cls._forward_core_decode_non_spec = hcu_forward_core_decode_non_spec
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
