# SPDX-License-Identifier: Apache-2.0
# vllm-hcu: fused RMSNorm + MRoPE for Qwen3.5 full-attention layers, ported
# from vllm-hcu main 4e01bc8 ("fix(qwen3_next): fused RMSNorm-MRoPE reads
# positions by stride"). Upstream 0.28's native fused_qk_rmsnorm_rope_gate
# path requires current_platform.is_cuda() and never engages on HCU, so the
# unfused q_norm + k_norm + rotary_emb sequence runs instead.

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_next"
PATCH_ID = "worker.op_opt.qwen3_next.fused_rmsnorm_mrope"
TARGETS = (f"{TARGET_MODULE}.Qwen3NextAttention._project_qkv_gate",)
_MARKER = "_vllm_hcu_fused_rmsnorm_mrope_applied"
_WRAPPER = "_vllm_hcu_fused_rmsnorm_mrope_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    qn = load_exact_module(TARGET_MODULE, module)
    if getattr(qn, _MARKER, False):
        return False
    attn_cls = require_class(
        qn, "Qwen3NextAttention", f"{TARGET_MODULE}.Qwen3NextAttention")
    original = require_callable(
        attn_cls, "_project_qkv_gate", TARGETS[0])

    @functools.wraps(original)
    def hcu_project_qkv_gate(self, qkv, positions):
        import vllm_hcu.platforms.envs as henvs

        rot_emb = self.rotary_emb
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_FUSED_RMSNORM_MROPE
            and positions.ndim == 2
            and self.head_dim == 256
            and type(rot_emb).__name__ == "MRotaryEmbedding"
            and getattr(rot_emb, "is_neox_style", False)
            and rot_emb.rotary_dim == 64
            and getattr(rot_emb, "mrope_interleaved", False)
            and tuple(getattr(rot_emb, "mrope_section", ())) == (11, 11, 10)
        ):
            from vllm_hcu.model_executor.layers.rotary_embedding import (
                fused_rmsnorm_mrope as _fused,
            )
            if self.attn_output_gate:
                q_gate, k, v = qkv.split(
                    [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
                )
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*orig_shape, -1)
                gate = gate.reshape(*orig_shape, -1)
            else:
                q, k, v = qkv.split(
                    [self.q_size, self.kv_size, self.kv_size], dim=-1)
                gate = None
            cache = (
                rot_emb._match_cos_sin_cache_dtype(q)
                if hasattr(rot_emb, "_match_cos_sin_cache_dtype")
                else rot_emb.cos_sin_cache
            )
            q, k = _fused.fused_rmsnorm_mrope(
                q, k,
                self.q_norm.weight, self.k_norm.weight,
                cache,
                positions,
                head_dim=self.head_dim,
                rotary_dim=rot_emb.rotary_dim,
                eps=self.q_norm.variance_epsilon,
            )
            return q, k, v, gate
        return original(self, qkv, positions)

    setattr(hcu_project_qkv_gate, _WRAPPER, True)
    attn_cls._vllm_hcu_original_project_qkv_gate = original
    attn_cls._project_qkv_gate = hcu_project_qkv_gate
    setattr(qn, _MARKER, True)
    return True
