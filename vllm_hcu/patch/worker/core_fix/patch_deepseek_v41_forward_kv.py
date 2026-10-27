# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add DeepseekV4Attention.forward_kv for decoder-side SWA bounded replay."""

from __future__ import annotations

from types import ModuleType
from typing import Any, cast

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
)

TARGET_MODULE = "vllm.models.deepseek_v41.attention"
PATCH_ID = "worker.core_fix.deepseek_v41.forward_kv"
TARGETS = (f"{TARGET_MODULE}.DeepseekV4Attention.forward_kv",)
_MARKER = "_vllm_hcu_dsv41_forward_kv_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        target, "DeepseekV4Attention", f"{TARGET_MODULE}.DeepseekV4Attention"
    )
    if getattr(cls, _MARKER, False):
        return False
    if hasattr(cls, "forward_kv") and getattr(
        vars(cls).get("forward_kv", None), _MARKER, False
    ):
        return False

    # Upstream forward_kv depends on these helpers existing on the class.
    for name in (
        "_run_parallel_input_projections",
        "_split_qkv_and_norm",
        "swa_cache_layer",
    ):
        # swa_cache_layer is an instance attr; method helpers must exist.
        if name == "swa_cache_layer":
            continue
        if not callable(getattr(cls, name, None)):
            raise PatchCompatibilityError(
                f"required HCU patch target {TARGET_MODULE}.DeepseekV4Attention."
                f"{name} is missing"
            )

    def forward_kv(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> None:
        """Write only the KV ``forward`` writes (SWA KV, compressed KV, index K)."""
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        qr_kv, kv_score, _ = self._run_parallel_input_projections(hidden_states)
        _, _, kv = self._split_qkv_and_norm(qr_kv)
        if isinstance(attn_metadata, dict):
            cache = self.swa_cache_layer.kv_cache
            swa_metadata = cast(Any, attn_metadata[self.swa_cache_layer.prefix])
            slot_mapping = swa_metadata.slot_mapping
            cache_kv = kv
            cache_positions = positions
            if getattr(swa_metadata, "pcp_has_global_prefill", False):
                from vllm_hcu.model_executor.layers.attention.pcp import (
                    globalize_pcp_slot_mapping,
                    pcp_global_positions,
                    restore_pcp_rows_to_global,
                )

                cache_kv = restore_pcp_rows_to_global(kv, swa_metadata)
                cache_positions = pcp_global_positions(swa_metadata)
                slot_mapping = globalize_pcp_slot_mapping(slot_mapping, swa_metadata)
            # PCP builders keep an expanded rank-major slot mapping for the
            # cache writer; this kernel pairs slots 1:1 with local positions.
            elif slot_mapping.numel() != positions.numel():
                from vllm_hcu.model_executor.layers.attention.pcp import (
                    pcp_local_slot_view,
                )

                try:
                    from vllm.distributed.parallel_state import get_pcp_group

                    pcp_size = int(get_pcp_group().world_size)
                except Exception:
                    pcp_size = 1
                if pcp_size > 1 and slot_mapping.numel() % pcp_size == 0:
                    slot_mapping = pcp_local_slot_view(slot_mapping, pcp_size)
                if slot_mapping.numel() != positions.numel():
                    # Replay pad / trim can leave a longer mapping; keep the
                    # leading rows that align with positions.
                    slot_mapping = slot_mapping.reshape(-1)[: positions.numel()]
            torch.ops._C.fused_deepseek_v4_kv_rope_insert(
                cache_kv,
                cache,
                slot_mapping,
                cache_positions,
                self.rotary_emb.cos_sin_cache,
                self.swa_cache_layer.block_size,
                getattr(self, "_flashinfer_fp8_kv_scale", None)
                if cache.dtype == torch.float8_e4m3fn
                else None,
                getattr(self, "kv_mxfp8", False),
            )
        if self.compressor is not None:
            latent = self.compressor(kv_score, positions)
            self.compressor.insert_cache(latent, positions, self.rotary_emb)
            if self.indexer is not None and self.indexer.owns_k:
                self.indexer._produce_k(latent, positions, self.indexer_rotary_emb)

    setattr(forward_kv, _MARKER, True)
    setattr(cls, "forward_kv", forward_kv)
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
