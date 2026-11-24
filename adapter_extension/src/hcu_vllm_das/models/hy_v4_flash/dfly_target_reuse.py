# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFly target reuse adapted from vLLM PR #68296 for Core v0.33.

This component does not register a speculator or lift the runtime DFly guard.
The input/index kernel bodies are retained in ``dfly_inputs``; owned wrappers
add shape/device/capacity validation.
"""

from dataclasses import replace

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, MLAAttentionSpec
from vllm.v1.worker.utils import customize_attention_spec

from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
from .dfly_inputs import build_dfly_reuse_indices
from .dfly_rotary import HYV4DFlyRotaryEmbedding


class HYV4DFlyTargetReuse:
    def __init__(
        self, target: nn.Module, draft: nn.Module, vllm_config: VllmConfig
    ):
        config = draft.config
        # The public Core normally validates this while deriving the draft.
        # Keep the old-Core plugin component safe before that wiring is active.
        count = len(draft.model.layers)
        language_model = (
            target.get_language_model()
            if hasattr(target, "get_language_model")
            else target
        )
        target_count = len(language_model.model.layers)
        for name in ("target_index_layer", "target_kv_layers"):
            selected = getattr(config, name)
            if (not isinstance(selected, list) or count == 0
                    or len(selected) != count
                    or any(type(i) is not int or not 1 <= i < target_count
                           for i in selected)):
                raise ValueError(f"DFly {name} needs one valid layer per draft layer")
        if len(set(config.target_kv_layers)) != count:
            raise ValueError("DFly needs a distinct target KV layer per draft layer")
        if vllm_config.cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8_ds_mla",
        ):
            raise ValueError("DFly requires BF16 or fp8_ds_mla MLA KV cache")
        language_model = (
            target.get_language_model()
            if hasattr(target, "get_language_model")
            else target
        )
        target_layers = language_model.model.layers
        self.index_layers: dict[int, HYV4MLAAttention] = {}
        self.layer_pairs: list[tuple[HYV4MLAAttention, HYV4MLAAttention]] = []
        self.aliases: dict[str, str] = {}
        self.draft = draft
        self.config = config

        def target_attention(index: int) -> HYV4MLAAttention:
            attn = target_layers[index].self_attn
            if not isinstance(attn, HYV4MLAAttention):
                # Flash derives from the installed Core MLA class, whereas
                # this backported draft uses the plugin-owned HYV4 base.
                from .attention import HYV4FlashAttention
                if not isinstance(attn, HYV4FlashAttention):
                    raise ValueError(f"DFly target layer {index} is not HYV4 MLA")
            return attn

        for index in config.target_index_layer:
            attn = target_attention(index)
            if (
                attn.indexer is None
                or attn.skip_topk
                or attn.topk_indices_buffer is None
            ):
                raise ValueError(
                    f"DFly target index layer {index} must run its own indexer"
                )
            if attn.topk_indices_buffer.shape[-1] != config.target_index_topk:
                raise ValueError(
                    f"DFly target index layer {index} has incompatible top-k width"
                )
            self.index_layers[index] = attn
        for layer, index in zip(draft.model.layers, config.target_kv_layers, strict=True):
            source, dest = target_attention(index), layer.self_attn
            source_rope, dest_rope = source.rotary_emb, dest.rotary_emb
            if isinstance(dest_rope, HYV4DFlyRotaryEmbedding):
                dest_rope = dest_rope.rotary
            if type(source_rope) is not type(dest_rope):
                raise ValueError(f"DFly target KV layer {index} uses incompatible RoPE")
            if source.num_local_heads != dest.num_local_heads:
                raise ValueError("DFly target and draft must use matching attention TP")
            storage_specs: list[MLAAttentionSpec] = []
            for attn in (source, dest):
                # v0.33 returns the storage spec directly, before the newer
                # LayerKVCacheConfig API used by PR #68296.
                spec = attn.mla_attn.get_kv_cache_spec(vllm_config)
                if not isinstance(spec, MLAAttentionSpec):
                    raise ValueError("DFly requires MLA KV cache specs")
                customized_spec = customize_attention_spec(
                    attn.mla_attn.get_attn_backend(), spec
                )
                if not isinstance(customized_spec, MLAAttentionSpec):
                    raise ValueError("DFly backend must preserve the MLA KV cache spec")
                storage_specs.append(
                    replace(
                        customized_spec,
                        topk_buffer_width=None,
                        is_index_group_leader=False,
                    )
                )
            if storage_specs[0] != storage_specs[1]:
                raise ValueError(
                    f"DFly target KV layer {index} has incompatible cache spec"
                )
            if (
                source.learnable_sink
                and not source.mla_attn.get_attn_backend().supports_sink()
            ):
                raise ValueError("DFly target backend must apply its learnable sink")
            self.layer_pairs.append((source, dest))
            self.aliases[dest.mla_attn.layer_name] = source.mla_attn.layer_name
        if len(set(self.aliases.values())) != len(self.aliases):
            raise ValueError("DFly needs a distinct target KV layer per draft layer")
        # One stacked snapshot buffer, so one launch builds every draft layer's
        # indices; each draft layer reads the slot of its target index layer.
        index_layer_ids = list(self.index_layers)
        first = next(iter(self.index_layers.values())).topk_indices_buffer
        assert first is not None
        self.snapshots = torch.full(
            (len(index_layer_ids), *first.shape),
            -1,
            dtype=first.dtype,
            device=first.device,
        )
        for slot, index in enumerate(index_layer_ids):
            self.index_layers[index].topk_snapshot = self.snapshots[slot]
        self.layer_snapshot_slots = torch.tensor(
            [index_layer_ids.index(i) for i in config.target_index_layer],
            dtype=torch.int32,
            device=first.device,
        )
        self._set_storage_aliases(True)

    def _set_storage_aliases(self, enabled: bool) -> None:
        """Toggle the KV-sharing alias on the draft layers.

        The allocation marker makes Core attn_utils omit separate draft storage
        and bind each draft layer to its target storage. It must remain enabled
        across profiling and final KV-cache initialization, which rediscover
        aliases independently. Core033 MLA writes through the forward-context
        slot mapping regardless of this marker; query metadata/slots stay owned
        by the draft.
        """
        for source, dest in self.layer_pairs:
            dest.mla_attn.kv_sharing_target_layer_name = (
                source.mla_attn.layer_name if enabled else None
            )

    def validate_groups(self, config: KVCacheConfig) -> None:
        self._set_storage_aliases(True)
        groups = {
            name: i
            for i, group in enumerate(config.kv_cache_groups)
            for name in group.layer_names
        }
        for draft, target in self.aliases.items():
            if draft not in groups or groups[draft] != groups.get(target):
                raise ValueError(
                    f"DFly alias {draft} and {target} must share a KV group"
                )

    def validate_storage(self) -> None:
        for source, dest in self.layer_pairs:
            a, b = source.mla_attn.kv_cache, dest.mla_attn.kv_cache
            if (
                a.numel() == 0
                or a.data_ptr() != b.data_ptr()
                or a.shape != b.shape
                or a.stride() != b.stride()
                or a.dtype != b.dtype
                or a.device != b.device
                or a.storage_offset() != b.storage_offset()
            ):
                raise ValueError(
                    "DFly query and target KV must alias the same local layout"
                )

    def update_indices(
        self,
        active: torch.Tensor,
        source_rows: torch.Tensor,
        anchors: torch.Tensor,
        padded_batch: int,
    ) -> None:
        build_dfly_reuse_indices(
            self.draft.model.topk_indices,
            self.snapshots,
            self.layer_snapshot_slots,
            active,
            source_rows,
            anchors,
            self.config.target_index_topk,
            self.config.context_local_window or 0,
            self.draft.model.query_width,
            padded_batch,
        )
