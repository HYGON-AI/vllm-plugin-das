from __future__ import annotations

import torch


def patch_hcu_lora_column_parallel_linear() -> None:
    # Patch vllm/lora/layers/column_parallel_linear.py:
    # - ColumnParallelLinearWithLoRA.__init__
    # - MergedColumnParallelLinearWithLoRA.__init__
    # - MergedColumnParallelLinearWithLoRA.slice_lora_b
    # - can_replace_layer() on the LoRA wrapper classes in this module
    from vllm.lora.layers import column_parallel_linear as lora_linear

    if getattr(lora_linear, "_hcu_lora_column_parallel_linear_patch_applied", False):
        return

    def _is_hcu_linear_type(layer_or_type, expected_name: str) -> bool:
        layer_type = (
            layer_or_type if isinstance(layer_or_type, type) else type(layer_or_type)
        )
        return (
            layer_type.__module__ == "vllm_hcu.model_executor.layers.linear"
            and layer_type.__name__ == expected_name
        )

    def _is_column_linear(layer_or_type) -> bool:
        return (
            type(layer_or_type) is type
            and (
                layer_or_type is lora_linear.ColumnParallelLinear
                or _is_hcu_linear_type(layer_or_type, "ColumnParallelLinear")
            )
        ) or (
            type(layer_or_type) is not type
            and (
                type(layer_or_type) is lora_linear.ColumnParallelLinear
                or _is_hcu_linear_type(layer_or_type, "ColumnParallelLinear")
            )
        )

    def _is_merged_column_linear(layer_or_type) -> bool:
        return (
            type(layer_or_type) is type
            and (
                layer_or_type is lora_linear.MergedColumnParallelLinear
                or _is_hcu_linear_type(layer_or_type, "MergedColumnParallelLinear")
            )
        ) or (
            type(layer_or_type) is not type
            and (
                type(layer_or_type) is lora_linear.MergedColumnParallelLinear
                or _is_hcu_linear_type(layer_or_type, "MergedColumnParallelLinear")
            )
        )

    def _is_qkv_linear(layer_or_type) -> bool:
        return (
            type(layer_or_type) is type
            and (
                layer_or_type is lora_linear.QKVParallelLinear
                or _is_hcu_linear_type(layer_or_type, "QKVParallelLinear")
            )
        ) or (
            type(layer_or_type) is not type
            and (
                type(layer_or_type) is lora_linear.QKVParallelLinear
                or _is_hcu_linear_type(layer_or_type, "QKVParallelLinear")
            )
        )

    original_init = lora_linear.ColumnParallelLinearWithLoRA.__init__
    original_merged_init = lora_linear.MergedColumnParallelLinearWithLoRA.__init__
    original_slice_lora_b = lora_linear.MergedColumnParallelLinearWithLoRA.slice_lora_b

    def patched_init(self, base_layer):
        original_init(self, base_layer)
        self.is_merged_col_linear = _is_merged_column_linear(base_layer)

    def patched_merged_init(self, base_layer):
        original_merged_init(self, base_layer)
        prefix = getattr(base_layer, "prefix", "")
        output_sizes = getattr(base_layer, "output_sizes", ())
        if not (
            isinstance(prefix, str)
            and prefix.endswith("in_proj_qkvz")
            and len(output_sizes) == 4
        ):
            self._hcu_grouped_qkvz_lora = False
            return

        self._hcu_grouped_qkvz_lora = True
        self._hcu_grouped_qkvz_output_sizes = tuple(output_sizes)
        qkv_group_size = sum(output_sizes[:3]) // self.tp_size
        z_group_size = output_sizes[3] // self.tp_size
        self.output_slices = (qkv_group_size, z_group_size)
        self.n_slices = 2
        self.output_ids = (self.tp_rank, self.tp_rank)

    def patched_slice_lora_b(self, lora_b):
        if not getattr(self, "_hcu_grouped_qkvz_lora", False):
            return original_slice_lora_b(self, lora_b)

        qkv_lora_b, z_lora_b = lora_b
        output_sizes = self._hcu_grouped_qkvz_output_sizes
        qkv_shards = []
        offset = 0
        for output_size in output_sizes[:3]:
            shard_size = output_size // self.tp_size
            qkv_shards.append(
                qkv_lora_b[
                    offset + shard_size * self.tp_rank : offset
                    + shard_size * (self.tp_rank + 1),
                    :,
                ]
            )
            offset += output_size
        qkv_lora_b = (
            qkv_shards[0].new_empty((0, qkv_lora_b.shape[1]))
            if not qkv_shards
            else torch.cat(qkv_shards, dim=0)
        )

        z_shard_size = output_sizes[3] // self.tp_size
        z_lora_b = z_lora_b[
            z_shard_size * self.tp_rank : z_shard_size * (self.tp_rank + 1),
            :,
        ]
        return [qkv_lora_b, z_lora_b]

    def _not_fully_sharded(lora_config, decorate: bool) -> bool:
        return True if not decorate else not lora_config.fully_sharded_loras

    def column_can_replace_layer(
        cls,
        source_layer,
        lora_config,
        packed_modules_list,
        model_config=None,
        *,
        decorate: bool = True,
    ) -> bool:
        if not _not_fully_sharded(lora_config, decorate):
            return False
        if _is_column_linear(source_layer):
            return True
        if _is_merged_column_linear(source_layer):
            if len(packed_modules_list) != 1:
                return False
            return not (
                hasattr(source_layer, "output_sizes")
                and len(source_layer.output_sizes) >= 3
            )
        return False

    def merged_can_replace_layer(
        cls,
        source_layer,
        lora_config,
        packed_modules_list,
        model_config=None,
        *,
        decorate: bool = True,
    ) -> bool:
        return (
            _not_fully_sharded(lora_config, decorate)
            and _is_merged_column_linear(source_layer)
            and len(packed_modules_list) == 2
        )

    def qkv_can_replace_layer(
        cls,
        source_layer,
        lora_config,
        packed_modules_list,
        model_config=None,
        *,
        decorate: bool = True,
    ) -> bool:
        return (
            _not_fully_sharded(lora_config, decorate)
            and _is_qkv_linear(source_layer)
            and len(packed_modules_list) == 1
        )

    def merged_qkv_can_replace_layer(
        cls,
        source_layer,
        lora_config,
        packed_modules_list,
        model_config=None,
        *,
        decorate: bool = True,
    ) -> bool:
        return (
            _not_fully_sharded(lora_config, decorate)
            and _is_qkv_linear(source_layer)
            and len(packed_modules_list) == 3
        )

    def variable_slice_can_replace_layer(
        cls,
        source_layer,
        lora_config,
        packed_modules_list,
        model_config=None,
        *,
        decorate: bool = True,
    ) -> bool:
        if not _not_fully_sharded(lora_config, decorate):
            return False
        if not _is_merged_column_linear(source_layer):
            return False
        if len(packed_modules_list) >= 3:
            return True
        if len(packed_modules_list) == 2:
            return False
        return (
            hasattr(source_layer, "output_sizes")
            and len(source_layer.output_sizes) >= 3
        )

    lora_linear.ColumnParallelLinearWithLoRA.__init__ = patched_init
    lora_linear.MergedColumnParallelLinearWithLoRA.__init__ = patched_merged_init
    lora_linear.MergedColumnParallelLinearWithLoRA.slice_lora_b = patched_slice_lora_b
    lora_linear.ColumnParallelLinearWithLoRA.can_replace_layer = classmethod(
        column_can_replace_layer
    )
    lora_linear.MergedColumnParallelLinearWithLoRA.can_replace_layer = classmethod(
        merged_can_replace_layer
    )
    lora_linear.QKVParallelLinearWithLoRA.can_replace_layer = classmethod(
        qkv_can_replace_layer
    )
    lora_linear.MergedQKVParallelLinearWithLoRA.can_replace_layer = classmethod(
        merged_qkv_can_replace_layer
    )
    lora_linear.MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer = classmethod(
        variable_slice_can_replace_layer
    )
    lora_linear._hcu_lora_column_parallel_linear_patch_applied = True
