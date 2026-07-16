# SPDX-License-Identifier: Apache-2.0
"""Column-parallel LoRA compatibility for HCU linear layers."""

from __future__ import annotations

import importlib
from types import ModuleType

import torch


_TARGET_MODULE = "vllm.lora.layers.column_parallel_linear"
_PATCH_MARKER = "_hcu_lora_column_parallel_linear_patch_applied"
_BINDING_MARKER = "_hcu_lora_column_parallel_compat_binding"

_REQUIRED_TYPES = (
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ColumnParallelLinearWithLoRA",
    "MergedColumnParallelLinearWithLoRA",
    "QKVParallelLinearWithLoRA",
    "MergedQKVParallelLinearWithLoRA",
    "MergedColumnParallelLinearVariableSliceWithLoRA",
)

_PATCHED_BINDINGS = (
    ("ColumnParallelLinearWithLoRA", "__init__", "column_init"),
    ("MergedColumnParallelLinearWithLoRA", "__init__", "merged_init"),
    (
        "MergedColumnParallelLinearWithLoRA",
        "slice_lora_b",
        "merged_slice_lora_b",
    ),
    (
        "ColumnParallelLinearWithLoRA",
        "can_replace_layer",
        "column_can_replace",
    ),
    (
        "MergedColumnParallelLinearWithLoRA",
        "can_replace_layer",
        "merged_can_replace",
    ),
    ("QKVParallelLinearWithLoRA", "can_replace_layer", "qkv_can_replace"),
    (
        "MergedQKVParallelLinearWithLoRA",
        "can_replace_layer",
        "merged_qkv_can_replace",
    ),
    (
        "MergedColumnParallelLinearVariableSliceWithLoRA",
        "can_replace_layer",
        "variable_slice_can_replace",
    ),
)


def _require_lora_module(module: ModuleType | None) -> ModuleType:
    if module is None:
        # The zero-argument direct API deliberately imports the complete child
        # name.  Runtime callbacks always pass the completed module and never
        # execute this import path.
        module = importlib.import_module(_TARGET_MODULE)
    if not isinstance(module, ModuleType) or module.__name__ != _TARGET_MODULE:
        actual_name = getattr(module, "__name__", None)
        raise TypeError(f"expected module {_TARGET_MODULE!r}, got {actual_name!r}")
    for type_name in _REQUIRED_TYPES:
        if not isinstance(getattr(module, type_name, None), type):
            raise RuntimeError(
                f"required LoRA runtime target {_TARGET_MODULE}.{type_name} is missing"
            )
    return module


def _mark_binding(function, role: str) -> None:
    setattr(function, _BINDING_MARKER, role)


def _binding_function(owner: type, attribute: str):
    binding = owner.__dict__.get(attribute)
    if isinstance(binding, classmethod):
        return binding.__func__
    return binding


def _verify_patched_bindings(lora_linear: ModuleType) -> None:
    for class_name, attribute, expected_role in _PATCHED_BINDINGS:
        owner = getattr(lora_linear, class_name)
        function = _binding_function(owner, attribute)
        if getattr(function, _BINDING_MARKER, None) != expected_role:
            raise RuntimeError(
                "HCU LoRA column-parallel postcondition failed for "
                f"{_TARGET_MODULE}.{class_name}.{attribute}"
            )


def install_hcu_lora_column_parallel_compat(
    lora_linear: ModuleType | None = None,
) -> None:
    """Install HCU linear-type and grouped-QKVZ LoRA compatibility."""

    lora_linear = _require_lora_module(lora_linear)

    if getattr(lora_linear, _PATCH_MARKER, False):
        _verify_patched_bindings(lora_linear)
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

    _mark_binding(patched_init, "column_init")
    _mark_binding(patched_merged_init, "merged_init")
    _mark_binding(patched_slice_lora_b, "merged_slice_lora_b")
    _mark_binding(column_can_replace_layer, "column_can_replace")
    _mark_binding(merged_can_replace_layer, "merged_can_replace")
    _mark_binding(qkv_can_replace_layer, "qkv_can_replace")
    _mark_binding(merged_qkv_can_replace_layer, "merged_qkv_can_replace")
    _mark_binding(variable_slice_can_replace_layer, "variable_slice_can_replace")

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
    variable_slice_cls = (
        lora_linear.MergedColumnParallelLinearVariableSliceWithLoRA
    )
    variable_slice_cls.can_replace_layer = classmethod(
        variable_slice_can_replace_layer,
    )
    _verify_patched_bindings(lora_linear)
    # Publish only after every exact binding has passed its postcondition.
    setattr(lora_linear, _PATCH_MARKER, True)
