# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-only adapters for the explicit HYV4 signed INT4 formats."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator

import torch

from .slimquant_facade import SlimQuantW4A8Facade

CHECKPOINT_FORMAT = "hy4-w4a8-custom-v1"
NATIVE_FORMAT = "hy4_w4a8_v1"
PACKING = "signed nibble, even input in low bits"


class HYV4W4A8Facade(SlimQuantW4A8Facade):
    """Extend only materialization at the existing SlimQuant registry seam."""

    @classmethod
    def from_config(cls, config):
        if config.get("quant_method") == "slimquant_w4a8":
            if config.get("checkpoint_format") == CHECKPOINT_FORMAT:
                from .hyv4_w4a8 import HYV4W4A8Config

                return HYV4W4A8Config.from_config(config)
            if config.get("checkpoint_format") == NATIVE_FORMAT:
                from .hyv4_native import HYV4NativeW4A8Config

                return HYV4NativeW4A8Config.from_config(config)
        return super().from_config(config)


def validate_metadata(metadata: dict, name: str) -> None:
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid HYV4 metadata for {name}")
    for field, expected in (("bits", 4), ("group_size", -1),
                            ("symmetric", True), ("packing", PACKING)):
        if field in metadata and (
            type(metadata[field]) is not type(expected) or metadata[field] != expected
        ):
            raise ValueError(f"Unsupported HYV4 {field} for {name}")


def quantized_module_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("HYV4 quantized tensor names must be nonempty strings")
    name = name.removesuffix(".weight")
    if re.search(r"\.experts\.(?:\d+\.)?(?:gate_up_proj|gate_proj|up_proj|down_proj)$", name):
        return name.split(".experts.")[0] + ".experts"
    return re.sub(r"\.(?:gate_proj|up_proj)$", ".gate_up_proj", name)


def read_quantized_modules(manifest_path: str) -> frozenset[str]:
    with open(manifest_path) as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict) or manifest.get("provenance", {}).get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Expected {CHECKPOINT_FORMAT} conversion manifest")
    tensors = manifest.get("quantized_tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("HYV4 conversion manifest needs quantized_tensors")
    modules = set()
    for name, metadata in tensors.items():
        validate_metadata(metadata, name)
        if metadata.get("packing") != PACKING:
            raise ValueError(f"Unsupported HYV4 INT4 packing for {name}")
        modules.add(quantized_module_name(name))
    return frozenset(modules)


def to_aiter_packing(weight: torch.Tensor) -> torch.Tensor:
    """Swap low-first checkpoint nibbles to current high-first packed INT4."""
    if weight.dtype not in (torch.uint8, torch.int8):
        raise ValueError("HYV4 packed INT4 must have byte storage")
    unsigned = weight.view(torch.uint8)
    return ((unsigned << 4) | (unsigned >> 4)).view(torch.int8)


def validate_channel_scale(weight: torch.Tensor, name: str) -> None:
    if (weight.dtype != torch.float32 or not weight.numel()
            or not torch.all(torch.isfinite(weight) & (weight > 0)).item()):
        raise ValueError(f"Invalid HYV4 channel scale: {name}")


def validate_identity_smoothing(weight: torch.Tensor, name: str) -> None:
    if not weight.numel() or not torch.all(weight == 1).item():
        raise ValueError(f"HYV4 W4A8 requires identity input_scale: {name}")


def adapt_weights(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    smoothing_names = set()
    for name, weight in weights:
        if name.endswith(".input_scale"):
            if name in smoothing_names:
                raise RuntimeError(f"Duplicate HYV4 checkpoint smoothing: {name}")
            smoothing_names.add(name)
            validate_identity_smoothing(weight, name)
        elif name.endswith(".int4_packed"):
            if weight.dtype != torch.uint8 or weight.ndim not in (2, 3):
                raise ValueError(f"Expected rank-2/3 UINT8 HYV4 packed tensor: {name}")
            yield name.removesuffix(".int4_packed"), weight
        elif name.endswith(".scale"):
            validate_channel_scale(weight, name)
            if weight.ndim not in (1, 2):
                raise ValueError(f"Invalid HYV4 channel scale shape: {name}")
            yield name.removesuffix(".scale") + "_scale", weight.unsqueeze(-1)
        else:
            yield name, weight


def adapt_checkpoint_weights(quant_config, weights):
    """Leave every non-HYV4 owner and its original tensor stream untouched."""
    if getattr(quant_config, "checkpoint_format", None) in (CHECKPOINT_FORMAT, NATIVE_FORMAT):
        from .hyv4_w4a8 import HYV4W4A8Config

        if isinstance(quant_config, HYV4W4A8Config):
            return quant_config.adapt_weights(weights)
    get_name = getattr(quant_config, "get_name", None)
    if callable(get_name) and get_name() == "compressed-tensors":
        return _adapt_compressed_tensors_experts(quant_config, weights)
    return weights


def _matches_compressed_target(target: str, name: str) -> bool:
    if target in ("Linear", "RoutedExperts"):
        return True
    if target.startswith("re:"):
        return re.fullmatch(target[3:], name) is not None
    return name == target or name.endswith("." + target)


def _is_supported_compressed_expert(quant_config, name: str) -> bool:
    config = getattr(quant_config, "config", None)
    if not isinstance(config, dict):
        return False
    for group in config.get("config_groups", {}).values():
        targets = group.get("targets", ())
        if not any(_matches_compressed_target(target, name) for target in targets):
            continue
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        return bool(
            group.get("format", config.get("format")) == "pack-quantized"
            and weight.get("num_bits") == 4
            and weight.get("type") == "int"
            and weight.get("strategy") == "channel"
            and weight.get("symmetric") is True
            and weight.get("dynamic") is False
            and weight.get("group_size") in (None, -1)
            and activation.get("num_bits") == 8
            and activation.get("type") == "int"
            and activation.get("strategy") == "token"
            and activation.get("symmetric") is True
            and activation.get("dynamic") is True
        )
    return False


def _adapt_compressed_tensors_experts(
    quant_config,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Rename split expert tensors before PP-local parameter loading."""
    expert = re.compile(r"\.mlp\.experts\.\d+\.")
    for name, weight in weights:
        match = expert.search(name)
        projection_name = re.sub(
            r"\.(?:weight_packed|weight_scale|weight_shape)$", "", name
        )
        if match is None or not _is_supported_compressed_expert(
            quant_config, projection_name
        ):
            yield name, weight
            continue
        if name.endswith(".weight_shape"):
            continue
        if name.endswith(".weight_packed"):
            if weight.dtype != torch.int32 or weight.ndim != 2:
                raise ValueError(
                    "HYV4 compressed-tensors expert weight_packed must be "
                    "rank-2 INT32"
                )
            yield name.removesuffix(".weight_packed") + ".weight", weight
            continue
        yield name, weight
