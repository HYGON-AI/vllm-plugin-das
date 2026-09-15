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
    return weights
