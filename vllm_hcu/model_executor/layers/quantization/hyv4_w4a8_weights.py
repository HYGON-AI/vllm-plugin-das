# SPDX-License-Identifier: Apache-2.0
"""Streaming adaptation of the explicit HYV4 custom INT4 checkpoint format."""

import json
from collections.abc import Iterable, Iterator

import torch

CHECKPOINT_FORMAT = "hy4-w4a8-custom-v1"


def read_quantized_modules(manifest_path: str) -> frozenset[str]:
    with open(manifest_path) as source:
        manifest = json.load(source)
    if manifest.get("provenance", {}).get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Expected {CHECKPOINT_FORMAT} conversion manifest")
    modules = set()
    for name, metadata in manifest["quantized_tensors"].items():
        if metadata.get("packing") != "signed nibble, even input in low bits":
            raise ValueError(f"Unsupported HYV4 INT4 packing for {name}")
        name = name.removesuffix(".weight")
        if ".experts." in name:
            name = name.rsplit(".", 1)[0]
        else:
            name = name.replace(".gate_proj", ".gate_up_proj")
            name = name.replace(".up_proj", ".gate_up_proj")
        modules.add(name)
    return frozenset(modules)


def to_aiter_packing(weight: torch.Tensor) -> torch.Tensor:
    """Signed two's-complement INT4: checkpoint low-first -> aiter high-first."""
    if weight.dtype not in (torch.uint8, torch.int8):
        raise ValueError("HYV4 packed INT4 must have byte storage")
    unsigned = weight.view(torch.uint8)
    return ((unsigned << 4) | (unsigned >> 4)).view(torch.int8)


def adapt_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Map names without expanding or retaining whole expert tensors on CPU.

    Nibble conversion occurs after local parameter sharding. Input scales in
    this uncalibrated RTN format are identity smoothing vectors, not runtime
    per-token activation scales; reject nonidentity vectors explicitly.
    """
    for name, weight in weights:
        if name.endswith(".input_scale"):
            if not torch.all(weight == 1).item():
                raise ValueError(f"HYV4 W4A8 requires identity input_scale: {name}")
            continue
        if name.endswith(".int4_packed"):
            if weight.dtype != torch.uint8:
                raise ValueError(f"Expected UINT8 packed checkpoint tensor: {name}")
            yield name.removesuffix(".int4_packed"), weight
        elif name.endswith(".scale"):
            if not torch.all(torch.isfinite(weight) & (weight > 0)).item():
                raise ValueError(f"Invalid HYV4 W4A8 channel scale: {name}")
            yield name.removesuffix(".scale") + "_scale", weight.unsqueeze(-1)
        else:
            yield name, weight
