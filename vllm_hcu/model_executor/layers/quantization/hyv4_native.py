# SPDX-License-Identifier: Apache-2.0
"""Read native HYV4 GPTQ channel INT4 checkpoints with existing aiter methods."""

import json
import re
from pathlib import Path

import torch

from .hyv4_w4a8 import HYV4W4A8Config

NATIVE_FORMAT = "hy4_w4a8_v1"


class HYV4NativeW4A8Config(HYV4W4A8Config):
    checkpoint_format = NATIVE_FORMAT

    def __init__(self, checkpoint_index: str):
        index_path = Path(checkpoint_index)
        with index_path.open() as source:
            index = json.load(source)
        if index.get("format") != NATIVE_FORMAT or index.get("complete") is not True:
            raise ValueError("Expected complete hy4_w4a8_v1 checkpoint index")
        with (index_path.parent / "config.json").open() as source:
            mtp_start = json.load(source)["num_hidden_layers"]
        modules = set()
        for name, entry in index["parameters"].items():
            if entry["kind"] != "quantized":
                continue
            # MTP module construction uses model.layers.N as its quant prefix.
            name = re.sub(r"^model\.mtp_layers\.(\d+)\.",
                          lambda m: f"model.layers.{mtp_start + int(m[1])}.", name)
            name = name.removesuffix(".weight")
            if re.search(r"\.experts\.\d+\.", name):
                name = name.split(".experts.")[0] + ".experts"
            else:
                name = name.replace(".gate_proj", ".gate_up_proj")
                name = name.replace(".up_proj", ".gate_up_proj")
            modules.add(name)
        self.quantized_modules = frozenset(modules)


def adapt_native_weights(weights):
    """Native scales already have [N,1]; packed signed INT4 is low-first."""
    for name, weight in weights:
        if name.endswith(".weight.packed"):
            if weight.dtype != torch.uint8 or weight.ndim != 2:
                raise ValueError(f"Expected rank-2 UINT8 native INT4 weight: {name}")
            yield name.removesuffix(".packed"), weight
        elif name.endswith(".weight.scale"):
            if (weight.dtype != torch.float32 or weight.ndim != 2
                    or weight.shape[-1] != 1
                    or not torch.all(torch.isfinite(weight) & (weight > 0)).item()):
                raise ValueError(f"Invalid native HYV4 channel scale: {name}")
            yield name.removesuffix(".scale") + "_scale", weight
        elif name.endswith(".weight"):
            # The native writer appends .weight even to scalar/non-linear params.
            yield name.removesuffix(".weight"), weight
        else:
            yield name, weight
