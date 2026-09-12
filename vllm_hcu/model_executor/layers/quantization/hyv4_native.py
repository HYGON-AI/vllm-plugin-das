# SPDX-License-Identifier: Apache-2.0
"""Native HYV4 UINT8/checkpoint-index adaptation over current SlimQuant."""
import json
import re
from pathlib import Path

import torch

from .hyv4_w4a8 import HYV4W4A8Config
from .hyv4_w4a8_weights import (
    NATIVE_FORMAT, quantized_module_name, validate_channel_scale,
    validate_identity_smoothing, validate_metadata,
)


def adapt_native_weights(weights):
    smoothing_names = set()
    for name, weight in weights:
        if name.endswith(".input_scale.weight"):
            name = name.removesuffix(".weight")
        if name.endswith(".input_scale"):
            if name in smoothing_names:
                raise RuntimeError(f"Duplicate HYV4 checkpoint smoothing: {name}")
            smoothing_names.add(name)
            validate_identity_smoothing(weight, name)
        elif name.endswith(".weight.packed"):
            if weight.dtype != torch.uint8 or weight.ndim != 2:
                raise ValueError(f"Expected rank-2 UINT8 native HYV4 INT4 weight: {name}")
            yield name.removesuffix(".packed"), weight
        elif name.endswith(".weight.scale"):
            validate_channel_scale(weight, name)
            if weight.ndim != 2 or weight.shape[-1] != 1:
                raise ValueError(f"Invalid native HYV4 channel scale shape: {name}")
            yield name.removesuffix(".scale") + "_scale", weight
        elif name.endswith(".weight"):
            # The native writer adds one storage suffix even to retained
            # scalars, norms and parameters already named *.weight.
            yield name.removesuffix(".weight"), weight
        else:
            yield name, weight


class HYV4NativeW4A8Config(HYV4W4A8Config):
    checkpoint_format = NATIVE_FORMAT
    manifest_field = "checkpoint_index"
    adapt_weights = staticmethod(adapt_native_weights)

    def __init__(self, checkpoint_index: str):
        path = Path(checkpoint_index)
        with path.open() as source:
            index = json.load(source)
        if (not isinstance(index, dict) or index.get("format") != NATIVE_FORMAT
                or index.get("complete") is not True):
            raise ValueError(f"Expected complete {NATIVE_FORMAT} checkpoint index")
        parameters = index.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("HYV4 native index requires parameters")
        with (path.parent / "config.json").open() as source:
            mtp_start = json.load(source).get("num_hidden_layers")
        if type(mtp_start) is not int or mtp_start <= 0:
            raise ValueError("HYV4 native config requires positive num_hidden_layers")
        modules = set()
        for name, metadata in parameters.items():
            validate_metadata(metadata, name)
            if metadata.get("kind") not in ("retained", "quantized"):
                raise ValueError(f"Unsupported HYV4 native parameter kind: {name}")
            if metadata["kind"] == "retained":
                continue
            if name.startswith("model.mtp_layers.") and not name.startswith("model.mtp_layers.0."):
                raise ValueError(f"HYV4 requires exactly one checkpoint MTP layer: {name}")
            name = re.sub(r"^model\.mtp_layers\.(\d+)\.",
                          lambda match: f"model.layers.{mtp_start + int(match[1])}.", name)
            modules.add(quantized_module_name(name))
        if not modules:
            raise ValueError("HYV4 native index requires quantized parameters")
        self.quantized_modules = frozenset(modules)
