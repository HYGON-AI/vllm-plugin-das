# SPDX-License-Identifier: Apache-2.0
"""Immutable 128-slot SGLang decode placement on the existing DP8 mesh."""
from dataclasses import dataclass
import re

from .minimax31_decode_contract import digest


ATTRIBUTE = "hcu_minimax_decode_placement"
LAYERS = tuple(range(3, 60))
_FIELDS = {
    "schema", "expert_dispatch_algorithm", "ep_size", "logical_experts",
    "physical_experts", "slots_per_rank", "source", "source_sha256",
    "map_sha256", "model_layer_ids", "physical_to_logical",
}


@dataclass(frozen=True)
class DecodePlacement:
    map_sha256: str
    logical_to_physical: tuple[tuple[int, ...], ...]

    def for_layer(self, name):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match is None or (index := int(match[1])) not in LAYERS:
            raise ValueError("decode placement requires a MoE layer from 3 through 59")
        return index, self.logical_to_physical[index - LAYERS[0]]


def validate_placement(value):
    expected = {
        "schema": "minimax31-decode-static128-v1", "expert_dispatch_algorithm": "static",
        "ep_size": 8, "logical_experts": 128, "physical_experts": 128, "slots_per_rank": 16,
        "model_layer_ids": list(LAYERS),
    }
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("invalid decode static permutation fields")
    if any(type(value[key]) is not type(v) or value[key] != v for key, v in expected.items()):
        raise ValueError("decode static permutation requires the SGLang static 128-slot contract")
    if not isinstance(value["source"], str) or not value["source"]:
        raise ValueError("decode static placement requires an artifact source")
    if not isinstance(value["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]):
        raise ValueError("decode static placement requires the SGLang artifact SHA256")
    rows = value["physical_to_logical"]
    if not isinstance(rows, list) or len(rows) != len(LAYERS):
        raise ValueError("decode static placement must contain all 57 MoE layers")
    target = set(range(128))
    inverses = []
    for row in rows:
        if (not isinstance(row, list) or len(row) != 128
                or any(type(expert) is not int for expert in row) or set(row) != target):
            raise ValueError("each decode layer requires a one-to-one 128-expert permutation")
        logical_to_physical = [0] * 128
        for physical, logical in enumerate(row):
            logical_to_physical[logical] = physical
        inverses.append(tuple(logical_to_physical))
    actual = digest({"model_layer_ids": list(LAYERS), "physical_to_logical": rows})
    if value["map_sha256"] != actual:
        raise ValueError("decode static expert map SHA256 differs")
    return DecodePlacement(actual, tuple(inverses))


def configured_placement(model_config):
    hf = model_config.hf_config
    outer = getattr(hf, ATTRIBUTE, None)
    nested = getattr(getattr(hf, "text_config", None), ATTRIBUTE, None)
    if outer is not None and nested is not None and outer != nested:
        raise ValueError("nested decode placement differs from the outer model")
    return outer if outer is not None else nested


def placement_for_model(model_config):
    value = configured_placement(model_config)
    return validate_placement(value) if value is not None else None
