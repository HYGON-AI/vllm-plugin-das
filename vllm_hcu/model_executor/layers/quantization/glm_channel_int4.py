# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Config checks and weight-layout conversion for GLM-5.3 channel INT4.

Routed experts use compressed-tensors pack-quantized storage: eight INT4
values are packed into each int32 from the low bits, stored as uint4b8
(value plus 8). Existing W4A8 kernels consume the SlimQuant layout, where
the high nibble of each byte is even K, the low nibble is odd K, and the
values are two's complement. This module only converts between those two
layouts, applies the AITER gemm2 reorder, and classifies checkpoint tensor
names.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import torch

_INT4_PACK_FACTOR = 8


def repack_slimquant_w4a8_for_aiter(weight: torch.Tensor) -> torch.Tensor:
    """Regroup high-nibble-first packed INT4 bytes for AITER W4A8.

    ``convert_compressed_tensors_int4_to_slimquant`` fixes the nibble order
    used by the checkpoint.  AITER's W4A8 MoE layout additionally groups eight
    logical INT4 values as ``[0..3]`` in the high nibbles and ``[4..7]`` in
    the low nibbles.  This conversion is deliberately separate from the
    checkpoint loader so it is applied once to the rank-3 expert tensor after
    all experts have been assembled.
    """
    if weight.dtype != torch.int8 or weight.ndim < 1 or weight.shape[-1] <= 0:
        raise ValueError(
            "AITER W4A8 repack expects a non-empty int8 packed tensor, "
            f"got dtype={weight.dtype} shape={tuple(weight.shape)}."
        )

    packed_u8 = weight.contiguous().view(torch.uint8)
    nibbles = torch.stack(
        ((packed_u8 >> 4) & 0x0F, packed_u8 & 0x0F), dim=-1
    ).reshape(*packed_u8.shape[:-1], -1)
    if nibbles.shape[-1] % 8:
        raise ValueError(
            "AITER W4A8 repack requires the logical INT4 dimension to be "
            f"divisible by 8, got shape={tuple(weight.shape)}."
        )

    blocks = nibbles.reshape(*packed_u8.shape[:-1], -1, 8)
    repacked = ((blocks[..., :4] << 4) | blocks[..., 4:]).reshape_as(packed_u8)
    return repacked.contiguous().view(torch.int8)


def repack_and_shuffle_w4a8_gemm2(
    weight: torch.Tensor,
    shuffle_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Convert SlimQuant-packed ``[E, N, K/2]`` experts to the AITER moe_c gemm2 layout.

    Regroup eight nibbles first, then call AITER's
    ``w4a8_moe_layout_shuffle_gemm2`` on each expert. The result is a one-shot
    destructive conversion. It must run once at load time and must not run
    again in forward.
    """
    if weight.ndim != 3 or weight.dtype != torch.int8:
        raise ValueError(
            "W4A8 gemm2 repack expects [E, N, K/2] int8 weights, "
            f"got dtype={weight.dtype} shape={tuple(weight.shape)}."
        )
    if shuffle_fn is None:
        try:
            from aiter.ops.shuffle import (
                w4a8_moe_layout_shuffle_gemm2 as shuffle_fn,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GLM-5.3 W4A8 AITER path requires "
                "aiter.ops.shuffle.w4a8_moe_layout_shuffle_gemm2; this AITER "
                "build does not provide it. Use --moe-backend triton instead."
            ) from exc

    repacked = repack_slimquant_w4a8_for_aiter(weight).view(torch.uint8)
    shuffled = torch.stack(
        [shuffle_fn(repacked[index]) for index in range(weight.shape[0])]
    ).reshape(weight.shape)
    return shuffled.contiguous().view(torch.int8)


def is_glm_moe_dsa_model(hf_config: Any) -> bool:
    """Return whether the config is a GLM MoE DSA model."""
    if getattr(hf_config, "model_type", None) == "glm_moe_dsa":
        return True
    architectures = getattr(hf_config, "architectures", None) or ()
    return "GlmMoeDsaForCausalLM" in architectures


# Shared experts and attention projections are not probes. Layer indexes come
# from the model depth and from literal indexes written in the targets.
_MTP_BLOCK_MARKER = ".mtp_block."
_LAYER_INDEX_IN_TARGET = re.compile(r"layers\\*\.(\d+)")
# Class-name targets such as Linear stay on the parent method. RoutedExperts
# is the fused expert module and is matched the same way as the facade.
_GENERIC_CLASS_TARGETS = frozenset({"Linear", "Module", "object"})
# Backbone and MTP experts share this shape. The index is what a published
# regex often fails to extend past ``num_hidden_layers``.
_LAYER_EXPERT_PREFIX = re.compile(
    r"^(?P<head>.*?layers\.)(?P<idx>\d+)"
    r"(?P<tail>(?:\.mtp_block)?\.mlp\.experts(?:\..*)?)$"
)


def _quant_field(value: Any, name: str, default: Any = None) -> Any:
    """Read one quantization field from a raw dict or QuantizationArgs."""
    if value is None:
        return None
    if isinstance(value, dict):
        field = value.get(name, default)
    else:
        field = getattr(value, name, default)
    return getattr(field, "value", field)


def _is_channel_int4_w4a8_scheme(
    weights: Any,
    activations: Any,
    group_format: Any,
) -> bool:
    # Match QuantizationArgs' default without accepting explicit false/null.
    return (
        _quant_field(weights, "num_bits") == 4
        and _quant_field(weights, "strategy") == "channel"
        and _quant_field(weights, "type") == "int"
        and _quant_field(weights, "symmetric", True) is True
        and _quant_field(activations, "num_bits") == 8
        and _quant_field(activations, "strategy") == "token"
        and _quant_field(activations, "dynamic") is True
        and _quant_field(activations, "type") == "int"
        and _quant_field(activations, "symmetric") is not False
        and getattr(group_format, "value", group_format) == "pack-quantized"
    )


def _is_channel_int8_scheme(weights: Any, activations: Any) -> bool:
    return (
        _quant_field(weights, "num_bits") == 8
        and _quant_field(weights, "strategy") == "channel"
        and _quant_field(weights, "type") == "int"
        and _quant_field(activations, "num_bits") == 8
        and _quant_field(activations, "strategy") == "token"
        and _quant_field(activations, "dynamic") is True
        and _quant_field(activations, "type") == "int"
    )


def _group_targets(group: dict[str, Any]) -> list[str]:
    targets = group.get("targets") or []
    if not isinstance(targets, list):
        return []
    return [target for target in targets if isinstance(target, str)]


def _layer_indexes_from_targets(targets: list[str]) -> set[int]:
    indexes: set[int] = set()
    for target in targets:
        for match in _LAYER_INDEX_IN_TARGET.finditer(target):
            indexes.add(int(match.group(1)))
    return indexes


def _probe_layer_indexes(targets: list[str], hf_config: Any) -> list[int]:
    """Layer indexes an INT4 expert target might name.

    Uses the model depth, including MTP layers, plus any literal layer index
    written in a target. An unanchored pattern with neither source is checked
    against layer 0.
    """
    indexes = _layer_indexes_from_targets(targets)
    if hf_config is not None:
        count = getattr(hf_config, "num_hidden_layers", None)
        nextn = getattr(hf_config, "num_nextn_predict_layers", 0) or 0
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            if not isinstance(nextn, int) or isinstance(nextn, bool) or nextn < 0:
                nextn = 0
            indexes.update(range(count + nextn))
    if not indexes:
        indexes.add(0)
    return sorted(indexes)


def _routed_expert_probes(
    targets: list[str], hf_config: Any = None
) -> list[tuple[str, str]]:
    """Return ``(module_prefix, match_name)`` for candidate expert layers."""
    probes = []
    for index in _probe_layer_indexes(targets, hf_config):
        prefix = f"model.layers.{index}.mlp.experts"
        probes.append((prefix, prefix))
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            probes.append((prefix, f"{prefix}.0.{proj_name}"))
    return probes


def _targets_cover_routed_experts(
    targets: list[str],
    ignore: Any = None,
    hf_config: Any = None,
) -> bool:
    """Return whether an INT4 target selects a routed expert that is not ignored.

    Early dense layers may be ignored or absent from the INT4 regex. One later
    expert layer is enough. Shared experts and attention projections do not
    match these probes.
    """
    from vllm.model_executor.layers.quantization.utils.config_utils import (
        is_equal_or_regex_match,
    )

    for module_prefix, probe in _routed_expert_probes(targets, hf_config):
        if _ignore_covers_expert(None, module_prefix, ignore, {}):
            continue
        for target in targets:
            if target == "RoutedExperts" or is_equal_or_regex_match(probe, target):
                return True
    return False


def _ignore_covers_prefix(prefix: str, ignore: Any, fused_mapping: Any = None) -> bool:
    if not ignore:
        return False
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    return should_ignore_layer(
        prefix,
        ignore=ignore,
        fused_mapping=fused_mapping or {},
    )


def is_glm_channel_int4_w4a8(
    hf_quant_cfg: dict[str, Any] | None,
    hf_config: Any = None,
) -> bool:
    """Return whether the quant config mixes channel INT4 experts and channel INT8 linears.

    The INT4 group must target a routed-expert layer that ``ignore`` does not
    exclude. Early dense layers may be ignored or absent from the INT4 regex.
    An INT4 group that only covers non-expert layers does not select this method.
    """
    if not isinstance(hf_quant_cfg, dict):
        return False
    if hf_quant_cfg.get("quant_method") != "compressed-tensors":
        return False
    groups = hf_quant_cfg.get("config_groups")
    if not isinstance(groups, dict):
        return False
    int4_targets: list[str] = []
    has_int8 = False
    ignore = hf_quant_cfg.get("ignore") or []
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights") or {}
        activations = group.get("input_activations") or {}
        group_format = group.get("format") or hf_quant_cfg.get("format")
        if _is_channel_int4_w4a8_scheme(weights, activations, group_format):
            int4_targets.extend(_group_targets(group))
        if _is_channel_int8_scheme(weights, activations):
            has_int8 = True
    if not has_int8 or not int4_targets:
        return False
    return _targets_cover_routed_experts(int4_targets, ignore, hf_config)


def _effective_scheme_format(scheme: Any, config: Any) -> Any:
    """Return the group format, or the config-wide format when the group omits it.

    The native parser stores a missing group format as None and fills
    ``config.quant_format`` only inside ``get_scheme_dict``.
    """
    group_format = scheme.get("format") if isinstance(scheme, dict) else None
    if group_format is not None:
        return group_format
    return getattr(config, "quant_format", None)


def _scheme_is_channel_int4(scheme: Any, config: Any = None) -> bool:
    if not isinstance(scheme, dict):
        return False
    return _is_channel_int4_w4a8_scheme(
        scheme.get("weights"),
        scheme.get("input_activations"),
        _effective_scheme_format(scheme, config),
    )


def _match_target_by_name(
    name: str,
    targets: Any,
    fused_mapping: Any,
) -> str | None:
    """Match a layer name to a compressed-tensors target, without class fallback."""
    from vllm.model_executor.layers.quantization.utils.config_utils import (
        is_equal_or_regex_match,
    )

    target_list = [target for target in targets if isinstance(target, str)]
    for target in target_list:
        if is_equal_or_regex_match(name, target):
            return target
    mapping = fused_mapping or {}
    fused = next((key for key in mapping if name.endswith(key)), None)
    if fused is None:
        return None
    unfused_paths = [name.replace(fused, part) for part in mapping[fused]]
    matched: list[str] = []
    for unfused in unfused_paths:
        found = next(
            (
                target
                for target in target_list
                if is_equal_or_regex_match(unfused, target)
            ),
            None,
        )
        if found is None:
            return None
        matched.append(found)
    return matched[0] if matched else None


def _match_target_by_class(layer: Any, targets: Any) -> str | None:
    """Match ``RoutedExperts`` and other non-generic class targets."""
    if layer is None:
        return None
    target_list = [
        target
        for target in targets
        if isinstance(target, str) and not target.startswith("re:")
    ]
    for cls in type(layer).__mro__:
        name = cls.__name__
        if name in _GENERIC_CLASS_TARGETS:
            continue
        if name in target_list:
            return name
    return None


def _projection_names(layer: Any, prefix: str) -> tuple[str, str, str]:
    return tuple(f"{prefix}.0.{proj_name}" for proj_name in _expert_proj_names(layer))


def _expert_proj_names(layer: Any) -> tuple[str, str, str]:
    return (
        getattr(layer, "ckpt_gate_proj_name", "gate_proj"),
        getattr(layer, "ckpt_up_proj_name", "up_proj"),
        getattr(layer, "ckpt_down_proj_name", "down_proj"),
    )


def _ignore_covers_expert(
    layer: Any, prefix: str, ignore: Any, fused_mapping: Any
) -> bool:
    """Apply owner/projection exclusions to runtime and checkpoint paths."""
    prefixes = [prefix]
    if _MTP_BLOCK_MARKER in prefix:
        prefixes.append(prefix.replace(_MTP_BLOCK_MARKER, "."))
    return any(
        _ignore_covers_prefix(candidate, ignore, fused_mapping)
        or any(
            _ignore_covers_prefix(proj_name, ignore, fused_mapping)
            for proj_name in _projection_names(layer, candidate)
        )
        for candidate in prefixes
    )


def _explicit_expert_scheme(
    config: Any, layer: Any, prefix: str
) -> tuple[bool, Any]:
    """Return ``(matched, scheme)`` for a routed-expert prefix.

    ``matched`` is false when no target names this prefix, its class, or all
    three projections. Projections may use different target strings; they
    select INT4 only when every projection's scheme is channel INT4. A generic
    ``Linear`` class target is not an expert scheme.
    """
    scheme_map = getattr(config, "target_scheme_map", None) or {}
    fused_mapping = getattr(config, "packed_modules_mapping", None) or {}
    matched = _match_target_by_name(prefix, scheme_map, fused_mapping)
    if matched is None:
        matched = _match_target_by_class(layer, scheme_map)
    if matched is not None:
        return True, scheme_map.get(matched)
    proj_schemes = []
    for proj_name in _projection_names(layer, prefix):
        proj_target = _match_target_by_name(proj_name, scheme_map, fused_mapping)
        if proj_target is None:
            return False, None
        proj_schemes.append(scheme_map.get(proj_target))
    if all(_scheme_is_channel_int4(scheme, config) for scheme in proj_schemes):
        return True, proj_schemes[0]
    return True, None


def _infer_num_hidden_layers() -> int | None:
    """Return the active model backbone size, when vLLM has one."""
    try:
        from vllm.config import get_current_vllm_config

        hf_config = get_current_vllm_config().model_config.hf_config
    except Exception:
        return None
    count = getattr(hf_config, "num_hidden_layers", None)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def _mtp_alias_prefixes(prefix: str, num_hidden_layers: int | None) -> list[str]:
    """Prefixes to retry when a published expert regex misses MTP.

    ``.mtp_block.`` is only inserted into some checkpoint names. The quant
    prefix itself stays ``model.layers.{N}.mlp.experts``, with ``N`` at or
    past ``num_hidden_layers``. Backbone experts keep their own index and are
    not rewritten.
    """
    aliases: list[str] = []
    if _MTP_BLOCK_MARKER in prefix:
        aliases.append(prefix.replace(_MTP_BLOCK_MARKER, "."))
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        return _dedupe_prefixes(prefix, aliases)
    match = _LAYER_EXPERT_PREFIX.fullmatch(prefix)
    if match is None or int(match.group("idx")) < num_hidden_layers:
        return _dedupe_prefixes(prefix, aliases)
    head = match.group("head")
    tail = match.group("tail")
    tail_without_mtp = tail.replace(_MTP_BLOCK_MARKER, ".")
    for backbone in range(num_hidden_layers):
        aliases.append(f"{head}{backbone}{tail}")
        if tail_without_mtp != tail:
            aliases.append(f"{head}{backbone}{tail_without_mtp}")
    return _dedupe_prefixes(prefix, aliases)


def _dedupe_prefixes(prefix: str, aliases: list[str]) -> list[str]:
    unique: list[str] = []
    for alias in aliases:
        if alias != prefix and alias not in unique:
            unique.append(alias)
    return unique


def routed_expert_uses_glm_channel_int4(
    config: Any,
    layer: Any,
    prefix: str,
    *,
    num_hidden_layers: int | None = None,
) -> bool:
    """Return whether this routed-expert prefix should load channel INT4.

    ``ignore`` wins. A prefix matches when its name, or all three projection
    names, select the channel INT4 scheme. Different target strings with that
    same scheme still match. MTP is retried only by removing ``.mtp_block.``
    or by comparing a layer index at or past ``num_hidden_layers`` with the
    backbone expert scheme.
    """
    ignore = getattr(config, "ignore", None) or []
    fused_mapping = getattr(config, "packed_modules_mapping", None) or {}
    if _ignore_covers_expert(layer, prefix, ignore, fused_mapping):
        return False
    matched, scheme = _explicit_expert_scheme(config, layer, prefix)
    if matched:
        return _scheme_is_channel_int4(scheme, config)
    unwrapped = prefix.replace(_MTP_BLOCK_MARKER, ".")
    if unwrapped != prefix:
        matched, scheme = _explicit_expert_scheme(config, layer, unwrapped)
        if matched:
            return _scheme_is_channel_int4(scheme, config)
    if num_hidden_layers is None:
        num_hidden_layers = _infer_num_hidden_layers()
    for alias in _mtp_alias_prefixes(prefix, num_hidden_layers):
        if _ignore_covers_expert(layer, alias, ignore, fused_mapping):
            continue
        matched, scheme = _explicit_expert_scheme(config, layer, alias)
        if matched and _scheme_is_channel_int4(scheme, config):
            return True
    return False


def _expert_owner_prefixes(name: str) -> list[str]:
    """Runtime expert-module prefixes that can own this checkpoint tensor."""
    marker = ".mlp.experts."
    if marker not in name:
        return []
    head = name.split(marker, 1)[0]
    owners = [head + ".mlp.experts"]
    if ".mtp_block" not in head:
        owners.append(head + ".mtp_block.mlp.experts")
    return owners


def _expert_weight_modules(owner: str) -> tuple[str, str]:
    """Parameter prefixes that can hold this owner's fused expert weights.

    Current FusedMoE registers weights on ``experts.routed_experts``. Older
    modules registered them directly on ``experts``.
    """
    return owner, f"{owner}.routed_experts"


def _module_uses_unpacked_expert_weight(
    module: str, params_dict: Mapping[str, Any]
) -> bool | None:
    """Return the naming style of one expert module, or None when it is absent."""
    if (
        f"{module}.w13_weight_packed" in params_dict
        or f"{module}.w2_weight_packed" in params_dict
    ):
        return False
    if f"{module}.w13_weight" in params_dict or f"{module}.w2_weight" in params_dict:
        return True
    return None


def _owner_uses_unpacked_expert_weight(
    name: str, params_dict: Mapping[str, Any]
) -> bool:
    """Return whether every located owner stores ``w13_weight`` or ``w2_weight``."""
    decisions = []
    for owner in _expert_owner_prefixes(name):
        for module in _expert_weight_modules(owner):
            decision = _module_uses_unpacked_expert_weight(module, params_dict)
            if decision is not None:
                decisions.append(decision)
                break
    return bool(decisions) and all(decisions)


def normalize_routed_expert_checkpoint_name(
    name: str,
    experts_use_unpacked_weight: bool = False,
    params_dict: Mapping[str, Any] | None = None,
) -> str | None:
    """Rewrite packed routed-expert tensor names so the existing loader can match them.

    When ``params_dict`` is given, only the expert module that owns this tensor
    is consulted. Weights may be registered on ``experts`` or on
    ``experts.routed_experts``. A W4A16 owner keeps ``weight_packed``. A W4A8
    owner, including an MTP module registered under ``.mtp_block``, rewrites
    ``weight_packed`` to ``weight`` and drops ``weight_shape``. The boolean is
    the whole-model fallback used when the parameter map is absent.
    """
    if ".mlp.experts." not in name:
        return name
    if params_dict is not None:
        experts_use_unpacked_weight = _owner_uses_unpacked_expert_weight(
            name, params_dict
        )
    if not experts_use_unpacked_weight:
        return name
    if name.endswith(".weight_shape"):
        return None
    if name.endswith(".weight_packed"):
        return name.replace(".weight_packed", ".weight")
    return name


def convert_compressed_tensors_int4_to_slimquant(
    packed: torch.Tensor,
) -> torch.Tensor:
    """Convert compressed-tensors int32 INT4 packing to SlimQuant int8 packing.

    The input's last dimension is ``K / 8``. Each nibble is uint4b8 (value
    plus 8). The output's last dimension is ``K / 2``: the high nibble is
    even K, the low nibble is odd K, and the values are INT4 two's complement.
    """
    if packed.dtype != torch.int32 or packed.ndim < 1 or packed.shape[-1] <= 0:
        raise ValueError(
            "compressed-tensors INT4 weights must be int32 with a positive "
            f"last dimension, got dtype={packed.dtype} shape={tuple(packed.shape)}."
        )
    raw = packed.contiguous().view(torch.uint8)
    if raw.shape[-1] != packed.shape[-1] * (_INT4_PACK_FACTOR // 2):
        raise ValueError(
            "INT4 int32 unpacked byte count is wrong: "
            f"packed={tuple(packed.shape)} bytes={tuple(raw.shape)}."
        )
    # Swap nibbles, then XOR 0x88: flip the high bit of each nibble,
    # which subtracts the uint4b8 bias of 8.
    swapped = (((raw & 0x0F) << 4) | (raw >> 4)) ^ 0x88
    return swapped.view(torch.int8)
