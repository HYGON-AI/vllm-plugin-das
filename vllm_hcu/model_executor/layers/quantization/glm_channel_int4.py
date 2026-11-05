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
from collections.abc import Callable, Iterable, Mapping
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
    """Return whether the config is a GLM MoE DSA model.

    This covers ``model_type == "glm_moe_dsa"`` and the ``GlmMoeDsaForCausalLM``
    architecture, which the plugin registers as a ``DeepseekV2ForCausalLM``
    subclass. The expert checkpoint-name handling below is wired into that
    model's loader and the DeepSeek MTP loader. ``Glm5Next*`` checkpoints are
    loaded by ``vllm.models.glm5next`` instead, which goes through neither
    loader, so they are intentionally not selected here even though they are
    also called GLM-5.3.
    """
    if getattr(hf_config, "model_type", None) == "glm_moe_dsa":
        return True
    architectures = getattr(hf_config, "architectures", None) or ()
    return "GlmMoeDsaForCausalLM" in architectures


# Shared experts and attention projections are not probes. Layer indexes come
# from the model depth and from literal indexes written in the targets.
_MTP_BLOCK_MARKER = ".mtp_block."
_LAYER_INDEX_IN_TARGET = re.compile(r"layers\\*\.(\d+)")
# Class name that class-name targets are tried on during detection; a test ties
# it to the class so the two cannot drift apart.
_ROUTED_EXPERTS_CLASS = "RoutedExperts"


def _parse_quant_args(value: Any) -> Any:
    """Parse a raw ``weights`` / ``input_activations`` dict like native does.

    ``CompressedTensorsConfig`` builds every scheme with
    ``QuantizationArgs.model_validate``: it fills field defaults, infers
    ``strategy`` from ``group_size``, lower-cases enum strings and rejects
    unsupported combinations. Auto-detection reads the raw HF dict before that
    parse, so it uses the same parser instead of keeping a copy of those rules.
    Anything native cannot parse either (a dict the schema rejects, or any other
    JSON value) selects nothing here and the parent reports the error. ``None``
    and arguments that are already parsed pass through unchanged.
    """
    if isinstance(value, dict):
        from compressed_tensors.quantization import QuantizationArgs

        try:
            return QuantizationArgs.model_validate(value)
        except Exception:
            # Whatever the validator raises (pydantic's ValidationError, or a
            # TypeError / NotImplementedError from a compressed-tensors check),
            # native cannot load this group either, and the parent reports it
            # with its own error. Detection must not abort config resolution.
            return None
    if isinstance(value, (list, tuple, str, bytes, int, float)):
        return None
    return value


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
    # Parsed arguments already carry every default; the fallbacks below also
    # cover argument objects built by hand. An explicit false/null is rejected.
    return (
        _quant_field(weights, "num_bits") == 4
        and _quant_field(weights, "strategy") == "channel"
        and _quant_field(weights, "type", "int") == "int"
        and _quant_field(weights, "symmetric", True) is True
        and _quant_field(activations, "num_bits", 8) == 8
        and _quant_field(activations, "strategy") == "token"
        and _quant_field(activations, "dynamic") is True
        and _quant_field(activations, "type", "int") == "int"
        and _quant_field(activations, "symmetric") is not False
        and getattr(group_format, "value", group_format) == "pack-quantized"
    )


def _is_channel_int8_scheme(weights: Any, activations: Any, group_format: Any) -> bool:
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        is_activation_quantization_format,
    )

    # Native only builds dynamic-token W8A8 for an activation-quantization
    # format and for symmetric weights (input quantization may be either), so
    # a group failing either cannot serve the linears.
    return (
        is_activation_quantization_format(getattr(group_format, "value", group_format))
        and _quant_field(weights, "num_bits", 8) == 8
        and _quant_field(weights, "symmetric", True) is True
        and _quant_field(weights, "strategy") == "channel"
        and _quant_field(weights, "type", "int") == "int"
        and _quant_field(activations, "num_bits", 8) == 8
        and _quant_field(activations, "strategy") == "token"
        and _quant_field(activations, "dynamic") is True
        and _quant_field(activations, "type", "int") == "int"
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
) -> Iterable[tuple[str, str, int]]:
    """Yield owner/projection probes for every logical checkpoint expert."""
    count = getattr(hf_config, "n_routed_experts", 1)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        count = 1
    for index in _probe_layer_indexes(targets, hf_config):
        prefix = f"model.layers.{index}.mlp.experts"
        for expert_id in range(count):
            yield prefix, prefix, expert_id
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                yield prefix, f"{prefix}.{expert_id}.{proj_name}", expert_id


def _class_name_matches_target(class_name: str, target: str) -> bool:
    """Match a module class name against one target the way native does.

    ``find_matched_target`` compares the class name with ``re:`` targets using
    ``re.match`` and with plain targets by case-insensitive containment, so
    ``RoutedExperts`` and ``re:^RoutedExperts$`` select the same module.
    """
    from vllm.model_executor.layers.quantization.utils.config_utils import (
        is_equal_or_regex_match,
    )

    return is_equal_or_regex_match(class_name, target, check_contains=True)


def _probe_matcher(targets: list[str]):
    """Return ``matches(probe)`` for path targets, as native matches layer names.

    A plain target matches by equality and a ``re:`` target by ``re.match``.
    Plain targets go in a set, so a config that lists every expert projection
    by name costs one lookup per probe instead of one comparison per target.
    """
    from vllm.model_executor.layers.quantization.utils.config_utils import (
        is_equal_or_regex_match,
    )

    exact = {target for target in targets if not target.startswith("re:")}
    patterns = [target for target in targets if target.startswith("re:")]

    def matches(probe: str) -> bool:
        return probe in exact or any(
            is_equal_or_regex_match(probe, pattern) for pattern in patterns
        )

    return matches


def _targets_cover_routed_experts(
    targets: list[str],
    ignore: Any = None,
    hf_config: Any = None,
) -> bool:
    """Return whether an INT4 target selects a routed expert that is not ignored.

    Early dense layers may be ignored or absent from the INT4 regex. One later
    expert layer is enough. Shared experts and attention projections do not
    match these probes. A class-name target covers every routed expert that
    ``ignore`` leaves in place.
    """
    # A class-name target covers every expert that ``ignore`` leaves in place,
    # whatever the probe is, so it is decided once outside the probe loop.
    covers_by_class = any(
        _class_name_matches_target(_ROUTED_EXPERTS_CLASS, target)
        for target in targets
    )
    matches = _probe_matcher(targets)
    # The ignore rules depend on the expert, not on which of its four probes
    # matched, so each expert is checked once.
    ignored: dict[tuple[str, int], bool] = {}
    for module_prefix, probe, expert_id in _routed_expert_probes(targets, hf_config):
        # Target matching is cheap and usually fails for non-expert targets,
        # while the ignore check runs several regexes. Test targets first so
        # only matching probes pay for the ignore check.
        if not (covers_by_class or matches(probe)):
            continue
        key = (module_prefix, expert_id)
        if key not in ignored:
            ignored[key] = _ignore_covers_expert(
                None, module_prefix, ignore, {}, expert_id
            )
        if not ignored[key]:
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
        # Parse each entry the way native does before comparing its fields.
        weights = _parse_quant_args(group.get("weights") or {})
        activations = _parse_quant_args(group.get("input_activations") or {})
        group_format = group.get("format") or hf_quant_cfg.get("format")
        if _is_channel_int4_w4a8_scheme(weights, activations, group_format):
            int4_targets.extend(_group_targets(group))
        if _is_channel_int8_scheme(weights, activations, group_format):
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
    """Match the layer's own class name against class-name targets, as native does.

    Targets are tried in order with the rule shared with detection, so a literal
    name, a substring and a ``re:`` pattern all resolve alike. Only the concrete
    class is compared: native does not look at its ancestors either.
    """
    if layer is None:
        return None
    class_name = type(layer).__name__
    for target in targets:
        if isinstance(target, str) and _class_name_matches_target(class_name, target):
            return target
    return None


def _match_target_by_names(
    names: list[str], targets: Any, fused_mapping: Any
) -> str | None:
    """Preserve target order across runtime/checkpoint aliases of one name."""
    matches = {_match_target_by_name(name, targets, fused_mapping) for name in names}
    return next(
        (target for target in targets if target is not None and target in matches),
        None,
    )


def _projection_names(
    layer: Any, prefix: str, expert_id: int = 0
) -> tuple[str, str, str]:
    return tuple(
        f"{prefix}.{expert_id}.{proj_name}" for proj_name in _expert_proj_names(layer)
    )


def _expert_proj_names(layer: Any) -> tuple[str, str, str]:
    return (
        getattr(layer, "ckpt_gate_proj_name", "gate_proj"),
        getattr(layer, "ckpt_up_proj_name", "up_proj"),
        getattr(layer, "ckpt_down_proj_name", "down_proj"),
    )


def _ignore_covers_expert(
    layer: Any, prefix: str, ignore: Any, fused_mapping: Any, expert_id: int = 0
) -> bool:
    """Apply owner/projection exclusions to runtime and checkpoint paths."""
    prefixes = [prefix]
    if _MTP_BLOCK_MARKER in prefix:
        prefixes.append(prefix.replace(_MTP_BLOCK_MARKER, "."))
    return any(
        _ignore_covers_prefix(candidate, ignore, fused_mapping)
        or any(
            _ignore_covers_prefix(proj_name, ignore, fused_mapping)
            for proj_name in _projection_names(layer, candidate, expert_id)
        )
        for candidate in prefixes
    )


def routed_expert_owner_is_ignored(config: Any, prefix: str) -> bool:
    """Return whether an ignore rule covers the whole fused expert owner."""
    ignore = getattr(config, "ignore", None) or []
    fused_mapping = getattr(config, "packed_modules_mapping", None) or {}
    prefixes = [prefix]
    unwrapped = prefix.replace(_MTP_BLOCK_MARKER, ".")
    if unwrapped != prefix:
        prefixes.append(unwrapped)
    return any(
        _ignore_covers_prefix(candidate, ignore, fused_mapping)
        for candidate in prefixes
    )


def _expert_projection_schemes(
    config: Any, layer: Any, prefix: str, expert_id: int
) -> list[Any]:
    """Resolve one expert's projections in native target order.

    Fused-owner targets are compatibility aliases only when none of this
    expert's projection names match. Generic ``Linear`` class targets stay
    on the parent method.
    """
    scheme_map = getattr(config, "target_scheme_map", None) or {}
    fused_mapping = getattr(config, "packed_modules_mapping", None) or {}
    prefixes = [prefix]
    unwrapped = prefix.replace(_MTP_BLOCK_MARKER, ".")
    if unwrapped != prefix:
        prefixes.append(unwrapped)

    # Native MoE dispatch resolves each checkpoint projection in target order.
    # An owner regex may also match these names, but must not bypass a target
    # that appears earlier for a projection.
    proj_targets = [
        _match_target_by_names(
            [f"{candidate}.{expert_id}.{proj_name}" for candidate in prefixes],
            scheme_map,
            fused_mapping,
        )
        for proj_name in _expert_proj_names(layer)
    ]
    if all(target is None for target in proj_targets):
        # Exact/anchored fused-owner targets are an HCU compatibility alias.
        # Only use them when no projection name has selected a scheme.
        matched = _match_target_by_names(prefixes, scheme_map, fused_mapping)
        if matched is not None:
            return [scheme_map.get(matched)] * 3

    # Parent dispatch can add RoutedExperts -> Linear to this shared config.
    # Match projection names first and use the class only for missing names,
    # so the class match is only computed when some projection needs it.
    class_target = (
        _match_target_by_class(layer, scheme_map) if None in proj_targets else None
    )
    return [
        scheme_map.get(target if target is not None else class_target)
        for target in proj_targets
    ]


def _logical_expert_count(layer: Any) -> int | None:
    """Use logical IDs, excluding EP locality and redundant EPLB slots."""
    moe = getattr(layer, "moe_config", None)
    for count in (
        getattr(moe, "num_logical_experts", None),
        getattr(layer, "num_logical_experts", None),
        getattr(layer, "global_num_experts", None),
        getattr(moe, "num_experts", None),
    ):
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            return count
    return None


def _uniform_glm_channel_int4_experts(
    config: Any, layer: Any, prefix: str, count: int
) -> bool:
    """Reject heterogeneous checkpoint experts before allocating fused weights."""
    ignore = getattr(config, "ignore", None) or []
    fused_mapping = getattr(config, "packed_modules_mapping", None) or {}
    prefixes = [prefix]
    unwrapped = prefix.replace(_MTP_BLOCK_MARKER, ".")
    if unwrapped != prefix:
        prefixes.append(unwrapped)
    if routed_expert_owner_is_ignored(config, prefix):
        return False
    first_int4 = first_other = None
    for expert_id in range(count):
        schemes = _expert_projection_schemes(config, layer, prefix, expert_id)
        for projection, scheme in zip(_expert_proj_names(layer), schemes):
            name = f"{prefix}.{expert_id}.{projection}"
            ignored = any(
                _ignore_covers_prefix(
                    f"{candidate}.{expert_id}.{projection}", ignore, fused_mapping
                )
                for candidate in prefixes
            )
            if not ignored and _scheme_is_channel_int4(scheme, config):
                first_int4 = first_int4 or name
            else:
                first_other = first_other or name
            if first_int4 is not None and first_other is not None:
                raise NotImplementedError(
                    f"GLM channel INT4 fused MoE at {prefix!r} requires channel "
                    "INT4 W4A8 for all logical experts and projections; "
                    f"{first_int4!r} selects it but {first_other!r} is ignored "
                    "or selects another scheme. Heterogeneous expert "
                    "quantization is not supported."
                )
    return first_int4 is not None


def routed_expert_uses_glm_channel_int4(
    config: Any,
    layer: Any,
    prefix: str,
    *,
    num_hidden_layers: int | None = None,
) -> bool:
    """Return whether this routed-expert prefix should load channel INT4.

    All logical experts and projections must select channel INT4 without
    being ignored. Mixing it with another scheme/ignore raises before fused
    weights are allocated. Different target strings with the same scheme
    still match. MTP is retried only by removing ``.mtp_block.``;
    it never borrows another layer's scheme. ``num_hidden_layers`` remains
    accepted for existing callers but does not affect target matching.
    """
    count = _logical_expert_count(layer)
    if count is not None:
        return _uniform_glm_channel_int4_experts(config, layer, prefix, count)
    # Name-only probes have no runtime MoE config. Actual RoutedExperts carries
    # the logical count before get_quant_method and always takes the check above.
    ignore = getattr(config, "ignore", None) or []
    fused_mapping = getattr(config, "packed_modules_mapping", None) or {}
    if _ignore_covers_expert(layer, prefix, ignore, fused_mapping):
        return False
    return all(
        _scheme_is_channel_int4(scheme, config)
        for scheme in _expert_projection_schemes(config, layer, prefix, 0)
    )


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
