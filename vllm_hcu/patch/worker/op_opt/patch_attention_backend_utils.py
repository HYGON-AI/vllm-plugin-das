# SPDX-License-Identifier: Apache-2.0
"""Adapt audited attention utility behavior required by HCU backends."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.backends.utils"
PATCH_ID = "worker.op_opt.attention.causal_conv_metadata"
TARGETS = (
    f"{TARGET_MODULE}.compute_causal_conv1d_metadata",
    f"{TARGET_MODULE}.split_decodes_and_prefills",
)
_MARKER = "_vllm_hcu_causal_conv_metadata_applied"
_WRAPPER = "_vllm_hcu_causal_conv_metadata_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    utils = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (utils, "compute_causal_conv1d_metadata", TARGETS[0], _WRAPPER),
        (utils, "split_decodes_and_prefills", TARGETS[1], _WRAPPER),
    )
    if already_applied(utils, _MARKER, wrapped):
        return False

    original = require_callable(
        utils, "compute_causal_conv1d_metadata", TARGETS[0]
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("query_start_loc_p_cpu",),
        keyword_only=("device",),
    )
    original_split = require_callable(
        utils, "split_decodes_and_prefills", TARGETS[1]
    )
    require_exact_signature(
        original_split,
        TARGETS[1],
        positional=(
            "common_attn_metadata",
            "decode_threshold",
            "require_uniform",
            "treat_short_extends_as_decodes",
        ),
        defaults={
            "decode_threshold": 1,
            "require_uniform": False,
            "treat_short_extends_as_decodes": True,
        },
    )

    @functools.wraps(original)
    def hcu_metadata(query_start_loc_p_cpu, *, device):
        result = original(query_start_loc_p_cpu, device=device)
        if (
            not isinstance(result, tuple)
            or len(result) != 3
            or not isinstance(result[0], dict)
        ):
            raise RuntimeError(
                "vLLM causal-conv metadata returned an incompatible value"
            )
        result[0]["seqlens"] = query_start_loc_p_cpu.diff().tolist()
        return result

    @functools.wraps(original_split)
    def hcu_split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        # vLLM 0.25 owns the flag and is_prefilling metadata.  Its uniform
        # fast path returns before consulting is_prefilling, so adapt only
        # that exact case and delegate every other classification.
        if not require_uniform or treat_short_extends_as_decodes:
            return original_split(
                common_attn_metadata,
                decode_threshold,
                require_uniform,
                treat_short_extends_as_decodes,
            )

        query_start_loc = common_attn_metadata.query_start_loc_cpu
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        if query_lens[0].item() > decode_threshold:
            return original_split(
                common_attn_metadata,
                decode_threshold,
                require_uniform,
                treat_short_extends_as_decodes,
            )
        is_uniform = ((query_lens == query_lens[0]) | (query_lens == 0)).all()
        if not bool(is_uniform.item()):
            return original_split(
                common_attn_metadata,
                decode_threshold,
                require_uniform,
                treat_short_extends_as_decodes,
            )

        is_prefilling = common_attn_metadata.is_prefilling
        if is_prefilling is None:
            raise AssertionError(
                "uniform short-extend classification requires is_prefilling"
            )
        is_prefill = is_prefilling.clone()
        is_prefill |= query_lens > decode_threshold
        is_prefill &= query_lens > 0
        if not bool(is_prefill.any().item()):
            return original_split(
                common_attn_metadata,
                decode_threshold,
                require_uniform,
                treat_short_extends_as_decodes,
            )

        first_prefill = is_prefill.int().argmax(dim=-1).item()
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        num_decode_tokens = query_start_loc[first_prefill].item()
        return (
            first_prefill,
            num_reqs - first_prefill,
            num_decode_tokens,
            num_tokens - num_decode_tokens,
        )

    for function in (hcu_metadata, hcu_split_decodes_and_prefills):
        setattr(function, _WRAPPER, True)
    setattr(
        utils,
        "_vllm_hcu_original_compute_causal_conv1d_metadata",
        original,
    )
    setattr(
        utils,
        "_vllm_hcu_original_split_decodes_and_prefills",
        original_split,
    )
    setattr(utils, "compute_causal_conv1d_metadata", hcu_metadata)
    setattr(utils, "split_decodes_and_prefills", hcu_split_decodes_and_prefills)
    setattr(utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
