# SPDX-License-Identifier: Apache-2.0
"""Reuse CPU scheduler metadata for Kimi's existing HCU convolution route."""
import os

LENGTHS_KEY = "_vllm_hcu_kimi_prefill_sequence_lengths"


def enabled():
    return all(os.environ.get(name, default).lower() in ("1", "true") for name, default in (
        ("VLLM_HCU_USE_CUSTOM_OPS", "true"),
        ("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS", "false"),
    ))


def prefill_sequence_lengths(metadata, query_start_loc):
    """Only reuse data owned by this metadata instance, never device addresses."""
    values = (metadata.nums_dict or {}).get(LENGTHS_KEY)
    if enabled() and values is not None:
        return values
    return query_start_loc.diff().tolist()
