# SPDX-License-Identifier: Apache-2.0
"""CPU metadata and conservative shape policies for Kimi prefill kernels."""
import os
from contextvars import ContextVar

LENGTHS_KEY = "_vllm_hcu_kimi_prefill_sequence_lengths"
PREFILL_LENGTHS = ContextVar("hcu_kimi_prefill_lengths", default=None)


def enabled(name):
    return all(os.environ.get(key, "1").lower() in ("1", "true")
               for key in ("VLLM_HCU_USE_CUSTOM_OPS", name))


def cpu_lengths(metadata):
    if not enabled("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS"):
        return None
    return (getattr(metadata, "nums_dict", None) or {}).get(LENGTHS_KEY)


def kernel_policy(name, lengths, *, tokens=None, sequences=None, capturing=False):
    """Auto uses CPU lengths only, avoiding a device synchronization.

    Conv1D auto accepts balanced prefill >=128 tokens in eager and >=8192
    tokens during graph capture. FlashKDA excludes strongly skewed
    multi-sequence batches. 1/hcu/flash explicitly force the external kernel;
    0/triton retain the upstream path.
    """
    backend = os.environ.get(name, "auto").lower()
    if backend in ("1", "hcu", "flash"):
        return True
    if backend in ("0", "triton"):
        return False
    if backend != "auto":
        raise ValueError(f"{name} must be auto, 1/hcu/flash, or 0/triton")
    if not lengths or min(lengths) <= 0:
        return False
    if sequences is not None and len(lengths) != sequences:
        return False
    if tokens is not None and sum(lengths) != tokens:
        return False
    if name == "VLLM_KIMI_CONV1D_BACKEND":
        balanced = max(lengths) <= 4 * (sum(lengths) / len(lengths))
        return balanced and sum(lengths) >= (8192 if capturing else 128)
    return max(lengths) <= 4 * (sum(lengths) / len(lengths))
