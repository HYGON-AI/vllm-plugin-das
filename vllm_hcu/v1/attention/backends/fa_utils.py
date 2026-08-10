# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
from typing import Any

from flash_attn import (
    flash_attn_varlen_func,
    hg_flash_attn_varlen_func,
    varlen_fwd_unified,
    vllm_flash_attn_varlen_func,
)
from torch import Tensor

import vllm_hcu.hcu_ops as hcu_ops


# Hcu doesn't use scheduler metadata (FA3 feature), provide stub
def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
    return None


def reshape_and_cache_flash(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None:
    """Write FlashAttention KV pages using the active physical layout.

    AITER's writer assumes token-major NHD storage inside each page.  The
    target vLLM Mooncake contract selects HND for heterogeneous-TP transfer,
    where the logical ``[B, N, H, D]`` view has head-major physical strides.
    vLLM's Triton writer consumes the real tensor strides and therefore
    handles both the HND slot/head order and a padded physical block stride.
    """
    # Logical cache views remain [block, token, head, dim] in both layouts.
    # HND is distinguishable by a token stride smaller than the head stride.
    # If either dimension is one, selecting NHD is also address-equivalent.
    #
    # AITER's flash cache writer corrupts FP8 E4M3 pages on HCU (including
    # NaN bit patterns).  vLLM's Triton writer explicitly casts through the
    # platform FP8 dtype and is stride-aware, so use it for FP8 caches as well
    # as for HND storage. Keep other quantization formats on their existing
    # route until they have an explicit HCU cache contract.
    if kv_cache_dtype.startswith("fp8") or (
        key_cache.ndim == 4 and key_cache.stride(1) < key_cache.stride(2)
    ):
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )

        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    from aiter.ops.cache import (
        reshape_and_cache_flash as aiter_reshape_and_cache_flash,
    )

    aiter_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None:
    return 2


def flash_attn_supports_fp8() -> bool:
    return True


def flash_attn_supports_sinks() -> bool:
    return True


def flash_attn_supports_mla():
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True


def flash_attn_supports_quant_query_input() -> bool:
    return True


def is_fa_version_supported(fa_version: int) -> bool:
    return False
