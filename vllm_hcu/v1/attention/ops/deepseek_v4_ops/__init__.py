# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from .cache_utils import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.v1.attention.ops.deepseek_v4_ops import (
    MXFP4_BLOCK_SIZE,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
)

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "combine_topk_swa_indices",
    "compute_global_topk_indices_and_lens",
    "dequantize_and_gather_k_cache",
    "fused_indexer_q_rope_quant",
    "fused_inv_rope_fp8_quant",
    "fused_q_kv_rmsnorm",
    "quantize_and_insert_k_cache",
]
