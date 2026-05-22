"""
Patch for vllm/model_executor/layers/deepseek_v4_attention.py
"""

PATCHES = [
(
"""
import torch
""",
"""
import math

import torch
""",
),

(
"""
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
""",
"""
    try:
        fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
    except RuntimeError as e:
        if HAS_TRITON and (
            "not available" in str(e).lower() or "outdated" in str(e).lower()
        ):
            from vllm_hcu.ops.fp8_einsum_fallback import deepseek_v4_fp8_einsum_fallback_triton
            deepseek_v4_fp8_einsum_fallback_triton(a, a_scale, b, b_scale, out, equation)
            return
        raise
""",
),

(
"""
            out=output.unsqueeze(1),
        )
""",
"""
            # out=output.unsqueeze(1),
        )
        output.copy_(out.squeeze(1).to(output.dtype))
""",
),

(
"""
                    out=output[query_start:query_end],
                )
""",
"""
                    out=output[query_start:query_end],
                )
                output[query_start:query_end].copy_(output_chunk.to(output.dtype))
""",
),
]