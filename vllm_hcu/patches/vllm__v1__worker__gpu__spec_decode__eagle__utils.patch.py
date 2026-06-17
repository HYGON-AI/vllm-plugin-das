# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.v1.worker.gpu.spec_decode.eagle.utils
"""

PATCHES = [
(
"""
    return eagle_model
""",
"""
    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = getattr(target_language_model, "model", None)
    draft_inner = getattr(eagle_model, "model", None)

    # MTP shares topk_indices_buffer with the target model. Update every
    # draft module that holds a buffer reference so per-layer indexers and
    # sparse-attention backends all point to the target's buffer.
    if (
        target_inner is not None
        and draft_inner is not None
        and hasattr(target_inner, "topk_indices_buffer")
    ):
        target_buffer = target_inner.topk_indices_buffer
        if target_buffer is not None:
            for _, module in draft_inner.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer

    return eagle_model
"""
),
]
