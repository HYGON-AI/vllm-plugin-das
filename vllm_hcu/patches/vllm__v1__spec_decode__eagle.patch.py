# SPDX-License-Identifier: Apache-2.0

"""
vllm.v1.models.spec_decode.eagle  __init__
"""

PATCHES = [
(
"""
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)
""",
"""
        # if current_platform.is_rocm():
        #     from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        #         ROCMAiterMLASparseMetadata,
        #     )
        #     from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

        #     rocm_types = [
        #         TritonAttentionMetadata,
        #         RocmAttentionMetadata,
        #         ROCMAiterMLASparseMetadata,
        #     ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            # if find_spec(
            #     AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            # ):
            #     from vllm.v1.attention.backends.rocm_aiter_fa import (
            #         AiterFlashAttentionMetadata,
            #     )

            #     rocm_types.append(AiterFlashAttentionMetadata)

            # # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            # from vllm.model_executor.layers.attention.mla_attention import (
            #     MLACommonMetadata,
            # )

            # rocm_types.append(MLACommonMetadata)

            # # FlexAttention backend support
            # from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            # rocm_types.append(FlexAttentionMetadata)

            # self.allowed_attn_types = tuple(rocm_types)
""",
),

]
