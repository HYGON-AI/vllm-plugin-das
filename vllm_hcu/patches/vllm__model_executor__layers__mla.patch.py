# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.mla
"""

PATCHES = [
(
"""
from vllm.model_executor.layers.quantization import QuantizationConfig
""",
"""
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.distributed import (
    tensor_model_parallel_all_gather,
)
"""
),

(
"""
        prefix: str = "",
    ) -> None:
""",
"""
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
"""
),

(
"""
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        if self.indexer is not None:
""",
"""
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        if self.indexer is not None:
"""
),

(
"""
        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )
""",
"""
        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
"""
),

(
"""
        if llama_4_scaling is not None:
            q *= llama_4_scaling
""",
"""
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        enable_lightly_cp = get_forward_context().enable_lightly_cp
        enable_lightly_cplb = get_forward_context().enable_lightly_cplb

        if enable_lightly_cp:
            kv_c_normed = tensor_model_parallel_all_gather(
                kv_c_normed.contiguous(), 0
            )
            k_pe = tensor_model_parallel_all_gather(
                k_pe.contiguous(), 0
            )

            gather_indexes_tensor = get_forward_context().gather_indexes_tensor
            if enable_lightly_cplb and gather_indexes_tensor is not None:
                # Reorder kv after pcp allgather.
                kv_c_normed = torch.index_select(kv_c_normed, 0, gather_indexes_tensor)
                k_pe = torch.index_select(k_pe, 0, gather_indexes_tensor)
"""
),
]
