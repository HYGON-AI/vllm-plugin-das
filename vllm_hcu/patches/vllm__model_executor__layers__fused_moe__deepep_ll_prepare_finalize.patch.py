# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize:
- DeepEP LL int8 dispatch + expert_num_tokens in _do_quant
- low_latency_dispatch: use quant_type / fp8_round_scale (HCU DeepEP API, v0.15 style)
- FP8: current_platform import + recognize DeepEP-prequantized fp8 tuples in _do_quant
"""

PATCHES = [
(
"""
from collections.abc import Callable

import deep_ep
""",
"""
from collections.abc import Callable
from typing import Optional

import deep_ep
""",
),
(
"""
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__()

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
""",
"""
        local_expert_global_ids: torch.Tensor | None = None,
        use_int8_dispatch: bool = False,
    ):
        super().__init__()

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.use_int8_dispatch = use_int8_dispatch
""",
),
(
"""
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
""",
"""
        quant_config: FusedMoEQuantConfig,
        expert_num_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
""",
),
(
"""
            x = dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)
""",
"""
            x = dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)

        elif self.use_int8_dispatch:
            x, x_scales = x
            return x, x_scales
""",
),
(
"""
            # TODO (varun): Optimization - Use a batched version of quant
            x = x.view((-1, hidden_dim))
""",
"""
            # TODO (varun): Optimization - Use a batched version of quant
            if expert_num_tokens is None:
                x = x.view((-1, hidden_dim))
""",
),
(
"""
            use_fp8=self.use_fp8_dispatch,
            round_scale=self.use_ue8m0_dispatch,
""",
"""
            use_fp8=self.use_fp8_dispatch or self.use_int8_dispatch,
            use_int8=self.use_int8_dispatch,
            round_scale=self.use_ue8m0_dispatch,
""",
),
(
"""
        expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)

        expert_tokens_meta = mk.ExpertTokensMetadata(
""",
"""
        expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config, expert_num_tokens)

        expert_tokens_meta = mk.ExpertTokensMetadata(
""",
),
(
"""
        # Dispatch
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.low_latency_dispatch(
            a1,
            dispatch_topk_ids,
            self.max_tokens_per_rank,
            num_experts,
            use_fp8=self.use_fp8_dispatch or self.use_int8_dispatch,
            use_int8=self.use_int8_dispatch,
            round_scale=self.use_ue8m0_dispatch,
            use_ue8m0=self.use_ue8m0_dispatch,
            **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
            **(
                dict(x_global_scale=qc_a1_gscale_or_scale)
                if qc_a1_gscale_or_scale is not None
                else dict()
            ),
            async_finish=False,
            return_recv_hook=True,
        )
""",
"""
        # Dispatch
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)

        quant_type = 0
        if self.use_int8_dispatch:
            quant_type = 1
        elif self.use_fp8_dispatch:
            quant_type = 2
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.low_latency_dispatch(
            a1,
            dispatch_topk_ids,
            self.max_tokens_per_rank,
            num_experts,
            quant_type = quant_type,
            fp8_round_scale=False,
            async_finish=False,
            return_recv_hook=True,
        )
""",
),
    # FP8 (apply after INT8 patches above on clean vllm018)
(
"""
logger = init_logger(__name__)
""",
"""
from vllm.platforms import current_platform
logger = init_logger(__name__)
""",
),
(
"""
            if block_k == DEEPEP_QUANT_BLOCK_SIZE:
""",
"""
            if block_k == DEEPEP_QUANT_BLOCK_SIZE or (isinstance(x, tuple) and x[0].dtype == current_platform.fp8_dtype()):
""",
),
]
