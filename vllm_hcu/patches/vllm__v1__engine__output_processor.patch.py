# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.v1.engine.output_processor
Mooncake TTFT trace: d_first_token event
"""

PATCHES = [
    (
        '''import torch

from vllm.lora.request import LoRARequest
''',
        '''import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    log_ttft_event,
)
from vllm.lora.request import LoRARequest
''',
    ),
    (
        '''            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                req_state.is_prefilling = False
''',
        '''            if req_state.is_prefilling:
                log_ttft_event(
                    "d_first_token",
                    req_id=engine_core_output.request_id,
                    kv_params=engine_core_output.kv_transfer_params,
                )
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                req_state.is_prefilling = False
''',
    ),
]
