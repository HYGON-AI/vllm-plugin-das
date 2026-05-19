# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# coordinate_batch_across_dp: early-exit for deepep low-latency (v0.15 parity).

PATCHES = [
    (
        """from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import (""",
        """from vllm.logger import init_logger
import vllm_hcu.platforms.envs as henvs
from vllm.v1.worker.ubatch_utils import (""",
    ),
    (
        """    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return False, None, cudagraph_mode""",
        """    if (
        parallel_config.data_parallel_size == 1
        or henvs.VLLM_HCU_ALL2ALL_BACKEND == "deepep_low_latency"
    ):
        # Early exit.
        return False, None, cudagraph_mode""",
    ),
]
