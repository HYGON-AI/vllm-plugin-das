# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# coordinate_batch_across_dp: early-exit for deepep low-latency (v0.15 parity).

PATCHES = [
    (
        """    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return False, None, cudagraph_mode""",
        """    if (
        parallel_config.data_parallel_size == 1
        or parallel_config.all2all_backend == "deepep_low_latency"
    ):
        # Early exit.
        return False, None, cudagraph_mode""",
    ),
]
