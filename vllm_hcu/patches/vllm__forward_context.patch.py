# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# set_forward_context: skip dp_metadata when deepep low-latency all2all.
# 注意：import 必须顶格，与 forward_context.py 一致，不能写成 8 个空格开头。

PATCHES = [
    (
        """import vllm.envs as envs
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig""",
        """import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig""",
    ),
    (
        """    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):""",
        """    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
        and henvs.VLLM_HCU_ALL2ALL_BACKEND != "deepep_low_latency"
    ):""",
    ),
]
