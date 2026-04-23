# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.distributed.device_communicators custom_all_reduce
"""

PATCHES = [
    (
"""
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
""",
"""
        from vllm_hcu.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
""",
    )
]