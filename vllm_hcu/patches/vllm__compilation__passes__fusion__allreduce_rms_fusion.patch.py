# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.compilation.passes.fusion.allreduce_rms_fusion
Import HCU CustomAllreduce to fix isinstance check on HCU platform.
"""

PATCHES = [
(
"""
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
""",
"""
try:
    from vllm_hcu.distributed.device_communicators.custom_all_reduce import CustomAllreduce
except ImportError:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
""",
),
]
