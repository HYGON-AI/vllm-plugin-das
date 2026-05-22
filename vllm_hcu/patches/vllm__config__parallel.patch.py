# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ParallelConfig.__post_init__: VLLM_HCU_ALL2ALL_BACKEND from env when set.
# 第二段 old/new 必须包含 # Continue 与 self.world_size 两行，否则补丁不完整。

PATCHES = [
    (
        """import vllm.envs as envs
from vllm.config.utils import config""",
        """import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs
from vllm.config.utils import config""",
    ),
    (
        """    def __post_init__(self) -> None:
        # Continue with the rest of the initialization
        self.world_size = (""",
        """    def __post_init__(self) -> None:
        # Set all2all_backend from env var when explicitly set, with deprecation warning
        if henvs.is_set("VLLM_HCU_ALL2ALL_BACKEND"):
            logger.warning_once(
                "VLLM_HCU_ALL2ALL_BACKEND environment variable is deprecated and "
                "will be removed in v0.15.0. Please use the "
                "--all2all-backend command-line argument instead."
            )
            self.all2all_backend = henvs.VLLM_HCU_ALL2ALL_BACKEND
        # Continue with the rest of the initialization
        self.world_size = (""",
    ),

################ lightly cp###########################
(
'''
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """
''',
'''
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    enable_lightly_cp: bool = False
    """Use lightly context parallel."""

    enable_lightly_cplb: bool = False
    """Use lightly context parallel load balancing."""

'''
),
################ lightly cp###########################
]