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
    ),

    (
"""
    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_to_all_single(output, input)
        else:
            torch.distributed.all_to_all_single(
                output, input, group=self.device_group
            )
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
""",
"""

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
""",
    )
]