# SPDX-License-Identifier: Apache-2.0

"""
patch for vllm.distributed.device_communicators.base_device_communicator
"""

PATCHES = [
(
"""
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
""",
"""
    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        torch.distributed.all_to_all_single(
            output, input, group=self.device_group
        )
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
""",
),
]