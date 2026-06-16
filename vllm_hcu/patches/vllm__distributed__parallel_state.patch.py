# SPDX-License-Identifier: Apache-2.0

"""
patch for vllm.distributed.parallel_state
"""

PATCHES = [
(
"""
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
""",
"""
    def all_to_all_single(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_to_all_single(output, input)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
""",
),
]