# SPDX-License-Identifier: Apache-2.0

"""
patch for vllm.distributed.device_communicators.pynccl
"""

PATCHES = [
(
"""
    def reduce_scatter(
""",
"""
    def all_to_all_single(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream=None,
    ):
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        assert output_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the output tensor is on {output_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        count = input_tensor.numel() // self.world_size
        self.nccl.ncclAllToAll(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            count,
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def reduce_scatter(
""",
),
]