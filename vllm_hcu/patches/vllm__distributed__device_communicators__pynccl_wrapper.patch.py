# SPDX-License-Identifier: Apache-2.0

"""
patch for vllm.distributed.device_communicators.pynccl_wrapper
"""

PATCHES = [
(
"""
    def ncclSend(
""",
"""
    def ncclAllToAll(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclAllToAll"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def ncclSend(
""",
),

(
"""
        # ncclResult_t  ncclSend(
""",
"""
        # ncclResult_t  ncclAllToAll(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "ncclAllToAll",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclSend(
""",
),
]