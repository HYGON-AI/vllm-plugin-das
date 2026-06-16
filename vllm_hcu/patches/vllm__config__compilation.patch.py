# SPDX-License-Identifier: Apache-2.0

"""Patch for vllm.config.compilation used by HCU custom SP."""

PATCHES = [
    (
        """        is_profiling: bool = False,
    ) -> CUDAGraphMode:""",
        """        is_profiling: bool = False,
        enable_custom_sp: bool = False,
    ) -> CUDAGraphMode:""",
    ),
    (
        """            self.adjust_cudagraph_sizes_for_spec_decode(
                uniform_decode_query_len,
                tensor_parallel_size,
            )""",
        """            self.adjust_cudagraph_sizes_for_spec_decode(
                uniform_decode_query_len,
                tensor_parallel_size,
                enable_custom_sp,
            )""",
    ),
    (
        """    def adjust_cudagraph_sizes_for_spec_decode(
        self, uniform_decode_query_len: int, tensor_parallel_size: int
    ):
        multiple_of = uniform_decode_query_len
        if tensor_parallel_size > 1 and self.pass_config.enable_sp:""",
        """    def adjust_cudagraph_sizes_for_spec_decode(
        self,
        uniform_decode_query_len: int,
        tensor_parallel_size: int,
        enable_custom_sp: bool = False,
    ):
        multiple_of = uniform_decode_query_len
        if tensor_parallel_size > 1 and (
            self.pass_config.enable_sp or enable_custom_sp
        ):""",
    ),
]
