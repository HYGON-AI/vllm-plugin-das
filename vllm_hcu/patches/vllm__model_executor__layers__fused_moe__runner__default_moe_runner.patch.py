# SPDX-License-Identifier: Apache-2.0
#
# default_moe_runner.use_dp_chunking: do not enable Python DP chunking for
# DeepEP low-latency kernels (aligns with vLLM 0.15.x fused_moe.layer behavior).
# DeepEP still uses moe_config.max_num_tokens / VLLM_MOE_DP_CHUNK_SIZE for
# buffers and get_handle; only the forward_impl_chunked + dp_metadata path is
# skipped for deepep_ll.

PATCHES = [
    (
        """    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_config.moe_parallel_config.use_deepep_ll_kernels
            or self.moe_config.moe_parallel_config.use_mori_kernels
            or self.moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels
            or self.moe_config.moe_parallel_config.use_nixl_ep_kernels
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK""",
        """    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_config.moe_parallel_config.use_mori_kernels
            or self.moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels
            or self.moe_config.moe_parallel_config.use_nixl_ep_kernels
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK""",
    ),
]
