# SPDX-License-Identifier: Apache-2.0

"""
patch for vllm.distributed.device_communicators.all2all: EP INT8
"""

PATCHES = [
(
"""
import vllm.envs as envs
""",
"""
import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs
""",
),
(
"""
        self.num_sms = 20
""",
"""
        self.num_sms = 30
""",
),
(
"""
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1
""",
"""
        # Defaults for internode and intranode are taken from DeepEP tests.
        #num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_nvl_bytes = int(2e9/2)#1024 * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            # num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            # num_qps_per_rank = self.num_sms // 2
            num_rdma_bytes = int(1e9/2) #1024 * 1024 * 1024
            num_qps_per_rank = 30 #self.num_sms // 2
            self.num_sms = 30
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1
            self.num_sms = 60
""",
),
(
"""
        # Right now the buffers are sized for only what the kernels were
        # created with. So we can only reduce the number of SMS used
        # but not increase it.
        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)
""",
"""
        # Right now the buffers are sized for only what the kernels were
        # created with. So we can only reduce the number of SMS used
        # but not increase it.
        if henvs.VLLM_HCU_DEEPEP_NUM_SMS is not None:
            num_sms = henvs.VLLM_HCU_DEEPEP_NUM_SMS
        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)
""",
),
]
