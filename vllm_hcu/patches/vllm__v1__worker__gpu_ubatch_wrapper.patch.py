# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.v1.worker.gpu_ubatch_wrapper.

Disable compute-side DeepGEMM SMS control until the local DeepGEMM Python
package exports set_num_sms/get_num_sms. Communication-side SMS control remains
unchanged.
"""

PATCHES = [
(
"""
from vllm.utils.deep_gemm import set_num_sms as deep_gemm_set_num_sms
from vllm.utils.import_utils import has_deep_gemm
""",
"""
""",
),
(
"""
        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        if has_deep_gemm() and comm_sms > 0:
            set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)
""",
"""
        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        # TODO(yql): Re-enable compute-side SM control after the local
        # DeepGEMM Python package exports set_num_sms/get_num_sms.
        # if has_deep_gemm() and comm_sms > 0:
        #     set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)
""",
),
]
