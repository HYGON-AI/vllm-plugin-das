# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.gpt_oss
"""

PATCHES = [
(
"""
            g = rocm_unquantized_gemm(
                self, x[:, : self.hidden_size], self.router.weight, self.router.bias
            )
""",
"""
            g = torch.nn.functional.linear(
                x[:, : self.hidden_size], self.router.weight.t(), self.router.bias
            )
""",
),
]