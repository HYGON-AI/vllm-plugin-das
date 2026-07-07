# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm/model_executor/layers/activation.py.
"""

PATCHES = [
(
"""
    def __init__(self, swiglu_limit: float, *, compile_native: bool = True):
        super().__init__(compile_native=compile_native)
        self.swiglu_limit = float(swiglu_limit)
        if current_platform.is_rocm():
            self._forward_method = self.forward_native
        elif current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul_with_clamp
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
""",
"""
    def __init__(self, swiglu_limit: float, *, compile_native: bool = True):
        super().__init__(enforce_enable=True, compile_native=compile_native)
        self.swiglu_limit = float(swiglu_limit)
        if (
            current_platform.is_rocm()
            or current_platform.is_cuda_alike()
            or current_platform.is_xpu()
        ):
            self.op = torch.ops._C.silu_and_mul_with_clamp
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
""",
),
]
