"""
Patch for vllm/utils/deep_gemm.py
"""

PATCHES = [
(
"""
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
""",
"""
    if _tf32_hc_prenorm_gemm_impl is None:
        # return _missing()
        out.zero_()
        sqrsum.zero_()
        out[0].copy_(torch.matmul(x.to(torch.float32), fn.t().to(torch.float32)))
        sqrsum[0].copy_(x.to(torch.float32).square().sum(dim=-1))
        return out
""",
),
]