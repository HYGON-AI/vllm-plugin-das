# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# vllm.utils.import_utils: also detect HCU ``deepgemm`` package.

PATCHES = [
(
"""
    return _has_module("deep_gemm") or _has_module("vllm.third_party.deep_gemm")
""",
"""
    return _has_module("deepgemm")  or _has_module("deep_gemm") or _has_module("vllm.third_party.deep_gemm")
""",
),
]
