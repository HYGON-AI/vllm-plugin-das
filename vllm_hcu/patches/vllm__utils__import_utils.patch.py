# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# vllm.utils.import_utils: also detect HCU ``deepgemm`` package.

PATCHES = [
    (
        '    return _has_module("deep_gemm")',
        '    return _has_module("deepgemm")',
    ),
]
