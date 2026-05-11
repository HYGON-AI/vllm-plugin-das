# SPDX-License-Identifier: Apache-2.0

"""
vllm.v1.attention.backends.utils
"""

PATCHES = [
(
"""
    nums_dict = {}  # type: ignore
""",
"""
    nums_dict = {}  # type: ignore
    nums_dict["seqlens"] = seqlens.tolist()
""",
),
]
