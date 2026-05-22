# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm._aiter_ops is_aiter_found_and_supported.
"""

PATCHES = [
(
"""
        return on_mi3xx()
""",
"""
        from vllm_hcu.platforms.hcu import on_gfx93x

        return on_gfx93x() or on_mi3xx()
""",
),
]
