# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.platforms.rocm to extend the GFX9 arch allowlist for HCU.
"""

PATCHES = [
(
"""
_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
""",
"""
_ON_GFX9 = any(arch in _GCN_ARCH  for arch in ["gfx90a", "gfx942", "gfx950", "gfx936", "gfx938"])
""",
),
(
"""
        "bitsandbytes",
""",
"""
        "bitsandbytes",
        "slimquant_w4a8",
""",
),
]
