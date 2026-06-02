# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.attention.__init__
"""

PATCHES = [
###################### support fuse qkv_split+rmsnorm+rope+kvstore ######################
(
'''
from vllm.model_executor.layers.attention.attention import Attention
''',
'''
from vllm.model_executor.layers.attention.attention import Attention, FusedQkvSplitRmsNormRopeAttention
'''
),

(
'''
    "Attention",
''',
'''
    "Attention",
    "FusedQkvSplitRmsNormRopeAttention",
'''
),
###################### support fuse qkv_split+rmsnorm+rope+kvstore ######################
]
