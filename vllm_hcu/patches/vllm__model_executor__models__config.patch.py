# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.config verify_and_update_config DeepseekV32ForCausalLM
"""

PATCHES = [
(
"""
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec
""",
"""
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
from typing import TYPE_CHECKING
""",
"""
from typing import TYPE_CHECKING
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
        is_v32 = hasattr(hf_config, "index_topk")
""",
"""
        is_v32 = hasattr(hf_config, "index_topk") and not henvs.VLLM_HCU_DISABLE_DSA
""",
),

]
