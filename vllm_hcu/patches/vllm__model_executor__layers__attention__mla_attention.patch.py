# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.attention.mla_attention process_weights_after_loading
"""
import re

PATCHES = [
    (
        "import vllm.envs as envs",
        "import vllm.envs as envs\nimport vllm_hcu.platforms.envs as henvs",
    ),

    (
        """        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T
""",
        """        if henvs.VLLM_USE_NN:
            kv_b_proj_weight = get_and_maybe_dequant_weights(
                self.kv_b_proj, out_dtype=act_dtype
        )
        else:
            kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T
""",
    ),
]