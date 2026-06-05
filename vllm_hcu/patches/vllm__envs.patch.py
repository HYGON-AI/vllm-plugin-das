# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.envs
"""

PATCHES = [
(
'''
        if env.startswith("VLLM_") and env not in environment_variables:
            if hard_fail:
                raise ValueError(f"Unknown vLLM environment variable detected: {env}")
            else:
                logger.warning("Unknown vLLM environment variable detected: %s", env)
''',
'''
        if env.startswith("VLLM_") and env not in environment_variables:
            # Allow HCU plugin-specific environment variables without needing
            # per-variable registration in vLLM core env registry.
            if env.startswith("VLLM_HCU_"):
                continue
            if hard_fail:
                raise ValueError(f"Unknown vLLM environment variable detected: {env}")
            else:
                logger.warning("Unknown vLLM environment variable detected: %s", env)
''',
),

(
'''
    "VLLM_ROCM_USE_AITER_MOE": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MOE", "True").lower() in ("true", "1")
    ),
''',
'''
    "VLLM_ROCM_USE_AITER_MOE": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MOE", "False").lower() in ("true", "1")
    ),
''',
),
]
