# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 environment getters registered by the platform callback."""

import os
import warnings


def is_dsa_disabled() -> bool:
    """Read the shared DSA switch during config resolution and model import."""
    return os.getenv("VLLM_STEP5_DISABLE_DSA", "0").strip().lower() in ("1", "true")


# These names were exported by the old adapter but have no runtime consumers.
# Do not translate them into different optimizations with different semantics.
IGNORED_ENVIRONMENT_VARIABLES = {
    "VLLM_STEP5_DSA_INDEX_TP_SIZE": "Indexer sharding follows the model TP layout.",
    "VLLM_STEP5_ENABLE_QKVG_PROJ": (
        "Use VLLM_HCU_STEP5_FUSE_QKVG to control QKV/gate projection fusion."
    ),
    "VLLM_STEP5_FUSE_INDEXER_NORM": "Indexer normalization is selected internally.",
    "VLLM_STEP5_O_PROJ_REDUCE_SCATTER": (
        "The output projection follows the configured parallel layout."
    ),
}


def warn_ignored_step5_envs() -> None:
    """Make obsolete launch-script settings visible only for Step5 models."""
    for name, detail in IGNORED_ENVIRONMENT_VARIABLES.items():
        if name in os.environ:
            warnings.warn(
                f"{name} is obsolete and has no effect. {detail}",
                UserWarning,
                stacklevel=2,
            )


ENVIRONMENT_VARIABLES = {
    "VLLM_STEP_CC_LEVEL": lambda: int(os.getenv("VLLM_STEP_CC_LEVEL", "0")),
    "VLLM_STEP5_DISABLE_DSA": is_dsa_disabled,
    "VLLM_STEP5_TOOL_PARSER_MAX_HEADER_CHARS": lambda: int(
        os.getenv("VLLM_STEP5_TOOL_PARSER_MAX_HEADER_CHARS", "512")
    ),
    "VLLM_STEP5_DSA_FORCE_STABLE_TOPK": lambda: (
        os.getenv("VLLM_STEP5_DSA_FORCE_STABLE_TOPK", "true").lower() in ("1", "true")
    ),
    "VLLM_STEP5_SPARSE": lambda: (
        os.getenv("VLLM_STEP5_SPARSE", "false").lower() in ("1", "true")
    ),
    "VLLM_STEP5_SPARSE_ATTENTION_IMPL": lambda: os.getenv(
        "VLLM_STEP5_SPARSE_ATTENTION_IMPL", "sparse_gqa"
    ),
    "VLLM_STEP5_SPARSE_DECODE_SPLIT_MAX": lambda: int(
        os.getenv("VLLM_STEP5_SPARSE_DECODE_SPLIT_MAX", "16")
    ),
    "VLLM_STEP5_SPARSE_INDEXER_ROPE_DIM": lambda: int(
        os.getenv("VLLM_STEP5_SPARSE_INDEXER_ROPE_DIM", "32")
    ),
    "VLLM_STEP5_SPARSE_PROXY_DIM": lambda: int(
        os.getenv("VLLM_STEP5_SPARSE_PROXY_DIM", "256")
    ),
    "VLLM_STEP5_SPARSE_REGION_BLOCK_SIZE": lambda: int(
        os.getenv("VLLM_STEP5_SPARSE_REGION_BLOCK_SIZE", "8")
    ),
    "VLLM_STEP5_SPARSE_TOPK": lambda: int(os.getenv("VLLM_STEP5_SPARSE_TOPK", "512")),
    "VLLM_STEP5_JIT_CACHE_DIR": lambda: os.getenv("VLLM_STEP5_JIT_CACHE_DIR", None),
    "VLLM_DSA_OCC_EVERY": lambda: int(os.getenv("VLLM_DSA_OCC_EVERY", "0")),
    "VLLM_DSA_TSL_REDZONE": lambda: (
        os.getenv("VLLM_DSA_TSL_REDZONE", "false").lower() in ("1", "true")
    ),
}
