# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared Qwen4Exp PLE pipeline-partition checks."""

from __future__ import annotations


def first_stage_owns_all_ple_layers(vllm_config: object) -> bool:
    """Return whether PP stage zero owns every configured 1-based PLE layer."""

    parallel_config = getattr(vllm_config, "parallel_config")
    pp_size = int(getattr(parallel_config, "pipeline_parallel_size"))
    if pp_size <= 1:
        return False

    text_config = getattr(getattr(vllm_config, "model_config"), "hf_text_config")
    ple_layer_ids = tuple(int(layer_id) for layer_id in text_config.ple_layer_ids)
    if not ple_layer_ids:
        return False

    from vllm.distributed.utils import get_pp_indices

    _, first_stage_end = get_pp_indices(
        int(text_config.num_hidden_layers),
        pp_rank=0,
        pp_size=pp_size,
    )
    return all(1 <= layer_id <= first_stage_end for layer_id in ple_layer_ids)


__all__ = ["first_stage_owns_all_ple_layers"]
