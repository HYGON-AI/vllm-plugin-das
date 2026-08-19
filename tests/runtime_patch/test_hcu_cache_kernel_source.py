# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_path", "expected_launch"),
    [
        (
            "vllm_hcu/csrc/hcu_cache_kernel.cu",
            "dim3 block(std::min(kv_lora_rank, 256));",
        ),
        (
            "vllm_hcu/csrc/hcu_cache_kernel.hip",
            "dim3 block(::min(kv_lora_rank, 256));",
        ),
    ],
)
def test_mla_cache_launch_respects_hcu_thread_limit(
    relative_path: str, expected_launch: str
) -> None:
    source = (REPO / relative_path).read_text(encoding="utf-8")
    function = source.split("void concat_and_cache_mla_hcu(", 1)[1]

    assert expected_launch in function
    assert "min(kv_lora_rank, 512)" not in function
