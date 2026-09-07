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


def test_flash_cache_writer_uses_runtime_strides_and_mutating_schema() -> None:
    kernel_source = (
        REPO / "vllm_hcu/csrc/hcu_cache_kernel.cu"
    ).read_text(encoding="utf-8")
    kernel = kernel_source.split(
        "__global__ void reshape_and_cache_flash_kernel_hcu(", 1
    )[1].split("template <typename scalar_t", 1)[0]

    assert "block_idx * block_stride" in kernel
    assert "block_offset * page_stride" in kernel
    assert "head_idx * head_stride" in kernel
    assert "if (slot_idx < 0)" in kernel

    host = kernel_source.split("void reshape_and_cache_flash_hcu(", 1)[1].split(
        "void concat_and_cache_mla_hcu(", 1
    )[0]
    assert "std::min(num_heads * head_size, 256)" in host

    bindings = (REPO / "vllm_hcu/csrc/torch_bindings.cpp").read_text(
        encoding="utf-8"
    )
    assert (
        '"reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, "'
        in bindings
    )
    assert '"Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, "' in (
        bindings
    )
    assert 'ops.impl("reshape_and_cache_flash", torch::kCUDA' in bindings
