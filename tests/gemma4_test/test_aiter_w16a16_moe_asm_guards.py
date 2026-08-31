# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from vllm_hcu.model_executor.layers.fused_moe import (
    aiter_moe_dispatch,
    aiter_runtime,
    unquantized_fused_moe_method,
)


def test_w16a16_runtime_uses_the_shared_aiter_selector() -> None:
    assert (
        aiter_runtime.select_aiter_moe_config
        is aiter_moe_dispatch.select_aiter_moe_config
    )
    assert aiter_runtime.execute_aiter_moe is aiter_moe_dispatch.execute_aiter_moe


def test_w16a16_runtime_exposes_no_asm_only_selector_or_loader() -> None:
    assert not hasattr(aiter_runtime, "get_w16a16_moe_config")
    assert not hasattr(aiter_runtime, "get_w16a16_moe_solution_id")
    assert not hasattr(
        unquantized_fused_moe_method,
        "_raise_if_aiter_moe_asm_blocked",
    )
