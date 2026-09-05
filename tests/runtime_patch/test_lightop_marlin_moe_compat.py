# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contracts for the LightOp-backed SlimQuant Marlin MoE boundary."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _install_lightop_marlin_kernel(
    monkeypatch: pytest.MonkeyPatch,
    implementation_name: str,
    export_name: str,
):
    implementation = ModuleType(implementation_name)

    def incomplete_alignment(
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
    ):
        del expert_map, pad_sorted_ids, ignore_invalid_experts
        return (
            torch.zeros(topk_ids.numel() * block_size, dtype=torch.int32),
            topk_ids.flatten().to(torch.int32),
            torch.tensor([topk_ids.numel() * block_size], dtype=torch.int32),
        )

    implementation.moe_align_block_size = incomplete_alignment
    monkeypatch.setitem(sys.modules, implementation_name, implementation)

    class Kernel:
        def __call__(self, **kwargs):
            return sys.modules[self.__module__].moe_align_block_size(
                kwargs["topk_ids"],
                2,
                kwargs["global_num_experts"],
                kwargs.get("expert_map"),
            )

    kernel = Kernel()
    kernel.__module__ = implementation_name

    def fill_alignment(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map=None,
    ):
        del expert_map
        flat_topk_ids = topk_ids.flatten()
        # Match vLLM's native kernel: every padding slot is initialized to the
        # numel sentinel before valid routed token ids are written.
        sorted_token_ids.fill_(topk_ids.numel())
        expert_ids.fill_(-1)
        write_offset = 0
        block_index = 0
        for expert_id in range(num_experts):
            token_ids = torch.nonzero(
                flat_topk_ids == expert_id,
                as_tuple=False,
            ).flatten()
            for chunk_start in range(0, token_ids.numel(), block_size):
                chunk = token_ids[chunk_start : chunk_start + block_size]
                sorted_token_ids[
                    write_offset : write_offset + chunk.numel()
                ] = chunk.to(torch.int32)
                expert_ids[block_index] = expert_id
                write_offset += block_size
                block_index += 1
        num_tokens_post_pad.fill_(write_offset)

    def moe_align_block_size_out(*_args, **_kwargs):
        raise AssertionError("LightOp alignment must not be used by Marlin")

    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    moe = ModuleType("lightop.moe")
    setattr(moe, export_name, kernel)
    moe.moe_align_block_size_out = moe_align_block_size_out
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)
    from vllm import _custom_ops as ops

    monkeypatch.setattr(ops, "moe_align_block_size", fill_alignment)
    return kernel


def _moe_layer(*, global_num_experts: int, expert_map=None):
    return SimpleNamespace(
        w13_weight=torch.ones(1),
        w2_weight=torch.ones(1),
        w13_weight_scale=torch.ones(1),
        w2_weight_scale=torch.ones(1),
        w13_input_scale=None,
        w2_input_scale=None,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=False,
        activation=SimpleNamespace(value="silu"),
    )


def test_fp8_marlin_reuses_native_alignment_initialization_without_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    kernel = _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop._lmslim_native.layers.fused_moe.fp8_marlin_test",
        "fused_experts_impl_fp8_marlin",
    )
    marlin.ensure_safe_marlin_moe_alignment(kernel)
    method = object.__new__(marlin.CompressedTensorsW8A8FP8MarlinMoEMethod)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        sorted_token_ids, _, _ = method.fused_moe_forward(
            _moe_layer(global_num_experts=2),
            torch.ones((1, 4)),
            torch.ones((1, 2)),
            torch.tensor([[0, 1]], dtype=torch.int32),
            global_num_experts=2,
        )

    assert sorted_token_ids.tolist() == [0, 2, 1, 2]
    assert not any(
        event.key == "aten::full" for event in profile.key_averages()
    )


def test_fp8_marlin_patches_public_lightop_runner_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    kernel = _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop.moe.fp8_marlin_public_test",
        "fused_experts_impl_fp8_marlin",
    )
    marlin.ensure_safe_marlin_moe_alignment(kernel)
    method = object.__new__(marlin.CompressedTensorsW8A8FP8MarlinMoEMethod)
    sorted_token_ids, _, _ = method.fused_moe_forward(
        _moe_layer(global_num_experts=2),
        torch.ones((1, 4)),
        torch.ones((1, 2)),
        torch.tensor([[0, 1]], dtype=torch.int32),
        global_num_experts=2,
    )

    assert sorted_token_ids.tolist() == [0, 2, 1, 2]


def test_fp8_marlin_uses_stable_vllm_alignment_not_lightop_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    kernel = _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop._lmslim_native.layers.fused_moe.fp8_marlin_stable_test",
        "fused_experts_impl_fp8_marlin",
    )
    marlin.ensure_safe_marlin_moe_alignment(kernel)

    method = object.__new__(marlin.CompressedTensorsW8A8FP8MarlinMoEMethod)
    sorted_token_ids, expert_ids, num_tokens_post_pad = method.fused_moe_forward(
        _moe_layer(global_num_experts=2),
        torch.ones((3, 4)),
        torch.ones((3, 2)),
        torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int32),
        global_num_experts=2,
    )

    assert num_tokens_post_pad.item() == 8
    assert sorted_token_ids.tolist() == [1, 2, 5, 6, 0, 3, 4, 6]
    assert expert_ids.tolist() == [0, 0, 1, 1]


def test_int8_marlin_safely_remaps_global_expert_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    kernel = _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop._lmslim_native.layers.fused_moe.int8_marlin_test",
        "fused_experts_impl_int8_marlin",
    )
    marlin.ensure_safe_marlin_moe_alignment(kernel)
    expert_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    layer = _moe_layer(global_num_experts=4, expert_map=expert_map)
    method = object.__new__(marlin.CompressedTensorsW8A8Int8MarlinMoEMethod)
    method.moe = None
    method.moe_quant_config = None
    _, expert_ids, _ = method.apply(
        layer,
        torch.ones((1, 4)),
        torch.ones((1, 2)),
        torch.tensor([[1, 3]], dtype=torch.int32),
        None,
        None,
    )

    assert expert_ids.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("method_name", "kernel_name"),
    [
        (
            "CompressedTensorsW8A8FP8MarlinMoEMethod",
            "fused_experts_impl_fp8_marlin",
        ),
        (
            "CompressedTensorsW8A8Int8MarlinMoEMethod",
            "fused_experts_impl_int8_marlin",
        ),
    ],
)
def test_marlin_installs_alignment_during_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    kernel_name: str,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    kernel = _install_lightop_marlin_kernel(
        monkeypatch,
        f"lightop.moe.{kernel_name}_load_test",
        kernel_name,
    )
    installed = []
    monkeypatch.setattr(
        marlin,
        "ensure_safe_marlin_moe_alignment",
        installed.append,
    )
    monkeypatch.setattr(
        marlin,
        "get_w8a8_int8_marlin_weights",
        lambda weight: weight,
    )
    method = object.__new__(getattr(marlin, method_name))
    method.use_deepep = False
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(torch.ones((1, 1))),
        w2_weight=torch.nn.Parameter(torch.ones((1, 1))),
    )

    method.process_weights_after_loading(layer)

    assert installed == [kernel]


@pytest.mark.parametrize(
    ("method_name", "kernel_name"),
    [
        (
            "CompressedTensorsW8A8FP8MarlinMoEMethod",
            "fused_experts_impl_fp8_marlin",
        ),
        (
            "CompressedTensorsW8A8Int8MarlinMoEMethod",
            "fused_experts_impl_int8_marlin",
        ),
    ],
)
@pytest.mark.parametrize("shared_experts_overlap", [False, True])
def test_marlin_allocates_routed_output_only_during_shared_input_overlap(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    kernel_name: str,
    shared_experts_overlap: bool,
) -> None:
    """Concurrent shared reads require out-of-place routed output, not a copy."""

    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    def marlin_kernel(**kwargs):
        hidden_states = kwargs["hidden_states"]
        if kwargs["inplace"]:
            hidden_states.fill_(3)
            return hidden_states
        return torch.full_like(hidden_states, 3)

    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    moe = ModuleType("lightop.moe")
    setattr(moe, kernel_name, marlin_kernel)
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)

    def fail_if_installed_during_forward(_kernel) -> None:
        raise AssertionError("alignment installation escaped weight loading")

    monkeypatch.setattr(
        marlin,
        "ensure_safe_marlin_moe_alignment",
        fail_if_installed_during_forward,
    )

    class SharedExperts:
        def allows_inplace_routed_output(
            self,
            routed_input: torch.Tensor,
            shared_input: torch.Tensor,
        ) -> bool:
            return not (
                torch._C._is_alias_of(routed_input, shared_input)
                and shared_experts_overlap
            )

    method = object.__new__(getattr(marlin, method_name))
    method.moe = object()
    method.use_deepep = False
    method.moe_quant_config = None
    if kernel_name == "fused_experts_impl_fp8_marlin":
        method.fused_experts = method.fused_moe_forward
    layer = _moe_layer(global_num_experts=2)
    hidden_states = torch.ones((1, 4))
    original_hidden_states = hidden_states.clone()

    output = method.apply(
        layer,
        hidden_states,
        torch.ones((1, 2)),
        torch.tensor([[0, 1]], dtype=torch.int32),
        SharedExperts(),
        hidden_states,
    )

    assert torch._C._is_alias_of(output, hidden_states) is (
        not shared_experts_overlap
    )
    if shared_experts_overlap:
        torch.testing.assert_close(hidden_states, original_hidden_states)
    torch.testing.assert_close(output, torch.full_like(hidden_states, 3))
