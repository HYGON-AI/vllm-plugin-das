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


def test_fp8_marlin_prefills_unwritten_alignment_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop._lmslim_native.layers.fused_moe.fp8_marlin_test",
        "fused_experts_impl_fp8_marlin",
    )
    method = object.__new__(marlin.CompressedTensorsW8A8FP8MarlinMoEMethod)
    sorted_token_ids, _, _ = method.fused_moe_forward(
        _moe_layer(global_num_experts=2),
        torch.ones((1, 4)),
        torch.ones((1, 2)),
        torch.tensor([[0, 1]], dtype=torch.int32),
        global_num_experts=2,
    )

    assert sorted_token_ids.tolist() == [0, 2, 1, 2]


def test_fp8_marlin_patches_public_lightop_runner_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop.moe.fp8_marlin_public_test",
        "fused_experts_impl_fp8_marlin",
    )
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

    _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop._lmslim_native.layers.fused_moe.fp8_marlin_stable_test",
        "fused_experts_impl_fp8_marlin",
    )

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

    _install_lightop_marlin_kernel(
        monkeypatch,
        "lightop._lmslim_native.layers.fused_moe.int8_marlin_test",
        "fused_experts_impl_int8_marlin",
    )
    monkeypatch.setattr(
        marlin,
        "_is_hcu_aiter_w8a8_moe_requested",
        lambda *_args: False,
    )
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
