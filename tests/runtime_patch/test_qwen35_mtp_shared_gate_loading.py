# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType

import torch

from vllm_hcu.patch.worker.core_fix import patch_qwen3_5_mtp_shared_gate


def _module(model_class: type) -> ModuleType:
    module = ModuleType(patch_qwen3_5_mtp_shared_gate.TARGET_MODULE)
    module.Qwen3_5MTP = model_class
    return module


def test_unquantized_mtp_shared_gate_dequantizes_serialized_int8_pair() -> None:
    captured: list[tuple[str, torch.Tensor]] = []

    class Qwen3_5MTP:
        def named_parameters(self):
            yield "model.layers.0.mlp.shared_expert_gate.weight", torch.empty(
                1, dtype=torch.bfloat16
            )

        def load_weights(self, weights):
            captured.extend(weights)
            return {name for name, _ in captured}

    module = _module(Qwen3_5MTP)
    patch_qwen3_5_mtp_shared_gate.apply_to_module(module)
    model = Qwen3_5MTP()
    result = model.load_weights(
        iter(
            [
                ("other.weight", torch.tensor([7.0], dtype=torch.bfloat16)),
                (
                    "mtp.layers.0.mlp.shared_expert_gate.weight",
                    torch.tensor([[2, -4]], dtype=torch.int8),
                ),
                (
                    "mtp.layers.0.mlp.shared_expert_gate.weight_scale",
                    torch.tensor([[0.5]], dtype=torch.bfloat16),
                ),
            ]
        )
    )

    assert result == {
        "other.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    }
    assert captured[0][0] == "other.weight"
    name, weight = captured[1]
    assert name == "mtp.layers.0.mlp.shared_expert_gate.weight"
    assert weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        weight,
        torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16),
    )


def test_quantized_mtp_shared_gate_preserves_weight_and_scale() -> None:
    captured: list[tuple[str, torch.Tensor]] = []

    class Qwen3_5MTP:
        def named_parameters(self):
            yield "model.layers.0.mlp.shared_expert_gate.weight", torch.empty(1)
            yield "model.layers.0.mlp.shared_expert_gate.weight_scale", torch.empty(1)

        def load_weights(self, weights):
            captured.extend(weights)
            return {name for name, _ in captured}

    module = _module(Qwen3_5MTP)
    patch_qwen3_5_mtp_shared_gate.apply_to_module(module)
    model = Qwen3_5MTP()
    names = model.load_weights(
        iter(
            [
                (
                    "mtp.layers.0.mlp.shared_expert_gate.weight",
                    torch.tensor([[2]], dtype=torch.int8),
                ),
                (
                    "mtp.layers.0.mlp.shared_expert_gate.weight_scale",
                    torch.tensor([[0.5]], dtype=torch.bfloat16),
                ),
            ]
        )
    )

    assert names == {
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight_scale",
    }
    assert len(captured) == 2


def test_unquantized_bf16_shared_gate_without_scale_is_preserved() -> None:
    captured: list[tuple[str, torch.Tensor]] = []

    class Qwen3_5MTP:
        def named_parameters(self):
            yield "model.layers.0.mlp.shared_expert_gate.weight", torch.empty(1)

        def load_weights(self, weights):
            captured.extend(weights)
            return {name for name, _ in captured}

    module = _module(Qwen3_5MTP)
    patch_qwen3_5_mtp_shared_gate.apply_to_module(module)
    source = torch.tensor([[3.0]], dtype=torch.bfloat16)
    names = Qwen3_5MTP().load_weights(
        iter([("mtp.layers.0.mlp.shared_expert_gate.weight", source)])
    )

    assert names == {"mtp.layers.0.mlp.shared_expert_gate.weight"}
    assert captured == [("mtp.layers.0.mlp.shared_expert_gate.weight", source)]
