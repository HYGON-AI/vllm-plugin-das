# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.attention import lightop_concat_runtime


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        torch.arange(12, dtype=torch.bfloat16).reshape(2, 3, 2),
    )


@pytest.mark.parametrize(
    ("master", "leaf", "legacy", "uses_lightop"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_mla_concat_obeys_master_leaf_and_legacy_switches(
    monkeypatch: pytest.MonkeyPatch,
    master: bool,
    leaf: bool,
    legacy: bool,
    uses_lightop: bool,
) -> None:
    left, right = _inputs()
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    monkeypatch.setattr(
        lightop_concat_runtime,
        "henvs",
        SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=master,
            VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT=leaf,
            VLLM_USE_OPT_CAT=legacy,
        ),
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "is_lightop_mla_decode_concat_eligible",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "_load_lightop_ds_cat",
        lambda: object(),
    )

    def lightop_call(
        actual_left: torch.Tensor, actual_right: torch.Tensor
    ) -> torch.Tensor:
        calls.append((actual_left, actual_right))
        return torch.cat((actual_left, actual_right), dim=-1)

    monkeypatch.setattr(lightop_concat_runtime, "_call_registered_lightop", lightop_call)
    actual = lightop_concat_runtime.concat_mla_decode(left, right, dim=-1)

    torch.testing.assert_close(actual, torch.cat((left, right), dim=-1))
    assert bool(calls) is uses_lightop


@pytest.mark.parametrize("dim", [-1, 2])
def test_mla_concat_accepts_last_dimension(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    left, right = _inputs()
    monkeypatch.setattr(
        lightop_concat_runtime,
        "henvs",
        SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT=True,
            VLLM_USE_OPT_CAT=True,
        ),
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "is_lightop_mla_decode_concat_eligible",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "_load_lightop_ds_cat",
        lambda: object(),
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "_call_registered_lightop",
        lambda actual_left, actual_right: torch.cat(
            (actual_left, actual_right), dim=-1
        ),
    )

    torch.testing.assert_close(
        lightop_concat_runtime.concat_mla_decode(left, right, dim=dim),
        torch.cat((left, right), dim=dim),
    )


def test_mla_concat_rejects_unsupported_dimension() -> None:
    left, right = _inputs()
    with pytest.raises(ValueError, match="last dimension"):
        lightop_concat_runtime.concat_mla_decode(left, right, dim=1)


def test_mla_concat_ineligible_inputs_use_torch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _inputs()
    monkeypatch.setattr(
        lightop_concat_runtime,
        "henvs",
        SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT=True,
            VLLM_USE_OPT_CAT=True,
        ),
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "_call_registered_lightop",
        lambda *_args: pytest.fail("ineligible inputs must not call LightOp"),
    )

    actual = lightop_concat_runtime.concat_mla_decode(left, right, dim=-1)
    torch.testing.assert_close(actual, torch.cat((left, right), dim=-1))


def test_lightop_impl_uses_categorized_mode_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _inputs()
    calls: list[int] = []

    def ds_cat(
        actual_left: torch.Tensor,
        actual_right: torch.Tensor,
        output: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        calls.append(mode)
        output.copy_(torch.cat((actual_left, actual_right), dim=-1))
        return output

    monkeypatch.setattr(
        lightop_concat_runtime,
        "_load_lightop_ds_cat",
        lambda: ds_cat,
    )
    actual = lightop_concat_runtime.lightop_mla_decode_concat_impl(left, right)

    torch.testing.assert_close(actual, torch.cat((left, right), dim=-1))
    assert calls == [0]


def test_lightop_impl_does_not_hide_kernel_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _inputs()

    def fail(*_args: object) -> None:
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(lightop_concat_runtime, "_load_lightop_ds_cat", lambda: fail)
    with pytest.raises(RuntimeError, match="kernel failed"):
        lightop_concat_runtime.lightop_mla_decode_concat_impl(left, right)


def test_missing_categorized_operator_uses_torch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _inputs()
    monkeypatch.setattr(
        lightop_concat_runtime,
        "henvs",
        SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT=True,
            VLLM_USE_OPT_CAT=True,
        ),
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "is_lightop_mla_decode_concat_eligible",
        lambda *_args, **_kwargs: True,
    )

    def unavailable() -> object:
        raise lightop_concat_runtime.LightOpMlaConcatUnavailable("missing")

    monkeypatch.setattr(lightop_concat_runtime, "_load_lightop_ds_cat", unavailable)
    monkeypatch.setattr(
        lightop_concat_runtime,
        "_call_registered_lightop",
        lambda *_args: pytest.fail("missing dependency must not call the custom op"),
    )

    torch.testing.assert_close(
        lightop_concat_runtime.concat_mla_decode(left, right),
        torch.cat((left, right), dim=-1),
    )
