# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_hcu.model_executor.layers.attention import lightop_concat_runtime


class _CudaTensorMetadata:
    def __init__(
        self,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> None:
        self.shape = shape
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda")
        self._strides = strides

    def dim(self) -> int:
        return len(self.shape)

    def stride(self, dim: int) -> int:
        return self._strides[dim]


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


@pytest.mark.parametrize(
    ("tokens", "heads", "left_stride", "right_stride"),
    (
        (32, 16, (512, 16384, 1), (3072, 192, 1)),
        (128, 7, (512, 65536, 1), (1536, 192, 1)),
        (128, 64, (512, 65536, 1), (12288, 192, 1)),
        (128, 16, (8192, 512, 1), (1024, 64, 1)),
        (128, 16, (512, 65536, 1), (0, 192, 1)),
    ),
)
def test_mla_concat_eligibility_rejects_unmeasured_or_nonproduction_layouts(
    tokens: int,
    heads: int,
    left_stride: tuple[int, ...],
    right_stride: tuple[int, ...],
) -> None:
    left = _CudaTensorMetadata((tokens, heads, 512), left_stride)
    right = _CudaTensorMetadata((tokens, heads, 64), right_stride)

    assert not lightop_concat_runtime.is_lightop_mla_decode_concat_eligible(
        left, right
    )


def test_mla_concat_eligibility_accepts_benchmarked_flashmla_layout() -> None:
    tokens, heads = 128, 16
    left = _CudaTensorMetadata(
        (tokens, heads, 512),
        (512, 512 * tokens, 1),
    )
    right = _CudaTensorMetadata(
        (tokens, heads, 64),
        (1536 * (heads // 8), 192, 1),
    )

    assert lightop_concat_runtime.is_lightop_mla_decode_concat_eligible(left, right)


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
    messages: list[str] = []

    def fail(*_args: object) -> None:
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(lightop_concat_runtime, "_load_lightop_ds_cat", lambda: fail)
    monkeypatch.setattr(
        lightop_concat_runtime,
        "logger",
        SimpleNamespace(warning_once=messages.append),
    )
    with pytest.raises(RuntimeError, match="kernel failed"):
        lightop_concat_runtime.lightop_mla_decode_concat_impl(left, right)
    assert messages == []


def test_mla_concat_frontend_is_fullgraph_compilable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lightop_concat_runtime,
        "henvs",
        SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT=True,
            VLLM_USE_OPT_CAT=True,
        ),
    )
    with FakeTensorMode():
        tokens, heads = 128, 16
        left = torch.as_strided(
            torch.empty(tokens * heads * 512, device="cuda", dtype=torch.bfloat16),
            size=(tokens, heads, 512),
            stride=(512, 512 * tokens, 1),
        )
        right = torch.as_strided(
            torch.empty(
                1536 * (heads // 8) * tokens,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            size=(tokens, heads, 64),
            stride=(1536 * (heads // 8), 192, 1),
        )
        result = torch.compile(
            lightop_concat_runtime.concat_mla_decode,
            backend="eager",
            fullgraph=True,
        )(left, right)

    assert result.shape == (128, 16, 576)
    assert result.dtype is torch.bfloat16


def test_mla_concat_frontend_does_not_probe_optional_dependency(
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
    monkeypatch.setattr(
        lightop_concat_runtime,
        "_load_lightop_ds_cat",
        lambda: pytest.fail("dependency probing belongs inside the custom op"),
    )
    monkeypatch.setattr(
        lightop_concat_runtime,
        "_call_registered_lightop",
        lambda actual_left, actual_right: torch.cat(
            (actual_left, actual_right), dim=-1
        ),
    )

    torch.testing.assert_close(
        lightop_concat_runtime.concat_mla_decode(left, right),
        torch.cat((left, right), dim=-1),
    )


def test_non_callable_categorized_operator_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lightop = ModuleType("lightop")
    lightop.__path__ = []
    tensor = ModuleType("lightop.tensor")
    tensor.ds_cat = object()
    lightop.tensor = tensor
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.tensor", tensor)

    with pytest.raises(
        lightop_concat_runtime.LightOpMlaConcatUnavailable,
        match="unavailable",
    ):
        lightop_concat_runtime._load_lightop_ds_cat()


def test_missing_categorized_operator_falls_back_inside_registered_impl(
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
        lightop_concat_runtime.lightop_mla_decode_concat_impl,
    )

    torch.testing.assert_close(
        lightop_concat_runtime.concat_mla_decode(left, right),
        torch.cat((left, right), dim=-1),
    )
