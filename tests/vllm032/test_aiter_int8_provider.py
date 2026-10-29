from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PLUGIN_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
)
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))

from hcu_vllm_das.moe import fused_moe_aiter as provider


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], *, dtype: object = "bf16") -> None:
        self.shape = shape
        self.dtype = dtype


def inputs() -> dict[str, object]:
    return {
        "hidden_states": FakeTensor((3, 8)),
        "w1": FakeTensor((4, 12, 8), dtype="int8"),
        "w2": FakeTensor((4, 8, 6), dtype="int8"),
        "topk_weights": FakeTensor((3, 2), dtype="float32"),
        "topk_ids": FakeTensor((3, 2), dtype="int32"),
        "moe_config": SimpleNamespace(num_experts=8),
        "quant_config": SimpleNamespace(
            use_int8_w8a8=True,
            w1_scale="w1-scale",
            w2_scale="w2-scale",
            w1_zp="w1-zp",
            w2_zp="w2-zp",
            a1_scale="a1-scale",
            a2_scale="a2-scale",
        ),
        "activation": SimpleNamespace(value="silu"),
        "expert_map": "expert-map",
        "output_dtype": "bf16-output",
    }


def native_config(**overrides: object) -> SimpleNamespace:
    values = {
        "solution_type": "asm",
        "config": {"SOL_ID1": 10010, "SOL_ID2": 20000, "BLOCK_SIZE_M": 16},
        "need_shuffle": False,
        "need_shuffle_scale": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shape_evidence_uses_vllm_runtime_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str, tuple[object, ...]]] = []
    logger_names: list[str] = []
    runtime_logger = SimpleNamespace(
        log=lambda level, message, *args: calls.append((level, message, args))
    )

    def init_logger(name: str):
        logger_names.append(name)
        return runtime_logger

    def load_module(name: str):
        assert name == "vllm.logger"
        return SimpleNamespace(init_logger=init_logger)

    monkeypatch.setattr(provider, "import_module", load_module)
    provider._record_once(
        provider.logging.INFO,
        "selected",
        (7, 4, 12, 8, 8, 2),
        "asm",
        native_config(),
    )

    assert logger_names == ["vllm.hcu_vllm_das.moe.fused_moe_aiter"]
    assert calls == [
        (
            provider.logging.INFO,
            "HCU AITER INT8 MoE %s: M=%d E=%d N1=%d N2=%d K=%d top_k=%d "
            "solution=%s config=%s",
            (
                "selected",
                7,
                4,
                12,
                8,
                8,
                2,
                "asm",
                {"SOL_ID1": 10010, "SOL_ID2": 20000, "BLOCK_SIZE_M": 16},
            ),
        )
    ]


def test_native_config_invokes_public_aiter_w8a8_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    expected = object()
    arguments = inputs()
    selected_config = native_config()

    def get_config(*args: object, **kwargs: object):
        calls["config"] = (args, kwargs)
        return True, selected_config

    def aiter_moe(*args: object, **kwargs: object):
        calls["aiter"] = (args, kwargs)
        return expected

    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    monkeypatch.setattr(
        provider,
        "_load_triton_fallback",
        lambda: pytest.fail("native AITER config must not use Triton fallback"),
    )

    result = provider.apply_aiter_int8_moe(**arguments)

    assert result is expected
    assert calls["config"] == (
        (3, 4, 12, 8, 8, 2, 0, "bf16", "int8_w8a8"),
        {"activation": "silu", "use_shuffle": 0},
    )
    assert calls["aiter"] == (
        (),
        {
            "hidden_states": arguments["hidden_states"],
            "w1": arguments["w1"],
            "w2": arguments["w2"],
            "topk_weights": arguments["topk_weights"],
            "topk_ids": arguments["topk_ids"],
            "moe_config": selected_config,
            "inplace": False,
            "activation": "silu",
            "w1_scale": "w1-scale",
            "w2_scale": "w2-scale",
            "w1_zp": "w1-zp",
            "w2_zp": "w2-zp",
            "a1_scale": "a1-scale",
            "a2_scale": "a2-scale",
            "block_shape": None,
            "global_num_experts": 8,
            "expert_map": "expert-map",
            "routed_scaling_factor": 1.0,
            "use_weight_shuffle": False,
            "output_dtype": "bf16-output",
        },
    )


@pytest.mark.parametrize(
    ("found", "config"),
    (
        (False, native_config(solution_type=None, config=None)),
        (True, native_config(need_shuffle=True)),
        (True, native_config(need_shuffle_scale=True)),
    ),
    ids=("config-miss", "weight-shuffle", "scale-shuffle"),
)
def test_unsupported_aiter_layout_uses_official_int8_triton_fallback(
    monkeypatch: pytest.MonkeyPatch,
    found: bool,
    config: SimpleNamespace,
) -> None:
    calls: list[dict[str, object]] = []
    expected = object()
    arguments = inputs()
    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=lambda *args, **kwargs: (found, config),
            aiter_moe=lambda **kwargs: pytest.fail("unsupported config must fallback"),
        ),
    )

    def fallback(**kwargs: object):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(provider, "_load_triton_fallback", lambda: fallback)

    result = provider.apply_aiter_int8_moe(**arguments)

    assert result is expected
    assert calls == [
        {
            "hidden_states": arguments["hidden_states"],
            "w1": arguments["w1"],
            "w2": arguments["w2"],
            "topk_weights": arguments["topk_weights"],
            "topk_ids": arguments["topk_ids"],
            "activation": "silu",
            "apply_router_weight_on_input": False,
            "use_fp8_w8a8": False,
            "use_int8_w8a8": True,
            "use_int8_w8a16": False,
            "use_int4_w4a16": False,
            "per_channel_quant": True,
            "global_num_experts": 8,
            "expert_map": "expert-map",
            "w1_scale": "w1-scale",
            "w2_scale": "w2-scale",
            "w1_zp": "w1-zp",
            "w2_zp": "w2-zp",
            "a1_scale": "a1-scale",
            "a2_scale": "a2-scale",
            "block_shape": None,
        }
    ]


def test_vendor_failure_after_config_selection_does_not_change_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_vendor(**kwargs: object):
        raise RuntimeError("vendor kernel failed")

    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=lambda *args, **kwargs: (True, native_config()),
            aiter_moe=fail_vendor,
        ),
    )
    monkeypatch.setattr(
        provider,
        "_load_triton_fallback",
        lambda: pytest.fail("vendor execution errors must propagate"),
    )

    with pytest.raises(RuntimeError, match="vendor kernel failed"):
        provider.apply_aiter_int8_moe(**inputs())
