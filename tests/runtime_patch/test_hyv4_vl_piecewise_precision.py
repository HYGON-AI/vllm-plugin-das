"""Preserve HYV4-VL precision when prefill graph buckets add tokens."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/hcu_model_runner.py"


def _load_batch_dispatch():
    """Exercise the runner's dispatch closure without importing GPU extensions."""
    from enum import Enum
    from types import SimpleNamespace

    class Mode(Enum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    runner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    method = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_determine_batch_execution_and_padding"
    )
    dispatch = next(
        n
        for n in method.body
        if isinstance(n, ast.FunctionDef) and n.name == "dispatch_cudagraph"
    )
    namespace = {
        "CUDAGraphMode": Mode,
        "BatchDescriptor": lambda num_tokens: SimpleNamespace(num_tokens=num_tokens),
        "uniform_decode": False,
        "has_lora": False,
        "num_active_loras": 0,
        "force_eager": False,
    }
    exec(  # noqa: S102 - Execute only the repository method under test.
        compile(ast.Module(body=[dispatch], type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace, Mode


def test_hyv4_vl_piecewise_preserves_unpadded_prefill_shapes():
    from types import SimpleNamespace

    namespace, mode = _load_batch_dispatch()
    calls = []

    def dispatch(**kwargs):
        calls.append(kwargs)
        return mode.PIECEWISE, SimpleNamespace(num_tokens=128)

    namespace["self"] = SimpleNamespace(
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["HYV4VLForConditionalGeneration"])
        ),
    )
    actual_mode, descriptor = namespace["dispatch_cudagraph"](22)
    assert actual_mode == mode.NONE
    assert descriptor.num_tokens == 22
    assert calls == [
        {
            "num_tokens": 22,
            "has_lora": False,
            "uniform_decode": False,
            "num_active_loras": 0,
            "valid_modes": None,
            "invalid_modes": None,
        }
    ]


def test_hyv4_vl_guard_retains_exact_graphs_and_other_dispatch_contracts():
    from types import SimpleNamespace

    namespace, mode = _load_batch_dispatch()
    # Full decode, exact prefill, unrelated models and DP coordination retain
    # the descriptor selected by the existing dispatcher.
    for runtime_mode, actual, padded, architectures, dp_size in (
        (mode.PIECEWISE, 1024, 1024, ["HYV4VLForConditionalGeneration"], 1),
        (mode.FULL, 1, 2, ["HYV4VLForConditionalGeneration"], 1),
        (mode.NONE, 22, 22, ["HYV4VLForConditionalGeneration"], 1),
        (mode.PIECEWISE, 22, 128, ["OtherModel"], 1),
        (mode.PIECEWISE, 22, 128, None, 1),
        (mode.PIECEWISE, 22, 128, ["HYV4VLForConditionalGeneration"], 2),
    ):
        descriptor = SimpleNamespace(num_tokens=padded)
        selected = (runtime_mode, descriptor)
        namespace["self"] = SimpleNamespace(
            cudagraph_dispatcher=SimpleNamespace(
                dispatch=lambda selected=selected, **kwargs: selected
            ),
            parallel_config=SimpleNamespace(data_parallel_size=dp_size),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=architectures)
            ),
        )
        result = namespace["dispatch_cudagraph"](actual, disable_full=True)
        assert result == selected
        assert result[1] is descriptor
