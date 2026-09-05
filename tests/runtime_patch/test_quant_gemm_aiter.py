# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import builtins
import enum
import inspect
import logging
import os
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import (
    aiter_moe_dispatch,
    aiter_runtime,
)
from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
    AiterMoeProblem,
    HcuAiterMoeDispatchError,
    aiter_expert_map_for_solution,
    execute_aiter_moe,
    prepare_aiter_moe_scales,
    prepare_aiter_moe_weights,
    select_aiter_moe_config,
)
from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime,
    int8_runtime,
    lightop_fp8_runtime,
)
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.patch.worker.op_opt import (
    patch_activation,
    patch_aiter_ops,
    patch_compressed_tensors,
    patch_compressed_tensors_moe_w8a8_fp8,
    patch_compressed_tensors_moe_wna16,
    patch_compressed_tensors_scheme,
    patch_compressed_tensors_w8a8_fp8,
    patch_compressed_tensors_w8a8_int8,
    patch_deep_gemm,
    patch_input_quant_fp8,
    patch_layers_utils,
    patch_scaled_mm_linear_kernel,
    patch_w8a8_utils,
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _package(name: str, **attributes: object) -> ModuleType:
    module = _module(name, **attributes)
    module.__package__ = name
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_lightop_moe(
    monkeypatch: pytest.MonkeyPatch, **exports: object
) -> ModuleType:
    lightop = _package("lightop")
    moe = _module("lightop.moe", **exports)
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)
    return moe


def _install_lightop_activation(
    monkeypatch: pytest.MonkeyPatch, **exports: object
) -> ModuleType:
    lightop = _package("lightop")
    activation = _module("lightop.activation", **exports)
    lightop.activation = activation
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.activation", activation)
    return activation


def _reject_import_prefix(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    real_import = builtins.__import__

    def reject_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == prefix or name.startswith(f"{prefix}."):
            raise AssertionError(f"unexpected import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_import)


def _fp8_quant_abi_stub(
    x,
    scale=None,
    quant_dtype=torch.int8,
    num_rows=None,
    num_rows_factor=1,
):
    del quant_dtype, num_rows, num_rows_factor
    return x, scale


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("0", False),
        ("1", True),
    ],
)
def test_unified_aiter_moe_shuffle_env(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    expected: bool,
):
    if value is None:
        monkeypatch.delenv("VLLM_HCU_USE_AITER_MOE_SHUFFLE", raising=False)
    else:
        monkeypatch.setenv("VLLM_HCU_USE_AITER_MOE_SHUFFLE", value)

    henvs.resolve_aiter_moe_shuffle.cache_clear()
    assert henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE is expected


def test_removed_w16a16_shuffle_env_cannot_change_unified_behavior(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_HCU_USE_AITER_MOE_SHUFFLE", raising=False)
    monkeypatch.setenv("VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", "0")
    henvs.resolve_aiter_moe_shuffle.cache_clear()

    assert henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE is True
    with pytest.raises(AttributeError):
        getattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE")


def test_unified_aiter_moe_config_disable_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setenv("VLLM_HCU_USE_AITER_MOE_CONFIG", "0")
    henvs.resolve_aiter_moe_config_compat.cache_clear()

    with caplog.at_level("WARNING", logger=henvs.__name__):
        assert henvs.VLLM_HCU_USE_AITER_MOE_CONFIG is True

    assert any(
        "deprecated and ignored" in record.message for record in caplog.records
    )


def test_aiter_dispatch_selector_passes_problem_without_forcing_solution(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    expected_config = SimpleNamespace(solution_type="triton")

    def get_config(**kwargs: object):
        captured.update(kwargs)
        return True, expected_config

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=get_config),
    )
    problem = AiterMoeProblem(
        M=2,
        E=4,
        N1=16,
        N2=8,
        K=8,
        top_k=2,
        block_size=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        quant_type="w16a16",
        activation="silu",
        use_shuffle=True,
    )

    actual = select_aiter_moe_config(problem, cache_owner=torch.empty(1))

    assert actual is expected_config
    assert captured == {
        "M": 2,
        "E": 4,
        "N1": 16,
        "N2": 8,
        "K": 8,
        "top_k": 2,
        "block_size": 0,
        "dtype": torch.bfloat16,
        "quant_type": "w16a16",
        "activation": "silu",
        "use_shuffle": 1,
    }
    assert "spec_sol_type" not in captured
    assert "device" not in captured


def test_aiter_dispatch_can_pin_asm_shuffle_solution(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    expected = SimpleNamespace(solution_type="asm", need_shuffle=True)

    def get_config(**kwargs: object):
        captured.update(kwargs)
        return True, expected

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=get_config),
    )
    problem = AiterMoeProblem(
        M=1,
        E=2,
        N1=8,
        N2=4,
        K=4,
        top_k=2,
        block_size=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        quant_type="w16a16",
        use_shuffle=True,
    )

    actual = select_aiter_moe_config(
        problem,
        cache_owner=torch.empty(1),
        solution_type="asm",
    )

    assert actual is expected
    assert captured["spec_sol_type"] == "asm"
    assert captured["use_shuffle"] == 1


def test_aiter_dispatch_selector_returns_none_only_for_explicit_no_solution(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (False, None),
        ),
    )
    problem = AiterMoeProblem(
        M=1,
        E=2,
        N1=4,
        N2=2,
        K=2,
        top_k=1,
        block_size=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
        quant_type="w16a16",
    )

    assert select_aiter_moe_config(problem, cache_owner=object()) is None


def test_aiter_dispatch_prewarm_m1_does_not_gate_supported_runtime_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def get_config(**kwargs: object):
        m = int(kwargs["M"])
        calls.append(m)
        if m == 1:
            return False, None
        return True, SimpleNamespace(solution_type=f"route-m{m}")

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=get_config),
    )
    owner = torch.empty(1)
    runtime_problem = AiterMoeProblem(
        M=64,
        E=2,
        N1=4,
        N2=2,
        K=2,
        top_k=1,
        block_size=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
        quant_type="w8a8",
    )

    assert (
        aiter_moe_dispatch.prewarm_aiter_moe_config(
            runtime_problem,
            cache_owner=owner,
        )
        is None
    )
    runtime_config = select_aiter_moe_config(
        runtime_problem,
        cache_owner=owner,
    )
    cached_config = select_aiter_moe_config(
        runtime_problem,
        cache_owner=owner,
    )

    assert runtime_config.solution_type == "route-m64"
    assert cached_config is runtime_config
    assert calls == [1, 64]


def test_aiter_dispatch_prewarm_m1_keeps_actual_m_routing_dynamic(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def get_config(**kwargs: object):
        m = int(kwargs["M"])
        calls.append(m)
        return True, SimpleNamespace(solution_type=f"route-m{m}")

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=get_config),
    )
    owner = torch.empty(1)
    runtime_problem = AiterMoeProblem(
        M=64,
        E=2,
        N1=4,
        N2=2,
        K=2,
        top_k=1,
        block_size=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
        quant_type="w8a8",
    )

    warm_config = aiter_moe_dispatch.prewarm_aiter_moe_config(
        runtime_problem,
        cache_owner=owner,
    )
    runtime_config = select_aiter_moe_config(runtime_problem, cache_owner=owner)
    cached_config = select_aiter_moe_config(runtime_problem, cache_owner=owner)

    assert warm_config.solution_type == "route-m1"
    assert runtime_config.solution_type == "route-m64"
    assert cached_config is runtime_config
    assert calls == [1, 64]


def test_aiter_dispatch_selector_rejects_success_without_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (True, None),
        ),
    )
    problem = AiterMoeProblem(
        M=3,
        E=2,
        N1=4,
        N2=2,
        K=2,
        top_k=1,
        block_size=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
        quant_type="w8a8",
    )

    with pytest.raises(HcuAiterMoeDispatchError, match="status=True"):
        select_aiter_moe_config(problem, cache_owner=object())


def test_aiter_dispatch_selector_rejects_non_boolean_status(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (None, object()),
        ),
    )
    problem = AiterMoeProblem(
        M=3,
        E=2,
        N1=4,
        N2=2,
        K=2,
        top_k=1,
        block_size=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
        quant_type="w8a8",
    )

    with pytest.raises(HcuAiterMoeDispatchError, match="boolean status"):
        select_aiter_moe_config(problem, cache_owner=object())


def test_aiter_dispatch_selection_cache_retains_more_than_eight_m_buckets(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def get_config(**kwargs: object):
        calls.append(int(kwargs["M"]))
        return True, SimpleNamespace(solution_type="asm")

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=get_config),
    )
    owner = torch.empty(1)
    problems = [
        AiterMoeProblem(
            M=m,
            E=2,
            N1=4,
            N2=2,
            K=2,
            top_k=1,
            block_size=0,
            dtype=torch.float16,
            device=torch.device("cpu"),
            quant_type="w8a8",
        )
        for m in range(1, 17)
    ]
    for problem in problems:
        select_aiter_moe_config(problem, cache_owner=owner)
    select_aiter_moe_config(problems[0], cache_owner=owner)

    assert calls == list(range(1, 17))


@pytest.mark.parametrize(
    ("status", "config", "message", "level"),
    [
        (
            True,
            SimpleNamespace(solution_type="asm"),
            "selected ASM",
            logging.DEBUG,
        ),
        (
            False,
            None,
            "falling back to vLLM Triton MoE",
            logging.WARNING,
        ),
    ],
)
def test_aiter_dispatch_logs_each_problem_once_across_cache_owners(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: bool,
    config: object | None,
    message: str,
    level: int,
):
    monkeypatch.setattr(
        aiter_moe_dispatch,
        "_ROUTE_LOG_CACHE",
        OrderedDict(),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (status, config),
        ),
    )
    problem = AiterMoeProblem(
        M=3,
        E=2,
        N1=4,
        N2=2,
        K=2,
        top_k=1,
        block_size=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
        quant_type="w8a8",
    )
    owners = (torch.empty(1), torch.empty(1))

    with caplog.at_level(
        logging.DEBUG,
        logger="vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch",
    ):
        for owner in owners:
            select_aiter_moe_config(problem, cache_owner=owner)

    matching = [
        record for record in caplog.records if message in record.getMessage()
    ]
    assert len(matching) == 1
    assert matching[0].levelno == level


def test_aiter_dispatch_route_log_cache_is_atomic_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
):
    route_cache: OrderedDict[AiterMoeProblem, None] = OrderedDict()
    monkeypatch.setattr(aiter_moe_dispatch, "_ROUTE_LOG_CACHE", route_cache)
    problem = AiterMoeProblem(
        M=1,
        E=2,
        N1=4,
        N2=2,
        K=2,
        top_k=1,
        block_size=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
        quant_type="w8a8",
    )

    with ThreadPoolExecutor(max_workers=16) as pool:
        recorded = list(
            pool.map(
                lambda _: aiter_moe_dispatch._mark_route_logged(problem),
                range(64),
            )
        )

    assert sum(recorded) == 1
    limit = aiter_moe_dispatch._ROUTE_LOG_CACHE_LIMIT
    for m in range(2, limit + 2):
        assert aiter_moe_dispatch._mark_route_logged(
            replace(problem, M=m)
        )
    assert len(route_cache) == limit
    assert problem not in route_cache


@pytest.mark.parametrize(
    ("solution", "need_shuffle", "should_shuffle"),
    [
        ("asm", True, True),
        ("moe_c", True, True),
        ("triton", False, False),
        ("ck", False, False),
    ],
)
def test_aiter_dispatch_prepares_solution_layout(
    monkeypatch: pytest.MonkeyPatch,
    solution: str,
    need_shuffle: bool,
    should_shuffle: bool,
):
    calls: list[object] = []

    def shuffle(w1: torch.Tensor, w2: torch.Tensor, config: object):
        calls.append(config)
        return w1.clone(), w2.clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle),
    )
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type=solution,
        need_shuffle=need_shuffle,
        config={"layout": solution},
    )
    w1 = torch.ones((2, 8, 4))
    w2 = torch.ones((2, 4, 4))

    actual_w1, actual_w2 = prepare_aiter_moe_weights(
        w1,
        w2,
        config,
        cache_owner=w1,
    )

    assert bool(calls) is should_shuffle
    assert (actual_w1 is not w1) is should_shuffle
    assert (actual_w2 is not w2) is should_shuffle


def test_aiter_dispatch_prepares_weight_cache_tracks_generation_and_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, str]] = []

    def shuffle(w1: torch.Tensor, w2: torch.Tensor, config: object):
        calls.append(dict(config.config))
        return w1.clone(), w2.clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle),
    )
    config = SimpleNamespace(
        quant_type="w8a8",
        solution_type="asm",
        need_shuffle=True,
        config={"PADDED_K": 4},
    )
    w1 = torch.ones((2, 8, 4))
    w2 = torch.ones((2, 4, 4))

    first = prepare_aiter_moe_weights(w1, w2, config, cache_owner=w1)
    second = prepare_aiter_moe_weights(w1, w2, config, cache_owner=w1)
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert calls == [{"PADDED_K": 4}]

    w1.add_(1)
    third = prepare_aiter_moe_weights(w1, w2, config, cache_owner=w1)
    assert third[0] is not first[0]

    config.config["PADDED_K"] = 8
    fourth = prepare_aiter_moe_weights(w1, w2, config, cache_owner=w1)
    assert fourth[0] is not third[0]
    assert calls == [
        {"PADDED_K": 4},
        {"PADDED_K": 4},
        {"PADDED_K": 8},
    ]


def test_aiter_dispatch_can_preserve_weights_from_mutating_shuffle(
    monkeypatch: pytest.MonkeyPatch,
):
    def shuffle(w1: torch.Tensor, w2: torch.Tensor, config: object):
        del config
        w1.add_(1)
        w2.add_(2)
        return w1, w2

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle),
    )
    config = SimpleNamespace(
        quant_type="w4a8",
        solution_type="moe_c",
        need_shuffle=True,
        config={},
    )
    w1 = torch.ones((2, 8, 4), dtype=torch.int8)
    w2 = torch.ones((2, 4, 4), dtype=torch.int8)

    prepared_w1, prepared_w2 = prepare_aiter_moe_weights(
        w1,
        w2,
        config,
        cache_owner=w1,
        preserve_inputs=True,
    )

    torch.testing.assert_close(w1, torch.ones_like(w1))
    torch.testing.assert_close(w2, torch.ones_like(w2))
    torch.testing.assert_close(prepared_w1, torch.full_like(w1, 2))
    torch.testing.assert_close(prepared_w2, torch.full_like(w2, 3))
    assert prepared_w1 is not w1
    assert prepared_w2 is not w2


def test_aiter_dispatch_weight_cache_ignores_tuning_solution_ids(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []

    def shuffle(w1: torch.Tensor, w2: torch.Tensor, config: object):
        calls.append(config)
        return w1.clone(), w2.clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle),
    )
    first_config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="asm",
        need_shuffle=True,
        config={"SOL_ID1": 10006, "SOL_ID2": 20000, "BLOCK_SIZE_M": 16},
    )
    second_config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="asm",
        need_shuffle=True,
        config={"SOL_ID1": 11002, "SOL_ID2": 21001, "BLOCK_SIZE_M": 32},
    )
    w1 = torch.ones((2, 8, 4))
    w2 = torch.ones((2, 4, 4))

    first = prepare_aiter_moe_weights(w1, w2, first_config, cache_owner=w1)
    second = prepare_aiter_moe_weights(w1, w2, second_config, cache_owner=w1)

    assert len(calls) == 1
    assert second[0] is first[0]
    assert second[1] is first[1]


def test_aiter_dispatch_accepts_public_padded_k_weight_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    def shuffle(w1: torch.Tensor, w2: torch.Tensor, config: object):
        del config
        return torch.zeros((2, 8, 6)), torch.zeros((2, 6, 4))

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
        config={"ORIGINAL_K": 4, "PADDED_K": 6},
    )

    prepared_w1, prepared_w2 = prepare_aiter_moe_weights(
        torch.ones((2, 8, 4)),
        torch.ones((2, 4, 4)),
        config,
        cache_owner=object(),
    )

    assert prepared_w1.shape == (2, 8, 6)
    assert prepared_w2.shape == (2, 6, 4)


def test_aiter_runtime_maps_swigluoai_activation_method():
    assert aiter_runtime._activation_name(2) == "swigluoai"


def test_aiter_dispatch_prepares_scale_layout_and_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []

    def shuffle(scale1: torch.Tensor, scale2: torch.Tensor, config: object):
        calls.append(config)
        return scale1.clone(), scale2.clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_scale=shuffle),
    )
    config = SimpleNamespace(
        quant_type="w8a8",
        solution_type="asm",
        need_shuffle_scale=True,
        config={"scale_layout": 1},
    )
    scale1 = torch.ones((2, 8))
    scale2 = torch.ones((2, 4))

    first = prepare_aiter_moe_scales(
        scale1,
        scale2,
        config,
        cache_owner=scale1,
    )
    second = prepare_aiter_moe_scales(
        scale1,
        scale2,
        config,
        cache_owner=scale1,
    )

    assert first[0] is not scale1
    assert first[1] is not scale2
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert calls == [config]


@pytest.mark.parametrize("solution", ["moe_c", "triton", "ck"])
def test_aiter_dispatch_expert_map_preserves_non_asm_solutions(solution: str):
    expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int64)
    config = SimpleNamespace(solution_type=solution)

    actual = aiter_expert_map_for_solution(expert_map, config, 4)

    assert actual is expert_map


def test_aiter_dispatch_expert_map_converts_asm_to_mask():
    expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int64)
    config = SimpleNamespace(solution_type="asm")

    actual = aiter_expert_map_for_solution(expert_map, config, 4)

    torch.testing.assert_close(
        actual,
        torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32),
    )


def test_aiter_dispatch_execute_uses_public_aiter_moe(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    expected = torch.ones((2, 4))

    def aiter_moe(**kwargs: object):
        captured.update(kwargs)
        return expected

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe=aiter_moe),
    )
    hidden_states = torch.ones((2, 4))
    w1 = torch.ones((2, 8, 4))
    w2 = torch.ones((2, 4, 4))
    topk_weights = torch.ones((2, 1))
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)
    config = SimpleNamespace(solution_type="triton")

    actual = execute_aiter_moe(
        config,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        global_num_experts=2,
    )

    assert actual is expected
    assert captured["moe_config"] is config
    assert captured["w1"] is w1
    assert captured["w2"] is w2
    assert captured["global_num_experts"] == 2


def test_aiter_asm_int8_quant_context_routes_only_enabled_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object]] = []

    def native_quant(x):
        calls.append(("aiter", x))
        return "native"

    def boltops_quant(x):
        calls.append(("boltops", x))
        return "aligned"

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_int8=native_quant,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        asm_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_quant,
        ),
    )

    assert asm_module.per_token_quant_int8("before") == "native"
    with aiter_runtime.aiter_asm_boltops_int8_quant_context(enabled=True):
        assert asm_module.per_token_quant_int8("inside") == "aligned"
    assert asm_module.per_token_quant_int8("after") == "native"
    assert calls == [
        ("aiter", "before"),
        ("boltops", "inside"),
        ("aiter", "after"),
    ]


def test_aiter_asm_boltops_fp8_quant_context_aligns_both_fp8_quant_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object, object]] = []

    def native_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del scale, num_rows, num_rows_factor
        calls.append(("aiter", x, quant_dtype))
        return "aiter_quant"

    boltops_output = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    boltops_scale = torch.ones((1, 1), dtype=torch.float32)

    def boltops_quant(x, scale=None, quant_dtype=torch.int8, **kwargs):
        assert scale is None
        assert kwargs == {}
        calls.append(("boltops", x, quant_dtype))
        return boltops_output, boltops_scale

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=native_quant,
    )
    boltops_module = _module(
        "boltops.fused_moe.triton.moe_compat",
        per_token_quant_hip=boltops_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        boltops_module,
    )

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        gemm1_output = asm_module.per_token_quant_hip(
            "gemm1_input", quant_dtype=torch.float8_e4m3fn
        )
        gemm2_output = asm_module.per_token_quant_hip(
            "gemm2_input", quant_dtype=torch.float8_e4m3fn
        )
        torch.testing.assert_close(gemm1_output[0], boltops_output)
        torch.testing.assert_close(gemm1_output[1], boltops_scale)
        torch.testing.assert_close(gemm2_output[0], boltops_output)
        torch.testing.assert_close(gemm2_output[1], boltops_scale)
        assert asm_module.per_token_quant_hip(
            "int8_input", quant_dtype=torch.int8
        ) == "aiter_quant"
    assert asm_module.per_token_quant_hip(
        "outside", quant_dtype=torch.float8_e4m3fn
    ) == "aiter_quant"
    assert calls == [
        ("boltops", "gemm1_input", torch.float8_e4m3fn),
        ("boltops", "gemm2_input", torch.float8_e4m3fn),
        ("aiter", "int8_input", torch.int8),
        ("aiter", "outside", torch.float8_e4m3fn),
    ]


def test_aiter_asm_boltops_fp8_quant_repairs_zero_scale_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    quantized = torch.full((2, 4), float("nan"), dtype=torch.float8_e4m3fn)
    scales = torch.zeros((2, 1), dtype=torch.float32)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=lambda x, **kwargs: (quantized, scales),
        ),
    )

    output, output_scales = aiter_runtime._boltops_per_token_quant_fp8(
        torch.zeros((2, 4), dtype=torch.bfloat16)
    )

    torch.testing.assert_close(output.float(), torch.zeros((2, 4)))
    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(
        output_scales,
        torch.full_like(scales, 1.0e-10)
        * (1.0 / torch.finfo(torch.float8_e4m3fn).max),
    )


def test_aiter_asm_boltops_fp8_quant_context_preserves_native_activation(
    monkeypatch: pytest.MonkeyPatch,
):
    def native_activation(
        activation,
        is_gated,
        activated_out,
        ffn1_out_2d,
        gemm1_alpha,
        gemm1_limit,
    ):
        del activation, is_gated, ffn1_out_2d, gemm1_alpha, gemm1_limit
        activated_out.fill_(1)

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        _apply_activation=native_activation,
        per_token_quant_hip=_fp8_quant_abi_stub,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=lambda x, **kwargs: (x, kwargs),
        ),
    )
    output = torch.empty((2, 4))

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        assert asm_module._apply_activation is native_activation
        asm_module._apply_activation(
            "silu", True, output, torch.empty((2, 8)), None, None
        )
    torch.testing.assert_close(output, torch.ones_like(output))


def test_aiter_asm_boltops_fp8_quant_context_nested_disable_restores_state(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def native_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del x, scale, quant_dtype, num_rows, num_rows_factor
        calls.append("aiter")
        return "aiter"

    boltops_output = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    boltops_scale = torch.ones((1, 1), dtype=torch.float32)

    def boltops_quant(x, **kwargs):
        del x, kwargs
        calls.append("boltops")
        return boltops_output, boltops_scale

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=native_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_quant,
        ),
    )

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        torch.testing.assert_close(
            asm_module.per_token_quant_hip(
                "outer", quant_dtype=torch.float8_e4m3fn
            )[0],
            boltops_output,
        )
        with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=False):
            assert asm_module.per_token_quant_hip(
                "disabled", quant_dtype=torch.float8_e4m3fn
            ) == "aiter"
        with pytest.raises(RuntimeError, match="cleanup"):
            with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
                raise RuntimeError("cleanup")
        torch.testing.assert_close(
            asm_module.per_token_quant_hip(
                "outer-again", quant_dtype=torch.float8_e4m3fn
            )[0],
            boltops_output,
        )

    assert asm_module.per_token_quant_hip(
        "outside", quant_dtype=torch.float8_e4m3fn
    ) == "aiter"
    assert calls == ["boltops", "aiter", "boltops", "aiter"]


@pytest.mark.parametrize(
    "quant_kwargs",
    [
        {"quant_dtype": torch.int8},
        {"quant_dtype": torch.float8_e4m3fn, "scale": torch.ones(1)},
        {"quant_dtype": torch.float8_e4m3fn, "num_rows": torch.ones(1)},
        {"quant_dtype": torch.float8_e4m3fn, "num_rows_factor": 2},
    ],
)
def test_aiter_asm_boltops_fp8_quant_context_preserves_unsupported_quant_modes(
    monkeypatch: pytest.MonkeyPatch,
    quant_kwargs: dict[str, object],
):
    calls: list[str] = []

    def native_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del x, scale, quant_dtype, num_rows, num_rows_factor
        calls.append("aiter")
        return "aiter"

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=native_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=lambda x, **kwargs: "boltops",
        ),
    )

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        assert asm_module.per_token_quant_hip("input", **quant_kwargs) == "aiter"
    assert calls == ["aiter"]


def test_aiter_asm_boltops_fp8_quant_context_rejects_incompatible_quant_abi(
    monkeypatch: pytest.MonkeyPatch,
):
    def incompatible_quant(x):
        return x

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=incompatible_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="per_token_quant_hip exposes unsupported arguments",
    ):
        with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
            pass
    assert asm_module.per_token_quant_hip is incompatible_quant


@pytest.mark.parametrize("abi_change", ["extra_parameter", "changed_default"])
def test_aiter_asm_boltops_fp8_quant_context_rejects_subtle_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
    abi_change: str,
):
    if abi_change == "extra_parameter":

        def incompatible_quant(
            x,
            scale=None,
            quant_dtype=torch.int8,
            num_rows=None,
            num_rows_factor=1,
            stochastic=False,
        ):
            del x, scale, quant_dtype, num_rows, num_rows_factor, stochastic

    else:

        def incompatible_quant(
            x,
            scale=None,
            quant_dtype=torch.int8,
            num_rows=None,
            num_rows_factor=2,
        ):
            del x, scale, quant_dtype, num_rows, num_rows_factor

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=incompatible_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="per_token_quant_hip exposes unsupported",
    ):
        with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
            pass
    assert asm_module.per_token_quant_hip is incompatible_quant


def test_int8_oracle_keeps_aiter_weights_and_packs_hcu_deep_gemm_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.patch.worker.op_opt.moe import patch_int8_oracle
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    import vllm.config as vllm_config_module

    class Int8MoeBackend(enum.Enum):
        TRITON = "TRITON"
        HUMMING = "HUMMING"
        CPU = "CPU"

    class AiterExperts:
        pass

    def backend_to_kernel_cls(backend):
        return [f"original:{backend.value}"]

    def map_int8_backend(runner_backend):
        if runner_backend == "triton":
            return Int8MoeBackend.TRITON
        raise ValueError(runner_backend)

    def select_int8_moe_backend(config, weight_key, activation_key):
        del config, weight_key, activation_key
        return "official-select", "official-experts"

    def convert_to_int8_moe_kernel_format(
        int8_backend,
        w13,
        w2,
        layer=None,
        w13_scale=None,
    ):
        del layer, w13_scale
        if int8_backend != Int8MoeBackend.TRITON:
            raise ValueError(int8_backend)
        return w13 + 1, w2 + 1

    def make_int8_moe_quant_config(
        int8_backend,
        w1_scale,
        w2_scale,
        a1_scale=None,
        a2_scale=None,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
        layer=None,
    ):
        del (
            int8_backend,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            w1_bias,
            w2_bias,
            per_act_token_quant,
            layer,
        )
        return "original-w8a16"

    def int8_w8a8_moe_quant_config(
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
        gemm1_clamp_limit=None,
    ):
        return SimpleNamespace(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            per_act_token_quant=per_act_token_quant,
            gemm1_clamp_limit=gemm1_clamp_limit,
            use_int8_w8a8=True,
        )

    def make_int8_moe_kernel(
        int8_backend,
        moe_quant_config,
        moe_config,
        experts_cls,
        routing_tables=None,
    ):
        del (
            int8_backend,
            moe_quant_config,
            moe_config,
            experts_cls,
            routing_tables,
        )
        return "official-int8-kernel"

    target = _module(
        patch_int8_oracle.TARGET_MODULE,
        Enum=enum.Enum,
        Int8MoeBackend=Int8MoeBackend,
        backend_to_kernel_cls=backend_to_kernel_cls,
        map_int8_backend=map_int8_backend,
        select_int8_moe_backend=select_int8_moe_backend,
        convert_to_int8_moe_kernel_format=convert_to_int8_moe_kernel_format,
        make_int8_moe_quant_config=make_int8_moe_quant_config,
        int8_w8a8_moe_quant_config=int8_w8a8_moe_quant_config,
        make_int8_moe_kernel=make_int8_moe_kernel,
        mk=SimpleNamespace(
            FusedMoEActivationFormat=SimpleNamespace(
                Standard="standard",
                BatchedExperts="batched",
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
        _module(
            "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
            AiterExperts=AiterExperts,
        ),
    )

    class DeepGemmExperts:
        @staticmethod
        def is_supported_config(
            cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        ):
            del cls, config
            return (
                (weight_key, activation_key, activation_format)
                == ("weight", "activation", "standard"),
                "requires standard activation format",
            )

    class BatchedDeepGemmExperts:
        @staticmethod
        def is_supported_config(
            cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        ):
            del cls, config
            return (
                (weight_key, activation_key, activation_format)
                == ("weight", "activation", "batched"),
                "requires batched activation format",
            )

    class DeepEPAutoInt8Experts:
        pass

    auto_kernel_calls: list[tuple[object, object, object]] = []
    auto_processed_layers: list[object] = []

    class AutoExperts:
        def process_weights_after_loading(self, layer):
            auto_processed_layers.append(layer)

    auto_kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(experts=AutoExperts())
    )

    def make_deepep_auto_deepgemm_int8_moe_kernel(
        *, moe_quant_config, moe_config, routing_tables
    ):
        auto_kernel_calls.append(
            (moe_quant_config, moe_config, routing_tables)
        )
        return auto_kernel

    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.model_executor.layers.fused_moe.experts.deep_gemm_moe",
        _module(
            "vllm_hcu.model_executor.layers.fused_moe.experts.deep_gemm_moe",
            DeepGemmExperts=DeepGemmExperts,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe",
        _module(
            "vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe",
            BatchedDeepGemmExperts=BatchedDeepGemmExperts,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        (
            "vllm_hcu.model_executor.layers.fused_moe.experts."
            "dpsk_v4_deep_gemm_moe"
        ),
        _module(
            (
                "vllm_hcu.model_executor.layers.fused_moe.experts."
                "dpsk_v4_deep_gemm_moe"
            ),
            DeepEPDeepGemmContiguousExperts=DeepEPAutoInt8Experts,
            make_deepep_auto_deepgemm_int8_moe_kernel=(
                make_deepep_auto_deepgemm_int8_moe_kernel
            ),
        ),
    )
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config_or_none",
        lambda: None,
    )

    assert patch_int8_oracle.apply_to_module(target) is True
    assert target.map_int8_backend("aiter") == target.Int8MoeBackend.AITER
    assert target.map_int8_backend("triton").value == "TRITON"
    assert target.backend_to_kernel_cls(target.Int8MoeBackend.AITER) == [
        AiterExperts
    ]

    w13 = torch.zeros((2, 4, 3), dtype=torch.int8)
    w2 = torch.zeros((2, 3, 2), dtype=torch.int8)
    converted_w13, converted_w2 = target.convert_to_int8_moe_kernel_format(
        target.Int8MoeBackend.AITER,
        w13,
        w2,
    )
    assert converted_w13 is w13
    assert converted_w2 is w2

    w1_scale = torch.ones((2, 4, 1))
    w2_scale = torch.ones((2, 3, 1))
    quant_config = target.make_int8_moe_quant_config(
        target.Int8MoeBackend.AITER,
        w1_scale,
        w2_scale,
        per_act_token_quant=True,
    )
    assert quant_config.use_int8_w8a8 is True
    assert quant_config.per_act_token_quant is True
    assert quant_config.a1_scale is None
    assert quant_config.a2_scale is None
    aiter_moe_config = SimpleNamespace(
        experts_per_token=2,
        in_dtype=torch.bfloat16,
        activation=SimpleNamespace(value="silu"),
        moe_parallel_config=SimpleNamespace(
            use_deepep_auto_kernels=False,
        ),
    )
    assert target.make_int8_moe_kernel(
        target.Int8MoeBackend.AITER,
        quant_config,
        aiter_moe_config,
        AiterExperts,
    ) == "official-int8-kernel"

    config = SimpleNamespace(
        moe_backend="deep_gemm",
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=False,
        ),
        _hcu_vllm_config=SimpleNamespace(
            additional_config={
                "hcu": {
                    "moe_backend": "deep_gemm",
                }
            },
            model_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
            ),
        ),
    )
    backend, experts = target.select_int8_moe_backend(
        config, "weight", "activation"
    )
    assert backend is target.Int8MoeBackend.HCU_DEEPGEMM
    assert experts is DeepGemmExperts
    assert target.map_int8_backend("deep_gemm") is backend
    deep_gemm_quant_config = target.make_int8_moe_quant_config(
        backend,
        w1_scale,
        w2_scale,
        per_act_token_quant=True,
        layer=SimpleNamespace(swiglu_limit=10.0),
    )
    assert getattr(deep_gemm_quant_config, "use_int8_w8a8", False) is True
    assert deep_gemm_quant_config.per_act_token_quant is True
    assert deep_gemm_quant_config.gemm1_clamp_limit == 10.0

    config.moe_parallel_config.use_batched_activation_format = True
    backend, experts = target.select_int8_moe_backend(
        config, "weight", "activation"
    )
    assert experts is BatchedDeepGemmExperts

    config.moe_backend = "auto"
    config._hcu_vllm_config.additional_config["hcu"].update(
        deepep_auto=True,
        moe_backend="auto",
    )
    auto_backend, auto_experts = target.select_int8_moe_backend(
        config,
        kInt8StaticChannelSym,
        kInt8DynamicTokenSym,
    )
    assert auto_backend is target.Int8MoeBackend.HCU_DEEPGEMM
    assert auto_experts is DeepEPAutoInt8Experts
    config.moe_parallel_config.use_deepep_auto_kernels = True
    auto_quant_config = SimpleNamespace()
    assert target.make_int8_moe_kernel(
        auto_backend,
        auto_quant_config,
        config,
        auto_experts,
        routing_tables="routing",
    ) is auto_kernel
    assert auto_kernel_calls == [(auto_quant_config, config, "routing")]
    assert auto_processed_layers == []

    config._hcu_vllm_config.model_config.architectures = [
        "GlmMoeDsaForCausalLM"
    ]
    with pytest.raises(ValueError, match="DeepSeek-V4"):
        target.select_int8_moe_backend(
            config,
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )
    config._hcu_vllm_config.model_config.architectures = [
        "DeepseekV4ForCausalLM"
    ]
    config._hcu_vllm_config.additional_config["hcu"]["deepep_auto"] = False
    config.moe_parallel_config.use_deepep_auto_kernels = False
    config._hcu_vllm_config.additional_config["hcu"]["moe_backend"] = "deep_gemm"
    config.moe_backend = "deep_gemm"

    deep_gemm_w13 = torch.arange(2 * 16 * 64, dtype=torch.int32).to(torch.int8)
    deep_gemm_w13 = deep_gemm_w13.reshape(2, 16, 64)
    deep_gemm_w2 = torch.arange(2 * 64 * 64, dtype=torch.int32).to(torch.int8)
    deep_gemm_w2 = deep_gemm_w2.reshape(2, 64, 64)
    deepgemm = _module("deepgemm")
    m_group_gemm = _module("deepgemm.m_group_gemm")

    def real_contiguous_pack(weight: torch.Tensor) -> torch.Tensor:
        return weight.flip(-1).clone()

    def real_masked_pack(weight: torch.Tensor) -> torch.Tensor:
        return weight.flip(-2).clone()

    deepgemm.marlin_i8_contiguous_weight = real_contiguous_pack
    deepgemm.marlin_i8_masked_weight = real_masked_pack
    deepgemm.m_group_gemm = m_group_gemm
    m_group_gemm.pack_int8_weight_enk_to_w6_low_latency = lambda weight: weight
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)
    monkeypatch.setitem(sys.modules, "deepgemm.m_group_gemm", m_group_gemm)

    contiguous_calls: list[torch.Tensor] = []
    masked_calls: list[torch.Tensor] = []

    def tracked_contiguous_pack(weight: torch.Tensor) -> torch.Tensor:
        contiguous_calls.append(weight)
        return real_contiguous_pack(weight)

    def tracked_masked_pack(weight: torch.Tensor) -> torch.Tensor:
        masked_calls.append(weight)
        return real_masked_pack(weight)

    def reject_w6_packer(*_args, **_kwargs):
        raise AssertionError("W6 low-latency packer invoked")

    monkeypatch.setattr(
        deepgemm,
        "marlin_i8_contiguous_weight",
        tracked_contiguous_pack,
    )
    monkeypatch.setattr(
        deepgemm,
        "marlin_i8_masked_weight",
        tracked_masked_pack,
    )
    monkeypatch.setattr(
        m_group_gemm,
        "pack_int8_weight_enk_to_w6_low_latency",
        reject_w6_packer,
    )

    auto_layout_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=True,
                use_deepep_auto_kernels=True,
            )
        )
    )
    auto_w13, auto_w2 = target.convert_to_int8_moe_kernel_format(
        backend,
        deep_gemm_w13,
        deep_gemm_w2,
        layer=auto_layout_layer,
    )
    assert auto_w13 is deep_gemm_w13
    assert auto_w2 is deep_gemm_w2
    assert contiguous_calls == []
    assert masked_calls == []

    standard_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=False,
            )
        )
    )
    expected_standard_w13 = real_contiguous_pack(deep_gemm_w13.clone())
    expected_standard_w2 = real_contiguous_pack(deep_gemm_w2.clone())
    standard_w13, standard_w2 = target.convert_to_int8_moe_kernel_format(
        backend,
        deep_gemm_w13,
        deep_gemm_w2,
        layer=standard_layer,
    )
    assert contiguous_calls == [deep_gemm_w13, deep_gemm_w2]
    assert masked_calls == []
    torch.testing.assert_close(standard_w13, expected_standard_w13)
    torch.testing.assert_close(standard_w2, expected_standard_w2)

    contiguous_calls.clear()
    batched_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=True,
            )
        )
    )
    expected_batched_w13 = real_masked_pack(deep_gemm_w13.clone())
    expected_batched_w2 = real_masked_pack(deep_gemm_w2.clone())
    batched_w13, batched_w2 = target.convert_to_int8_moe_kernel_format(
        backend,
        deep_gemm_w13,
        deep_gemm_w2,
        layer=batched_layer,
    )
    assert contiguous_calls == []
    assert masked_calls == [deep_gemm_w13, deep_gemm_w2]
    torch.testing.assert_close(batched_w13, expected_batched_w13)
    torch.testing.assert_close(batched_w2, expected_batched_w2)

    config._hcu_vllm_config.additional_config["hcu"]["moe_backend"] = "auto"
    assert target.select_int8_moe_backend(
        config, "weight", "activation"
    ) == ("official-select", "official-experts")

    config._hcu_vllm_config.additional_config["hcu"][
        "moe_backend"
    ] = "deep_gemm"
    config.moe_backend = "triton"
    with pytest.raises(ValueError, match="official backend must match 'deep_gemm'"):
        target.select_int8_moe_backend(config, "weight", "activation")


def test_int8_oracle_reports_quant_config_target_for_abi_drift():
    from vllm_hcu.patch.worker.op_opt.moe import patch_int8_oracle

    class Int8MoeBackend(enum.Enum):
        TRITON = "TRITON"

    def backend_to_kernel_cls(backend):
        del backend

    def map_int8_backend(runner_backend):
        del runner_backend

    def select_int8_moe_backend(config, weight_key, activation_key):
        del config, weight_key, activation_key

    def convert_to_int8_moe_kernel_format(
        int8_backend,
        w13,
        w2,
        layer,
        w13_scale,
    ):
        del int8_backend, w13, w2, layer, w13_scale

    def make_int8_moe_quant_config(incompatible_parameter):
        del incompatible_parameter

    def make_int8_moe_kernel(
        int8_backend,
        moe_quant_config,
        moe_config,
        experts_cls,
        routing_tables=None,
        layer=None,
    ):
        del (
            int8_backend,
            moe_quant_config,
            moe_config,
            experts_cls,
            routing_tables,
            layer,
        )

    target = _module(
        patch_int8_oracle.TARGET_MODULE,
        Int8MoeBackend=Int8MoeBackend,
        backend_to_kernel_cls=backend_to_kernel_cls,
        map_int8_backend=map_int8_backend,
        select_int8_moe_backend=select_int8_moe_backend,
        convert_to_int8_moe_kernel_format=convert_to_int8_moe_kernel_format,
        make_int8_moe_quant_config=make_int8_moe_quant_config,
        int8_w8a8_moe_quant_config=lambda: None,
        make_int8_moe_kernel=make_int8_moe_kernel,
    )

    with pytest.raises(RuntimeError) as error:
        patch_int8_oracle.apply_to_module(target)

    assert patch_int8_oracle.TARGETS[5] in str(error.value)
    assert patch_int8_oracle.TARGETS[4] not in str(error.value)


def test_worker_registers_int8_aiter_oracle_before_quantized_methods():
    from vllm_hcu.patch import worker

    callbacks = worker.worker_callback_names()
    int8_entry = (
        "worker.op_opt.moe.oracle.int8_aiter",
        "vllm.model_executor.layers.fused_moe.oracle.int8",
    )
    fp8_method_entry = (
        "worker.op_opt.compressed_tensors.moe_w8a8_fp8",
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8",
    )
    assert int8_entry in callbacks
    assert callbacks.index(int8_entry) < callbacks.index(fp8_method_entry)


def test_hcu_deep_gemm_experts_accept_rocm_lightop_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.model_executor.layers.fused_moe  # noqa: F401

    from vllm_hcu.model_executor.layers.fused_moe.experts import (
        batched_deep_gemm_moe,
        deep_gemm_moe,
    )

    monkeypatch.setattr(
        deep_gemm_moe.current_platform,
        "is_rocm",
        lambda: True,
    )
    monkeypatch.setattr(
        deep_gemm_moe,
        "is_deep_gemm_supported",
        lambda: False,
    )
    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "is_deep_gemm_supported",
        lambda: False,
    )

    assert deep_gemm_moe.DeepGemmExperts._supports_current_device()
    assert batched_deep_gemm_moe.BatchedDeepGemmExperts._supports_current_device()


def _install_fake_vllm_envs(
    monkeypatch: pytest.MonkeyPatch,
    **attributes: object,
) -> ModuleType:
    """Install a complete fake package edge for ``import vllm.envs``.

    Adding only the child to ``sys.modules`` still makes Python import the real
    parent package.  That both defeats isolation and can leave a partially
    initialized ``vllm`` behind for later tests when the child is deliberately
    minimal.
    """

    envs = _module("vllm.envs", **attributes)
    vllm = _package("vllm", envs=envs)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    return envs


def _install_fake_vllm_torch_utils(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Provide only the custom-op registration dependency under test."""

    def direct_register_custom_op(**kwargs):
        return kwargs

    torch_utils = _module(
        "vllm.utils.torch_utils",
        direct_register_custom_op=direct_register_custom_op,
    )
    utils = _package("vllm.utils", torch_utils=torch_utils)
    vllm = _package("vllm", utils=utils)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    return direct_register_custom_op


def test_aiter_gfx93x_capability_extends_upstream(monkeypatch: pytest.MonkeyPatch):
    fake_hcu = _module("vllm_hcu.platforms.hcu", on_gfx93x=lambda: True)
    monkeypatch.setitem(sys.modules, "vllm_hcu.platforms.hcu", fake_hcu)
    platform = SimpleNamespace(is_rocm=lambda: True)
    assert aiter_runtime.is_aiter_found_and_supported(
        lambda: False, platform, True
    )
    assert not aiter_runtime.is_aiter_found_and_supported(
        lambda: False, platform, False
    )


class _QuantType(enum.IntEnum):
    No = 0
    Other = 1


def _install_fake_aiter(
    monkeypatch: pytest.MonkeyPatch, **attributes: object
) -> ModuleType:
    module = _module("aiter", QuantType=_QuantType, **attributes)
    module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiter", module)
    return module


def test_aiter_w8a8_tuning_capability_requires_runtime_and_target_config(
    monkeypatch: pytest.MonkeyPatch,
):
    aiter = _install_fake_aiter(
        monkeypatch,
        gemm_a8w8_bpreshuffle=lambda: None,
        gemm_a8w8_CK=lambda: None,
    )
    configs = SimpleNamespace(
        AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE="/configs/preshuffle.csv",
        AITER_CONFIG_GEMM_A8W8_FILE="/configs/per-token.csv",
    )
    monkeypatch.setitem(sys.modules, "aiter.ops", _package("aiter.ops"))
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.gemm_op_a8w8",
        _module("aiter.ops.gemm_op_a8w8", AITER_CONFIGS=configs),
    )

    assert aiter_runtime.get_w8a8_tuned_config_path(
        "gemm_a8w8_bpreshuffle",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
    ) == "/configs/preshuffle.csv"
    assert aiter_runtime.get_w8a8_tuned_config_path(
        "gemm_a8w8_CK",
        "AITER_CONFIG_GEMM_A8W8_FILE",
    ) == "/configs/per-token.csv"

    delattr(aiter, "gemm_a8w8_bpreshuffle")
    assert (
        aiter_runtime.get_w8a8_tuned_config_path(
            "gemm_a8w8_bpreshuffle",
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        )
        is None
    )


def test_aiter_w8a8_tuning_capability_fails_closed_on_expected_api_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch, gemm_a8w8_CK=lambda: None)
    monkeypatch.setitem(sys.modules, "aiter.ops", _package("aiter.ops"))

    for name, gemm_module in (
        ("missing-config-owner", _module("aiter.ops.gemm_op_a8w8")),
        (
            "missing-config-attribute",
            _module(
                "aiter.ops.gemm_op_a8w8",
                AITER_CONFIGS=SimpleNamespace(),
            ),
        ),
        (
            "invalid-empty-path",
            _module(
                "aiter.ops.gemm_op_a8w8",
                AITER_CONFIGS=SimpleNamespace(AITER_CONFIG_GEMM_A8W8_FILE=""),
            ),
        ),
    ):
        monkeypatch.setitem(sys.modules, "aiter.ops.gemm_op_a8w8", gemm_module)
        assert (
            aiter_runtime.get_w8a8_tuned_config_path(
                "gemm_a8w8_CK",
                "AITER_CONFIG_GEMM_A8W8_FILE",
            )
            is None
        ), name


def test_aiter_w8a8_tuning_capability_does_not_hide_unexpected_abi_error(
    monkeypatch: pytest.MonkeyPatch,
):
    aiter = _module("aiter", gemm_a8w8_CK=lambda: None)

    for error in (
        OSError("unexpected AITER loader ABI error"),
        ImportError("undefined symbol: proprietary_aiter_abi"),
        AttributeError("unexpected module initialization failure"),
    ):
        def fake_import(name: str, *, failure=error):
            if name == "aiter":
                return aiter
            raise failure

        monkeypatch.setattr(aiter_runtime, "import_module", fake_import)
        with pytest.raises(type(error), match=str(error)):
            aiter_runtime.get_w8a8_tuned_config_path(
                "gemm_a8w8_CK",
                "AITER_CONFIG_GEMM_A8W8_FILE",
            )


def test_aiter_w8a8_tuning_capability_short_circuits_broken_submodule(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if name == "aiter":
            return _module("aiter")
        raise AssertionError("the unavailable runtime must short-circuit")

    monkeypatch.setattr(aiter_runtime, "import_module", fake_import)
    assert (
        aiter_runtime.get_w8a8_tuned_config_path(
            "gemm_a8w8_CK",
            "AITER_CONFIG_GEMM_A8W8_FILE",
        )
        is None
    )
    assert calls == ["aiter"]


def test_optional_aiter_module_distinguishes_absence_from_transitive_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    requested = "aiter.ops.gemm_op_a8w8"

    def missing_requested(name: str):
        raise ModuleNotFoundError(
            f"No module named {name!r}",
            name=name,
        )

    monkeypatch.setattr(aiter_runtime, "import_module", missing_requested)
    assert aiter_runtime._import_optional_aiter_module(requested) is None

    def missing_transitive(name: str):
        raise ModuleNotFoundError(
            "No module named 'proprietary_abi_dependency'",
            name="proprietary_abi_dependency",
        )

    monkeypatch.setattr(aiter_runtime, "import_module", missing_transitive)
    with pytest.raises(ModuleNotFoundError, match="proprietary_abi_dependency"):
        aiter_runtime._import_optional_aiter_module(requested)


def test_aiter_triton_fp8_bmm_capability_is_symbol_aware(
    monkeypatch: pytest.MonkeyPatch,
):
    symbol = aiter_runtime._AITER_TRITON_FP8_BMM_SYMBOL
    module = _module(
        aiter_runtime._AITER_TRITON_FP8_BMM_MODULE,
        **{symbol: lambda: None},
    )
    monkeypatch.setattr(aiter_runtime, "import_module", lambda name: module)
    assert aiter_runtime.has_triton_fp8_bmm()

    delattr(module, symbol)
    assert not aiter_runtime.has_triton_fp8_bmm()

    def missing_requested(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(aiter_runtime, "import_module", missing_requested)
    assert not aiter_runtime.has_triton_fp8_bmm()

    def missing_transitive(name: str):
        raise ModuleNotFoundError(
            "No module named 'triton_runtime_abi'",
            name="triton_runtime_abi",
        )

    monkeypatch.setattr(aiter_runtime, "import_module", missing_transitive)
    with pytest.raises(ModuleNotFoundError, match="triton_runtime_abi"):
        aiter_runtime.has_triton_fp8_bmm()


def test_aiter_triton_fp8_bmm_env_gates_short_circuit_capability_probe(
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_probe():
        raise AssertionError("disabled environment gates must short-circuit")

    monkeypatch.setattr(aiter_runtime, "has_triton_fp8_bmm", unexpected_probe)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(False, False)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(False, True)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(True, False)

    monkeypatch.setattr(aiter_runtime, "has_triton_fp8_bmm", lambda: False)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(True, True)
    monkeypatch.setattr(aiter_runtime, "has_triton_fp8_bmm", lambda: True)
    assert aiter_runtime.is_triton_fp8_bmm_enabled(True, True)


def _operator_schema(*names: str) -> SimpleNamespace:
    return SimpleNamespace(
        arguments=[SimpleNamespace(name=name) for name in names]
    )


@pytest.mark.parametrize(
    ("op_name", "legacy_arguments"),
    tuple(aiter_runtime._AITER_RMSNORM_DYNAMIC_QUANT_ARGUMENTS.items()),
)
@pytest.mark.parametrize(
    ("profile", "suffix"),
    (
        ("legacy-default", ()),
        (
            "model-sensitive",
            (aiter_runtime._AITER_MODEL_SENSITIVE_RMSNORM_ARGUMENT,),
        ),
    ),
)
def test_aiter_rmsnorm_dynamic_quant_abi_requires_exact_known_schema(
    monkeypatch: pytest.MonkeyPatch,
    op_name: str,
    legacy_arguments: tuple[str, ...],
    profile: str,
    suffix: tuple[str, ...],
):
    overload = SimpleNamespace(
        _schema=_operator_schema(*(legacy_arguments + suffix))
    )
    namespace = SimpleNamespace(
        **{op_name: SimpleNamespace(default=overload)}
    )
    monkeypatch.setattr(
        aiter_runtime,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(aiter=namespace)),
    )

    assert aiter_runtime._aiter_rmsnorm_dynamic_quant_abi(op_name) == profile


@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("a0", "a1", "a2", "a3", "a4"),
        ("input", "out", "yscale", "weight", "epsilon"),
        ("out", "input", "yscale", "weight", "epsilon", "unexpected"),
    ),
)
def test_aiter_rmsnorm_dynamic_quant_abi_fails_closed_on_unknown_schema(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
):
    op_name = "rmsnorm2d_fwd_with_dynamicquant"
    overload = SimpleNamespace(_schema=_operator_schema(*arguments))
    namespace = SimpleNamespace(
        **{op_name: SimpleNamespace(default=overload)}
    )
    monkeypatch.setattr(
        aiter_runtime,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(aiter=namespace)),
    )

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="no readable operator schema|unsupported arguments",
    ):
        aiter_runtime._aiter_rmsnorm_dynamic_quant_abi(op_name)


def test_aiter_rmsnorm_dynamic_quant_abi_fails_closed_when_op_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        aiter_runtime,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(aiter=SimpleNamespace())),
    )
    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="no readable operator schema",
    ):
        aiter_runtime._aiter_rmsnorm_dynamic_quant_abi(
            "rmsnorm2d_fwd_with_dynamicquant"
        )


@pytest.mark.parametrize("fused_add", [False, True])
@pytest.mark.parametrize(
    ("profile", "expected_kwargs"),
    (
        ("legacy-default", {}),
        ("model-sensitive", {"use_model_sensitive_rmsnorm": 0}),
    ),
)
def test_aiter_rmsnorm_int8_calls_each_supported_abi_once(
    monkeypatch: pytest.MonkeyPatch,
    fused_add: bool,
    profile: str,
    expected_kwargs: dict[str, int],
):
    monkeypatch.setattr(
        aiter_runtime,
        "_aiter_rmsnorm_dynamic_quant_abi",
        lambda op_name: profile,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def operation(*args, **kwargs):
        calls.append((args, kwargs))

    x = torch.ones(2, 4)
    weight = torch.ones(4)
    if fused_add:
        result = aiter_runtime.rmsnorm_add_dynamic_quant_impl(
            operation,
            x,
            torch.full_like(x, 2),
            weight,
            1e-6,
            torch.int8,
        )
        assert len(result) == 3
        assert len(calls[0][0]) == 7
    else:
        result = aiter_runtime.rmsnorm_dynamic_quant_impl(
            operation, x, weight, 1e-6, torch.int8
        )
        assert len(result) == 2
        assert len(calls[0][0]) == 5
    assert len(calls) == 1
    assert calls[0][1] == expected_kwargs


@pytest.mark.parametrize("fused_add", [False, True])
def test_legacy_aiter_rmsnorm_fp8_uses_vllm_native_fallback_only(
    monkeypatch: pytest.MonkeyPatch,
    fused_add: bool,
):
    monkeypatch.setattr(
        aiter_runtime,
        "_aiter_rmsnorm_dynamic_quant_abi",
        lambda op_name: "legacy-default",
    )
    native_calls: list[tuple[object, ...]] = []

    def native(x, weight, epsilon, quant_dtype, residual=None):
        native_calls.append((x, weight, epsilon, quant_dtype, residual))
        output = torch.empty_like(x, dtype=quant_dtype)
        scale = torch.empty(x.shape[0], 1)
        residual_out = residual.clone() if residual is not None else None
        return output, scale, residual_out

    monkeypatch.setattr(
        aiter_runtime,
        "_vllm_native_rmsnorm_dynamic_quant",
        native,
    )

    def unexpected_aiter(*args, **kwargs):
        raise AssertionError("legacy AITER FP8 kernel must not run")

    x = torch.ones(2, 4)
    weight = torch.ones(4)
    if fused_add:
        residual = torch.full_like(x, 2)
        output, residual_out, scale = (
            aiter_runtime.rmsnorm_add_dynamic_quant_impl(
                unexpected_aiter,
                x,
                residual,
                weight,
                1e-6,
                torch.float8_e4m3fn,
            )
        )
        assert native_calls[0][4] is residual
        assert residual_out is not residual
        torch.testing.assert_close(residual_out, residual)
    else:
        output, scale = aiter_runtime.rmsnorm_dynamic_quant_impl(
            unexpected_aiter,
            x,
            weight,
            1e-6,
            torch.float8_e4m3fn,
        )
        assert native_calls[0][4] is None
    assert output.dtype is torch.float8_e4m3fn
    assert scale.shape == (2, 1)
    assert len(native_calls) == 1


def test_vllm_native_rmsnorm_fallback_validates_schema_and_clones_residual(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []

    class NativeOperation:
        _schema = _operator_schema(
            *aiter_runtime._VLLM_NATIVE_RMSNORM_DYNAMIC_QUANT_ARGUMENTS
        )

        def __call__(self, *args):
            calls.append(args)
            output, x, _weight, scale, _epsilon, scale_ub, residual = args
            output.fill_(1)
            scale.fill_(2)
            assert scale_ub is None
            if residual is not None:
                residual.add_(x)

    torch_proxy = SimpleNamespace(
        empty=torch.empty,
        float32=torch.float32,
        ops=SimpleNamespace(
            _C=SimpleNamespace(
                rms_norm_dynamic_per_token_quant=SimpleNamespace(
                    default=NativeOperation()
                )
            )
        ),
    )
    monkeypatch.setattr(aiter_runtime, "torch", torch_proxy)
    monkeypatch.setattr(aiter_runtime, "import_module", lambda name: object())

    x = torch.ones(2, 4)
    residual = torch.full_like(x, 2)
    original_x = x.clone()
    original_residual = residual.clone()
    output, scale, residual_out = (
        aiter_runtime._vllm_native_rmsnorm_dynamic_quant(
            x,
            torch.ones(4),
            1e-6,
            torch.float8_e4m3fn,
            residual,
        )
    )

    assert len(calls) == 1
    assert output.dtype is torch.float8_e4m3fn
    assert torch.all(scale == 2)
    assert residual_out is not None and residual_out is not residual
    torch.testing.assert_close(residual_out, original_x + original_residual)
    torch.testing.assert_close(x, original_x)
    torch.testing.assert_close(residual, original_residual)


@pytest.mark.parametrize("fused_add", [False, True])
def test_aiter_rmsnorm_kernel_errors_propagate_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    fused_add: bool,
):
    monkeypatch.setattr(
        aiter_runtime,
        "_aiter_rmsnorm_dynamic_quant_abi",
        lambda op_name: "model-sensitive",
    )
    calls = 0

    def broken_operation(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("AITER kernel execution failed")

    x = torch.ones(2, 4)
    if fused_add:
        call = lambda: aiter_runtime.rmsnorm_add_dynamic_quant_impl(
            broken_operation,
            x,
            torch.ones_like(x),
            torch.ones(4),
            1e-6,
            torch.int8,
        )
    else:
        call = lambda: aiter_runtime.rmsnorm_dynamic_quant_impl(
            broken_operation,
            x,
            torch.ones(4),
            1e-6,
            torch.int8,
        )
    with pytest.raises(RuntimeError, match="AITER kernel execution failed"):
        call()
    assert calls == 1


def test_aiter_replacement_rmsnorm_wrappers_delegate_without_retry_logic():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    expected = {
        "_rocm_aiter_rmsnorm_fused_dynamic_quant_impl": (
            "rmsnorm_dynamic_quant_impl"
        ),
        "_rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl": (
            "rmsnorm_add_dynamic_quant_impl"
        ),
    }
    for function_name, runtime_name in expected.items():
        function = functions[function_name]
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == runtime_name
        ]
        assert len(calls) == 1, function_name
        assert not any(isinstance(node, ast.Try) for node in ast.walk(function))


def test_aiter_replacement_maps_each_optional_capability_exactly():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "rocm_aiter_ops"
    )
    methods = {
        node.name: node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
    }
    expected = {
        "is_shuffled_per_token_w8a8_gemm_tuned": (
            "gemm_a8w8_bpreshuffle",
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        ),
        "is_per_token_w8a8_gemm_tuned": (
            "gemm_a8w8_CK",
            "AITER_CONFIG_GEMM_A8W8_FILE",
        ),
    }
    for method_name, expected_arguments in expected.items():
        calls = [
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_w8a8_tuned_config_path"
        ]
        assert len(calls) == 1, method_name
        assert tuple(ast.literal_eval(arg) for arg in calls[0].args) == expected_arguments
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_check_kernel_tuned"
            for node in ast.walk(methods[method_name])
        ), method_name

    fp8_bmm_source = ast.unparse(methods["is_fp8bmm_enabled"])
    assert "_hcu_runtime.is_triton_fp8_bmm_enabled" in fp8_bmm_source
    fused_moe_source = ast.unparse(methods["is_fused_moe_enabled"])
    assert "_hcu_runtime.is_aiter_moe_requested()" in fused_moe_source


def test_aiter_replacement_uses_workspace_aiter_module_layout():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    ).read_text(encoding="utf-8")

    assert "from aiter.fused_moe_asm import asm_moe_tkw1" in source
    assert "from aiter.ops.triton.rope import (" in source
    assert "_hcu_runtime.triton_rope_and_cache_impl" in source
    assert "aiter.fused_moe_bf16_asm" not in source
    assert "aiter.ops.triton.rope.rope" not in source
    assert "aiter.ops.triton.fused_kv_cache" not in source
    assert "aiter.ops.triton.fused_fp8_quant" not in source
    assert "aiter.ops.triton.fused_add_rmsnorm_pad" not in source
    assert "aiter.ops.fused_qk_rmsnorm_group_quant" not in source


def test_workspace_aiter_rope_and_cache_composes_public_ops(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def rope(*args, **kwargs):
        calls.append(("rope", args, kwargs))

    def cache_flash(*args, **kwargs):
        calls.append(("cache_flash", args, kwargs))

    def cache(*args, **kwargs):
        calls.append(("cache", args, kwargs))

    aiter = _package("aiter")
    ops = _package("aiter.ops")
    triton = _package("aiter.ops.triton")
    cache_module = _module(
        "aiter.ops.cache",
        reshape_and_cache=cache,
    )
    fa_utils_module = _module(
        "vllm_hcu.v1.attention.backends.fa_utils",
        reshape_and_cache_flash=cache_flash,
    )
    rope_module = _module(
        "aiter.ops.triton.rope",
        rope_cached_thd_positions_2c_fwd_inplace=rope,
    )
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton", triton)
    monkeypatch.setitem(sys.modules, "aiter.ops.cache", cache_module)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.rope", rope_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.v1.attention.backends.fa_utils",
        fa_utils_module,
    )

    query = torch.zeros(2, 8)
    key = torch.zeros(2, 4)
    value = torch.zeros(2, 1, 4)
    positions = torch.tensor([0, 1])
    cos_sin_cache = torch.zeros(4, 8)
    key_cache = torch.zeros(1)
    value_cache = torch.zeros(1)
    slots = torch.tensor([0, 1])
    k_scale = torch.tensor(2.0)
    v_scale = torch.tensor(3.0)

    aiter_runtime.triton_rope_and_cache_impl(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        True,
        key_cache,
        value_cache,
        slots,
        k_scale,
        v_scale,
        True,
        True,
    )
    assert [call[0] for call in calls] == ["rope", "cache_flash"]
    assert calls[0][1][0].shape == (2, 2, 4)
    assert calls[0][1][1].shape == (2, 1, 4)
    assert calls[0][1][5] == 0
    assert calls[1][1][5] == "fp8"

    calls.clear()
    aiter_runtime.triton_rope_and_cache_impl(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        False,
        key_cache,
        value_cache,
        slots,
        k_scale,
        v_scale,
        False,
        False,
    )
    assert [call[0] for call in calls] == ["rope", "cache"]
    assert calls[0][1][5] == 1
    assert calls[1][1][5:] == ("auto", 2.0, 3.0, False)


def test_hcu_aiter_moe_uses_v0251_out_of_place_contract():
    signature = inspect.signature(execute_aiter_moe)
    assert signature.parameters["inplace"].default is False
    assert "disable_inplace" not in signature.parameters


def test_aiter_gelu_tanh_and_feature_off_delegation(
    monkeypatch: pytest.MonkeyPatch,
):
    class ActivationType:
        GeluTanh = object()

    _install_fake_aiter(monkeypatch, ActivationType=ActivationType)
    fused_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fused_moe(*args, **kwargs):
        fused_calls.append((args, kwargs))
        return "gelu_tanh"

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe",
        _module("aiter.fused_moe", fused_moe=fused_moe),
    )
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    calls: list[tuple[object, ...]] = []

    def original(*args):
        calls.append(args)
        return "upstream"

    x = torch.ones(1, 2)
    w1 = torch.ones(1, 4, 2)
    w2 = torch.ones(1, 2, 2)
    topk_weight = torch.ones(1, 1)
    topk_ids = torch.zeros(1, 1, dtype=torch.int64)
    assert (
        aiter_runtime.fused_moe_impl(
            original, x, w1, w2, topk_weight, topk_ids, activation_method=0
        )
        == "upstream"
    )
    assert len(calls) == 1
    assert (
        aiter_runtime.fused_moe_impl(
            original, x, w1, w2, topk_weight, topk_ids, activation_method=3
        )
        == "gelu_tanh"
    )
    assert fused_calls[0][0][6] == "gelu_tanh"
    assert aiter_runtime.get_gelu_tanh_activation_type() is ActivationType.GeluTanh


def test_aiter_activation_string_mapping(monkeypatch: pytest.MonkeyPatch):
    sentinel = object()

    class ActivationType:
        GeluTanh = sentinel

    _install_fake_aiter(monkeypatch, ActivationType=ActivationType)
    original = lambda value: {"silu": "silu"}.get(value)
    assert (
        aiter_runtime.get_aiter_activation_type(original, "gelu_tanh")
        is sentinel
    )
    assert (
        aiter_runtime.get_aiter_activation_type(
            original, "GELU_PYTORCH_TANH"
        )
        is sentinel
    )
    assert aiter_runtime.get_aiter_activation_type(original, "silu") == "silu"


def test_aiter_gate_mode_requires_compatible_abi():
    assert aiter_runtime.aiter_gate_mode_kwargs(
        "",
        supports_gate_mode=False,
    ) == {}
    assert aiter_runtime.aiter_gate_mode_kwargs(
        "interleave",
        supports_gate_mode=True,
    ) == {"gate_mode": "interleave"}

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="does not support.*gate_mode='separated'",
    ):
        aiter_runtime.aiter_gate_mode_kwargs(
            "separated",
            supports_gate_mode=False,
        )


def test_aiter_w16a16_routes_public_api_with_canonical_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", True)
    x = torch.ones(2, 4)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    w1.is_shuffled = False
    w2.is_shuffled = False
    w1._hcu_aiter_moe_solution_type = "moe_c"
    w2._hcu_aiter_moe_solution_type = "moe_c"
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)
    expected = torch.full_like(x, 7)
    selector_calls: list[dict[str, object]] = []
    execute_calls: list[dict[str, object]] = []
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="moe_c",
        need_shuffle=False,
        config={},
    )

    def get_config(**kwargs: object):
        selector_calls.append(kwargs)
        return True, config

    def aiter_moe(**kwargs: object):
        execute_calls.append(kwargs)
        return expected

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )

    actual = aiter_runtime.fused_moe_impl(
        lambda *unused: pytest.fail("selected AITER config must not delegate"),
        x,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation_method=3,
        swiglu_limit=7.5,
    )

    assert actual is expected
    assert selector_calls[0]["N1"] == 8
    assert selector_calls[0]["N2"] == 4
    assert selector_calls[0]["K"] == 4
    assert selector_calls[0]["use_shuffle"] == 1
    assert selector_calls[0]["spec_sol_type"] == "moe_c"
    assert execute_calls[0]["w1"] is w1
    assert execute_calls[0]["w2"] is w2
    assert execute_calls[0]["activation"] == "gelu_tanh"
    assert execute_calls[0]["gemm1_limit"] == 7.5


def test_aiter_w16a16_reuses_installed_asm_layout_without_reshuffle(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", True)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    w1.is_shuffled = True
    w2.is_shuffled = True
    w1._hcu_aiter_moe_solution_type = "asm"
    w2._hcu_aiter_moe_solution_type = "asm"
    selector_calls: list[dict[str, object]] = []
    execute_calls: list[dict[str, object]] = []
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="asm",
        need_shuffle=True,
        config={},
    )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (
                selector_calls.append(kwargs) or True,
                config,
            ),
            aiter_moe_shfl_weight=lambda *_args, **_kwargs: pytest.fail(
                "installed ASM weights must not be shuffled again"
            ),
            aiter_moe=lambda **kwargs: execute_calls.append(kwargs)
            or kwargs["hidden_states"].clone(),
        ),
    )

    aiter_runtime.fused_moe_impl(
        lambda *_args: pytest.fail("must not delegate"),
        torch.ones(2, 4),
        w1,
        w2,
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
        activation_method=0,
    )

    assert selector_calls[0]["spec_sol_type"] == "asm"
    assert execute_calls[0]["w1"] is w1
    assert execute_calls[0]["w2"] is w2
    assert execute_calls[0]["use_weight_shuffle"] is True


def test_aiter_w16a16_shuffled_weights_never_fall_back_to_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", True)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    w1.is_shuffled = True
    w2.is_shuffled = True
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **_kwargs: (False, None),
        ),
    )

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="no ASM solution",
    ):
        aiter_runtime.fused_moe_impl(
            lambda *_args: pytest.fail("must not delegate"),
            torch.ones(2, 4),
            w1,
            w2,
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int64),
            activation_method=0,
        )


def test_aiter_w16a16_installed_canonical_layout_never_runtime_shuffles(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    for weight in (w1, w2):
        weight.is_shuffled = False
        weight._hcu_aiter_moe_solution_type = "triton"
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="triton",
        need_shuffle=True,
        config={},
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **_kwargs: (True, config),
            aiter_moe_shfl_weight=lambda *_args: pytest.fail(
                "installed canonical weights must not be shuffled at runtime"
            ),
        ),
    )

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="layout does not match",
    ):
        aiter_runtime.fused_moe_impl(
            lambda *_args: pytest.fail("must not delegate"),
            torch.ones(2, 4),
            w1,
            w2,
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int64),
            activation_method=0,
        )


def test_aiter_w16a16_rejects_runtime_config_for_different_asm_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    for weight in (w1, w2):
        weight.is_shuffled = True
        weight._hcu_aiter_moe_solution_type = "asm"
        weight._hcu_aiter_moe_weight_layout = (
            "w16a16",
            "ASM",
            True,
            64,
        )
    runtime_config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="asm",
        need_shuffle=True,
        config={"PADDED_K": 128},
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **_kwargs: (True, runtime_config),
        ),
    )

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="physical layout",
    ):
        aiter_runtime.fused_moe_impl(
            lambda *_args: pytest.fail("must not delegate"),
            torch.ones(2, 4),
            w1,
            w2,
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int64),
            activation_method=0,
        )


def test_aiter_w16a16_runtime_selects_with_installed_logical_dimensions(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="asm",
        need_shuffle=True,
        config={"ORIGINAL_K": 4, "PADDED_K": 8},
    )
    layout = (
        "w16a16",
        "ASM",
        True,
        8,
    )
    w1 = torch.ones(3, 8, 8)
    w2 = torch.ones(3, 4, 4)
    for weight in (w1, w2):
        weight.is_shuffled = True
        weight._hcu_aiter_moe_solution_type = "asm"
        weight._hcu_aiter_moe_weight_layout = layout
        weight._hcu_aiter_moe_logical_shape = (3, 8, 4, 4)
    selector_calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (
                selector_calls.append(kwargs) or True,
                config,
            ),
            aiter_moe=lambda **kwargs: kwargs["hidden_states"].clone(),
        ),
    )

    aiter_runtime.fused_moe_impl(
        lambda *_args: pytest.fail("must not delegate"),
        torch.ones(2, 4),
        w1,
        w2,
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
        activation_method=0,
    )

    assert selector_calls[0]["E"] == 3
    assert selector_calls[0]["N1"] == 8
    assert selector_calls[0]["N2"] == 4
    assert selector_calls[0]["K"] == 4


def test_aiter_w16a16_installed_native_fallback_bypasses_selector(
    monkeypatch: pytest.MonkeyPatch,
):
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)

    expected = torch.full((2, 4), 5.0)
    monkeypatch.setattr(
        aiter_runtime,
        "select_aiter_moe_config",
        lambda *_args, **_kwargs: pytest.fail(
            "native fallback must not reselect AITER at runtime"
        ),
    )
    monkeypatch.setattr(
        fused_moe_module,
        "fused_experts_impl",
        lambda *_args, **_kwargs: expected,
    )
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    for weight in (w1, w2):
        weight.is_shuffled = False
        weight._hcu_aiter_moe_solution_type = "native"

    actual = aiter_runtime.fused_moe_impl(
        lambda *_args: pytest.fail("must not delegate"),
        torch.ones(2, 4),
        w1,
        w2,
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
        activation_method=0,
    )

    assert actual is expected


def test_aiter_w16a16_post_load_replaces_parameters_with_asm_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as hcu_unquantized,
    )

    w13 = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    w2 = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    layer = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        w13_bias=None,
        w2_bias=None,
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        layer_name="model.layers.0.mlp.experts",
    )
    method = object.__new__(hcu_unquantized.HcuUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(
        num_experts=2,
        experts_per_token=2,
        in_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        activation=SimpleNamespace(value="silu"),
    )
    method.unquantized_backend = hcu_unquantized.UnquantizedMoeBackend.AITER
    method.experts_cls = object
    method.get_fused_moe_quant_config = lambda unused_layer: object()

    monkeypatch.setattr(
        hcu_unquantized,
        "_is_hcu_aiter_moe_requested",
        lambda method=None: True,
    )
    kernels: list[dict[str, object]] = []
    monkeypatch.setattr(
        hcu_unquantized,
        "make_unquantized_moe_kernel",
        lambda **kwargs: kernels.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        hcu_unquantized,
        "replace_parameter",
        lambda owner, name, value, **_kwargs: setattr(
            owner,
            name,
            torch.nn.Parameter(value, requires_grad=False),
        ),
    )
    select_calls: list[tuple[AiterMoeProblem, object, object]] = []
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="asm",
        need_shuffle=True,
        config={"PADDED_K": 64, "ORIGINAL_K": 4},
    )
    monkeypatch.setattr(
        hcu_unquantized,
        "select_aiter_moe_config",
        lambda problem, cache_owner, solution_type=None: select_calls.append(
            (problem, cache_owner, solution_type)
        ) or config,
    )
    shuffled_w13 = torch.full((2, 8, 64), 13.0)
    shuffled_w2 = torch.full((2, 64, 4), 2.0)
    shuffle_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shuffle_weights(current_w13, current_w2, _config):
        shuffle_calls.append((current_w13, current_w2))
        return shuffled_w13.clone(), shuffled_w2.clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe", _module(
            "aiter.moe",
            aiter_moe_shfl_weight=shuffle_weights,
        ),
    )

    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)

    # Upstream layerwise reload restores new canonical Parameters but does not
    # know about legacy HCU-owned layer markers.
    layer._hcu_aiter_moe_initialized = True
    reloaded_w13 = torch.nn.Parameter(torch.full_like(w13, 5), requires_grad=False)
    reloaded_w2 = torch.nn.Parameter(torch.full_like(w2, 6), requires_grad=False)
    layer.w13_weight = reloaded_w13
    layer.w2_weight = reloaded_w2
    method.process_weights_after_loading(layer)

    assert layer.w13_weight is not w13
    assert layer.w2_weight is not w2
    torch.testing.assert_close(layer.w13_weight, shuffled_w13)
    torch.testing.assert_close(layer.w2_weight, shuffled_w2)
    assert layer.w13_weight.is_shuffled is True
    assert layer.w2_weight.is_shuffled is True
    assert layer.w13_weight._hcu_aiter_moe_solution_type == "asm"
    assert layer.w2_weight._hcu_aiter_moe_solution_type == "asm"
    assert layer.w13_weight._hcu_aiter_moe_weight_layout == (
        "w16a16",
        "ASM",
        True,
        64,
    )
    assert layer.w2_weight._hcu_aiter_moe_weight_layout == (
        "w16a16",
        "ASM",
        True,
        64,
    )
    assert layer.w13_weight._hcu_aiter_moe_logical_shape == (2, 8, 4, 4)
    assert layer.w2_weight._hcu_aiter_moe_logical_shape == (2, 8, 4, 4)
    assert len(kernels) == 2
    assert method.moe_kernel is not None
    assert len(select_calls) == 2
    assert len(shuffle_calls) == 2
    assert shuffle_calls[0][0] is w13
    assert shuffle_calls[0][1] is w2
    assert shuffle_calls[1][0] is reloaded_w13
    assert shuffle_calls[1][1] is reloaded_w2
    problem, cache_owner, solution_type = select_calls[0]
    assert problem == AiterMoeProblem(
        M=1,
        E=2,
        N1=8,
        N2=4,
        K=4,
        top_k=2,
        block_size=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        quant_type="w16a16",
        activation="silu",
        use_shuffle=True,
    )
    assert cache_owner is w13
    assert solution_type == "asm"


def test_aiter_w16a16_post_load_locks_canonical_triton_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as hcu_unquantized,
    )

    w13 = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    w2 = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    layer = SimpleNamespace(w13_weight=w13, w2_weight=w2)
    method = object.__new__(hcu_unquantized.HcuUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(
        num_experts=2,
        experts_per_token=2,
        in_dtype=torch.bfloat16,
        activation=SimpleNamespace(value="silu"),
    )
    method.unquantized_backend = hcu_unquantized.UnquantizedMoeBackend.AITER
    method.experts_cls = object
    method.get_fused_moe_quant_config = lambda _layer: object()
    monkeypatch.setattr(hcu_unquantized, "_is_hcu_aiter_moe_requested", lambda *_: True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", False)
    select_calls: list[tuple[AiterMoeProblem, object, object]] = []
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="triton",
        need_shuffle=False,
        config={},
    )
    monkeypatch.setattr(
        hcu_unquantized,
        "select_aiter_moe_config",
        lambda problem, cache_owner, solution_type=None: select_calls.append(
            (problem, cache_owner, solution_type)
        ) or config,
    )
    monkeypatch.setattr(
        hcu_unquantized,
        "make_unquantized_moe_kernel",
        lambda **_kwargs: object(),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight is w13
    assert layer.w2_weight is w2
    assert layer.w13_weight.is_shuffled is False
    assert layer.w2_weight.is_shuffled is False
    assert layer.w13_weight._hcu_aiter_moe_solution_type == "triton"
    assert layer.w2_weight._hcu_aiter_moe_solution_type == "triton"
    assert len(select_calls) == 1
    assert select_calls[0][1] is w13
    assert select_calls[0][2] is None


def test_aiter_w16a16_post_load_locks_native_fallback_on_m1_miss(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as hcu_unquantized,
    )

    w13 = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    w2 = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    layer = SimpleNamespace(w13_weight=w13, w2_weight=w2)
    method = object.__new__(hcu_unquantized.HcuUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(
        num_experts=2,
        experts_per_token=2,
        in_dtype=torch.bfloat16,
        activation=SimpleNamespace(value="silu"),
    )
    method.unquantized_backend = hcu_unquantized.UnquantizedMoeBackend.AITER
    method.experts_cls = object
    method.get_fused_moe_quant_config = lambda _layer: object()
    monkeypatch.setattr(hcu_unquantized, "_is_hcu_aiter_moe_requested", lambda *_: True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", True)
    monkeypatch.setattr(
        hcu_unquantized,
        "select_aiter_moe_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        hcu_unquantized,
        "make_unquantized_moe_kernel",
        lambda **_kwargs: object(),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight is w13
    assert layer.w2_weight is w2
    assert layer.w13_weight.is_shuffled is False
    assert layer.w2_weight.is_shuffled is False
    assert layer.w13_weight._hcu_aiter_moe_solution_type == "native"
    assert layer.w2_weight._hcu_aiter_moe_solution_type == "native"


def test_explicit_aiter_backend_uses_public_router_without_auto_env_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", False)
    x = torch.ones(2, 4)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)
    expected = torch.full_like(x, 3)
    calls: list[dict[str, object]] = []
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="triton",
        need_shuffle=False,
        config={},
    )

    def execute(**kwargs: object):
        calls.append(kwargs)
        return expected

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (True, config),
            aiter_moe=execute,
        ),
    )

    with aiter_runtime.aiter_moe_request_context(
        SimpleNamespace(moe_backend="aiter")
    ):
        actual = aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("explicit AITER must not delegate"),
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
        )
    assert actual is expected
    assert calls[0]["w1"] is w1
    assert calls[0]["w2"] is w2
    assert calls[0]["activation"] == "silu"


def test_explicit_aiter_backend_enables_mask_construction_from_current_config(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    config = _module(
        "vllm.config",
        get_current_vllm_config_or_none=lambda: SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="aiter")
        ),
    )
    setattr(sys.modules["vllm"], "config", config)
    monkeypatch.setitem(sys.modules, "vllm.config", config)

    assert aiter_runtime.is_aiter_moe_requested()


def test_explicit_triton_backend_overrides_enabled_aiter_env(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )
    moe_config = SimpleNamespace(moe_backend="triton", num_experts=4)

    assert not aiter_runtime.is_aiter_moe_requested(moe_config)
    with aiter_runtime.aiter_moe_request_context(moe_config):
        assert not aiter_runtime.is_aiter_moe_requested()


def test_explicit_triton_backend_overrides_legacy_w4a16_aiter_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W4A16_MOE", True)

    layer = SimpleNamespace(
        moe_config=SimpleNamespace(moe_backend="triton")
    )

    assert not patch_compressed_tensors_moe_wna16._aiter_requested(layer)


def test_aiter_w16a16_no_solution_falls_back_to_vllm_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", True)
    fallback_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    expected = torch.full((2, 6), 5, dtype=torch.bfloat16)

    def fallback(*args: object, **kwargs: object):
        fallback_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(fused_moe_module, "fused_experts_impl", fallback)

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (False, None),
            aiter_moe=lambda **kwargs: pytest.fail(
                "no-solution must not execute AITER"
            ),
        ),
    )

    hidden_states = torch.ones(2, 6, dtype=torch.bfloat16)
    w1 = torch.ones(2, 8, 6, dtype=torch.bfloat16)
    w2 = torch.ones(2, 6, 4, dtype=torch.bfloat16)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)
    native_expert_map = torch.tensor([1, -1, 0, -1], dtype=torch.int32)
    expert_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32)
    expert_mask._vllm_hcu_native_expert_map = native_expert_map

    with aiter_runtime.aiter_moe_request_context(
        SimpleNamespace(moe_backend="aiter", num_experts=4)
    ):
        actual = aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("no-solution uses vLLM Triton directly"),
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask=expert_mask,
        )

    assert actual is expected
    assert fallback_calls[0][0][1] is w1
    assert fallback_calls[0][0][2] is w2
    assert fallback_calls[0][1]["use_fp8_w8a8"] is False
    assert fallback_calls[0][1]["use_int8_w8a8"] is False
    assert fallback_calls[0][1]["use_int8_w8a16"] is False
    assert fallback_calls[0][1]["use_int4_w4a16"] is False
    assert fallback_calls[0][1]["global_num_experts"] == 4
    assert fallback_calls[0][1]["expert_map"] is native_expert_map


def test_aiter_w16a16_config_fault_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    fallback_calls: list[object] = []
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )
    monkeypatch.setattr(
        fused_moe_module,
        "fused_experts_impl",
        lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
    )

    def config_fault(**kwargs: object):
        raise RuntimeError("aiter config fault")

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=config_fault),
    )
    tensors = (
        torch.ones(2, 4),
        torch.ones(3, 8, 4),
        torch.ones(3, 4, 4),
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
    )

    with pytest.raises(RuntimeError, match="aiter config fault"):
        aiter_runtime.fused_moe_impl(lambda *unused: None, *tensors)
    assert fallback_calls == []


def test_aiter_w16a16_selector_uses_w2_output_dim(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(aiter_runtime, "is_aiter_moe_requested", lambda: True)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", True)
    captured: dict[str, object] = {}
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="triton",
        need_shuffle=False,
        config={},
    )

    def get_config(**kwargs: object):
        captured.update(kwargs)
        return True, config

    expected = torch.ones(2, 6, dtype=torch.bfloat16)
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=get_config,
            aiter_moe=lambda **kwargs: expected,
        ),
    )
    x = torch.ones(2, 6, dtype=torch.bfloat16)
    w1 = torch.ones(3, 8, 6, dtype=torch.bfloat16)
    w2 = torch.ones(3, 6, 4, dtype=torch.bfloat16)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)

    actual = aiter_runtime.fused_moe_impl(
        lambda *unused: pytest.fail("selected AITER config must not delegate"),
        x,
        w1,
        w2,
        topk_weight,
        topk_ids,
    )

    assert actual is expected
    assert captured["N1"] == 8
    assert captured["E"] == 3
    assert captured["N2"] == 6
    assert captured["K"] == 6
    assert captured["use_shuffle"] == 1

    with pytest.raises(ValueError, match="unexpected MoE weight layout"):
        aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("invalid layout must fail early"),
            x,
            torch.ones(3, 10, 6, dtype=torch.bfloat16),
            w2,
            topk_weight,
            topk_ids,
        )


def test_aiter_w16a16_non_asm_route_uses_global_expert_map_for_ep(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", True)
    captured: dict[str, object] = {}
    calls: list[dict[str, object]] = []
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="triton",
        need_shuffle=False,
        config={},
    )

    def get_config(**kwargs: object):
        captured.update(kwargs)
        return True, config

    expected = torch.ones(2, 6, dtype=torch.bfloat16)

    def execute(**kwargs: object):
        calls.append(kwargs)
        return expected

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=get_config,
            aiter_moe=execute,
        ),
    )
    x = torch.ones(2, 6, dtype=torch.bfloat16)
    w1 = torch.ones(2, 8, 6, dtype=torch.bfloat16)
    w2 = torch.ones(2, 6, 4, dtype=torch.bfloat16)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)
    native_expert_map = torch.tensor([1, -1, 0, -1], dtype=torch.int32)
    expert_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32)
    expert_mask._vllm_hcu_native_expert_map = native_expert_map

    with aiter_runtime.aiter_moe_request_context(
        SimpleNamespace(moe_backend="aiter", num_experts=4)
    ):
        actual = aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("selected AITER config must not delegate"),
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask=expert_mask,
        )

    assert actual is expected
    assert captured["E"] == 4
    assert calls[0]["global_num_experts"] == 4
    assert calls[0]["expert_map"] is native_expert_map


def test_aiter_w16a16_asm_requires_mask_sentinel_and_rejects_expert_map(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_SHUFFLE", False)
    x = torch.ones(2, 4)
    w1 = torch.ones(2, 8, 4)
    w2 = torch.ones(2, 4, 4)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)
    expert_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32)
    calls: list[dict[str, object]] = []
    expected = torch.full_like(x, 9)
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type="asm",
        need_shuffle=False,
        config={"SOL_ID1": 1, "SOL_ID2": 2},
    )

    def execute(**kwargs: object):
        calls.append(kwargs)
        return expected

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            get_aiter_moe_config=lambda **kwargs: (True, config),
            aiter_moe=execute,
        ),
    )

    moe_config = SimpleNamespace(moe_backend="aiter", num_experts=4)
    with aiter_runtime.aiter_moe_request_context(moe_config):
        actual = aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("AITER ASM must not delegate"),
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask=expert_mask,
        )

    assert actual is expected
    assert calls[0]["global_num_experts"] == 4
    assert calls[0]["expert_map"] is expert_mask

    global_to_local_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    with aiter_runtime.aiter_moe_request_context(moe_config):
        with pytest.raises(
            aiter_runtime.HcuAiterRuntimeError,
            match="trailing sentinel.*global-to-local expert map",
        ):
            aiter_runtime.fused_moe_impl(
                lambda *unused: pytest.fail("invalid expert map must fail early"),
                x,
                w1,
                w2,
                topk_weight,
                topk_ids,
                expert_mask=global_to_local_map,
            )

    with aiter_runtime.aiter_moe_request_context(moe_config):
        with pytest.raises(
            aiter_runtime.HcuAiterRuntimeError,
            match="requires an expert mask for EP",
        ):
            aiter_runtime.fused_moe_impl(
                lambda *unused: pytest.fail("missing EP mask must fail early"),
                x,
                w1,
                w2,
                topk_weight,
                topk_ids,
            )


def test_aiter_feature_off_delegates_v0251_fused_moe_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    calls: list[tuple[object, ...]] = []

    def original(*args):
        calls.append(args)
        return "target"

    tensors = (
        torch.ones(2, 4),
        torch.ones(3, 8, 4),
        torch.ones(3, 4, 4),
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
    )
    assert (
        aiter_runtime.fused_moe_impl(
            original,
            *tensors,
            gate_mode="separated",
            moe_sorting_dispatch_policy=3,
            swiglu_limit=4.5,
        )
        == "target"
    )
    assert calls[0][17] == "separated"
    assert calls[0][20] == 3
    assert calls[0][21] == 4.5


@pytest.mark.parametrize("extended", [False, True])
def test_aiter_topk_supports_old_and_new_abi(
    monkeypatch: pytest.MonkeyPatch, extended: bool
):
    calls: list[tuple[object, ...]] = []
    if extended:

        def topk(a, b, c, d, e, num_shared_experts=0, shared_expert_scoring_func=""):
            calls.append(
                (a, b, c, d, e, num_shared_experts, shared_expert_scoring_func)
            )

    else:

        def topk(a, b, c, d, e):
            calls.append((a, b, c, d, e))

    _install_fake_aiter(monkeypatch, topk_softmax=topk)
    tensors = [torch.empty(1) for _ in range(4)]
    aiter_runtime.topk_softmax_impl(*tensors, True, 2, "sigmoid")
    assert len(calls[0]) == (7 if extended else 5)


def test_scaled_mm_prequantized_input_bypasses_quantizer():
    calls: list[tuple[object, ...]] = []

    class FP8ScaledMMLinearKernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def apply_weights(self, layer, x, bias=None):
            calls.append(("original", layer, x, bias))
            return "original"

    module = _module(
        patch_scaled_mm_linear_kernel.TARGET_MODULE,
        FP8ScaledMMLinearKernel=FP8ScaledMMLinearKernel,
    )
    patch_scaled_mm_linear_kernel.apply_to_module(module)
    kernel = FP8ScaledMMLinearKernel()
    kernel.config = SimpleNamespace(out_dtype=None)
    weight = torch.ones(3, 5, dtype=torch.int8)
    weight_scale = torch.ones(5)
    kernel._get_layer_params = lambda layer: (weight, weight_scale, None, None)
    kernel.apply_scaled_mm = lambda **kwargs: calls.append(("scaled", kwargs)) or kwargs
    x = torch.ones(2, 3)
    x_q = torch.ones(2, 3, dtype=torch.int8)
    x_scale = torch.ones(2, 1)
    result = kernel.apply_weights(object(), x, x_and_scale_quanted=(x_q, x_scale))
    assert calls[0][0] == "scaled"
    assert result["A"] is x_q and result["As"] is x_scale
    assert kernel.supports_quanted_inputs() is True
    assert kernel.apply_weights(object(), x) == "original"


def test_scaled_mm_prequantized_scale_shape_preserves_eager_value_error():
    class FP8ScaledMMLinearKernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def apply_weights(self, layer, x, bias=None):
            return x

    module = _module(
        patch_scaled_mm_linear_kernel.TARGET_MODULE,
        FP8ScaledMMLinearKernel=FP8ScaledMMLinearKernel,
    )
    patch_scaled_mm_linear_kernel.apply_to_module(module)
    kernel = FP8ScaledMMLinearKernel()
    kernel.config = SimpleNamespace(out_dtype=None)
    weight = torch.ones(3, 5, dtype=torch.int8)
    weight_scale = torch.ones(5)
    kernel._get_layer_params = lambda layer: (weight, weight_scale, None, None)
    kernel.apply_scaled_mm = lambda **kwargs: kwargs

    with pytest.raises(ValueError, match="scale must be scalar or per-token"):
        kernel.apply_weights(
            object(),
            torch.ones(2, 3),
            x_and_scale_quanted=(
                torch.ones(2, 3, dtype=torch.int8),
                torch.ones(2, 2),
            ),
        )


def test_scaled_mm_prequantized_scale_shape_is_one_strict_dynamic_graph(monkeypatch):
    from sympy.logic.boolalg import Boolean

    # Some ROCm-enabled PyTorch builds report accelerator support to Dynamo
    # even on a CPU-only runner, which makes compilation snapshot a CUDA RNG
    # state and initialize HIP.  This graph is deliberately CPU-only.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class FP8ScaledMMLinearKernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def apply_weights(self, layer, x, bias=None):
            return x

        def apply_scaled_mm(
            self,
            *,
            A,
            B,
            out_dtype,
            As,
            Bs,
            bias,
            output_shape,
        ):
            # Keep this CPU-only graph shape dependent while avoiding any
            # device kernel.  The regression target is the wrapper's symbolic
            # scale-shape validation before this target-owned call.
            return A.to(out_dtype) * As

    module = _module(
        patch_scaled_mm_linear_kernel.TARGET_MODULE,
        FP8ScaledMMLinearKernel=FP8ScaledMMLinearKernel,
    )
    patch_scaled_mm_linear_kernel.apply_to_module(module)
    kernel = FP8ScaledMMLinearKernel()
    kernel.config = SimpleNamespace(out_dtype=torch.float32)
    weight = torch.ones(3, 5, dtype=torch.int8)
    weight_scale = torch.ones(5)
    kernel._get_layer_params = lambda layer: (weight, weight_scale, None, None)
    layer = object()
    compile_count = 0
    captured_graphs = []

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        captured_graphs.append(graph_module)
        return graph_module.forward

    def apply_prequantized(x, x_2d_q, x_scale):
        return kernel.apply_weights(
            layer,
            x,
            x_and_scale_quanted=(x_2d_q, x_scale),
        )

    compiled = torch.compile(
        apply_prequantized,
        backend=counting_backend,
        dynamic=True,
        fullgraph=True,
    )
    for num_tokens in (2, 33, 65, 129):
        x = torch.ones(num_tokens, 3)
        x_2d_q = torch.ones(num_tokens, 3, dtype=torch.int8)
        x_scale = torch.ones(num_tokens, 1)
        for tensor in (x, x_2d_q, x_scale):
            torch._dynamo.mark_dynamic(
                tensor,
                0,
                min=1,
                max=10240,
            )

        result = compiled(x, x_2d_q, x_scale)
        assert result.shape == (num_tokens, 3)
        torch.testing.assert_close(result, torch.ones(num_tokens, 3))

    assert compile_count == 1
    symbolic_boolean_nodes = []
    for graph_module in captured_graphs:
        for node in graph_module.graph.nodes:
            value = node.meta.get("example_value")
            if isinstance(value, (torch.SymBool, Boolean)):
                symbolic_boolean_nodes.append((node.name, repr(value)))
    assert symbolic_boolean_nodes == [], symbolic_boolean_nodes


def test_clamp_swiglu_enforces_rocm_custom_op():
    sentinel = object()

    class CustomOp:
        def __init__(self, *, enforce_enable=False, compile_native=False):
            self.base_args = (enforce_enable, compile_native)
            self._forward_method = "dispatched"

    class SiluAndMulWithClamp(CustomOp):
        def __init__(
            self,
            swiglu_limit: float,
            alpha: float = 1.0,
            beta: float = 0.0,
            *,
            compile_native: bool = True,
        ):
            super().__init__(compile_native=compile_native)
            self.swiglu_limit = float(swiglu_limit)
            self.alpha = float(alpha)
            self.beta = float(beta)

        def forward_native(self, x):
            return x

    platform = SimpleNamespace(
        is_rocm=lambda: True,
        is_cuda_alike=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
    )
    module = _module(
        patch_activation.TARGET_MODULE,
        SiluAndMulWithClamp=SiluAndMulWithClamp,
        current_platform=platform,
        torch=SimpleNamespace(
            ops=SimpleNamespace(
                _C=SimpleNamespace(silu_and_mul_with_clamp=sentinel)
            )
        ),
    )
    patch_activation.apply_to_module(module)
    instance = SiluAndMulWithClamp(7.0, 1.5, 0.25, compile_native=False)
    assert instance.base_args == (True, False)
    assert instance.op is sentinel
    assert instance.alpha == 1.5
    assert instance.beta == 0.25


def test_compressed_linear_only_forwards_supported_prequantized_input():
    class CompressedTensorsLinearMethod:
        def apply(self, layer, x, bias=None):
            return layer.scheme.apply_weights(layer, x, bias=bias)

    module = _module(
        patch_compressed_tensors.TARGET_MODULE,
        CompressedTensorsLinearMethod=CompressedTensorsLinearMethod,
    )
    patch_compressed_tensors.apply_to_module(module)
    calls: list[dict[str, object]] = []
    scheme = SimpleNamespace(
        supports_quanted_inputs=lambda: True,
        apply_weights=lambda layer, x, **kwargs: calls.append(kwargs) or "quantized",
    )
    method = CompressedTensorsLinearMethod()
    pair = (torch.ones(1, 2, dtype=torch.int8), torch.ones(1, 1))
    assert (
        method.apply(
            SimpleNamespace(scheme=scheme),
            torch.ones(1, 2),
            x_and_scale_quanted=pair,
        )
        == "quantized"
    )
    assert calls == [{"bias": None, "x_and_scale_quanted": pair}]
    assert method.supports_quanted_inputs() is True


def test_compressed_scheme_inactive_anchor_is_corrected_at_runtime():
    class CompressedTensorsScheme:
        def process_weights_after_loading(self, layer):
            raise NotImplementedError()

    module = _module(
        patch_compressed_tensors_scheme.TARGET_MODULE,
        CompressedTensorsScheme=CompressedTensorsScheme,
    )
    assert patch_compressed_tensors_scheme.apply_to_module(module) is True
    assert CompressedTensorsScheme().supports_quanted_inputs() is False
    assert patch_compressed_tensors_scheme.apply_to_module(module) is False


@pytest.mark.hcu
def test_slimquant_w4a8_dispatches_current_routed_experts_to_aiter_method():
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    # FusedMoE is a factory in v0.25.1. It constructs RoutedExperts, whose
    # constructor immediately calls get_quant_method. Build that exact public
    # layer type without recursively entering the dispatch under test.
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(
        moe_backend="aiter",
        moe_parallel_config=SimpleNamespace(
            tp_size=8,
            dp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
        ),
    )
    config = slimquant_w4a8.SlimQuantW4A8Int8Config()

    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

    assert type(method) is slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod
    assert method.moe is layer.moe_config


@pytest.mark.hcu
def test_slimquant_w4a8_dispatch_preserves_linear_method():
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    layer = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(layer)

    method = slimquant_w4a8.SlimQuantW4A8Int8Config().get_quant_method(
        layer, "model.layers.0.self_attn.q_proj"
    )

    assert type(method) is slimquant_w4a8.SlimQuantW4A8Int8LinearMethod
    assert isinstance(layer.scheme, slimquant_w4a8.CompressedTensorsW8A8Int8)


@pytest.mark.hcu
def test_slimquant_w4a8_dispatch_returns_none_for_unsupported_layer():
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = slimquant_w4a8.SlimQuantW4A8Int8Config().get_quant_method(
        torch.nn.Embedding(4, 4), "model.embed_tokens"
    )

    assert method is None


@pytest.mark.hcu
def test_slimquant_w4a8_moe_quant_config_uses_int4_weight_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    calls: list[dict[str, object]] = []

    class FusedMoEQuantConfig:
        @staticmethod
        def make(*args, **kwargs):
            calls.append({"args": args, **kwargs})
            return SimpleNamespace(kind="w4a8_quant_config")

    monkeypatch.setattr(
        slimquant_w4a8, "FusedMoEQuantConfig", FusedMoEQuantConfig
    )
    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    layer = SimpleNamespace(
        w13_weight_scale=torch.ones(2, 4, 1),
        w2_weight_scale=torch.ones(2, 2, 1),
        w13_input_scale=None,
        w2_input_scale=None,
    )

    quant_config = method.get_fused_moe_quant_config(layer)

    assert quant_config.kind == "w4a8_quant_config"
    assert method.moe_quant_config is quant_config
    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == (torch.int8,)
    torch.testing.assert_close(
        call["w1_scale"], torch.full_like(layer.w13_weight_scale, 16.0)
    )
    torch.testing.assert_close(
        call["w2_scale"], torch.full_like(layer.w2_weight_scale, 16.0)
    )
    torch.testing.assert_close(layer.w13_weight_scale, torch.ones(2, 4, 1))
    torch.testing.assert_close(layer.w2_weight_scale, torch.ones(2, 2, 1))
    assert call["a1_scale"] is None
    assert call["a2_scale"] is None
    assert call["per_act_token_quant"] is True
    assert call["per_out_ch_quant"] is False
    assert call["block_shape"] is None
    assert call["weight_dtype"] == "int4"


@pytest.mark.hcu
def test_slimquant_w4a8_moe_method_is_a_direct_fused_moe_method():
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=object(),
    )

    assert type(method) is slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod
    assert isinstance(method, slimquant_w4a8.FusedMoEMethodBase)


@pytest.mark.hcu
def test_slimquant_w4a8_deepep_auto_uses_w4a8_deepgemm_factory_not_aiter(
    monkeypatch: pytest.MonkeyPatch,
):
    """DP+EP W4A8 must own an auto DeepGEMM kernel, not an AITER fallback."""

    from vllm_hcu.model_executor.layers.fused_moe.experts import (
        dpsk_v4_deep_gemm_moe as deepgemm_module,
    )
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    factory_calls: list[tuple[object, object, object]] = []
    processed_layers: list[object] = []

    class W4A8Experts:
        def process_weights_after_loading(self, layer: object) -> None:
            processed_layers.append(layer)

    class W4A8Kernel:
        fused_experts = SimpleNamespace(experts=W4A8Experts())

        @staticmethod
        def apply(**kwargs: object) -> torch.Tensor:
            return kwargs["hidden_states"] + 3

    def make_deepep_auto_deepgemm_w4a8_moe_kernel(
        *,
        moe_quant_config: object,
        moe_config: object,
        routing_tables: object = None,
    ) -> W4A8Kernel:
        factory_calls.append((moe_quant_config, moe_config, routing_tables))
        return W4A8Kernel()

    monkeypatch.setattr(
        deepgemm_module,
        "make_deepep_auto_deepgemm_w4a8_moe_kernel",
        make_deepep_auto_deepgemm_w4a8_moe_kernel,
        raising=False,
    )
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a8_moe",
        lambda *_args: pytest.fail(
            "deepep_auto W4A8 must create DeepGEMM experts, not prewarm AITER"
        ),
    )
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "apply_aiter_w4a8_moe",
        lambda *_args: pytest.fail(
            "deepep_auto W4A8 must execute its DeepGEMM kernel, not AITER"
        ),
    )

    moe = SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            dp_size=2,
            use_ep=True,
            all2all_backend="deepep_auto",
            use_deepep_auto_kernels=True,
        ),
        _hcu_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"]
            )
        ),
    )
    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(object(), moe)
    routing_tables = (object(), object(), object())
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros((2, 8, 2), dtype=torch.int8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros((2, 4, 2), dtype=torch.int8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones((2, 8, 1)), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones((2, 4, 1)), requires_grad=False
        ),
        w13_input_scale=None,
        w2_input_scale=None,
        _expert_routing_tables=lambda: routing_tables,
    )

    assert method.moe_quant_config is None

    method.process_weights_after_loading(layer)

    quant_config = method.moe_quant_config
    assert quant_config is not None
    assert quant_config.weight_quant_dtype == "int4"
    assert factory_calls == [(quant_config, moe, routing_tables)]
    assert processed_layers == [layer]
    assert method.moe_kernel is not None
    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.apply(
            layer,
            x,
            torch.ones((2, 1), dtype=torch.float32),
            torch.zeros((2, 1), dtype=torch.int32),
            None,
            None,
        ),
        x + 3,
    )


@pytest.mark.hcu
def test_slimquant_w4a8_deepep_auto_rejects_non_deepseek_v4_architecture():
    from vllm_hcu.model_executor.layers.fused_moe.deepep_runtime import (
        slimquant_w4a8_uses_deepep_auto,
    )

    moe = SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            dp_size=2,
            use_ep=True,
            all2all_backend="deepep_auto",
            use_deepep_auto_kernels=True,
        ),
        _hcu_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV3ForCausalLM"]
            )
        ),
    )

    with pytest.raises(ValueError, match="validated only for DeepSeek-V4"):
        slimquant_w4a8_uses_deepep_auto(moe)


@pytest.mark.hcu
def test_slimquant_w4a8_deepep_auto_rejects_non_silu_activation():
    from vllm_hcu.model_executor.layers.fused_moe.deepep_runtime import (
        slimquant_w4a8_uses_deepep_auto,
    )

    moe = SimpleNamespace(
        activation=SimpleNamespace(value="gelu"),
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            dp_size=2,
            use_ep=True,
            all2all_backend="deepep_auto",
            use_deepep_auto_kernels=True,
        ),
        _hcu_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"]
            )
        ),
    )

    with pytest.raises(ValueError, match="supports only SiLU activation"):
        slimquant_w4a8_uses_deepep_auto(moe)


@pytest.mark.hcu
def test_slimquant_w4a8_deepep_auto_rejects_non_rocm_platform(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime

    monkeypatch.setattr(
        deepep_runtime,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: False),
        raising=False,
    )
    moe = SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            dp_size=2,
            use_ep=True,
            all2all_backend="deepep_auto",
            use_deepep_auto_kernels=True,
        ),
        _hcu_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"]
            )
        ),
    )

    with pytest.raises(RuntimeError, match="requires the HCU ROCm runtime"):
        deepep_runtime.slimquant_w4a8_uses_deepep_auto(moe)


@pytest.mark.hcu
def test_slimquant_w4a8_deepep_auto_rejects_incomplete_hipc_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime

    deepep_runtime._require_slimquant_w4a8_hipc_runtime.cache_clear()
    noop = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "deepgemm",
        _module(
            "deepgemm",
            pack_w4a8_moe_hipc_weight=noop,
            view_w4a8_moe_hipc_weight_n32_layout=noop,
            m_grouped_w4a8_gemm_nt_contiguous_hipc=noop,
        ),
    )
    _install_lightop_activation(
        monkeypatch,
        fuse_silu_mul_quant=noop,
        fuse_silu_mul_quant_ep=noop,
    )
    moe = SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            dp_size=2,
            use_ep=True,
            all2all_backend="deepep_auto",
            use_deepep_auto_kernels=True,
        ),
        _hcu_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"]
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="m_grouped_w4a8_gemm_nt_masked_hipc",
    ):
        deepep_runtime.slimquant_w4a8_uses_deepep_auto(moe)


@pytest.mark.hcu
def test_slimquant_w4a8_deepep_auto_rejects_missing_ll_lightop(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime

    deepep_runtime._require_slimquant_w4a8_hipc_runtime.cache_clear()
    noop = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "deepgemm",
        _module(
            "deepgemm",
            pack_w4a8_moe_hipc_weight=noop,
            view_w4a8_moe_hipc_weight_n32_layout=noop,
            m_grouped_w4a8_gemm_nt_contiguous_hipc=noop,
            m_grouped_w4a8_gemm_nt_masked_hipc=noop,
        ),
    )
    _install_lightop_activation(
        monkeypatch,
        fuse_silu_mul_quant=noop,
    )
    moe = SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            dp_size=2,
            use_ep=True,
            all2all_backend="deepep_auto",
            use_deepep_auto_kernels=True,
        ),
        _hcu_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"]
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"lightop\.activation\.fuse_silu_mul_quant_ep",
    ):
        deepep_runtime.slimquant_w4a8_uses_deepep_auto(moe)


@pytest.mark.hcu
@pytest.mark.parametrize(
    (
        "dp_size",
        "use_ep",
        "all2all_backend",
        "auto_kernels",
        "moe_backend",
        "match",
    ),
    [
        (
            2,
            True,
            "deepep_high_throughput",
            False,
            "auto",
            "requires all2all_backend='deepep_auto'",
        ),
        (
            2,
            True,
            "deepep_low_latency",
            False,
            "auto",
            "requires all2all_backend='deepep_auto'",
        ),
        (
            1,
            False,
            "deepep_auto",
            True,
            "auto",
            "requires dp_size > 1 and expert parallelism",
        ),
        (
            2,
            False,
            "deepep_auto",
            True,
            "auto",
            "requires dp_size > 1 and expert parallelism",
        ),
        (
            2,
            True,
            "deepep_auto",
            False,
            "auto",
            "incompatible use_deepep_auto_kernels metadata",
        ),
        (
            2,
            True,
            "deepep_auto",
            True,
            "aiter",
            "requires moe_backend='auto' or 'deep_gemm'",
        ),
    ],
)
def test_slimquant_w4a8_deepep_routing_fails_closed_before_tp_fallback(
    monkeypatch: pytest.MonkeyPatch,
    dp_size: int,
    use_ep: bool,
    all2all_backend: str,
    auto_kernels: bool,
    moe_backend: str,
    match: str,
):
    """Invalid DeepEP metadata must never fall through to TP AITER/Triton."""

    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a8_moe",
        lambda *_args: pytest.fail("invalid DeepEP metadata entered TP AITER"),
    )
    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        object(),
        SimpleNamespace(
            moe_backend=moe_backend,
            moe_parallel_config=SimpleNamespace(
                dp_size=dp_size,
                use_ep=use_ep,
                all2all_backend=all2all_backend,
                use_deepep_auto_kernels=auto_kernels,
            ),
        ),
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros((2, 8, 2), dtype=torch.int8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros((2, 4, 2), dtype=torch.int8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones((2, 8, 1)), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones((2, 4, 1)), requires_grad=False
        ),
        w13_input_scale=None,
        w2_input_scale=None,
    )
    method.get_fused_moe_quant_config(layer)

    with pytest.raises(ValueError, match=match):
        method.process_weights_after_loading(layer)


@pytest.mark.hcu
def test_slimquant_w4a8_installs_moe_c_layout_at_load(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    calls: list[dict[str, object]] = []
    selected = SimpleNamespace(
        quant_type="w4a8",
        solution_type="moe_c",
        need_shuffle=True,
        need_shuffle_scale=False,
    )

    class MoeQuantType:
        W4A8 = "w4a8"

    def get_config(**kwargs: object):
        calls.append(kwargs)
        return True, selected

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe_shfl_weight=lambda w1, w2, _config: (
                w1.clone().add_(1),
                w2.clone().add_(2),
            ),
        ),
    )
    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=SimpleNamespace(
            experts_per_token=2,
            hidden_dim=4,
            in_dtype=torch.bfloat16,
            activation=SimpleNamespace(value="silu"),
            num_experts=2,
        ),
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros(2, 8, 2, dtype=torch.int8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros(2, 4, 2, dtype=torch.int8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(2, 8, 1), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(2, 4, 1), requires_grad=False
        ),
        w13_input_scale=None,
        w2_input_scale=None,
    )
    assert method.moe_quant_config is None

    method.process_weights_after_loading(layer)

    assert calls == [
        {
            "M": 1,
            "E": 2,
            "N1": 8,
            "N2": 4,
            "K": 4,
            "top_k": 2,
            "block_size": 0,
            "dtype": torch.bfloat16,
            "quant_type": "w4a8",
            "activation": "silu",
            "use_shuffle": 1,
            "spec_sol_type": "moe_c",
        }
    ]
    assert layer.w13_weight.is_shuffled is True
    assert layer.w2_weight.is_shuffled is True
    assert layer.w13_weight._hcu_aiter_moe_solution_type == "moe_c"
    assert layer.w2_weight._hcu_aiter_moe_solution_type == "moe_c"
    assert method.moe_quant_config is not None
    torch.testing.assert_close(
        method.moe_quant_config.w1_scale,
        torch.full_like(method.moe_quant_config.w1_scale, 16.0),
    )
    torch.testing.assert_close(
        method.moe_quant_config.w2_scale,
        torch.full_like(method.moe_quant_config.w2_scale, 16.0),
    )
    torch.testing.assert_close(
        layer.w13_weight_scale, torch.ones_like(layer.w13_weight_scale)
    )
    torch.testing.assert_close(
        layer.w2_weight_scale, torch.ones_like(layer.w2_weight_scale)
    )
    torch.testing.assert_close(layer.w13_weight, torch.ones_like(layer.w13_weight))
    torch.testing.assert_close(layer.w2_weight, torch.full_like(layer.w2_weight, 2))


def test_w8a8_prewarm_replaces_raw_weights_with_selected_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    config = SimpleNamespace(
        quant_type="w8a8",
        solution_type="asm",
        need_shuffle=True,
        need_shuffle_scale=False,
        config={},
    )

    class MoeQuantType:
        W8A8 = "w8a8"

    shuffle_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shuffle_weights(w1, w2, _config):
        shuffle_calls.append((w1, w2))
        return w1.clone().add_(3), w2.clone().add_(4)

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **_kwargs: (True, config),
            aiter_moe_shfl_weight=shuffle_weights,
        ),
    )
    layer = _fp8_moe_layer()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros((3, 8, 4), dtype=torch.int8), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros((3, 4, 4), dtype=torch.int8), requires_grad=False
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        block_shape=None,
    )

    compressed_tensors_moe_runtime.prewarm_aiter_quantized_moe(
        layer,
        SimpleNamespace(
            experts_per_token=2,
            in_dtype=torch.bfloat16,
            activation=SimpleNamespace(value="silu"),
        ),
        quant_config,
    )
    compressed_tensors_moe_runtime.prewarm_aiter_quantized_moe(
        layer,
        SimpleNamespace(
            experts_per_token=2,
            in_dtype=torch.bfloat16,
            activation=SimpleNamespace(value="silu"),
        ),
        quant_config,
    )

    assert layer.w13_weight._hcu_aiter_moe_solution_type == "asm"
    assert layer.w2_weight._hcu_aiter_moe_solution_type == "asm"
    assert layer.w13_weight._hcu_aiter_moe_weight_layout == (
        "w8a8",
        "ASM",
        True,
        None,
    )
    assert layer.w2_weight._hcu_aiter_moe_weight_layout == (
        "w8a8",
        "ASM",
        True,
        None,
    )
    torch.testing.assert_close(layer.w13_weight, torch.full_like(layer.w13_weight, 3))
    torch.testing.assert_close(layer.w2_weight, torch.full_like(layer.w2_weight, 4))
    assert len(shuffle_calls) == 1

    reloaded_w13 = torch.nn.Parameter(
        torch.zeros((3, 8, 4), dtype=torch.int8), requires_grad=False
    )
    reloaded_w2 = torch.nn.Parameter(
        torch.zeros((3, 4, 4), dtype=torch.int8), requires_grad=False
    )
    layer.w13_weight = reloaded_w13
    layer.w2_weight = reloaded_w2
    compressed_tensors_moe_runtime.prewarm_aiter_quantized_moe(
        layer,
        SimpleNamespace(
            experts_per_token=2,
            in_dtype=torch.bfloat16,
            activation=SimpleNamespace(value="silu"),
        ),
        quant_config,
    )

    assert len(shuffle_calls) == 2
    assert shuffle_calls[1][0] is reloaded_w13
    assert shuffle_calls[1][1] is reloaded_w2
    torch.testing.assert_close(layer.w13_weight, torch.full_like(layer.w13_weight, 3))
    torch.testing.assert_close(layer.w2_weight, torch.full_like(layer.w2_weight, 4))


def test_w8a8_padded_layout_keeps_original_logical_problem_dimensions(
    monkeypatch: pytest.MonkeyPatch,
):
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
        need_shuffle_scale=False,
        config={"PADDED_K": 8, "ORIGINAL_K": 4},
    )
    config_calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        config_calls.append(kwargs)
        return True, config

    def shuffle_weights(w1, w2, _config):
        return (
            torch.zeros((w1.shape[0], w1.shape[1], 8), dtype=w1.dtype),
            torch.zeros((w2.shape[0], 8, w2.shape[2]), dtype=w2.dtype),
        )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe_shfl_weight=shuffle_weights,
            aiter_moe=lambda **kwargs: kwargs["hidden_states"],
        ),
    )
    layer = _fp8_moe_layer()
    quant_config = SimpleNamespace(
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        block_shape=None,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a1_scale=None,
        a2_scale=None,
    )
    moe_config = SimpleNamespace(
        experts_per_token=2,
        in_dtype=torch.bfloat16,
        activation=SimpleNamespace(value="silu"),
        num_experts=3,
    )

    compressed_tensors_moe_runtime.prewarm_aiter_quantized_moe(
        layer, moe_config, quant_config
    )
    compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=torch.ones((2, 2), dtype=torch.bfloat16),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=moe_config,
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=None,
        quant_config=quant_config,
    )

    assert layer.w13_weight.shape[2] == 8
    assert layer.w2_weight.shape[1] == 8
    assert config_calls[1]["K"] == 4
    assert config_calls[1]["N1"] == 8
    assert config_calls[1]["N2"] == 4


def test_padded_layout_validation_rejects_wrong_w2_axis(
    monkeypatch: pytest.MonkeyPatch,
):
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
        config={"PADDED_K": 8, "ORIGINAL_K": 4},
    )
    w1 = torch.zeros((2, 8, 4))
    w2 = torch.zeros((2, 4, 4))
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=lambda *_args: (
                torch.zeros((2, 8, 8)),
                torch.zeros((2, 4, 8)),
            ),
        ),
    )

    with pytest.raises(
        HcuAiterMoeDispatchError,
        match="incompatible w2 shape",
    ):
        prepare_aiter_moe_weights(
            w1,
            w2,
            config,
            cache_owner=object(),
        )


def test_padded_scale_layout_accepts_weight_matching_k_axes(
    monkeypatch: pytest.MonkeyPatch,
):
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle_scale=True,
        config={"PADDED_K": 8, "ORIGINAL_K": 4},
    )
    scale1 = torch.ones((2, 8, 4))
    scale2 = torch.ones((2, 4, 4))
    expected1 = torch.full((2, 8, 8), 3.0)
    expected2 = torch.full((2, 8, 4), 4.0)
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_scale=lambda *_args: (expected1, expected2),
        ),
    )

    actual1, actual2 = prepare_aiter_moe_scales(
        scale1, scale2, config, cache_owner=object()
    )

    assert actual1 is expected1
    assert actual2 is expected2


def test_padded_scale_layout_rejects_wrong_w2_axis(
    monkeypatch: pytest.MonkeyPatch,
):
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle_scale=True,
        config={"PADDED_K": 8, "ORIGINAL_K": 4},
    )
    scale1 = torch.ones((2, 8, 4))
    scale2 = torch.ones((2, 4, 4))
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_scale=lambda *_args: (
                torch.zeros((2, 8, 8)),
                torch.zeros((2, 4, 8)),
            ),
        ),
    )

    with pytest.raises(HcuAiterMoeDispatchError, match="incompatible w2_scale shape"):
        prepare_aiter_moe_scales(scale1, scale2, config, cache_owner=object())


def test_w8a8_m1_miss_locks_native_weights_without_runtime_relayout(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoeQuantType:
        W8A8 = "w8a8"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **_kwargs: (False, None),
        ),
    )
    layer = _fp8_moe_layer()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros((3, 8, 4), dtype=torch.int8), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros((3, 4, 4), dtype=torch.int8), requires_grad=False
    )

    compressed_tensors_moe_runtime.prewarm_aiter_quantized_moe(
        layer,
        SimpleNamespace(
            experts_per_token=2,
            in_dtype=torch.bfloat16,
            activation=SimpleNamespace(value="silu"),
        ),
        SimpleNamespace(
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            block_shape=None,
        ),
    )

    assert layer.w13_weight._hcu_aiter_moe_solution_type == "native"
    assert layer.w2_weight._hcu_aiter_moe_solution_type == "native"
    assert layer.w13_weight.is_shuffled is False
    assert layer.w2_weight.is_shuffled is False


def test_quantized_installed_weights_reject_different_physical_layout():
    w1 = torch.zeros((2, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((2, 4, 4), dtype=torch.int8)
    installed_layout = ("w8a8", "MOE_C", True, 64)
    for weight in (w1, w2):
        weight.is_shuffled = True
        weight._hcu_aiter_moe_solution_type = "moe_c"
        weight._hcu_aiter_moe_weight_layout = installed_layout
    runtime_config = SimpleNamespace(
        quant_type="w8a8",
        solution_type="moe_c",
        need_shuffle=True,
        config={"PADDED_K": 128},
    )

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="physical layout",
    ):
        compressed_tensors_moe_runtime._weights_for_selected_config(
            w1,
            w2,
            runtime_config,
            installed_solution="moe_c",
        )


def test_w8a8_prewarm_installs_selected_scale_layout_once(
    monkeypatch: pytest.MonkeyPatch,
):
    config = SimpleNamespace(
        quant_type="w8a8",
        solution_type="moe_c",
        need_shuffle=False,
        need_shuffle_scale=True,
        config={},
    )

    class MoeQuantType:
        W8A8 = "w8a8"

    scale_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shuffle_scales(scale1, scale2, _config):
        scale_calls.append((scale1, scale2))
        return scale1.clone().add_(3), scale2.clone().add_(4)

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **_kwargs: (True, config),
            aiter_moe_shfl_scale=shuffle_scales,
        ),
    )
    layer = _fp8_moe_layer()
    original_w1_scale = layer.w13_weight_scale
    original_w2_scale = layer.w2_weight_scale
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        block_shape=None,
        w1_scale=original_w1_scale,
        w2_scale=original_w2_scale,
    )

    compressed_tensors_moe_runtime.prewarm_aiter_quantized_moe(
        layer,
        SimpleNamespace(
            experts_per_token=2,
            in_dtype=torch.bfloat16,
            activation=SimpleNamespace(value="silu"),
        ),
        quant_config,
    )
    compressed_tensors_moe_runtime.prewarm_aiter_quantized_moe(
        layer,
        SimpleNamespace(
            experts_per_token=2,
            in_dtype=torch.bfloat16,
            activation=SimpleNamespace(value="silu"),
        ),
        quant_config,
    )

    assert len(scale_calls) == 1
    assert layer.w13_weight_scale is not original_w1_scale
    assert layer.w2_weight_scale is not original_w2_scale
    assert quant_config.w1_scale is layer.w13_weight_scale
    assert quant_config.w2_scale is layer.w2_weight_scale
    assert layer.w13_weight_scale._hcu_aiter_moe_scale_layout == (
        "w8a8",
        "MOE_C",
        True,
        None,
        None,
    )
    assert layer.w2_weight_scale._hcu_aiter_moe_scale_layout == (
        "w8a8",
        "MOE_C",
        True,
        None,
        None,
    )


def test_installed_scale_layout_rejects_different_padded_k(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoeQuantType:
        W8A8 = "w8a8"

    first_config = SimpleNamespace(
        quant_type="w8a8",
        solution_type="moe_c",
        need_shuffle_scale=True,
        config={"PADDED_K": 8, "ORIGINAL_K": 4},
    )
    second_config = SimpleNamespace(
        quant_type="w8a8",
        solution_type="moe_c",
        need_shuffle_scale=True,
        config={"PADDED_K": 16, "ORIGINAL_K": 4},
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            aiter_moe_shfl_scale=lambda scale1, scale2, _config: (
                torch.zeros((2, 8, 8), dtype=scale1.dtype),
                torch.zeros((2, 8, 4), dtype=scale2.dtype),
            ),
        ),
    )
    layer = SimpleNamespace(
        w13_weight_scale=torch.ones((2, 8, 4)),
        w2_weight_scale=torch.ones((2, 4, 4)),
    )
    quant_config = SimpleNamespace(
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
    )

    compressed_tensors_moe_runtime.install_aiter_moe_scale_layout(
        layer, quant_config, first_config
    )

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="selected layout",
    ):
        compressed_tensors_moe_runtime.install_aiter_moe_scale_layout(
            layer, quant_config, second_config
        )


@pytest.mark.hcu
def test_slimquant_w4a8_tp_aiter_keeps_selected_canonical_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    """The DP auto ownership exception must not transform pure-TP weights."""

    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=SimpleNamespace(moe_backend="aiter"),
    )
    w13 = torch.nn.Parameter(
        torch.arange(16, dtype=torch.int8).reshape(1, 8, 2),
        requires_grad=False,
    )
    w2 = torch.nn.Parameter(
        torch.arange(8, dtype=torch.int8).reshape(1, 4, 2),
        requires_grad=False,
    )
    layer = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=torch.nn.Parameter(
            torch.ones((1, 8, 1), dtype=torch.float32), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones((1, 4, 1), dtype=torch.float32), requires_grad=False
        ),
        w13_input_scale=None,
        w2_input_scale=None,
    )
    expected_w13 = w13.detach().clone()
    expected_w2 = w2.detach().clone()
    assert method.moe_quant_config is None
    selected = SimpleNamespace(
        quant_type="w4a8",
        solution_type="moe_c",
        need_shuffle=False,
        need_shuffle_scale=False,
        config={},
    )
    prewarm_calls: list[object] = []

    def prewarm(_method, owner):
        prewarm_calls.append(owner)
        return selected

    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a8_moe",
        prewarm,
    )

    method.process_weights_after_loading(layer)

    assert method.moe_quant_config is not None
    assert prewarm_calls == [layer]
    assert layer.w13_weight is w13
    assert layer.w2_weight is w2
    torch.testing.assert_close(layer.w13_weight, expected_w13)
    torch.testing.assert_close(layer.w2_weight, expected_w2)
    assert not hasattr(layer, "_slimquant_w4a8_deepep_auto_layout")

    installed_quant_config = method.moe_quant_config
    method.process_weights_after_loading(layer)
    assert method.moe_quant_config is installed_quant_config
    assert prewarm_calls == [layer]

    with torch.no_grad():
        layer.w13_weight_scale.add_(1.0)
        layer.w2_weight_scale.add_(2.0)
    method.process_weights_after_loading(layer)
    assert method.moe_quant_config is not installed_quant_config
    assert prewarm_calls == [layer]
    torch.testing.assert_close(
        method.moe_quant_config.w1_scale,
        torch.full((1, 8, 1), 32.0),
    )
    torch.testing.assert_close(
        method.moe_quant_config.w2_scale,
        torch.full((1, 4, 1), 48.0),
    )


@pytest.mark.hcu
def test_slimquant_w4a8_explicit_triton_prepares_only_vllm_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a8_moe",
        lambda *args: pytest.fail("explicit Triton must not prewarm AITER"),
    )
    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=SimpleNamespace(
            moe_backend="triton",
            experts_per_token=1,
            hidden_dim=4,
            in_dtype=torch.float16,
            activation=SimpleNamespace(value="silu"),
            num_experts=1,
        ),
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.tensor(
                [[[0x8F, 0x70], [0x1E, 0xA5], [0x70, 0x8F], [0xA5, 0x1E]]],
                dtype=torch.uint8,
            ).view(torch.int8),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.tensor(
                [[[0x8F], [0x70], [0x1E], [0xA5]]], dtype=torch.uint8
            ).view(torch.int8),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(1, 4, 1), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(1, 4, 1), requires_grad=False
        ),
        w13_input_scale=None,
        w2_input_scale=None,
    )
    assert method.moe_quant_config is None
    packed_w13 = layer.w13_weight.detach().clone()
    packed_w2 = layer.w2_weight.detach().clone()

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (1, 4, 4)
    assert layer.w2_weight.shape == (1, 4, 2)
    assert layer.w13_weight._hcu_vllm_w4a8_unpacked is True
    assert layer.w2_weight._hcu_vllm_w4a8_unpacked is True
    assert not hasattr(layer.w13_weight, "_hcu_vllm_w4a8_fallback_weights")
    assert not hasattr(layer.w13_weight, "_hcu_aiter_moe_m1_supported")
    assert method.moe_quant_config is not None
    torch.testing.assert_close(
        method.moe_quant_config.w1_scale,
        layer.w13_weight_scale * 16.0,
    )
    torch.testing.assert_close(
        method.moe_quant_config.w2_scale,
        layer.w2_weight_scale * 16.0,
    )

    from vllm.model_executor.layers.fused_moe import fused_moe

    kernel_calls: list[dict[str, object]] = []

    def fused_experts_impl(hidden_states, *_args, **kwargs):
        kernel_calls.append(kwargs)
        return hidden_states + 1

    monkeypatch.setattr(fused_moe, "fused_experts_impl", fused_experts_impl)
    hidden_states = torch.zeros((2, 4), dtype=torch.float16)
    result = method.apply(
        layer,
        hidden_states,
        torch.ones((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.int64),
        None,
        None,
    )

    torch.testing.assert_close(result, hidden_states + 1)
    assert kernel_calls[0]["w1_scale"] is method.moe_quant_config.w1_scale
    assert kernel_calls[0]["w2_scale"] is method.moe_quant_config.w2_scale

    installed_w13 = layer.w13_weight
    installed_w2 = layer.w2_weight
    installed_quant_config = method.moe_quant_config
    method.process_weights_after_loading(layer)
    assert layer.w13_weight is installed_w13
    assert layer.w2_weight is installed_w2
    assert method.moe_quant_config is installed_quant_config

    with torch.no_grad():
        layer.w13_weight_scale.add_(1.0)
        layer.w2_weight_scale.add_(2.0)
    method.process_weights_after_loading(layer)
    scale_updated_quant_config = method.moe_quant_config
    assert layer.w13_weight is installed_w13
    assert layer.w2_weight is installed_w2
    assert scale_updated_quant_config is not installed_quant_config
    torch.testing.assert_close(
        scale_updated_quant_config.w1_scale,
        torch.full((1, 4, 1), 32.0),
    )
    torch.testing.assert_close(
        scale_updated_quant_config.w2_scale,
        torch.full((1, 4, 1), 48.0),
    )

    layer.w13_weight = torch.nn.Parameter(packed_w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(packed_w2, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.full((1, 4, 1), 2.0), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.full((1, 4, 1), 3.0), requires_grad=False
    )
    method.process_weights_after_loading(layer)

    assert method.moe_quant_config is not installed_quant_config
    torch.testing.assert_close(
        method.moe_quant_config.w1_scale,
        torch.full((1, 4, 1), 32.0),
    )
    torch.testing.assert_close(
        method.moe_quant_config.w2_scale,
        torch.full((1, 4, 1), 48.0),
    )


@pytest.mark.hcu
def test_slimquant_w4a8_m1_miss_installs_single_native_triton_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    class MoeQuantType:
        W4A8 = "w4a8"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **_: (False, None),
        ),
    )
    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=SimpleNamespace(
            experts_per_token=1,
            hidden_dim=4,
            in_dtype=torch.float16,
            activation=SimpleNamespace(value="silu"),
            num_experts=1,
        ),
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.tensor(
                [[[0x8F, 0x70], [0x1E, 0xA5], [0x70, 0x8F], [0xA5, 0x1E]]],
                dtype=torch.uint8,
            ).view(torch.int8),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.tensor(
                [[[0x8F], [0x70], [0x1E], [0xA5]]], dtype=torch.uint8
            ).view(torch.int8),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(1, 4, 1), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(1, 4, 1), requires_grad=False
        ),
        w13_input_scale=None,
        w2_input_scale=None,
    )

    method.process_weights_after_loading(layer)

    assert method.moe_quant_config is not None
    assert layer.w13_weight.shape == (1, 4, 4)
    assert layer.w2_weight.shape == (1, 4, 2)
    assert layer.w13_weight._hcu_vllm_w4a8_unpacked is True
    assert layer.w2_weight._hcu_vllm_w4a8_unpacked is True
    assert layer.w13_weight._hcu_aiter_moe_solution_type == "native"
    assert layer.w2_weight._hcu_aiter_moe_solution_type == "native"


@pytest.mark.hcu
def test_slimquant_w4a8_legacy_raw_weights_pin_actual_m_to_moe_c(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    selected = SimpleNamespace(
        quant_type="w4a8",
        solution_type="moe_c",
        need_shuffle=False,
        need_shuffle_scale=False,
    )

    class MoeQuantType:
        W4A8 = "w4a8"

    def get_config(**kwargs: object):
        calls["selector"] = kwargs
        return True, selected

    def aiter_moe(**kwargs: object):
        calls["execute"] = kwargs
        return kwargs["hidden_states"] + 1

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    x = torch.zeros(3, 4, dtype=torch.bfloat16)
    topk_weights = torch.ones(3, 2, dtype=torch.float32)
    topk_ids = torch.zeros(3, 2, dtype=torch.int32)
    w1 = torch.zeros(2, 8, 2, dtype=torch.int8)
    w2 = torch.zeros(2, 4, 2, dtype=torch.int8)
    w1._hcu_aiter_moe_m1_supported = False
    w1_scale = torch.full((2, 8, 1), 16.0)
    w2_scale = torch.full((2, 4, 1), 16.0)
    method = SimpleNamespace(
        moe=SimpleNamespace(num_experts=2),
        moe_quant_config=SimpleNamespace(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
        ),
    )
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=w2,
        global_num_experts=2,
        _expert_map=None,
        expert_mask=None,
    )

    result = compressed_tensors_moe_runtime.apply_aiter_w4a8_moe(
        method, layer, x, topk_weights, topk_ids
    )

    torch.testing.assert_close(result, x + 1)
    assert calls["selector"] == {
        "M": 3,
        "E": 2,
        "N1": 8,
        "N2": 4,
        "K": 4,
        "top_k": 2,
        "block_size": 0,
        "dtype": torch.bfloat16,
        "quant_type": "w4a8",
        "activation": "silu",
        "use_shuffle": 1,
        "spec_sol_type": "moe_c",
    }
    execute = calls["execute"]
    assert execute["moe_config"] is selected
    assert execute["w1"] is w1
    assert execute["w2"] is w2
    assert execute["w1_scale"] is w1_scale
    assert execute["w2_scale"] is w2_scale


@pytest.mark.hcu
def test_slimquant_w4a8_actual_m_no_config_uses_cached_selection_and_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers.fused_moe import fused_moe

    calls: list[dict[str, object]] = []
    selector_calls: list[int] = []

    class MoeQuantType:
        W4A8 = "w4a8"

    def get_config(**kwargs: object):
        selector_calls.append(int(kwargs["M"]))
        return False, None

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
        ),
    )

    def fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        *,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        global_num_experts=-1,
        expert_map=None,
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    ):
        args = (hidden_states, w1, w2, topk_weights, topk_ids)
        kwargs = {
            "activation": activation,
            "apply_router_weight_on_input": apply_router_weight_on_input,
            "use_fp8_w8a8": use_fp8_w8a8,
            "use_int8_w8a8": use_int8_w8a8,
            "use_int8_w8a16": use_int8_w8a16,
            "use_int4_w4a16": use_int4_w4a16,
            "per_channel_quant": per_channel_quant,
            "global_num_experts": global_num_experts,
            "expert_map": expert_map,
            "w1_scale": w1_scale,
            "w2_scale": w2_scale,
            "a1_scale": a1_scale,
            "a2_scale": a2_scale,
            "block_shape": block_shape,
        }
        calls.append({"args": args, "kwargs": kwargs})
        return hidden_states + 1

    monkeypatch.setattr(fused_moe, "fused_experts_impl", fused_experts_impl)
    packed_w1 = torch.tensor(
        [[[0x8F, 0x70], [0x1E, 0xA5], [0x70, 0x8F], [0xA5, 0x1E]]],
        dtype=torch.uint8,
    ).view(torch.int8)
    packed_w2 = torch.tensor(
        [[[0x8F], [0x70], [0x1E], [0xA5]]], dtype=torch.uint8
    ).view(torch.int8)
    method = SimpleNamespace(
        moe=SimpleNamespace(num_experts=1),
        moe_quant_config=SimpleNamespace(
            w1_scale=torch.full((1, 4, 1), 16.0),
            w2_scale=torch.full((1, 4, 1), 16.0),
            a1_scale=None,
            a2_scale=None,
        ),
    )
    layer = SimpleNamespace(
        w13_weight=packed_w1,
        w2_weight=packed_w2,
        global_num_experts=1,
        _expert_map=None,
        expert_mask=None,
    )
    x = torch.zeros(2, 4, dtype=torch.float16)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)

    first = compressed_tensors_moe_runtime.apply_aiter_w4a8_moe(
        method, layer, x, topk_weights, topk_ids
    )
    second = compressed_tensors_moe_runtime.apply_aiter_w4a8_moe(
        method, layer, x, topk_weights, topk_ids
    )

    torch.testing.assert_close(first, x + 1)
    torch.testing.assert_close(second, x + 1)
    assert selector_calls == [2]
    assert len(calls) == 2
    first_args = calls[0]["args"]
    second_args = calls[1]["args"]
    assert first_args[1] is not second_args[1]
    assert first_args[2] is not second_args[2]
    torch.testing.assert_close(
        first_args[1],
        torch.tensor(
            [[
                [-8, -1, 7, 0],
                [1, -2, -6, 5],
                [7, 0, -8, -1],
                [-6, 5, 1, -2],
            ]],
            dtype=torch.int8,
        ),
    )
    torch.testing.assert_close(
        first_args[2],
        torch.tensor(
            [[[-8, -1], [7, 0], [1, -2], [-6, 5]]], dtype=torch.int8
        ),
    )
    kwargs = calls[0]["kwargs"]
    assert kwargs["use_int8_w8a8"] is True
    assert kwargs["use_int4_w4a16"] is False
    assert kwargs["per_channel_quant"] is True


@pytest.mark.hcu
def test_slimquant_w4a8_explicit_triton_runtime_never_selects_aiter(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers.fused_moe import fused_moe
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "select_aiter_moe_config",
        lambda *args, **kwargs: pytest.fail(
            "explicit Triton must not query the AITER selector"
        ),
    )
    expected = torch.full((2, 4), 3.0, dtype=torch.float16)
    monkeypatch.setattr(
        fused_moe,
        "fused_experts_impl",
        lambda *args, **kwargs: expected,
    )
    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=SimpleNamespace(
            moe_backend="triton",
            hidden_dim=4,
            activation=SimpleNamespace(value="silu"),
            num_experts=1,
        ),
    )
    method.moe_quant_config = SimpleNamespace(
        w1_scale=torch.full((1, 4, 1), 16.0),
        w2_scale=torch.full((1, 4, 1), 16.0),
        a1_scale=None,
        a2_scale=None,
    )
    layer = SimpleNamespace(
        w13_weight=torch.zeros((1, 4, 2), dtype=torch.int8),
        w2_weight=torch.zeros((1, 4, 1), dtype=torch.int8),
        global_num_experts=1,
        _expert_map=None,
        expert_mask=None,
    )
    x = torch.zeros((2, 4), dtype=torch.float16)

    actual = method.apply(
        layer,
        x,
        torch.ones((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.int32),
        None,
        None,
    )

    assert actual is expected


@pytest.mark.hcu
def test_slimquant_w4a8_apply_fails_closed_for_unsupported_inputs(
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    layer = SimpleNamespace()
    x = torch.zeros(3, 2, dtype=torch.float16)
    topk_weights = torch.ones(3, 1, dtype=torch.float16)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="rank-2 hidden states"):
        method.apply(layer, x.unsqueeze(0), topk_weights, topk_ids, None, None)
    with pytest.raises(ValueError, match="same shape"):
        method.apply(layer, x, topk_weights[:, :0], topk_ids, None, None)
    with pytest.raises(ValueError, match="same token count"):
        method.apply(layer, x, topk_weights[:2], topk_ids[:2], None, None)


@pytest.mark.hcu
def test_slimquant_w4a8_apply_accepts_runner_owned_shared_experts(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    layer = SimpleNamespace()
    x = torch.zeros(3, 2, dtype=torch.float16)
    topk_weights = torch.ones(3, 1, dtype=torch.float16)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "apply_aiter_w4a8_moe",
        lambda *args: args[2],
    )

    result = method.apply(
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=object(),
        shared_experts_input=x,
    )

    assert result is x


@pytest.mark.hcu
def test_slimquant_w4a8_deepep_auto_restores_native_nonidentity_expert_map(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    monkeypatch.setattr(
        deepep_runtime,
        "slimquant_w4a8_uses_deepep_auto",
        lambda _moe: True,
    )
    calls: list[dict[str, object]] = []

    class W4A8Kernel:
        @staticmethod
        def apply(**kwargs: object) -> torch.Tensor:
            calls.append(kwargs)
            return kwargs["hidden_states"]

    native_expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    expert_mask = torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32)
    expert_mask._vllm_hcu_native_expert_map = native_expert_map
    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    method.moe = SimpleNamespace(num_experts=4)
    method.moe_kernel = W4A8Kernel()
    layer = SimpleNamespace(
        w13_weight=torch.zeros((2, 8, 2), dtype=torch.int8),
        w2_weight=torch.zeros((2, 4, 2), dtype=torch.int8),
        activation=SimpleNamespace(value="silu"),
        global_num_experts=4,
        expert_map=expert_mask,
        apply_router_weight_on_input=False,
    )
    x = torch.zeros((2, 4), dtype=torch.bfloat16)

    result = method.apply(
        layer,
        x,
        torch.ones((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.int32),
        None,
        None,
    )

    assert result is x
    assert calls[0]["expert_map"] is native_expert_map


@pytest.mark.hcu
def test_slimquant_w4a8_apply_reuses_initialized_deepep_auto_route(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.config

    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: None,
    )

    class W4A8Kernel:
        @staticmethod
        def apply(**kwargs: object) -> torch.Tensor:
            return kwargs["hidden_states"]

    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    method.moe = SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        moe_backend="auto",
        num_experts=2,
        moe_parallel_config=SimpleNamespace(
            dp_size=2,
            use_ep=True,
            all2all_backend="deepep_auto",
            use_deepep_auto_kernels=True,
        ),
    )
    method.moe_kernel = W4A8Kernel()
    layer = SimpleNamespace(
        w13_weight=torch.zeros((2, 8, 2), dtype=torch.int8),
        w2_weight=torch.zeros((2, 4, 2), dtype=torch.int8),
        activation=SimpleNamespace(value="silu"),
        global_num_experts=2,
        expert_map=torch.tensor([0, 1], dtype=torch.int32),
        apply_router_weight_on_input=False,
    )
    x = torch.zeros((2, 4), dtype=torch.bfloat16)

    result = method.apply(
        layer,
        x,
        torch.ones((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.int32),
        None,
        None,
    )

    assert result is x


def _fake_moe_fp8_module():
    channel = object()
    token = object()
    tensor = object()

    class CompressedTensorsW8A8Fp8MoEMethod:
        init_calls: list[object] = []
        selected_backend = "TRITON"

        def __init__(self, weight_quant, input_quant, moe, layer_name=None):
            type(self).init_calls.append(moe)
            self.weight_quant = weight_quant
            self.input_quant = input_quant
            self.moe = moe
            self.layer_name = layer_name
            self.fp8_backend = SimpleNamespace(value=type(self).selected_backend)

        def process_weights_after_loading(self, layer):
            layer.upstream_processed = True

        def apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        ):
            return (
                "upstream",
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )

    return _module(
        patch_compressed_tensors_moe_w8a8_fp8.TARGET_MODULE,
        CompressedTensorsW8A8Fp8MoEMethod=CompressedTensorsW8A8Fp8MoEMethod,
        QuantizationStrategy=SimpleNamespace(
            CHANNEL=channel,
            TOKEN=token,
            TENSOR=tensor,
        ),
    )


def _channel_fp8_moe_args(module: ModuleType) -> tuple[object, object]:
    strategy = module.QuantizationStrategy
    return (
        SimpleNamespace(strategy=strategy.CHANNEL),
        SimpleNamespace(strategy=strategy.TOKEN),
    )


def _tensor_fp8_moe_args(module: ModuleType) -> tuple[object, object]:
    strategy = module.QuantizationStrategy
    return (
        SimpleNamespace(strategy=strategy.TENSOR),
        SimpleNamespace(strategy=strategy.TENSOR),
    )


def _fp8_moe_layer() -> SimpleNamespace:
    return SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        w13_weight=torch.ones(3, 8, 4),
        w2_weight=torch.ones(3, 4, 4),
        w13_weight_scale=torch.ones(3, 8, 1),
        w2_weight_scale=torch.ones(3, 4, 1),
        w13_input_scale=None,
        w2_input_scale=None,
        global_num_experts=3,
        expert_map=None,
        layer_name="model.layers.0.mlp.experts",
    )


def test_moe_fp8_target_triton_preserves_process_and_apply_behavior(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    target_apply = method_class.apply
    assert patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module) is True
    assert patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module) is False
    assert method_class.apply is target_apply
    assert not hasattr(method_class, "_get_aiter_moe_runtime_config")
    assert not hasattr(method_class, "_get_aiter_weights_for_solution")

    method = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="triton"),
    )
    layer = _fp8_moe_layer()
    method.process_weights_after_loading(layer)
    assert layer.upstream_processed is True
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    shared = object()
    result = method.apply(layer, x, weights, ids, shared, None)
    assert result == ("upstream", layer, x, weights, ids, shared, None)
    assert method_class.init_calls == [method.moe]


@pytest.mark.parametrize(
    ("target_aiter", "hcu_aiter"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_moe_fp8_explicit_aiter_ignores_legacy_environment_half_states(
    monkeypatch: pytest.MonkeyPatch,
    target_aiter: bool,
    hcu_aiter: bool,
):
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    method_class.selected_backend = "AITER"
    if target_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    else:
        monkeypatch.delenv("VLLM_ROCM_USE_AITER_MOE", raising=False)
    if hcu_aiter:
        monkeypatch.setenv("VLLM_HCU_USE_AITER_W8A8_FP8_MOE", "1")
    else:
        monkeypatch.delenv(
            "VLLM_HCU_USE_AITER_W8A8_FP8_MOE",
            raising=False,
        )
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="aiter"),
    )
    assert method.fp8_backend.value == "AITER"
    assert len(method_class.init_calls) == 1


def test_moe_fp8_aiter_prewarms_m1_during_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    method_class.selected_backend = "AITER"
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="aiter"),
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        block_shape=None,
    )
    method.moe_quant_config = quant_config
    calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_quantized_moe",
        lambda layer, moe, config: calls.append((layer, moe, config)),
    )
    layer = _fp8_moe_layer()

    method.process_weights_after_loading(layer)

    assert layer.upstream_processed is True
    assert calls == [(layer, method.moe, quant_config)]


def test_moe_fp8_explicit_deepgemm_accepts_hcu_oracle_selection():
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    method_class.selected_backend = "HCU_DEEPGEMM"
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)

    method = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="deep_gemm"),
    )

    assert method.fp8_backend.value == "HCU_DEEPGEMM"
    assert len(method_class.init_calls) == 1


def test_moe_fp8_auto_delegates_selection_and_explicit_checks_match():
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    automatic = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="auto"),
    )
    assert automatic.fp8_backend.value == "TRITON"
    assert len(method_class.init_calls) == 1

    method_class.selected_backend = "AITER"
    with pytest.raises(RuntimeError, match="selected='AITER'"):
        method_class(
            *_channel_fp8_moe_args(module),
            SimpleNamespace(moe_backend="triton"),
        )
    assert len(method_class.init_calls) == 2

    method_class.selected_backend = "TRITON"
    with pytest.raises(RuntimeError, match="selected='TRITON'"):
        method_class(
            *_channel_fp8_moe_args(module),
            SimpleNamespace(moe_backend="aiter"),
        )
    assert len(method_class.init_calls) == 3


def test_moe_fp8_non_channel_routes_delegate_target_without_triton_policy(
):
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    method_class.selected_backend = "AITER"
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = method_class(
        *_tensor_fp8_moe_args(module),
        SimpleNamespace(moe_backend="auto"),
    )
    assert method.fp8_backend.value == "AITER"
    assert len(method_class.init_calls) == 1


def test_moe_fp8_aiter_route_is_layer_aware_and_cached(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        calls.append(kwargs)
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="MOE_C",
            need_shuffle=False,
            config={"serial": len(calls)},
        )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=lambda **kwargs: kwargs["hidden_states"].clone(),
        ),
    )
    method = SimpleNamespace(moe=object())
    x = torch.ones(2, 4)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    first_layer = _fp8_moe_layer()
    weights = torch.ones(2, 2)

    def run(layer: object):
        return compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            method,
            layer,
            x,
            weights,
            ids,
            None,
            None,
        )

    run(first_layer)
    run(first_layer)
    run(_fp8_moe_layer())
    assert len(calls) == 2
    assert calls[0]["quant_type"] == "fp8_w8a8"
    assert calls[0]["M"] == 2 and calls[0]["top_k"] == 2
    assert calls[0]["use_shuffle"] == 1


def test_moe_fp8_padded_layout_routes_with_installed_logical_dimensions(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
        need_shuffle_scale=False,
        config={"PADDED_K": 8, "ORIGINAL_K": 4},
    )

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        calls.append(kwargs)
        return True, config

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=lambda **kwargs: kwargs["hidden_states"],
        ),
    )
    layer = _fp8_moe_layer()
    layer.w13_weight = torch.zeros((3, 8, 8))
    layer.w2_weight = torch.zeros((3, 8, 4))
    logical_shape = (3, 8, 4, 4)
    layout = ("fp8_w8a8", "MOE_C", True, 8)
    for weight in (layer.w13_weight, layer.w2_weight):
        weight._hcu_aiter_moe_solution_type = "moe_c"
        weight._hcu_aiter_moe_logical_shape = logical_shape
        weight._hcu_aiter_moe_weight_layout = layout
        weight.is_shuffled = True

    compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
        SimpleNamespace(moe=object()),
        layer,
        torch.ones((2, 4)),
        torch.ones((2, 2)),
        torch.zeros((2, 2), dtype=torch.int64),
        None,
        None,
    )

    assert calls[0]["E"] == 3
    assert calls[0]["N1"] == 8
    assert calls[0]["N2"] == 4
    assert calls[0]["K"] == 4


def test_quantized_installed_logical_contract_rejects_invalid_dimensions():
    w1 = torch.zeros((3, 8, 4))
    w2 = torch.zeros((3, 4, 4))
    w1._hcu_aiter_moe_logical_shape = (3, 8, 4, -1)
    w2._hcu_aiter_moe_logical_shape = (3, 8, 4, -1)

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="invalid logical dimensions",
    ):
        compressed_tensors_moe_runtime._installed_weight_logical_shape(w1, w2)


def test_quantized_installed_logical_contract_rejects_invalid_layout_signature():
    w1 = torch.zeros((3, 8, 4))
    w2 = torch.zeros((3, 4, 4))
    for weight in (w1, w2):
        weight._hcu_aiter_moe_logical_shape = (3, 8, 4, 4)
        weight._hcu_aiter_moe_weight_layout = ("fp8_w8a8",)

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="invalid physical layout signature",
    ):
        compressed_tensors_moe_runtime._installed_weight_logical_shape(w1, w2)


def test_moe_fp8_installed_layout_rejects_hidden_width_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", MoeQuantType=MoeQuantType),
    )
    layer = _fp8_moe_layer()
    for weight in (layer.w13_weight, layer.w2_weight):
        weight._hcu_aiter_moe_solution_type = "native"
        weight._hcu_aiter_moe_logical_shape = (3, 8, 4, 4)
        weight.is_shuffled = False

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="hidden states do not match logical K",
    ):
        compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            SimpleNamespace(moe=object()),
            layer,
            torch.ones((2, 5)),
            torch.ones((2, 2)),
            torch.zeros((2, 2), dtype=torch.int64),
            None,
            None,
        )


@pytest.mark.parametrize("entrypoint", ["generic", "explicit"])
def test_quantized_native_layout_accepts_non_gated_geometry(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", MoeQuantType=MoeQuantType),
    )
    expected = torch.ones((2, 4))
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )
    monkeypatch.setattr(
        fused_moe_module,
        "fused_experts_impl",
        lambda *_args, **_kwargs: expected,
    )
    w1 = torch.zeros((2, 6, 4))
    w2 = torch.zeros((2, 4, 6))
    for weight in (w1, w2):
        weight._hcu_aiter_moe_solution_type = "native"
        weight._hcu_aiter_moe_logical_shape = (2, 6, 4, 4)
        weight.is_shuffled = False
    hidden_states = torch.ones((2, 4))
    topk_weights = torch.ones((2, 1))
    topk_ids = torch.zeros((2, 1), dtype=torch.int64)
    if entrypoint == "generic":
        quant_config = SimpleNamespace(
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            block_shape=None,
            w1_scale=torch.ones((2, 6, 1)),
            w2_scale=torch.ones((2, 4, 1)),
            a1_scale=None,
            a2_scale=None,
        )
        actual = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            SimpleNamespace(num_experts=2),
            SimpleNamespace(value="relu2"),
            False,
            None,
            quant_config,
        )
    else:
        layer = SimpleNamespace(
            activation=SimpleNamespace(value="relu2"),
            apply_router_weight_on_input=False,
            w13_weight=w1,
            w2_weight=w2,
            w13_weight_scale=torch.ones((2, 6, 1)),
            w2_weight_scale=torch.ones((2, 4, 1)),
            w13_input_scale=None,
            w2_input_scale=None,
            global_num_experts=2,
            expert_map=None,
        )
        actual = compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            SimpleNamespace(moe=object()),
            layer,
            hidden_states,
            topk_weights,
            topk_ids,
            None,
            None,
        )
    assert actual is expected


def test_moe_fp8_uses_unified_shuffle_cache_and_invalidates_on_weight_change(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    kernel_weights: list[torch.Tensor] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def shuffle_weights(w1, w2, config):
        assert config is moe_config
        calls.append("w1")
        calls.append("w2")
        return w1 + 1, w2 + 2

    moe_config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
        config={},
    )

    def execute(**kwargs: object):
        kernel_weights.append(kwargs["w1"])
        return kwargs["hidden_states"].clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **kwargs: (True, moe_config),
            aiter_moe_shfl_weight=shuffle_weights,
            aiter_moe=execute,
        ),
    )
    layer = _fp8_moe_layer()
    method = SimpleNamespace(moe=object())
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)

    def run():
        return compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            method,
            layer,
            x,
            weights,
            ids,
            None,
            None,
        )

    run()
    run()
    assert kernel_weights[0] is kernel_weights[1]
    assert calls == ["w1", "w2"]
    layer.w13_weight.add_(1)
    run()
    assert kernel_weights[2] is not kernel_weights[1]
    assert calls == ["w1", "w2", "w1", "w2"]


def test_moe_fp8_hcu_aiter_flag_defaults_off(monkeypatch: pytest.MonkeyPatch):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.delenv("VLLM_HCU_USE_AITER_W8A8_FP8_MOE", raising=False)
    assert henvs.VLLM_HCU_USE_AITER_W8A8_FP8_MOE is False
    monkeypatch.setenv("VLLM_HCU_USE_AITER_W8A8_FP8_MOE", "1")
    assert henvs.VLLM_HCU_USE_AITER_W8A8_FP8_MOE is True


def test_moe_fp8_aiter_path_accepts_v0251_shared_expert_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel_calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        return True, SimpleNamespace(solution_type="ASM")

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        return torch.full((2, 4), 7.0)

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            per_token_quant_hip=_fp8_quant_abi_stub,
        ),
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    assert "disable_inplace" not in FusedMoEConfig.__dataclass_fields__
    target_moe_config = object.__new__(FusedMoEConfig)
    method = SimpleNamespace(
        moe=target_moe_config,
        _hcu_aiter_moe_config_cache={},
    )
    layer = _fp8_moe_layer()
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    output = compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
        method, layer, x, weights, ids, None, None
    )
    torch.testing.assert_close(output, torch.full((2, 4), 7.0))
    assert kernel_calls[0]["hidden_states"] is x
    assert kernel_calls[0]["inplace"] is False
    assert kernel_calls[0]["topk_ids"].dtype is torch.int32
    shared = object()
    output_with_shared_contract = (
        compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            method, layer, x, weights, ids, shared, x
        )
    )
    torch.testing.assert_close(
        output_with_shared_contract, torch.full((2, 4), 7.0)
    )
    method.moe = None
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="vLLM v0.25.1 MoE configuration",
    ):
        compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            method, layer, x, weights, ids, None, None
        )


def test_moe_fp8_no_solution_falls_back_to_vllm_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **kwargs: (False, None),
        ),
    )
    expected = torch.full((2, 4), 8.0)
    fallback_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )

    def fallback(*args: object, **kwargs: object):
        fallback_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(fused_moe_module, "fused_experts_impl", fallback)
    layer = _fp8_moe_layer()
    x = torch.ones(2, 4)

    actual = compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
        SimpleNamespace(moe=object()),
        layer,
        x,
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
        None,
        None,
    )

    assert actual is expected
    assert fallback_calls[0][0][0] is x
    assert fallback_calls[0][0][1] is layer.w13_weight
    assert fallback_calls[0][0][2] is layer.w2_weight
    assert fallback_calls[0][1]["use_fp8_w8a8"] is True
    assert fallback_calls[0][1]["per_channel_quant"] is True


def test_moe_fp8_config_fault_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def config_fault(**kwargs: object):
        raise RuntimeError("aiter config fault")

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=config_fault,
        ),
    )
    fallback_calls: list[object] = []
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )
    monkeypatch.setattr(
        fused_moe_module,
        "fused_experts_impl",
        lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="aiter config fault"):
        compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            SimpleNamespace(moe=object()),
            _fp8_moe_layer(),
            torch.ones(2, 4),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int64),
            None,
            None,
        )
    assert fallback_calls == []


def test_moe_fp8_runtime_owns_no_duplicate_aiter_selector_helpers():
    assert not hasattr(
        compressed_tensors_moe_runtime,
        "get_aiter_w8a8_runtime_config",
    )
    assert not hasattr(
        compressed_tensors_moe_runtime,
        "get_aiter_weights_for_solution",
    )


@pytest.mark.parametrize(
    ("use_fp8", "use_int8", "expected_quant_type"),
    [
        (True, False, "fp8_w8a8"),
        (False, True, "int8_w8a8"),
    ],
)
def test_quantized_aiter_runtime_selects_exact_quant_type(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    expected_quant_type: str,
):
    config_calls: list[dict[str, object]] = []
    kernel_calls: list[dict[str, object]] = []
    expected_output = torch.full((2, 4), 9.0)

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        config_calls.append(kwargs)
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="asm",
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        return expected_output

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            per_token_quant_int8=lambda x: (x, None),
            per_token_quant_hip=_fp8_quant_abi_stub,
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    w1 = torch.zeros((3, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((3, 4, 4), dtype=torch.int8)
    topk_weights = torch.ones((2, 2), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 2), dtype=torch.int64)
    w1_scale = torch.ones((3, 8, 1), dtype=torch.float32)
    w2_scale = torch.ones((3, 4, 1), dtype=torch.float32)
    a1q_scale = torch.ones((2, 1), dtype=torch.float32)
    expert_map = torch.tensor([0, 1, 2], dtype=torch.int32)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    output = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        vllm_moe_config=SimpleNamespace(num_experts=3),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=expert_map,
        quant_config=quant_config,
        a1q_scale=a1q_scale,
        output_dtype=torch.bfloat16,
    )

    assert output is expected_output
    assert config_calls[0]["quant_type"] == expected_quant_type
    assert config_calls[0]["M"] == 2
    assert config_calls[0]["E"] == 3
    assert config_calls[0]["top_k"] == 2
    assert config_calls[0]["use_shuffle"] == 1
    call = kernel_calls[0]
    assert call["hidden_states"] is hidden_states
    assert call["w1"] is w1 and call["w2"] is w2
    assert call["w1_scale"] is w1_scale and call["w2_scale"] is w2_scale
    assert call["a1_scale"] is a1q_scale
    torch.testing.assert_close(
        call["expert_map"],
        torch.tensor([1, 1, 1, 0], dtype=torch.int32),
    )
    assert call["global_num_experts"] == 3
    assert call["inplace"] is False
    assert call["use_weight_shuffle"] is False
    assert call["output_dtype"] is torch.bfloat16
    assert call["topk_weights"].dtype is torch.float32
    assert call["topk_ids"].dtype is torch.int32


@pytest.mark.parametrize(
    ("use_fp8", "use_int8"),
    [(True, False), (False, True)],
)
@pytest.mark.parametrize(
    ("solution_type", "expected_map"),
    [
        ("ASM", [1, 0, 1, 0, 0]),
        ("MOE_C", [-1, 0, 1, -1]),
    ],
)
def test_quantized_aiter_runtime_converts_only_asm_ep_map_to_binary_mask(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    solution_type: str,
    expected_map: list[int],
):
    calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution_type,
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        calls.append(kwargs)
        return torch.zeros((2, 4))

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )

    def native_fp8_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del scale, quant_dtype, num_rows, num_rows_factor
        return x, torch.ones((*x.shape[:-1], 1), dtype=torch.float32)

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            per_token_quant_int8=lambda x: (x, torch.ones((x.shape[0], 1))),
            per_token_quant_hip=native_fp8_quant,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=native_fp8_quant,
        ),
    )

    expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int64)
    expert_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32)
    expert_mask._vllm_hcu_native_expert_map = expert_map
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=torch.ones((2, 8, 1)),
        w2_scale=torch.ones((2, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )
    compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        w1=torch.zeros((2, 8, 4), dtype=torch.int8),
        w2=torch.zeros((2, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(num_experts=4),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=expert_mask,
        quant_config=quant_config,
    )

    passed_map = calls[0]["expert_map"]
    assert isinstance(passed_map, torch.Tensor)
    expected_dtype = torch.int32 if solution_type == "ASM" else torch.int64
    assert passed_map.dtype is expected_dtype
    torch.testing.assert_close(
        passed_map.cpu(),
        torch.tensor(expected_map, dtype=passed_map.dtype),
    )
    if solution_type == "ASM":
        assert passed_map is expert_mask
    else:
        assert passed_map is expert_map


@pytest.mark.parametrize(
    ("runtime_kwargs", "message"),
    [
        ({"num_local_tokens": torch.tensor([2], dtype=torch.int32)}, "num_local_tokens"),
        ({"moe_sorting_dispatch_policy": 7}, "moe_sorting_dispatch_policy"),
    ],
)
def test_quantized_aiter_runtime_rejects_unsupported_parallel_metadata(
    runtime_kwargs: dict[str, object],
    message: str,
):
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((2, 8, 1)),
        w2_scale=torch.ones((2, 4, 1)),
        block_shape=None,
    )
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match=message,
    ):
        compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
            w1=torch.zeros((2, 8, 4), dtype=torch.int8),
            w2=torch.zeros((2, 4, 4), dtype=torch.int8),
            topk_weights=torch.ones((2, 2)),
            topk_ids=torch.zeros((2, 2), dtype=torch.int64),
            vllm_moe_config=SimpleNamespace(num_experts=2),
            activation=SimpleNamespace(value="silu"),
            apply_router_weight_on_input=False,
            expert_map=None,
            quant_config=quant_config,
            **runtime_kwargs,
        )


@pytest.mark.parametrize(
    ("use_fp8", "use_int8", "solution_type", "expected"),
    [
        (False, True, "asm", "aligned"),
        (True, False, "asm", "native"),
        (False, True, "moe_c", "native"),
    ],
)
def test_quantized_aiter_runtime_scopes_boltops_quant_to_int8_asm(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    solution_type: str,
    expected: str,
):
    observed_quantizers: list[str] = []
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def native_quant(x):
        del x
        return "native"

    def boltops_quant(x):
        del x
        return "aligned"

    def native_activation(
        activation,
        is_gated,
        activated_out,
        ffn1_out_2d,
        gemm1_alpha,
        gemm1_limit,
    ):
        del (
            activation,
            is_gated,
            activated_out,
            ffn1_out_2d,
            gemm1_alpha,
            gemm1_limit,
        )

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_int8=native_quant,
        _apply_activation=native_activation,
        per_token_quant_hip=_fp8_quant_abi_stub,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        asm_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_quant,
        ),
    )

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution_type,
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        observed_quantizers.append(
            asm_module.per_token_quant_int8(kwargs["hidden_states"])
        )
        return kwargs["hidden_states"].clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    output = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=hidden_states,
        w1=torch.zeros((3, 8, 4), dtype=torch.int8),
        w2=torch.zeros((3, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(num_experts=3),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=None,
        quant_config=quant_config,
    )

    torch.testing.assert_close(output, hidden_states)
    assert observed_quantizers == [expected]


@pytest.mark.parametrize(
    ("use_fp8", "use_int8", "solution_type", "activation", "expected"),
    [
        (True, False, "asm", "silu", 1.0),
        (False, True, "asm", "silu", 1.0),
        (True, False, "moe_c", "silu", 1.0),
        (True, False, "asm", "gelu", 1.0),
    ],
)
def test_quantized_aiter_runtime_scopes_boltops_quant_to_both_fp8_asm_stages(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    solution_type: str,
    activation: str,
    expected: float,
):
    calls: list[str] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def native_activation(
        activation,
        is_gated,
        activated_out,
        ffn1_out_2d,
        gemm1_alpha,
        gemm1_limit,
    ):
        del activation, is_gated, ffn1_out_2d, gemm1_alpha, gemm1_limit
        calls.append("aiter_activation")
        activated_out.fill_(1)

    def native_fp8_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del scale, quant_dtype, num_rows, num_rows_factor
        calls.append("aiter_fp8_quant")
        return x, torch.ones((x.shape[0], 1))

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        _apply_activation=native_activation,
        per_token_quant_int8=lambda x: (x, torch.ones((x.shape[0], 1))),
        per_token_quant_hip=native_fp8_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)

    def boltops_fp8_quant(x, scale=None, quant_dtype=torch.int8, **kwargs):
        assert scale is None
        assert quant_dtype == torch.float8_e4m3fn
        assert kwargs == {}
        calls.append("boltops_fp8_quant")
        return x, torch.ones((x.shape[0], 1))

    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_fp8_quant,
        ),
    )

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution_type,
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        if use_fp8:
            asm_module.per_token_quant_hip(
                kwargs["hidden_states"],
                quant_dtype=torch.float8_e4m3fn,
            )
        output = torch.empty((2, 4))
        asm_module._apply_activation(
            activation=kwargs["activation"],
            is_gated=True,
            activated_out=output,
            ffn1_out_2d=torch.empty((2, 8)),
            gemm1_alpha=None,
            gemm1_limit=None,
        )
        if use_fp8:
            asm_module.per_token_quant_hip(
                output,
                quant_dtype=torch.float8_e4m3fn,
            )
        return output

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    output = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        w1=torch.zeros((3, 8, 4), dtype=torch.int8),
        w2=torch.zeros((3, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(num_experts=3),
        activation=SimpleNamespace(value=activation),
        apply_router_weight_on_input=False,
        expert_map=None,
        quant_config=quant_config,
    )

    torch.testing.assert_close(output, torch.full_like(output, expected))
    if use_fp8:
        if solution_type == "asm":
            assert calls == [
                "boltops_fp8_quant",
                "aiter_activation",
                "boltops_fp8_quant",
            ]
        else:
            assert calls == [
                "aiter_fp8_quant",
                "aiter_activation",
                "aiter_fp8_quant",
            ]


def test_quantized_aiter_runtime_caches_config_and_invalidates_shuffled_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    config_calls: list[dict[str, object]] = []
    shuffle_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    kernel_calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        config_calls.append(kwargs)
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="moe_c",
            need_shuffle=True,
        )

    def shuffle_weights(w1, w2, config):
        assert config.solution_type == "moe_c"
        shuffle_calls.append((w1, w2))
        return w1.clone(), w2.clone()

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        return kwargs["hidden_states"].clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe_shfl_weight=shuffle_weights,
            aiter_moe=aiter_moe,
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    w1 = torch.zeros((3, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((3, 4, 4), dtype=torch.int8)
    topk_weights = torch.ones((2, 2))
    topk_ids = torch.zeros((2, 2), dtype=torch.int64)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    def run(x=hidden_states, weights=topk_weights, ids=topk_ids):
        return compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=weights,
            topk_ids=ids,
            vllm_moe_config=SimpleNamespace(num_experts=3),
            activation=SimpleNamespace(value="silu"),
            apply_router_weight_on_input=False,
            expert_map=None,
            quant_config=quant_config,
        )

    run()
    run()
    assert len(config_calls) == 1
    assert len(shuffle_calls) == 1
    assert kernel_calls[0]["w1"] is kernel_calls[1]["w1"]
    assert kernel_calls[0]["use_weight_shuffle"] is True

    w1.add_(1)
    run()
    assert len(config_calls) == 1
    assert len(shuffle_calls) == 2
    assert kernel_calls[2]["w1"] is not kernel_calls[1]["w1"]

    larger_x = torch.ones((3, 4), dtype=torch.bfloat16)
    run(larger_x, torch.ones((3, 2)), torch.zeros((3, 2), dtype=torch.int64))
    assert len(config_calls) == 2


@pytest.mark.parametrize(
    ("invalid_case", "message"),
    [
        ("topk_shape", "matching rank-2 top-k"),
        ("router_weight", "apply_router_weight_on_input=True"),
        ("block_quant", "channel/token W8A8"),
        ("ambiguous_quant", "exactly one FP8-W8A8 or INT8-W8A8"),
    ],
)
def test_quantized_aiter_runtime_rejects_invalid_explicit_contracts(
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
    message: str,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="asm",
            need_shuffle=False,
        )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=lambda **kwargs: kwargs["hidden_states"].clone(),
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 2))
    topk_ids = torch.zeros((2, 2), dtype=torch.int64)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )
    apply_router_weight_on_input = False
    if invalid_case == "topk_shape":
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
    elif invalid_case == "router_weight":
        apply_router_weight_on_input = True
    elif invalid_case == "block_quant":
        quant_config.block_shape = [128, 128]
    elif invalid_case == "ambiguous_quant":
        quant_config.use_int8_w8a8 = False

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match=message,
    ):
        compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states=hidden_states,
            w1=torch.zeros((3, 8, 4), dtype=torch.int8),
            w2=torch.zeros((3, 4, 4), dtype=torch.int8),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            vllm_moe_config=SimpleNamespace(num_experts=3),
            activation=SimpleNamespace(value="silu"),
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=None,
            quant_config=quant_config,
        )


def test_quantized_aiter_runtime_no_solution_falls_back_to_vllm_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **kwargs: (False, None),
            aiter_moe=lambda **kwargs: pytest.fail(
                "no-solution must not execute AITER"
            ),
        ),
    )
    expected = torch.full((2, 4), 6.0)
    fallback_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )

    def fallback(*args: object, **kwargs: object):
        fallback_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(fused_moe_module, "fused_experts_impl", fallback)
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    w1 = torch.zeros((3, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((3, 4, 4), dtype=torch.int8)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )
    native_expert_map = torch.tensor([0, -1, 1], dtype=torch.int32)
    expert_mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    expert_mask._vllm_hcu_native_expert_map = native_expert_map

    actual = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(num_experts=3),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=expert_mask,
        quant_config=quant_config,
    )

    assert actual is expected
    assert fallback_calls[0][0][1] is w1
    assert fallback_calls[0][0][2] is w2
    assert fallback_calls[0][1]["use_int8_w8a8"] is True
    assert fallback_calls[0][1]["per_channel_quant"] is True
    assert fallback_calls[0][1]["expert_map"] is native_expert_map


def test_quantized_aiter_runtime_config_fault_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    class MoeQuantType:
        W8A8 = "int8_w8a8"

    def config_fault(**kwargs: object):
        raise RuntimeError("aiter config fault")

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=config_fault,
        ),
    )
    fallback_calls: list[object] = []
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )
    monkeypatch.setattr(
        fused_moe_module,
        "fused_experts_impl",
        lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        block_shape=None,
    )

    with pytest.raises(RuntimeError, match="aiter config fault"):
        compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
            w1=torch.zeros((3, 8, 4), dtype=torch.int8),
            w2=torch.zeros((3, 4, 4), dtype=torch.int8),
            topk_weights=torch.ones((2, 2)),
            topk_ids=torch.zeros((2, 2), dtype=torch.int64),
            vllm_moe_config=SimpleNamespace(num_experts=3),
            activation=SimpleNamespace(value="silu"),
            apply_router_weight_on_input=False,
            expert_map=None,
            quant_config=quant_config,
        )
    assert fallback_calls == []


def test_moe_fp8_hcu_deepgemm_restores_layouts_after_kernel_recreation(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.fused_moe.experts import (
        dpsk_v4_deep_gemm_moe as deepgemm_module,
    )

    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    method_class.selected_backend = "HCU_DEEPGEMM"
    created_experts = []

    def make_auto_experts():
        experts = object.__new__(deepgemm_module.DeepEPAutoDeepGemmExperts)
        experts.ht_experts = object.__new__(
            deepgemm_module.DeepEPDeepGemmContiguousExperts
        )
        experts.ll_experts = object.__new__(
            deepgemm_module.DeepEPDeepGemmMaskedExperts
        )
        for child in (experts.ht_experts, experts.ll_experts):
            child._deepgemm_w13 = None
            child._deepgemm_w2 = None
        created_experts.append(experts)
        return experts

    def upstream_process(self, layer):
        layer.upstream_process_count += 1
        self.moe_kernel = SimpleNamespace(fused_experts=make_auto_experts())

    method_class.process_weights_after_loading = upstream_process
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="deep_gemm"),
    )

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros((1, 128, 64), dtype=torch.int8),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros((1, 64, 64), dtype=torch.int8),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(torch.ones((1, 128)))
    layer.w2_weight_scale = torch.nn.Parameter(torch.ones((1, 64)))
    layer.weight_block_size = None
    layer.upstream_process_count = 0

    ht_w13 = torch.full((1, 1, 8, 4, 16, 16), 31, dtype=torch.int8)
    ht_w2 = torch.full((1, 1, 4, 4, 16, 16), 32, dtype=torch.int8)
    ll_w13 = torch.full((1, 1, 8, 4, 16, 16), 47, dtype=torch.int8)
    ll_w2 = torch.full((1, 1, 4, 4, 16, 16), 48, dtype=torch.int8)
    monkeypatch.setattr(
        deepgemm_module,
        "marlin_fp8_contiguous_weight",
        lambda weight: ht_w13.clone() if weight.size(1) == 128 else ht_w2.clone(),
    )
    monkeypatch.setattr(
        deepgemm_module,
        "marlin_fp8_masked_weight",
        lambda weight: ll_w13.clone() if weight.size(1) == 128 else ll_w2.clone(),
    )

    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)

    assert layer.upstream_process_count == 2
    assert len(created_experts) == 2
    replacement = created_experts[-1]
    assert replacement.ht_experts._deepgemm_w13 is layer.w13_weight
    assert replacement.ht_experts._deepgemm_w2 is layer.w2_weight
    assert torch.equal(replacement.ll_experts._deepgemm_w13, ll_w13)
    assert torch.equal(replacement.ll_experts._deepgemm_w2, ll_w2)


def _fake_moe_wna16_module():
    attrs_calls: list[tuple[torch.Tensor, dict[str, object]]] = []
    config_calls: list[dict[str, object]] = []

    def set_weight_attrs(weight, attrs):
        attrs_calls.append((weight, dict(attrs)))

    def config_builder(**kwargs):
        config_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    class CompressedTensorsWNA16MoEMethod:
        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            del params_dtype, extra_weight_attrs
            layer.register_parameter(
                "w13_weight_packed",
                torch.nn.Parameter(
                    torch.empty(num_experts, hidden_size, 1), requires_grad=False
                ),
            )
            layer.register_parameter(
                "w2_weight_packed",
                torch.nn.Parameter(
                    torch.empty(
                        num_experts, intermediate_size_per_partition, 1
                    ),
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                "w13_weight_scale",
                torch.nn.Parameter(torch.ones(1), requires_grad=False),
            )
            layer.register_parameter(
                "w2_weight_scale",
                torch.nn.Parameter(torch.ones(1), requires_grad=False),
            )
            return "upstream-create"

        def get_fused_moe_quant_config(self, layer):
            return ("upstream-config", layer)

    module = _module(
        patch_compressed_tensors_moe_wna16.TARGET_MODULE,
        CompressedTensorsWNA16MoEMethod=CompressedTensorsWNA16MoEMethod,
        set_weight_attrs=set_weight_attrs,
        int4_w4a16_moe_quant_config=config_builder,
    )
    return module, attrs_calls, config_calls


def _wna16_method(module, *, gated: bool, num_bits: int = 4):
    method = module.CompressedTensorsWNA16MoEMethod()
    method.num_bits = num_bits
    method.group_size = 4
    method.strategy = "group"
    method.moe = SimpleNamespace(is_act_and_mul=gated)
    return method


def test_moe_wna16_feature_off_delegates_exactly(
    monkeypatch: pytest.MonkeyPatch,
):
    module, _, config_calls = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda _layer=None: False,
    )
    assert patch_compressed_tensors_moe_wna16.apply_to_module(module) is True
    assert patch_compressed_tensors_moe_wna16.apply_to_module(module) is False
    method = _wna16_method(module, gated=True)
    layer = torch.nn.Module()
    assert method.create_weights(layer, 2, 16, 24, torch.bfloat16) == (
        "upstream-create"
    )
    assert not hasattr(layer, "w13_qzeros") and not hasattr(layer, "w2_qzeros")
    assert method.get_fused_moe_quant_config(layer) == ("upstream-config", layer)
    assert config_calls == []


@pytest.mark.parametrize("gated,shards", [(False, 1), (True, 2)])
def test_moe_wna16_allocates_correct_initialized_qzeros(
    monkeypatch: pytest.MonkeyPatch,
    gated: bool,
    shards: int,
):
    module, attrs_calls, _ = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda _layer=None: True,
    )
    patch_compressed_tensors_moe_wna16.apply_to_module(module)
    method = _wna16_method(module, gated=gated)
    layer = torch.nn.Module()
    method.create_weights(layer, 2, 16, 24, torch.bfloat16, loader="test")
    # AITER/vLLM pack two output-channel zero points per byte, while the
    # K/group axis remains unpacked.
    assert layer.w13_qzeros.shape == (2, shards * 12, 4)
    assert layer.w2_qzeros.shape == (2, 8, 6)
    assert layer.w13_qzeros.dtype is torch.uint8
    assert torch.all(layer.w13_qzeros == 0x88)
    assert torch.all(layer.w2_qzeros == 0x88)
    assert attrs_calls[-2][1] == {
        "loader": "test",
        "is_transposed": True,
        "quant_method": "group",
    }


def test_moe_wna16_quant_config_requires_registered_qzeros(
    monkeypatch: pytest.MonkeyPatch,
):
    module, _, config_calls = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda _layer=None: True,
    )
    patch_compressed_tensors_moe_wna16.apply_to_module(module)
    method = _wna16_method(module, gated=True)
    prewarm_calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a16_moe",
        lambda owner, layer, config: prewarm_calls.append(
            (owner, layer, config)
        ),
        raising=False,
    )
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="w13_weight_scale",
    ):
        method.get_fused_moe_quant_config(torch.nn.Module())

    layer = torch.nn.Module()
    method.create_weights(layer, 2, 16, 24, torch.bfloat16)
    config = method.get_fused_moe_quant_config(layer)
    assert config.w1_zp is layer.w13_qzeros
    assert config.w2_zp is layer.w2_qzeros
    assert config.block_shape == [0, 4]
    assert len(config_calls) == 1
    assert prewarm_calls == [(method, layer, config)]

    invalid = _wna16_method(module, gated=True, num_bits=8)
    invalid_layer = torch.nn.Module()
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="requires 4-bit",
    ):
        invalid.create_weights(invalid_layer, 2, 16, 24, torch.bfloat16)


def _fake_fp8_scheme_module():
    channel = object()

    class CompressedTensorsW8A8Fp8:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(
                layer.weight.t(), requires_grad=False
            )
            layer.weight.input_dim = 0
            layer.weight.output_dim = 1

        def apply_weights(self, layer, x, bias=None):
            return self.fp8_linear.apply_weights(layer, x, bias)

    return _module(
        patch_compressed_tensors_w8a8_fp8.TARGET_MODULE,
        CompressedTensorsW8A8Fp8=CompressedTensorsW8A8Fp8,
        QuantizationStrategy=SimpleNamespace(CHANNEL=channel),
    ), channel


def test_fp8_channel_weight_layout_requires_hcu_kernel(monkeypatch: pytest.MonkeyPatch):
    module, channel = _fake_fp8_scheme_module()
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)
    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = object()
    with pytest.raises(RuntimeError, match="target Triton scaled-mm adapter"):
        scheme.process_weights_after_loading(SimpleNamespace(weight=torch.ones(2, 3)))


def test_fp8_target_triton_route_is_independent_of_general_custom_gemm_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", False)
    module, channel = _fake_fp8_scheme_module()
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)

    class Kernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = Kernel()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 3)))
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (3, 2)
    assert layer.weight.stride() == (1, 3)


def test_fp8_scheme_forwards_prequantized_input(monkeypatch: pytest.MonkeyPatch):
    module, channel = _fake_fp8_scheme_module()
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)
    calls: list[tuple[object, ...]] = []

    class Kernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def supports_quanted_inputs(self):
            return True

        def apply_weights(self, *args, **kwargs):
            calls.append((*args, kwargs))
            return "fp8"

    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = Kernel()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.arange(6.0).view(2, 3)))
    original = layer.weight.detach().clone()
    scheme.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight, original.t())
    assert layer.weight.stride() == (1, 3)
    assert layer.weight.input_dim == 0 and layer.weight.output_dim == 1
    assert scheme.supports_quanted_inputs() is True
    pair = (torch.ones(1, 3, dtype=torch.int8), torch.ones(1, 1))
    assert (
        scheme.apply_weights(layer, torch.ones(1, 3), x_and_scale_quanted=pair)
        == "fp8"
    )
    assert calls[0][-1]["x_and_scale_quanted"] is pair


def test_int8_hcu_owned_kernel_validates_and_computes_shapes(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", False)

    def gemm(a, b, scale_a, scale_b, m, n, k, flag, out_dtype):
        assert (m, n, k, flag) == (2, 2, 3, "NT")
        output = (a.float() * scale_a) @ (b.float() * scale_b).t()
        return True, output.to(out_dtype)

    lightop = _module("lightop")
    lightop.__path__ = []
    gemm_ops = _module("lightop.gemm_ops", hipblaslt_w8a8_gemm=gemm)
    lightop.gemm_ops = gemm_ops
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", gemm_ops)
    x = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    x_q = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.int8)
    x_scale = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
    weight = torch.tensor([[1, 0, -1], [2, 1, 0]], dtype=torch.int8)
    weight_scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    result = int8_runtime.apply_int8_linear(
        x,
        weight,
        weight_scale,
        torch.bfloat16,
        x_and_scale_quanted=(x_q, x_scale),
    )
    expected = (x_q.reshape(2, 3).float() * x_scale.reshape(2, 1)) @ (
        weight.float() * weight_scale
    ).t()
    torch.testing.assert_close(result.reshape(2, 2).float(), expected)
    with pytest.raises(
        int8_runtime.HcuInt8LinearError,
        match="quantized input/scale shapes",
    ):
        int8_runtime.apply_int8_linear(
            torch.ones(1, 2, 6, dtype=torch.bfloat16),
            weight,
            weight_scale,
            torch.bfloat16,
            x_and_scale_quanted=(x_q, x_scale),
        )
    with pytest.raises(int8_runtime.HcuInt8LinearError, match="symmetric"):
        int8_runtime.apply_int8_linear(
            x,
            weight,
            weight_scale,
            torch.bfloat16,
            input_zero_point=torch.ones(1, dtype=torch.int8),
            x_and_scale_quanted=(x_q, x_scale),
        )


def test_int8_linear_prefers_categorized_lightop_quant_and_gemm(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", False)
    calls: list[tuple[str, tuple[object, ...]]] = []

    def quant(value: torch.Tensor):
        calls.append(("quant", (value,)))
        return torch.ones_like(value, dtype=torch.int8), torch.ones(
            (*value.shape[:-1], 1), dtype=torch.float32
        )

    def gemm(*args: object):
        calls.append(("gemm", args))
        activation, weight, activation_scale, weight_scale, m, n, k, layout, dtype = args
        assert activation.shape == (m, k)
        assert weight.shape == (n, k)
        assert activation_scale.shape == (m, 1)
        assert weight_scale.shape == (n, 1)
        assert layout == "NT"
        return True, torch.full((m, n), 3, dtype=dtype)

    lightop = _module("lightop")
    lightop.__path__ = []
    quant_module = _module("lightop.quant", per_token_quant_int8=quant)
    gemm_module = _module("lightop.gemm_ops", hipblaslt_w8a8_gemm=gemm)
    lightop.quant = quant_module
    lightop.gemm_ops = gemm_module
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.quant", quant_module)
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", gemm_module)
    _reject_import_prefix(monkeypatch, "lmslim")

    input = torch.ones((1, 2, 3), dtype=torch.bfloat16)
    weight = torch.ones((4, 3), dtype=torch.int8)
    weight_scale = torch.ones((4, 1), dtype=torch.float32)
    actual = int8_runtime.apply_int8_linear(
        input, weight, weight_scale, torch.bfloat16
    )

    assert [name for name, _ in calls] == ["quant", "gemm"]
    assert actual.shape == (1, 2, 4)
    assert torch.equal(actual, torch.full_like(actual, 3))


def test_int8_quant_missing_categorized_export_fails_without_lmslim_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", False)
    lightop = _package("lightop")
    lightop.quant = _module("lightop.quant")
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.quant", lightop.quant)
    _reject_import_prefix(monkeypatch, "lmslim")

    with pytest.raises(
        int8_runtime.HcuInt8LinearError, match="lightop.quant"
    ):
        int8_runtime.apply_int8_linear(
            torch.ones((1, 3), dtype=torch.bfloat16),
            torch.ones((2, 3), dtype=torch.int8),
            torch.ones((2, 1), dtype=torch.float32),
            torch.bfloat16,
        )
def test_int8_gemm_missing_categorized_export_fails_without_lmslim_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", False)
    lightop = _package("lightop")
    lightop.gemm_ops = _module("lightop.gemm_ops")
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", lightop.gemm_ops)
    _reject_import_prefix(monkeypatch, "lmslim")
    activation = torch.ones((1, 3), dtype=torch.bfloat16)
    quantized = torch.ones_like(activation, dtype=torch.int8)
    activation_scale = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(
        int8_runtime.HcuInt8LinearError, match="lightop.gemm_ops"
    ):
        int8_runtime.apply_int8_linear(
            activation,
            torch.ones((2, 3), dtype=torch.int8),
            torch.ones((2, 1), dtype=torch.float32),
            torch.bfloat16,
            x_and_scale_quanted=(quantized, activation_scale),
        )


def _fake_int8_scheme_module():
    class CompressedTensorsW8A8Int8:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(
                layer.weight.t().contiguous(), requires_grad=False
            )

        def apply_weights(self, layer, x, bias):
            return ("upstream", layer, x, bias)

    return _module(
        patch_compressed_tensors_w8a8_int8.TARGET_MODULE,
        CompressedTensorsW8A8Int8=CompressedTensorsW8A8Int8,
    )


def test_int8_scheme_layout_and_feature_off_delegation(monkeypatch: pytest.MonkeyPatch):
    module = _fake_int8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: False,
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(module)
    scheme = module.CompressedTensorsW8A8Int8()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 3)))
    assert scheme.apply_weights(layer, "x", None)[0] == "upstream"
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (3, 2)

    feature_module = _fake_int8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        int8_runtime,
        "apply_int8_linear",
        lambda **kwargs: calls.append(kwargs) or "hcu",
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(feature_module)
    feature_scheme = feature_module.CompressedTensorsW8A8Int8()
    feature_layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(2, 3)),
        weight_scale=torch.ones(2, 1),
        params_dtype=torch.bfloat16,
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
    )
    feature_scheme.process_weights_after_loading(feature_layer)
    assert feature_layer.weight.shape == (2, 3)
    assert feature_layer.weight.is_contiguous()
    assert feature_scheme.apply_weights(feature_layer, "x", None) == "hcu"
    assert calls[0]["weight"] is feature_layer.weight


def test_int8_weight_layout_rolls_back_if_upstream_processing_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    class CompressedTensorsW8A8Int8:
        def process_weights_after_loading(self, layer):
            raise RuntimeError("packing failed")

        def apply_weights(self, layer, x, bias):
            return None

    module = _module(
        patch_compressed_tensors_w8a8_int8.TARGET_MODULE,
        CompressedTensorsW8A8Int8=CompressedTensorsW8A8Int8,
    )
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(module)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    )
    original_parameter = layer.weight
    original_value = layer.weight.detach().clone()
    with pytest.raises(RuntimeError, match="packing failed"):
        module.CompressedTensorsW8A8Int8().process_weights_after_loading(layer)
    assert layer.weight is original_parameter
    assert layer.weight.shape == (2, 3)
    torch.testing.assert_close(layer.weight, original_value)


def test_lightop_fp8_registration_is_single_owner_and_latched():
    lightop_fp8_runtime._reset_for_tests()
    calls: list[dict[str, object]] = []
    lightop_fp8_runtime.ensure_registered(
        torch.float8_e4m3fn,
        lambda **kwargs: calls.append(kwargs),
    )
    lightop_fp8_runtime.ensure_registered(
        torch.float8_e4m3fn,
        lambda **kwargs: calls.append(kwargs),
    )
    assert len(calls) == 1
    assert calls[0]["op_name"] == "lightop_per_token_quant_fp8"

    lightop_fp8_runtime._reset_for_tests()
    with pytest.raises(lightop_fp8_runtime.HcuLightOpRegistrationError):
        lightop_fp8_runtime.ensure_registered(
            torch.float8_e4m3fn,
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("duplicate")),
        )
    with pytest.raises(
        lightop_fp8_runtime.HcuLightOpRegistrationError,
        match="previously failed",
    ):
        lightop_fp8_runtime.ensure_registered(
            torch.float8_e4m3fn, lambda **kwargs: None
        )
    lightop_fp8_runtime._reset_for_tests()


def test_lightop_fp8_adapter_has_no_import_time_registration():
    lightop_fp8_runtime._reset_for_tests()
    module, _ = _fake_input_quant_module()
    patch_input_quant_fp8.apply_to_module(module)
    assert lightop_fp8_runtime._REGISTERED is False
    assert lightop_fp8_runtime._REGISTRATION_ERROR is None


def _fake_input_quant_module():
    per_token = object()

    class QuantFP8:
        def forward_cuda(self, x, scale=None, scale_ub=None, use_triton=False):
            return ("cuda", x, scale, scale_ub, use_triton)

        def forward_native(self, x, scale=None, scale_ub=None, use_triton=False):
            return ("native", x, scale, scale_ub, use_triton)

    return _module(
        patch_input_quant_fp8.TARGET_MODULE,
        QuantFP8=QuantFP8,
        GroupShape=SimpleNamespace(PER_TOKEN=per_token),
        _FP8_DTYPE=torch.float8_e4m3fn,
    ), per_token


@pytest.mark.parametrize("method_name", ["forward_cuda", "forward_native"])
def test_quant_fp8_eligibility_and_feature_off(
    monkeypatch: pytest.MonkeyPatch, method_name: str
):
    _install_fake_vllm_torch_utils(monkeypatch)
    module, per_token = _fake_input_quant_module()
    patch_input_quant_fp8.apply_to_module(module)
    instance = module.QuantFP8()
    instance.group_shape = per_token
    instance.num_token_padding = None
    x = torch.ones(2, 4)
    monkeypatch.setattr(patch_input_quant_fp8, "_lightop_requested", lambda: True)
    monkeypatch.setattr(
        lightop_fp8_runtime,
        "quantize",
        lambda value, dtype, register: ("lightop", value, dtype),
    )
    method = getattr(instance, method_name)
    assert method(x)[0] == "lightop"
    assert method(x.t())[0] == method_name.removeprefix("forward_")
    monkeypatch.setattr(patch_input_quant_fp8, "_lightop_requested", lambda: False)
    assert method(x)[0] == method_name.removeprefix("forward_")


def test_weight8bit_marlin2_layout_2d_3d_and_validation():
    module = _module(patch_w8a8_utils.TARGET_MODULE)
    patch_w8a8_utils.apply_to_module(module)
    assert (
        module.weight8bit_nt_kpack2_marlin2
        is int8_runtime.weight8bit_nt_kpack2_marlin2
    )
    weight = torch.arange(16 * 64, dtype=torch.int32).to(torch.int8).view(16, 64)
    result = module.weight8bit_nt_kpack2_marlin2(weight)
    reference = (
        weight.reshape(1, 16, 1, 4, 16)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .reshape(1, 1024)
    )
    torch.testing.assert_close(result, reference)
    experts = torch.stack((weight, weight + 1))
    result_3d = module.weight8bit_nt_kpack2_marlin2(experts)
    assert result_3d.shape == (2, 1, 1024)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        module.weight8bit_nt_kpack2_marlin2(torch.ones(16, dtype=torch.int8))


def test_marlin_moe_never_imports_lmslim(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    calls: list[tuple[str, dict[str, object]]] = []

    def fp8_kernel(**kwargs):
        calls.append(("fp8", kwargs))
        return kwargs["hidden_states"] + 2

    def int8_kernel(**kwargs):
        calls.append(("int8", kwargs))
        return kwargs["hidden_states"] + 3

    _install_lightop_moe(
        monkeypatch,
        fused_experts_impl_fp8_marlin=fp8_kernel,
        fused_experts_impl_int8_marlin=int8_kernel,
    )
    _reject_import_prefix(monkeypatch, "lmslim")
    monkeypatch.setattr(
        marlin, "_is_hcu_aiter_w8a8_moe_requested", lambda *args: False
    )
    layer = _fp8_moe_layer()
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)

    fp8_method = object.__new__(marlin.CompressedTensorsW8A8FP8MarlinMoEMethod)
    fp8_output = fp8_method.fused_moe_forward(layer, x, weights, ids)
    torch.testing.assert_close(fp8_output, x + 2)
    assert calls[0][1]["hidden_states"] is x
    assert calls[0][1]["topk_ids"] is ids
    assert calls[0][1]["use_fp8_w8a8"] is True

    int8_method = object.__new__(marlin.CompressedTensorsW8A8Int8MarlinMoEMethod)
    int8_method.moe = None
    int8_method.moe_quant_config = "int8-config"
    int8_output = int8_method.apply(layer, x, weights, ids, None, None)
    torch.testing.assert_close(int8_output, x + 3)
    assert calls[1][1]["hidden_states"] is x
    assert calls[1][1]["topk_ids"] is ids
    assert calls[1][1]["quant_config"] == "int8-config"
    assert calls[1][1]["use_int8_w8a8"] is True
    assert [kind for kind, _ in calls] == ["fp8", "int8"]


def test_missing_lightop_marlin_export_fails_without_lmslim_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    lightop = _package("lightop")
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.delitem(sys.modules, "lightop.moe", raising=False)
    _reject_import_prefix(monkeypatch, "lmslim")
    monkeypatch.setattr(marlin, "_is_hcu_aiter_w8a8_moe_requested", lambda: False)

    layer = _fp8_moe_layer()
    x = torch.ones(2, 4)
    with pytest.raises((ImportError, AttributeError)):
        object.__new__(marlin.CompressedTensorsW8A8FP8MarlinMoEMethod).fused_moe_forward(
            layer,
            x,
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int64),
        )


def test_slimquant_marlin_module_imports_before_worker_patch():
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["PYTHONPATH"] = os.pathsep.join((str(repo), env.get("PYTHONPATH", "")))
    script = """
from vllm.model_executor.layers.quantization.utils import w8a8_utils
assert not hasattr(w8a8_utils, "weight8bit_nt_kpack2_marlin2")
from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe_marlin,
)
from vllm_hcu.model_executor.layers.quantization import int8_runtime
from vllm.model_executor.layers.fused_moe import config as fused_moe_config
assert (
    compressed_tensors_moe_marlin.weight8bit_nt_kpack2_marlin2
    is int8_runtime.weight8bit_nt_kpack2_marlin2
)
calls = []
def hcu_int8_config(**kwargs):
    calls.append(kwargs)
    return "HCU_INT8_CONFIG"
fused_moe_config.int8_w8a8_moe_quant_config = hcu_int8_config
method = object.__new__(
    compressed_tensors_moe_marlin.CompressedTensorsW8A8Int8MarlinMoEMethod
)
layer = type("Layer", (), {
    "w13_weight_scale": object(),
    "w2_weight_scale": object(),
    "w13_input_scale": object(),
    "w2_input_scale": object(),
})()
assert method.get_fused_moe_quant_config(layer) == "HCU_INT8_CONFIG"
assert calls[0]["block_shape"] is None
print("SLIMQUANT_PREPATCH_IMPORT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SLIMQUANT_PREPATCH_IMPORT_OK" in result.stdout


def test_marlin_aiter_moe_no_solution_uses_native_triton_not_lmslim(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    method_class = marlin.CompressedTensorsW8A8Int8MarlinMoEMethod
    assert not hasattr(method_class, "_get_aiter_moe_runtime_config")
    assert not hasattr(method_class, "_get_aiter_weights_for_solution")
    method = object.__new__(method_class)
    method.moe = SimpleNamespace(num_experts=3)
    method.moe_quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )
    monkeypatch.setattr(
        marlin,
        "_is_hcu_aiter_w8a8_moe_requested",
        lambda _moe=None: True,
    )
    monkeypatch.setattr(marlin.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)

    class MoeQuantType:
        W8A8 = "int8_w8a8"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=lambda **kwargs: (False, None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "lmslim.layers.fused_moe.fuse_moe_int8_marlin",
        _module(
            "lmslim.layers.fused_moe.fuse_moe_int8_marlin",
            fused_experts_impl_int8_marlin=lambda **kwargs: pytest.fail(
                "AITER no-solution must use vLLM Triton, not LMSlim"
            ),
        ),
    )
    expected = torch.full((2, 4), 4.0)
    fallback_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )

    def fallback(*args: object, **kwargs: object):
        fallback_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(fused_moe_module, "fused_experts_impl", fallback)
    layer = _fp8_moe_layer()
    layer.w13_weight = torch.zeros((3, 8, 4), dtype=torch.int8)
    layer.w2_weight = torch.zeros((3, 4, 4), dtype=torch.int8)

    actual = method.apply(
        layer,
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.ones((2, 2)),
        torch.zeros((2, 2), dtype=torch.int64),
        None,
        None,
    )

    assert actual is expected
    assert fallback_calls[0][1]["use_int8_w8a8"] is True


def test_marlin_explicit_triton_backend_bypasses_aiter_env(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )

    assert not marlin._is_hcu_aiter_w8a8_moe_requested(
        SimpleNamespace(moe_backend="triton")
    )


def test_marlin_aiter_moe_prewarms_m1_during_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    method = object.__new__(marlin.CompressedTensorsW8A8Int8MarlinMoEMethod)
    method.moe = SimpleNamespace(
        num_experts=3,
        experts_per_token=2,
        in_dtype=torch.bfloat16,
        activation=SimpleNamespace(value="silu"),
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        block_shape=None,
    )
    method.get_fused_moe_quant_config = lambda unused_layer: quant_config
    monkeypatch.setattr(
        marlin,
        "_is_hcu_aiter_w8a8_moe_requested",
        lambda _moe=None: True,
    )
    monkeypatch.setattr(marlin.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_quantized_moe",
        lambda layer, moe, config: calls.append((layer, moe, config)),
        raising=False,
    )
    layer = _fp8_moe_layer()
    layer.w13_weight = torch.zeros((3, 8, 4), dtype=torch.int8)
    layer.w2_weight = torch.zeros((3, 4, 4), dtype=torch.int8)

    method.process_weights_after_loading(layer)

    assert method.moe_quant_config is quant_config
    assert calls == [(layer, method.moe, quant_config)]


def test_marlin_aiter_moe_config_fault_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as marlin,
    )

    method = object.__new__(marlin.CompressedTensorsW8A8Int8MarlinMoEMethod)
    method.moe = SimpleNamespace(num_experts=3)
    method.moe_quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        block_shape=None,
    )
    monkeypatch.setattr(
        marlin,
        "_is_hcu_aiter_w8a8_moe_requested",
        lambda _moe=None: True,
    )
    monkeypatch.setattr(marlin.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)

    class MoeQuantType:
        W8A8 = "int8_w8a8"

    def config_fault(**kwargs: object):
        raise RuntimeError("aiter config fault")

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=config_fault,
        ),
    )
    fallback_calls: list[object] = []
    fused_moe_module = __import__(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fromlist=["fused_experts_impl"],
    )
    monkeypatch.setattr(
        fused_moe_module,
        "fused_experts_impl",
        lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
    )
    layer = _fp8_moe_layer()
    layer.w13_weight = torch.zeros((3, 8, 4), dtype=torch.int8)
    layer.w2_weight = torch.zeros((3, 4, 4), dtype=torch.int8)

    with pytest.raises(RuntimeError, match="aiter config fault"):
        method.apply(
            layer,
            torch.ones((2, 4), dtype=torch.bfloat16),
            torch.ones((2, 2)),
            torch.zeros((2, 2), dtype=torch.int64),
            None,
            None,
        )
    assert fallback_calls == []


@pytest.mark.parametrize("is_rocm", [False, True])
def test_unquantized_gemm_dispatch_only_changes_rocm(is_rocm: bool):
    default = lambda *args: "default"
    rocm = lambda *args: "rocm"
    calls = []

    def dispatch_unquantized_gemm(linear_backend="auto"):
        calls.append(linear_backend)
        return rocm if is_rocm else "other"

    module = _module(
        patch_layers_utils.TARGET_MODULE,
        current_platform=SimpleNamespace(is_rocm=lambda: is_rocm),
        default_unquantized_gemm=default,
        dispatch_unquantized_gemm=dispatch_unquantized_gemm,
    )
    patch_layers_utils.apply_to_module(module)
    assert module.dispatch_unquantized_gemm("flashinfer-cutlass") is (
        default if is_rocm else "other"
    )
    assert calls == ([] if is_rocm else ["flashinfer-cutlass"])


def test_tf32_hc_prenorm_cpu_fallback_and_backend_delegation():
    calls: list[str] = []

    def lazy_init():
        calls.append("lazy")

    def original(x, fn, out, sqrsum, num_split):
        calls.append("backend")
        return "backend"

    module = _module(
        patch_deep_gemm.TARGET_MODULE,
        torch=torch,
        _lazy_init=lazy_init,
        _tf32_hc_prenorm_gemm_impl=None,
        tf32_hc_prenorm_gemm=original,
    )
    patch_deep_gemm.apply_to_module(module)
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    fn = torch.tensor([[2.0, 1.0], [0.5, -2.0], [1.0, 1.0]])
    out = torch.empty(2, 2, 3)
    sqrsum = torch.empty(2, 2)
    result = module.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 2)
    assert result is out
    torch.testing.assert_close(out[0], x @ fn.t())
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]))
    torch.testing.assert_close(sqrsum[0], x.square().sum(-1))
    torch.testing.assert_close(sqrsum[1], torch.zeros_like(sqrsum[1]))

    module._tf32_hc_prenorm_gemm_impl = object()
    assert module.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 2) == "backend"
    assert calls[-1] == "backend"
