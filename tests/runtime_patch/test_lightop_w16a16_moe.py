# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import enum
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)

from vllm_hcu.platforms import envs as henvs


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _config(
    *, backend: str = "auto", max_num_tokens: int = 16
) -> SimpleNamespace:
    return SimpleNamespace(
        moe_backend=backend,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        hidden_dim=2048,
        intermediate_size_per_partition=512,
        num_experts=256,
        experts_per_token=8,
        max_num_tokens=max_num_tokens,
        is_act_and_mul=True,
        is_lora_enabled=False,
        has_bias=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        routing_method=RoutingMethodType.Default,
        router_logits_dtype=torch.bfloat16,
        moe_parallel_config=SimpleNamespace(
            dp_size=1,
            tp_size=1,
            ep_size=1,
            use_ep=False,
            use_all2all_kernels=False,
            use_batched_activation_format=False,
            use_fi_nvl_two_sided_kernels=False,
            use_fi_nvl_one_sided_kernels=False,
        ),
    )


def _target_module():
    class UnquantizedMoeBackend(enum.Enum):
        AITER = "ROCm AITER"
        TRITON = "TRITON"

    calls: list[tuple[str, object]] = []
    live: dict[str, ModuleType] = {}

    def backend_to_kernel_cls(backend):
        calls.append(("backend", backend))
        if not isinstance(backend, live["module"].UnquantizedMoeBackend):
            raise ValueError("backend enum does not match the live oracle enum")
        return "official-experts"

    def map_unquantized_backend(runner_backend):
        calls.append(("map", runner_backend))
        return live["module"].UnquantizedMoeBackend.AITER

    def select_unquantized_moe_backend(moe_config):
        calls.append(("select", moe_config))
        return live["module"].UnquantizedMoeBackend.AITER, "official-experts"

    def convert_to_unquantized_kernel_format(
        unquantized_backend,
        moe_config,
        w13_weight,
        w2_weight,
    ):
        del moe_config
        if not isinstance(
            unquantized_backend, live["module"].UnquantizedMoeBackend
        ):
            raise ValueError("converter received a stale oracle enum")
        calls.append(("convert", (w13_weight, w2_weight)))
        return w13_weight, w2_weight

    def make_unquantized_moe_kernel(
        quant_config,
        moe_config,
        backend,
        experts_cls,
        routing_tables=None,
    ):
        del quant_config, moe_config, backend, experts_cls, routing_tables
        return "official-kernel"

    module = _module(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized",
        Enum=enum.Enum,
        UnquantizedMoeBackend=UnquantizedMoeBackend,
        backend_to_kernel_cls=backend_to_kernel_cls,
        map_unquantized_backend=map_unquantized_backend,
        select_unquantized_moe_backend=select_unquantized_moe_backend,
        convert_to_unquantized_kernel_format=convert_to_unquantized_kernel_format,
        make_unquantized_moe_kernel=make_unquantized_moe_kernel,
        mk=SimpleNamespace(
            FusedMoEActivationFormat=SimpleNamespace(
                Standard=FusedMoEActivationFormat.Standard,
                BatchedExperts=FusedMoEActivationFormat.BatchedExperts,
            )
        ),
    )
    live["module"] = module
    return module, calls


def test_w16a16_packing_is_exact_non_mutating_and_idempotent() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.lightop_w16a16_runtime import (
        pack_lightop_w16a16_weights,
    )

    w13 = torch.arange(2 * 64 * 32, dtype=torch.float32).reshape(2, 64, 32)
    w2 = torch.arange(2 * 32 * 32, dtype=torch.float32).reshape(2, 32, 32)
    before13 = w13.clone()
    before2 = w2.clone()

    packed13, packed2, layout = pack_lightop_w16a16_weights(w13, w2)

    assert torch.equal(w13, before13)
    assert torch.equal(w2, before2)
    assert packed13.shape == (2, 2, 1024)
    assert packed2.shape == (2, 2, 512)
    assert layout.logical_w13_shape == (2, 64, 32)
    assert layout.logical_w2_shape == (2, 32, 32)
    assert layout.packed_w13_shape == tuple(packed13.shape)
    assert layout.packed_w2_shape == tuple(packed2.shape)
    assert getattr(packed13, "_hcu_lightop_w16a16_generation") == 1
    assert getattr(packed2, "_hcu_lightop_w16a16_generation") == 1

    repacked13, repacked2, repeated_layout = pack_lightop_w16a16_weights(
        packed13, packed2
    )
    assert repacked13 is packed13
    assert repacked2 is packed2
    assert repeated_layout == layout


def _run_mocked_w16a16_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_stage: str | None = None,
) -> list[str]:
    from vllm_hcu.model_executor.layers.fused_moe import lightop_w16a16_runtime

    layout = lightop_w16a16_runtime.LightopW16A16Layout(
        logical_w13_shape=(2, 32, 32),
        logical_w2_shape=(2, 32, 16),
        packed_w13_shape=(2, 2, 16),
        packed_w2_shape=(2, 1, 16),
    )
    w13 = torch.empty(layout.packed_w13_shape, dtype=torch.bfloat16)
    w2 = torch.empty(layout.packed_w2_shape, dtype=torch.bfloat16)
    lightop_w16a16_runtime.mark_lightop_w16a16_weights(w13, w2, layout)
    messages: list[str] = []

    def stage(name: str):
        def operator(*_args, **_kwargs) -> None:
            if fail_stage == name:
                raise RuntimeError(f"{name} failed")

        return operator

    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "_load_lightop_w16a16",
        lambda: SimpleNamespace(
            get_config=object(),
            align=stage("align"),
            gemm=stage("gemm"),
            activate=stage("activate"),
            reduce=stage("reduce"),
        ),
    )
    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "select_lightop_w16a16_config",
        lambda *_args, **_kwargs: (
            {"BLOCK_SIZE_M": 1},
            {"BLOCK_SIZE_M": 1},
        ),
    )
    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "logger",
        SimpleNamespace(warning_once=messages.append),
    )

    def call() -> None:
        lightop_w16a16_runtime.run_lightop_w16a16(
            output=torch.empty((1, 32), dtype=torch.bfloat16),
            hidden_states=torch.empty((1, 32), dtype=torch.bfloat16),
            w13=w13,
            w2=w2,
            topk_weights=torch.ones((1, 1), dtype=torch.float32),
            topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            workspace13=torch.empty((1, 32), dtype=torch.bfloat16),
            workspace2=torch.empty((1, 16), dtype=torch.bfloat16),
            global_num_experts=2,
        )
    if fail_stage is None:
        call()
    else:
        with pytest.raises(RuntimeError, match=f"{fail_stage} failed"):
            call()
    return messages


def test_w16a16_success_marker_requires_complete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_mocked_w16a16_pipeline(monkeypatch) == [
        "Using LightOp W16A16 Marlin MoE backend."
    ]


@pytest.mark.parametrize("fail_stage", ("align", "gemm", "activate", "reduce"))
def test_w16a16_failed_pipeline_does_not_emit_success_marker(
    monkeypatch: pytest.MonkeyPatch,
    fail_stage: str,
) -> None:
    assert _run_mocked_w16a16_pipeline(monkeypatch, fail_stage=fail_stage) == []


def test_w16a16_packing_does_not_emit_backend_success_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import lightop_w16a16_runtime

    messages: list[str] = []
    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "logger",
        SimpleNamespace(warning_once=messages.append),
    )
    lightop_w16a16_runtime.pack_lightop_w16a16_weights(
        torch.empty((2, 32, 32), dtype=torch.bfloat16),
        torch.empty((2, 32, 16), dtype=torch.bfloat16),
    )

    assert "Using LightOp W16A16 Marlin MoE backend." not in messages


@pytest.mark.parametrize(
    "w13_shape,w2_shape",
    (
        ((2, 64, 31), (2, 31, 32)),
        ((2, 62, 32), (2, 32, 31)),
        ((2, 64, 32), (3, 32, 32)),
    ),
)
def test_w16a16_packing_rejects_incompatible_logical_layouts(
    w13_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe.lightop_w16a16_runtime import (
        pack_lightop_w16a16_weights,
    )

    with pytest.raises(ValueError, match="W16A16"):
        pack_lightop_w16a16_weights(
            torch.empty(w13_shape, dtype=torch.bfloat16),
            torch.empty(w2_shape, dtype=torch.bfloat16),
        )


def test_w16a16_auto_selects_lightop_only_when_master_and_leaf_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import lightop_w16a16_runtime
    from vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe import (
        LightopW16A16Experts,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, calls = _target_module()
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", True, raising=False
    )
    monkeypatch.setattr(
        LightopW16A16Experts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )
    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "select_lightop_w16a16_config",
        lambda *args, **kwargs: ({"BLOCK_SIZE_M": 16}, {"BLOCK_SIZE_M": 16}),
    )

    assert patch_unquantized_oracle.apply_to_module(module) is True
    backend, experts = module.select_unquantized_moe_backend(_config())

    assert backend.name == "HCU_LIGHTOP_W16A16"
    assert experts is LightopW16A16Experts
    assert not [call for call in calls if call[0] == "select"]


def test_w16a16_backend_to_kernel_cls_preserves_official_list_contract() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe import (
        LightopW16A16Experts,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, _ = _target_module()
    assert patch_unquantized_oracle.apply_to_module(module) is True

    result = module.backend_to_kernel_cls(
        module.UnquantizedMoeBackend.HCU_LIGHTOP_W16A16
    )

    assert result == [LightopW16A16Experts]


@pytest.mark.parametrize(
    "binding",
    (
        "UnquantizedMoeBackend",
        "backend_to_kernel_cls",
        "map_unquantized_backend",
        "select_unquantized_moe_backend",
        "convert_to_unquantized_kernel_format",
        "make_unquantized_moe_kernel",
    ),
)
def test_w16a16_oracle_rejects_stale_applied_binding(binding: str) -> None:
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle
    from vllm_hcu.patch.worker.op_opt.moe._common import PatchCompatibilityError

    module, _ = _target_module()
    assert patch_unquantized_oracle.apply_to_module(module) is True
    setattr(module, binding, lambda: None)

    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch_unquantized_oracle.apply_to_module(module)


@pytest.mark.parametrize(
    "master,leaf,backend",
    (
        (False, True, "auto"),
        (True, False, "auto"),
        (True, True, "aiter"),
        (True, True, "triton"),
    ),
)
def test_w16a16_disabled_or_explicit_backend_preserves_official_selection(
    monkeypatch: pytest.MonkeyPatch,
    master: bool,
    leaf: bool,
    backend: str,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe import (
        LightopW16A16Experts,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, calls = _target_module()
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", leaf, raising=False
    )
    monkeypatch.setattr(
        LightopW16A16Experts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )
    patch_unquantized_oracle.apply_to_module(module)

    result = module.select_unquantized_moe_backend(_config(backend=backend))

    assert result[0].name == "AITER"
    assert [call for call in calls if call[0] == "select"]


def test_w16a16_no_config_preserves_original_selector_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import lightop_w16a16_runtime
    from vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe import (
        LightopW16A16Experts,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, calls = _target_module()
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", True, raising=False
    )
    monkeypatch.setattr(
        LightopW16A16Experts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )
    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "select_lightop_w16a16_config",
        lambda *args, **kwargs: None,
    )
    patch_unquantized_oracle.apply_to_module(module)
    w13 = torch.empty((2, 64, 32))
    w2 = torch.empty((2, 32, 32))

    backend, experts = module.select_unquantized_moe_backend(_config())
    converted = module.convert_to_unquantized_kernel_format(
        backend, _config(), w13, w2
    )

    assert backend.name == "AITER"
    assert experts == "official-experts"
    assert converted == (w13, w2)
    assert [call for call in calls if call[0] == "select"]
    assert [call for call in calls if call[0] == "convert"]


def test_w16a16_selection_requires_configs_for_every_reachable_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import lightop_w16a16_runtime
    from vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe import (
        LightopW16A16Experts,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, calls = _target_module()
    probed: list[int] = []
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", True, raising=False
    )
    monkeypatch.setattr(
        LightopW16A16Experts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )

    def select_config(_config, expected_m, device):
        del device
        probed.append(expected_m)
        if expected_m == 3:
            return None
        return {"BLOCK_SIZE_M": 16}, {"BLOCK_SIZE_M": 16}

    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "select_lightop_w16a16_config",
        select_config,
    )
    patch_unquantized_oracle.apply_to_module(module)

    backend, experts = module.select_unquantized_moe_backend(
        _config(max_num_tokens=4)
    )

    assert backend.name == "AITER"
    assert experts == "official-experts"
    assert probed == [1, 2, 3]
    assert [call for call in calls if call[0] == "select"]


def test_w16a16_unverified_token_range_preserves_official_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe import (
        LightopW16A16Experts,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, calls = _target_module()
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", True, raising=False
    )
    monkeypatch.setattr(
        LightopW16A16Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    patch_unquantized_oracle.apply_to_module(module)

    backend, experts = module.select_unquantized_moe_backend(
        _config(max_num_tokens=32)
    )

    assert backend.name == "AITER"
    assert experts == "official-experts"
    assert [call for call in calls if call[0] == "select"]
    supported, reason = LightopW16A16Experts.is_supported_config(
        LightopW16A16Experts,
        _config(max_num_tokens=32),
        None,
        None,
        FusedMoEActivationFormat.Standard,
    )
    assert supported is False
    assert "max_num_tokens" in str(reason)


def test_w16a16_ineligible_config_does_not_probe_lightop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe.experts import (
        lightop_w16a16_moe,
    )

    config = _config()
    config.in_dtype = torch.float16
    monkeypatch.setattr(
        lightop_w16a16_moe.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        lightop_w16a16_moe,
        "is_lightop_w16a16_available",
        lambda: pytest.fail("ineligible config must not probe LightOp"),
    )

    supported, reason = lightop_w16a16_moe.LightopW16A16Experts.is_supported_config(
        lightop_w16a16_moe.LightopW16A16Experts,
        config,
        None,
        None,
        FusedMoEActivationFormat.Standard,
    )

    assert supported is False
    assert "BF16" in str(reason)


@pytest.mark.parametrize(
    ("attribute", "value"),
    (
        ("swiglu_limit", 7.0),
        ("swiglu_alpha", 1.25),
        ("swiglu_beta", 0.5),
    ),
)
def test_w16a16_modified_swiglu_is_rejected_before_lightop_probe(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: float,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe.experts import (
        lightop_w16a16_moe,
    )

    config = _config()
    setattr(config, attribute, value)
    monkeypatch.setattr(
        lightop_w16a16_moe.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        lightop_w16a16_moe,
        "is_lightop_w16a16_available",
        lambda: pytest.fail("modified SwiGLU must be rejected before LightOp probe"),
    )

    supported, reason = lightop_w16a16_moe.LightopW16A16Experts.is_supported_config(
        lightop_w16a16_moe.LightopW16A16Experts,
        config,
        None,
        None,
        FusedMoEActivationFormat.Standard,
    )

    assert supported is False
    assert "SwiGLU" in str(reason)


def test_w16a16_missing_lightop_preserves_official_backend_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe.experts import (
        lightop_w16a16_moe,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, calls = _target_module()
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", True, raising=False
    )
    monkeypatch.setattr(
        lightop_w16a16_moe,
        "is_lightop_w16a16_available",
        lambda: False,
    )
    monkeypatch.setattr(
        lightop_w16a16_moe.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    patch_unquantized_oracle.apply_to_module(module)
    w13 = torch.empty((2, 64, 32))
    w2 = torch.empty((2, 32, 32))

    backend, experts = module.select_unquantized_moe_backend(_config())
    converted = module.convert_to_unquantized_kernel_format(
        backend, _config(), w13, w2
    )

    assert backend.name == "AITER"
    assert experts == "official-experts"
    assert converted == (w13, w2)
    assert [call for call in calls if call[0] == "select"]
    assert [call for call in calls if call[0] == "convert"]


def test_w16a16_converter_packs_only_owned_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import lightop_w16a16_runtime
    from vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe import (
        LightopW16A16Experts,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle

    module, _ = _target_module()
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", True, raising=False
    )
    monkeypatch.setattr(
        LightopW16A16Experts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )
    monkeypatch.setattr(
        lightop_w16a16_runtime,
        "select_lightop_w16a16_config",
        lambda *args, **kwargs: ({"BLOCK_SIZE_M": 16}, {"BLOCK_SIZE_M": 16}),
    )
    patch_unquantized_oracle.apply_to_module(module)
    backend, _ = module.select_unquantized_moe_backend(_config())
    w13 = torch.empty((2, 64, 32), dtype=torch.bfloat16)
    w2 = torch.empty((2, 32, 32), dtype=torch.bfloat16)

    packed13, packed2 = module.convert_to_unquantized_kernel_format(
        backend, _config(), w13, w2
    )

    assert packed13.shape == (2, 2, 1024)
    assert packed2.shape == (2, 2, 512)
    assert getattr(packed13, "_hcu_lightop_w16a16_packed") is True
    assert getattr(packed2, "_hcu_lightop_w16a16_packed") is True


def test_hcu_unquantized_constructor_uses_patched_oracle_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm_hcu.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as method_module,
    )

    selected = SimpleNamespace(name="HCU_LIGHTOP_W16A16")
    calls: list[object] = []

    def select(*, moe_config):
        calls.append(moe_config)
        return selected, object

    monkeypatch.setattr(
        method_module, "select_unquantized_moe_backend", select, raising=False
    )
    config = _config()
    with set_current_vllm_config(
        VllmConfig(device_config=DeviceConfig(device="cuda"))
    ):
        method = method_module.HcuUnquantizedFusedMoEMethod(config)

    assert calls == [config]
    assert method.unquantized_backend is selected


def test_w16a16_weight_lifecycle_installs_one_packed_parameter_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as method_module,
    )

    method_cls = method_module.HcuUnquantizedFusedMoEMethod
    method = method_cls.__new__(method_cls)
    object.__setattr__(
        method,
        "unquantized_backend",
        SimpleNamespace(name="HCU_LIGHTOP_W16A16"),
    )
    object.__setattr__(method, "moe_kernel", None)
    object.__setattr__(method, "moe_quant_config", None)
    object.__setattr__(method, "moe", object())
    object.__setattr__(method, "experts_cls", object)
    object.__setattr__(
        method, "get_fused_moe_quant_config", lambda layer: "quant-config"
    )
    built: list[dict[str, object]] = []

    def make_kernel(**kwargs):
        built.append(kwargs)
        return object()

    monkeypatch.setattr(method_module, "make_unquantized_moe_kernel", make_kernel)

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter(
                "w13_weight",
                torch.nn.Parameter(torch.empty((2, 64, 32)), requires_grad=False),
            )
            self.register_parameter(
                "w2_weight",
                torch.nn.Parameter(torch.empty((2, 32, 32)), requires_grad=False),
            )
            self.w13_weight.weight_loader = "w13-loader"
            self.w2_weight.weight_loader = "w2-loader"

        def _expert_routing_tables(self):
            return None

    layer = Layer()
    method.process_weights_after_loading(layer)
    installed13 = layer.w13_weight
    installed2 = layer.w2_weight

    assert tuple(installed13.shape) == (2, 2, 1024)
    assert tuple(installed2.shape) == (2, 2, 512)
    assert callable(installed13.weight_loader)
    assert callable(installed2.weight_loader)
    with pytest.raises(RuntimeError, match="cannot reload"):
        installed13.weight_loader(param=installed13, loaded_weight=torch.empty(1))
    with pytest.raises(RuntimeError, match="cannot reload"):
        installed2.weight_loader(param=installed2, loaded_weight=torch.empty(1))
    assert getattr(installed13, "_hcu_lightop_w16a16_packed") is True
    assert getattr(installed2, "_hcu_lightop_w16a16_packed") is True
    assert len(built) == 1

    method.process_weights_after_loading(layer)
    assert layer.w13_weight is installed13
    assert layer.w2_weight is installed2
    assert len(built) == 1


def test_worker_registers_unquantized_oracle_before_unquantized_method() -> None:
    from vllm_hcu.patch import worker

    callbacks = worker.worker_callback_names()
    oracle = (
        "worker.op_opt.moe.oracle.unquantized_lightop_w16a16",
        "vllm.model_executor.layers.fused_moe.oracle.unquantized",
    )
    layer = (
        "worker.op_opt.moe.layer",
        "vllm.model_executor.layers.fused_moe",
    )
    assert oracle in callbacks
    assert layer in callbacks
    assert callbacks.index(oracle) < callbacks.index(layer)
