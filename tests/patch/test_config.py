# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import pickle
import warnings
from types import SimpleNamespace

import pytest

import vllm_hcu.patch.config as hcu_config_module
from vllm_hcu.patch.config import (
    HcuFeatureConfig,
    bind_hcu_eplb_config,
    get_hcu_config,
    pop_hcu_feature_kwargs,
    set_hcu_config,
    write_hcu_config,
)


def test_defaults_and_round_trip_through_object_sidecar() -> None:
    vllm_config = SimpleNamespace(additional_config={"unrelated": 7})
    normalized = set_hcu_config(
        vllm_config,
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
        enable_custom_sp=True,
        enable_multi_layers_mtp=True,
        moe_backend="deep_gemm",
    )

    assert normalized == HcuFeatureConfig(
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
        enable_custom_sp=True,
        enable_multi_layers_mtp=True,
        moe_backend="deep_gemm",
    )
    assert vllm_config.additional_config["unrelated"] == 7
    assert get_hcu_config(vllm_config) == normalized


def test_dict_vllm_config_uses_canonical_storage_path() -> None:
    vllm_config: dict[str, object] = {}
    normalized = set_hcu_config(vllm_config, {"enable_lightly_cp": True})

    assert vllm_config == {
        "additional_config": {
            "hcu": {
                "enable_lightly_cp": True,
                "enable_lightly_cplb": False,
                "enable_custom_sp": False,
                "enable_multi_layers_mtp": False,
                "deepep_auto": False,
                "fused_qwen3_rms_rope": True,
                "moe_backend": "auto",
                "hcu_flash_attn_mode": None,
                "expert_map_path": None,
                "expert_map_record_path": None,
                "eplb_disable_rearrange": False,
                "eplb_static_dispatch_policy": "nearest",
            }
        }
    }
    assert get_hcu_config(vllm_config) == normalized


def test_direct_dict_and_object_payloads_are_accepted() -> None:
    assert get_hcu_config({"enable_custom_sp": True}).enable_custom_sp is True
    payload = SimpleNamespace(enable_multi_layers_mtp=True)
    assert get_hcu_config(payload).enable_multi_layers_mtp is True


def test_sidecar_is_pickle_safe_for_spawned_process_config() -> None:
    original = {"additional_config": {"hcu": HcuFeatureConfig(enable_custom_sp=True).to_dict()}}
    restored = pickle.loads(pickle.dumps(original))
    assert get_hcu_config(restored) == HcuFeatureConfig(enable_custom_sp=True)


def test_offline_eplb_sidecar_round_trip_and_parallel_binding() -> None:
    feature_config = HcuFeatureConfig(
        expert_map_path="/maps/hy4.json",
        eplb_disable_rearrange=True,
        eplb_static_dispatch_policy="locality_fair",
    )
    vllm_config = SimpleNamespace(
        additional_config={"hcu": feature_config.to_dict()},
        parallel_config=SimpleNamespace(),
    )

    restored = pickle.loads(pickle.dumps(vllm_config))
    bind_hcu_eplb_config(restored)

    assert get_hcu_config(restored) == feature_config
    assert restored.parallel_config._vllm_hcu_expert_map_path == "/maps/hy4.json"
    assert restored.parallel_config._vllm_hcu_expert_map_record_path is None
    assert restored.parallel_config._vllm_hcu_eplb_disable_rearrange is True
    assert (
        restored.parallel_config._vllm_hcu_eplb_static_dispatch_policy
        == "locality_fair"
    )


def test_offline_eplb_binding_supports_mapping_parallel_config() -> None:
    vllm_config = {
        "additional_config": {
            "hcu": HcuFeatureConfig(
                expert_map_record_path="/maps/record.json"
            ).to_dict()
        },
        "parallel_config": {},
    }

    bind_hcu_eplb_config(vllm_config)

    assert vllm_config["parallel_config"] == {
        "_vllm_hcu_expert_map_path": None,
        "_vllm_hcu_expert_map_record_path": "/maps/record.json",
        "_vllm_hcu_eplb_disable_rearrange": False,
        "_vllm_hcu_eplb_static_dispatch_policy": "nearest",
    }


def test_pop_legacy_keywords_leaves_upstream_kwargs_untouched() -> None:
    kwargs = {
        "model": "example",
        "enable_lightly_cp": True,
        "enable_lightly_cplb": True,
        "moe_backend": "deep_gemm",
    }
    config = pop_hcu_feature_kwargs(kwargs)
    assert kwargs == {"model": "example"}
    assert config.enable_lightly_cp and config.enable_lightly_cplb
    assert config.moe_backend == "deep_gemm"


def test_legacy_deep_gemm_sidecar_is_normalized_with_one_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hcu_config_module,
        "_legacy_backend_warning_emitted",
        False,
        raising=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = get_hcu_config({"moe_backend": "dpsk_deep_gemm"})
        second = get_hcu_config(
            {"additional_config": {"hcu": {"moe_backend": "dpsk_deep_gemm"}}}
        )
        direct = HcuFeatureConfig(moe_backend="dpsk_deep_gemm")

    assert first.moe_backend == "deep_gemm"
    assert second.moe_backend == "deep_gemm"
    assert direct.moe_backend == "deep_gemm"
    assert [warning.category for warning in caught] == [FutureWarning]
    assert "dpsk_deep_gemm" in str(caught[0].message)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"enable_lightly_cp": 1}, TypeError),
        ({"moe_backend": "triton"}, ValueError),
        ({"hcu_flash_attn_mode": "future"}, ValueError),
        ({"expert_map_path": ""}, ValueError),
        ({"expert_map_path": "   "}, ValueError),
        ({"expert_map_path": 7}, TypeError),
        ({"expert_map_record_path": ""}, ValueError),
        ({"expert_map_record_path": 7}, TypeError),
        ({"eplb_disable_rearrange": 1}, TypeError),
        ({"eplb_static_dispatch_policy": "round_robin"}, ValueError),
        ({"eplb_static_dispatch_policy": 1}, TypeError),
        (
            {
                "expert_map_path": "/maps/load.json",
                "expert_map_record_path": "/maps/record.json",
            },
            ValueError,
        ),
        ({"future_typo": True}, ValueError),
        ({"enable_lightly_cplb": True}, ValueError),
    ],
)
def test_invalid_values_are_rejected(payload: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        HcuFeatureConfig.from_mapping(payload)


def test_write_requires_mutable_mapping() -> None:
    with pytest.raises(TypeError, match="mutable mapping"):
        write_hcu_config(({},), HcuFeatureConfig())  # type: ignore[arg-type]
