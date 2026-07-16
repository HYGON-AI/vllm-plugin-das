# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from vllm_hcu.patch.config import (
    HcuFeatureConfig,
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
        moe_backend="dpsk_deep_gemm",
    )

    assert normalized == HcuFeatureConfig(
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
        enable_custom_sp=True,
        enable_multi_layers_mtp=True,
        moe_backend="dpsk_deep_gemm",
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
                "moe_backend": "auto",
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


def test_pop_legacy_keywords_leaves_upstream_kwargs_untouched() -> None:
    kwargs = {
        "model": "example",
        "enable_lightly_cp": True,
        "enable_lightly_cplb": True,
        "moe_backend": "dpsk_deep_gemm",
    }
    config = pop_hcu_feature_kwargs(kwargs)
    assert kwargs == {"model": "example"}
    assert config.enable_lightly_cp and config.enable_lightly_cplb
    assert config.moe_backend == "dpsk_deep_gemm"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"enable_lightly_cp": 1}, TypeError),
        ({"moe_backend": "triton"}, ValueError),
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
