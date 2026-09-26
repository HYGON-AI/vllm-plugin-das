# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fail-closed configuration tests for the first Hy4 DCP topology."""

from types import SimpleNamespace

import pytest

from vllm_hcu.patch.config import HcuFeatureConfig


@pytest.fixture
def make_dcp_config():
    def build(**changes):
        values = dict(
            architecture="HYV4ForCausalLM",
            use_v2=True,
            use_mla=True,
            heads=64,
            tp=8,
            dcp=2,
            pcp=1,
            pp=1,
            dp=1,
            ep=True,
            qrep=False,
            comm="ag_rs",
            interleave=1,
            kv_dtype="fp8_e4m3",
            moe_backend="aiter",
            method=None,
            tokens=0,
            lora=False,
            multimodal=False,
            kv_offload=False,
            kv_connector=None,
            multi_layer_mtp=False,
            lightly_cp=False,
        )
        values.update(changes)
        return SimpleNamespace(
            use_v2_model_runner=values["use_v2"],
            model_config=SimpleNamespace(
                architectures=[values["architecture"]],
                use_mla=values["use_mla"],
                hf_config=SimpleNamespace(num_attention_heads=values["heads"]),
                is_multimodal_model=values["multimodal"],
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=values["tp"],
                decode_context_parallel_size=values["dcp"],
                prefill_context_parallel_size=values["pcp"],
                pipeline_parallel_size=values["pp"],
                data_parallel_size=values["dp"],
                enable_expert_parallel=values["ep"],
                dcp_q_replicate=values["qrep"],
                dcp_comm_backend=values["comm"],
                cp_kv_cache_interleave_size=values["interleave"],
            ),
            cache_config=SimpleNamespace(
                cache_dtype=values["kv_dtype"],
                kv_offloading_size=1.0 if values["kv_offload"] else None,
            ),
            kernel_config=SimpleNamespace(moe_backend=values["moe_backend"]),
            speculative_config=(
                None
                if values["method"] is None
                else SimpleNamespace(
                    method=values["method"],
                    num_speculative_tokens=values["tokens"],
                )
            ),
            lora_config=SimpleNamespace() if values["lora"] else None,
            kv_transfer_config=(
                None
                if values["kv_connector"] is None
                else SimpleNamespace(kv_connector=values["kv_connector"])
            ),
            additional_config={
                "hcu": HcuFeatureConfig(
                    enable_multi_layers_mtp=values["multi_layer_mtp"],
                    enable_lightly_cp=values["lightly_cp"],
                ).to_dict()
            },
        )

    return build


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tp": 4}, "topology"),
        ({"dcp": 4}, "topology"),
        ({"pcp": 2}, "topology"),
        ({"pp": 2}, "topology"),
        ({"dp": 2}, "topology"),
        ({"kv_dtype": "bfloat16"}, "fp8_e4m3"),
        ({"kv_dtype": "auto"}, "fp8_e4m3"),
        ({"kv_dtype": "fp8"}, "fp8_e4m3"),
        ({"use_v2": False}, "Model Runner V2"),
        ({"use_mla": False}, "Model Runner V2"),
        ({"heads": 32}, "64 attention heads"),
        ({"comm": "all_gather"}, "ag_rs"),
        ({"interleave": 2}, "interleave size 1"),
        ({"ep": False}, "expert parallelism"),
        ({"moe_backend": "auto"}, "AITER"),
        ({"qrep": True}, "query replication"),
        ({"method": "mtp", "tokens": 2}, "MTP3"),
        ({"method": "ngram", "tokens": 3}, "MTP3"),
        ({"lora": True}, "LoRA"),
        ({"multimodal": True}, "multimodal"),
        ({"kv_offload": True}, "offload"),
        ({"kv_connector": "MooncakeConnector"}, "P/D"),
        ({"multi_layer_mtp": True}, "multi-layer MTP"),
        ({"lightly_cp": True}, "lightly-CP"),
    ],
)
def test_hy4_dcp2_rejects_unvalidated_configuration(
    make_dcp_config, changes, message
):
    from vllm_hcu.models.hy_v4.dcp_config import validate_hy4_dcp_config

    with pytest.raises(ValueError, match=message):
        validate_hy4_dcp_config(make_dcp_config(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"kv_dtype": "fp8_ds_mla"},
        {"method": "mtp", "tokens": 3},
    ],
)
def test_hy4_dcp2_accepts_valid_fp8_mrv2_configuration(
    make_dcp_config, changes
):
    from vllm_hcu.models.hy_v4.dcp_config import validate_hy4_dcp_config

    assert validate_hy4_dcp_config(make_dcp_config(**changes)) is True


def test_hy4_dcp2_dtype_error_lists_both_supported_formats(make_dcp_config):
    from vllm_hcu.models.hy_v4.dcp_config import validate_hy4_dcp_config

    with pytest.raises(ValueError) as error:
        validate_hy4_dcp_config(make_dcp_config(kv_dtype="fp8_ds_mla_typo"))

    assert "--kv-cache-dtype" in str(error.value)
    assert "fp8_e4m3" in str(error.value)
    assert "fp8_ds_mla" in str(error.value)


def test_hy4_dcp2_rejects_query_replication_environment(
    monkeypatch, make_dcp_config
):
    from vllm_hcu.models.hy_v4.dcp_config import validate_hy4_dcp_config

    monkeypatch.setenv("VLLM_DCP_Q_REPLICATE", "1")
    with pytest.raises(ValueError, match="query replication"):
        validate_hy4_dcp_config(make_dcp_config())


def test_hy4_dcp1_and_other_models_are_outside_gate(make_dcp_config):
    from vllm_hcu.models.hy_v4.dcp_config import validate_hy4_dcp_config

    assert (
        validate_hy4_dcp_config(
            make_dcp_config(dcp=1, kv_dtype="bfloat16", ep=False)
        )
        is False
    )
    assert (
        validate_hy4_dcp_config(
            make_dcp_config(architecture="Qwen3ForCausalLM")
        )
        is False
    )


def test_platform_config_hook_rejects_hy4_dcp_before_later_binding(
    make_dcp_config,
):
    from vllm_hcu.patch.platform.core_fix import patch_vllm_config

    with pytest.raises(ValueError, match="fp8_e4m3"):
        patch_vllm_config.validate_and_update_hcu_config(
            make_dcp_config(kv_dtype="bfloat16")
        )
