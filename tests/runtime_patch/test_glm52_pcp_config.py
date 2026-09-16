# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fail-closed Model Runner V2 PCP configuration contracts."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.vllm import VllmConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm_hcu.patch.config import HcuFeatureConfig
from vllm_hcu.patch.platform.core_fix import patch_vllm_config
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError


def _make_pcp_config(**overrides: object) -> object:
    """Build a CPU-safe object matching vLLM 0.25.1 config field names."""

    architecture = overrides.pop("architecture", "GlmMoeDsaForCausalLM")
    use_v2 = overrides.pop("use_v2", True)
    use_mla = overrides.pop("use_mla", True)
    pcp = overrides.pop("pcp", 2)
    tp = overrides.pop("tp", 4)
    pp = overrides.pop("pp", 1)
    dcp = overrides.pop("dcp", 1)
    dp = overrides.pop("dp", 1)
    enable_expert_parallel = overrides.pop("enable_expert_parallel", True)
    enforce_eager = overrides.pop("enforce_eager", True)
    speculative = overrides.pop("speculative", False)
    speculative_method = overrides.pop("speculative_method", "mtp")
    num_speculative_tokens = overrides.pop("num_speculative_tokens", 1)
    lora = overrides.pop("lora", False)
    multimodal = overrides.pop("multimodal", False)
    hybrid = overrides.pop("hybrid", False)
    kv_offload = overrides.pop("kv_offload", False)
    kv_transfer = overrides.pop("kv_transfer", False)
    enable_lightly_cp = overrides.pop("enable_lightly_cp", False)
    enable_multi_layers_mtp = overrides.pop("enable_multi_layers_mtp", False)
    attention_backend = overrides.pop(
        "attention_backend", AttentionBackendEnum.FLASH_ATTN
    )
    if overrides:
        raise AssertionError(f"unknown PCP fixture override(s): {sorted(overrides)}")

    return SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_config=SimpleNamespace(
            architectures=[architecture],
            use_mla=use_mla,
            enforce_eager=enforce_eager,
            is_multimodal_model=multimodal,
            is_hybrid=hybrid,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp,
            data_parallel_size=dp,
            enable_expert_parallel=enable_expert_parallel,
        ),
        attention_config=SimpleNamespace(backend=attention_backend),
        speculative_config=(
            SimpleNamespace(
                method=speculative_method,
                num_speculative_tokens=num_speculative_tokens,
            )
            if speculative
            else None
        ),
        lora_config=(SimpleNamespace() if lora else None),
        cache_config=SimpleNamespace(kv_offloading_size=(1.0 if kv_offload else None)),
        kv_transfer_config=(
            SimpleNamespace(kv_connector="MooncakeConnector") if kv_transfer else None
        ),
        additional_config={
            "hcu": HcuFeatureConfig(
                enable_lightly_cp=enable_lightly_cp,
                enable_multi_layers_mtp=enable_multi_layers_mtp,
            ).to_dict()
        },
    )


@pytest.fixture
def make_pcp_config():
    return _make_pcp_config


@pytest.fixture
def make_hyv4_pp2_pcp4_config(monkeypatch, make_pcp_config):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "41,37")

    def make_config(**overrides):
        values = dict(architecture="HYV4ForCausalLM", pp=2, tp=1, pcp=4)
        values.update(overrides)
        return make_pcp_config(**values)

    return make_config


def test_hyv4_pp2_pcp4_exact_target_only_topology_is_allowed(
    make_hyv4_pp2_pcp4_config,
) -> None:
    assert patch_vllm_config._validate_hcu_pcp_scope(
        make_hyv4_pp2_pcp4_config()
    ) is True


def _native_hyv4_mtp_config(make_config, **overrides):
    config = make_config(speculative=True, num_speculative_tokens=3, **overrides)
    config.model_config.hf_config = SimpleNamespace(num_nextn_predict_layers=1)
    config.model_config.model = "native-hyv4"
    config.speculative_config.draft_model_config = SimpleNamespace(
        architectures=["HYV4MTPModel"], model="native-hyv4",
        hf_config=SimpleNamespace(num_nextn_predict_layers=1, n_predict=1),
    )
    config.parallel_config.all2all_backend = "deepep_high_throughput"
    config.parallel_config.enable_eplb = False
    config.kernel_config = SimpleNamespace(moe_backend="deep_gemm")
    config.cache_config.cache_dtype = "fp8_e4m3"
    return config


def test_hyv4_exact_pp2_pcp4_native_mtp3_is_allowed(make_hyv4_pp2_pcp4_config):
    config = _native_hyv4_mtp_config(make_hyv4_pp2_pcp4_config)
    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True
    assert config.model_config.architectures == ["HYV4ForCausalLM"]
    assert config.speculative_config.draft_model_config.architectures == ["HYV4MTPModel"]


def test_hyv4_mtp3_revalidation_accepts_current_sparse_cache_canonicalization(
    make_hyv4_pp2_pcp4_config,
):
    from vllm_hcu.models.hy_v4.attention import _normalize_hy_v4_kv_cache_dtype
    config = _native_hyv4_mtp_config(make_hyv4_pp2_pcp4_config)
    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True
    # Native draft attention consumes the same CacheConfig and canonicalizes
    # its public E4M3 alias before initialize_kv_cache builds the PCP manager.
    config.cache_config.cache_dtype = _normalize_hy_v4_kv_cache_dtype(
        config.cache_config.cache_dtype, use_sparse=True)
    assert config.cache_config.cache_dtype == "fp8_ds_mla"
    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True


@pytest.mark.parametrize("override", [
    {"pp": 1}, {"pp": 3}, {"tp": 2}, {"pcp": 2}, {"pcp": 8},
    {"dp": 2}, {"dcp": 2}, {"enable_expert_parallel": False},
    {"enforce_eager": False}, {"use_v2": False}, {"lora": True},
    {"multimodal": True}, {"kv_offload": True}, {"kv_transfer": True},
    {"enable_multi_layers_mtp": True}, {"speculative_method": "eagle"},
    {"architecture": "HYV4MTPModel"},
])
def test_hyv4_mtp3_neighbors_stay_closed(make_hyv4_pp2_pcp4_config, override):
    config = _native_hyv4_mtp_config(make_hyv4_pp2_pcp4_config, **override)
    with pytest.raises(ValueError):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("path,value", [
    ("speculative_config.num_speculative_tokens", 1),
    ("speculative_config.num_speculative_tokens", 2),
    ("speculative_config.num_speculative_tokens", 4),
    ("speculative_config.draft_model_config.architectures", ["UnknownDraft"]),
    ("speculative_config.draft_model_config.model", "separate-checkpoint"),
    ("speculative_config.draft_model_config.hf_config.n_predict", 2),
    ("model_config.hf_config.num_nextn_predict_layers", 2),
    ("parallel_config.enable_eplb", True),
    ("parallel_config.all2all_backend", "deepep_low_latency"),
    ("kernel_config.moe_backend", "aiter"),
    ("cache_config.cache_dtype", "auto"),
])
def test_hyv4_mtp3_exact_native_contract_is_required(
    make_hyv4_pp2_pcp4_config, path, value,
):
    config = _native_hyv4_mtp_config(make_hyv4_pp2_pcp4_config)
    owner = config
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)
    with pytest.raises(ValueError):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("hcu", [
    {"expert_map_path": "/static.json", "eplb_disable_rearrange": True},
    {"expert_map_path": "/load.json"},
    {"expert_map_record_path": "/record.json"},
    {"eplb_disable_rearrange": True},
])
def test_hyv4_mtp3_eplb_sidecars_stay_closed(make_hyv4_pp2_pcp4_config, hcu):
    config = _native_hyv4_mtp_config(make_hyv4_pp2_pcp4_config)
    config.additional_config["hcu"].update(hcu)
    with pytest.raises(ValueError):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("partition", [None, "", "39,39", "41, 37", "37,41"])
def test_hyv4_mtp3_requires_exact_partition(
    make_hyv4_pp2_pcp4_config, monkeypatch, partition,
):
    config = _native_hyv4_mtp_config(make_hyv4_pp2_pcp4_config)
    if partition is None:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION")
    else:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    with pytest.raises(ValueError):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("override", [
    {"pp": 3}, {"tp": 2}, {"pcp": 2}, {"pcp": 3}, {"pcp": 8},
    {"dp": 2}, {"dcp": 2}, {"enable_expert_parallel": False},
    {"enforce_eager": False},
])
def test_hyv4_pp2_pcp4_rejects_nearby_topologies(
    make_hyv4_pp2_pcp4_config, override,
) -> None:
    with pytest.raises(ValueError, match="Hy4 PP2.*PCP4"):
        patch_vllm_config._validate_hcu_pcp_scope(
            make_hyv4_pp2_pcp4_config(**override)
        )


@pytest.mark.parametrize("partition", [None, "", "39,39", "40,38", "41, 37", "37,41"])
def test_hyv4_pp2_pcp4_requires_exact_layer_partition(
    make_hyv4_pp2_pcp4_config, monkeypatch, partition,
) -> None:
    config = make_hyv4_pp2_pcp4_config()
    if partition is None:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION")
    else:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    with pytest.raises(ValueError, match="VLLM_PP_LAYER_PARTITION=41,37"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("architecture,use_mla", [
    ("HYV4MTPModel", True), ("GlmMoeDsaForCausalLM", True),
    ("DeepseekV2ForCausalLM", True), ("Qwen3ForCausalLM", False),
    ("HYV4ForCausalLM", False),
])
def test_hyv4_pp2_pcp4_does_not_authorize_other_architectures(
    make_hyv4_pp2_pcp4_config, architecture, use_mla,
) -> None:
    with pytest.raises(ValueError):
        patch_vllm_config._validate_hcu_pcp_scope(
            make_hyv4_pp2_pcp4_config(architecture=architecture, use_mla=use_mla)
        )


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("override,message", [
    ({"speculative": True}, "speculative"),
    ({"speculative": True, "speculative_method": "eagle"}, "speculative"),
    ({"kv_transfer": True}, "P/D disaggregation"),
    ({"use_v2": False}, "Model Runner V2"),
    ({"lora": True}, "LoRA"),
    ({"multimodal": True}, "multimodal"),
    ({"kv_offload": True}, "KV offload"),
    ({"enable_lightly_cp": True}, "lightly-CP"),
    ({"enable_multi_layers_mtp": True}, "multi-layer MTP"),
])
def test_hyv4_pcp_keeps_unvalidated_features_rejected(
    make_hyv4_pp2_pcp4_config, pp, override, message,
) -> None:
    with pytest.raises(ValueError, match=message):
        patch_vllm_config._validate_hcu_pcp_scope(
            make_hyv4_pp2_pcp4_config(pp=pp, **override)
        )


@pytest.mark.parametrize("pcp,tp", [(2, 4), (4, 1), (8, 1)])
def test_hyv4_pp1_target_only_uses_existing_mla_pcp_ep_scope(
    make_pcp_config, monkeypatch, pcp, tp,
) -> None:
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    assert patch_vllm_config._validate_hcu_pcp_scope(
        make_pcp_config(architecture="HYV4ForCausalLM", pcp=pcp, tp=tp)
    ) is True


@pytest.mark.parametrize("pcp,tp", [(2, 4), (4, 1), (8, 1)])
def test_hyv4_pd_allows_only_mooncake_producer_pcp(make_pcp_config, pcp, tp):
    config = make_pcp_config(architecture="HYV4ForCausalLM", pcp=pcp, tp=tp)
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeConnector", kv_role="kv_producer",
        kv_buffer_device="cpu",
    )
    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True


@pytest.mark.parametrize("role", ["kv_consumer", "kv_both", None, "prefill"])
def test_hyv4_pd_rejects_nonproducer_pcp(make_pcp_config, role):
    config = make_pcp_config(architecture="HYV4ForCausalLM")
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="MooncakeConnector", kv_role=role,
        kv_connector_module_path=None,
    )
    with pytest.raises(ValueError, match="P/D disaggregation"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("connector,module_path", [
    ("OtherConnector", None), (None, None),
    ("MooncakeConnector", "custom.connector"), ("MooncakeConnector", ""),
])
def test_hyv4_pd_rejects_unknown_or_custom_producer_connector(
    make_pcp_config, connector, module_path,
):
    config = make_pcp_config(architecture="HYV4ForCausalLM")
    config.kv_transfer_config = KVTransferConfig(
        kv_connector=connector, kv_role="kv_producer",
        kv_connector_module_path=module_path, kv_buffer_device="cpu",
    )
    with pytest.raises(ValueError, match="P/D disaggregation"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("pp", [2, 3])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
def test_hyv4_pp_pcp_pd_is_rejected(make_hyv4_pp2_pcp4_config, pp, role):
    config = make_hyv4_pp2_pcp4_config(pp=pp)
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeConnector", kv_role=role, kv_buffer_device="cpu",
    )
    with pytest.raises(ValueError, match="PP.*PCP.*P/D"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


def test_hyv4_pd_consumer_without_pcp_preserves_existing_validation(make_pcp_config):
    config = make_pcp_config(architecture="HYV4ForCausalLM", pcp=1)
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeConnector", kv_role="kv_consumer", kv_buffer_device="cpu",
    )
    assert patch_vllm_config._validate_hcu_pcp_scope(config) is False


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("use_mla", [False, True])
def test_hyv4_mtp_architecture_is_never_a_pcp_target(
    make_hyv4_pp2_pcp4_config, pp, use_mla,
) -> None:
    with pytest.raises(ValueError, match="HYV4MTPModel"):
        patch_vllm_config._validate_hcu_pcp_scope(
            make_hyv4_pp2_pcp4_config(
                architecture="HYV4MTPModel", pp=pp, use_mla=use_mla,
            )
        )


@pytest.mark.parametrize("pp", [1, 2])
def test_hyv4_pcp_rejects_incomplete_pd_configuration(
    make_hyv4_pp2_pcp4_config, pp,
) -> None:
    config = make_hyv4_pp2_pcp4_config(pp=pp)
    config.kv_transfer_config = SimpleNamespace(kv_connector=None)
    with pytest.raises(ValueError, match="P/D disaggregation"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("use_mla", [False, True])
def test_hyv4_pcp_rejects_ambiguous_architecture_lists(
    make_hyv4_pp2_pcp4_config, use_mla,
) -> None:
    config = make_hyv4_pp2_pcp4_config(pp=1, use_mla=use_mla)
    config.model_config.architectures = ["HYV4ForCausalLM", "Qwen3ForCausalLM"]
    with pytest.raises(ValueError):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize("override,message", [
    ({"dcp": 2}, "decode context"), ({"dp": 2}, "data parallel"),
    ({"enable_expert_parallel": False}, "expert parallel"),
    ({"enforce_eager": False}, "eager"), ({"use_mla": False}, "MLA"),
])
def test_hyv4_pp1_preserves_existing_mla_pcp_restrictions(
    make_hyv4_pp2_pcp4_config, override, message,
) -> None:
    with pytest.raises(ValueError, match=message):
        patch_vllm_config._validate_hcu_pcp_scope(
            make_hyv4_pp2_pcp4_config(pp=1, **override)
        )


def test_hyv4_pp2_pcp4_removes_only_audited_upstream_rejection(
    make_hyv4_pp2_pcp4_config,
) -> None:
    module = _make_vllm_module()
    patch_vllm_config.apply_to_module(module)
    config = _as_fake_vllm_config(module, make_hyv4_pp2_pcp4_config())
    assert config._get_v2_model_runner_unsupported_features() == []
    config._validate_v2_model_runner()

    config.parallel_config.tensor_parallel_size = 2
    assert config._get_v2_model_runner_unsupported_features() == [
        "prefill context parallelism"
    ]
    with pytest.raises(ValueError, match="prefill context parallelism"):
        config._validate_v2_model_runner()


def test_glm52_mrv2_mla_pcp2_eager_is_allowed(make_pcp_config) -> None:
    """Removing the accepted GLM PCP branch must reject this configuration."""

    config = make_pcp_config(
        architecture="GlmMoeDsaForCausalLM",
        use_mla=True,
        pcp=2,
        tp=4,
        enable_expert_parallel=True,
        enforce_eager=True,
    )
    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True


def test_gqa_mrv2_flash_pcp_is_allowed(make_pcp_config) -> None:
    """GQA PCP must not remain trapped behind the former MLA-only gate."""

    config = make_pcp_config(
        architecture="Qwen3ForCausalLM",
        use_mla=False,
        pcp=2,
        tp=2,
        dcp=1,
        enable_expert_parallel=False,
    )

    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True


def test_gqa_pcp_rejects_triton_attention_backend(make_pcp_config) -> None:
    """Triton lacks the PCP KV gather and metadata path used during prefill."""

    config = make_pcp_config(
        architecture="Qwen3ForCausalLM",
        use_mla=False,
        pcp=2,
        tp=2,
        dcp=1,
        enable_expert_parallel=False,
        attention_backend=AttentionBackendEnum.TRITON_ATTN,
    )

    with pytest.raises(ValueError, match="only supports FLASH_ATTN"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


def test_gqa_pcp_rejects_automatic_attention_backend(make_pcp_config) -> None:
    """Fail closed instead of allowing auto-selection to fall back to Triton."""

    config = make_pcp_config(
        architecture="Qwen3ForCausalLM",
        use_mla=False,
        pcp=2,
        tp=2,
        dcp=1,
        enable_expert_parallel=False,
        attention_backend=None,
    )

    with pytest.raises(ValueError, match="only supports FLASH_ATTN"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


def test_gqa_pcp_rejects_decode_context_parallelism(make_pcp_config) -> None:
    """DCP reuses TP ranks, so it cannot be treated as the PCP rank group."""

    config = make_pcp_config(
        architecture="Qwen3ForCausalLM",
        use_mla=False,
        pcp=2,
        tp=2,
        dcp=2,
        enable_expert_parallel=False,
    )

    with pytest.raises(ValueError, match="does not support decode context"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


def test_gqa_pcp_allows_piecewise_graph_execution(make_pcp_config) -> None:
    """FlashAttention opts out of capture itself, so global eager is unnecessary."""

    config = make_pcp_config(
        architecture="Qwen3ForCausalLM",
        use_mla=False,
        pcp=2,
        dcp=1,
        enable_expert_parallel=False,
        enforce_eager=False,
    )

    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True


@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
def test_glm52_pcp_allows_validated_builtin_mtp_depths(
    make_pcp_config, num_speculative_tokens: int
) -> None:
    """Rejecting either validated draft depth breaks PCP+MTP service startup."""

    config = make_pcp_config(
        pcp=2,
        speculative=True,
        speculative_method="mtp",
        num_speculative_tokens=num_speculative_tokens,
    )

    assert patch_vllm_config._validate_hcu_pcp_scope(config) is True


def test_gqa_pcp_rejects_speculative_decoding(make_pcp_config) -> None:
    """FlashAttention PCP has no replicated speculative-decode contract."""

    config = make_pcp_config(
        architecture="Qwen3ForCausalLM",
        use_mla=False,
        pcp=2,
        tp=2,
        enable_expert_parallel=False,
        speculative=True,
    )

    with pytest.raises(ValueError, match="speculative decoding"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


def test_gqa_pcp_rejects_hybrid_kv_cache_groups(make_pcp_config) -> None:
    """One PCP plan cannot address distinct block tables for hybrid KV groups."""

    config = make_pcp_config(
        architecture="Qwen3ForCausalLM",
        use_mla=False,
        pcp=2,
        tp=2,
        enable_expert_parallel=False,
        hybrid=True,
    )

    with pytest.raises(ValueError, match="hybrid"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_v2": False}, "Model Runner V2"),
        ({"architecture": "DeepseekV2ForCausalLM"}, "GLM-5.2"),
        ({"use_mla": False}, "MLA"),
        ({"pp": 2}, "pipeline parallel"),
        ({"dcp": 2}, "decode context parallel"),
        ({"dp": 2}, "data parallel"),
        ({"enable_expert_parallel": False}, "expert parallel"),
        (
            {"speculative": True, "speculative_method": "eagle"},
            "only supports built-in MTP",
        ),
        (
            {"speculative": True, "num_speculative_tokens": 3},
            "one or two speculative tokens",
        ),
        ({"enforce_eager": False}, "eager"),
        ({"lora": True}, "LoRA"),
        ({"multimodal": True}, "multimodal"),
        ({"kv_offload": True}, "KV offload"),
        ({"kv_transfer": True}, "P/D disaggregation"),
        ({"enable_lightly_cp": True}, "lightly-CP"),
        ({"enable_multi_layers_mtp": True}, "multi-layer MTP"),
    ],
)
def test_glm52_pcp_scope_rejects_unsupported_combinations(
    make_pcp_config, override, message
) -> None:
    """Each unsupported PCP dimension must fail closed with its own reason."""

    with pytest.raises(ValueError, match=message):
        patch_vllm_config._validate_hcu_pcp_scope(
            make_pcp_config(pcp=2, **override)
        )


def _make_vllm_module() -> ModuleType:
    module = ModuleType(patch_vllm_config.TARGET_MODULE)

    class ModelConfig:
        def get_model_arch_config(self) -> object:
            return None

        def verify_with_parallel_config(self, parallel_config) -> None:
            if (
                parallel_config.decode_context_parallel_size > 1
                and not self.use_mla
            ):
                raise AssertionError("legacy GQA DCP head constraint")

    class VllmConfig:
        def __post_init__(self) -> None:
            return None

        def with_hf_config(self, hf_config: object, architectures=None):
            del hf_config, architectures
            return self

        def _set_cudagraph_sizes(self) -> None:
            return None

        def _get_v2_model_runner_unsupported_features(self) -> list[str]:
            if self.parallel_config.prefill_context_parallel_size > 1:
                return ["prefill context parallelism"]
            return ["upstream feature"]

        def _validate_v2_model_runner(self) -> None:
            unsupported = self._get_v2_model_runner_unsupported_features()
            if unsupported:
                raise ValueError(
                    f"Model Runner V2 does not yet support: {', '.join(unsupported)}"
                )

    module.ModelConfig = ModelConfig
    module.VllmConfig = VllmConfig
    return module


def _as_fake_vllm_config(module: ModuleType, config: object) -> object:
    patched_config = object.__new__(module.VllmConfig)
    patched_config.__dict__.update(vars(config))
    return patched_config


def test_valid_glm52_pcp_removes_only_the_upstream_pcp_rejection(
    make_pcp_config,
) -> None:
    """Restoring the original unsupported list must reject valid GLM PCP."""

    module = _make_vllm_module()
    assert patch_vllm_config.apply_to_module(module) is True
    config = _as_fake_vllm_config(module, make_pcp_config(pcp=2))

    assert config._get_v2_model_runner_unsupported_features() == []
    config._validate_v2_model_runner()


def test_gqa_pcp_dcp_preserves_upstream_head_partition_constraint() -> None:
    """PCP and DCP use different rank groups, so upstream DCP checks must run."""

    module = _make_vllm_module()
    assert patch_vllm_config.apply_to_module(module) is True
    model_config = module.ModelConfig()
    model_config.use_mla = False
    parallel_config = SimpleNamespace(
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
    )

    with pytest.raises(AssertionError, match="legacy GQA DCP head constraint"):
        model_config.verify_with_parallel_config(parallel_config)
    assert parallel_config.decode_context_parallel_size == 2


def test_pcp1_preserves_upstream_v2_unsupported_feature_and_validation(
    make_pcp_config,
) -> None:
    """Accidentally bypassing non-PCP V2 validation must fail this test."""

    module = _make_vllm_module()
    assert patch_vllm_config.apply_to_module(module) is True
    config = _as_fake_vllm_config(module, make_pcp_config(pcp=1))

    assert config._get_v2_model_runner_unsupported_features() == ["upstream feature"]
    with pytest.raises(ValueError, match="upstream feature"):
        config._validate_v2_model_runner()


def test_invalid_pcp_preserves_upstream_pcp_rejection(make_pcp_config) -> None:
    """Relaxing PCP before its HCU scope passes must remain impossible."""

    module = _make_vllm_module()
    assert patch_vllm_config.apply_to_module(module) is True
    config = _as_fake_vllm_config(
        module,
        make_pcp_config(
            architecture="Qwen3ForCausalLM",
            use_mla=False,
            pcp=2,
            dcp=4,
            enable_expert_parallel=False,
        ),
    )

    assert config._get_v2_model_runner_unsupported_features() == [
        "prefill context parallelism"
    ]
    with pytest.raises(ValueError, match="prefill context parallelism"):
        config._validate_v2_model_runner()


def test_pcp_patch_rejects_v0251_wrapper_signature_drift() -> None:
    """Changing either v0.25.1 wrapper boundary must fail patch installation."""

    module = _make_vllm_module()

    def incompatible_unsupported(self, feature: object) -> list[str]:
        del self, feature
        return []

    module.VllmConfig._get_v2_model_runner_unsupported_features = (
        incompatible_unsupported
    )
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_vllm_config.apply_to_module(module)


def test_model_arch_config_signature_drift_names_exact_target() -> None:
    module = _make_vllm_module()

    def incompatible_model_arch_config(self, feature: object) -> object:
        del self, feature
        return None

    module.ModelConfig.get_model_arch_config = incompatible_model_arch_config

    with pytest.raises(PatchCompatibilityError) as error:
        patch_vllm_config.apply_to_module(module)

    assert "vllm.config.model.ModelConfig.get_model_arch_config" in str(error.value)
    assert "VllmConfig._get_v2_model_runner_unsupported_features" not in str(error.value)


class _LifecycleCompilationConfig:
    def __init__(self) -> None:
        self.cudagraph_mode = SimpleNamespace(has_full_cudagraphs=lambda: False)


def test_forced_v1_pcp_is_rejected_during_platform_config_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    make_pcp_config,
) -> None:
    """Removing the platform gate would let forced V1 PCP finish validation."""

    import torch

    assert VllmConfig.__module__ == "vllm.config.vllm"

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx936"),
    )
    from vllm_hcu.patch.platform.framework_opt import (
        patch_multiproc_executor,
        patch_scheduler,
    )
    from vllm_hcu.platforms.hcu import HCUPlatform

    monkeypatch.setattr(
        patch_scheduler, "select_hcu_scheduler", lambda config: False
    )
    monkeypatch.setattr(
        patch_multiproc_executor, "select_hcu_multiproc_executor", lambda config: False
    )
    config = make_pcp_config(use_v2=False, pcp=2)
    config.compilation_config = _LifecycleCompilationConfig()
    config.kernel_config = SimpleNamespace(moe_backend="auto")
    config.cache_config.user_specified_block_size = True
    config.parallel_config.worker_cls = "auto"

    with pytest.raises(ValueError, match="Model Runner V2"):
        HCUPlatform.check_and_update_config(config)
