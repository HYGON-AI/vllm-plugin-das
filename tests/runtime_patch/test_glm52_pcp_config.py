# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fail-closed Model Runner V2 PCP configuration contracts."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

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

    assert patch_vllm_config.TARGETS[4] in str(error.value)
    assert patch_vllm_config.TARGETS[2] not in str(error.value)


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
