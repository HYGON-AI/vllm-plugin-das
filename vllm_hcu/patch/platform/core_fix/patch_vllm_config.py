# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""VllmConfig adapters for HCU sidecar validation and graph buckets."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import Any

from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config, set_hcu_config

from ._common import PatchCompatibilityError, apply_once, load_exact_module
from .patch_compilation_config import bind_hcu_config

TARGET_MODULE = "vllm.config.vllm"
PATCH_ID = "platform.core_fix.hcu_config.vllm"
TARGETS = (
    f"{TARGET_MODULE}.VllmConfig.with_hf_config",
    f"{TARGET_MODULE}.VllmConfig._set_cudagraph_sizes",
    f"{TARGET_MODULE}.VllmConfig._get_v2_model_runner_unsupported_features",
    f"{TARGET_MODULE}.VllmConfig._validate_v2_model_runner",
    "vllm.config.model.ModelConfig.get_model_arch_config",
    "vllm_hcu.platforms.hcu.HCUPlatform.check_and_update_config",
)
_MARKER = "_vllm_hcu_feature_config_patch_applied"
_REQUEST_CAPTURE_SIZES = (
    *range(1, 9),
    *range(10, 33, 2),
    *range(40, 65, 4),
    *range(72, 257, 8),
)


def _require_hcu_pcp_attribute(owner: object, name: str, owner_name: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError as exc:
        raise PatchCompatibilityError(
            f"GLM-5.2 PCP requires {owner_name}.{name} in vLLM 0.25.1"
        ) from exc


def _require_mrv2_mla_eager_pcp_contract(vllm_config: object) -> None:
    """Reject every Model Runner V2 PCP configuration outside HCU support."""

    if not _require_hcu_pcp_attribute(
        vllm_config, "use_v2_model_runner", "VllmConfig"
    ):
        raise ValueError("GLM-5.2 PCP requires Model Runner V2.")

    model_config = _require_hcu_pcp_attribute(
        vllm_config, "model_config", "VllmConfig"
    )
    parallel_config = _require_hcu_pcp_attribute(
        vllm_config, "parallel_config", "VllmConfig"
    )
    cache_config = _require_hcu_pcp_attribute(
        vllm_config, "cache_config", "VllmConfig"
    )

    architectures = _require_hcu_pcp_attribute(
        model_config, "architectures", "ModelConfig"
    )
    if architectures != ["GlmMoeDsaForCausalLM"]:
        raise ValueError(
            "GLM-5.2 PCP only supports architecture "
            "GlmMoeDsaForCausalLM."
        )
    if not _require_hcu_pcp_attribute(model_config, "use_mla", "ModelConfig"):
        raise ValueError("GLM-5.2 PCP requires MLA or sparse MLA.")
    if _require_hcu_pcp_attribute(
        parallel_config, "pipeline_parallel_size", "ParallelConfig"
    ) != 1:
        raise ValueError("GLM-5.2 PCP does not support pipeline parallelism.")
    if _require_hcu_pcp_attribute(
        parallel_config, "decode_context_parallel_size", "ParallelConfig"
    ) != 1:
        raise ValueError(
            "GLM-5.2 PCP does not support decode context parallelism."
        )
    if _require_hcu_pcp_attribute(
        parallel_config, "data_parallel_size", "ParallelConfig"
    ) != 1:
        raise ValueError("GLM-5.2 PCP does not support data parallelism.")
    if not _require_hcu_pcp_attribute(
        parallel_config, "enable_expert_parallel", "ParallelConfig"
    ):
        raise ValueError("GLM-5.2 PCP requires expert parallelism.")
    if not _require_hcu_pcp_attribute(model_config, "enforce_eager", "ModelConfig"):
        raise ValueError("GLM-5.2 PCP requires eager execution without graphs.")
    if _require_hcu_pcp_attribute(
        vllm_config, "speculative_config", "VllmConfig"
    ) is not None:
        raise ValueError("GLM-5.2 PCP does not support speculative decoding or MTP.")
    if _require_hcu_pcp_attribute(vllm_config, "lora_config", "VllmConfig") is not None:
        raise ValueError("GLM-5.2 PCP does not support LoRA.")
    if _require_hcu_pcp_attribute(
        model_config, "is_multimodal_model", "ModelConfig"
    ):
        raise ValueError("GLM-5.2 PCP does not support multimodal models.")
    if _require_hcu_pcp_attribute(
        cache_config, "kv_offloading_size", "CacheConfig"
    ) is not None:
        raise ValueError("GLM-5.2 PCP does not support KV offload.")

    kv_transfer_config = _require_hcu_pcp_attribute(
        vllm_config, "kv_transfer_config", "VllmConfig"
    )
    if kv_transfer_config is not None and _require_hcu_pcp_attribute(
        kv_transfer_config, "kv_connector", "KVTransferConfig"
    ) is not None:
        raise ValueError("GLM-5.2 PCP does not support P/D disaggregation.")

    feature_config = get_hcu_config(vllm_config)
    if feature_config.enable_lightly_cp:
        raise ValueError("GLM-5.2 PCP does not support lightly-CP.")
    if feature_config.enable_multi_layers_mtp:
        raise ValueError("GLM-5.2 PCP does not support speculative decoding or MTP.")


def _validate_hcu_pcp_scope(vllm_config: object) -> bool:
    """Return whether this configuration is in the HCU PCP support scope."""

    parallel_config = _require_hcu_pcp_attribute(
        vllm_config, "parallel_config", "VllmConfig"
    )
    pcp_size = _require_hcu_pcp_attribute(
        parallel_config, "prefill_context_parallel_size", "ParallelConfig"
    )
    if pcp_size <= 1:
        return False

    _require_mrv2_mla_eager_pcp_contract(vllm_config)
    return True


def validate_and_update_hcu_config(vllm_config: object) -> HcuFeatureConfig:
    """Validate cross-config invariants and bind the compilation adapter."""

    _validate_hcu_pcp_scope(vllm_config)
    feature_config = get_hcu_config(vllm_config)
    updates: dict[str, str] = {}
    if feature_config.hcu_flash_attn_mode is None:
        # Persist the resolved sub-mode before vLLM computes compilation cache
        # hashes. Classic, CUTLASS, and CUSTOM do not share a KV-cache ABI.
        from vllm_hcu.platforms import envs as hcu_envs

        updates["hcu_flash_attn_mode"] = hcu_envs.resolve_hcu_flash_attn_mode(None)
    if updates:
        feature_config = feature_config.with_updates(**updates)
    # Persist the resolved mode so it enters vLLM's compilation hash.
    set_hcu_config(vllm_config, feature_config)

    feature_config = bind_hcu_config(vllm_config)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    kernel_config = getattr(vllm_config, "kernel_config", None)

    if parallel_config is not None:
        setattr(
            parallel_config,
            "_vllm_hcu_deepep_auto",
            feature_config.deepep_auto,
        )
    if feature_config.deepep_auto:
        if parallel_config is None:
            raise PatchCompatibilityError(
                "deepep_auto requires VllmConfig.parallel_config"
            )
        if getattr(parallel_config, "all2all_backend", None) != "deepep_low_latency":
            raise ValueError(
                "HCU deepep_auto must be normalized to the vLLM 0.25 "
                "deepep_low_latency configuration contract"
            )
        if feature_config.moe_backend not in ("auto", "dpsk_deep_gemm"):
            raise ValueError(
                "deepep_auto requires HCU moe_backend='auto' or "
                "'dpsk_deep_gemm'"
            )

    if feature_config.enable_lightly_cp:
        if model_config is None:
            raise PatchCompatibilityError(
                "Lightly-CP requires VllmConfig.model_config"
            )
        if not getattr(model_config, "enforce_eager", False):
            raise ValueError(
                "Lightly context parallel currently only supports eager mode."
            )
        if parallel_config is None:
            raise PatchCompatibilityError(
                "Lightly-CP requires VllmConfig.parallel_config"
            )
        if getattr(parallel_config, "decode_context_parallel_size", 1) > 1:
            raise ValueError(
                "Lightly context parallel and DCP cannot be enabled simultaneously."
            )

    if feature_config.moe_backend == "dpsk_deep_gemm":
        if kernel_config is None:
            raise PatchCompatibilityError(
                "dpsk_deep_gemm requires VllmConfig.kernel_config"
            )
        upstream_backend = getattr(kernel_config, "moe_backend", None)
        if upstream_backend == "dpsk_deep_gemm":
            # Defensive normalization for programmatic objects that bypassed
            # EngineArgs.  Pydantic's official Literal must never see this.
            setattr(kernel_config, "moe_backend", "auto")
        elif upstream_backend != "auto":
            raise ValueError(
                "HCU sidecar selects dpsk_deep_gemm but upstream "
                f"KernelConfig.moe_backend selects {upstream_backend!r}"
            )
    return feature_config


def _request_cudagraph_buckets_enabled() -> bool:
    from vllm_hcu.platforms import envs as hcu_envs

    return bool(
        hcu_envs.VLLM_HCU_USE_CUSTOM_OPS
        and hcu_envs.VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS
    )


def _replace_with_request_cudagraph_buckets(
    vllm_config: object,
    *,
    compile_sizes_template: list[int | str] | None,
) -> None:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if compilation_config is None or scheduler_config is None:
        raise PatchCompatibilityError(
            "request cudagraph buckets require compilation_config and scheduler_config"
        )
    current_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    max_size = getattr(compilation_config, "max_cudagraph_capture_size", None)
    if not current_sizes or not isinstance(max_size, int) or max_size < 1:
        return

    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = getattr(speculative_config, "num_speculative_tokens", 0)
    decode_query_len = 1 + (num_speculative_tokens or 0)
    sizes = [
        request_size * decode_query_len
        for request_size in _REQUEST_CAPTURE_SIZES
        if request_size * decode_query_len <= max_size
    ]

    max_num_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
    if (
        isinstance(max_num_tokens, int)
        and max_num_tokens <= max_size
        and max_num_tokens not in sizes
    ):
        sizes.append(max_num_tokens)
    sizes = sorted(set(sizes))
    if not sizes:
        raise ValueError(
            "No valid request-oriented cudagraph bucket fits within "
            f"max_cudagraph_capture_size={max_size}"
        )

    compilation_config.cudagraph_capture_sizes = sizes
    compilation_config.max_cudagraph_capture_size = sizes[-1]
    # Upstream consumes the symbolic cudagraph sentinel during its first
    # post-init.  Restore the template and recompute after changing the list.
    compilation_config.compile_sizes = compile_sizes_template
    post_init = getattr(compilation_config, "post_init_cudagraph_sizes", None)
    if not callable(post_init):
        raise PatchCompatibilityError(
            "CompilationConfig.post_init_cudagraph_sizes is missing"
        )
    post_init()


def apply_to_module(module: ModuleType) -> bool:
    vllm_module = load_exact_module(TARGET_MODULE, module)
    vllm_config = getattr(vllm_module, "VllmConfig", None)
    model_config_class = getattr(vllm_module, "ModelConfig", None)
    if not isinstance(vllm_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.VllmConfig is missing"
        )
    if not isinstance(model_config_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.config.model.ModelConfig is missing"
        )
    if getattr(vllm_config, _MARKER, False):
        return False

    with_hf_config = vars(vllm_config).get("with_hf_config")
    set_cudagraph_sizes = vars(vllm_config).get("_set_cudagraph_sizes")
    get_v2_unsupported_features = vars(vllm_config).get(
        "_get_v2_model_runner_unsupported_features"
    )
    validate_v2_model_runner = vars(vllm_config).get("_validate_v2_model_runner")
    get_model_arch_config = vars(model_config_class).get("get_model_arch_config")
    if (
        not callable(with_hf_config)
        or not callable(set_cudagraph_sizes)
        or not callable(get_v2_unsupported_features)
        or not callable(validate_v2_model_runner)
        or not callable(get_model_arch_config)
    ):
        raise PatchCompatibilityError(
            "required HCU VllmConfig compatibility methods are missing"
        )
    model_arch_signature = inspect.signature(get_model_arch_config)
    if tuple(model_arch_signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[4]} has incompatible "
            f"signature {model_arch_signature}"
        )
    with_hf_signature = inspect.signature(with_hf_config)
    if tuple(with_hf_signature.parameters) != (
        "self",
        "hf_config",
        "architectures",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {with_hf_signature}"
        )
    cudagraph_signature = inspect.signature(set_cudagraph_sizes)
    if tuple(cudagraph_signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible "
            f"signature {cudagraph_signature}"
        )
    unsupported_features_signature = inspect.signature(get_v2_unsupported_features)
    if tuple(unsupported_features_signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[2]} has incompatible "
            f"signature {unsupported_features_signature}"
        )
    validate_v2_signature = inspect.signature(validate_v2_model_runner)
    if tuple(validate_v2_signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[3]} has incompatible "
            f"signature {validate_v2_signature}"
        )

    @functools.wraps(get_model_arch_config)
    def hcu_get_model_arch_config(self):
        hf_config = getattr(self, "hf_config", None)
        get_text_config = getattr(hf_config, "get_text_config", None)
        if not callable(get_text_config):
            raise PatchCompatibilityError(
                "ModelConfig.hf_config.get_text_config is missing"
            )
        self.hf_text_config = get_text_config()
        return get_model_arch_config(self)

    setattr(
        model_config_class,
        "_vllm_hcu_original_get_model_arch_config",
        get_model_arch_config,
    )
    setattr(
        model_config_class,
        "get_model_arch_config",
        hcu_get_model_arch_config,
    )

    @functools.wraps(with_hf_config)
    def hcu_with_hf_config(self, hf_config: object, architectures=None):
        updated = with_hf_config(self, hf_config, architectures)
        model_config = getattr(updated, "model_config", None)
        if model_config is None:
            return updated
        installed_hf_config = getattr(model_config, "hf_config", None)
        get_text_config = getattr(installed_hf_config, "get_text_config", None)
        if not callable(get_text_config):
            raise PatchCompatibilityError(
                "updated HuggingFace config does not expose get_text_config"
            )
        model_config.hf_text_config = get_text_config()
        refresh_model_arch_config = getattr(
            model_config, "get_model_arch_config", None
        )
        if not callable(refresh_model_arch_config):
            raise PatchCompatibilityError(
                "ModelConfig.get_model_arch_config is missing"
            )
        model_config.model_arch_config = refresh_model_arch_config()
        return updated

    @functools.wraps(set_cudagraph_sizes)
    def hcu_set_cudagraph_sizes(self) -> Any:
        # VllmConfig.__post_init__ performs its first cudagraph-size pass
        # before current_platform.check_and_update_config().  Bind directly
        # from the authoritative sidecar at this earlier boundary so the
        # CompilationConfig custom-SP wrapper observes the requested feature
        # during that first pass.  The later platform validation intentionally
        # keeps rebinding after spawn/unpickle.
        bind_hcu_config(self)
        compilation_config = getattr(self, "compilation_config", None)
        if compilation_config is None:
            raise PatchCompatibilityError("VllmConfig.compilation_config is missing")
        explicit_sizes = compilation_config.cudagraph_capture_sizes is not None
        compile_sizes = getattr(compilation_config, "compile_sizes", None)
        compile_sizes_template = (
            list(compile_sizes) if compile_sizes is not None else None
        )

        result = set_cudagraph_sizes(self)
        if explicit_sizes or not _request_cudagraph_buckets_enabled():
            return result
        _replace_with_request_cudagraph_buckets(
            self,
            compile_sizes_template=compile_sizes_template,
        )
        return result

    @functools.wraps(get_v2_unsupported_features)
    def hcu_get_v2_model_runner_unsupported_features(self) -> list[str]:
        unsupported = get_v2_unsupported_features(self)
        try:
            hcu_pcp_enabled = _validate_hcu_pcp_scope(self)
        except ValueError:
            # Leave every out-of-scope PCP configuration on the exact upstream
            # rejection path. The direct scope check provides its actionable
            # diagnostic without relaxing vLLM's generic V2 contract.
            return unsupported
        if not hcu_pcp_enabled:
            return unsupported
        return [
            feature
            for feature in unsupported
            if feature != "prefill context parallelism"
        ]

    @functools.wraps(validate_v2_model_runner)
    def hcu_validate_v2_model_runner(self) -> Any:
        parallel_config = _require_hcu_pcp_attribute(
            self, "parallel_config", "VllmConfig"
        )
        pcp_size = _require_hcu_pcp_attribute(
            parallel_config, "prefill_context_parallel_size", "ParallelConfig"
        )
        if pcp_size > 1:
            try:
                _validate_hcu_pcp_scope(self)
            except ValueError:
                # Preserve v0.25.1 validation for every rejected PCP shape.
                return validate_v2_model_runner(self)
        return validate_v2_model_runner(self)

    setattr(vllm_config, "_vllm_hcu_original_with_hf_config", with_hf_config)
    setattr(vllm_config, "with_hf_config", hcu_with_hf_config)
    setattr(
        vllm_config,
        "_vllm_hcu_original_set_cudagraph_sizes",
        set_cudagraph_sizes,
    )
    setattr(vllm_config, "_set_cudagraph_sizes", hcu_set_cudagraph_sizes)
    setattr(
        vllm_config,
        "_vllm_hcu_original_get_v2_model_runner_unsupported_features",
        get_v2_unsupported_features,
    )
    setattr(
        vllm_config,
        "_get_v2_model_runner_unsupported_features",
        hcu_get_v2_model_runner_unsupported_features,
    )
    setattr(
        vllm_config,
        "_vllm_hcu_original_validate_v2_model_runner",
        validate_v2_model_runner,
    )
    setattr(
        vllm_config,
        "_validate_v2_model_runner",
        hcu_validate_v2_model_runner,
    )
    setattr(vllm_config, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    vllm_module = load_exact_module(TARGET_MODULE, module)
    vllm_config = getattr(vllm_module, "VllmConfig", None)
    if not isinstance(vllm_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.VllmConfig is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=vllm_config,
        marker=_MARKER,
        callback=lambda: apply_to_module(vllm_module),
    )


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "validate_and_update_hcu_config",
]
