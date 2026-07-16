# SPDX-License-Identifier: Apache-2.0
"""Runtime migration of the transformers qk-rope config correction."""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.transformers_utils.model_arch_config_convertor"
PATCH_ID = "platform.core_fix.model_arch_config_convertor.qk_rope"
TARGETS = (
    f"{TARGET_MODULE}.ModelArchConfigConvertorBase.get_head_size",
    f"{TARGET_MODULE}.ModelArchConfigConvertorBase._get_qk_rope_head_dim",
)
_MARKER = "_vllm_hcu_qk_rope_config_patch_applied"


def apply_to_module(module: ModuleType) -> bool:
    """Apply to an exact module from the import coordinator, without reporting."""

    convertor_module = load_exact_module(TARGET_MODULE, module)
    convertor_class = getattr(convertor_module, "ModelArchConfigConvertorBase", None)
    if not isinstance(convertor_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.transformers_utils."
            "model_arch_config_convertor.ModelArchConfigConvertorBase is missing"
        )
    if getattr(convertor_class, _MARKER, False):
        return False

    def install() -> None:
        if "_get_qk_rope_head_dim" in convertor_class.__dict__:
            raise PatchCompatibilityError(
                "vllm ModelArchConfigConvertorBase already defines "
                "_get_qk_rope_head_dim; refusing to overwrite an unknown API"
            )
        original_get_head_size = require_callable(
            convertor_class,
            "get_head_size",
            "vllm.transformers_utils.model_arch_config_convertor."
            "ModelArchConfigConvertorBase.get_head_size",
        )
        require_positional_signature(
            original_get_head_size,
            "vllm.transformers_utils.model_arch_config_convertor."
            "ModelArchConfigConvertorBase.get_head_size",
            ("self",),
        )
        is_deepseek_mla = require_callable(
            convertor_class,
            "is_deepseek_mla",
            "vllm.transformers_utils.model_arch_config_convertor."
            "ModelArchConfigConvertorBase.is_deepseek_mla",
        )
        require_positional_signature(
            is_deepseek_mla,
            "vllm.transformers_utils.model_arch_config_convertor."
            "ModelArchConfigConvertorBase.is_deepseek_mla",
            ("self",),
        )
        logger = getattr(convertor_module, "logger", None)
        if not callable(getattr(logger, "info", None)):
            raise PatchCompatibilityError(
                "required HCU patch target vllm.transformers_utils."
                "model_arch_config_convertor.logger.info is missing"
            )

        def get_qk_rope_head_dim(self) -> int:
            config = self.hf_text_config
            rope_dim = getattr(config, "qk_rope_head_dim", 0)
            nope_dim = getattr(config, "qk_nope_head_dim", 0)
            if rope_dim == 0 or rope_dim != nope_dim:
                return rope_dim

            model_path = getattr(self.hf_config, "name_or_path", None)
            if not model_path:
                return rope_dim
            try:
                repo_utils = importlib.import_module(
                    "vllm.transformers_utils.repo_utils"
                )
            except Exception as exc:
                raise PatchCompatibilityError(
                    "required qk-rope fallback could not import "
                    "vllm.transformers_utils.repo_utils"
                ) from exc
            get_raw_config = require_callable(
                repo_utils,
                "get_hf_file_to_dict",
                "vllm.transformers_utils.repo_utils.get_hf_file_to_dict",
            )
            raw_config = get_raw_config("config.json", model_path)
            if raw_config and "qk_rope_head_dim" in raw_config:
                corrected = raw_config["qk_rope_head_dim"]
                if corrected != rope_dim:
                    logger.info(
                        "Fixing qk_rope_head_dim: %d -> %d "
                        "(transformers attribute_map bug)",
                        rope_dim,
                        corrected,
                    )
                    config.qk_rope_head_dim = corrected
                    return corrected
            return rope_dim

        @functools.wraps(original_get_head_size)
        def hcu_get_head_size(self) -> int:
            if self.is_deepseek_mla() and not hasattr(
                self.hf_text_config, "compress_ratios"
            ):
                self._get_qk_rope_head_dim()
            return original_get_head_size(self)

        setattr(
            convertor_class,
            "_vllm_hcu_original_get_head_size",
            original_get_head_size,
        )
        setattr(convertor_class, "_get_qk_rope_head_dim", get_qk_rope_head_dim)
        setattr(convertor_class, "get_head_size", hcu_get_head_size)
        setattr(convertor_class, _MARKER, True)

    install()
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Recover ``qk_rope_head_dim`` from raw config when HF aliases corrupt it."""

    convertor_module = load_exact_module(TARGET_MODULE, module)
    convertor_class = getattr(convertor_module, "ModelArchConfigConvertorBase", None)
    if not isinstance(convertor_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.transformers_utils."
            "model_arch_config_convertor.ModelArchConfigConvertorBase is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=convertor_class,
        marker=_MARKER,
        callback=lambda: apply_to_module(convertor_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
