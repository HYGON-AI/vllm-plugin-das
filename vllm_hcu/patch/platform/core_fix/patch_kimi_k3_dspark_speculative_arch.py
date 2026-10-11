# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep the K3 DSpark draft architecture out of the DeepSeek-V4 rewrite.

``SpeculativeConfig.__post_init__`` normalises every ``dspark`` method request
to the DeepSeek-V4 draft trunk (``model_type="deepseek_v4"`` and
``architectures=["DSparkDraftModel"]``) unless the draft architecture is
exempted by name.  The K3 DSpark draft checkpoint declares
``architectures=["K3DSparkModel"]`` on ``model_type="k3_dspark"`` and is not
covered by the shipped exemption, so the draft is routed onto the
DeepSeek-V4 trunk and crashes on ``config.hc_mult`` (an attribute that only
IHC-enabled DSv4 configs define).

The plugin model registry already maps ``K3DSparkModel`` onto the HCU K3
DSpark implementation, so preserving the draft identity is sufficient.  The
rewrite only reassigns string fields, therefore this adapter runs AFTER the
wheel normalisation and identifies the K3 draft by its surviving config
*class* (``K3DSparkConfig``): the string fields are restored and
``update_arch_`` rebuilds every derived draft-model field from the restored
architecture.  ``draft_model_config`` is ``None`` until ``__post_init__``
assigns it, so the restoration must not depend on state captured before the
wrapped call.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.config.speculative"
PATCH_ID = "platform.core_fix.kimi_k3_dspark_speculative_arch"
TARGETS = (f"{TARGET_MODULE}.SpeculativeConfig.__post_init__",)
_MARKER = "_vllm_hcu_kimi_k3_dspark_speculative_arch_applied"

_DSPARK_METHOD = "dspark"
_K3_CONFIG_NAME = "K3DSparkConfig"
_K3_CONFIG_TYPE = "k3_dspark"
_K3_DSPARK_ARCH = "K3DSparkModel"
_REWRITE_ARCH = "DSparkDraftModel"
_REWRITE_MODEL_TYPE = "deepseek_v4"


def _restore_k3_dspark_identity(self) -> None:
    draft_model_config = getattr(self, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    if type(hf_config).__name__ != _K3_CONFIG_NAME:
        return
    if getattr(hf_config, "architectures", None) == [_REWRITE_ARCH]:
        hf_config.architectures = [_K3_DSPARK_ARCH]
    if getattr(hf_config, "model_type", None) == _REWRITE_MODEL_TYPE:
        hf_config.model_type = _K3_CONFIG_TYPE
    self.update_arch_()


def _install_wrapped_post_init(speculative_config: type) -> bool:
    if getattr(speculative_config, _MARKER, False):
        return False
    owner = f"{TARGET_MODULE}.SpeculativeConfig"
    original = require_callable(
        speculative_config, "__post_init__", f"{owner}.__post_init__"
    )
    signature = inspect.signature(original)
    if tuple(signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"SpeculativeConfig.__post_init__ has incompatible signature {signature}"
        )
    if not callable(getattr(speculative_config, "update_arch_", None)):
        raise PatchCompatibilityError(
            f"required HCU patch target {owner}.update_arch_ is missing"
        )

    @functools.wraps(original)
    def __post_init__(self):  # noqa: ANN001 - mirrors the wrapped signature
        original(self)
        if getattr(self, "method", None) != _DSPARK_METHOD:
            return
        _restore_k3_dspark_identity(self)

    speculative_config.__post_init__ = __post_init__
    setattr(speculative_config, _MARKER, True)
    return True


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    speculative_config = getattr(target, "SpeculativeConfig", None)
    if not isinstance(speculative_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.SpeculativeConfig is missing"
        )
    installed = _install_wrapped_post_init(speculative_config)
    if installed:
        # The class marker guards against double wrapping; the module marker
        # is the bookkeeping owner required by ``apply_once``.
        setattr(target, _MARKER, True)
    return installed


def apply(module: ModuleType | None = None) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=target,
        marker=_MARKER,
        callback=lambda: apply_to_module(target),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
