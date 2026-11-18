# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Process-local, serialized admission of the optional native RMS provider."""
from __future__ import annotations

from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

from .extension import DasExtensionError

_ACTION = "das-lightop-rms-provider"
_lifecycle = ProcessLocalLifecycle(runtime_action=_ACTION, error_type=DasExtensionError)
_provider = None


def get_provider(importer, validate):
    def resolve():
        global _provider
        try:
            candidate = importer("lightop.norm")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required LightOp norm module: {error}"
            ) from error
        validate(candidate)
        _provider = candidate

    _lifecycle.run_once(_ACTION, resolve)
    return _provider


def check_provider_health() -> None:
    """Report prior resolution failures without loading or repairing a provider."""
    _lifecycle.require_runtime_healthy()
