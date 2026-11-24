# SPDX-License-Identifier: Apache-2.0
"""Translate the retired TTFT switch before Core freezes its environment."""

from __future__ import annotations

from importlib import import_module
import os

from hcu_adapter.extensions import ExtensionSpec


LEGACY_FLAG = "VLLM_HCU_MOONCAKE_TTFT_TRACE"
CORE_FLAG = "VLLM_MOONCAKE_TTFT_TRACE"
_policy = None


def _raw_policy():
    return os.environ.get(LEGACY_FLAG), os.environ.get(CORE_FLAG)


def activate_extension(context):
    """Keep an explicit Core setting; translate enabled legacy values once."""
    global _policy
    if _policy is not None:
        check_health()
        return
    legacy, canonical = _raw_policy()
    if canonical is None and (legacy or "").lower() in {"1", "true"}:
        envs = import_module("vllm.envs")
        if CORE_FLAG not in envs.environment_variables:
            raise RuntimeError(
                "Mooncake tracing requires Core's VLLM_MOONCAKE_TTFT_TRACE API"
            )
        if envs._is_envs_cache_enabled() and not getattr(envs, CORE_FLAG):
            raise RuntimeError(
                "Mooncake tracing was requested after Core cached its environment; "
                "set VLLM_MOONCAKE_TTFT_TRACE=1 before process startup"
            )
        os.environ.setdefault(CORE_FLAG, "1")
    _policy = _raw_policy()


def check_health():
    """Detect policy drift without changing Core's cache or output methods."""
    if _policy is None or _raw_policy() != _policy:
        raise RuntimeError("Mooncake tracing policy changed; start a fresh process")


def extension_spec():
    """Discover the alias bridge without loading a framework or provider."""
    return ExtensionSpec(
        name="das-hcu-mooncake-trace",
        api_version=1,
        priority=120,
        capabilities=frozenset({"mooncake-ttft-trace-env"}),
        activate=activate_extension,
        check_health=check_health,
    )
