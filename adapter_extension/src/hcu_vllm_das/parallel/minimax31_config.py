# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Deployment artifacts for the admitted MiniMax path; no environment switches.

Numerical choices belong to quantization/attention config. Performance paths are
on by default within their validated scope. These options only locate artifacts.
"""
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RuntimeArtifacts:
    msa_extension_dir: str | None = None
    dataset_manifest: str | None = None
    loading_receipts: str | None = None
    ready_manifest: str | None = None
    consumer_receipts: str | None = None

    def require(self, name):
        value = getattr(self, name)
        if value is None:
            raise ValueError(f'additional_config.minimax31.{name} is required')
        return value


def artifacts(config=None):
    if config is None:
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
    additional = getattr(config, 'additional_config', None)
    if additional is None:
        additional = {}
    if not isinstance(additional, dict):
        raise ValueError('additional_config must be an object')
    value = additional.get('minimax31', {})
    allowed = {f.name for f in fields(RuntimeArtifacts)}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('unknown MiniMax deployment artifact option')
    if any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError('MiniMax deployment artifact paths must be nonempty strings')
    return RuntimeArtifacts(**value)
