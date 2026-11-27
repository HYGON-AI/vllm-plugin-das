# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Deployment artifacts and startup options for the admitted MiniMax path."""
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RuntimeArtifacts:
    msa_extension_dir: str | None = None
    msa_stage2_receipt: str | None = None
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


@dataclass(frozen=True)
class DecodeOptions:
    shared_expert_schedule: str = "auto"
    shared_w8a8_fused: bool = False
    hipc_verify_tuned: bool = True
    lightop_moe_tuned: bool = False

    def __post_init__(self):
        if self.shared_expert_schedule not in ("auto", "stock", "combine", "dual_window", "aux"):
            raise ValueError("unknown MiniMax decode shared_expert_schedule")
        for name in ("shared_w8a8_fused", "hipc_verify_tuned", "lightop_moe_tuned"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"MiniMax decode {name} must be a boolean")
        if self.shared_w8a8_fused and self.shared_expert_schedule == "stock":
            raise ValueError("MiniMax shared W8A8 fusion requires a shared overlap schedule")


def decode_options(config):
    additional = getattr(config, "additional_config", None)
    if additional is None:
        additional = {}
    if not isinstance(additional, dict):
        raise ValueError("additional_config must be an object")
    value = additional.get("minimax31_decode", {})
    if not isinstance(value, dict) or set(value) - {field.name for field in fields(DecodeOptions)}:
        raise ValueError("unknown MiniMax decode option")
    return DecodeOptions(**value)
