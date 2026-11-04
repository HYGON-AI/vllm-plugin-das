# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Mode-gated, exact cold imports for LocateAnything Fast.

The original vLLM installation remains read-only. Large implementations live
in vllm_hcu.v1.locateanything and are loaded by the common import coordinator.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from functools import lru_cache


def locateanything_mode() -> str:
    mode = os.environ.get("VLLM_HCU_LOCATEANYTHING_MODE", "slow")
    if mode not in {"slow", "fast"}:
        raise ValueError("VLLM_HCU_LOCATEANYTHING_MODE must be slow or fast")
    return mode


@lru_cache(maxsize=1)
def fast_source_contract() -> dict:
    path = Path(__file__).resolve().parents[1] / "v1/locateanything/source_contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_fast_sources(contract: dict, vllm_root: Path | None = None) -> None:
    """Fail before any exchange if the qualified upstream source has drifted."""
    if vllm_root is None:
        vllm_root = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    for name, spec in contract.items():
        source = vllm_root / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != spec["source_sha256"]:
            raise RuntimeError(
                f"LocateAnything Fast vLLM source contract mismatch: {name}; "
                "use the qualified HCU vLLM build or rebase and validate the Fast runtime"
            )


@lru_cache(maxsize=None)
def _canonical_factory(canonical: str, owner: str):
    """Keep one module/globals object while satisfying exact-name callbacks."""
    def load():
        module = importlib.import_module(owner)
        if module.__name__ not in {owner, canonical}:
            raise RuntimeError(f"Unexpected LocateAnything module identity: {module.__name__}")
        module.__name__ = canonical
        return module
    return load


def register_locateanything_exchanges(coordinator) -> None:
    if locateanything_mode() != "fast":
        return
    contract = fast_source_contract()
    validate_fast_sources(contract)
    for name, spec in contract.items():
        canonical = "vllm." + name[:-3].replace("/", ".")
        coordinator.register_replacement(
            "module_exchange.locateanything.fast." + Path(name).stem + "." + canonical,
            canonical,
            _canonical_factory(canonical, spec["module"]),
            targets=(canonical,),
        )
