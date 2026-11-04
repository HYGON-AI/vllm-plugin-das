# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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
from functools import wraps


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


def _patch_detection_endpoint(module, class_name, method_name):
    """Keep the checkpoint's output syntax for this model in both modes."""
    from vllm_hcu.patch.platform.framework_opt._common import (
        require_callable, require_class, require_signature_prefix,
    )

    target_name = f"{module.__name__}.{class_name}.{method_name}"
    cls = require_class(module, class_name, target_name)
    original = require_callable(cls, method_name, target_name)
    if getattr(original, "_locateanything_detection_tokens", False):
        return False
    require_signature_prefix(original, target_name, ("self", "request", "raw_request"))

    @wraps(original)
    async def preserve_detection_tokens(self, request, *args, **kwargs):
        if self.model_config.hf_config.model_type == "locateanything":
            # BOX/REF and all coordinate tokens are special tokens. Dropping
            # them destroys detections; these flags describe the model format,
            # including when a client explicitly requests the generic defaults.
            request = request.model_copy(update={
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
            })
        return await original(self, request, *args, **kwargs)

    preserve_detection_tokens._locateanything_detection_tokens = True
    setattr(cls, method_name, preserve_detection_tokens)
    return True


def _patch_chat_detection_tokens(module):
    return _patch_detection_endpoint(module, "OpenAIServingChat", "create_chat_completion")


def _patch_completion_detection_tokens(module):
    return _patch_detection_endpoint(module, "OpenAIServingCompletion", "create_completion")


def register_locateanything_api_callbacks(coordinator):
    # These callbacks wrap shared Slow or exchanged Fast endpoints, selected by
    # the serving model's config, and leave other model families untouched.
    for module, cls, method, callback in (
        ("vllm.entrypoints.openai.chat_completion.serving", "OpenAIServingChat",
         "create_chat_completion", _patch_chat_detection_tokens),
        ("vllm.entrypoints.openai.completion.serving", "OpenAIServingCompletion",
         "create_completion", _patch_completion_detection_tokens),
    ):
        coordinator.register_callback(
            "locateanything.api." + method, module, callback,
            targets=(f"{module}.{cls}.{method}",),
        )
