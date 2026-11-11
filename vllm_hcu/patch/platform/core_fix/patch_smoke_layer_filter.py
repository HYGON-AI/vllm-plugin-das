# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K3 smoke layer loading, reproducing the 877e core layering in plugin.

Mirrors the upstream (kimi-k3 dev line) implementation structure so behaviour
matches the core design without modifying deployed vLLM code:

* ``resolve_smoke_layer_limit(model_config)`` — validate ``VLLM_SKIP_WEIGHT``
  plus an explicit reduced ``text_config.num_hidden_layers`` override against
  the ORIGINAL checkpoint ``config.json``; returns the layer limit or ``None``
  (weight_utils.py:184 upstream).
* ``should_skip_smoke_layer_weight(weight_name, layer_limit)`` — key-level
  predicate on the same regex as upstream (weight_utils.py:162).
* ``disable_smoke_layer_limit()`` — context manager that mutes resolution
  process-wide (weight_utils.py:174).  The DSpark draft load is wrapped with
  it by ``worker/framework_opt/patch_smoke_draft_guard.py`` (upstream
  c4fc87056e wraps ``get_model`` inside ``load_dspark_model``).
* ``DefaultModelLoader.load_weights`` resolves the limit once per load,
  ``_get_weights_iterator`` routes to a skip-BEFORE-READ safetensors iterator
  (saves the full 96-shard I/O a post-filter would pay) and rejects loader
  combinations the upstream design rejects (fastsafetensors / instanttensor /
  npcache / torchao / non-safetensors checkpoints).

Registered through the standard platform/core import coordinator; arms the
first time ``vllm.model_executor.model_loader.default_loader`` is imported.
This module must stay importable without vLLM (the worker draft guard imports
``disable_smoke_layer_limit`` from here).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

PATCH_ID = "platform.core.kimi_k3_smoke_layer_filter"
TARGET_MODULE = "vllm.model_executor.model_loader.default_loader"
TARGETS = (
    f"{TARGET_MODULE}.DefaultModelLoader.load_weights",
    f"{TARGET_MODULE}.DefaultModelLoader._get_weights_iterator",
)

_WEIGHT_UTILS_MODULE = "vllm.model_executor.model_loader.weight_utils"
_LAYER_KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Same truthy set as upstream envs.VLLM_SKIP_WEIGHT.
_SMOKE_TRUTHY = ("1", "true")
_REJECTED_LOAD_FORMATS = ("npcache", "fastsafetensors", "instanttensor")

_PATCHED_FLAG = "_k3_smoke_filter_patched"
_WRAPPER_FLAG = "_k3_smoke_filter_wrapper"

# Upstream keeps a save/restore bool; with interleaved exits from different
# threads that restores False while another holder is still inside.  A locked
# depth counter keeps the same process-wide visibility without that race.
_DISABLE_LOCK = threading.Lock()
_DISABLE_DEPTH = 0
_ENABLED_LOGGED = False

_log = logging.getLogger("vllm_hcu.patch.kimi_k3_smoke_layer_filter")

__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply_to_module",
    "disable_smoke_layer_limit",
    "parse_smoke_layer_id",
    "resolve_smoke_layer_limit",
    "should_skip_smoke_layer_weight",
    "smoke_layer_limit_disabled",
    "smoke_weight_env_enabled",
]


def smoke_weight_env_enabled() -> bool:
    return os.environ.get("VLLM_SKIP_WEIGHT", "0").strip().lower() in _SMOKE_TRUTHY


def smoke_layer_limit_disabled() -> bool:
    return _DISABLE_DEPTH > 0


@contextmanager
def disable_smoke_layer_limit():
    """Mute smoke resolution; mirrors upstream disable_smoke_layer_limit().

    Reentrant.  If a future core ships its own switch, it is entered as well
    so core-native resolution is muted too.
    """
    global _DISABLE_DEPTH
    core_cm = getattr(
        sys.modules.get(_WEIGHT_UTILS_MODULE), "disable_smoke_layer_limit", None
    )
    with _DISABLE_LOCK:
        _DISABLE_DEPTH += 1
    try:
        if core_cm is not None:
            with core_cm():
                yield
        else:
            yield
    finally:
        with _DISABLE_LOCK:
            _DISABLE_DEPTH -= 1


def parse_smoke_layer_id(weight_name: str) -> int | None:
    match = _LAYER_KEY_RE.search(weight_name)
    return int(match.group(1)) if match else None


def should_skip_smoke_layer_weight(weight_name: str, layer_limit: int | None) -> bool:
    if layer_limit is None:
        return False
    layer_id = parse_smoke_layer_id(weight_name)
    return layer_id is not None and layer_id >= layer_limit


def resolve_smoke_layer_limit(model_config) -> int | None:
    """Validate and return the debug layer limit, mirroring upstream 184."""
    global _ENABLED_LOGGED

    if not smoke_weight_env_enabled() or smoke_layer_limit_disabled():
        return None

    text_config = getattr(model_config.hf_config, "text_config", None)
    effective_layers = getattr(text_config, "num_hidden_layers", None)
    config_path = Path(model_config.model) / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "VLLM_SKIP_WEIGHT is debug-only and requires a local checkpoint "
            "config.json to validate the explicit reduced layer override."
        )
    with config_path.open(encoding="utf-8") as config_file:
        original_config = json.load(config_file)
    original_layers = original_config.get("text_config", original_config).get(
        "num_hidden_layers"
    )
    if not isinstance(effective_layers, int) or not isinstance(original_layers, int):
        raise ValueError(
            "VLLM_SKIP_WEIGHT requires text_config.num_hidden_layers to be an int."
        )
    if not 0 < effective_layers < original_layers:
        raise ValueError(
            "VLLM_SKIP_WEIGHT requires an explicit reduced "
            "--hf-overrides '{\"text_config\":{\"num_hidden_layers\":N}}' "
            f"with 0 < N < {original_layers}; got N={effective_layers}."
        )
    if not _ENABLED_LOGGED:
        _ENABLED_LOGGED = True
        _log.warning(
            "VLLM_SKIP_WEIGHT debug mode enabled: loading decoder layers [0, %d) "
            "out of %d; global checkpoint weights remain enabled.",
            effective_layers,
            original_layers,
        )
    return effective_layers


def _format_name(value) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _smoke_safetensors_iterator(
    hf_weights_files,
    layer_limit: int,
    *,
    use_tqdm_on_load: bool,
    local_expert_ids,
):
    """Skip-before-read iterator; mirrors the smoke branch of upstream
    ``safetensors_weights_iterator`` (no prefetch, no eager whole-file read,
    EP expert filter kept, tensors read only for surviving keys)."""
    from safetensors import safe_open

    wu = sys.modules.get(_WEIGHT_UTILS_MODULE)
    sort_key = getattr(wu, "_natural_sort_key", None)
    should_skip_weight = getattr(wu, "should_skip_weight", None)
    if local_expert_ids is not None and should_skip_weight is None:
        raise ValueError(
            "VLLM_SKIP_WEIGHT: loader has local_expert_ids but weight_utils "
            "exposes no should_skip_weight; refusing to load all experts."
        )

    files = sorted(hf_weights_files, key=sort_key) if sort_key else sorted(hf_weights_files)
    tqdm = getattr(wu, "tqdm", None)
    enable_tqdm = getattr(wu, "enable_tqdm", None)
    if tqdm is not None and enable_tqdm is not None:
        files = tqdm(
            files,
            desc="Loading safetensors checkpoint shards (smoke)",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=getattr(wu, "_BAR_FORMAT", None),
        )

    skipped = 0
    for st_file in files:
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if (
                    local_expert_ids is not None
                    and should_skip_weight(name, local_expert_ids)
                ) or should_skip_smoke_layer_weight(name, layer_limit):
                    skipped += 1
                    continue
                yield name, f.get_tensor(name)
    _log.info(
        "[k3-smoke-filter] skipped %d tensors before read (layers>=%d or "
        "non-local experts)",
        skipped,
        layer_limit,
    )


def _apply(module: ModuleType) -> bool:
    wu = sys.modules.get(_WEIGHT_UTILS_MODULE)
    if wu is not None and hasattr(wu, "resolve_smoke_layer_limit"):
        # Core already implements the smoke layering; don't stack a second one.
        _log.info("[k3-smoke-filter] core has native smoke layering; not patching")
        return False

    loader_cls = getattr(module, "DefaultModelLoader", None)
    if loader_cls is None:
        raise RuntimeError(f"{TARGET_MODULE}.DefaultModelLoader is missing")
    if loader_cls.__dict__.get(_PATCHED_FLAG, False):
        for name in ("load_weights", "_get_weights_iterator"):
            if not getattr(loader_cls.__dict__.get(name), _WRAPPER_FLAG, False):
                raise RuntimeError(
                    f"[k3-smoke-filter] marker set but {name} was replaced; "
                    "restart the process"
                )
        return False

    original_get_iterator = loader_cls._get_weights_iterator
    original_load_weights = loader_cls.load_weights

    # Plain function (not a generator): the non-smoke path returns the
    # original iterator untouched, so its eager work/exceptions keep timing.
    def _get_weights_iterator(self, source):
        layer_limit = getattr(self, "smoke_layer_limit", None)
        if layer_limit is None:
            return original_get_iterator(self, source)

        load_config = self.load_config
        load_format = _format_name(getattr(load_config, "load_format", None))
        if load_format in _REJECTED_LOAD_FORMATS:
            raise ValueError(
                f"VLLM_SKIP_WEIGHT does not support load-format={load_format}; "
                "use the default safetensors loader so tensors are skipped before I/O."
            )
        if _format_name(getattr(load_config, "safetensors_load_strategy", None)) == "torchao":
            raise ValueError(
                "VLLM_SKIP_WEIGHT does not support torchao safetensors; "
                "use the default safetensors loader."
            )
        _hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if not use_safetensors:
            raise ValueError(
                "VLLM_SKIP_WEIGHT requires a safetensors checkpoint and the "
                "default loader."
            )
        weights_iterator = _smoke_safetensors_iterator(
            hf_weights_files,
            layer_limit,
            use_tqdm_on_load=getattr(load_config, "use_tqdm_on_load", False),
            local_expert_ids=getattr(self, "local_expert_ids", None),
        )
        if getattr(self, "counter_before_loading_weights", None) == 0.0:
            import time

            self.counter_before_loading_weights = time.perf_counter()
        prefix = source.prefix
        return ((prefix + name, tensor) for (name, tensor) in weights_iterator)

    def load_weights(self, model, model_config):
        self.smoke_layer_limit = resolve_smoke_layer_limit(model_config)
        try:
            return original_load_weights(self, model, model_config)
        finally:
            self.smoke_layer_limit = None

    for wrapper, original in (
        (_get_weights_iterator, original_get_iterator),
        (load_weights, original_load_weights),
    ):
        wrapper.__name__ = original.__name__
        wrapper.__qualname__ = original.__qualname__
        wrapper.__doc__ = original.__doc__
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(wrapper, _WRAPPER_FLAG, True)

    loader_cls._get_weights_iterator = _get_weights_iterator
    loader_cls.load_weights = load_weights
    setattr(loader_cls, _PATCHED_FLAG, True)

    _log.info(
        "[k3-smoke-filter] armed (VLLM_SKIP_WEIGHT=%r)",
        os.environ.get("VLLM_SKIP_WEIGHT"),
    )
    return True


def apply_to_module(module: ModuleType) -> bool:
    try:
        return _apply(module)
    except Exception:
        if smoke_weight_env_enabled():
            # Fail closed: a silently unarmed filter would load every layer.
            raise
        _log.exception("[k3-smoke-filter] failed to arm")
        return False
