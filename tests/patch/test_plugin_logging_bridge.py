# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Review 2026-10-15 P2: the ``vllm_hcu`` logger must MIRROR the level
configured on the ``vllm`` logger instead of pinning INFO.

The two logger trees are siblings under the global root, so an unset plugin
level canNOT inherit VLLM_LOGGING_LEVEL via the hierarchy; the bridge now
copies the configured ``vllm`` level (NOTSET -> INFO legacy default).  The
old hard ``INFO`` dropped plugin ``.debug()`` diagnostics in DEBUG
deployments and let plugin INFO ride through at WARNING+.
"""

from __future__ import annotations

import logging

import pytest

import vllm_hcu as plugin  # noqa: F401  (importing the package arms the bridge)


def _rebridge(root: logging.Logger) -> None:
    root._hcu_bridge_done = False  # type: ignore[attr-defined]
    plugin._bridge_plugin_logging()


def test_bridge_mirrors_configured_vllm_level() -> None:
    # Mirror-path premise (R9 P3-b): a handlerless vllm logger makes the
    # bridge defer without touching ANY plugin state, so the mirror
    # assertions below would hard-fail in that exotic environment.
    if not logging.getLogger("vllm").handlers:
        pytest.skip("vllm logger unconfigured; test targets the mirror path")
    root = logging.getLogger("vllm_hcu")
    vllm_log = logging.getLogger("vllm")
    saved_flag = getattr(root, "_hcu_bridge_done", False)
    saved_level, saved_vllm_level = root.level, vllm_log.level
    saved_handlers = list(root.handlers)
    try:
        vllm_log.setLevel(logging.DEBUG)
        root.handlers.clear()
        _rebridge(root)
        assert root.level == logging.DEBUG
        assert root.handlers, "bridge must attach the vllm handlers"

        vllm_log.setLevel(logging.WARNING)
        root.handlers.clear()
        _rebridge(root)
        assert root.level == logging.WARNING

        # Unconfigured ``vllm`` logger: keep the legacy INFO default.
        vllm_log.setLevel(logging.NOTSET)
        root.handlers.clear()
        _rebridge(root)
        assert root.level == logging.INFO
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        vllm_log.setLevel(saved_vllm_level)
        if saved_flag:
            root._hcu_bridge_done = True  # type: ignore[attr-defined]


def test_rebridge_does_not_duplicate_handlers() -> None:
    # Premise (round-6 #C): targets the MIRROR path.  Since round 8 the
    # bridge no longer mints a fallback handler at all, but an exotic
    # handlerless environment would still make `before` empty.
    if not logging.getLogger("vllm").handlers:
        pytest.skip("vllm logger unconfigured; test targets the mirror path")
    root = logging.getLogger("vllm_hcu")
    saved_flag = getattr(root, "_hcu_bridge_done", False)
    try:
        root._hcu_bridge_done = False  # type: ignore[attr-defined]
        plugin._bridge_plugin_logging()
        before = list(root.handlers)
        assert before
        # Force a second pass WITH the latch cleared so the per-handler
        # dedup branch is actually exercised (round-5 review #C).
        for handler in logging.getLogger("vllm").handlers:
            if handler not in root.handlers:  # pragma: no cover - hygiene
                root.addHandler(handler)
        root._hcu_bridge_done = False  # type: ignore[attr-defined]
        plugin._bridge_plugin_logging()
        assert list(root.handlers) == before
        plugin._bridge_plugin_logging()  # latched: no re-add either
        assert list(root.handlers) == before
    finally:
        if saved_flag:
            root._hcu_bridge_done = True  # type: ignore[attr-defined]


def test_handlerless_vllm_defers_to_operator_root() -> None:
    """Review 2026-10-15 round 8 P2: when the ``vllm`` logger has no
    handlers (unconfigured vLLM or VLLM_CONFIGURE_LOGGING=0), the plugin
    must NOT install its own fallback handler nor latch: its records keep
    following the operator's root policy, and the cleared latch leaves any
    future bridge retry open (nothing schedules one today -- R9 P3-c)."""

    root = logging.getLogger("vllm_hcu")
    vllm_log = logging.getLogger("vllm")
    saved_flag = getattr(root, "_hcu_bridge_done", False)
    saved_level, saved_propagate = root.level, root.propagate
    saved_handlers = list(root.handlers)
    saved_vllm_handlers = list(vllm_log.handlers)
    try:
        vllm_log.handlers.clear()
        root.handlers.clear()
        root.setLevel(logging.NOTSET)
        root.propagate = True
        root._hcu_bridge_done = False  # type: ignore[attr-defined]

        plugin._bridge_plugin_logging()
        assert root.handlers == [], "no plugin-owned fallback handler"
        assert root.propagate is True, "records follow the root policy"
        # R10 P3: only the latch state is asserted; a later bridge call is
        # NOT blocked by the latch, but nothing schedules one today.
        assert getattr(root, "_hcu_bridge_done", False) is False

        plugin._bridge_plugin_logging()  # repeat: still a no-op
        assert root.handlers == []
    finally:
        vllm_log.handlers[:] = saved_vllm_handlers
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        root.propagate = saved_propagate
        if saved_flag:
            root._hcu_bridge_done = True  # type: ignore[attr-defined]
