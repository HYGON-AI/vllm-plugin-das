# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the DSV4.1 channel-FP8 admission.

The registration moved to `hcu_adapter.dsv41_activation`. This module keeps the
historical import path and symbol surface alive for entry-point discovery and for
the existing replays.

Two things are deliberate. Import stays framework-free: the plugin providers pull
in vLLM, so they are registered as a binder that only runs at activation, inside
the framework. And the module replaces itself in `sys.modules` with the adapter
module once the binder is registered, so attribute access and `patch.object`
reach the real state instead of a copy.
"""
from __future__ import annotations

import sys


def _bind_providers() -> None:
    """Install the plugin mechanism; called by the adapter at activation."""
    from . import _seams

    _seams.bind_dsv41()


from hcu_adapter import dsv41_activation as _impl  # noqa: E402

_impl.set_provider_binder(_bind_providers)

sys.modules[__name__] = _impl
