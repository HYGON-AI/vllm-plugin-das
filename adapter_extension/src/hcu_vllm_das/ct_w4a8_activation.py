# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the CT W4A8 admission.

The registration moved to `hcu_adapter.ct_w4a8_registration`. This module keeps
the historical import path and symbol surface alive for entry-point discovery and
for the existing replays.

Import stays framework-free: the plugin config class pulls in vLLM, so it is
registered as a binder that only runs at activation. The module then replaces
itself in `sys.modules` with the adapter module, so attribute access and
`patch.object` reach the real state instead of a copy.
"""
from __future__ import annotations

import sys


def _bind_providers() -> None:
    """Install the plugin mechanism; called by the adapter at activation."""
    from . import _seams

    _seams.bind_ct_w4a8()


from hcu_vllm_das._seams import activation_owner  # noqa: E402

# Check only the framework-free binder here. Full provider checks run when the
# binder is invoked at activation, inside the framework.
_impl = activation_owner("ct_w4a8_registration")

_impl.set_provider_binder(_bind_providers)

sys.modules[__name__] = _impl
