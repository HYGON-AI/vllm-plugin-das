# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the DSV4.1 Channel-FP8 config.

The config class moved to `hcu_adapter.dsv41_channel_fp8`. This module keeps the
historical import path alive and replaces itself in `sys.modules` with the
adapter module, so `patch.object(candidate, ...)` and attribute reads reach the
real state instead of a copy.
"""
from __future__ import annotations

import sys

from .. import _seams

_seams.bind_dsv41()

from hcu_adapter import dsv41_channel_fp8 as _impl  # noqa: E402

sys.modules[__name__] = _impl
