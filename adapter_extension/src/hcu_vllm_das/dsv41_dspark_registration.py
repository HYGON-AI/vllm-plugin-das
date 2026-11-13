# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the DSpark draft model registration.

The public model-registry handoff moved to `hcu_adapter.dsv41_dspark_registration`.
This module keeps the historical import path alive and replaces itself in
`sys.modules` with the adapter module, so attribute access reaches the real
registry state instead of a copy.
"""
from __future__ import annotations

import sys

from hcu_adapter import dsv41_dspark_registration as _impl

sys.modules[__name__] = _impl
