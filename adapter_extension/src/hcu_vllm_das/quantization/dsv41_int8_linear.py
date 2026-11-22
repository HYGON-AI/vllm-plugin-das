# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the DSV4.1 CT W8A8 INT8 empty-row method.

The method moved to `hcu_adapter.dsv41_int8_linear`. This module keeps the
historical import path alive and replaces itself in `sys.modules` with the
adapter module, so attribute access reaches the real class instead of a copy.
"""
from __future__ import annotations

from ..adapter_api import adapter_module as _adapter_module

import sys

_impl = _adapter_module('dsv41_int8_linear')

sys.modules[__name__] = _impl
