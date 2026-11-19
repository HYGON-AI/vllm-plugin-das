# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for Adapter-owned PCP speculative compressor history."""
import sys
from importlib import import_module

try:
    _impl = import_module('hcu_adapter.kernels.pcp_dspark')
except ModuleNotFoundError as error:
    if error.name != 'hcu_adapter.kernels.pcp_dspark':
        raise
    raise RuntimeError(
        'DeepSeek V4.1 PCP DSpark requires the companion hcu-adapter PCP '
        'DSpark implementation (HYGON-AI/adapter-das#139).'
    ) from error

sys.modules[__name__] = _impl
