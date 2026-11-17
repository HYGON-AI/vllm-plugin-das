# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for Adapter-owned PCP speculative compressor history."""
import sys
from hcu_adapter.kernels import pcp_dspark as _impl

sys.modules[__name__] = _impl
