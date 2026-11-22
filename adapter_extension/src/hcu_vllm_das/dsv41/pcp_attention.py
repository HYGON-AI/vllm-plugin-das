# SPDX-License-Identifier: Apache-2.0
"""Compatibility names for the adapter-owned PCP attention classes.

All names resolve to the existing binding; this module owns no second attention
implementation, scorer or admission policy.
"""
from .. import v41_pcp as _binding

validate_config = _binding.validate_config
_NAMES = {
    "HcuDSV41PcpAttention": "HcuV41PcpAttention",
    "HcuDSV41PcpIndexer": "HcuV41PcpIndexer",
    "HcuDSV41PcpSparseIndexer": "HcuV41PcpSparseIndexer",
    "HcuDSV41PcpBackend": "HcuV41PcpBackend",
    "HcuDSV41PcpSwaBackend": "HcuV41PcpSwaBackend",
}


def __getattr__(name):
    if name not in _NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_binding, _NAMES[name])


__all__ = ["validate_config", *_NAMES]
