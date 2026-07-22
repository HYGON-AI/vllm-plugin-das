# SPDX-License-Identifier: Apache-2.0
"""vLLM-HCU release-series and installed-distribution version metadata."""

from __future__ import annotations

import importlib.metadata as importlib_metadata


# This source constant defines the vLLM release series audited by this branch.
# ``setup.py`` computes build provenance for wheel metadata without rewriting
# this tracked file.
__version__ = "0.25.0"
__version_tuple__ = (0, 25, 0)


def get_hcu_version() -> str:
    """Return installed wheel metadata, or the source-tree base version."""

    try:
        return importlib_metadata.version("vllm_hcu")
    except importlib_metadata.PackageNotFoundError:
        return __version__


__hcu_version__ = get_hcu_version()


__all__ = [
    "__hcu_version__",
    "__version__",
    "__version_tuple__",
    "get_hcu_version",
]
