# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""vLLM-HCU release-series and installed-distribution version metadata."""

from __future__ import annotations

import importlib.metadata as importlib_metadata


# This source constant defines the vLLM release series audited by this branch.
# ``setup.py`` computes build provenance for wheel metadata without rewriting
# this tracked file.
__version__ = "0.28.1rc1.dev489"
__version_tuple__ = (0, 28, 1)
__vllm_target_version__ = (
    "0.28.1rc1.dev489+g4574da606.das.4574da6.dtk2604"
)
__vllm_upstream_sha__ = "58ad1f3b8973b23943107b51230d594050b42ec3"
__vllm_opendas_sha__ = "4574da606553cad5c22448d498f144630a23641e"


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
