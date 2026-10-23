# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""FlashAttention distribution checks without importing native attention code."""

import importlib.metadata as importlib_metadata

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


# This range describes the Python/native API line exercised by the plugin. It
# does not attest that a particular vendor artifact contains the native
# long-prefill gather workspace fix; deployment images must validate that
# binary capability separately.
_FLASH_ATTN_COMPATIBLE_VERSIONS = SpecifierSet(
    ">=2.8.4,<2.9", prereleases=False
)


def require_compatible_flash_attn() -> None:
    try:
        actual = importlib_metadata.version("flash_attn")
    except importlib_metadata.PackageNotFoundError:
        actual = None

    try:
        parsed = None if actual is None else Version(actual)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"Invalid flash_attn distribution version: {actual!r}"
        ) from exc

    if parsed is None or parsed not in _FLASH_ATTN_COMPATIBLE_VERSIONS:
        raise RuntimeError(
            "HCU requires a compatible flash_attn release "
            f"({_FLASH_ATTN_COMPATIBLE_VERSIONS}); found {actual!r}. "
            "The deployed vendor wheel must separately be validated for "
            "native long-prefill gather."
        )
