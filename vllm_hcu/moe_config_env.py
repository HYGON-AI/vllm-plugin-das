# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Point vLLM's tuned-MoE-config lookup at the configs owned by AITER.

``vllm.model_executor.layers.fused_moe.fused_moe.get_moe_configs`` resolves a
layer's tile configuration from ``E=..,N=..,device_name=..[,dtype=..].json``,
looking first in ``VLLM_TUNED_CONFIG_FOLDER`` and then in the configs shipped
with vLLM.

AITER owns the concrete vLLM-compatible tuning data in
``aiter/configs/vllm_moe``.  This module keeps only the discovery hook that
points vLLM at that package directory; it ships no tuning row of its own.  A
user-selected folder is left untouched: it is the documented way to tune a
shape AITER does not ship, and vLLM consults it before the bundled configs.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import MutableMapping


logger = logging.getLogger(__name__)

TUNED_CONFIG_FOLDER_ENV = "VLLM_TUNED_CONFIG_FOLDER"
AITER_PACKAGE = "aiter"
AITER_MOE_CONFIG_SUBDIR = ("configs", "vllm_moe")


def _aiter_package_dir() -> str | None:
    """Return AITER's installed package directory without importing it.

    Plugin discovery must stay dependency-light, and importing ``aiter`` pulls
    in torch and the JIT layer.  ``find_spec`` only locates the package.
    """

    try:
        spec = importlib.util.find_spec(AITER_PACKAGE)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    if locations:
        return os.path.abspath(next(iter(locations)))
    if spec.origin:
        return os.path.dirname(os.path.abspath(spec.origin))
    return None


def get_tuned_moe_config_folder() -> str | None:
    """Return AITER's vLLM-schema MoE config folder, or ``None`` if absent."""

    package_dir = _aiter_package_dir()
    if package_dir is None:
        return None
    folder = os.path.join(package_dir, *AITER_MOE_CONFIG_SUBDIR)
    return folder if os.path.isdir(folder) else None


# Resolved at import for callers that only need the path; configure() resolves
# it again so tests and late installs are not stuck with an import-time miss.
MOE_CONFIG_FOLDER = get_tuned_moe_config_folder()


def configure_tuned_moe_config_folder(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Default ``VLLM_TUNED_CONFIG_FOLDER`` to AITER's tuned MoE configs.

    A user-selected folder is left untouched.  When AITER is not installed the
    variable stays unset and vLLM keeps its own lookup and fallback; the plugin
    never serves a duplicate copy of the tuning data.
    """

    environ = os.environ if environ is None else environ
    if environ.get(TUNED_CONFIG_FOLDER_ENV):
        return

    folder = get_tuned_moe_config_folder()
    if folder is None:
        logger.warning(
            "AITER's vLLM-compatible MoE config folder was not found; "
            "leaving %s unset so vLLM uses its own configs.",
            TUNED_CONFIG_FOLDER_ENV,
        )
        return

    environ[TUNED_CONFIG_FOLDER_ENV] = folder
    logger.info(
        "Serving tuned MoE configs from AITER at %s; set %s to override.",
        folder,
        TUNED_CONFIG_FOLDER_ENV,
    )


__all__ = [
    "AITER_MOE_CONFIG_SUBDIR",
    "AITER_PACKAGE",
    "MOE_CONFIG_FOLDER",
    "TUNED_CONFIG_FOLDER_ENV",
    "configure_tuned_moe_config_folder",
    "get_tuned_moe_config_folder",
]
