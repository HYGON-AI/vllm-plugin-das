# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Arm vLLM's tuned-MoE-config lookup with the configs shipped by the plugin.

``vllm.model_executor.layers.fused_moe.fused_moe.get_moe_configs`` resolves a
layer's tile configuration from ``E=..,N=..,device_name=..[,dtype=..].json``,
looking first in ``VLLM_TUNED_CONFIG_FOLDER`` and then in the configs shipped
with vLLM.  Every file under ``moe_configs/`` is a single (shape, device) key,
so the specialization stays as narrow as the file name: a shape without a file
keeps vLLM's own lookup, and the user folder still wins when set.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping


logger = logging.getLogger(__name__)

TUNED_CONFIG_FOLDER_ENV = "VLLM_TUNED_CONFIG_FOLDER"
MOE_CONFIG_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_executor",
    "layers",
    "fused_moe",
    "moe_configs",
)


def configure_tuned_moe_config_folder(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Default ``VLLM_TUNED_CONFIG_FOLDER`` to the plugin's tuned MoE configs.

    A user-selected folder is left untouched: it is the documented way to tune
    a shape this plugin does not ship, and vLLM consults it before the bundled
    configs.  Point the variable at any folder without a matching file to fall
    back to vLLM's own configs.
    """

    environ = os.environ if environ is None else environ
    if environ.get(TUNED_CONFIG_FOLDER_ENV):
        return
    environ[TUNED_CONFIG_FOLDER_ENV] = MOE_CONFIG_FOLDER
    logger.info(
        "Serving tuned MoE configs from %s; set %s to override.",
        MOE_CONFIG_FOLDER,
        TUNED_CONFIG_FOLDER_ENV,
    )


__all__ = [
    "MOE_CONFIG_FOLDER",
    "TUNED_CONFIG_FOLDER_ENV",
    "configure_tuned_moe_config_folder",
]
