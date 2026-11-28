# SPDX-License-Identifier: Apache-2.0
"""Explicit new-target identity without changing ordinary CT/MTP dispatch."""


def uses_standard_target(hf_config, *, method):
    kind = getattr(hf_config, "model_type", None)
    if kind == "hy_v4_vl":
        return True
    if kind != "hy_v4":
        return False
    return getattr(hf_config, "hy4_standard_vl", False) is True or method == "dfly"
