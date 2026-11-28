# SPDX-License-Identifier: Apache-2.0
"""Explicit standard HY4 DFly admission; imports are inert until selected."""

from hcu_adapter.extensions import ExtensionSpec

_active = False


def check_health():
    if _active:
        from .models.hy_v4_flash.dfly_compat import check_health as owned_health

        owned_health()


def activate(context):
    global _active
    selected = (
        "hy4-dfly" in context.requested_capabilities
        or "das_hcu_hy4_dfly" in context.requested_extensions
    )
    if not selected:
        if _active:
            raise RuntimeError("HY4 DFly publication cannot be disabled")
        return
    if (
        "hy4-static-eplb" in context.requested_capabilities
        or "das_hcu_hy4_static" in context.requested_extensions
    ):
        raise ValueError("HY4 DFly does not admit the static EPLB model owner")
    from .models.hy_v4_flash.dfly_compat import install_compat

    install_compat()
    _active = True
    check_health()


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-hy4-dfly",
        api_version=1,
        priority=103,
        capabilities=frozenset({"hy4-dfly"}),
        activate=activate,
        check_health=check_health,
    )
