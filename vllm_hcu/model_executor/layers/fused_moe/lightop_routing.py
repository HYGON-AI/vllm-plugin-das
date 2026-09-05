# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compatibility checks for the LightOp fused-MoE routing ABI.

The original ``lightop.moe.moe_fused_gate`` ABI has no arguments for the
router scoring function or for top-k renormalization.  The kernel therefore
has one fixed semantic contract (sigmoid scoring followed by
renormalization).  Keeping that fact in one small helper prevents the two HCU
router entry points from drifting apart.

Newer LightOp versions can extend the contract without a vLLM change by
exporting ``supports_moe_fused_gate_routing`` from ``lightop.moe``.  The hook
must have the following keyword-only shape::

    supports_moe_fused_gate_routing(
        *, scoring_func: str, renormalize: bool
    ) -> bool

When it returns ``True`` for a non-legacy mode, the corresponding
``moe_fused_gate`` wrapper must also accept ``scoring_func=`` and
``renormalize=`` keyword arguments.  A missing hook is deliberately treated
as an old LightOp build, so unsupported modes fail safe by using the official
router instead of silently changing expert IDs or weights.
"""

from __future__ import annotations

from typing import Any


_LEGACY_SCORING_FUNC = "sigmoid"


def lightop_moe_gate_kwargs(
    lightop_moe: Any,
    scoring_func: str | None,
    renormalize: bool | None,
) -> dict[str, Any] | None:
    """Return LightOp routing kwargs, or ``None`` for standard-router fallback.

    ``{}`` means that the existing positional ABI is safe to call.  A
    non-empty mapping is used only after a newer backend explicitly advertises
    support through the capability hook described in this module.
    """

    if scoring_func == _LEGACY_SCORING_FUNC and bool(renormalize):
        return {}

    capability = getattr(lightop_moe, "supports_moe_fused_gate_routing", None)
    if capability is None:
        # Older LightOp releases have no way to receive these options and
        # always execute sigmoid + renormalize.  Do not call them for another
        # vLLM routing configuration.
        return None
    if not callable(capability):
        raise RuntimeError(
            "lightop.moe.supports_moe_fused_gate_routing must be callable"
        )

    try:
        supported = capability(
            scoring_func=scoring_func,
            renormalize=bool(renormalize),
        )
    except TypeError as exc:
        raise RuntimeError(
            "lightop.moe.supports_moe_fused_gate_routing must accept "
            "scoring_func= and renormalize= keyword arguments"
        ) from exc
    if not bool(supported):
        return None

    # The capability hook and this keyword forwarding form a versioned
    # protocol: a backend opting in must implement the matching gate ABI.
    return {
        "scoring_func": scoring_func,
        "renormalize": bool(renormalize),
    }


__all__ = ["lightop_moe_gate_kwargs"]
