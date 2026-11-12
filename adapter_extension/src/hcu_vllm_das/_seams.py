# SPDX-License-Identifier: Apache-2.0
"""Plugin-side bindings for the adapter-owned DSV4.1 admission code.

The adapter owns the public quantization registration, but three hooks in it are
plugin-private mechanism: the Channel-FP8 MoE method, the Engram/EDP guard and
the DSpark Graph gate, plus the CT W4A8 config class and the plugin draft model
class path. They are injected here as lazy providers so the plugin import stays on
demand and the adapter never imports or names the plugin.
"""
from __future__ import annotations

from hcu_adapter import (
    ct_w4a8_registration,
    dsv41_activation,
    dsv41_channel_fp8,
    dsv41_dspark_registration,
)

# The adapter seams this plugin binds against. The two repos are separate
# checkouts, so nothing at build time stops a plugin revision from being installed
# against an adapter that predates the seam it calls. Without this check that
# mismatch surfaces as a bare `AttributeError` deep inside entry-point discovery,
# which reads like a plugin bug; naming the missing seam and the seam's purpose
# makes the real cause -- a mixed revision pair -- obvious.
_REQUIRED_SEAMS = (
    (dsv41_channel_fp8, "set_moe_provider",
     "Channel-FP8 MoE provider injection (adapter seam, added for this plugin)"),
    (dsv41_activation, "set_provider_binder",
     "DSV4.1 activation binder (adapter seam, added for this plugin)"),
    (dsv41_activation, "set_engram_edp_hooks",
     "Engram/EDP guard injection (adapter seam)"),
    (dsv41_activation, "set_dspark_graph_hooks",
     "DSpark graph gate injection (adapter seam)"),
    (dsv41_dspark_registration, "set_plugin_reference",
     "draft model class hand-off (adapter seam)"),
    (ct_w4a8_registration, "set_config_provider",
     "CT W4A8 config provider injection (adapter seam)"),
)

_bound: set[str] = set()


class AdapterContractError(RuntimeError):
    """The installed adapter does not provide a seam this plugin revision needs."""


def require_adapter_seams() -> None:
    """Refuse a mixed revision pair instead of failing on a bare AttributeError.

    Called before any binding so the refusal names the seam, not the call site.
    """
    missing = [
        f"{module.__name__}.{name} -- {purpose}"
        for module, name, purpose in _REQUIRED_SEAMS
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise AdapterContractError(
            "hcu_vllm_das plugin requires a newer hcu_adapter; missing: "
            + "; ".join(missing)
        )


def _moe_provider():
    from .quantization.channel_fp8_moe import HcuDSV41ChannelFP8MoEMethod

    return HcuDSV41ChannelFP8MoEMethod


def _ct_w4a8_config_provider():
    from .quantization.ct_w4a8 import HcuDSV41CTW4A8Config

    return HcuDSV41CTW4A8Config


def _engram_edp_install():
    from .v41_engram_edp import install_engram_edp_guard

    return install_engram_edp_guard()


def _engram_edp_check():
    from .v41_engram_edp import check_engram_edp_guard

    return check_engram_edp_guard()


def _dspark_graph_install():
    from .dspark_graph import install_dspark_graph_support

    return install_dspark_graph_support()


def _dspark_graph_check():
    from .dspark_graph import check_dspark_graph_support

    return check_dspark_graph_support()


def bind_dsv41() -> None:
    """Bind the plugin providers the DSV4.1 channel-FP8 admission consumes."""
    if "dsv41" in _bound:
        return
    require_adapter_seams()
    dsv41_channel_fp8.set_moe_provider(_moe_provider)
    dsv41_activation.set_engram_edp_hooks(_engram_edp_install, _engram_edp_check)
    dsv41_activation.set_dspark_graph_hooks(_dspark_graph_install, _dspark_graph_check)
    dsv41_dspark_registration.set_plugin_reference((
        "hcu_vllm_das.models.dsv41_dspark", "HcuDSparkDeepseekV41ForCausalLM",
    ))
    _bound.add("dsv41")


def bind_ct_w4a8() -> None:
    """Bind the plugin CT W4A8 config class the adapter registration consumes."""
    if "ct_w4a8" in _bound:
        return
    require_adapter_seams()
    ct_w4a8_registration.set_config_provider(_ct_w4a8_config_provider)
    _bound.add("ct_w4a8")
