# SPDX-License-Identifier: Apache-2.0
"""Plugin-side bindings for the adapter-owned DSV4.1 admission code.

The adapter owns the public quantization registration, but three hooks in it are
plugin-private mechanism: the Channel-FP8 MoE method, the Engram/EDP guard and
the DSpark Graph gate, plus the CT W4A8 config class and the plugin draft model
class path. They are injected here as lazy providers so the plugin import stays on
demand and the adapter never imports or names the plugin.
"""
from __future__ import annotations

# A mixed revision pair can be missing a whole module, not just a function: the
# adapter module that owns a seam may not exist yet in the installed revision.
# Importing eagerly at module scope turns that into a `ModuleNotFoundError` from
# entry-point discovery, which is exactly the opaque failure this check exists
# to replace -- and it happens before `require_adapter_seams()` can run. Import
# lazily and report the missing module as a contract violation instead.
_SEAM_MODULES = (
    "ct_w4a8_registration",
    "dsv41_activation",
    "dsv41_channel_fp8",
    "dsv41_dspark_registration",
)


class AdapterContractError(RuntimeError):
    """The installed adapter does not provide a seam this plugin revision needs."""


def _seam_modules():
    """Import the adapter modules the seam table covers, naming any that is absent."""
    from importlib import import_module

    import hcu_adapter

    modules, missing = {}, []
    for name in _SEAM_MODULES:
        try:
            modules[name] = import_module("hcu_adapter." + name)
        except ModuleNotFoundError as error:
            # Only treat a missing *adapter* module as a revision mismatch; a
            # missing third-party dependency inside it is a different problem
            # and must not be reported as "upgrade the adapter".
            if error.name not in (None, "hcu_adapter." + name):
                raise
            missing.append("hcu_adapter." + name)
    if missing:
        raise AdapterContractError(
            "hcu_vllm_das plugin requires a newer hcu_adapter; missing modules: "
            + "; ".join(missing)
        )
    return modules


def _required_seams(modules):
    """The seam table, resolved against the modules that actually imported."""
    return (
        (modules["dsv41_channel_fp8"], "set_moe_provider",
         "Channel-FP8 MoE provider injection (adapter seam, added for this plugin)"),
        (modules["dsv41_activation"], "set_provider_binder",
         "DSV4.1 activation binder (adapter seam, added for this plugin)"),
        (modules["dsv41_activation"], "set_engram_edp_hooks",
         "Engram/EDP guard injection (adapter seam)"),
        (modules["dsv41_activation"], "set_dspark_graph_hooks",
         "DSpark graph gate injection (adapter seam)"),
        (modules["dsv41_dspark_registration"], "set_plugin_reference",
         "draft model class hand-off (adapter seam)"),
        (modules["ct_w4a8_registration"], "set_config_provider",
         "CT W4A8 config provider injection (adapter seam)"),
        # `ct_w4a8_activation.py` calls this on *its own* adapter module at
        # import time, before `bind_ct_w4a8()` runs. Naming the DSV4.1 module
        # here would check the wrong object: the two modules each own a
        # `set_provider_binder`, and only this one is on the CT path.
        (modules["ct_w4a8_registration"], "set_provider_binder",
         "CT W4A8 activation binder (adapter seam, called at import time)"),
    )


_bound: set[str] = set()


def require_adapter_seams() -> None:
    """Refuse a mixed revision pair instead of failing on a bare AttributeError.

    Called before any binding so the refusal names the seam, not the call site.
    """
    modules = _seam_modules()
    missing = [
        f"{module.__name__}.{name} -- {purpose}"
        for module, name, purpose in _required_seams(modules)
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise AdapterContractError(
            "hcu_vllm_das plugin requires a newer hcu_adapter; missing: "
            + "; ".join(missing)
        )


def _bindings():
    """The adapter modules, imported lazily so a missing one is a named refusal."""
    return _seam_modules()


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
    modules = _bindings()
    require_adapter_seams()
    modules["dsv41_channel_fp8"].set_moe_provider(_moe_provider)
    modules["dsv41_activation"].set_engram_edp_hooks(
        _engram_edp_install, _engram_edp_check)
    modules["dsv41_activation"].set_dspark_graph_hooks(
        _dspark_graph_install, _dspark_graph_check)
    modules["dsv41_dspark_registration"].set_plugin_reference((
        "hcu_vllm_das.models.dsv41_dspark", "HcuDSparkDeepseekV41ForCausalLM",
    ))
    _bound.add("dsv41")


def bind_ct_w4a8() -> None:
    """Bind the plugin CT W4A8 config class the adapter registration consumes."""
    if "ct_w4a8" in _bound:
        return
    modules = _bindings()
    require_adapter_seams()
    modules["ct_w4a8_registration"].set_config_provider(_ct_w4a8_config_provider)
    _bound.add("ct_w4a8")
