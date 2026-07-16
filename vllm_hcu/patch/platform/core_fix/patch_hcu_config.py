# SPDX-License-Identifier: Apache-2.0
"""Platform pre-registration coordinator for HCU feature configuration."""

from __future__ import annotations

import argparse

from ._common import PatchCompatibilityError

_HCU_BOOLEAN_OPTIONS = {
    "enable_lightly_cp": "Enable HCU lightly context parallelism.",
    "enable_lightly_cplb": "Enable HCU lightly-CP load balancing.",
    "enable_custom_sp": "Enable HCU custom runtime sequence parallelism.",
}
_DPSK_BACKEND = "dpsk_deep_gemm"


def _actions_by_dest(parser: object) -> dict[str, argparse.Action]:
    actions = getattr(parser, "_actions", None)
    if not isinstance(actions, list):
        raise PatchCompatibilityError(
            "HCU platform CLI registration requires an argparse-compatible parser"
        )
    return {
        action.dest: action
        for action in actions
        if isinstance(action, argparse.Action)
    }


def register_hcu_cli_args(parser: object) -> None:
    """Add legacy CLI switches once and extend the existing MoE choice."""

    by_dest = _actions_by_dest(parser)
    add_group = getattr(parser, "add_argument_group", None)
    if not callable(add_group):
        raise PatchCompatibilityError(
            "HCU platform CLI parser does not expose add_argument_group"
        )

    # Validate the expected upstream action before mutating the parser, so a
    # platform-hook ordering error cannot leave half-registered HCU options.
    moe_action = by_dest.get("moe_backend")
    if moe_action is None:
        raise PatchCompatibilityError(
            "HCU platform expected the upstream --moe-backend action to exist"
        )
    choices = moe_action.choices
    if choices is None:
        raise PatchCompatibilityError(
            "upstream --moe-backend action does not expose finite choices"
        )

    for dest in _HCU_BOOLEAN_OPTIONS:
        option = "--" + dest.replace("_", "-")
        existing = by_dest.get(dest)
        if existing is not None:
            if (
                option not in existing.option_strings
                or existing.nargs != 0
                or existing.const is not True
            ):
                raise PatchCompatibilityError(
                    f"CLI destination {dest!r} is already registered with "
                    "incompatible semantics"
                )
            # An omitted store_true option must not manufacture False and
            # overwrite a True value supplied via --additional-config.
            existing.default = argparse.SUPPRESS

    group = None
    for dest, help_text in _HCU_BOOLEAN_OPTIONS.items():
        if dest in by_dest:
            continue
        option = "--" + dest.replace("_", "-")
        if group is None:
            group = add_group(
                title="HCUFeatureConfig",
                description=(
                    "HCU-only options stored in VllmConfig.additional_config['hcu']."
                ),
            )
        group.add_argument(
            option,
            dest=dest,
            action="store_true",
            default=argparse.SUPPRESS,
            help=help_text,
        )

    # EngineArgs.add_cli_args owns this action.  The platform hook executes
    # afterwards, so it may extend choices without changing KernelConfig's
    # upstream Literal/Pydantic schema.
    if _DPSK_BACKEND not in choices:
        moe_action.choices = [*choices, _DPSK_BACKEND]


def pre_register_and_update(parser: object | None = None) -> None:
    """Use vLLM's public platform hook to arm config/runtime adapters."""

    # Imports are local so loading HCUPlatform itself does not eagerly import
    # model/quantization modules or custom kernels.
    from . import (
        patch_compilation_config,
        patch_engine_args,
        patch_slimquant_registry,
        patch_vllm_config,
    )

    patch_engine_args.apply()
    patch_compilation_config.apply()
    patch_vllm_config.apply()
    patch_slimquant_registry.apply()
    if parser is not None:
        register_hcu_cli_args(parser)


__all__ = ["pre_register_and_update", "register_hcu_cli_args"]
