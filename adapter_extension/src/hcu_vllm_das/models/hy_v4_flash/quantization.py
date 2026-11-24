# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local Flash quantization admission; no registration or config mutation.

The serialized-recipe validators deliberately have no Torch/Core dependency so
model and DFly admission can share them without importing execution kernels.
Kernel availability is a separate, real Core/Adapter construction-time check.
"""


def _matches(actual, expected):
    return isinstance(actual, dict) and all(
        type(actual.get(key)) is type(value) and actual.get(key) == value
        for key, value in expected.items()
    )


def _groups(config, error):
    if (
        not isinstance(config, dict)
        or any(
            config.get(key) is not None
            for key in ("kv_cache_scheme", "transform_config")
        )
        or config.get("sparsity_config") not in (None, {})
    ):
        raise ValueError(error)
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(error)
    ignore = config.get("ignore", [])
    if not isinstance(ignore, list) or any(
        not isinstance(t, str) or not t for t in ignore
    ):
        raise ValueError(error)
    for group in groups.values():
        if not isinstance(group, dict) or (
            not isinstance(group.get("targets"), list)
            or not group["targets"]
            or any(not isinstance(t, str) or not t for t in group["targets"])
            or group.get("output_activations") is not None
        ):
            raise ValueError(error)
    return groups.values()


def _plain_args(args, *, channel=False):
    return (
        isinstance(args, dict)
        and args.get("observer_kwargs") in (None, {})
        and all(args.get(k) is None for k in ("actorder", "scale_dtype", "zp_dtype"))
        and (
            not channel
            or (
                args.get("group_size") in (None, -1)
                and args.get("block_structure") is None
            )
        )
    )


def validate_fp8_config(config: object) -> None:
    """Admit serialized native block-FP8 or CT block/channel W8A8 FP8."""
    error = (
        "HYV4 Flash requires serialized W8A8 FP8: 128x128 blocks with dynamic "
        "group-128 activations, or CT channel weights with dynamic token "
        "activations; KV-cache schemes/transforms are not admitted"
    )
    if not isinstance(config, dict) or any(
        config.get(k) is not None for k in ("kv_cache_scheme", "transform_config")
    ):
        raise ValueError(error)
    if config.get("quant_method") == "fp8":
        if not _matches(
            config, dict(activation_scheme="dynamic", weight_block_size=[128, 128])
        ) or any(
            config.get(k) not in choices
            for k, choices in (
                ("fmt", (None, "e4m3")),
                ("store_dtype", (None, "fp8")),
                ("expert_dtype", (None, "fp8")),
            )
        ):
            raise ValueError(error)
        return
    if (
        config.get("quant_method") != "compressed-tensors"
        or config.get("format") != "float-quantized"
        or config.get("quantization_status") not in (None, "compressed")
    ):
        raise ValueError(error)
    for group in _groups(config, error):
        w, a = group.get("weights"), group.get("input_activations")
        common_w = dict(num_bits=8, type="float", symmetric=True, dynamic=False)
        common_a = dict(num_bits=8, type="float", symmetric=True, dynamic=True)
        channel = (
            _matches(w, dict(common_w, strategy="channel"))
            and _matches(a, dict(common_a, strategy="token"))
            and _plain_args(w, channel=True)
            and _plain_args(a, channel=True)
        )
        block = (
            _matches(w, dict(common_w, strategy="block", block_structure=[128, 128]))
            and _matches(a, dict(common_a, strategy="group", group_size=128))
            and _plain_args(w)
            and _plain_args(a)
            and w.get("group_size") in (None, -1)
            and a.get("block_structure") is None
        )
        if not (channel or block) or group.get("format") not in (
            None,
            "float-quantized",
        ):
            raise ValueError(error)


def validate_channel_w4a8_config(config: object) -> None:
    """Admit channel INT4 experts + channel INT8 linears, dynamic token INT8.

    Both storage groups must exist. Actual projection/target selection is checked
    by the model-local CT config at real RoutedExperts construction, not inferred
    from regex text here. No activation metadata is removed or changed to W4A16.
    """
    error = (
        "HYV4 Flash W4A8 requires symmetric channel INT4/INT8 weights, dynamic "
        "token INT8 activations, and compressed packed CT metadata"
    )
    if not isinstance(config, dict) or (
        config.get("quant_method") != "compressed-tensors"
        or config.get("format") != "pack-quantized"
        or config.get("quantization_status") != "compressed"
    ):
        raise ValueError(error)
    bits_seen = set()
    for group in _groups(config, error):
        w, a = group.get("weights"), group.get("input_activations")
        bits = w.get("num_bits") if isinstance(w, dict) else None
        if type(bits) is not int or bits not in (4, 8):
            raise ValueError(error)
        if (
            not _matches(
                w,
                dict(
                    num_bits=bits,
                    type="int",
                    strategy="channel",
                    symmetric=True,
                    dynamic=False,
                ),
            )
            or not _matches(
                a,
                dict(
                    num_bits=8,
                    type="int",
                    strategy="token",
                    symmetric=True,
                    dynamic=True,
                ),
            )
            or not _plain_args(w, channel=True)
            or not _plain_args(a, channel=True)
            or group.get("format")
            != ("pack-quantized" if bits == 4 else "int-quantized")
        ):
            raise ValueError(error)
        bits_seen.add(bits)
    if bits_seen != {4, 8}:
        raise ValueError(error)


def validate_flash_quant_config(config: object) -> str:
    """Return ``fp8`` or ``channel_w4a8`` without mutating serialized input."""
    if isinstance(config, dict) and config.get("format") == "pack-quantized":
        validate_channel_w4a8_config(config)
        return "channel_w4a8"
    validate_fp8_config(config)
    return "fp8"


# Preserve the validator spellings used by the selective legacy references.
_validate_fp8_config = validate_fp8_config
_validate_channel_w4a8_config = validate_channel_w4a8_config


def validate_flash_quant_parallelism(vllm_config, *, enable_eplb=False):
    """Admission only: ordinary TP; not an assertion of distributed acceptance."""
    p = vllm_config.parallel_config
    if (
        p.enable_expert_parallel
        or p.enable_elastic_ep
        or p.data_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.pipeline_parallel_size != 1
        or p.use_sequence_parallel_moe
        or p.enable_eplb
        or enable_eplb
        or p.eplb_config.num_redundant_experts != 0
        or p.enable_dbo
        or vllm_config.lora_config is not None
    ):
        raise NotImplementedError(
            "Flash quantized MoE requires ordinary TP (no EP/DP/PCP/DCP/PP/SP/EPLB/DBO/LoRA)"
        )


def prepare_flash_quant_config(config, quant_config, *, for_moe: bool = False):
    """Compose real Core CT and model-local W4A8; never register a quant method.

    Dense/DFly linears retain Core CT. CT FP8 retains Core's own weight loader,
    backend selection and apply; unsupported platform kernels fail at that real
    boundary. W4A8 is explicit opt-in eager HCU only, not a native Core claim.
    """
    backend = getattr(config, "flash_w4a8_backend", "disabled")
    if backend not in ("disabled", "adapter_reference"):
        raise ValueError("flash_w4a8_backend must be disabled or adapter_reference")
    if quant_config is None:
        if backend != "disabled":
            raise ValueError("adapter_reference requires serialized channel W4A8 CT")
        return None
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    validate_flash_quant_parallelism(get_current_vllm_config())
    if isinstance(quant_config, Fp8Config):
        if not quant_config.is_checkpoint_fp8_serialized:
            raise ValueError("Flash FP8 requires a serialized checkpoint")
        validate_fp8_config(
            dict(
                quant_method="fp8",
                activation_scheme=quant_config.activation_scheme,
                weight_block_size=quant_config.weight_block_size,
                store_dtype=quant_config.store_dtype,
            )
        )
        family = "fp8"
    elif isinstance(quant_config, CompressedTensorsConfig):
        if (
            quant_config.kv_cache_scheme is not None
            or quant_config.transform_config is not None
        ):
            raise ValueError("Flash does not admit CT KV-cache schemes/transforms")
        family = validate_flash_quant_config(quant_config.config)
        if (
            quant_config.quant_format != quant_config.config["format"]
            or quant_config.online_quantization_config is not None
        ):
            raise ValueError(
                "Flash requires consistent serialized CT format and no online quantization"
            )
    else:
        raise NotImplementedError(
            "Flash requires real Core FP8 or compressed-tensors config"
        )
    if family == "channel_w4a8":
        from .runtime import is_hcu_platform

        if backend != "adapter_reference" or not is_hcu_platform():
            raise ValueError(
                "Channel W4A8 requires HCU and flash_w4a8_backend=adapter_reference"
            )
        if for_moe:
            from .w4a8 import HYV4FlashW4A8Config

            return HYV4FlashW4A8Config.for_moe(quant_config)
    elif backend != "disabled":
        raise ValueError("adapter_reference requires serialized channel W4A8 CT")
    return quant_config
