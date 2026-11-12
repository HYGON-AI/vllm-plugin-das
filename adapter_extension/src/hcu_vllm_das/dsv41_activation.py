# SPDX-License-Identifier: Apache-2.0
"""Opt-in registration for DeepSeek V4.1 PCP + MegaMoE on HCU."""
from hcu_adapter.extensions import ExtensionSpec

from .environment import _flag
from .extension import _require_vllm_033_release_line

QUANTIZATION_METHOD = "hcu_dsv41_channel_fp8"
ARCH = "DeepseekV41ForCausalLM"
MODEL = ("hcu_vllm_das.models.dsv41_pcp", "HcuDSV41PcpForCausalLM")


def activate_extension(context):
    _require_vllm_033_release_line(context.vllm_version)
    if not _flag("VLLM_HCU_DSV41_PCP_ATTENTION", default=False):
        return
    from vllm.platforms import current_platform
    if not (current_platform.is_rocm() and current_platform.device_name == "hcu"):
        return
    from vllm.model_executor.layers import quantization
    from vllm.model_executor.models import ModelRegistry
    from .quantization.dsv41_channel_fp8 import HcuDSV41ChannelFP8Config

    entry = ModelRegistry.models.get(ARCH)
    owner = (getattr(entry, "module_name", None), getattr(entry, "class_name", None))
    if owner not in (("vllm.models.deepseek_v41", ARCH), MODEL):
        raise RuntimeError("V4.1 PCP model has a foreign registry owner")
    if QUANTIZATION_METHOD in quantization.QUANTIZATION_METHODS:
        if quantization.get_quantization_config(QUANTIZATION_METHOD) is not HcuDSV41ChannelFP8Config:
            raise RuntimeError("V4.1 Channel-FP8 config has a foreign registry owner")
    else:
        quantization.register_quantization_config(QUANTIZATION_METHOD)(HcuDSV41ChannelFP8Config)
    if owner != MODEL:
        ModelRegistry.register_model(ARCH, ":".join(MODEL))


def extension_spec():
    return ExtensionSpec(name="das-hcu-dsv41", api_version=1, priority=116,
        capabilities=frozenset({"dsv41-pcp-megamoe"}), activate=activate_extension)
