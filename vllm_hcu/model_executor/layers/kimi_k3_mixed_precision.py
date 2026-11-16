# SPDX-License-Identifier: Apache-2.0
"""K3 SlimQuant mixed projections, preserving checkpoint branch precision."""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none


def _get_offloader():
    try:
        from vllm.model_executor.offloader.base import get_offloader
        return get_offloader()
    except Exception:  # pragma: no cover - offloading is optional
        return None


def _is_offloaded_parameter(param) -> bool:
    """Whether vLLM's weight offloader owns ``param``.

    The fused QKV copy is a private tensor outside the offloader's storage, so
    building it from offloaded branches would duplicate the weights in
    accelerator memory.
    """
    if getattr(param, "_vllm_is_uva_offloaded", False):
        return True
    offloader = _get_offloader()
    if offloader is None:
        return False
    for module_offloader in getattr(offloader, "module_offloaders", ()):
        for param_offloader in getattr(module_offloader, "_param_offloaders", {}).values():
            try:
                if param_offloader._param is param:
                    return True
            except (AttributeError, RuntimeError):
                continue
    # UVA's CPU fallback (UVA unavailable or disabled) leaves the parameter on
    # CPU between forwards and swaps in a device copy through functional_call.
    if (getattr(offloader, "uva_offloading", None) is False
            and param.data.device.type == "cpu"):
        return True
    return False


def is_mixed_slimquant(config) -> bool:
    return (config is not None
            and config.get_name() == "slimquant_w4a8"
            and getattr(config, "w8a8_include", None) is not None)


class KimiK3MixedKDAProjection(nn.Module):
    """INT8 Q/K/V plus the native replicated-Fa floating gate projection.

    The caller's packed output order and padding are unchanged. Keeping the
    branches separate avoids casting integer checkpoint values into BF16 or
    quantizing the full-rank gate with Q/K/V's scales.
    """

    _hcu_mixed_kda = True

    def __init__(self, input_size, output_sizes, replicated_shard_id, tp_size,
                 *, column_linear, merged_linear, quant_config, prefix, **kwargs):
        super().__init__()
        if replicated_shard_id != 4 or len(output_sizes) not in (6, 7):
            raise ValueError("Mixed K3 KDA requires full-rank Q/K/V/G/Fa/B layout")
        fuse_qkv = os.environ.get("VLLM_HCU_KIMI_KDA_FUSE_QKV", "0") == "1"
        vllm_config = get_current_vllm_config_or_none()
        model_config = getattr(vllm_config, "model_config", None)
        if fuse_qkv and getattr(model_config, "enable_sleep_mode", False):
            raise ValueError(
                "VLLM_HCU_KIMI_KDA_FUSE_QKV=1 is incompatible with sleep mode: "
                "fused QKV storage is allocated outside the weights pool. "
                "Disable fused KDA QKV or sleep mode."
            )
        parent = prefix.removesuffix(".in_proj_qkvgfab")
        for name, size in zip(("q_proj", "k_proj", "v_proj"), output_sizes[:3]):
            layer = column_linear(input_size, size, quant_config=quant_config,
                                  prefix=f"{parent}.{name}", **kwargs)
            if layer.weight.dtype != torch.int8 or not hasattr(layer, "weight_scale"):
                raise ValueError(f"Mixed K3 KDA {parent}.{name} must select W8A8")
            self.add_module(name, layer)
        self.fp_gate = merged_linear(
            input_size, output_sizes[3:], replicated_shard_id=1, tp_size=tp_size,
            quant_config=None, prefix=f"{prefix}.fp_gate", **kwargs)
        # Record ownership after loading, before an offloader's functional_call
        # temporarily replaces CPU parameters with device tensors.
        self._qkv_offloaded = False
        self._fuse_qkv = fuse_qkv
        self._fused_qkv = None
        self._fused_refresh_deferred = False

    @property
    def weight(self):
        # Upstream zeros only the final padding rows through this property.
        return self.fp_gate.weight

    def forward(self, x):
        if self._fuse_qkv:
            if self._fused_qkv is None:
                self._fused_qkv = self._make_fused_qkv(x.shape[-1])
            qkv = self.q_proj.quant_method.apply(self._fused_qkv, x, bias=None)
            values = [qkv, self.fp_gate(x)[0]]
        else:
            values = [getattr(self, name)(x)[0]
                      for name in ("q_proj", "k_proj", "v_proj", "fp_gate")]
        return torch.cat(values, dim=-1), None

    def _make_fused_qkv(self, input_size):
        """Combine post-processed rank-local weights for fused QKV."""
        layers = [self.q_proj, self.k_proj, self.v_proj]
        candidates = (param for layer in layers
                      for param in (layer.weight, layer.weight_scale))
        if getattr(self, "_qkv_offloaded", False) or any(_is_offloaded_parameter(param) for param in candidates):
            raise ValueError(
                "VLLM_HCU_KIMI_KDA_FUSE_QKV is incompatible with weight "
                "offloading: the fused QKV copy is not managed by the "
                "offloader. Disable fused KDA QKV or exclude Q/K/V from "
                "offloading."
            )
        scheme = layers[0].scheme
        if (getattr(scheme, "is_static_input_scheme", True)
                or not getattr(scheme, "input_symmetric", False)
                or str(getattr(scheme, "strategy", "")).lower().split(".")[-1] != "channel"):
            raise ValueError("Fused KDA QKV requires dynamic symmetric channelwise W8A8")
        for layer in layers:
            s = layer.scheme
            if (type(layer.quant_method) is not type(layers[0].quant_method)
                    or type(s) is not type(scheme)
                    or (s.strategy, s.is_static_input_scheme, s.input_symmetric)
                    != (scheme.strategy, scheme.is_static_input_scheme, scheme.input_symmetric)
                    or layer.weight.dtype != torch.int8
                    or layer.weight_scale.dtype != torch.float32
                    or layer.weight_scale.ndim != 2
                    or layer.weight_scale.shape[1] != 1
                    or any(getattr(layer, name, None) is not None for name in
                           ("input_scale", "input_zero_point", "azp_adj", "bias"))):
                raise ValueError("Incompatible KDA QKV quantization or weight layout")
        # Post-load kernels use either [N,K] (HCU) or [K,N] (Triton).
        shape = layers[0].weight.shape
        n = layers[0].weight_scale.shape[0]
        if shape == (n, input_size) and shape != (input_size, n):
            axis = 0
        elif shape == (input_size, n) and shape != (n, input_size):
            axis = 1
        else:
            raise ValueError("Ambiguous or unsupported KDA QKV weight layout")
        if any(layer.weight.shape != shape or layer.weight_scale.shape != (n, 1)
               or layer.weight.device != layers[0].weight.device
               or layer.weight_scale.device != layers[0].weight.device
               for layer in layers):
            raise ValueError("KDA QKV rank-local weights or scales do not match")
        with torch.no_grad():
            return SimpleNamespace(
                weight=torch.cat([layer.weight for layer in layers], dim=axis).contiguous(),
                weight_scale=torch.cat([layer.weight_scale for layer in layers], dim=0),
                input_scale=None, input_zero_point=None, azp_adj=None,
                scheme=scheme, params_dtype=layers[0].params_dtype,
            )

    def refresh_fused_qkv(self):
        """Refresh in place so captured graphs retain valid tensor addresses."""
        if self._fused_qkv is None:
            return
        fused = self._fused_qkv
        layers = [self.q_proj, self.k_proj, self.v_proj]
        axis = 0 if fused.weight.shape[0] == sum(x.weight.shape[0] for x in layers) else 1
        with torch.inference_mode():
            torch.cat([x.weight for x in layers], dim=axis, out=fused.weight)
            torch.cat([x.weight_scale for x in layers], dim=0, out=fused.weight_scale)
        self._fused_refresh_deferred = False

    def load_branch(self, branch, suffix, tensor):
        if branch in ("q_proj", "k_proj", "v_proj"):
            if suffix not in ("weight", "weight_scale"):
                raise ValueError(f"Unsupported K3 W8A8 parameter {branch}.{suffix}")
            if suffix == "weight" and tensor.dtype != torch.int8:
                raise TypeError(f"{branch}.weight must be INT8")
            if suffix == "weight_scale" and tensor.dtype != torch.float32:
                raise TypeError(f"{branch}.weight_scale must be FP32")
            param = getattr(getattr(self, branch), suffix)
            param.weight_loader(param, tensor)
            layers = (self.q_proj, self.k_proj, self.v_proj)
            if any(getattr(layer, name).is_meta for layer in layers
                   for name in ("weight", "weight_scale")):
                self._fused_refresh_deferred = True
            elif not self._fused_refresh_deferred:
                self.refresh_fused_qkv()
            return f"{branch}.{suffix}"
        if suffix != "weight" or not tensor.is_floating_point():
            raise ValueError(f"K3 floating gate requires {branch}.weight")
        shard = {"g_proj": 0, "f_a_proj": 1, "b_proj": 2}[branch]
        param = self.fp_gate.weight
        param.weight_loader(param, tensor, shard)
        return "fp_gate.weight"


def refresh_kimi_kda_fused_qkv(model):
    if model is not None:
        for module in model.modules():
            if isinstance(module, KimiK3MixedKDAProjection):
                module.refresh_fused_qkv()


def record_kimi_kda_offload_ownership(model):
    """Record ownership without retaining the pre-postprocessing parameters."""
    if model is not None:
        for module in model.modules():
            if isinstance(module, KimiK3MixedKDAProjection):
                module._qkv_offloaded = any(
                    _is_offloaded_parameter(param)
                    for layer in (module.q_proj, module.k_proj, module.v_proj)
                    for param in (layer.weight, layer.weight_scale)
                )
