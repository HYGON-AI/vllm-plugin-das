# SPDX-License-Identifier: Apache-2.0
"""Unregistered mixed KDA projection candidate for K3.1 SlimQuant loading."""
from __future__ import annotations
import torch
from torch import nn


class MixedKDAProjection(nn.Module):
    """Preserve Q/K/V INT8 and G/Fa/B BF16 checkpoint branches separately.

    Loader calls use official supplied Column/MergedLinear methods. No model
    monkeypatch or automatic registration is performed by this module.
    """
    def __init__(self, input_size, output_sizes, replicated_shard_id, tp_size,
                 *, column_linear, merged_linear, quant_config, prefix, **kwargs):
        super().__init__()
        if replicated_shard_id != 4 or len(output_sizes) not in (6,7):
            raise ValueError("expected full-rank KDA Q/K/V/G/Fa/B layout")
        if not prefix.endswith(".in_proj_qkvgfab"):
            raise ValueError("unexpected projection prefix")
        parent=prefix.removesuffix(".in_proj_qkvgfab")
        column_kwargs = dict(kwargs)
        # The custom merged constructor accepts this; ColumnParallelLinear does not.
        column_kwargs.pop("padding_size", None)
        for name,size in zip(("q_proj","k_proj","v_proj"),output_sizes[:3]):
            layer=column_linear(input_size,size,quant_config=quant_config,
                                prefix=f"{parent}.{name}", **column_kwargs)
            if layer.weight.dtype != torch.int8 or not hasattr(layer,"weight_scale"):
                raise ValueError(f"{name} must select W8A8")
            self.add_module(name,layer)
        self.fp_gate=merged_linear(input_size,output_sizes[3:],replicated_shard_id=1,
                                   tp_size=tp_size,quant_config=None,
                                   prefix=f"{prefix}.fp_gate",**kwargs)
        if not self.fp_gate.weight.is_floating_point():
            raise ValueError("gate projection must be floating-point")

    @property
    def weight(self):
        # Upstream zeros padding through this property at construction.
        return self.fp_gate.weight

    def load_branch(self, branch, suffix, tensor):
        if branch in ("q_proj","k_proj","v_proj"):
            if suffix not in ("weight","weight_scale"):
                raise ValueError("unexpected INT8 branch parameter")
            if tensor.dtype != (torch.int8 if suffix == "weight" else torch.float32):
                raise TypeError(f"invalid {branch}.{suffix} dtype")
            param=getattr(getattr(self,branch),suffix)
            param.weight_loader(param,tensor)
            return f"{branch}.{suffix}"
        if branch not in ("g_proj","f_a_proj","b_proj") or suffix != "weight" or tensor.dtype != torch.bfloat16:
            raise ValueError("expected BF16 KDA gate/f_a/b weight")
        param=self.fp_gate.weight
        param.weight_loader(param,tensor,{"g_proj":0,"f_a_proj":1,"b_proj":2}[branch])
        return "fp_gate.weight"

    def forward(self, x):
        raise RuntimeError("mixed KDA preload candidate does not support forward")


def load_mixed_kda_weights(model, weights, native_load):
    """Route split KDA checkpoint entries before native model weight loading.

    ``native_load`` is the original bound load_weights method. The incoming
    names must already use the model loader namespace. No globals are patched.
    """
    owners = {name.removesuffix(".in_proj_qkvgfab"): (name, module)
              for name, module in model.named_modules()
              if isinstance(module, MixedKDAProjection)
              and name.endswith(".in_proj_qkvgfab")}
    if not owners:
        raise ValueError("no mixed KDA projections in model")
    loaded = set()
    seen = set()

    def remaining():
        for entry in weights:
            if len(entry) not in (2, 3):
                raise ValueError("unsupported checkpoint entry")
            name, tensor = entry[:2]
            parent, _, suffix = name.rpartition(".")
            attention, _, branch = parent.rpartition(".")
            owner = owners.get(attention)
            if owner is None or branch not in ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"):
                yield entry
                continue
            if name in seen:
                raise ValueError(f"duplicate KDA checkpoint tensor: {name}")
            if len(entry) == 3 and entry[2]:
                raise ValueError(f"unsupported KDA loader metadata: {name}")
            seen.add(name)
            module_name, projection = owner
            param_name = projection.load_branch(branch, suffix, tensor)
            loaded.add(f"{module_name}.{param_name}")
    native_loaded = native_load(remaining())
    if not isinstance(native_loaded, set):
        raise TypeError("native loader must return a loaded parameter set")
    return native_loaded | loaded
