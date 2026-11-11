# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route Kimi-K3.1 DSA full-attention layers through the HCU sparse MLA path.

Three exact-name post-import hooks on ``vllm.models.kimi_k3.amd.linear``:

* ``KimiLinearModel.__init__`` -- for ``model_type == kimi_k31`` checkpoints
  with ``text_config.dsa_config``, create the model-level shared
  ``topk_indices_buffer`` and bind a construction-scope context so every DSA
  layer receives the SAME buffer (DeepSeek V3.2/HY-V4 contract, per DP worker).
* ``KimiMLAAttention.__init__`` -- for K3.1 decoder layers whose layer index
  is listed in ``dsa_layers`` (1-based in the checkpoint config), construct the
  sparse ``KimiK31DSAAttention`` state into ``self`` INSTEAD of the dense MLA
  path.  Dense K3/Kimi-Linear layers (and every other model) keep the stock
  construction untouched.
* ``KimiMoE.__init__`` -- K3.1 W4A8 checkpoints store the latent routed-expert
  projections quantized, while stock Kimi-K3 builds them UNquantized and its
  mixed-precision patch injects a quant config only for ``w8a8_include``
  checkpoints.  For K3.1 the layer's SlimQuant config is transplanted onto
  exactly ``routed_expert_{up,down}_proj`` during the stock constructor.

This adapter must be registered AFTER ``patch_kimi_k3_mla_gate`` so the
outermost ``KimiMLAAttention.__init__`` wrapper short-circuits before the
merged-gate rebind can touch a DSA layer.  The merged gate is additionally
unreachable on the W4A8 deployment because its merge requires
``UnquantizedLinearMethod`` projections.
"""

from __future__ import annotations

import functools
import re
from contextvars import ContextVar
from inspect import signature
from types import ModuleType
from typing import Any

from vllm.config import get_current_vllm_config_or_none

from ._common import load_exact_module

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k31.dsa_layers"
TARGETS = (
    f"{TARGET_MODULE}.KimiLinearModel.__init__",
    f"{TARGET_MODULE}.KimiMLAAttention.__init__",
    f"{TARGET_MODULE}.KimiMoE.__init__",
)
_MARKER = "_vllm_hcu_kimi_k31_dsa_applied"

_LAYER_PREFIX_RE = re.compile(r"layers\.(\d+)\.self_attn$")


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False

    from vllm_hcu.models.kimi_k31 import (
        K31_DSA_CTX,
        KimiK31DSAAttention,
        KimiK31DsaContext,
    )

    original_model_init = target.KimiLinearModel.__init__
    original_attn_init = target.KimiMLAAttention.__init__
    attn_signature = signature(original_attn_init)
    model_signature = signature(original_model_init)
    original_moe_init = target.KimiMoE.__init__
    moe_signature = signature(original_moe_init)
    _EXPERT_SCOPE: ContextVar = ContextVar(
        "vllm_hcu_k31_expert_quant", default=None
    )

    class _K31QuantReplicated(target.ReplicatedLinear):
        # Latent-MoE routed expert projections are constructed WITHOUT a
        # quant config on stock Kimi-K3; its mixed W4A8/W8A8 checkpoints
        # inject one afterwards through patch_kimi_k3_mixed_linear.  K3.1
        # ships W4A8 expert weights outright, so transplant the layer's
        # SlimQuant config onto exactly these two modules while the stock
        # constructor runs.
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            scope = _EXPERT_SCOPE.get()
            if scope is not None and kwargs.get("quant_config") is None:
                quant_cfg, parent = scope
                if kwargs.get("prefix") in (
                    f"{parent}.routed_expert_down_proj",
                    f"{parent}.routed_expert_up_proj",
                ):
                    kwargs["quant_config"] = quant_cfg
            super().__init__(*args, **kwargs)

    @functools.wraps(original_moe_init)
    def moe_init(self: Any, *args: Any, **kwargs: Any) -> None:
        ctx = K31_DSA_CTX.get()
        quant_cfg = None
        prefix: Any = None
        if ctx is not None:
            try:
                bound = moe_signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                quant_cfg = bound.arguments.get("quant_config")
                prefix = bound.arguments.get("prefix")
            except TypeError:
                quant_cfg = None
        get_name = getattr(quant_cfg, "get_name", None)
        if (
            ctx is None
            or quant_cfg is None
            or get_name is None
            or get_name() != "slimquant_w4a8"
            or getattr(quant_cfg, "w8a8_include", "missing") is not None
        ):
            original_moe_init(self, *args, **kwargs)
            return
        token = _EXPERT_SCOPE.set((quant_cfg, str(prefix)))
        prev_cls = target.ReplicatedLinear
        target.ReplicatedLinear = _K31QuantReplicated
        try:
            original_moe_init(self, *args, **kwargs)
        finally:
            target.ReplicatedLinear = prev_cls
            _EXPERT_SCOPE.reset(token)

    @functools.wraps(original_model_init)
    def model_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Argument-driven: KimiLinearModel is constructed with (at least)
        # vllm_config; prefer that over the ambient thread-local config,
        # which is NOT guaranteed to be set inside worker model builds.
        vllm_config = None
        try:
            bound = model_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            vllm_config = bound.arguments.get("vllm_config")
        except TypeError:
            pass
        if vllm_config is None:
            vllm_config = get_current_vllm_config_or_none()
        text_config = getattr(
            getattr(vllm_config, "model_config", None),
            "hf_text_config",
            None,
        )
        if vllm_config is None or text_config is None:
            original_model_init(self, *args, **kwargs)
            return
        from vllm_hcu.models.kimi_k31 import kimi_k31_dsa_layers

        if kimi_k31_dsa_layers(text_config) is None:
            original_model_init(self, *args, **kwargs)
            return
        with KimiK31DsaContext(
            vllm_config=vllm_config,
            cache_config=getattr(vllm_config, "cache_config", None),
        ) as _ctx:
            original_model_init(self, *args, **kwargs)

    @functools.wraps(original_attn_init)
    def attn_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Argument-driven DSA detection: the stock KimiMLAAttention ctor
        # receives the TEXT config itself, which carries ``dsa_config`` on
        # K3.1 checkpoints — no ambient engine config needed.  Surplus
        # caller kwargs (layer_idx, model_config, ...) land in the stock
        # ``**kwargs`` catch-all, so flatten them back before lookup.
        arguments: dict[str, Any] | None = None
        try:
            bound = attn_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            extra = arguments.pop("kwargs", None) or {}
            arguments.update(extra)
        except TypeError:
            arguments = None
        if arguments is not None:
            config = arguments.get("config")
            prefix = arguments.get("prefix")
            if config is not None and prefix is not None:
                from vllm_hcu.models.kimi_k31 import kimi_k31_dsa_layers

                dsa_layers = kimi_k31_dsa_layers(config)
                layer_idx = arguments.get("layer_idx")
                is_dsa = (
                    layer_idx in dsa_layers
                    if dsa_layers and isinstance(layer_idx, int)
                    else (
                        (_m := _LAYER_PREFIX_RE.search(str(prefix)))
                        is not None
                        and dsa_layers is not None
                        and int(_m.group(1)) in dsa_layers
                    )
                )
                if is_dsa:
                    ctx = K31_DSA_CTX.get()
                    if not ctx or ctx.get("buffer") is None:
                        raise RuntimeError(
                            f"{prefix}: K3.1 DSA layer constructed outside "
                            "the KimiLinearModel DSA context"
                        )
                    # Populate the sparse projection/indexer state DIRECTLY
                    # onto this instance.  The stock KimiMLAAttention.forward
                    # body (``output[:] = self.mla_attn(positions,
                    # hidden_states)``) is byte-identical to the DSA forward,
                    # so the stock class keeps dispatching the layer; the
                    # static forward context registers this layer name once.
                    kwargs.pop("self", None)
                    KimiK31DSAAttention.__init__(
                        self,
                        config=config,
                        hidden_size=arguments["hidden_size"],
                        num_heads=arguments["num_heads"],
                        qk_nope_head_dim=arguments["qk_nope_head_dim"],
                        qk_rope_head_dim=arguments["qk_rope_head_dim"],
                        v_head_dim=arguments["v_head_dim"],
                        q_lora_rank=arguments["q_lora_rank"],
                        kv_lora_rank=arguments["kv_lora_rank"],
                        cache_config=arguments.get("cache_config"),
                        quant_config=arguments.get("quant_config"),
                        model_config=arguments.get("model_config"),
                        prefix=str(prefix),
                        use_nope=arguments.get("use_nope", True),
                        vllm_config=ctx.get("vllm_config"),
                    )
                    self._vllm_hcu_kimi_k31_dsa = True
                    return
        original_attn_init(self, *args, **kwargs)

    target.KimiLinearModel.__init__ = model_init
    target.KimiMLAAttention.__init__ = attn_init
    target.KimiMoE.__init__ = moe_init
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
