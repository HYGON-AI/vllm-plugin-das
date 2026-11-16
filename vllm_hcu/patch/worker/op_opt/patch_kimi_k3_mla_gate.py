# SPDX-License-Identifier: Apache-2.0
"""Merge Kimi's replicated Q/KV-A and sharded output gate without vLLM edits."""
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from types import ModuleType
from ._common import load_exact_module

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.mla_gate"
TARGETS = (f"{TARGET_MODULE}.KimiMLAAttention.__init__",
           f"{TARGET_MODULE}.KimiLinearModel.__init__",
           f"{TARGET_MODULE}.KimiLinearModel.load_weights")
_MARKER = "_hcu_kimi_merged_mla_gate"
_LOADER_SCOPE = ContextVar("kimi_mla_gate_loader_scope", default=False)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm_hcu.model_executor.layers.kimi_k3_merged_mla_gate import (
        KimiK3MergedQKVGateLinear, enabled,
    )
    from vllm_hcu.model_executor.layers.kimi_k3_mla import HcuKimiK3MergedMLAWrapper
    original_init = target.KimiMLAAttention.__init__
    init_signature = signature(original_init)
    original_load = target.KimiLinearModel.load_weights
    original_model_init = target.KimiLinearModel.__init__

    @wraps(original_model_init)
    def model_init(self, *args, **kwargs):
        # Only this owner has the gate-shard loader below. MTP and DSpark
        # construct attention independently and must keep their native layout.
        token = _LOADER_SCOPE.set(True)
        try:
            original_model_init(self, *args, **kwargs)
        finally:
            _LOADER_SCOPE.reset(token)

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from vllm_hcu.platforms import envs as henvs
        if not (_LOADER_SCOPE.get() and enabled()
                and henvs.optional_custom_op_enabled(True)
                and self.use_output_gate and self.q_lora_rank is not None
                and not getattr(self.mla_attn, "dcp_q_replicate", False)):
            return
        # Merge only BF16/FP16 projections. Quantized layouts have separate
        # scale contracts and must retain their original loader/execution.
        old = self.fused_qkv_a_proj
        gate = self.g_proj
        if (type(old.quant_method).__name__ != "UnquantizedLinearMethod"
                or type(gate.quant_method).__name__ != "UnquantizedLinearMethod"):
            return
        bound = init_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        prefix = bound.arguments['prefix']
        merged = KimiK3MergedQKVGateLinear(
            self.hidden_size, self.q_lora_rank, self.kv_lora_rank,
            self.qk_rope_head_dim, self.num_heads, self.v_head_dim,
            bias=False, quant_config=None, prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.fused_qkv_a_proj = merged
        self.mla_attn.fused_qkv_a_proj = merged
        self.mla_attn.g_proj = None
        del self.g_proj
        self.mla_attn.__class__ = HcuKimiK3MergedMLAWrapper
        self.mla_attn._hcu_fused_output_gate = True
        from vllm.logger import init_logger
        init_logger(__name__).info_once("HCU Kimi fused MLA Q/KV-A + gate enabled")

    @wraps(original_load)
    def load(self, weights):
        # Original Q/KV-A mappings still address shards 0/1. Intercept gate
        # tensors only for modules which actually constructed a merged layer.
        merged_names = {
            name: sub for name, sub in self.named_modules()
            if isinstance(sub, KimiK3MergedQKVGateLinear)
        }
        loaded = set()

        def tensors():
            for entry in weights:
                name, weight = entry[:2]
                if '.g_proj.' in name:
                    mapped = name.replace('.g_proj.', '.fused_qkv_a_proj.')
                    module_name, _, param_name = mapped.rpartition('.')
                    sub = merged_names.get(module_name)
                    if sub is not None:
                        param = getattr(sub, param_name)
                        param.weight_loader(param, weight, 2)
                        loaded.add(mapped)
                        continue
                yield entry
        result = original_load(self, tensors())
        return result | loaded

    target.KimiMLAAttention.__init__ = init
    target.KimiLinearModel.__init__ = model_init
    target.KimiLinearModel.load_weights = load
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
