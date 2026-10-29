# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM #61023 workspace policy, adapted to the v0.25.1 HCU sidecar.

Upstream: d6aca837c3c6b773a557448d526db219a5a5718b.
Do not add fields to the installed vLLM dataclass/Pydantic schema.
"""
from __future__ import annotations

import functools
from types import FunctionType

from vllm_hcu.patch.config import get_hcu_config, set_hcu_config


def resolve_workspace_factor(config):
    model = getattr(config, "model_config", None)
    if model is None:
        return None
    factor = get_hcu_config(config).sparse_mla_prefill_workspace_factor
    if factor is None and model is not None:
        factor = 5 if model.max_model_len <= 256 * 1024 else 2
        set_hcu_config(config, sparse_mla_prefill_workspace_factor=factor)
    return factor


def workspace_rows(config):
    factor = get_hcu_config(config).sparse_mla_prefill_workspace_factor
    if factor is None:
        raise ValueError("sparse MLA workspace factor must be resolved before allocation")
    return config.model_config.max_model_len * factor


def indexer_rows(config):
    return workspace_rows(config) * (576 * 2 // 132)


def bind_workspace_helper(function, config):
    """Rebind only the audited helper, leaving shared module globals intact."""
    if not isinstance(function, FunctionType):
        raise TypeError("sparse MLA workspace adapter requires a Python function")
    factor = get_hcu_config(config).sparse_mla_prefill_workspace_factor
    if factor is None:
        raise ValueError("sparse MLA workspace factor was not resolved at config init")
    namespace = dict(function.__globals__)
    namespace['get_prefill_workspace_size'] = lambda max_model_len: max_model_len * factor
    bound = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def wrap_workspace_builder(function, cache_attribute):
    if "get_prefill_workspace_size" not in function.__code__.co_names:
        return function
    @functools.wraps(function)
    def build(self, common_attn_metadata):
        bound = getattr(self, cache_attribute, None)
        if bound is None:
            bound = bind_workspace_helper(function, self.vllm_config)
            setattr(self, cache_attribute, bound)
        return bound(self, common_attn_metadata)
    return build


def install_workspace_adapters(module):
    """Called after the existing FlashMLA patch has validated its targets."""
    if getattr(module, '_vllm_hcu_workspace_factor_applied', False):
        return
    impl = module.FlashMLASparseImpl
    init = impl.__init__
    get_config = module.get_current_vllm_config

    @functools.wraps(init)
    def initialize(self, *args, **kwargs):
        config = get_config()
        return bind_workspace_helper(init, config)(self, *args, **kwargs)

    separate = module.FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode
    impl.__init__ = initialize
    module.FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode = (
        wrap_workspace_builder(separate, '_hcu_workspace_separate_build'))
    module._vllm_hcu_workspace_factor_applied = True
