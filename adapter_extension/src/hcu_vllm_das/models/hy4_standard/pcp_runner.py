# SPDX-License-Identifier: Apache-2.0
"""Owned PCP/VL target plumbing through the existing Adapter runner factory."""

import os

from hcu_adapter.features.hy4 import pcp_profile
from hcu_adapter.infra.kill_switch import op_disabled
from vllm.v1.worker.gpu import model_runner as native
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from ...environment import custom_ops_enabled
from ...hy4_constructor import bind_constructor
from .pcp_manager import HYV4PCPProducerManager
from .pcp_model_state import HYV4PCPProducerModelState
from .pd_policy import classify_standard_hy4_role, is_standard_vl_target

_policy = None
_failure = None
_STOCK_INITIALIZE = GPUModelRunner.initialize_kv_cache
_STOCK_ATTN_INIT = native.init_attn_backend
_PROFILE_BASE = pcp_profile.runner_class()


def _profile_selected():
    return custom_ops_enabled() and not op_disabled(pcp_profile.OP)


def prepare():
    global _policy
    if _policy is None:
        _policy = (os.getpid(), _profile_selected())
    check_health()


def check_health():
    global _failure
    if _failure is not None:
        raise RuntimeError(_failure)
    if _policy is None:
        raise RuntimeError("HY4 PCP producer runner publication was not prepared")
    if (
        _policy != (os.getpid(), _profile_selected())
        or native.init_attn_backend is not _STOCK_ATTN_INIT
        or GPUModelRunner.initialize_kv_cache is not _STOCK_INITIALIZE
        or _PROFILE_BASE.initialize_kv_cache is not _STOCK_INITIALIZE
    ):
        _failure = "HY4 PCP runner process/profile policy/native owner changed; restart"
        raise RuntimeError(_failure)
    if _policy[1]:
        pcp_profile.require_healthy()


def target_attention_names(kv_cache_config, draft_layer_names):
    names = {
        name for group in kv_cache_config.kv_cache_groups for name in group.layer_names
    }
    selected = names - set(draft_layer_names)
    if not selected:
        raise ValueError("HY4 PCP target has no real attention layers")
    return selected


class _ProducerRunner:
    @property
    def pcp_manager_cls(self):
        return HYV4PCPProducerManager

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if not isinstance(self.model_state, HYV4PCPProducerModelState):
            raise TypeError("HY4 PCP producer requires its owned VL ModelState hook")
        self.model_state.configure_pcp(self.pcp_manager)
        return result

    def initialize_kv_cache(self, *args, **kwargs):
        check_health()

        def target_init(
            cache_config,
            config,
            device,
            active_layer_names=None,
            draft_layer_names=None,
        ):
            actual = set(getattr(self.speculator, "draft_attn_layer_names", ()))
            if draft_layer_names is not None and set(draft_layer_names) != actual:
                raise RuntimeError(
                    "HY4 PCP draft-layer owner differs at KV initialization"
                )
            active = target_attention_names(cache_config, actual)
            if active_layer_names is not None:
                active.intersection_update(active_layer_names)
            return _STOCK_ATTN_INIT(
                cache_config,
                config,
                device,
                active_layer_names=active,
                draft_layer_names=draft_layer_names,
            )

        initialize = bind_constructor(
            _STOCK_INITIALIZE, {"init_attn_backend": target_init}
        )
        # Original allocation, alias discovery and speculator.set_attn remain.
        return initialize(self, *args, **kwargs)


class HYV4PCPProducerRunner(_ProducerRunner, GPUModelRunner):
    """Stock profiling when the master/profile gate is disabled."""


class HYV4PCPProducerProfileRunner(_ProducerRunner, _PROFILE_BASE):
    """Retain Adapter's local profile bound only when explicitly enabled."""


def matches(config):
    check_health()
    if not is_standard_vl_target(config.model_config.hf_config):
        return False
    return classify_standard_hy4_role(config) == "producer"


def create(config, device):
    check_health()
    if not matches(config):
        raise ValueError("HY4 PCP runner factory requires its producer role")
    if pcp_profile.matches(config):
        raise RuntimeError("HY4 PCP producer and Adapter profile factories overlap")
    cls = HYV4PCPProducerProfileRunner if _policy[1] else HYV4PCPProducerRunner
    return cls(config, device)
