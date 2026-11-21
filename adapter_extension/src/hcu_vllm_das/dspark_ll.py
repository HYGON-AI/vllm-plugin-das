# SPDX-License-Identifier: Apache-2.0
"""Single native target buffer sized for both W4A8 LL passes.

Keep the native cache entry and destructor. Owned complete-model entries record
one native clean in each graph; unknown entries retain a clean per dispatch.
This transport policy is limited to single-node legacy IPC.
"""
from __future__ import annotations

from functools import lru_cache
import logging
import os
import weakref


_MODEL_ENTRY_OWNERS = weakref.WeakKeyDictionary()


def validate_legacy_ipc(buffer_cls):
    if os.environ.get('ROCSHMEM_BACKEND', 'ipc') != 'ipc':
        raise ValueError('W4A8 shared LL requires ROCSHMEM IPC')
    for name in ('is_ll_layered_opt_enabled', 'is_ll_layered_develop_enabled'):
        select = getattr(buffer_cls, name, None)
        if not callable(select) or select(8):
            raise ValueError('W4A8 shared LL requires native legacy IPC selection')
    for name in ('ROCSHMEM_IPC_MNVL', 'DEEP_EP_NORMAL_MNVL'):
        if os.environ.get(name, '0').strip().lower() not in ('0', 'false', 'off', ''):
            raise ValueError('W4A8 shared LL does not support ' + name)
    if os.environ.get('DEEPEP_TEST_FORCE_ROUTE_MODE', '').strip():
        raise ValueError('W4A8 shared LL does not support forced routes')


def install_model_entry_reset(model, config, num_experts):
    """Record layout initialization on every complete model invocation.

    The clean is a GPU operation captured inside each full model graph. Never
    infer the last GPU layout from Python state: graph replay skips Python.
    Layers within one invocation finish dispatch/combine in the same layout.
    Unsupported model geometry keeps the per-dispatch initialization policy.
    """
    from .dspark_backend import native_draft_ll_requested
    if not native_draft_ll_requested(config):
        return False
    mc = config.model_config
    hf = getattr(mc, "hf_text_config", None)
    if hf is None:
        hf = getattr(mc.hf_config, "text_config", mc.hf_config)
    names = ("hidden_size", "n_routed_experts", "num_experts_per_tok",
             "dspark_n_routed_experts", "dspark_num_experts_per_tok",
             "num_hidden_layers")
    if tuple(getattr(hf, n, None) for n in names) != (5120, 384, 6, 128, 3, 40):
        return False
    if num_experts not in (128, 384):
        raise ValueError("Unsupported model entry IPC expert count")
    from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
    from .dspark_graph import token_capacity
    manager = get_ep_all2all_manager()
    capacity = token_capacity(config).query_rows
    policy = SharedIPCManager(manager, capacity)
    owners = _MODEL_ENTRY_OWNERS.setdefault(manager, {})
    previous = owners.get(num_experts)
    if previous is not None and previous() is not None:
        raise ValueError("Duplicate live model entry IPC reset owner")
    buffer = None
    kwargs = dict(max_num_tokens_per_dp_rank=capacity, token_hidden_size=5120,
                  num_ep_ranks=8, num_global_experts=num_experts,
                  num_local_experts=num_experts // 8)

    def clean_at_entry(_module, _args):
        nonlocal buffer
        if buffer is None:
            buffer = policy.get_handle(kwargs)
            logging.getLogger("vllm." + __name__).info(
                "HCU W4A8 model entry IPC clean executing: experts=%d "
                "capacity=%d buffer_id=%d", num_experts, capacity, id(buffer))
        buffer.clean_low_latency_buffer(capacity, 5120, num_experts,
                                        quant_group_size=0,
                                        num_topk=6 if num_experts == 384 else 3)

    model.register_forward_pre_hook(clean_at_entry)
    owners[num_experts] = weakref.ref(model)
    return True


@lru_cache(maxsize=64)
def _record_layout(experts, capacity, buffer_id):
    # Keep only an integer identity, never another owner of the native buffer.
    logging.getLogger("vllm." + __name__).info(
        "HCU W4A8 shared IPC LL admitted: experts=%d capacity=%d "
        "buffer_id=%d wire=%s", experts, capacity, buffer_id,
        "BF16/native-AGRS-receiver" if experts == 128 else "INT8/native-target")


class SharedIPCManager:
    """Native factory argument; never replaces the EP group's manager."""

    def __init__(self, manager, capacity):
        import deep_ep
        from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager

        if (not isinstance(manager, DeepEPLLAll2AllManager)
                or manager.internode or manager.world_size != 8
                or manager.dp_world_size != 8 or manager.support_fault_tolerance):
            raise ValueError("W4A8 shared LL requires single-node native DP8/EP8")
        validate_legacy_ipc(deep_ep.Buffer)
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('W4A8 shared LL capacity must be a positive integer')
        self.capacity = capacity
        self.native = manager
        self.world_size = manager.world_size
        self.dp_world_size = manager.dp_world_size

    def has_model_entry_reset(self, num_experts):
        owner = _MODEL_ENTRY_OWNERS.get(self.native, {}).get(num_experts)
        return owner is not None and owner() is not None

    def get_handle(self, kwargs):
        expected = dict(max_num_tokens_per_dp_rank=self.capacity, token_hidden_size=5120,
                        num_ep_ranks=8, num_global_experts=384, num_local_experts=48)
        experts = kwargs.get("num_global_experts")
        actual = dict(expected, num_global_experts=experts,
                      num_local_experts=experts // 8 if type(experts) is int else None)
        if experts not in (128, 384) or kwargs != actual:
            raise ValueError("W4A8 shared LL geometry disagrees with configured capacity")
        # Request the same native cache key for both layouts. Adding a second
        # cache alias would give the native manager two destruction owners.
        buffer = self.native.get_handle(expected)
        _record_layout(experts, self.capacity, id(buffer))
        return buffer
