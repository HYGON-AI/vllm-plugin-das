# SPDX-License-Identifier: Apache-2.0
"""Keep each HY4 PCP draft draw identical before the next cache write.

The post-load seam binds only the selected speculator instance. Native model,
proposal loops, verification and request-state updates remain unchanged.
"""
from dataclasses import dataclass
import os
from types import MethodType

from vllm.distributed import get_pcp_group

_STATE = '_hcu_hy4_pcp_mtp_binding'


@dataclass
class _Binding:
    original: object
    group: object
    pid: int
    failure: str | None = None


def _healthy(instance, state):
    if state.failure is not None:
        raise RuntimeError(state.failure)
    if (os.getpid() != state.pid
            or getattr(instance.sample_draft, '__func__', None) is not _sample_draft
            or type(instance).sample_draft is not state.original
            or get_pcp_group() is not state.group):
        state.failure = 'HY4 PCP MTP sampling changed owner/process/group; restart'
        raise RuntimeError(state.failure)


def _sample_draft(self, hidden_states, sample_src_positions, idx_mapping,
                  temperature, seeds, draft_step, draft_logits,
                  spec_step_idx=0, num_steps=1):
    state = getattr(self, _STATE)
    _healthy(self, state)
    if draft_logits is not None:
        raise ValueError('HY4 PCP MTP synchronization requires greedy draft sampling')
    tokens = state.original(self, hidden_states, sample_src_positions, idx_mapping,
        temperature, seeds, draft_step, draft_logits,
        spec_step_idx=spec_step_idx, num_steps=num_steps)
    # Synchronize here, not at the end of propose(): the next draft step has
    # already consumed this token and written its KV by that later boundary.
    return state.group.broadcast(tokens, src=0)


def applies(cfg):
    spec = getattr(cfg, 'speculative_config', None)
    return (getattr(cfg.model_config.hf_config, 'model_type', None) == 'hy_v4'
        and cfg.parallel_config.prefill_context_parallel_size > 1
        and spec is not None and spec.method == 'mtp')


def _policy(cfg):
    p = cfg.parallel_config
    if (not cfg.use_v2_model_runner or not cfg.model_config.enforce_eager
            or any(getattr(p, field) != 1 for field in (
                'tensor_parallel_size', 'pipeline_parallel_size',
                'data_parallel_size', 'decode_context_parallel_size'))
            or cfg.speculative_config.draft_sample_method != 'greedy'):
        raise ValueError('HY4 PCP MTP synchronization requires eager MRV2/TP1/PP1/DP1/DCP1 and greedy drafts')


def prepare(cfg):
    """Require and register the existing load seam before allocating weights."""
    if not applies(cfg):
        return
    _policy(cfg)
    from hcu_adapter.bridges.vllm_model_load import (
        register_post_model_load, require_post_model_load,
    )
    require_post_model_load(cfg)
    register_post_model_load('hy4-static-pcp-mtp', on_model_loaded)


def bind_speculator(instance, cfg):
    if not applies(cfg):
        return
    _policy(cfg)
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
    if type(instance) is not MTPSpeculator:
        raise RuntimeError('HY4 PCP MTP requires the native MTP speculator owner')
    state = getattr(instance, _STATE, None)
    if state is not None:
        _healthy(instance, state)
        return
    original = DraftModelSpeculator.sample_draft
    if (type(instance).sample_draft is not original
            or 'sample_draft' in instance.__dict__):
        raise RuntimeError('HY4 PCP MTP draft sampling has a foreign owner')
    group = get_pcp_group()
    if group.world_size != cfg.parallel_config.prefill_context_parallel_size:
        raise ValueError('HY4 PCP MTP group size disagrees with configuration')
    if (instance.acceptance_estimator is not None
            or instance.draft_watermarker is not None):
        raise ValueError('HY4 PCP MTP adaptive verification/watermarking is not validated')
    setattr(instance, _STATE, _Binding(original, group, os.getpid()))
    instance.sample_draft = MethodType(_sample_draft, instance)


def on_model_loaded(context):
    if applies(context.vllm_config):
        bind_speculator(context.worker.model_runner.speculator, context.vllm_config)
