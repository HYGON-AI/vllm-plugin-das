# SPDX-License-Identifier: Apache-2.0
"""Reproduce native DFlash attention-PCP reset with native MoE membership."""
import copy
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from vllm.config import ParallelConfig, get_current_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import config as moe_config
from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from hcu_vllm_das.dsv41.profile_worker import HT_FIXED_CAPACITY_KEY, HT_SENDER_INT8_KEY
from hcu_vllm_das.models.dsv41_dspark import HcuDSparkDeepseekV41ForCausalLM
from hcu_vllm_das.pcp_dspark_draft import draft_config, moe_pcp_size, needs_isolated_ht_buffer
from hcu_vllm_das.quantization.ct_w4a8 import HcuDSV41CTW4A8Config, validate_topology
from test_pcp_dspark import config


def native_draft_config(*, target_fixed_ht=False):
    cfg = config()
    cfg.additional_config[HT_FIXED_CAPACITY_KEY] = target_fixed_ht
    cfg.parallel_config = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1, prefill_context_parallel_size=8,
        enable_expert_parallel=True, all2all_backend='deepep_high_throughput',
        distributed_executor_backend='mp')
    cfg.speculative_config.target_parallel_config = cfg.parallel_config
    cfg.speculative_config.draft_model_config.use_mla = False
    cfg.quant_config = object.__new__(HcuDSV41CTW4A8Config)
    cfg.compilation_config.cudagraph_capture_sizes = []
    captured = []

    class Captured(Exception):
        pass

    def stop_before_device_allocation(self, draft, device):
        captured.append(draft)
        raise Captured

    with patch.object(DraftModelSpeculator, '__init__', stop_before_device_allocation):
        with pytest.raises(Captured):
            DFlashSpeculator.__init__(object.__new__(DFlashSpeculator), cfg, torch.device('cpu'))
    draft = captured[0]
    assert cfg.parallel_config.prefill_context_parallel_size == 8
    assert draft.parallel_config.prefill_context_parallel_size == 1
    return draft


def test_real_native_draft_attention_and_moe_have_distinct_pcp_widths():
    cfg = native_draft_config()
    with patch.object(moe_config, 'get_pcp_group', return_value=NS(rank_in_group=0)):
        actual = moe_config.FusedMoEParallelConfig.make(
            tp_size_=1, pcp_size_=8, dp_size_=1, sp_size_=1,
            vllm_parallel_config=cfg.parallel_config)
    moe = NS(moe_parallel_config=actual, max_num_tokens=32768,
             num_experts=64, num_local_experts=8)
    with pytest.raises(ValueError, match='membership mismatch'):
        validate_topology(cfg, moe)
    local = draft_config(cfg)
    validate_topology(local, moe)
    assert local.parallel_config is cfg.parallel_config
    assert moe_pcp_size(local) == 8
    assert 'hcu_static_eplb' not in local.additional_config
    assert 'hcu_static_eplb' in cfg.additional_config
    assert actual.ep_size == actual.pcp_size == 8


@pytest.mark.parametrize("target_fixed_ht", [False, True])
def test_draft_remembers_target_ht_mode_without_enabling_fixed_receive(target_fixed_ht):
    native = native_draft_config(target_fixed_ht=target_fixed_ht)
    native.additional_config[HT_SENDER_INT8_KEY] = target_fixed_ht
    local = draft_config(native)
    assert HT_FIXED_CAPACITY_KEY not in local.additional_config
    assert HT_SENDER_INT8_KEY not in local.additional_config
    assert native.additional_config[HT_SENDER_INT8_KEY] is target_fixed_ht
    assert needs_isolated_ht_buffer(local) is target_fixed_ht
    assert draft_config(local).additional_config == local.additional_config
    assert moe_pcp_size(local) == 8


def test_owned_model_binds_private_config_and_restores_parent():
    cfg = native_draft_config()
    seen = []

    def native_init(self, *, vllm_config, prefix=''):
        torch.nn.Module.__init__(self)
        assert vllm_config is get_current_vllm_config()
        assert vllm_config.parallel_config.prefill_context_parallel_size == 1
        seen.append(moe_pcp_size(vllm_config))

    with set_current_vllm_config(cfg):
        with patch.object(DSparkDeepseekV4ForCausalLM, '__init__', native_init):
            HcuDSparkDeepseekV41ForCausalLM(vllm_config=cfg)
        assert get_current_vllm_config() is cfg
    assert seen == [8]


@pytest.mark.parametrize('mutation', ['capacity', 'width', 'backend'])
def test_pcp_draft_rejects_changed_ownership(mutation):
    cfg = draft_config(native_draft_config())
    cfg = copy.deepcopy(cfg)
    if mutation == 'capacity':
        cfg.scheduler_config.max_num_seqs += 1
    elif mutation == 'width':
        cfg.speculative_config.target_parallel_config.prefill_context_parallel_size = 4
    else:
        cfg.parallel_config.all2all_backend = 'deepep_low_latency'
    with pytest.raises(ValueError, match='PCP DSpark draft'):
        moe_pcp_size(cfg)
