# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""PCP Engram history must follow each virtual row's fragment boundary."""

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_model_state as patch


def _state(pcp_size):
    module = ModuleType(patch.TARGET_MODULE)

    class DefaultModelState:
        def prepare_inputs(self, input_batch, req_states):
            return {"input_ids": input_batch.input_ids}

    class DeepseekV41ModelState(DefaultModelState):
        def __init__(self, vllm_config, model, encoder_cache, device):
            self.vllm_config = vllm_config
            self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, 3), -1, dtype=torch.int32, device=device
            )

        def prepare_inputs(self, input_batch, req_states):
            return {"upstream": True}

    module.DefaultModelState = DefaultModelState
    module.DeepseekV41ModelState = DeepseekV41ModelState
    patch.apply_to_module(module)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
    )
    return DeepseekV41ModelState(config, None, None, torch.device("cpu"))


@pytest.mark.parametrize("pcp_size", [2, 8])
def test_fragment_history_includes_previous_chunk_and_generated_tokens(pcp_size):
    """More virtual rows than requests, repeated mappings, and a partial window."""
    state = _state(pcp_size)
    tokens = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23, 24, 25, 26, 27]],
        dtype=torch.int32,
    )
    batch = SimpleNamespace(
        num_reqs=4,
        idx_mapping=torch.tensor([1, 0, 1, 0]),
        num_computed_tokens_np=np.array([0, 2, 4, 7], dtype=np.int32),
        input_ids=torch.tensor([20, 12, 24, 17]),
    )
    states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=tokens),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([0, 0])),
    )
    result = state.prepare_inputs(batch, states)
    assert result["input_ids"] is batch.input_ids
    assert result["lookback_token_ids"] is state.lookback_token_ids
    torch.testing.assert_close(
        result["lookback_token_ids"][:4],
        torch.tensor(
            [[-1, -1, -1], [11, 10, -1], [23, 22, 21], [16, 15, 14]], dtype=torch.int32
        ),
    )
    batch.num_reqs = 1
    batch.num_computed_tokens_np[0] = 1
    result = state.prepare_inputs(batch, states)
    torch.testing.assert_close(
        result["lookback_token_ids"][0], torch.tensor([20, -1, -1], dtype=torch.int32)
    )
    assert torch.all(result["lookback_token_ids"][1:] == -1)


def test_non_pcp_model_state_keeps_upstream_prepare_inputs():
    assert _state(1).prepare_inputs(None, None) == {"upstream": True}
