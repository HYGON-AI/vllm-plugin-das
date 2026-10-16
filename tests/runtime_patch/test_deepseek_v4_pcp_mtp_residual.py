# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Native V4 residual writes and sampling with real PCP layouts on CPU."""

from __future__ import annotations

import ast
from collections import namedtuple
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import vllm

from tests.runtime_patch.test_glm52_pcp_manager import _make_batch, _make_managers
from tests.runtime_patch.test_glm52_pcp_runner import _config, pcp_runner_module
from vllm_hcu.v1 import pcp_manager


def _native_method(relative_path, class_name, method_name, namespace):
    # Load the complete installed method without importing GPU-only model
    # dependencies. No statements from the native method are substituted.
    path = Path(vllm.__file__).parent / relative_path
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    code = ast.Module(body=[method], type_ignores=[])
    exec(compile(code, str(path), "exec", dont_inherit=False), namespace)
    return namespace[method_name]


def _native_target():
    namespace = {
        "torch": torch,
        "islice": islice,
        "get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    }
    forward = _native_method("models/deepseek_v4/amd/model.py",
                             "DeepseekV4Model", "forward", namespace)
    getter = _native_method("models/deepseek_v4/amd/model.py",
                            "DeepseekV4ForCausalLM", "get_mtp_target_hidden_states",
                            namespace)

    class Layer:
        use_fused_mhc = False

        def __call__(self, hidden, positions, input_ids, *args):
            return hidden + positions[:, None, None], None, None, None

    model = SimpleNamespace(
        embed_input_ids=lambda ids: torch.stack((ids.float(), ids.float() * 10), -1),
        hc_mult=2, layers=[Layer()], start_layer=0, end_layer=1,
        aux_hidden_state_layers=set(),
        _mtp_hidden_buffer=torch.full((128, 4), -999.0),
        hc_head_op=lambda hidden, *args: hidden.sum(1) + 1000,
        hc_head_fn=None, hc_head_scale=None, hc_head_base=None,
        rms_norm_eps=1e-6, hc_eps=1e-6, norm=lambda hidden: hidden,
    )
    model.forward = MethodType(forward, model)
    target = SimpleNamespace(model=model, compute_logits=lambda *args: None)
    target.get_mtp_target_hidden_states = MethodType(getter, target)
    return target


def _native_sampler():
    namespace = {
        "torch": torch,
        "step_eplb_after": lambda: (lambda function: function),
        "pcp": pcp_manager,
        "use_workspace_lane": lambda *args: nullcontext(),
        "ModelRunnerOutput": lambda **kwargs: SimpleNamespace(**kwargs),
        "AsyncOutput": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    return _native_method("v1/worker/gpu/model_runner.py", "GPUModelRunner",
                          "sample_tokens", namespace)


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
@pytest.mark.parametrize("layout", ["prefill", "mixed", "short", "decode"])
def test_native_mtp_propose_receives_global_pre_hc_residual(
    pcp_runner_module, monkeypatch, pcp_size, layout,
):
    module, _ = pcp_runner_module
    monkeypatch.setattr(module.GPUModelRunner, "sample_tokens", _native_sampler())
    monkeypatch.setattr(module, "build_slot_mappings_by_layer", lambda *args: {})
    managers, groups = _make_managers(pcp_size)
    models = [_native_target() for _ in managers]
    reference = _native_target()
    State = namedtuple("State", (
        "input_batch attn_metadata slot_mappings_by_layer hidden_states "
        "aux_hidden_states dp_sync finished_req_ids ec_connector_output "
        "routed_experts cudagraph_stats"
    ))

    # Reuse models and PCP managers across steps to expose stale buffer tails.
    for step in range(2):
        offset = step * 100
        if layout == "mixed":
            requests = [("d", [offset + 50], 30, False),
                        ("p", list(range(offset + 1, offset + 16)), 20, True)]
        elif layout == "decode":
            requests = [("d", [offset + 50], 30, False)]
        else:
            length = 1 if layout == "short" else 16
            requests = [("p", list(range(offset + 1, offset + length + 1)),
                         length + step, True)]
        batch = _make_batch(requests)
        local_batches = [manager.partition_batch(batch) for manager in managers]
        expected_hidden = reference.model.forward(batch.input_ids, batch.positions, None)
        expected_residual = reference.get_mtp_target_hidden_states()[:batch.num_tokens].clone()
        outputs = [model.model.forward(local.input_ids, local.positions, None)
                   for model, local in zip(models, local_batches)]
        residuals = [model.get_mtp_target_hidden_states()[:output.shape[0]].clone()
                     for model, output in zip(models, outputs)]

        for rank, (manager, group, model, local, output) in enumerate(zip(
            managers, groups, models, local_batches, outputs,
        )):
            calls = []

            def gather(tensor, dim=0):
                peers = residuals if tensor.shape[1] == 4 else outputs
                torch.testing.assert_close(tensor, peers[rank])
                calls.append(tensor.shape[1])
                return torch.cat(peers, dim=dim)

            monkeypatch.setattr(group, "all_gather", gather)
            monkeypatch.setattr(manager, "prepare_global_attn", lambda: ((), None))
            config = _config(pcp_size)
            config.model_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
            runner = module.HcuGPUModelRunnerV2(config, "cpu")
            runner.model = model
            runner.pcp_manager = manager
            runner.is_last_pp_rank = True
            runner.model_state = SimpleNamespace(prepare_attn=lambda *args: {})
            runner.kv_cache_config = runner.attn_groups = None
            proposals = []

            def propose(input_batch, metadata, slots, hidden, *args, **kwargs):
                assert input_batch is batch
                torch.testing.assert_close(hidden, expected_residual)
                assert hidden.shape[1] == 4  # pre-hc, not post-hc hidden states
                proposals.append(hidden.clone())
                return torch.zeros((batch.num_reqs, 2), dtype=torch.int32)

            runner.speculator = SimpleNamespace(supports_mm_inputs=False, propose=propose)
            states = SimpleNamespace(temperature=SimpleNamespace(gpu=None),
                                     seeds=SimpleNamespace(gpu=None))
            runner.sampler = SimpleNamespace(sampling_states=states)

            def sample(hidden, input_batch, grammar):
                assert input_batch is batch
                torch.testing.assert_close(hidden, expected_hidden)
                return SimpleNamespace(sampled_token_ids=None), None, None

            runner.sample = sample
            runner.req_states = SimpleNamespace(
                all_token_ids=SimpleNamespace(gpu=None),
                num_computed_tokens=SimpleNamespace(gpu=None),
                prompt_len=SimpleNamespace(np=None),
                last_sampled_tokens=None, next_prefill_tokens=None,
                draft_tokens=torch.zeros((32, 2), dtype=torch.int32),
            )
            runner.prompt_logprobs_worker = SimpleNamespace(
                compute_prompt_logprobs=lambda *args: None,
            )
            runner.pp_handler = None
            runner.main_stream = runner.output_copy_stream = None
            runner.check_ep_fault = False
            runner.postprocess_sampled = lambda *args: None
            runner._draft_workspace_lane = None
            runner.adaptive_verification = None
            runner.num_speculative_steps = 0
            runner.kv_connector = SimpleNamespace(post_forward=lambda *args: None)
            runner.execute_model_state = State(
                local, {}, {}, output, None, None, set(), None, None, None,
            )
            persistent_buffer = model.get_mtp_target_hidden_states()
            runner.sample_tokens(None)
            assert len(proposals) == 1
            assert model.get_mtp_target_hidden_states() is persistent_buffer
            assert runner.pcp_manager is manager
            assert runner.execute_model_state is None
            expected_calls = [] if manager._hidden_restore_idx is None else [4, 2]
            assert calls == expected_calls  # no duplicate restore by native sampler
