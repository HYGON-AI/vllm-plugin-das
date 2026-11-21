# SPDX-License-Identifier: Apache-2.0
"""Run real runner entry/dispatch code on CPU, stopping before graph selection."""
import ast
import __future__
import inspect
import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.deepseek_v4_runtime import is_deepseek_v4
from vllm_hcu.v1.pcp_manager import (
    pcp_dispatch_metadata_scope, set_pcp_dispatch_metadata,
)
from tests.runtime_patch.test_glm52_pcp_manager import _make_batch, _make_managers

ROOT = Path(__file__).resolve().parents[2]


def _dispatch_prefix(source):
    fn = ast.parse(textwrap.dedent(source)).body[0]
    fn.decorator_list = []
    stop = next(i for i, node in enumerate(fn.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "num_active_loras"
                        for t in node.targets))
    fn.body = fn.body[:stop] + [ast.parse(
        "return num_toks, batch_req_state, uniform_tok_count").body[0]]
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 "<real-dispatch-prefix>", "exec", flags=__future__.annotations.compiler_flag), ns)
    return ns[fn.name]


def dispatch_before_prepare(
    runner, output, *, step5=False, architecture="DeepseekV4ForCausalLM",
):
    # Compile the actual HCU entry and gather methods over the real upstream
    # execute prefix. No InputBatch exists until this call has returned.
    upstream = _dispatch_prefix(inspect.getsource(GPUModelRunner.execute_model))
    if step5:
        tree = ast.parse((ROOT / "vllm_hcu/v1/step5_execution.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "execute_model")
        upstream = _dispatch_prefix(ast.unparse(fn))

    class Base:
        execute_model = upstream
        gather_batch_req_state = GPUModelRunner.gather_batch_req_state

    tree = ast.parse((ROOT / "vllm_hcu/v1/hcu_model_runner_v2.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and any(isinstance(m, ast.FunctionDef) and m.name == "gather_batch_req_state"
                       for m in n.body))
    methods = [m for m in cls.body if isinstance(m, ast.FunctionDef)
               and m.name in ("gather_batch_req_state", "execute_model")]
    for method in methods:
        method.decorator_list = []
    adapter = ast.ClassDef(name="Adapter", bases=[ast.Name(id="Base", ctx=ast.Load())],
                          keywords=[], body=methods, decorator_list=[])
    ns = dict(Base=Base, nullcontext=nullcontext,
              pcp_dispatch_metadata_scope=pcp_dispatch_metadata_scope,
              set_pcp_dispatch_metadata=set_pcp_dispatch_metadata,
              deepep_auto_request_phase_scope=nullcontext,
              _uses_global_pcp_batch=lambda config: False,
              _DEEPEP_AUTO_DUMMY_PHASE=SimpleNamespace(get=lambda: None),
              henvs=SimpleNamespace(custom_ops_enabled=lambda: False),
              is_deepseek_v4=is_deepseek_v4)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adapter], type_ignores=[])),
                 "<real-hcu-entry>", "exec"), ns)
    obj = ns["Adapter"]()
    obj.__dict__.update(vars(runner))
    obj.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
    )
    obj.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    for name in ("update_pp_decode_requests", "finish_requests", "free_states",
                 "add_requests", "update_requests"):
        setattr(obj, name, lambda *args: None)
    obj.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    return obj.execute_model(output)


@pytest.mark.parametrize("prefill,length,expected", [(True, 128, 64), (False, 1, 1)])
def test_step5_batch_req_state_dispatch(prefill, length, expected):
    output = SimpleNamespace(num_scheduled_tokens={"req": length},
                             total_num_scheduled_tokens=length,
                             scheduled_spec_decode_tokens={})
    managers, _ = _make_managers(2)
    runner = SimpleNamespace(
        pcp_manager=managers[0], decode_query_len=1, adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"req": 0},
            prefill_len=SimpleNamespace(np=np.array([128], dtype=np.int32)),
            num_computed_prefill_tokens=np.array([0 if prefill else 128], dtype=np.int32),
        ),
    )
    padding, state, _ = dispatch_before_prepare(runner, output, step5=True)
    assert padding == expected
    assert not hasattr(state, "num_draft_tokens_per_req")


def test_dispatch_metadata_scope_restores_after_error():
    from vllm_hcu.v1.pcp_manager import _DISPATCH_METADATA
    original = _DISPATCH_METADATA.get()
    with pytest.raises(RuntimeError), pcp_dispatch_metadata_scope():
        set_pcp_dispatch_metadata(
            SimpleNamespace(req_ids=["r"], num_scheduled_tokens=np.array([6])),
            SimpleNamespace(scheduled_spec_decode_tokens={"r": [0] * 5}), 1)
        assert _DISPATCH_METADATA.get() is not None
        raise RuntimeError("execute failed")
    assert _DISPATCH_METADATA.get() is original


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
def test_pre_dispatch_uses_gather_order_and_keeps_real_prefill(pcp_size):
    output = SimpleNamespace(
        num_scheduled_tokens={"prefill": 128, "verify": 6},
        total_num_scheduled_tokens=134,
        scheduled_spec_decode_tokens={"verify": [-1] * 5},
    )
    managers, _ = _make_managers(pcp_size)
    runner = SimpleNamespace(
        pcp_manager=managers[0], decode_query_len=6, adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"prefill": 0, "verify": 1},
            prefill_len=SimpleNamespace(np=np.array([256, 1], dtype=np.int32)),
            num_computed_prefill_tokens=np.array([0, 0], dtype=np.int32),
        ),
    )
    padding, state, uniform = dispatch_before_prepare(runner, output)
    assert padding == 128 // pcp_size + 6
    assert state.is_prefilling_np.all()
    assert state.has_prefill
    assert uniform is None


@pytest.mark.parametrize("architecture", [
    "GlmMoeDsaForCausalLM", "HYV4ForCausalLM", "DeepseekV4ForCausalLM",
])
@pytest.mark.parametrize("step5", [False, True])
@pytest.mark.parametrize("pcp_size", [2, 4, 8])
@pytest.mark.parametrize("spec_tokens", [1, 2, 3])
def test_speculative_short_prefill_dispatch_matches_partition(
    architecture, step5, pcp_size, spec_tokens,
):
    width = spec_tokens + 1
    output = SimpleNamespace(
        num_scheduled_tokens={"short-prefill": width},
        total_num_scheduled_tokens=width,
        scheduled_spec_decode_tokens={"short-prefill": [-1] * spec_tokens},
    )
    managers, _ = _make_managers(pcp_size)
    runner = SimpleNamespace(
        pcp_manager=managers[0], decode_query_len=width,
        adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"short-prefill": 0},
            prefill_len=SimpleNamespace(np=np.array([1], dtype=np.int32)),
            num_computed_prefill_tokens=np.array([0], dtype=np.int32),
        ),
    )
    padding, state, _ = dispatch_before_prepare(
        runner, output, step5=step5, architecture=architecture,
    )
    # MRV2 still sees a prefill, while speculative token/logit widths require
    # the complete row on every rank, including GLM-5.2 and HY-V4 runners.
    assert state.is_prefilling_np.tolist() == [True]
    assert padding == width

    batch = _make_batch([("short-prefill", list(range(width)), width, True)])
    batch.prefill_len_np = state.prefill_len_np
    batch.num_computed_prefill_tokens_np = state.num_computed_prefill_tokens_np
    batch.num_draft_tokens = spec_tokens
    batch.num_draft_tokens_per_req = np.array([spec_tokens], dtype=np.int32)
    batch.cu_num_logits_np = np.array([0, width], dtype=np.int32)
    batch.cu_num_logits = torch.from_numpy(batch.cu_num_logits_np.copy())
    batch.logits_indices = torch.arange(width)
    for manager in managers:
        local = manager.partition_batch(batch, padded_num_tokens=padding)
        assert local.num_scheduled_tokens.tolist() == [width]
        assert local.num_draft_tokens_per_req.tolist() == [spec_tokens]
        assert local.cu_num_logits_np.tolist() == [0, width]
        assert not local.is_prefilling_np.any()
        assert not local.is_padding.any()
        torch.testing.assert_close(local.input_ids, batch.input_ids)
