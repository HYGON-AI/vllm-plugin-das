# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import ast
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _class(path: str, name: str) -> ast.ClassDef:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _methods(node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        member.name: member
        for member in node.body
        if isinstance(member, ast.FunctionDef)
    }


def test_platform_inherits_target_rocm_and_has_no_empty_block_hook() -> None:
    platform = _class("vllm_hcu/platforms/hcu.py", "HCUPlatform")

    assert [ast.unparse(base) for base in platform.bases] == ["RocmPlatform"]
    methods = _methods(platform)
    assert "update_block_size_for_backend" not in methods
    assert {
        "get_valid_backends",
        "apply_config_platform_defaults",
        "check_and_update_config",
        "supports_fp8",
    } <= methods.keys()


def test_worker_keeps_target_runner_and_lifecycle_selection() -> None:
    worker_path = REPO_ROOT / "vllm_hcu/v1/worker.py"
    worker_tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in worker_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HcuGPUWorker"
    )

    assert [ast.unparse(base) for base in worker.bases] == ["Worker"]
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_create_model_runner"
        for node in worker_tree.body
    )
    methods = _methods(worker)
    assert set(methods) == {"__init__", "load_model", "init_device"}
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
        and call.func.attr == "init_device"
        for call in ast.walk(methods["init_device"])
        if isinstance(call, ast.Call)
    )


def test_hcu_v2_runner_inherits_target_v028_input_and_execute_paths() -> None:
    """MRv2 must not accidentally inherit the similarly named V1 runner."""

    code = r'''
import ast
import inspect
import textwrap

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2

assert "prepare_inputs" not in vars(HcuGPUModelRunnerV2)
assert "execute_model" not in vars(HcuGPUModelRunnerV2)
assert HcuGPUModelRunnerV2.prepare_inputs is GPUModelRunner.prepare_inputs
assert HcuGPUModelRunnerV2.execute_model is GPUModelRunner.execute_model

prepare_tree = ast.parse(
    textwrap.dedent(inspect.getsource(GPUModelRunner.prepare_inputs))
)
combine_call = next(
    node
    for node in ast.walk(prepare_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "combine_sampled_and_draft_tokens"
)
arguments = {ast.unparse(argument) for argument in combine_call.args}
assert "self.req_states.draft_tokens" in arguments
assert "scheduler_output.scheduled_spec_decode_tokens" not in arguments
print("V028_MRV2_SELECTED_CONTRACT_OK")
'''
    environment = dict(os.environ)
    environment["VLLM_PLUGINS"] = "__disabled__"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "V028_MRV2_SELECTED_CONTRACT_OK" in result.stdout


def test_v028_mrv2_uses_placeholder_cardinality_not_placeholder_values() -> None:
    """MRv2 scheduler placeholders describe shape, not embedding token IDs."""

    from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids

    num_tokens = {"req-a": 3, "req-b": 2, "req-c": 3}
    real_drafts = {"req-a": [31, 32], "req-b": [], "req-c": [41, 42]}
    placeholders = {"req-a": [-1, -1], "req-b": [], "req-c": [-1, -1]}

    assert sort_batch_req_ids(num_tokens, real_drafts, 3) == (
        sort_batch_req_ids(num_tokens, placeholders, 3)
    )


def test_v028_mrv2_request_state_owns_zero_initialized_draft_tokens() -> None:
    """Embeddings read worker-local drafts, whose unused slots are valid IDs."""

    from vllm.v1.worker.gpu.states import RequestState

    state = RequestState(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=8,
        num_speculative_steps=3,
        vocab_size=1024,
        device=torch.device("cpu"),
    )

    assert state.draft_tokens.shape == (2, 3)
    assert torch.count_nonzero(state.draft_tokens).item() == 0
