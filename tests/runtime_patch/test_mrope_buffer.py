# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Differential checks against the installed vLLM runner's actual methods."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.utils import CpuGpuBuffer
import vllm

from vllm_hcu.v1 import mrope_buffer as feature


def load_methods(path, class_name, names):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in methods} == set(names)
    namespace = {
        "torch": torch,
        "np": np,
        "PIN_MEMORY": False,
        "MRotaryEmbedding": MRotaryEmbedding,
        "length_from_prompt_token_ids_or_embeds": length_from_prompt_token_ids_or_embeds,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


NATIVE = Path(vllm.__file__).parent / "v1/worker/gpu_model_runner.py"
METHODS = load_methods(
    NATIVE, "GPUModelRunner", ["_calc_mrope_positions", "_get_positions"]
)
CALC = METHODS["_calc_mrope_positions"]
GET = METHODS["_get_positions"]


def buffer(enabled, tokens=129, device="cpu"):
    if enabled:
        return feature.TokenMajorMropeBuffer(
            tokens, device=torch.device(device), pin_memory=False
        )
    return CpuGpuBuffer(
        3, tokens, dtype=torch.int64, device=torch.device(device), pin_memory=False
    )


@pytest.mark.parametrize("mode", ["prefill", "decode", "cross_boundary", "reordered"])
def test_position_calculation_matches_original(mode):
    requests = {}
    for i in range(4):
        values = torch.arange(3 * 20).view(3, 20) + 100 * i
        requests[str(i)] = NS(
            prompt_token_ids=list(range(20)),
            prompt_embeds=None,
            mrope_positions=values,
            mrope_position_delta=7 * i,
        )
    ids = list(requests)
    if mode == "reordered":
        ids.reverse()
    computed = np.array(
        {
            "prefill": [0, 3, 5, 7],
            "decode": [21, 30, 25, 39],
            "cross_boundary": [18, 19, 17, 22],
            "reordered": [2, 22, 18, 8],
        }[mode]
    )
    scheduled = {rid: i + 2 for i, rid in enumerate(ids)}
    output = NS(num_scheduled_tokens=scheduled)
    runners = [
        NS(
            mrope_positions=buffer(enabled),
            requests=requests,
            input_batch=NS(req_ids=ids, num_computed_tokens_cpu=computed),
            uses_mrope=True,
            uses_xdrope_dim=0,
        )
        for enabled in (False, True)
    ]
    for runner in runners:
        CALC(runner, output)
    torch.testing.assert_close(
        runners[0].mrope_positions.cpu, runners[1].mrope_positions.cpu
    )
    assert runners[1].mrope_positions.cpu.stride() == (1, 3)
    assert np.shares_memory(
        runners[1].mrope_positions.np, runners[1].mrope_positions.flat_cpu.numpy()
    )


@pytest.mark.parametrize(
    "index", [0, 1, 64, 128, torch.tensor([0, 2, 127, 128]), slice(3, 8)]
)
def test_existing_get_positions_and_dummy_slot(index):
    logical = torch.arange(3 * 129).view(3, 129)
    runners = []
    for enabled in (False, True):
        buf = buffer(enabled)
        buf.cpu.copy_(logical)
        # This is the unchanged native _prepare_inputs expression.
        buf.gpu[:, :129].copy_(buf.cpu[:, :129], non_blocking=True)
        runners.append(NS(uses_mrope=True, uses_xdrope_dim=0, mrope_positions=buf))
    torch.testing.assert_close(GET(runners[0], index), GET(runners[1], index))


def test_partial_flat_copy_preserves_padding():
    buf = buffer(True)
    buf.flat_gpu.fill_(-1)
    buf.cpu[:, :5] = torch.arange(15).view(3, 5)
    result = buf.copy_to_gpu(5)
    assert result.shape == (3, 5)
    torch.testing.assert_close(result, buf.cpu[:, :5])
    assert torch.all(buf.gpu[:, 5:] == -1)
    buf.gpu[:, :2].fill_(99)
    buf.copy_to_cpu(2)
    assert torch.all(buf.cpu[:, :2] == 99)
