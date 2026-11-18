# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contract coverage for in-place and out-of-place indexer RoPE."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
MODEL_SOURCE = (REPO / "vllm_hcu/models/deepseek_v2.py").read_text(encoding="utf-8")


def _load_indexer_forward(is_rocm):
    # Follow the isolated source-contract tests: do not import the model and
    # initialize HCU, runtime patches, or tensor-parallel state during collection.
    tree = ast.parse(MODEL_SOURCE)
    indexer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Indexer"
    )
    forward = next(
        node
        for node in indexer.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    namespace = {
        "torch": torch,
        "current_platform": SimpleNamespace(is_rocm=lambda: is_rocm),
        "on_gfx938": lambda: True,
        "get_forward_context": lambda: SimpleNamespace(enable_lightly_cp=False),
        "per_token_group_quant_fp8": lambda q, *args, **kwargs: (
            q.clone(),
            torch.full((q.shape[0], 1), 2.0, dtype=q.dtype),
        ),
    }
    module = ast.Module(body=[forward], type_ignores=[])
    filename = str(REPO / "vllm_hcu/models/deepseek_v2.py")
    exec(compile(module, filename, "exec"), namespace)
    return namespace["forward"]


@pytest.mark.parametrize("is_rocm", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("extra_leading_dim", [False, True])
def test_indexer_consumes_rope_return_values(is_rocm, in_place, extra_leading_dim):
    tokens, heads, dim, rope_dim = 2, 4, 8, 4
    query = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(
        tokens, heads, dim
    ) / 16
    key = torch.arange(tokens * dim, dtype=torch.float32).reshape(tokens, dim) / 8
    weights = torch.arange(tokens * heads, dtype=torch.float32).reshape(tokens, heads)
    positions = torch.tensor([2, 17])

    def rotary_emb(positions, q, k):
        offset = positions[:, None, None].to(q.dtype)
        q_rot, k_rot = q + offset / 8, k - offset / 4
        if in_place:
            q.copy_(q_rot)
            k.copy_(k_rot)
            q_rot, k_rot = q, k
        if extra_leading_dim:
            q_rot, k_rot = q_rot.unsqueeze(0), k_rot.unsqueeze(0)
        return q_rot, k_rot

    indexer = SimpleNamespace(
        wq_b=lambda qr: (query.clone().reshape(tokens, -1), None),
        wk_weights_proj=lambda hidden: (torch.cat((key, weights), dim=-1), None),
        k_norm=lambda k: k,
        head_dim=dim,
        n_head=heads,
        rope_dim=rope_dim,
        quant_block_size=dim,
        scale_fmt="ue8m0",
        softmax_scale=0.5,
        indexer_op=lambda hidden, q, k, w: (q, k, w),
    )
    q, k, w = _load_indexer_forward(is_rocm)(
        indexer, torch.zeros(tokens, 1), torch.zeros(tokens, 1), positions, rotary_emb
    )
    expected_q = query.clone()
    expected_k = key.clone()
    expected_q[..., :rope_dim] += positions[:, None, None] / 8
    expected_k[..., :rope_dim] -= positions[:, None] / 4

    assert q.shape == query.shape
    assert k.shape == key.shape
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(w, weights * 2 * 0.5 * heads**-0.5)
    # The unrotated NoPE components must not be changed by either backend.
    torch.testing.assert_close(q[..., rope_dim:], query[..., rope_dim:])
    torch.testing.assert_close(k[..., rope_dim:], key[..., rope_dim:])
