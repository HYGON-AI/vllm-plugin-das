# SPDX-License-Identifier: Apache-2.0
"""Numerical and ownership contracts for the former vLLM source delta."""

from types import SimpleNamespace
import functools
import os
import subprocess
import sys

import pytest
import torch


def _isolated_model_test(function):
    # Model registration installs cold import replacements. Other test modules
    # intentionally import upstream MoE during collection, so bootstrap these
    # model contracts in a fresh process, as the projection-fusion tests do.
    @functools.wraps(function)
    def run(*args, **kwargs):
        node = kwargs["request"].node.nodeid
        if os.environ.get("VLLM_HCU_STEP5_DELTA_TEST_CHILD") == node:
            return function(*args, **kwargs)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-xq", node],
            env=dict(os.environ, VLLM_HCU_STEP5_DELTA_TEST_CHILD=node),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


def _model():
    from vllm_hcu import hcu_platform_register_model

    hcu_platform_register_model()
    from vllm_hcu.models import step5_text

    return step5_text


@_isolated_model_test
def test_qkvg_includes_indexer_rows_and_releases_only_consumed_weights(
    monkeypatch, request
):
    m = _model()
    from vllm_hcu.ops.step5 import fp8_dense

    monkeypatch.setattr(m, "QKV_INDEXER", True)
    captured = []

    def quantize(weight):
        captured.append(weight.clone())
        return weight.clone(), torch.ones(weight.shape[1])

    monkeypatch.setattr(fp8_dense, "quantize_weight_fp8", quantize)
    a = m.Step5Attention.__new__(m.Step5Attention)
    torch.nn.Module.__init__(a)
    a.qkvg_weight = torch.arange(48, dtype=torch.bfloat16).reshape(4, 12)
    a.qkv_proj = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    a.g_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    a.indexer = SimpleNamespace(pre_projected=False)
    for name, rows in [("q", 8), ("k", 4), ("z", 4), ("w", 2)]:
        setattr(
            a,
            f"sparse_indexer_{name}",
            torch.nn.Linear(4, rows, bias=False, dtype=torch.bfloat16),
        )
    expected = torch.cat(
        [
            a.qkvg_weight,
            torch.cat([getattr(a, f"sparse_indexer_{n}").weight for n in "qkzw"]).t(),
        ],
        dim=1,
    )
    a._quantize_qkvg()
    torch.testing.assert_close(captured[0], expected)
    assert a.indexer.pre_projected and (a.idx_off, a.idx_n) == (12, 18)
    assert a.qkvg_weight is None
    assert a.qkv_proj.weight.numel() == a.g_proj.weight.numel() == 0
    assert all(getattr(a, f"sparse_indexer_{n}").weight.numel() == 0 for n in "qkz")
    assert a.sparse_indexer_w.weight.numel() == 8


def test_preprojected_indexer_uses_strided_columns_without_linear(monkeypatch):
    from vllm_hcu.model_executor.layers import step4_dsa as dsa

    indexer = object.__new__(dsa.Step4Indexer)
    indexer.owner = SimpleNamespace()
    indexer.geometry = SimpleNamespace(proxy_dim=4)
    indexer.num_heads = 2
    indexer.heads_per_group = 2
    indexer.num_kv_groups = 1
    indexer.pre_projected = True
    indexer.rope_theta = 10000.0
    indexer.rope_scaling = None
    indexer.max_position = 16
    indexer.eps = 1e-6
    indexer.state = SimpleNamespace(
        rope_tables=lambda **kw: (torch.ones(2, 2), torch.ones(2, 2))
    )
    indexer._cast_params = lambda dtype: (None,) * 4
    indexer._merged_projection = lambda dtype: pytest.fail(
        "preprojected path repeated GEMM"
    )
    monkeypatch.setattr(
        dsa.F, "linear", lambda *args, **kw: pytest.fail("repeated linear")
    )
    monkeypatch.setattr(
        dsa, "indexer_norm_rope", lambda q, k, z, *args, **kw: (q, k, z)
    )
    storage = torch.arange(2 * 25, dtype=torch.bfloat16).reshape(2, 25)
    columns = storage[:, 3:21]
    q, k, z, weights = indexer._project(columns, torch.arange(2), 2)
    torch.testing.assert_close(q, columns[:, :8].reshape(2, 1, 2, 4))
    torch.testing.assert_close(k, columns[:, 8:12].reshape(2, 1, 4))
    torch.testing.assert_close(z, columns[:, 12:16].reshape(2, 1, 4))
    torch.testing.assert_close(weights, columns[:, 16:].reshape(2, 1, 2))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@_isolated_model_test
def test_placement_permutates_experts_scales_and_router_together(
    tmp_path, monkeypatch, dtype, request
):
    m = _model()
    placement = tmp_path / "placement.pt"
    permutation = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    torch.save({"physical_to_logical_map": permutation}, placement)
    monkeypatch.setattr(m, "EXPERT_PLACEMENT", str(placement))
    names = [
        "gate_proj.weight",
        "up_proj.weight_scale",
        "down_proj.weight_scale_inv",
        "gate.weight",
        "router_bias",
    ]
    tensors = [
        ("layers.3.moe." + n, torch.arange(8).reshape(4, 2).to(dtype)) for n in names
    ]
    other = ("layers.4.moe.gate.weight", torch.ones(4, 2))
    actual = list(m._place_experts(iter(tensors + [other]), [3]))
    for (_, value), (_, original) in zip(actual[:-1], tensors):
        torch.testing.assert_close(
            value.float(), original.float()[permutation[0].long()]
        )
    assert actual[-1][1] is other[1]
    # Repeated checkpoint streams start from their original logical ordering.
    again = list(m._place_experts(iter(tensors), [3]))
    torch.testing.assert_close(again[0][1].float(), actual[0][1].float())


@pytest.mark.parametrize(
    "mapping",
    [
        torch.tensor([[0, 0, 2]]),
        torch.tensor([0, 1, 2]),
        torch.tensor([[0.0, 1.0, 2.0]]),
    ],
)
@_isolated_model_test
def test_invalid_placement_rejected_before_weights_change(
    tmp_path, monkeypatch, mapping, request
):
    m = _model()
    placement = tmp_path / "invalid.pt"
    torch.save({"physical_to_logical_map": mapping}, placement)
    monkeypatch.setattr(m, "EXPERT_PLACEMENT", str(placement))
    with pytest.raises(ValueError, match="placement"):
        list(m._place_experts([], [3]))


def test_previous_step_retired_row_survives_new_request_before_rollback():
    from vllm_hcu.model_executor.layers.step4_dsa import Step4PendingTable

    table = Step4PendingTable(2)
    table.begin_step()
    old = table.acquire(10)
    table.retire(10)
    table.begin_step()
    live = table.acquire(20)
    table.retire(20)
    table.begin_step()
    assert table.acquire(30) == old
    assert table.get(20) == live
    with pytest.raises(RuntimeError, match="no pending tokens"):
        table.get(10)


@pytest.mark.hcu
@pytest.mark.parametrize("past", [0, 19, 32768])
def test_causal_fp8_prefill_scores_preserve_group_major_strided_output(past):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    from vllm_hcu.model_executor.layers.step4_dsa_kernels import prefill_indexer_logits

    torch.manual_seed(42)
    tokens, groups, heads, dim, regions = 137, 2, 2, 256, 257
    q = torch.randn(tokens, groups, heads, dim, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(regions, 1, dim, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(tokens, groups, heads, device="cuda", dtype=torch.bfloat16)
    stride, base = tokens + 11, 3
    out = torch.full((groups * stride, regions + 5), float("nan"), device="cuda")
    prefill_indexer_logits(
        q,
        weights,
        k,
        first_position=past,
        out=out,
        out_row_base=base,
        out_group_stride=stride,
    )
    expected = (
        torch.einsum("tghd,rd->tghr", q.float(), k[:, 0].float()).relu()
        * weights.float()[..., None]
    ).sum(dim=2)
    visible = ((past + torch.arange(tokens, device="cuda")) // 8).clamp(max=regions)
    mask = torch.arange(regions, device="cuda")[None, :] < visible[:, None]
    for g in range(groups):
        actual = out[g * stride + base : g * stride + base + tokens, :regions]
        torch.testing.assert_close(
            actual[mask], expected[:, g][mask], rtol=2e-4, atol=3e-3
        )
        assert out[g * stride : g * stride + base].isnan().all()
        assert out[g * stride + base + tokens : (g + 1) * stride].isnan().all()
    assert out[:, regions:].isnan().all()


@pytest.mark.hcu
@pytest.mark.parametrize("regions", [33, 4096, 8192, 16384, 16385])
@pytest.mark.parametrize("topk", [7, 512])
def test_resident_prefill_selection_matches_radix_with_ties_and_padding(
    regions, topk, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    from vllm_hcu.model_executor.layers import step4_dsa_kernels as k

    torch.manual_seed(17)
    rows = 7
    scores = torch.randn(rows, regions, device="cuda")
    scores[1].fill_(0.0)
    scores[2] = scores[2].round()
    scores[3] = -scores[3].abs()
    lengths = torch.tensor(
        [
            0,
            min(12, regions),
            regions,
            regions,
            min(512, regions),
            min(513, regions),
            1,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    positions = lengths * 8 + 3
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
    pages = (regions + 7) // 8
    table = torch.stack(
        [
            torch.arange(pages, device="cuda", dtype=torch.int32) + 5,
            torch.arange(pages, device="cuda", dtype=torch.int32) + 5000,
        ]
    )

    def select():
        logical = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
        return k.region_topk_pack(
            scores,
            lengths,
            positions,
            table,
            topk=topk,
            regions_per_page=8,
            request_indices=requests,
            _logical_out=logical,
        )

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    reference = select()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    for _ in range(3):
        actual = select()
        torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
