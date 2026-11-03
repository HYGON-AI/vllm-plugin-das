# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exercise the actual profile boundary without allocating enormous tables."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def reservation(config):
    source = (Path(__file__).resolve().parents[2]
              / 'vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_reserve_lightop_identity_page_table_for_profile')
    tables, lengths = [], []

    def current_config():
        if config is None:
            raise AssertionError('No active VllmConfig')
        return config

    namespace = {
        'torch': torch,
        'henvs': SimpleNamespace(VLLM_HCU_USE_CUSTOM_OPS=True),
        '_use_lightop_fast_topk_transform': lambda: True,
        '_lightop_fast_topk_transform': lambda: object(),
        'current_platform': SimpleNamespace(is_rocm=lambda: True),
        'on_gfx938': lambda: True,
        'get_current_vllm_config': current_config,
        '_lightop_identity_page_table': lambda device, rows, width: tables.append((rows, width)),
        '_lightop_unit_query_cu_seqlens': lambda device, rows: lengths.append(rows),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'profile-boundary', 'exec'), namespace)
    return namespace[function.name], tables, lengths, namespace


def config(requests, budget, speculative_tokens):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=requests, max_num_batched_tokens=budget),
        speculative_config=None if speculative_tokens == 'off' else SimpleNamespace(num_speculative_tokens=speculative_tokens),
    )


@pytest.mark.parametrize(('requests', 'budget', 'speculative_tokens', 'profile_rows', 'expected_rows'), [
    (1024, 65536, 3, 65536, 4096),
    (256, 65536, 3, 65536, 1024),
    (256, 65536, 'off', 65536, 256),
    # Padding eight decode requests of lengths [4, 1, 1, 1, 1, 1, 1, 1]
    # consumes 32 kernel rows even though only eleven tokens were scheduled.
    (8, 11, 3, 65536, 32),
    (1024, 65536, 3, 7, 7),
])
def test_profile_reserves_possible_decode_rows_instead_of_prefill_rows(
    requests, budget, speculative_tokens, profile_rows, expected_rows,
):
    reserve, tables, lengths, _ = reservation(config(requests, budget, speculative_tokens))
    hidden = torch.empty((profile_rows, 1), device='meta')
    q = torch.empty((profile_rows, 32, 128), device='meta', dtype=torch.float8_e4m3fn)
    reserve(hidden, q, 2048, 1048576)
    assert tables == [(expected_rows, 1048576)]
    assert lengths == [expected_rows]


@pytest.mark.parametrize('cfg', [None, config(256, 65536, None)])
def test_unknown_decode_bound_keeps_conservative_profile_accounting(cfg):
    reserve, tables, lengths, _ = reservation(cfg)
    reserve(torch.empty((64, 1), device='meta'),
            torch.empty((64, 32, 128), device='meta', dtype=torch.float8_e4m3fn), 2048, 8192)
    assert tables == [(64, 8192)]
    assert lengths == [64]


def test_missing_fast_topk_api_does_not_reserve_unused_mapping():
    reserve, tables, lengths, namespace = reservation(config(1024, 65536, 3))
    namespace['_lightop_fast_topk_transform'] = lambda: None
    reserve(torch.empty((65536, 1), device='meta'),
            torch.empty((65536, 32, 128), device='meta', dtype=torch.float8_e4m3fn), 2048, 1048576)
    assert tables == [] and lengths == []
