# SPDX-License-Identifier: Apache-2.0
"""int8 lm_head: a tuned DeepGEMM config is only used when its K tile divides K."""

import sys
import types

import pytest

from vllm_hcu.patch.worker.op_opt import patch_int8_lm_head as m


@pytest.fixture(autouse=True)
def fake_deepgemm(monkeypatch):
    mod = types.SimpleNamespace(
        get_w8a16_linear_config=lambda mm, n, k: {"DEFAULT": True, "WEIGHT_LAYOUT": 0})
    monkeypatch.setitem(sys.modules, "deepgemm", mod)


@pytest.mark.parametrize("m_rows,k,expected", [
    (1, 2048, "SMALL"),      # Qwen3.5-35B: tuned shape
    (16, 2048, "SMALL"),
    (32, 2048, "LARGE"),
    (1, 2560, "K512"),       # Qwen3.5-4B: 1024-wide tile would leave a partial tile
    (16, 2560, "K512"),
    (32, 2560, "LARGE"),     # 128-wide tile divides 2560
    (1, 2304, "DEFAULT"),    # neither 1024 nor 512 divides
    (32, 2304, "LARGE"),
])
def test_config_k_tile(m_rows, k, expected):
    cfg = m._config(m_rows, 248320, k)
    name = {id(m._CFG_SMALL): "SMALL", id(m._CFG_SMALL_K512): "K512",
            id(m._CFG_LARGE): "LARGE"}.get(id(cfg), "DEFAULT")
    assert name == expected
    if name == "DEFAULT":
        assert cfg["WEIGHT_LAYOUT"] == 1
    else:
        assert k % m._k_tile(cfg) == 0
