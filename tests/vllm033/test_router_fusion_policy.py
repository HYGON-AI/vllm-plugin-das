# SPDX-License-Identifier: Apache-2.0
"""Master rollback must refuse binding before touching any model state."""
from importlib import import_module

import pytest


@pytest.mark.parametrize("module,function,expected", [('hcu_vllm_das.moe.router_stagger', 'bind_router', False)])
@pytest.mark.parametrize("value", ["0", "false", "no", "off", "NO", " 0 ", " false "])
def test_master_rollback_leaves_consumers_untouched(monkeypatch, module, function, expected, value):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", value)
    # Accessing config/model attributes would fail: OFF must exit first.
    assert getattr(import_module(module), function)(object(), object()) is expected
