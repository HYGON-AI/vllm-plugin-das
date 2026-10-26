from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.hcu

SUPPORTED_LIVE_ARCHITECTURES = frozenset({"gfx936", "gfx938"})


def _live_requested() -> bool:
    return os.environ.get("HCU_VLLM032_LIVE") == "1"


@pytest.mark.skipif(
    not _live_requested(),
    reason="set HCU_VLLM032_LIVE=1 for installed-wheel HCU validation",
)
def test_installed_lightop_silu_matches_reference_and_rollback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import torch.nn.functional as functional

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    assert current_platform.__class__.__module__ == "hcu_vllm_adapter.platform.hcu"
    load_general_plugins()

    from hcu_vllm_das.silu_and_mul import HcuSiluAndMul

    properties = torch.cuda.get_device_properties(0)
    architecture = properties.gcnArchName.split(":", 1)[0]
    assert architecture in SUPPORTED_LIVE_ARCHITECTURES

    torch.manual_seed(20261026)
    value = torch.randn((257, 8192), device="cuda", dtype=torch.bfloat16)
    expected = (
        functional.silu(value[..., :4096].float())
        * value[..., 4096:].float()
    ).to(torch.bfloat16)
    # CustomOp reads the current vLLM compilation config during construction.
    # A full engine sets this context; this focused kernel test mirrors vLLM's
    # own ``default_vllm_config`` fixture explicitly.
    with set_current_vllm_config(VllmConfig()):
        layer = SiluAndMul()
        assert isinstance(layer, HcuSiluAndMul)

        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_SILU_AND_MUL", "1")
        optimized = layer(value)
        torch.cuda.synchronize()
        torch.testing.assert_close(optimized, expected, rtol=2e-2, atol=2e-2)

        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
        master_rollback = layer(value)
        torch.testing.assert_close(master_rollback, expected, rtol=2e-2, atol=2e-2)

        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_SILU_AND_MUL", "0")
        child_rollback = layer(value)
        torch.testing.assert_close(child_rollback, expected, rtol=2e-2, atol=2e-2)
