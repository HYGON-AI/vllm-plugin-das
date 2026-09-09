# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tests for patch_base_communicator_pcp: widens use_all2all under PCP + EP."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.worker.framework_opt import patch_base_communicator_pcp


# ---------------------------------------------------------------------------
# Fake vLLM ``base_device_communicator`` module
# ---------------------------------------------------------------------------


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _fake_base_communicator_module() -> ModuleType:
    """Reproduce the audited vLLM 0.25.1 DeviceCommunicatorBase surface.

    Only the parts our patch inspects are modeled: the constructor signature,
    ``is_ep_communicator``/``use_all2all``/``all2all_backend`` attributes.
    """

    class DeviceCommunicatorBase:
        def __init__(
            self,
            cpu_group,
            device=None,
            device_group=None,
            unique_name: str = "",
            global_ranks=None,
            global_world_size=None,
        ):
            self.cpu_group = cpu_group
            self.device = device
            self.device_group = device_group
            self.unique_name = unique_name
            # Reproduce the upstream deduction of these three attributes;
            # only their final values matter to the patch.
            self.is_ep_communicator = unique_name.split(":")[0] == "ep"
            self.use_all2all = False
            self.all2all_backend = None

    return _module(
        patch_base_communicator_pcp.TARGET_MODULE,
        DeviceCommunicatorBase=DeviceCommunicatorBase,
    )


def _install_vllm_config(monkeypatch: pytest.MonkeyPatch, config: object) -> None:
    import vllm.config

    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config_or_none", lambda: config
    )


def _pcp_ep_config(
    *,
    pcp_size: int = 8,
    ep_enabled: bool = True,
    backend: str | None = "deepep_high_throughput",
) -> SimpleNamespace:
    parallel = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        enable_expert_parallel=ep_enabled,
        all2all_backend=backend,
        data_parallel_size=1,
    )
    return SimpleNamespace(parallel_config=parallel)


# ---------------------------------------------------------------------------
# Positive path: PCP + EP + DeepEP backend forces use_all2all=True
# ---------------------------------------------------------------------------


def test_base_pcp_ep_forces_use_all2all_for_pcp_ep_deepep_ht(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    _install_vllm_config(monkeypatch, _pcp_ep_config())
    communicator = module.DeviceCommunicatorBase(
        object(), unique_name="ep:0"
    )
    assert communicator.is_ep_communicator is True
    assert communicator.use_all2all is True


def test_base_pcp_ep_forces_use_all2all_for_pcp_ep_deepep_ll(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    _install_vllm_config(
        monkeypatch, _pcp_ep_config(backend="deepep_low_latency")
    )
    communicator = module.DeviceCommunicatorBase(
        object(), unique_name="ep:0"
    )
    assert communicator.use_all2all is True


# ---------------------------------------------------------------------------
# Negative paths: patch must leave use_all2all=False in every other scenario
# ---------------------------------------------------------------------------


def test_base_pcp_ep_ignores_non_ep_communicator(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    _install_vllm_config(monkeypatch, _pcp_ep_config())
    for unique_name in ("tp:0", "pp:0", "dp:0", "world", ""):
        communicator = module.DeviceCommunicatorBase(
            object(), unique_name=unique_name
        )
        assert communicator.is_ep_communicator is False
        assert communicator.use_all2all is False, unique_name


def test_base_pcp_ep_ignores_non_deepep_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    for backend in ("naive", "pynccl", "allgather_reducescatter", None):
        _install_vllm_config(
            monkeypatch, _pcp_ep_config(backend=backend)
        )
        communicator = module.DeviceCommunicatorBase(
            object(), unique_name="ep:0"
        )
        assert communicator.use_all2all is False, backend


def test_base_pcp_ep_ignores_pcp_size_one(monkeypatch: pytest.MonkeyPatch):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    _install_vllm_config(monkeypatch, _pcp_ep_config(pcp_size=1))
    communicator = module.DeviceCommunicatorBase(
        object(), unique_name="ep:0"
    )
    assert communicator.use_all2all is False


def test_base_pcp_ep_ignores_expert_parallel_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    _install_vllm_config(monkeypatch, _pcp_ep_config(ep_enabled=False))
    communicator = module.DeviceCommunicatorBase(
        object(), unique_name="ep:0"
    )
    assert communicator.use_all2all is False


def test_base_pcp_ep_preserves_upstream_use_all2all_true(
    monkeypatch: pytest.MonkeyPatch,
):
    """When upstream already flipped use_all2all=True (DP>1 case), do nothing."""
    module = _fake_base_communicator_module()

    class UpstreamAlreadyEnabled(module.DeviceCommunicatorBase):
        def __init__(
            self,
            cpu_group,
            device=None,
            device_group=None,
            unique_name: str = "",
            global_ranks=None,
            global_world_size=None,
        ):
            super().__init__(
                cpu_group,
                device=device,
                device_group=device_group,
                unique_name=unique_name,
                global_ranks=global_ranks,
                global_world_size=global_world_size,
            )
            # Simulate upstream having chosen to enable all2all already.
            self.use_all2all = True

    module.DeviceCommunicatorBase = UpstreamAlreadyEnabled
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    # Even if PCP+EP is off, upstream's True must remain True.
    _install_vllm_config(
        monkeypatch, _pcp_ep_config(pcp_size=1, ep_enabled=False)
    )
    communicator = module.DeviceCommunicatorBase(
        object(), unique_name="ep:0"
    )
    assert communicator.use_all2all is True


def test_base_pcp_ep_tolerates_missing_vllm_config(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    _install_vllm_config(monkeypatch, None)
    # Should not raise; use_all2all stays False.
    communicator = module.DeviceCommunicatorBase(
        object(), unique_name="ep:0"
    )
    assert communicator.use_all2all is False


def test_base_pcp_ep_tolerates_missing_parallel_config(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True

    _install_vllm_config(monkeypatch, SimpleNamespace(parallel_config=None))
    communicator = module.DeviceCommunicatorBase(
        object(), unique_name="ep:0"
    )
    assert communicator.use_all2all is False


# ---------------------------------------------------------------------------
# Contract tests: idempotency, signature validation
# ---------------------------------------------------------------------------


def test_base_pcp_ep_apply_is_idempotent():
    module = _fake_base_communicator_module()
    assert patch_base_communicator_pcp.apply_to_module(module) is True
    # Second call must report no-op via the marker path.
    assert patch_base_communicator_pcp.apply_to_module(module) is False


def test_base_pcp_ep_rejects_incompatible_init_signature():
    from vllm_hcu.patch.worker.framework_opt._common import (
        PatchCompatibilityError,
    )

    class DeviceCommunicatorBase:
        def __init__(self, cpu_group):  # missing keyword args
            self.cpu_group = cpu_group

    module = _module(
        patch_base_communicator_pcp.TARGET_MODULE,
        DeviceCommunicatorBase=DeviceCommunicatorBase,
    )
    with pytest.raises(PatchCompatibilityError):
        patch_base_communicator_pcp.apply_to_module(module)


def test_base_pcp_ep_rejects_missing_class():
    from vllm_hcu.patch.worker.framework_opt._common import (
        PatchCompatibilityError,
    )

    module = _module(patch_base_communicator_pcp.TARGET_MODULE)
    with pytest.raises(PatchCompatibilityError):
        patch_base_communicator_pcp.apply_to_module(module)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_base_pcp_ep_registered_as_deepep_gated_callback():
    from vllm_hcu.patch import worker as worker_dispatcher

    names = worker_dispatcher.worker_callback_names()
    assert (
        patch_base_communicator_pcp.PATCH_ID,
        patch_base_communicator_pcp.TARGET_MODULE,
    ) in names

    features = worker_dispatcher._patch_features()
    assert features[patch_base_communicator_pcp.PATCH_ID] == "deepep"


def test_base_pcp_ep_dispatched_before_deep_ep_runtime():
    """Must run before patch_all2all so use_all2all is set first."""
    from vllm_hcu.patch import worker as worker_dispatcher

    order = [
        patch_id
        for patch_id, _target in worker_dispatcher.worker_callback_names()
    ]
    assert (
        patch_base_communicator_pcp.PATCH_ID in order
        and "worker.framework_opt.communicator.deep_ep_runtime" in order
    )
    assert order.index(patch_base_communicator_pcp.PATCH_ID) < order.index(
        "worker.framework_opt.communicator.deep_ep_runtime"
    )


def test_base_pcp_ep_patch_id_matches_adapter():
    """PATCH_ID string must match the one declared inside the adapter."""
    assert (
        patch_base_communicator_pcp.PATCH_ID
        == "worker.framework_opt.communicator.base_pcp_ep"
    )
    assert (
        patch_base_communicator_pcp.TARGET_MODULE
        == "vllm.distributed.device_communicators.base_device_communicator"
    )
    assert patch_base_communicator_pcp.TARGETS == (
        f"{patch_base_communicator_pcp.TARGET_MODULE}"
        ".DeviceCommunicatorBase.__init__",
    )
