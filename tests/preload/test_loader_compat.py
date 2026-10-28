# SPDX-License-Identifier: Apache-2.0
import time
from types import SimpleNamespace

import pytest

from vllm_hcu.runtime_compat.weight_cache import (
    install_loader_compat, install_uuid_daemon,
)


class Unavailable(Exception):
    pass


class Mismatch(Exception):
    pass


def module_with_loader():
    class Loader:
        @staticmethod
        def _check_supported(vllm_config, model_config):
            return "checked"
        def _resolve_socket_path(self):
            raise AssertionError("old resolution")
        def _connect(self, timeout):
            self.connect_timeouts.append(timeout)
            return None
        def _apply_entries(self, model, entries, aliases, device_index):
            return "applied"
        def _fetch_entries(self, config):
            self._connect(self.state_timeout_s)
            self.calls += 1
            if self.calls <= self.failures:
                raise self.error("not ready")
            return "ok"

    torch = SimpleNamespace(
        accelerator=SimpleNamespace(current_device_index=lambda: 0),
        cuda=SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(uuid="GPU-123")),
    )
    return SimpleNamespace(IpcModelLoader=Loader, torch=torch,
                           get_socket_path=lambda gpu, directory: f"{directory}/{gpu}",
                           WeightCacheUnavailableError=Unavailable,
                           CacheConfigMismatchError=Mismatch,
                           _materialize_remaining_meta_tensors=lambda model, device: "materialized")


def test_uuid_resolution_and_explicit_socket():
    module = module_with_loader()
    install_loader_compat(module)
    loader = module.IpcModelLoader()
    loader.socket_path, loader.socket_dir = None, "/cache"
    assert loader._resolve_socket_path() == "/cache/uuid-GPU-123"
    loader.socket_path = "/manual/socket"
    assert loader._resolve_socket_path() == "/manual/socket"
    loader.socket_path = None
    module.torch.cuda.get_device_properties = lambda index: SimpleNamespace(uuid="../bad")
    with pytest.raises(Unavailable, match="UUID"):
        loader._resolve_socket_path()


def test_wait_only_for_unavailable_strict_loader(monkeypatch):
    module = module_with_loader()
    install_loader_compat(module)
    ticks = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + seconds))
    loader = module.IpcModelLoader()
    loader.calls, loader.failures, loader.error = 0, 2, Unavailable
    loader.connect_timeouts = []
    loader.fallback, loader.state_timeout_s = False, 2
    assert loader._fetch_entries(None) == "ok"
    assert loader.calls == 3
    assert loader.connect_timeouts == [2.0, 1.5, 1.0]
    loader.calls, loader.failures = 0, 99
    with pytest.raises(Unavailable, match="did not become ready"):
        loader._fetch_entries(None)
    loader.calls, loader.error = 0, Mismatch
    with pytest.raises(Mismatch):
        loader._fetch_entries(None)
    loader.calls, loader.error, loader.fallback = 0, Unavailable, True
    with pytest.raises(Unavailable, match="not ready"):
        loader._fetch_entries(None)
    assert loader.calls == 1


def test_strict_fetch_clears_deadline_after_failure(monkeypatch):
    module = module_with_loader()
    install_loader_compat(module)
    ticks = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + seconds))
    loader = module.IpcModelLoader()
    loader.calls, loader.failures, loader.error = 0, 99, Unavailable
    loader.connect_timeouts = []
    loader.fallback, loader.state_timeout_s = False, 1
    with pytest.raises(Unavailable, match="did not become ready"):
        loader._fetch_entries(None)
    assert not hasattr(loader, "_hcu_fetch_deadline")
    loader.fallback = True
    loader.error = Unavailable
    loader.calls = 0
    loader.failures = 0
    loader._fetch_entries(None)
    assert loader.connect_timeouts[-1] == 1


def test_daemon_uuid_matches_loader():
    module = module_with_loader()
    class Daemon:
        def _socket_path(self):
            pass
    module.WeightCacheDaemon = Daemon
    install_uuid_daemon(module)
    daemon = Daemon()
    daemon.socket_dir = "/cache"
    assert daemon._socket_path() == "/cache/uuid-GPU-123"


def test_full_model_launcher_rejects_without_explicit_opt_in(monkeypatch):
    import sys
    from vllm_hcu.preload import daemon
    monkeypatch.setattr(sys, "argv", ["vllm-hcu-preload", "--model", "example"])
    monkeypatch.setattr(daemon, "_prepare", lambda: pytest.fail("must reject before imports"))
    with pytest.raises(SystemExit, match="FULL model"):
        daemon.main()


def test_checkpoint_fingerprint_changes_with_shard(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str
        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("header-hash", "vllm")

    path = tmp_path / "part.safetensors"
    path.write_bytes(b"one")
    module = SimpleNamespace(WeightCacheKey=Key)
    install_key_fingerprint(module)
    config = SimpleNamespace(model=str(tmp_path))
    first = Key.from_model_config(config, 1, 0)
    path.write_bytes(b"new data")
    second = Key.from_model_config(config, 1, 0)
    assert first.checkpoint != second.checkpoint
    assert "|hcu:" in first.vllm_version


def test_launcher_installs_spawn_target_before_main(monkeypatch):
    import sys
    from vllm_hcu.preload import daemon as launcher
    called = []
    upstream = SimpleNamespace(_run_daemon=lambda *args: None)
    upstream.main = lambda: called.append(upstream._run_daemon)
    monkeypatch.setattr(launcher, "_prepare", lambda: upstream)
    monkeypatch.setattr(sys, "argv", ["vllm-hcu-preload", "--enable-full-model-preload", "--model", "x"])
    launcher.main()
    assert called == [launcher.run_daemon]
    assert "--enable-full-model-preload" not in sys.argv


def test_spawn_target_prepares_fresh_worker(monkeypatch):
    from vllm_hcu.preload import daemon as launcher
    calls = []
    upstream = SimpleNamespace(_run_daemon=lambda *args: calls.append(args))
    monkeypatch.setattr(launcher, "_prepare", lambda: upstream)
    launcher.run_daemon(0, "config")
    assert calls == [(0, "config")]


def test_checkpoint_fingerprint_captures_nested_config_and_overrides(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str
        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("same-header", "vllm")

    class HFConfig:
        def __init__(self):
            self.weight_bits = 4
        def to_dict(self):
            return {"text_config": {"compression_config": {"bits": self.weight_bits}}}

    module = SimpleNamespace(WeightCacheKey=Key)
    install_key_fingerprint(module)
    config = SimpleNamespace(model=str(tmp_path), hf_config=HFConfig(), model_class_overrides={})
    first = Key.from_model_config(config, 1, 0)
    config.hf_config.weight_bits = 8
    second = Key.from_model_config(config, 1, 0)
    config.hf_config.weight_bits = 4
    config.model_class_overrides = {"architecture": "OtherModel"}
    third = Key.from_model_config(config, 1, 0)
    assert len({first.checkpoint, second.checkpoint, third.checkpoint}) == 3


def test_incomplete_meta_parameters_fail_closed():
    import torch
    module = module_with_loader()
    install_loader_compat(module)
    with torch.device("meta"):
        layer = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="missing model tensors"):
        module._materialize_remaining_meta_tensors(layer, torch.device("cpu"))
    layer = torch.nn.Linear(2, 2)
    assert module._materialize_remaining_meta_tensors(layer, torch.device("cpu")) == "materialized"


def test_loader_rejects_non_cuda_platform(monkeypatch):
    import vllm.platforms
    module = module_with_loader()
    install_loader_compat(module)
    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace(is_cuda_alike=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA/ROCm"):
        module.IpcModelLoader._check_supported(None, None)
    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    assert module.IpcModelLoader._check_supported(None, None) == "checked"


def test_incomplete_meta_buffers_fail_closed():
    import torch
    module = module_with_loader()
    install_loader_compat(module)
    layer = torch.nn.Module()
    layer.register_buffer("cache", torch.empty(3, device="meta"), persistent=False)
    with pytest.raises(RuntimeError, match="missing model tensors: cache"):
        module._materialize_remaining_meta_tensors(layer, torch.device("cpu"))
