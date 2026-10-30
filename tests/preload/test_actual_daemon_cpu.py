# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed vLLM daemon protocol with CPU tensors only."""
import socket

import torch


def test_installed_daemon_exports_each_client_and_checks_key(monkeypatch):
    from vllm.model_executor.model_loader.weight_cache import daemon, protocol
    from vllm_hcu.runtime_compat.weight_cache import install_per_client_export

    cls = daemon.WeightCacheDaemon
    for attr in ("_export_entries", "_handle_get_state"):
        monkeypatch.setattr(cls, attr, getattr(cls, attr))
    monkeypatch.setattr(cls, "_hcu_per_client_export", False, raising=False)
    install_per_client_export(daemon)
    instance = object.__new__(cls)
    instance.model = torch.nn.Linear(3, 2, bias=False)
    instance.cache_config = protocol.WeightCacheKey(
        checkpoint="x", model_arch="test", tp_size=1, tp_rank=0,
        dtype="float32", quantization=None, quant_config_hash="",
        revision=None, vllm_version="test")
    instance._gpu_uuid = lambda: None
    instance._export_entries()
    export_original = daemon.export_entries
    calls = []
    def export(model):
        calls.append(model)
        return export_original(model)
    monkeypatch.setattr(daemon, "export_entries", export)

    def request(key):
        server, client = socket.socketpair()
        with server, client:
            instance._handle_get_state(server, {"cache_config": key})
            return protocol.recv_msg(client)

    first = request(instance.cache_config)
    second = request(instance.cache_config)
    assert len(calls) == 2
    assert first["status"] == second["status"] == "ok"
    assert torch.equal(first["entries"]["weight"].rebuild(0), instance.model.weight)
    assert torch.equal(second["entries"]["weight"].rebuild(0), instance.model.weight)
    import dataclasses
    assert request(dataclasses.replace(instance.cache_config, checkpoint="different"))["status"] == "mismatch"
    assert len(calls) == 2


def test_expert_layout_marker_survives_real_entry_export_and_load(monkeypatch):
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import daemon, ipc_loader
    from vllm_hcu.runtime_compat.weight_cache import (
        install_expert_export_metadata, install_loader_compat,
    )

    monkeypatch.setattr(daemon, "export_entries", daemon.export_entries)
    install_expert_export_metadata(daemon)
    cls = ipc_loader.IpcModelLoader
    for name in ("_fetch_entries", "_resolve_socket_path", "_check_supported", "_apply_entries"):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    monkeypatch.setattr(cls, "_hcu_weight_cache_loader", False, raising=False)
    monkeypatch.setattr(ipc_loader, "_materialize_remaining_meta_tensors",
                        ipc_loader._materialize_remaining_meta_tensors)
    install_loader_compat(ipc_loader)
    source = torch.nn.Module()
    source.w13_weight = torch.nn.Parameter(torch.ones(1, 2, 4, dtype=torch.int8), requires_grad=False)
    source.w13_weight._hcu_vllm_w4a8_unpacked = True
    entries, aliases = daemon.export_entries(source)
    assert entries["w13_weight"]._hcu_vllm_w4a8_unpacked is True
    target = torch.nn.Module()
    loader = cls(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"fallback": False}))
    loader._apply_entries(target, entries, aliases, 0)
    assert target.w13_weight._hcu_vllm_w4a8_unpacked is True
    assert torch.equal(target.w13_weight, source.w13_weight)
