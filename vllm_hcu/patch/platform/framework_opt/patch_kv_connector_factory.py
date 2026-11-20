# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register HCU KV connectors and add the optional DP-rank constructor path."""

from __future__ import annotations

import importlib
from types import ModuleType

from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)

TARGET_MODULE = "vllm.distributed.kv_transfer.kv_connector.factory"
PATCH_ID = "platform.framework_opt.kv_connector_factory"
TARGETS = (
    f"{TARGET_MODULE}.KVConnectorFactory.create_connector",
    f"{TARGET_MODULE}.KVConnectorFactory._registry[MooncakeConnector]",
    f"{TARGET_MODULE}.KVConnectorFactory._registry[MooncakeStoreConnector]",
    f"{TARGET_MODULE}.KVConnectorFactory._registry[DuSwiftConnector]",
    f"{TARGET_MODULE}.KVConnectorFactory._registry[DuSwiftConnectorDp]",
)
_MARKER = "_vllm_hcu_kv_connector_factory_applied"
_WRAPPER = "_vllm_hcu_kv_connector_factory_wrapper"
_HCU_CONNECTORS = {
    "MooncakeStoreConnector": (
        "vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake.store_connector",
        "HcuMooncakeStoreConnector",
    ),
    # MultiConnector metrics aggregation resolves the child class name, while
    # CLI config resolves the upstream connector name.
    "HcuMooncakeStoreConnector": (
        "vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake.store_connector",
        "HcuMooncakeStoreConnector",
    ),
    "MooncakeConnector": (
        "vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
        "MooncakeConnector",
    ),
    "DuSwiftConnector": (
        "vllm_hcu.distributed.kv_transfer.kv_connector.v1.du.du_swift_connector",
        "DuSwiftConnector",
    ),
    "DuSwiftConnectorDp": (
        "vllm_hcu.distributed.kv_transfer.kv_connector.v1.du.du_swift_connector_dp",
        "DuSwiftConnectorDp",
    ),
}


def _lazy_loader(module_path: str, class_name: str):
    def load():
        module = importlib.import_module(module_path)
        connector = getattr(module, class_name, None)
        if not isinstance(connector, type):
            raise RuntimeError(
                f"registered HCU connector {module_path}.{class_name} is missing"
            )
        return connector

    load._hcu_connector_target = (module_path, class_name)  # type: ignore[attr-defined]
    return load


def _patch_multi_connector_prom_metrics() -> None:
    """Ignore child stats for connectors that do not expose Prometheus metrics."""
    module = importlib.import_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    )
    metrics_cls = getattr(module, "MultiKVConnectorPromMetrics", None)
    if not isinstance(metrics_cls, type):
        raise PatchCompatibilityError(
            "required MultiConnector metrics class is missing"
        )
    marker = "_vllm_hcu_multi_connector_metrics_filter_applied"
    if getattr(metrics_cls, marker, False):
        return
    original_observe = getattr(metrics_cls, "observe", None)
    if not callable(original_observe):
        raise PatchCompatibilityError(
            "required MultiConnector metrics observe method is missing"
        )

    def hcu_observe(self, transfer_stats_data, engine_idx: int = 0):
        registered_stats = {
            connector_id: stats_data
            for connector_id, stats_data in transfer_stats_data.items()
            if connector_id in self._prom_metrics
        }
        return original_observe(self, registered_stats, engine_idx)

    hcu_observe.__name__ = original_observe.__name__
    hcu_observe.__qualname__ = original_observe.__qualname__
    hcu_observe.__doc__ = original_observe.__doc__
    metrics_cls.observe = hcu_observe
    setattr(metrics_cls, marker, True)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    factory = require_class(target, "KVConnectorFactory", f"{TARGET_MODULE}.KVConnectorFactory")
    registry = getattr(factory, "_registry", None)
    if not isinstance(registry, dict):
        raise PatchCompatibilityError("KVConnectorFactory._registry must be a dict")
    create_connector = require_callable(factory, "create_connector", TARGETS[0])
    require_callable(
        factory,
        "get_connector_class",
        f"{TARGET_MODULE}.KVConnectorFactory.get_connector_class",
    )
    if not callable(getattr(target, "supports_hma", None)):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.supports_hma is missing"
        )
    _patch_multi_connector_prom_metrics()
    if getattr(factory, _MARKER, False):
        if not getattr(create_connector, _WRAPPER, False):
            raise PatchCompatibilityError(
                "HCU KVConnectorFactory marker is stale; restart the process"
            )
        for name, expected in _HCU_CONNECTORS.items():
            loader = registry.get(name)
            if getattr(loader, "_hcu_connector_target", None) != expected:
                raise PatchCompatibilityError(
                    f"HCU connector registry entry {name!r} changed after installation"
                )
        return False

    require_signature_prefix(
        create_connector,
        TARGETS[0],
        ("config", "role", "kv_cache_config"),
    )
    descriptor = vars(factory).get("create_connector")
    if not isinstance(descriptor, classmethod):
        raise PatchCompatibilityError(f"required target {TARGETS[0]} is not a classmethod")
    original_create = create_connector

    def hcu_create_connector(
        cls,
        config,
        role,
        kv_cache_config,
        dp_rank: int = -1,
    ):
        if (
            henvs.VLLM_HCU_USE_DP_CONNECTOR
            and config.kv_transfer_config is not None
        ):
            connector_cls = cls.get_connector_class(config.kv_transfer_config)
            try:
                from vllm.utils.func_utils import supports_kw
            except ImportError as exc:
                raise RuntimeError("vLLM supports_kw is required for HCU DP connector") from exc
            if supports_kw(connector_cls, "dp_rank"):
                if dp_rank < 0:
                    dp_rank = config.parallel_config.data_parallel_rank
                # Preserve the official HMA validation before construction.
                hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
                if hma_enabled and not target.supports_hma(connector_cls):
                    raise ValueError(
                        f"Connector {connector_cls.__name__} does not support HMA but "
                        "HMA is enabled. Please set --disable-hybrid-kv-cache-manager."
                    )
                return connector_cls(
                    config,
                    role,
                    kv_cache_config,
                    dp_rank=dp_rank,
                )
        return original_create(config, role, kv_cache_config)

    hcu_create_connector.__name__ = descriptor.__func__.__name__
    hcu_create_connector.__qualname__ = descriptor.__func__.__qualname__
    hcu_create_connector.__doc__ = descriptor.__func__.__doc__
    setattr(hcu_create_connector, _WRAPPER, True)
    factory._vllm_hcu_original_create_connector = descriptor
    factory.create_connector = classmethod(hcu_create_connector)
    for name, (module_path, class_name) in _HCU_CONNECTORS.items():
        registry[name] = _lazy_loader(module_path, class_name)
    setattr(factory, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
