# SPDX-License-Identifier: Apache-2.0
"""Resolve Adapter-owned APIs in either supported directory layout.

Only the infrastructure namespace selects the layout. Missing dependencies or
missing owners inside an installed new layout propagate without legacy retry.
Importing this helper does not import an Adapter provider, Torch or a host.
"""
import importlib

_RELOCATED = {
    "runtime_lifecycle": "hcu_adapter.infra.runtime_lifecycle",
    "hcu_env": "hcu_adapter.infra.hcu_env",
    "registry": "hcu_adapter.infra.registry",
    "hcu_config": "hcu_adapter.infra.hcu_config",
    "kill_switch": "hcu_adapter.infra.kill_switch",
    "dsv41_activation": "hcu_adapter.features.dsv41.dsv41_activation",
    "ct_w4a8_registration": "hcu_adapter.features.ct_w4a8.ct_w4a8_registration",
    "v41_fp8_indexer": "hcu_adapter.features.dsv41.v41_fp8_indexer",
    "dsv41_channel_fp8": "hcu_adapter.features.dsv41.dsv41_channel_fp8",
    "fp4_indexer": "hcu_adapter.features.dsv41.fp4_indexer",
    "v41_fp8_indexer_triton": "hcu_adapter.features.dsv41.v41_fp8_indexer_triton",
    "dsv41_dspark_registration": "hcu_adapter.features.dsv41.dsv41_dspark_registration",
    "dsv41_draft_weight_alias": "hcu_adapter.features.dsv41.dsv41_draft_weight_alias",
    "dsv41_int8_linear": "hcu_adapter.features.dsv41.dsv41_int8_linear",
    "ct_w4a8_masked_activation": "hcu_adapter.features.ct_w4a8.masked_activation",
    "channel_fp8_empty_linear": "hcu_adapter.features.ct_w4a8.channel_fp8_empty_linear",
    "ct_w4a8_packed_loader": "hcu_adapter.features.ct_w4a8.ct_w4a8_packed_loader"
}


def adapter_module_name(name):
    legacy = "hcu_adapter." + name
    canonical = _RELOCATED.get(name)
    if canonical is None:
        return legacy
    try:
        importlib.import_module("hcu_adapter.infra")
    except ModuleNotFoundError as error:
        if error.name != "hcu_adapter.infra":
            raise
        return legacy
    return canonical


def adapter_module(name):
    """Return the actual owner module, preserving its identity and state."""
    return importlib.import_module(adapter_module_name(name))
