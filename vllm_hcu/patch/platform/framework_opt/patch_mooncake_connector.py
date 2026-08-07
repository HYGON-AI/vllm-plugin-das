# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contract validation for the HCU-owned Mooncake connector."""

from __future__ import annotations

from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_signature_prefix

TARGET_MODULE = (
    "vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
)
PATCH_ID = "platform.framework_opt.hcu_mooncake_connector"
TARGETS = (
    f"{TARGET_MODULE}.MooncakeConnector",
    f"{TARGET_MODULE}.MooncakeConnectorScheduler",
    f"{TARGET_MODULE}.MooncakeConnectorWorker",
    f"{TARGET_MODULE}.MooncakeXferMetadata",
)
_MARKER = "_vllm_hcu_mooncake_contract_validated"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    for name in (
        "MooncakeConnector",
        "MooncakeConnectorScheduler",
        "MooncakeConnectorWorker",
        "MooncakeXferMetadata",
    ):
        require_class(target, name, f"{TARGET_MODULE}.{name}")
    helper_contracts = {
        "transfer_id_from_req": ("req_id", "kv_params"),
        "log_ttft_event": ("event",),
        "_get_tp_ratio": ("local_tp_size", "remote_tp_size"),
        "_expand_transfer_regions": (
            "base_addrs",
            "block_lens",
            "kv_block_lens",
            "layer_names",
            "layer_indices",
            "is_kv_layout_blocks_first",
        ),
        "_validate_phase1_metadata": (),
        "should_launch_bootstrap_server": ("vllm_config",),
    }
    for name, prefix in helper_contracts.items():
        function = require_callable(target, name, f"{TARGET_MODULE}.{name}")
        require_signature_prefix(function, f"{TARGET_MODULE}.{name}", prefix)
    worker = target.MooncakeConnectorWorker
    for name in (
        "register_kv_caches",
        "receive_kv_from_single_worker",
        "process_pulling_result",
        "_fail_pull_metas",
        "receive_kv",
        "record_send_reqs",
        "_get_transfer_regions",
        "_get_sender_transfer_plan",
    ):
        require_callable(worker, name, f"{TARGET_MODULE}.MooncakeConnectorWorker.{name}")
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
