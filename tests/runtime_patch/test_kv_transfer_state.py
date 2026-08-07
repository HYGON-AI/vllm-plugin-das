# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Model-free KV-transfer and Mooncake scheduler state contracts."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from types import SimpleNamespace

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest

from tests.fixtures.kv_transfer import KVTransferTopology, TransferScope
from vllm_hcu.patch.platform.framework_opt import patch_mooncake_connector


@pytest.fixture()
def mooncake_module(monkeypatch: pytest.MonkeyPatch):
    fake_hcu_platform = ModuleType("vllm_hcu.platforms.hcu")
    fake_hcu_platform.get_hcu_flash_attn_mode = lambda: "classic"
    monkeypatch.setitem(sys.modules, "vllm_hcu.platforms.hcu", fake_hcu_platform)
    sys.modules.pop(patch_mooncake_connector.TARGET_MODULE, None)
    module = importlib.import_module(patch_mooncake_connector.TARGET_MODULE)
    patch_mooncake_connector.apply_to_module(module)
    return module


class _Blocks:
    def __init__(self, *groups: list[int]) -> None:
        self._groups = groups

    def get_unhashed_block_ids_all_groups(self) -> tuple[list[int], ...]:
        return self._groups


def _scheduler(mooncake_module, *, role: str):
    scheduler = object.__new__(mooncake_module.MooncakeConnectorScheduler)
    scheduler.is_kv_producer = role == "producer"
    scheduler.is_kv_consumer = role == "consumer"
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_not_processed = set()
    scheduler._is_hma_required = False
    scheduler._has_mamba = False
    scheduler.blocks_per_sw = []
    return scheduler


def _request(request_id: str, **kv_transfer_params: object) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[10, 11, 12, 13, 14],
        kv_transfer_params=kv_transfer_params,
        status=None,
    )


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        (KVTransferTopology(TransferScope.LOCAL_PROCESS, 1, 1), 2),
        (KVTransferTopology(TransferScope.LOCAL_NODE, 2, 4), 6),
        (KVTransferTopology(TransferScope.MULTI_NODE, 4, 8), 8),
    ],
)
def test_kv_transfer_topology_declares_local_device_budget(
    topology: KVTransferTopology,
    expected: int,
) -> None:
    assert topology.minimum_local_devices == expected


def test_mooncake_metadata_separates_recv_send_and_groups_recv_by_engine(
    mooncake_module,
) -> None:
    metadata = mooncake_module.MooncakeConnectorMetadata()

    metadata.add_new_req(
        "decoder-1",
        [[1, 2], [3]],
        {
            "transfer_id": "xfer-1",
            "remote_engine_id": "prefill-a",
            "remote_bootstrap_addr": "127.0.0.1:8800",
        },
    )
    metadata.add_new_req(
        "decoder-2",
        [[4]],
        {
            "transfer_id": "xfer-2",
            "remote_engine_id": "prefill-a",
            "remote_bootstrap_addr": "127.0.0.1:8800",
        },
    )
    metadata.add_new_req(
        "prefill-1",
        [[7, 8]],
        {"transfer_id": "xfer-3"},
        load_remote_cache=False,
    )

    assert sorted(metadata.reqs_to_recv) == ["prefill-a"]
    assert set(metadata.reqs_to_recv["prefill-a"]) == {"decoder-1", "decoder-2"}
    assert metadata.reqs_to_recv["prefill-a"]["decoder-1"].transfer_id == "xfer-1"
    assert metadata.reqs_to_send == {"prefill-1": ("xfer-3", [[7, 8]])}


def test_mooncake_remote_prefill_state_moves_to_metadata_once(
    mooncake_module,
) -> None:
    scheduler = _scheduler(mooncake_module, role="consumer")
    request = _request(
        "decoder",
        do_remote_prefill=True,
        transfer_id="xfer-r",
        remote_engine_id="prefill-a",
        remote_bootstrap_addr="127.0.0.1:8800",
    )

    assert scheduler.get_num_new_matched_tokens(request, 2) == (3, True)
    scheduler.update_state_after_alloc(request, _Blocks([1, 2], [3]), 3)

    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert set(scheduler._reqs_need_recv) == {"decoder"}

    metadata = scheduler.build_connector_meta(SimpleNamespace())
    pull = metadata.reqs_to_recv["prefill-a"]["decoder"]

    assert pull.transfer_id == "xfer-r"
    assert pull.local_block_ids == [[1, 2], [3]]
    assert scheduler._reqs_need_recv == {}
    assert scheduler.build_connector_meta(SimpleNamespace()).reqs_to_recv == {}


def test_mooncake_decode_finish_delays_free_until_send_metadata_is_built(
    mooncake_module,
) -> None:
    scheduler = _scheduler(mooncake_module, role="producer")
    request = _request(
        "prefill",
        do_remote_decode=True,
        transfer_id="xfer-d",
    )
    request.status = mooncake_module.RequestStatus.FINISHED_LENGTH_CAPPED

    delay_free, params = scheduler.request_finished(request, ([5, 6], []))

    assert delay_free is True
    assert params is None
    assert set(scheduler._reqs_need_send) == {"prefill"}

    metadata = scheduler.build_connector_meta(SimpleNamespace())
    assert metadata.reqs_to_send == {"prefill": ("xfer-d", [[5, 6], []])}
    assert scheduler._reqs_need_send == {}


def test_mooncake_decode_abort_records_not_processed_transfer_id(
    mooncake_module,
) -> None:
    scheduler = _scheduler(mooncake_module, role="producer")
    request = _request(
        "prefill-abort",
        do_remote_decode=True,
        transfer_id="xfer-abort",
    )
    request.status = mooncake_module.RequestStatus.FINISHED_ABORTED

    delay_free, params = scheduler.request_finished(request, ([5],))
    metadata = scheduler.build_connector_meta(SimpleNamespace())

    assert delay_free is False
    assert params is None
    assert metadata.reqs_to_send == {}
    assert metadata.reqs_not_processed == {"xfer-abort"}
    assert scheduler._reqs_not_processed == set()
