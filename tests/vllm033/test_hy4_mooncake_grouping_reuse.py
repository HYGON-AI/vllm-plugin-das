# SPDX-License-Identifier: Apache-2.0
"""Execute the complete current Core builder; isolate engine/device setup only."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from hcu_vllm_das import hy4_pcp_mooncake as owned
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as native


def worker(groups=1, ratio=1):
    obj = object.__new__(owned.HY4PCPMooncakeWorker)
    obj.tp_rank, obj.tp_size, obj.use_mla = 0, 1, True
    obj.transfer_topo = NS(local_replicates_kv_cache=False)
    obj._physical_blocks_per_logical_kv_block = ratio
    obj.kv_cache_config = NS(transfer_groups=[NS(kv_cache_spec=object())
                                            for _ in range(groups)],
                             has_mamba_layers=False)
    return obj


def build(obj, source, destination, groups=(0,), *, native_builder=False):
    local = [native.TransferRegion(str(i), i, 1000*(i+1), 8, 8, group)
             for i, group in enumerate(groups)]
    remote = [native.TransferRegion(str(i), i, 10000*(i+1), 8, 8, group)
              for i, group in enumerate(groups)]
    meta = NS(remote_hostname='test', remote_port=1, remote_tp_size=1,
              remote_tp_rank=0, req_blocks={'d': ('t', destination)})
    fn = (native.MooncakeConnectorWorker._build_transfer_params if native_builder
          else owned.HY4PCPMooncakeWorker._build_transfer_params)
    return asyncio.run(fn(obj, [('d', NS(local_block_ids=source))], meta, local, remote))


def test_repeated_regions_group_once_without_changing_descriptors(monkeypatch):
    obj = worker()
    source, destination = [[1, 2, 5, 6]], [[10, 11, 20, 21]]
    counted = Mock(wraps=native.group_concurrent_contiguous)
    monkeypatch.setattr(native, 'group_concurrent_contiguous', counted)
    expected = build(obj, source, destination, (0,)*78, native_builder=True)
    assert counted.call_count == 78
    counted.reset_mock()
    assert build(obj, source, destination, (0,)*78) == expected
    assert counted.call_count == 1
    assert native.group_concurrent_contiguous is counted
    assert source == [[1, 2, 5, 6]] and destination == [[10, 11, 20, 21]]


@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('source,destination', [
    ([[1, 2, 8], [30, 32]], [[10, 11], [40, 41]]),
    ([[1, 2], []], [[20, 21], []]),
    ([[1]], [[2], [3]]),
    ([[1], [2]], [[10, 11], [30]]),
])
def test_current_builder_group_and_error_semantics(source, destination, ratio):
    obj = worker(2, ratio)
    expected = build(obj, source, destination, (0, 1, 0, 1), native_builder=True)
    assert build(obj, source, destination, (0, 1, 0, 1)) == expected


def test_cache_does_not_leak_between_builder_calls(monkeypatch):
    obj = worker()
    counted = Mock(wraps=native.group_concurrent_contiguous)
    monkeypatch.setattr(native, 'group_concurrent_contiguous', counted)
    first = build(obj, [[1, 2]], [[10, 11]], (0, 0))
    second = build(obj, [[30, 31]], [[40, 41]], (0, 0))
    assert counted.call_count == 2
    assert first[0] != second[0] and first[1] != second[1]


def test_disabled_reuse_retains_native_builder(monkeypatch):
    monkeypatch.setattr(owned, 'op_disabled', lambda name: True)
    counted = Mock(wraps=native.group_concurrent_contiguous)
    monkeypatch.setattr(native, 'group_concurrent_contiguous', counted)
    obj = worker()
    result = build(obj, [[1, 2]], [[10, 11]], (0, 0))
    assert counted.call_count == 2
    assert result == build(obj, [[1, 2]], [[10, 11]], (0, 0), native_builder=True)
