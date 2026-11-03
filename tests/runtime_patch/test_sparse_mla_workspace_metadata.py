# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_hcu.v1.attention.backends.mla.workspace_dedup import build_workspace_metadata
from vllm_hcu.v1.workspace_block_tables import (
    HOST_TABLE_ATTRIBUTE,
    attach_workspace_block_tables,
    install_workspace_block_table_mirrors,
)


class FakeBlockTables:
    def __init__(self):
        self.block_tables = [
            SimpleNamespace(gpu=torch.zeros((4, 8))),
            SimpleNamespace(gpu=torch.zeros((4, 8))),
        ]
        self.blocks_per_kv_block = [2, 1]
        self.num_blocks = SimpleNamespace(np=np.zeros((2, 4), dtype=np.int32))

    def append_block_ids(self, req_index, new_block_ids, overwrite):
        for group, ids in enumerate(new_block_ids):
            if overwrite:
                self.num_blocks.np[group, req_index] = 0
            self.num_blocks.np[group, req_index] += (
                len(ids) * self.blocks_per_kv_block[group]
            )


def test_canonical_mirror_reuse_and_pcp_snapshots():
    tables = FakeBlockTables()
    install_workspace_block_table_mirrors(tables)
    install_workspace_block_table_mirrors(tables)
    tables.append_block_ids(2, ([5, 7], [6]), True)
    tables.append_block_ids(2, ([9], [8]), False)
    tensors = (torch.zeros((4, 8)), torch.zeros((4, 8)))
    batch = SimpleNamespace(
        idx_mapping_np=np.array([2, 2, 0]),
        is_prefilling_np=np.array([True, True, False]),
    )
    attach_workspace_block_tables(tables, tensors, batch)
    snapshot = getattr(tensors[0], HOST_TABLE_ATTRIBUTE)
    assert snapshot.tolist() == [[10, 11, 14, 15, 18, 19, 0, 0]] * 2 + [[0] * 8] * 2
    assert getattr(tensors[1], HOST_TABLE_ATTRIBUTE)[0].tolist() == [6, 8] + [0] * 6
    tables.append_block_ids(2, ([1], [3]), True)
    new_tensors = (torch.zeros((3, 8)), torch.zeros((3, 8)))
    attach_workspace_block_tables(tables, new_tensors, batch)
    assert getattr(new_tensors[0], HOST_TABLE_ATTRIBUTE)[0].tolist() == [2, 3] + [0] * 6
    assert snapshot[0].tolist() == [10, 11, 14, 15, 18, 19, 0, 0]
    decode = SimpleNamespace(is_prefilling_np=np.array([False]))
    untouched = (torch.zeros((1, 8)), torch.zeros((1, 8)))
    assert attach_workspace_block_tables(tables, untouched, decode) is untouched
    assert not hasattr(untouched[0], HOST_TABLE_ATTRIBUTE)


def test_mirror_must_precede_allocation():
    tables = FakeBlockTables()
    tables.num_blocks.np[0, 0] = 1
    with pytest.raises(RuntimeError, match="before block allocation"):
        install_workspace_block_table_mirrors(tables)


def make_metadata(decode=False, factor=2):
    table = np.array([[5, 1, 7], [5, 1, 8], [5, 1, 7]], dtype=np.int32)
    lengths = [130, 150, 190]
    widths = [2, 2, 2]
    if decode:
        table = np.vstack(([6, 2, 0], table)).astype(np.int32)
        lengths.insert(0, 70)
        widths.insert(0, 1)
    common = SimpleNamespace(
        block_table_tensor=torch.from_numpy(table),
        sparse_mla_workspace_block_table_cpu=np.pad(table, ((0, 2), (0, 0))),
        seq_lens_cpu_upper_bound=torch.tensor(lengths),
        query_start_loc_cpu=torch.tensor(np.cumsum([0] + widths)),
    )
    extra = SimpleNamespace(
        prefill=SimpleNamespace(),
        num_decodes=int(decode),
        num_prefills=3,
        num_prefill_tokens=6,
    )
    metadata = SimpleNamespace(fp8_extra_metadata=extra, fp8_use_mixed_batch=False)
    builder = SimpleNamespace(
        device=torch.device("cpu"),
        kv_cache_spec=SimpleNamespace(block_size=64),
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(
                sparse_mla_prefill_workspace_factor=factor
            ),
            model_config=SimpleNamespace(max_model_len=190),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
    )
    return builder, common, metadata


@pytest.mark.parametrize("decode", [False, True])
def test_metadata_merges_rows_without_reordering_queries(decode):
    builder, common, metadata = make_metadata(decode)
    plan = build_workspace_metadata(builder, common, metadata)
    assert len(plan.chunks) == 1
    assert plan.chunks[0].num_tokens == 212
    assert plan.chunks[0].tokens_slice == slice(int(decode), 6 + int(decode))
    assert plan.chunks[0].descriptors.tolist() == [
        [5, 64, 0],
        [1, 64, 64],
        [7, 62, 128],
        [8, 22, 190],
    ]
    assert plan.block_starts.tolist() == [[0, 64, 128], [0, 64, 190], [0, 64, 128]]


def test_capture_decode_and_mixed_path_fall_back():
    builder, common, metadata = make_metadata()
    common.sparse_mla_workspace_block_table_cpu = None
    assert build_workspace_metadata(builder, common, metadata) is None
    setattr(
        common.block_table_tensor,
        HOST_TABLE_ATTRIBUTE,
        np.array([[5, 1, 7], [5, 1, 8], [5, 1, 7]], dtype=np.int32),
    )
    assert build_workspace_metadata(builder, common, metadata) is not None
    metadata.fp8_use_mixed_batch = True
    assert build_workspace_metadata(builder, common, metadata) is None
    metadata.fp8_use_mixed_batch = False
    metadata.fp8_extra_metadata.prefill = None
    assert build_workspace_metadata(builder, common, metadata) is None


def test_backend_builder_is_opt_in_and_dcp_scoped(monkeypatch):
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadataBuilder,
    )
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseBackend,
        HcuFlashMLASparseMetadataBuilder,
    )

    assert (
        HcuFlashMLASparseBackend.get_builder_cls() is HcuFlashMLASparseMetadataBuilder
    )
    stub, common, metadata = make_metadata()
    monkeypatch.setattr(FlashMLASparseMetadataBuilder, "build", lambda *args: metadata)
    builder = HcuFlashMLASparseMetadataBuilder.__new__(HcuFlashMLASparseMetadataBuilder)
    builder.__dict__.update(vars(stub))
    monkeypatch.delenv("VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP", raising=False)
    assert not hasattr(builder.build(0, common), "hcu_workspace_dedup")
    monkeypatch.setenv("VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP", "1")
    assert builder.build(0, common).hcu_workspace_dedup is not None
    metadata = SimpleNamespace(fp8_extra_metadata=None, fp8_use_mixed_batch=False)
    builder.vllm_config.parallel_config.decode_context_parallel_size = 2
    assert not hasattr(builder.build(0, common), "hcu_workspace_dedup")
