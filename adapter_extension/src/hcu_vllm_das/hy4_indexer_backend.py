# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4-owned metadata builder over the native Indexer cache/backend ABI."""
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.hy_v4.nvidia.attention import Indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend, DeepseekV32IndexerMetadataBuilder,
)

from .hy4_constructor import bind_constructor
from .hy4_indexer_workspace import coalesce_indexer_prefill


class HcuHY4WorkspaceIndexerBuilder(DeepseekV32IndexerMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        return coalesce_indexer_prefill(self, common_attn_metadata, metadata)


class HcuHY4WorkspaceIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_builder_cls():
        return HcuHY4WorkspaceIndexerBuilder


class HcuHY4WorkspaceIndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self):
        return HcuHY4WorkspaceIndexerBackend


class HcuHY4WorkspaceIndexer(Indexer):
    __init__ = bind_constructor(Indexer.__init__,
        {'DeepseekV32IndexerCache': HcuHY4WorkspaceIndexerCache})
