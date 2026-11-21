# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Owned PCP metadata builder; Adapter remains the sparse kernel ABI owner."""
from hcu_adapter.kernels.vllm_sparse_flashmla_dtk import HcuSparseFlashMLABackend
from vllm.logger import init_logger

from .hy4_prefill_workspace import coalesce_prefill

logger = init_logger(__name__)


class HcuHY4PCPWorkspaceBuilder(HcuSparseFlashMLABackend.get_builder_cls()):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        import torch
        properties = torch.cuda.get_device_properties(device)
        self._hcu_workspace_l2_bytes = max(0, int(getattr(properties, 'L2_cache_size', 0) or 0))
        logger.info_once('HY4 PCP request-segment KV workspace enabled; L2=%d bytes',
                         self._hcu_workspace_l2_bytes)

    def _build_fp8_separate_prefill_decode(self, common_attn_metadata, metadata):
        native = super()._build_fp8_separate_prefill_decode(common_attn_metadata, metadata)
        return coalesce_prefill(self, common_attn_metadata, native,
                                num_decode_rows=metadata.num_decodes)


class HcuHY4PCPWorkspaceBackend(HcuSparseFlashMLABackend):
    @staticmethod
    def get_builder_cls():
        return HcuHY4PCPWorkspaceBuilder
