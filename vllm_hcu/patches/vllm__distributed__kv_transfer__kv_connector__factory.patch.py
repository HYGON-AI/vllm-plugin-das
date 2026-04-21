# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.attention.attention unified_kv_cache_update
"""

PATCHES = [
    (
"""
KVConnectorFactory.register_connector(
    "P2pNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector",
    "P2pNcclConnector",
)
""",
"""
KVConnectorFactory.register_connector(
    "P2pNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector",
    "P2pNcclConnector",
)

KVConnectorFactory.register_connector(
    "DuSwiftConnector",
    "vllm_hcu.distributed.kv_transfer.kv_connector.v1.du.du_swift_connector",
    "DuSwiftConnector",
)
""",
    )
]