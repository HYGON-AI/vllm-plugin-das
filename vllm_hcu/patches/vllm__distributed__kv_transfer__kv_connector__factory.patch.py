# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.attention.attention unified_kv_cache_update
"""

PATCHES = [
    (
"""
import importlib
""",
"""
import importlib
from vllm_hcu.platforms import envs as henvs
""",       
    ),
    (
"""
        kv_cache_config: "KVCacheConfig",
""",
"""
        kv_cache_config: "KVCacheConfig",
        dp_rank: int = -1,
""",  
    ),
    (
"""
        return connector_cls(config, role, kv_cache_config)
""",
"""  
        if henvs.VLLM_HCU_USE_DP_CONNECTOR:
            return connector_cls(config, role, kv_cache_config, dp_rank)
        else:
            return connector_cls(config, role, kv_cache_config)
""",        
    ),
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

KVConnectorFactory.register_connector(
    "DuSwiftConnectorDp",
    "vllm_hcu.distributed.kv_transfer.kv_connector.v1.du.du_swift_connector_dp",
    "DuSwiftConnectorDp",
)
""",
    )
]
