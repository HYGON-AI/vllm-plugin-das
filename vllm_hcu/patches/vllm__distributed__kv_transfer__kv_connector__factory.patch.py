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
        kv_cache_config: "KVCacheConfig | None" = None,
""",
"""
        kv_cache_config: "KVCacheConfig | None" = None,
        dp_rank: int = -1,
""",  
    ),
    (
"""
        if compat_sig:
            # Old signature: __init__(self, vllm_config, role)
            return connector_cls(config, role)
""",
"""
        if compat_sig:
            # Old signature: __init__(self, vllm_config, role)
            return connector_cls(config, role)
        elif henvs.VLLM_HCU_USE_DP_CONNECTOR:
            return connector_cls(config, role, kv_cache_config, dp_rank)
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
