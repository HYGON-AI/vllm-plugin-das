
"""
Patch for vllm.model_executor.layers.fused_moe.router.router_factory
Use HCU GroupedTopKRouter when building grouped-topk MoE routers.
"""

PATCHES = [
    (
        """        grouped_topk_router = """,
        """        from vllm_hcu.ops.fuse_moe_gate import HcuGroupedTopKRouter as GroupedTopKRouter
        grouped_topk_router = """,
    ),
]
