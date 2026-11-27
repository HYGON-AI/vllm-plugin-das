# SPDX-License-Identifier: Apache-2.0
"""Cache eager PCP forward allocations in stream-ordered HIP segments."""
from __future__ import annotations

from functools import wraps
from math import prod


def _reserve_attention_workspace(runner):
    import torch
    from vllm.v1.worker.workspace import current_workspace_manager

    # Core grows this workspace in its dummy forward. A native allocator trim
    # between persistent collectives may block a rank before its peers arrive.
    # Reserve the largest of Core's own requests before launching any layer.
    largest_bytes = 0
    largest_shapes = ()
    for attn in runner.model.modules():
        if not hasattr(attn, "_prefill_workspace_shapes"):
            continue
        rows = attn.max_num_batched_tokens
        ratio = attn.compress_ratio
        compressed = (attn.max_model_len + ratio - 1) // ratio if ratio else 0
        q = torch.empty((0, attn.head_dim), device="meta")
        shapes = attn._prefill_workspace_shapes(
            compressed + attn.window_size + rows, rows, q,
            gather=not attn._use_aiter_sparse_mla,
        )
        size = sum((prod(shape) * dtype.itemsize + 255) // 256 * 256
                   for shape, dtype in shapes)
        if size > largest_bytes:
            largest_bytes, largest_shapes = size, shapes
    if largest_shapes:
        current_workspace_manager().get_simultaneous(*largest_shapes)


def configure_forward_pool(context):
    cfg = context.vllm_config
    if (cfg.model_config.quantization != "hcu_dsv41_ct_w4a8"
            or cfg.kernel_config.moe_backend != "deep_gemm_mega_moe"):
        return
    import torch
    from deepgemm.mega import _C
    from vllm.logger import init_logger

    runner = context.worker.model_runner
    if getattr(runner, "_hcu_megamoe_forward_pool_enabled", False):
        return
    if torch.cuda.get_allocator_backend() != "native":
        raise RuntimeError("DSV4.1 cached HIP segments require the native PyTorch allocator")
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        _C.__file__, "dsv41_hip_segment_alloc", "dsv41_hip_segment_free",
    )

    def new_pool():
        return torch.cuda.MemPool(allocator.allocator(), use_on_oom=True)

    pool = new_pool()
    execute = runner.execute_model
    profile = runner.profile_run

    @wraps(execute)
    def execute_in_pool(*args, **kwargs):
        with torch.cuda.use_mem_pool(pool):
            return execute(*args, **kwargs)

    @wraps(profile)
    def profile_and_release(*args, **kwargs):
        nonlocal pool
        _reserve_attention_workspace(runner)
        result = profile(*args, **kwargs)
        # Profile and inference have very different allocation sizes. Release
        # the dummy batch's segments before Core allocates the real KV cache.
        torch.cuda.synchronize()
        pool = new_pool()
        return result

    runner.execute_model = execute_in_pool
    runner.profile_run = profile_and_release
    runner._hcu_megamoe_forward_pool_enabled = True
    init_logger(f"vllm.{__name__}").info_once("Enabled DSV4.1 MegaMoE forward memory pool")
