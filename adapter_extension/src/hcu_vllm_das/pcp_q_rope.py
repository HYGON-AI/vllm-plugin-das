# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Use the HCU rotary kernel for eager PCP prefill queries."""
from functools import lru_cache, partial
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from hcu_adapter import registry


@lru_cache(maxsize=1)
def load_operator():
    if torch.version.hip is None:
        raise RuntimeError("HCU inverse RoPE requires HIP PyTorch and hipcc")
    from torch.utils.cpp_extension import load, _get_build_directory

    root = Path(__file__).with_name("csrc")
    assets = [(name, (root / name).read_bytes())
              for name in ("inverse_rope.cpp", "inverse_rope.cu")]
    # Match Core's separate FP32 products and sum before the BF16 cast.
    cuda_flags = ["-O3", "-ffp-contract=off"]
    key = hashlib.sha256(
        (str(torch.__version__) + str(torch.version.hip)).encode()
        + repr(cuda_flags).encode()
        + b"".join(payload for _, payload in assets)).hexdigest()[:20]
    build = Path(_get_build_directory("hcu_vllm_das_inverse_rope", False)) / key
    build.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, payload in assets:
        destination = build / name
        if destination.exists():
            if destination.read_bytes() != payload:
                raise RuntimeError("HCU inverse-RoPE source cache is corrupt")
        else:
            # HIPify writes a sibling .hip file. Stage into the build cache so
            # compiling an installed wheel never writes into its package tree.
            with tempfile.TemporaryDirectory(dir=build) as temporary:
                staged = Path(temporary) / name
                staged.write_bytes(payload)
                os.replace(staged, destination)
        sources.append(str(destination))
    load(name="hcu_vllm_das_inverse_rope",
         sources=sources, build_directory=str(build),
         extra_cuda_cflags=cuda_flags, is_python_module=False, verbose=False)
    operator = torch.ops.hcu_vllm_das.deepseek_v4_inv_rope
    torch.library.register_fake("hcu_vllm_das::deepseek_v4_inv_rope")(
        lambda rope, position_ids, cos_sin_cache: None)
    registry.register(
        "vllm_pcp_q_rope", "hcu", "torch.ops.hcu_vllm_das.deepseek_v4_inv_rope",
        registry.MECH_SEAM, detail="Eager PCP prefill Q RoPE using a negated sine table",
    )
    return operator



@lru_cache(maxsize=4)
def forward_table(cos_sin_cache):
    """Reuse a reversed-angle copy across layers sharing a static Core table."""
    with torch.no_grad():
        table = cos_sin_cache.clone(memory_format=torch.contiguous_format)
        table[:, 32:].neg_()
    return table


@lru_cache(maxsize=None)
def _supported_device(device):
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0] == "gfx938"


def prepare(q, positions, cos_sin_cache):
    """Prepare the rotation, or defer unsupported inputs to the native path."""
    if (torch.version.hip is None or torch.is_grad_enabled()
            or not q.is_cuda or q.dtype != torch.bfloat16 or q.ndim != 3
            or q.shape[-1] != 512 or not q.is_contiguous()
            or not 0 < q.shape[0] * q.shape[1] <= 2**31 - 9
            or positions.device != q.device or positions.dtype != torch.int64
            or positions.shape != (q.shape[0],) or not positions.is_contiguous()
            or cos_sin_cache.device != q.device or cos_sin_cache.dtype != torch.float32
            or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64
            or not _supported_device(q.device)):
        return None

    # The native kernel applies inverse GPT-J rotation. Flip the sine to obtain
    # forward rotation; the original table is still used by Core for KV.
    return partial(load_operator(), q[..., -64:], positions, forward_table(cos_sin_cache))
