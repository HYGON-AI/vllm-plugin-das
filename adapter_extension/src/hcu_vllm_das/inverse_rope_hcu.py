# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU inverse-RoPE provider, prepared before model execution or Graph capture."""
from functools import lru_cache, partial
import hashlib
import os
from pathlib import Path
import tempfile

import torch


@lru_cache(maxsize=1)
def load_operator():
    if torch.version.hip is None:
        raise RuntimeError("HCU inverse RoPE requires HIP PyTorch and hipcc")
    from torch.utils.cpp_extension import load, _get_build_directory

    root = Path(__file__).with_name("csrc")
    assets = [(name, (root / name).read_bytes())
              for name in ("inverse_rope.cpp", "inverse_rope.cu")]
    # Expose both arithmetic modes without compiler-chosen contraction. Inverse
    # preparation selects the actual Core contract; PCP Q always uses False.
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
        lambda rope, position_ids, cos_sin_cache, fuse_cosine=True: None)
    return operator


def _probe_mode(original, operator, device):
    """Distinguish the two native modes using private cancellation witnesses.

    Run only during preparation, never on a caller's data or a captured path.
    Position dtypes must agree; an unknown/ambiguous lowering is not admitted.
    """
    values = torch.tensor([
        [-1.1484375, 0.83203125, -0.809720516204834, 0.5868157148361206],
        [0.546875, -2.359375, -0.22580166161060333, 0.9741733074188232],
        [-0.80078125, -1.0390625, -0.7926467657089233, 0.6096811890602112],
    ], dtype=torch.float32, device=device)
    source = torch.zeros(3, 1, 512, dtype=torch.bfloat16, device=device)
    source[..., 448::2] = values[:, 0, None, None]
    source[..., 449::2] = values[:, 1, None, None]
    cache = torch.cat((values[:, 2, None].expand(-1, 32),
                       values[:, 3, None].expand(-1, 32)), dim=-1)
    matches = {False, True}
    with torch.inference_mode():
        for dtype in (torch.int32, torch.int64):
            positions = torch.arange(3, device=device, dtype=dtype)
            expected = source.clone()
            original(expected, positions, cache, 64)
            for mode in tuple(matches):
                actual = source.clone()
                operator(actual[..., -64:], positions, cache, fuse_cosine=mode)
                if not torch.equal(actual, expected):
                    matches.remove(mode)
    if len(matches) != 1:
        raise RuntimeError("Core inverse-RoPE arithmetic is unsupported or ambiguous")
    return matches.pop()


def prepare_operator(original, device=None):
    """Bind Core-matching native arithmetic before model execution/Graph capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare inverse RoPE before Graph capture")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    operator = load_operator()
    mode = _probe_mode(original, operator, device)
    return partial(operator, fuse_cosine=mode)


def eligible(output, positions, cos_sin_cache, rope_head_dim):
    return (
        torch.version.hip is not None and not torch.is_grad_enabled()
        and output.is_cuda and output.dtype == torch.bfloat16
        and output.ndim == 3 and output.shape[-1] >= 64
        and rope_head_dim == 64 and output.stride(-1) == 1
        and output.stride(1) >= output.shape[-1]
        and output.stride(0) >= output.shape[1] * output.stride(1)
        and positions.device == output.device and positions.ndim == 1
        and positions.shape[0] == output.shape[0] and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and cos_sin_cache.device == output.device and cos_sin_cache.ndim == 2
        and cos_sin_cache.dtype == torch.float32 and cos_sin_cache.shape[1] == 64
        and cos_sin_cache.stride(1) == 1 and cos_sin_cache.stride(0) >= 64
    )


def route(original, operator):
    def inverse(output, positions, cos_sin_cache, rope_head_dim):
        if output.shape[0] == 0:
            return None
        if operator is None or not eligible(output, positions, cos_sin_cache, rope_head_dim):
            return original(output, positions, cos_sin_cache, rope_head_dim)
        return operator(output[..., -64:], positions, cos_sin_cache)
    return inverse
