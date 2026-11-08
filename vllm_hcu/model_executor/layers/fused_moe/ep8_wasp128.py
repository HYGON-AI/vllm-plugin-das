# SPDX-License-Identifier: Apache-2.0
"""Load the optional Step5 EP8 grouped FP8 GEMM operator."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from vllm.logger import init_logger


@lru_cache(maxsize=1)
def _operator():
    library=Path(os.environ["HCU_STEP5_EP8_WASP128_LIBRARY"])
    proof=json.loads(library.with_suffix('.json').read_text())
    if not proof.get('passed') or proof.get('torch') != torch.__version__:
        raise RuntimeError(f"Step5 EP8 WASP128 qualification does not match PyTorch: {library}")
    if hashlib.sha256(library.read_bytes()).hexdigest() != proof.get('sha256'):
        raise RuntimeError(f"Step5 EP8 WASP128 checksum mismatch: {library}")
    torch.ops.load_library(str(library))
    init_logger('vllm.hcu.step5_wasp128').info_once('EP8 WASP128 native G1/G2 active',scope='process')
    return torch.ops.step5_wasp128_probe.gemm


def grouped_fp8(a, b, out, experts):
    bn = 384 if a[0].size(1) == 4096 else 256
    return _operator()(
        a[0], b[0], out, a[1].contiguous(), b[1].contiguous(),
        experts.contiguous(), 128, bn, 4,
    )
