#!/usr/bin/env python3
"""Run the Channel-FP8 MoE replay inside a vLLM config context.

The replay constructs `RoutedExperts` directly, and the owned MoE method captures
`get_current_vllm_config()` at construction time because production always runs
inside that context. A CPU direct run has no such context, so the harness supplies
one; the replay itself is executed unchanged.

Scope: a deliberately small config surface. The replay already doubles every
hardware/transport dependency it needs, and nothing here replaces a factory,
group, method or forward.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from vllm.config import set_current_vllm_config


def context():
    machine = SimpleNamespace()
    compilation = SimpleNamespace(
        level=0,
        pass_config=SimpleNamespace(enable_fusion=False, enable_noop=False, enable_sequence_parallelism=False),
        splitting_ops=(),
        use_cudagraph=False,
        full_cuda_graph=False,
    )
    parallel = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=0,
        enable_expert_parallel=False,
        use_ray=False,
    )
    return SimpleNamespace(
        additional_config=None,
        compilation_config=compilation,
        parallel_config=parallel,
        model_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(),
        kernel_config=SimpleNamespace(monokernel=False),
        compilation_machine=machine,
    )


def main():
    target = Path(__file__).with_name("replay_channel_fp8_moe.py")
    with set_current_vllm_config(context()):
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    sys.exit(main() or 0)
