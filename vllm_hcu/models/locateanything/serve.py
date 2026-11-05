# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Launch LocateAnything using only the installed HCU package and vLLM."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def build_command(mode: str, model: str, max_num_seqs: int, extra: list[str]) -> list[str]:
    if max_num_seqs < 1:
        raise ValueError("max-num-seqs must be positive")
    if any(x == "--enforce-eager" or x.startswith("--enforce-eager=") for x in extra):
        raise ValueError("This LocateAnything launch profile requires graph execution")
    if mode == "fast" and any(
        x.split("=", 1)[0].replace("_", "-") == "--speculative-config"
        or x.split("=", 1)[0].replace("_", "-").startswith("--speculative-config.")
        for x in extra
    ):
        raise ValueError("LocateAnything Fast does not support --speculative-config")
    if mode == "fast":
        from vllm_hcu.v1.locateanything.limits import reject_pipeline_parallelism
        parallel_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
        parallel_parser.add_argument("--pipeline-parallel-size", "--pipeline_parallel_size", "-pp", type=int, default=1)
        parallel_parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", "-tp", type=int, default=1)
        parallel_parser.add_argument("--data-parallel-size", "--data_parallel_size", "-dp", type=int, default=1)
        parallel_parser.add_argument("--prefill-context-parallel-size", type=int, default=1)
        parallel_parser.add_argument("--decode-context-parallel-size", type=int, default=1)
        parallel_args, _ = parallel_parser.parse_known_args(extra)
        reject_pipeline_parallelism(parallel_args.pipeline_parallel_size)
        for name in ('tensor_parallel_size','data_parallel_size','prefill_context_parallel_size','decode_context_parallel_size'):
            if getattr(parallel_args,name) != 1:
                raise ValueError(f'LocateAnything Fast requires {name}=1')
    sizes = []; size = 1
    while size < max_num_seqs:
        sizes.append(size); size *= 2
    sizes.append(max_num_seqs)
    if mode == "fast":
        sizes = [12 * x for x in sizes]
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--served-model-name", f"locateanything-native-{mode}",
        "--dtype", "bfloat16", "--tensor-parallel-size", "1",
        "--max-model-len", "16390" if mode == "fast" else "16384",
        "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", "32768" if mode == "fast" else "16384",
        "--attention-backend", "TRITON_ATTN", "--mm-encoder-attn-backend", "FLASH_ATTN",
        "--limit-mm-per-prompt", '{"image":1}', "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching", "--generation-config", "vllm",
        "--chat-template", str(Path(__file__).with_name("chat_template.jinja")),
        "--chat-template-content-format", "openai",
        "--cudagraph-capture-sizes", *map(str, sizes), "--cudagraph-metrics"]
    if mode == "fast":
        argv += ["--no-enable-chunked-prefill", "--no-async-scheduling"]
    return argv + extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["slow", "fast"], default="slow")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    args, extra = parser.parse_known_args()
    env = os.environ.copy()
    env["VLLM_HCU_LOCATEANYTHING_MODE"] = args.mode
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env.setdefault("HIP_GRAPH_ACCUMULATE_DISPATCH", "1")
    env.setdefault("USE_CONTIGUOUS_FA", "1")
    os.execvpe(sys.executable, build_command(args.mode, args.model, args.max_num_seqs, extra), env)


if __name__ == "__main__":
    main()
