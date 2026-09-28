# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Changing-input Graph replay for the HCU AITER TP communicator.

Run directly with pytest on a node exposing at least four HCU devices. The
pytest controller starts an isolated four-rank worker job.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


REPOSITORY = Path(__file__).resolve().parents[4]


def _all_rank_max(value: float, group: dist.ProcessGroup) -> float:
    result = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX, group=group)
    return float(result.item())


def _worker_main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    gpu_group = dist.new_group(backend="nccl")

    from vllm_hcu.patch.platform import apply_platform_patches

    apply_platform_patches()
    from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
        AiterCustomAllreduce,
    )

    communicator = AiterCustomAllreduce(
        dist.group.WORLD,
        torch.device("cuda", local_rank),
        512 * 1024 * 1024,
    )
    try:
        assert not communicator.aiter_ca.enable_register_for_capturing
        results: list[dict[str, float | int]] = []
        for tokens in (1, 8, 64):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(1000 + rank + tokens)
            static_input = torch.randn(
                (tokens, 2048),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            static_residual = torch.randn(
                static_input.shape,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            weight = (
                torch.randn(
                    (2048,),
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                ).abs()
                + 0.5
            )

            # Initialize all native allocation and kernel paths before capture.
            for _ in range(2):
                communicator.custom_all_reduce(static_input)
                communicator.aiter_ca.custom_fused_ar_rms(
                    static_input,
                    static_residual,
                    weight,
                    1e-6,
                    use_1stage=True,
                )
            torch.cuda.synchronize()
            dist.barrier()

            graph = torch.cuda.CUDAGraph()
            with communicator.capture():
                with torch.cuda.graph(graph):
                    graph_plain = communicator.custom_all_reduce(static_input)
                    graph_fused, graph_residual = (
                        communicator.aiter_ca.custom_fused_ar_rms(
                            static_input,
                            static_residual,
                            weight,
                            1e-6,
                            use_1stage=True,
                        )
                    )
            assert graph_plain is not None
            torch.cuda.synchronize()
            dist.barrier()

            for replay in range(3):
                replay_input = torch.randn(
                    static_input.shape,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                ) + (rank + 1) * (replay + 1)
                replay_residual = torch.randn(
                    static_residual.shape,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                eager_plain = communicator.custom_all_reduce(replay_input).clone()
                reference_plain = replay_input.clone()
                dist.all_reduce(reference_plain, group=gpu_group)
                eager_fused, eager_residual = (
                    communicator.aiter_ca.custom_fused_ar_rms(
                        replay_input,
                        replay_residual,
                        weight,
                        1e-6,
                        use_1stage=True,
                    )
                )
                eager_fused = eager_fused.clone()
                eager_residual = eager_residual.clone()
                torch.cuda.synchronize()
                dist.barrier()

                # Preserve graph addresses while changing every replay's values.
                static_input.copy_(replay_input)
                static_residual.copy_(replay_residual)
                graph.replay()
                torch.cuda.synchronize()
                dist.barrier()

                row = {
                    "tokens": tokens,
                    "replay": replay,
                    "reference_max_abs": _all_rank_max(
                        (eager_plain.float() - reference_plain.float())
                        .abs()
                        .max()
                        .item(),
                        gpu_group,
                    ),
                    "reference_max_rel": _all_rank_max(
                        (
                            (eager_plain.float() - reference_plain.float())
                            .abs()
                            / reference_plain.float().abs().clamp_min(1.0)
                        )
                        .max()
                        .item(),
                        gpu_group,
                    ),
                    "plain_max_abs": _all_rank_max(
                        (graph_plain.float() - eager_plain.float())
                        .abs()
                        .max()
                        .item(),
                        gpu_group,
                    ),
                    "fused_max_abs": _all_rank_max(
                        (graph_fused.float() - eager_fused.float())
                        .abs()
                        .max()
                        .item(),
                        gpu_group,
                    ),
                    "residual_max_abs": _all_rank_max(
                        (graph_residual.float() - eager_residual.float())
                        .abs()
                        .max()
                        .item(),
                        gpu_group,
                    ),
                }
                if rank == 0:
                    print(json.dumps(row), flush=True)
                results.append(row)

        assert all(
            row["reference_max_abs"] <= 0.5
            and row["reference_max_rel"] <= 0.04
            for row in results
        ), results
        assert all(
            row["plain_max_abs"] == 0
            and row["fused_max_abs"] == 0
            and row["residual_max_abs"] == 0
            for row in results
        ), results
    finally:
        communicator.close()
        dist.destroy_process_group(gpu_group)
        dist.destroy_process_group()


def test_aiter_custom_all_reduce_graph_replays_changing_inputs() -> None:
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires a ROCm/HCU runtime")
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four visible HCU devices")
    if importlib.util.find_spec("aiter") is None:
        pytest.skip("requires the vendor AITER package")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY), environment.get("PYTHONPATH", ""))
    )
    environment["VLLM_PLUGINS"] = "__disabled__"
    environment["AITER_AR_TRANSPORT"] = "ipc"
    environment["AITER_AR_MAX_SIZE_MB"] = "256"
    environment.pop("AITER_AR_ENABLE_REG_CAPTURE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(Path(__file__).resolve()),
            "--worker",
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('"plain_max_abs": 0.0') == 9
    assert result.stdout.count('"fused_max_abs": 0.0') == 9
    assert result.stdout.count('"residual_max_abs": 0.0') == 9


if __name__ == "__main__":
    if "--worker" not in sys.argv:
        raise SystemExit("run this module with pytest")
    _worker_main()
