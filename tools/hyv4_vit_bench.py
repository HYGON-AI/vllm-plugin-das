# SPDX-License-Identifier: Apache-2.0
"""Compare real HYV4 vision weights with an isolated reference plugin checkout.

Uses one HCU (TP1), five warmups and 15 CUDA-event samples per input case.
Supply image cases with --case NAME IMAGE [IMAGE ...].
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)

from vllm_hcu.models.hy_v4.vl import HYV4VisionTransformer, HYV4VLProcessor
from vllm_hcu.models.hy_v4.vl_config import HYV4VLConfig


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case", nargs="+", action="append", required=True, metavar="NAME_OR_IMAGE"
    )
    parser.add_argument("--distributed-init-method", default="tcp://127.0.0.1:29872")
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()
    if any(len(case) < 2 for case in args.case):
        parser.error("--case requires a name and at least one image")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != args.tp_size:
        parser.error("--tp-size must match the torchrun world size")
    torch.cuda.set_device(local_rank)
    spec = importlib.util.spec_from_file_location(
        "vllm_hcu.models.hy_v4.baseline_vl",
        str(args.reference_checkout / "vllm_hcu/models/hy_v4/vl.py"),
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    model = args.model
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://"
        if world_size > 1
        else args.distributed_init_method,
    )
    cfg = HYV4VLConfig.from_pretrained(model)
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=args.tp_size))
    ):
        ensure_model_parallel_initialized(args.tp_size, 1)
        with torch.device(f"cuda:{local_rank}"):
            reference = base.HYV4VisionTransformer(cfg.vision_config, "vit").eval()
            optimized = HYV4VisionTransformer(cfg.vision_config, "vit").eval()
        index = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

        def weights():
            for shard in sorted({v for k, v in index.items() if k.startswith("vit.")}):
                with safe_open(model / shard, framework="pt", device="cpu") as f:
                    for name in f.keys():  # noqa: SIM118 - safe_open exposes keys(), not iteration
                        if name.startswith("vit."):
                            yield name[4:], f.get_tensor(name)

        for tower in (reference, optimized):
            loaded = tower.load_weights(weights())
            assert not set(dict(tower.named_parameters())) - loaded
        assert reference.state_dict().keys() == optimized.state_dict().keys()
        torch.set_default_dtype(torch.float32)
        proc = HYV4VLProcessor.from_pretrained(model, max_pixels=1048576)
        results = []
        with torch.inference_mode():
            for label, *paths in args.case:
                v = proc.image_processor(
                    images=[Image.open(p) for p in paths],
                    return_tensors="pt",
                    max_pixels=1048576,
                )
                pixels = v["pixel_values"].to("cuda", torch.bfloat16)
                grids = v["image_grid_thw"].tolist()
                outputs = []
                r = {
                    "label": label,
                    "grids": grids,
                    "tp_size": args.tp_size,
                    "image_sha256": [
                        hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
                    ],
                }
                for tag, tower in [("baseline", reference), ("optimized", optimized)]:
                    for _ in range(5):
                        tower(pixels, grids)
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(15):
                        s, e = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        s.record()
                        out = tower(pixels, grids)
                        e.record()
                        e.synchronize()
                        times.append(s.elapsed_time(e))
                    r[tag + "_ms"] = statistics.median(times)
                    outputs.append([o.detach().clone() for o in out])
                r["comparison"] = [
                    {
                        "cosine": float(
                            F.cosine_similarity(
                                a.flatten().float(), b.flatten().float(), dim=0
                            )
                        ),
                        "relative_l2": float(
                            (a.float() - b.float()).norm() / a.float().norm()
                        ),
                        "max_abs": float((a.float() - b.float()).abs().max()),
                    }
                    for a, b in zip(*outputs)
                ]
                # Joint encoding must preserve individual images and their order.
                if len(paths) > 1:
                    off = 0
                    ind = []
                    for g in grids:
                        length = g[1] * g[2]
                        ind.append(optimized(pixels[off : off + length], [g])[0])
                        off += length
                    r["packed_vs_individual"] = [
                        {
                            "cosine": float(
                                F.cosine_similarity(
                                    a.flatten().float(), b.flatten().float(), dim=0
                                )
                            ),
                            "relative_l2": float(
                                (a.float() - b.float()).norm() / a.float().norm()
                            ),
                        }
                        for a, b in zip(ind, outputs[1])
                    ]
                if rank == 0:
                    print(json.dumps(r), flush=True)
                results.append(r)
        if rank == 0:
            (root / "vit-comparison.json").write_text(json.dumps(results, indent=2))
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
