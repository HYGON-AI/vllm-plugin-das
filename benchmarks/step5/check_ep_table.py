"""Validate the effective EP8 or EP16 table before loading Step5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

KEY = "E=22,N=3072,K=4096,topk=8"


def check_ep_table(
    path: str | Path, ep_size: int = 16, physical_experts: int = 352,
    require_m: int | None = None,
) -> dict:
    if ep_size not in (8, 16):
        raise ValueError(f"Unsupported Step5 expert-parallel size: {ep_size}")
    if type(physical_experts) is not int or physical_experts not in (352, 368):
        raise ValueError("Step5 physical_experts must be 352 or 368")
    if physical_experts % ep_size:
        raise ValueError("Step5 physical experts must divide evenly across EP ranks")
    if require_m is not None and (type(require_m) is not int or require_m <= 0):
        raise ValueError("Step5 require_m must be a positive integer")
    key = f"E={physical_experts // ep_size},N=3072,K=4096,topk=8"
    path = Path(path).resolve()
    raw = path.read_bytes()
    table = json.loads(raw)
    entries = table.get(key)
    if not isinstance(entries, dict) or not entries:
        raise ValueError(
            f"Step5 EP{ep_size} requires {key} in {path}; the effective table would "
            "otherwise fall back to untuned library configurations"
        )
    points = {}
    for size, pair in entries.items():
        try:
            size = int(size)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Step5 EP{ep_size} token count: {size!r}") from error
        if size <= 0 or size in points or not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid Step5 EP{ep_size} GEMM pair at M={size}")
        for config in pair:
            if not isinstance(config, dict) or any(
                type(config.get(name)) is not int or config[name] <= 0
                for name in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
            ):
                raise ValueError(f"Invalid Step5 EP{ep_size} GEMM dimensions at M={size}")
        if pair[0]["BLOCK_SIZE_M"] != pair[1]["BLOCK_SIZE_M"]:
            raise ValueError(f"Step5 EP{ep_size} GEMM pair has different BLOCK_SIZE_M at M={size}")
        points[size] = pair
    if require_m is not None and require_m not in points:
        raise ValueError(f"Step5 {key} requires an explicit tuned row at M={require_m}")
    selected = {}
    for size in dict.fromkeys((256, 32768) + (() if require_m is None else (require_m,))):
        nearest = min(points, key=lambda candidate: abs(candidate-size))
        selected[str(size)] = {"table_m": nearest, "gemm1": points[nearest][0], "gemm2": points[nearest][1]}
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "key": key, "ep_size": ep_size, "physical_experts": physical_experts,
            "local_experts": physical_experts // ep_size,
            "entries": len(points), "selected": selected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--ep-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--physical-experts", type=int, choices=[352, 368], default=352)
    parser.add_argument("--require-m", type=int)
    args = parser.parse_args()
    try:
        print(f"STEP5_EP{args.ep_size}_TABLE " + json.dumps(check_ep_table(
            args.path, args.ep_size, args.physical_experts, args.require_m)))
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")
