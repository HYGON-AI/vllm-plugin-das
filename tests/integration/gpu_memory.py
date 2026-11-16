# SPDX-License-Identifier: Apache-2.0
"""Wait for configured device memory before starting a CI model process."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

WAIT_TIMEOUT_ENV = "HCU_CI_GPU_MEMORY_WAIT_TIMEOUT_S"
POLL_INTERVAL_ENV = "HCU_CI_GPU_MEMORY_POLL_INTERVAL_S"
_GIB = 1024 ** 3


class GpuMemoryUnavailableError(RuntimeError):
    """The model's memory requirement remained unavailable until the deadline."""


def configured_gpu_wait_timeout(env: Mapping[str, str] | None = None) -> int:
    environment = os.environ if env is None else env
    raw = environment.get(WAIT_TIMEOUT_ENV)
    if raw is None or raw == "":
        return 600 if environment.get("VLLM_HCU_IS_IN_CI") == "1" else 0
    timeout = int(raw)
    if timeout < 0:
        raise ValueError(f"{WAIT_TIMEOUT_ENV} must be nonnegative")
    return timeout


def _read_gpu_memory(
    required_cards: int, env: Mapping[str, str], timeout_s: float,
) -> list[dict[str, int]]:
    # Query in a short-lived process so this test driver does not retain GPU
    # contexts while the actual model allocates its weights and KV cache.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe", str(required_cards)],
        env=dict(env), capture_output=True, text=True, check=False, timeout=timeout_s,
    )
    if result.returncode:
        raise RuntimeError(
            f"GPU memory probe failed (rc={result.returncode}): "
            f"{(result.stderr or result.stdout)[-2000:].strip()}"
        )
    lines = result.stdout.strip().splitlines()
    devices = json.loads(lines[-1]) if lines else []
    if not isinstance(devices, list) or len(devices) != required_cards:
        raise ValueError("GPU memory probe returned the wrong device count")
    for device in devices:
        if not isinstance(device, dict) or any(
            not isinstance(device.get(key), int) or isinstance(device.get(key), bool)
            for key in ("device", "free", "total")
        ):
            raise ValueError("GPU memory probe returned invalid counters")
        if device["total"] <= 0 or not 0 <= device["free"] <= device["total"]:
            raise ValueError("GPU memory probe returned invalid memory sizes")
    return devices


def _default_gpu_memory_utilization(
    env: Mapping[str, str], timeout_s: float,
) -> float:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--default-utilization"],
        env=dict(env), capture_output=True, text=True, check=False, timeout=timeout_s,
    )
    if result.returncode:
        raise RuntimeError(
            "Unable to resolve vLLM's default GPU memory utilization: "
            f"{(result.stderr or result.stdout)[-2000:].strip()}"
        )
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise ValueError("vLLM default GPU memory probe returned no value")
    return float(json.loads(lines[-1]))


def wait_for_gpu_memory(
    *,
    required_cards: int,
    gpu_memory_utilization: float | None,
    env: Mapping[str, str] | None = None,
    report: Callable[[str], None] | None = None,
    timeout_s: float | None = None,
) -> None:
    environment = dict(os.environ if env is None else env)
    timeout = configured_gpu_wait_timeout(environment) if timeout_s is None else timeout_s
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("GPU memory wait timeout must be finite and nonnegative")
    if timeout == 0:
        return
    interval = int(environment.get(POLL_INTERVAL_ENV, "15") or "15")
    if interval <= 0:
        raise ValueError(f"{POLL_INTERVAL_ENV} must be positive")
    if required_cards <= 0 or (
        gpu_memory_utilization is not None
        and (not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1)
    ):
        raise ValueError("GPU memory wait requires positive cards and utilization in (0, 1]")
    emit = report or (lambda message: print(message, flush=True))
    start = time.monotonic()
    deadline = start + timeout
    last_observation = "no device snapshot"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            if gpu_memory_utilization is None:
                default_fraction = _default_gpu_memory_utilization(
                    environment, min(60.0, remaining)
                )
                if not math.isfinite(default_fraction) or not 0 < default_fraction <= 1:
                    raise ValueError("vLLM default GPU memory utilization is outside (0, 1]")
                gpu_memory_utilization = default_fraction
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
            devices = _read_gpu_memory(required_cards, environment, min(60.0, remaining))
            last_observation = "; ".join(
                f"GPU {device['device']}: free={device['free'] / _GIB:.2f} GiB, "
                f"required={device['total'] * gpu_memory_utilization / _GIB:.2f} GiB"
                for device in devices
            )
            if all(
                device["free"] >= device["total"] * gpu_memory_utilization
                for device in devices
            ):
                emit(f"HCU GPU memory ready after {time.monotonic() - start:.1f}s: "
                     f"{last_observation}")
                return
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            last_observation = f"GPU memory probe unavailable: {error}"
        emit(f"HCU waiting for GPU memory ({time.monotonic() - start:.1f}/{timeout:g}s): "
             f"{last_observation}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    message = f"HCU GPU memory unavailable after {timeout:g}s: {last_observation}"
    emit(message)
    raise GpuMemoryUnavailableError(message)


def gpu_memory_utilization_from_args(command: list[str]) -> float | None:
    value = None
    for index, argument in enumerate(command):
        if argument.startswith("--gpu-memory-utilization="):
            value = float(argument.split("=", 1)[1])
        elif argument == "--gpu-memory-utilization":
            if index + 1 == len(command):
                raise ValueError("--gpu-memory-utilization requires a value")
            value = float(command[index + 1])
    return value


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", type=int)
    mode.add_argument("--default-utilization", action="store_true")
    args = parser.parse_args()
    if args.default_utilization:
        from vllm.config import CacheConfig

        print(json.dumps(float(CacheConfig().gpu_memory_utilization)), flush=True)
        return 0
    import torch

    if args.probe <= 0 or torch.cuda.device_count() < args.probe:
        raise RuntimeError(f"GPU memory probe requires {args.probe} visible devices")
    devices = []
    for index in range(args.probe):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"device": index, "free": free, "total": total})
    print(json.dumps(devices), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
