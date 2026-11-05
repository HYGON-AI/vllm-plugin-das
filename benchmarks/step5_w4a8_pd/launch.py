#!/usr/bin/env python3
"""Launch one fixed P/D performance profile through the 96 GiB watchdog."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def plan(args):
    model = args.model.expanduser().resolve()
    tokenizer = (args.tokenizer or args.model).expanduser().resolve()
    if not model.is_dir() or not tokenizer.is_dir():
        raise ValueError("Model and tokenizer must be existing local directories")
    work = args.work_dir.expanduser().resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = work / "runs" / f"{args.role}-{stamp}"
    if run.exists():
        raise ValueError("The run directory already exists")
    environment = dict(os.environ)
    environment.update(
        MODEL_PATH=str(model),
        TOKENIZER_PATH=str(tokenizer),
        STEP5_EXP_RUN=str(run),
        STEP5_HOST=args.host,
        STEP5_PORT=str(args.port),
        STEP5_RESERVED_BYTES_PER_CARD="0",
        PLACEMENT_PATH=str(ROOT / "assets" / "expert_placement_prefill.pt"),
        TRITON_CACHE_DIR=str(work / "cache" / args.role / "triton"),
        VLLM_CACHE_ROOT=str(work / "cache" / args.role / "vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(work / "cache" / args.role / "inductor"),
    )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "budget_guard.py"),
        "--execute",
        "--",
        "bash",
        str(ROOT / "profiles" / f"{args.role}.sh"),
    ]
    return command, environment, run


def service_exit_code(guard_code, run):
    if guard_code:
        return guard_code
    result = run / "guard-result.json"
    if not result.is_file():
        return 1
    child_code = json.loads(result.read_text()).get("returncode")
    if not isinstance(child_code, int):
        return 1
    return child_code if child_code >= 0 else 128 - child_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--execute", action="store_true", help="start the selected service"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    try:
        command, environment, run = plan(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    if not args.execute:
        print(
            json.dumps(
                {
                    "role": args.role,
                    "command": command,
                    "run": str(run),
                    "fixed_reservation_started": False,
                    "service_limit_gib": 96,
                    "synthetic_mtp": True,
                    "model_started": False,
                },
                indent=2,
            )
        )
        return
    if sys.platform != "linux":
        parser.error("Service execution requires the Linux HCU container")
    try:
        from vllm_hcu.ops.step5.w4a8 import require_native_op

        require_native_op()
    except (ImportError, RuntimeError) as error:
        parser.error(
            "Install the plugin with the built-in Step5 W4A8 operator: " + str(error)
        )
    print(f"Profile: {args.role}; server log: {run / 'server.log'}", flush=True)
    print("Fixed synthetic MTP3 is enabled: this is a performance profile.", flush=True)
    guard_code = subprocess.call(command, env=environment)
    raise SystemExit(service_exit_code(guard_code, run))


if __name__ == "__main__":
    main()
