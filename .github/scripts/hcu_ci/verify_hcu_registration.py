# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Verify and inspect the literal-only HCU test registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from hcu_ci_register import (
    DEFAULT_REGISTRY,
    RegistrationError,
    load_live_estimates,
    parse_registry,
    partition_registrations,
    registrations_for_job,
    validate_registrations,
)
from select_hcu_tests import DEFAULT_CONFIG, _load_config, validate_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--job")
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--partition-size", type=int, default=1)
    parser.add_argument("--timings", type=Path)
    parser.add_argument(
        "--format",
        choices=("summary", "json", "pytest"),
        default="summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        jobs = validate_config(_load_config(args.config))
        registrations = parse_registry(args.registry)
        validate_registrations(registrations, jobs)
        selected = registrations
        if args.job:
            if args.job not in jobs:
                raise RegistrationError(f"unknown HCU registry job: {args.job}")
            selected = registrations_for_job(registrations, args.job)
            selected = partition_registrations(
                selected,
                args.partition_id,
                args.partition_size,
                live_estimates=load_live_estimates(args.timings),
            )
            if not selected:
                raise RegistrationError(
                    f"job {args.job!r} partition {args.partition_id}/"
                    f"{args.partition_size} selected no enabled tests"
                )
        if args.format == "pytest":
            print("\n".join(item.target for item in selected))
        elif args.format == "json":
            print(
                json.dumps(
                    [
                        {
                            "job": item.job,
                            "target": item.target,
                            "est_time": item.est_time,
                            "disabled": item.disabled,
                        }
                        for item in selected
                    ],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(
                f"HCU registry valid: {len(registrations)} registrations, "
                f"{len(jobs)} jobs"
            )
            for job in jobs:
                entries = registrations_for_job(registrations, job)
                print(
                    f"  {job}: {len(entries)} target(s), "
                    f"est={sum(item.est_time for item in entries):.0f}s"
                )
    except (OSError, RegistrationError, ValueError) as exc:
        print(f"HCU registration verification failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
