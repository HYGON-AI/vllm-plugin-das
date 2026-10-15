# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contracts for dispatching the v0.28.1 nightly from the default branch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


REPOSITORY = Path(__file__).resolve().parents[2]


def test_scheduled_default_branch_nightly_dispatches_v0281(tmp_path: Path) -> None:
    workflow = yaml.load(
        (REPOSITORY / ".github/workflows/nightly-hcu.yml").read_text(
            encoding="utf-8"
        ),
        Loader=yaml.BaseLoader,
    )

    assert workflow["on"]["schedule"]
    dispatch = workflow["jobs"]["dispatch-v0281-nightly"]
    assert workflow["permissions"] == {"contents": "read"}
    assert dispatch["permissions"] == {"contents": "read", "actions": "write"}
    assert dispatch["if"] == "${{ github.event_name == 'schedule' }}"
    step = next(
        step
        for step in dispatch["steps"]
        if step["name"] == "Dispatch v0.28.1 nightly"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "gh-args"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$GH_ARGS_FILE"\n',
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "GH_ARGS_FILE": str(captured),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO": "HYGON-AI/vllm-plugin-das",
            "TARGET_REF": "v0.28.1-dev",
        }
    )

    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        check=True,
        cwd=REPOSITORY,
        env=env,
    )
    assert captured.read_text(encoding="utf-8").splitlines() == [
        "workflow",
        "run",
        "nightly-hcu.yml",
        "--repo",
        "HYGON-AI/vllm-plugin-das",
        "--ref",
        "v0.28.1-dev",
        "-f",
        "ref=v0.28.1-dev",
    ]
