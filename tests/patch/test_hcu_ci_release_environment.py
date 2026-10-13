# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable contracts for release-specific HCU CI environment selection."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
CI_SCRIPTS = REPOSITORY / ".github" / "scripts" / "hcu_ci"
if str(CI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CI_SCRIPTS))

from resolve_hcu_ci_environment import (  # noqa: E402
    ReleaseEnvironmentError,
    main,
    resolve_release_environment,
)


def test_release_environment_resolver_avoids_post_python36_syntax_and_stdlib() -> None:
    source = (CI_SCRIPTS / "resolve_hcu_ci_environment.py").read_text(
        encoding="utf-8"
    )

    ast.parse(source, feature_version=(3, 6))
    assert "from __future__ import annotations" not in source
    assert "from dataclasses import dataclass" not in source
    assert " | None" not in source


def _workflow(name: str) -> dict:
    path = REPOSITORY / ".github" / "workflows" / name
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_v0281_ref_resolves_versioned_image_and_checked_out_lock() -> None:
    image = "harbor.example/vllm@sha256:" + "a" * 64

    resolved = resolve_release_environment(
        "v0.28.1-dev",
        {"HCU_CI_IMAGE_V0281": image},
    )

    assert resolved.image == image
    assert resolved.environment_lock == (
        ".github/workflows/configs/hcu-runner-environment.json"
    )


def test_v0281_ref_rejects_missing_versioned_image() -> None:
    with pytest.raises(
        ReleaseEnvironmentError,
        match="HCU_CI_IMAGE_V0281 is required for v0.28.1-dev",
    ):
        resolve_release_environment("v0.28.1-dev", {})


def test_resolver_rejects_unsupported_release_ref() -> None:
    with pytest.raises(
        ReleaseEnvironmentError,
        match="unsupported HCU CI release ref: v0.25.1",
    ):
        resolve_release_environment(
            "v0.25.1",
            {"HCU_CI_IMAGE_V0281": "harbor.example/vllm:v0.28.1"},
        )


def test_cli_failure_does_not_write_partial_github_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.delenv("HCU_CI_IMAGE_V0281", raising=False)

    assert main(["--ref", "v0.28.1-dev", "--github-output", str(output)]) == 2
    assert not output.exists()
    assert "HCU_CI_IMAGE_V0281 is required" in capsys.readouterr().err


def test_cli_writes_complete_github_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.setenv("HCU_CI_IMAGE_V0281", "harbor.example/vllm:v0.28.1")

    assert main(["--ref", "v0.28.1-dev", "--github-output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "image=harbor.example/vllm:v0.28.1",
        "environment_lock=.github/workflows/configs/hcu-runner-environment.json",
    ]


def test_pull_request_gates_target_v0281_dev() -> None:
    for workflow_name in ("hcu-pr-ci.yml", "quality-gate.yml"):
        workflow = _workflow(workflow_name)
        assert workflow["on"]["pull_request"]["branches"] == ["v0.28.1-dev"]


def test_pr_static_gate_exports_release_environment_to_consumers() -> None:
    workflow = _workflow("hcu-pr-ci.yml")
    static = workflow["jobs"]["static-and-select"]
    resolver = _named_step(static, "Resolve v0.28.1 HCU environment")

    assert resolver["id"] == "release-environment"
    assert resolver["env"] == {
        "BASE_REF": "${{ github.event.pull_request.base.ref }}",
        "HCU_CI_IMAGE_V0281": "${{ vars.HCU_CI_IMAGE_V0281 }}",
    }
    assert '[[ "$BASE_REF" != "v0.28.1-dev" ]]' in resolver["run"]
    assert "HCU_CI_IMAGE_V0281 is required" in resolver["run"]
    assert "image=$HCU_CI_IMAGE_V0281" in resolver["run"]
    assert (
        "environment_lock=.github/workflows/configs/"
        "hcu-runner-environment.json"
    ) in resolver["run"]
    assert static["outputs"]["image"] == (
        "${{ steps.release-environment.outputs.image }}"
    )
    assert static["outputs"]["environment_lock"] == (
        "${{ steps.release-environment.outputs.environment_lock }}"
    )
    verify = _named_step(static, "Verify patch contracts")
    assert (
        '--environment-lock "${{ '
        'steps.release-environment.outputs.environment_lock }}"'
    ) in verify["run"]

    authorization = workflow["jobs"]["authorize-hcu"]
    assert authorization["env"]["HCU_CI_IMAGE"] == (
        "${{ needs.static-and-select.outputs.image }}"
    )
    selected = workflow["jobs"]["selected-hcu-tests"]
    assert selected["with"]["image"] == (
        "${{ needs.static-and-select.outputs.image }}"
    )


def test_pr_ci_uses_immutable_event_shas_and_trusted_host_helpers() -> None:
    workflow = _workflow("hcu-pr-ci.yml")
    head_sha = "${{ github.event.pull_request.head.sha }}"
    base_sha = "${{ github.event.pull_request.base.sha }}"

    static = workflow["jobs"]["static-and-select"]
    trusted_checkout = _named_step(static, "Checkout trusted PR CI bootstrap")
    head_checkout = _named_step(static, "Checkout PR head revision")
    resolver = _named_step(static, "Resolve v0.28.1 HCU environment")
    verify = _named_step(static, "Verify patch contracts")
    select = _named_step(static, "Select HCU matrix")

    assert trusted_checkout["with"]["ref"] == base_sha
    assert head_checkout["with"]["ref"] == head_sha
    assert "resolve_hcu_ci_environment.py" not in resolver["run"]
    assert "$RUNNER_TEMP/hcu-ci-scripts/hcu_ci_run_control_container.sh" in verify[
        "run"
    ]
    assert "$RUNNER_TEMP/hcu-ci-scripts/hcu_ci_run_control_container.sh" in select[
        "run"
    ]

    authorization = workflow["jobs"]["authorize-hcu"]
    authorization_checkout = _named_step(
        authorization,
        "Checkout trusted authorization helpers",
    )
    assert authorization_checkout["with"]["ref"] == base_sha
    assert not any(
        step.get("name") == "Checkout PR runner helpers"
        for step in authorization["steps"]
    )

    selected = workflow["jobs"]["selected-hcu-tests"]
    assert selected["with"]["tested_ref"] == head_sha
    assert "merge_commit_sha" not in (REPOSITORY / ".github/workflows/hcu-pr-ci.yml").read_text(
        encoding="utf-8"
    )


def test_reusable_hardware_and_image_validation_never_fall_back_to_v0251() -> None:
    selected = _workflow("_selected-hcu-tests.yml")
    validation = _workflow("validate-docker-image.yml")

    assert selected["jobs"]["resolve-image"]["env"]["SOURCE_IMAGE"] == (
        "${{ inputs.image }}"
    )
    validation_source = _named_step(
        validation["jobs"]["validate"], "Resolve latest image digest"
    )["env"]["SOURCE_IMAGE"]
    assert validation_source == (
        "${{ inputs.image || vars.HCU_CI_IMAGE_V0281 }}"
    )
    assert "0.25.1-latest" not in json.dumps(selected)
    assert "0.25.1-latest" not in json.dumps(validation)


def test_full_enabled_dispatch_falls_back_to_general_hcu_image() -> None:
    full_enabled = _workflow("full-enabled-hcu.yml")
    selected = _workflow("_selected-hcu-tests.yml")

    assert full_enabled["jobs"]["hardware"]["with"]["image"] == (
        "${{ inputs.image || vars.HCU_CI_IMAGE }}"
    )
    assert selected["jobs"]["resolve-image"]["env"]["SOURCE_IMAGE"] == (
        "${{ inputs.image }}"
    )


def test_image_validation_uses_python_from_the_published_image_path() -> None:
    validation = _workflow("validate-docker-image.yml")
    command = _named_step(
        validation["jobs"]["validate"], "Validate published image"
    )["run"]

    assert "/usr/local/bin/python" not in command
    assert "python3 --version" in command
    assert "python3 .github/scripts/hcu_ci/hcu_ci_preflight.py" in command
    assert "python3 tools/run_patch_tests.py" in command


def test_image_validation_exposes_the_hcu_runtime_to_the_container() -> None:
    validation = _workflow("validate-docker-image.yml")
    command = _named_step(
        validation["jobs"]["validate"], "Validate published image"
    )["run"]

    assert "--privileged" not in command
    assert "--device /dev/kfd:/dev/kfd" in command
    assert "--device /dev/dri:/dev/dri" in command
    assert "-v /opt/hyhal:/opt/hyhal:ro" in command


def test_image_validation_rejects_images_outside_release_repository() -> None:
    validation = _workflow("validate-docker-image.yml")
    command = _named_step(
        validation["jobs"]["validate"], "Resolve latest image digest"
    )["run"]

    assert (
        'allowed_repository="${HARBOR_SITE}/${HARBOR_NAMESPACE}/vllm"'
        in command
    )
    assert 'source_repository="${SOURCE_IMAGE%%@*}"' in command
    assert 'if [[ "$source_repository" != "$allowed_repository" ]]' in command


def test_image_validation_manual_dispatch_has_no_image_override() -> None:
    validation = _workflow("validate-docker-image.yml")
    validate = validation["jobs"]["validate"]

    assert validation["on"]["workflow_dispatch"] == ""
    assert "if" not in validate
    authorization = _named_step(validate, "Authorize release ref")
    assert '[[ "$GITHUB_REF" != "refs/heads/v0.28.1-dev" ]]' in authorization["run"]
    assert "exit 2" in authorization["run"]


def test_image_validation_runs_portable_and_hcu_contracts() -> None:
    validation = _workflow("validate-docker-image.yml")
    command = _named_step(
        validation["jobs"]["validate"], "Validate published image"
    )["run"]

    assert "tools/run_patch_tests.py --suite contract --" in command
    assert "tools/run_patch_tests.py --suite contract-hcu --" in command
    assert (
        "--deselect tests/patch/test_clean_process_bootstrap.py::"
        "test_clean_hcu_process_imports_every_enabled_patch_target"
    ) in command


def test_nightly_and_patch_coverage_use_v0281_environment() -> None:
    nightly = _workflow("nightly-hcu.yml")
    patch_coverage = _workflow("patch-coverage.yml")

    assert nightly["on"]["workflow_dispatch"]["inputs"]["ref"]["default"] == (
        "v0.28.1-dev"
    )
    assert nightly["jobs"]["prepare"]["container"]["image"] == (
        "${{ vars.HCU_CI_IMAGE_V0281 }}"
    )
    assert nightly["jobs"]["hardware"]["with"]["image"] == (
        "${{ inputs.image || vars.HCU_CI_IMAGE_V0281 }}"
    )
    assert nightly["jobs"]["quarantine-retest"]["with"]["image"] == (
        "${{ inputs.image || vars.HCU_CI_IMAGE_V0281 }}"
    )

    assert patch_coverage["on"]["push"]["branches"] == ["v0.28.1-dev"]
    assert patch_coverage["jobs"]["static-patch-gate"]["container"][
        "image"
    ] == "${{ vars.HCU_CI_IMAGE_V0281 }}"
