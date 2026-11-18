# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only regression contracts for the v0.28.1 ARC PR lane."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.error import URLError

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY / ".github/scripts/hcu_ci"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_hcu_k8s as arc  # noqa: E402
from select_hcu_tests import (
    DEFAULT_CONFIG,
    _load_config,
    expand_job_partitions,
    validate_config,
)  # noqa: E402


def workflow(name):
    return yaml.load(
        (REPOSITORY / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader
    )


def job(runner="bw1100", arch="gfx938", cards=1, **extra):
    return dict(
        id="selected-job",
        runner=runner,
        arch=arch,
        cards=cards,
        pytest_args=["-m", "hcu"],
        partition_id=0,
        partition_size=1,
        **extra,
    )


@pytest.mark.parametrize("runner,arch", [("bw1000", "gfx936"), ("bw1100", "gfx938")])
@pytest.mark.parametrize("cards", [1, 2, 4, 8])
def test_pool_mapping_preserves_all_selection_fields(runner, arch, cards):
    original = job(runner, arch, cards)
    outputs = arc.split_matrix([original])
    assert json.loads(outputs["arc_matrix"]) == [
        dict(original, runner=f"hygon-{runner}-{cards}card")
    ]
    assert outputs["legacy_matrix"] == "[]"
    assert original["runner"] == runner


def test_all_real_registered_partitions_are_preserved_exactly_once():
    configured = list(validate_config(_load_config(DEFAULT_CONFIG)).values())
    configured_outputs = arc.split_matrix(configured)
    assert json.loads(
        configured_outputs["legacy_matrix"]
    ), "real config must cover isolation"
    matrix = expand_job_partitions(configured)
    before = {item["id"]: item for item in matrix}
    assert len(before) == len(matrix)
    outputs = arc.split_matrix(matrix)
    hardware = json.loads(outputs["arc_matrix"])
    isolated = json.loads(outputs["legacy_matrix"])
    assert len(hardware) + len(isolated) == len(matrix)
    assert {item["id"] for item in hardware + isolated} == set(before)
    for item in hardware:
        original = before[item["id"]]
        assert not original.get("isolate_generated_code", False)
        assert item == dict(
            original, runner=f"hygon-{original['runner']}-{original['cards']}card"
        )
    for item in isolated:
        assert item == before[item["id"]]
        assert item["isolate_generated_code"] is True


def test_isolated_jobs_keep_the_original_hardened_backend():
    original = job(isolate_generated_code=True)
    outputs = arc.split_matrix([original])
    assert outputs["arc_matrix"] == "[]"
    assert json.loads(outputs["legacy_matrix"]) == [original]


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        {},
        [None],
        [job(cards=0)],
        [job(cards=3)],
        [job(cards=True)],
        [job(cards="1")],
        [job(runner="custom-pool")],
        [job(runner=[])],
        [job(arch="gfx936")],
        [job(isolate_generated_code="false")],
        [dict(job(), id="bad\nid")],
        [job(), job()],
    ],
)
def test_invalid_matrix_never_publishes_partial_outputs(tmp_path, monkeypatch, matrix):
    output = tmp_path / "output"
    monkeypatch.setenv("MATRIX_JSON", json.dumps(matrix))
    assert arc.main(["matrix", "--github-output", str(output)]) == 2
    assert not output.exists()


def test_empty_matrix_has_two_explicit_empty_outputs():
    assert arc.split_matrix([]) == {"arc_matrix": "[]", "legacy_matrix": "[]"}


@pytest.mark.parametrize("repository", arc.REPOSITORIES)
def test_digest_inputs_do_not_require_registry_tools_or_a_network_call(
    monkeypatch, repository
):
    def refused(*args):
        pytest.fail("immutable input must not require a manifest lookup")

    monkeypatch.setattr(arc, "build_opener", refused)
    image = f"{arc.REGISTRY}/{repository}@sha256:" + "a" * 64
    assert arc.resolve_image(image) == image


@pytest.mark.parametrize(
    "image",
    [
        "",
        " ",
        "foreign.invalid/vllm:latest",
        arc.REGISTRY + "/foreign/vllm:latest",
        arc.REGISTRY + "/jenkins/model_test_env/vllm",
        arc.REGISTRY + "/jenkins/model_test_env/vllm:tag@sha256:" + "a" * 64,
        arc.REGISTRY + "/jenkins/model_test_env/vllm@sha256:" + "a" * 63,
        arc.REGISTRY + "/jenkins/model_test_env/vllm:tag\nimage=foreign",
    ],
)
def test_foreign_mutated_or_malformed_image_fails_before_outputs(
    tmp_path, monkeypatch, image
):
    output = tmp_path / "output"
    monkeypatch.setenv("SOURCE_IMAGE", image)
    assert arc.main(["image", "--github-output", str(output)]) == 2
    assert not output.exists()


def registry(
    monkeypatch, *, declared=None, media_type=None, body=b'{"schemaVersion":2}'
):
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    calls = []

    def open_manifest(request, timeout):
        calls.append((request, timeout))
        response = io.BytesIO(body)
        response.headers = {
            "Docker-Content-Digest": digest if declared is None else declared,
            "Content-Type": media_type or arc.MANIFEST_TYPES[0],
        }
        return response

    def opener(handler):
        assert isinstance(handler, arc._NoRedirect)
        return SimpleNamespace(open=open_manifest)

    monkeypatch.setattr(arc, "build_opener", opener)
    return digest, calls


def test_tag_resolves_from_exact_manifest_bytes_without_credentials(
    monkeypatch, tmp_path
):
    digest, calls = registry(monkeypatch)
    monkeypatch.setenv(
        "SOURCE_IMAGE", arc.REGISTRY + "/jenkins/model_test_env/vllm:release"
    )
    output = tmp_path / "output"
    assert arc.main(["image", "--github-output", str(output)]) == 0
    assert (
        output.read_text()
        == f"image={arc.REGISTRY}/jenkins/model_test_env/vllm@{digest}\n"
    )
    request, timeout = calls[0]
    assert (
        request.full_url
        == f"http://{arc.REGISTRY}/v2/jenkins/model_test_env/vllm/manifests/release"
    )
    assert timeout == 30
    assert set(dict(request.header_items())) == {"Accept"}
    assert (
        arc._NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://foreign.invalid"
        )
        is None
    )


@pytest.mark.parametrize(
    "options",
    [
        {"declared": "sha256:" + "0" * 64},
        {"declared": ""},
        {"media_type": "text/html"},
        {"body": b"x" * (arc.MAX_MANIFEST_BYTES + 1)},
    ],
)
def test_registry_digest_and_response_policy_fail_closed(monkeypatch, options):
    registry(monkeypatch, **options)
    with pytest.raises(arc.PreparationError):
        arc.resolve_image(arc.REGISTRY + "/jenkins/model_test_env/vllm:release")


def test_registry_failure_never_publishes_outputs(monkeypatch, tmp_path):
    def fail(request, timeout):
        raise URLError("registry unavailable")

    monkeypatch.setattr(arc, "build_opener", lambda *args: SimpleNamespace(open=fail))
    monkeypatch.setenv(
        "SOURCE_IMAGE", arc.REGISTRY + "/jenkins/model_test_env/vllm:release"
    )
    output = tmp_path / "output"
    assert arc.main(["image", "--github-output", str(output)]) == 2
    assert not output.exists()


def run_bash(script, environment):
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        env=dict(os.environ, **environment),
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize(
    "mode,arc_result,status",
    [
        ("", "success", 0),
        ("k8s", "success", 0),
        ("", "skipped", 1),
        ("k8s", "failure", 1),
        ("legacy", "skipped", 0),
        ("K8S", "success", 2),
        ("k8s ", "success", 2),
    ],
)
def test_outer_gate_checks_the_selected_backend_not_skipped_jobs(
    mode, arc_result, status
):
    gate = workflow("hcu-pr-ci.yml")["jobs"]["ci-gate"]
    env = dict(
        RUNNER_MODE=mode,
        ARC_RESULT=arc_result,
        STATIC_RESULT="success",
        AUTH_RESULT="success",
        HCU_RESULT="success",
        HAS_JOBS="true",
        AUTHORIZED="true",
        GROUPS="[]",
    )
    result = run_bash(gate["steps"][0]["run"], env)
    assert result.returncode == status, result.stdout + result.stderr
    assert gate["runs-on"] == "ubuntu-latest"


@pytest.mark.parametrize(
    "updates,status",
    [
        ({}, 0),
        ({"AUTHORIZED": "false"}, 1),
        ({"IMAGE_RESULT": "failure"}, 1),
        ({"STATIC_RESULT": "failure"}, 1),
        ({"AUTH_RESULT": "failure"}, 1),
        ({"ARC_RESULT": "skipped"}, 1),
        ({"LEGACY_MATRIX": "[{}]", "LEGACY_RESULT": "failure"}, 1),
        ({"HAS_JOBS": "", "ARC_MATRIX": "[]", "ARC_RESULT": "skipped"}, 1),
        ({"ARC_MATRIX": "[]", "ARC_RESULT": "skipped"}, 1),
        ({"HAS_JOBS": "false", "ARC_MATRIX": "[]", "ARC_RESULT": "skipped"}, 0),
    ],
)
def test_arc_gate_requires_both_backends_and_trusted_authorization(updates, status):
    gate = workflow("_hcu-pr-ci-k8s.yml")["jobs"]["result"]
    env = dict(
        IMAGE_RESULT="success",
        STATIC_RESULT="success",
        AUTH_RESULT="success",
        AUTHORIZED="true",
        HAS_JOBS="true",
        ARC_MATRIX="[{}]",
        LEGACY_MATRIX="[]",
        ARC_RESULT="success",
        LEGACY_RESULT="skipped",
    )
    env.update(updates)
    result = run_bash(gate["steps"][0]["run"], env)
    assert result.returncode == status, result.stdout + result.stderr


def test_workflow_keeps_pr_event_trusted_base_auth_and_head_only_execution():
    outer = workflow("hcu-pr-ci.yml")
    assert outer["on"]["pull_request"]["branches"] == ["v0.28.1-dev"]
    assert "pull_request_target" not in outer["on"]
    assert (
        outer["jobs"]["static-and-select"]["if"]
        == "vars.HCU_CI_RUNNER_MODE_V0281 == 'legacy'"
    )
    jobs = workflow("_hcu-pr-ci-k8s.yml")["jobs"]
    assert (
        jobs["authorize-hcu"]["steps"][0]["with"]["ref"]
        == "${{ github.event.pull_request.base.sha }}"
    )
    for job_id in ("resolve-image", "static-and-select", "hardware", "authorize-hcu"):
        for step in jobs[job_id]["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] == "false"
    assert (
        jobs["resolve-image"]["steps"][0]["with"]["ref"] == "${{ github.workflow_sha }}"
    )
    for job_id in ("static-and-select", "hardware"):
        checkouts = [
            step["with"]
            for step in jobs[job_id]["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        expected_refs = ["${{ github.workflow_sha }}"]
        if job_id == "static-and-select":
            expected_refs.append("${{ github.event.pull_request.base.sha }}")
        expected_refs.append("${{ github.event.pull_request.head.sha }}")
        assert [step["ref"] for step in checkouts] == expected_refs
        assert all(step["fetch-depth"] == "1" for step in checkouts)
    for job_id in ("hardware", "isolated-hardware"):
        assert "needs.authorize-hcu.outputs.authorized == 'true'" in jobs[job_id]["if"]
    assert (
        jobs["isolated-hardware"]["uses"]
        == "./.github/workflows/_selected-hcu-tests.yml"
    )


@pytest.mark.parametrize(
    "job_id,helper,preserve_name",
    [
        (
            "static-and-select",
            ".github/scripts/hcu_ci/prepare_hcu_k8s.py",
            "Preserve caller ARC matrix mapper",
        ),
        (
            "hardware",
            "scripts/ci/hcu/hcu_ci_run_k8s.sh",
            "Preserve caller ARC native launcher",
        ),
    ],
)
def test_bootstrap_survives_shallow_checkout_of_pre_migration_head(
    tmp_path, job_id, helper, preserve_name
):
    # Model an existing PR that diverged before the migration reached its base.
    # Run the actual preservation step between real depth-1 Git checkouts.
    remote = tmp_path / "remote"
    remote.mkdir()

    def git(root, *args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    git(remote, "init", "--initial-branch=main")
    git(remote, "config", "core.autocrlf", "false")
    git(remote, "config", "user.name", "ARC fixture")
    git(remote, "config", "user.email", "arc-fixture@example.invalid")
    (remote / "revision.txt").write_text("before migration\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "before migration")
    ancestor = git(remote, "rev-parse", "HEAD")
    git(remote, "checkout", "-b", "old-pr")
    (remote / "feature.txt").write_text("tested PR source\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "old PR head")
    head = git(remote, "rev-parse", "HEAD")
    git(remote, "checkout", "main")
    assert git(remote, "rev-parse", "HEAD") == ancestor
    helper_path = remote / helper
    helper_path.parent.mkdir(parents=True)
    expected_helper = (REPOSITORY / helper).read_bytes()
    helper_path.write_bytes(expected_helper)
    (remote / "base-only.txt").write_text("new base revision\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "migration on base")
    base = git(remote, "rev-parse", "HEAD")
    git(remote, "merge", "--no-ff", "old-pr", "-m", "PR workflow merge revision")
    workflow_sha = git(remote, "rev-parse", "HEAD")
    refs = {
        "${{ github.workflow_sha }}": workflow_sha,
        "${{ github.event.pull_request.base.sha }}": base,
        "${{ github.event.pull_request.head.sha }}": head,
    }
    tested = tmp_path / "tested workspace"
    tested.mkdir()
    git(tested, "init")
    git(tested, "config", "core.autocrlf", "false")
    git(tested, "remote", "add", "origin", remote.as_uri())
    runner_temp = tmp_path / "runner temp"
    runner_temp.mkdir()
    steps = workflow("_hcu-pr-ci-k8s.yml")["jobs"][job_id]["steps"]
    preserved = False
    for step in steps:
        if step.get("uses", "").startswith("actions/checkout@"):
            settings = step["with"]
            assert settings["fetch-depth"] == "1"
            ref = refs[settings["ref"]]
            git(tested, "fetch", "--no-tags", "--depth=1", "origin", ref)
            git(tested, "checkout", "--force", "--detach", ref)
        elif step.get("name") == preserve_name:
            result = run_bash(
                'cd "$GITHUB_WORKSPACE"\n' + step["run"],
                {
                    "GITHUB_WORKSPACE": posix(tested),
                    "RUNNER_TEMP": posix(runner_temp),
                },
            )
            assert result.returncode == 0, result.stdout + result.stderr
            preserved = True
        elif step.get("name") in (
            "Verify patch contracts",
            "Prepare container-local reports",
        ):
            break
    assert preserved
    assert git(tested, "rev-parse", "HEAD") == head
    assert not (tested / helper).exists()
    assert not (tested / "base-only.txt").exists()
    assert (tested / "feature.txt").read_text() == "tested PR source\n"
    preserved_helper = runner_temp / "hcu-ci-arc-bootstrap" / Path(helper).name
    assert preserved_helper.read_bytes() == expected_helper
    subsequent_runs = "\n".join(step.get("run", "") for step in steps)
    assert f'"$RUNNER_TEMP/hcu-ci-arc-bootstrap/{Path(helper).name}"' in subsequent_runs
    assert helper not in "\n".join(
        step.get("run", "")
        for step in steps[
            steps.index(next(s for s in steps if s.get("name") == preserve_name)) + 1 :
        ]
    )
    if job_id == "static-and-select":
        # The shallow base fetch must still be available after head checkout.
        assert git(tested, "cat-file", "-t", base) == "commit"
        changed = git(tested, "diff", "--name-only", base, head)
        assert "feature.txt" in changed and "base-only.txt" in changed


def test_arc_container_contract_and_failure_artifacts():
    jobs = workflow("_hcu-pr-ci-k8s.yml")["jobs"]
    assert jobs["resolve-image"]["runs-on"] == "hygon-cpu"
    assert jobs["authorize-hcu"]["runs-on"] == "hygon-cpu"
    assert jobs["static-and-select"]["runs-on"] == "hygon-build"
    assert jobs["hardware"]["runs-on"] == "${{ matrix.runner }}"
    for job_id in ("static-and-select", "hardware"):
        assert set(jobs[job_id]["container"]) == {"image"}
        assert jobs[job_id]["defaults"]["run"]["shell"] == "bash"
    hardware = jobs["hardware"]
    assert hardware["timeout-minutes"] == "${{ matrix.timeout_minutes }}"
    assert (
        hardware["env"]["HCU_CI_ISOLATE_GENERATED_CODE"]
        == "${{ matrix.isolate_generated_code && '1' || '0' }}"
    )
    for forbidden in (
        "OMP_NUM_THREADS",
        "MAX_JOBS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HCU_CI_IN_K8S",
    ):
        assert forbidden not in hardware["env"]
    upload = next(
        step
        for step in hardware["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "${{ runner.temp }}/hcu-ci-${{ matrix.id }}"
    source = (REPOSITORY / ".github/workflows/_hcu-pr-ci-k8s.yml").read_text()
    assert "secrets." not in source and "secrets: inherit" not in source
    assert "docker " not in source and "--privileged" not in source


def posix(path):
    if os.name != "nt":
        return str(path)
    return subprocess.check_output(
        ["bash", "-c", 'cygpath -u "$1"', "--", str(path)], text=True
    ).strip()


def test_native_helper_rejects_wrong_environment_and_isolated_jobs():
    source = (REPOSITORY / "scripts/ci/hcu/hcu_ci_run_k8s.sh").read_text()
    for environment in (
        {"HCU_CI_IN_K8S": "0"},
        {"HCU_CI_IN_K8S": "1", "HCU_CI_ISOLATE_GENERATED_CODE": "1"},
    ):
        result = run_bash(source, environment)
        assert result.returncode == 2
    for forbidden in (
        "docker ",
        "HIP_VISIBLE_DEVICES=",
        "CUDA_VISIBLE_DEVICES=",
        "OMP_NUM_THREADS=",
        "MAX_JOBS=",
        "CMAKE_BUILD_PARALLEL_LEVEL=",
        "--privileged",
        "rm -rf",
    ):
        assert forbidden not in source


def test_native_helper_preserves_resources_devices_and_thread_budgets(tmp_path):
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    (workspace / ".github/workflows/configs").mkdir(parents=True)
    model = tmp_path / "models with spaces"
    dataset = tmp_path / "datasets"
    model.mkdir()
    dataset.mkdir()
    (dataset / "ci_datasets").mkdir()  # Explicit dataset roots remain exact.
    home = tmp_path / "home"
    home.mkdir()
    dtk = tmp_path / "dtk-env.sh"
    dtk.write_text("export DTK_TEST_SOURCED=yes MAX_JOBS=99 HIP_VISIBLE_DEVICES=99\n")
    source = (REPOSITORY / "scripts/ci/hcu/hcu_ci_run_k8s.sh").read_text()
    # Only replace the unavailable DTK installation in this CPU fixture.
    assert "source /opt/dtk/env.sh" in source
    source = source.replace("source /opt/dtk/env.sh", 'source "' + posix(dtk) + '"')
    source = source.replace(
        'exec "$@"',
        'printf "%s\\n" "$DTK_TEST_SOURCED" "$VLLM_HCU_TEST_MODEL_ROOT" "$VLLM_HCU_TEST_DATASET_ROOT" "$OMP_NUM_THREADS" "$MAX_JOBS" "$CMAKE_BUILD_PARALLEL_LEVEL" "$HIP_VISIBLE_DEVICES" "$CUDA_VISIBLE_DEVICES" "$HCU_CI_ENVIRONMENT_LOCK"',
    )
    environment = dict(
        HCU_CI_IN_K8S="1",
        HCU_CI_ISOLATE_GENERATED_CODE="0",
        HOME=posix(home),
        GITHUB_WORKSPACE=posix(workspace),
        RUNNER_TEMP=posix(tmp_path),
        HCU_CI_JOB_ROOT=posix(tmp_path / "reports"),
        HCU_CI_MODEL_ROOT=posix(model),
        HCU_CI_DATASET_ROOT=posix(dataset),
        OMP_NUM_THREADS="7",
        MAX_JOBS="6",
        CMAKE_BUILD_PARALLEL_LEVEL="5",
        HIP_VISIBLE_DEVICES="2",
        CUDA_VISIBLE_DEVICES="2",
    )
    result = run_bash("set -- true\n" + source, environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[:8] == [
        "yes",
        posix(model),
        posix(dataset),
        "7",
        "6",
        "5",
        "2",
        "2",
    ]
    assert result.stdout.splitlines()[8].endswith(
        "/.github/workflows/configs/hcu-runner-environment.json"
    )
    assert (tmp_path / "reports").is_dir()
