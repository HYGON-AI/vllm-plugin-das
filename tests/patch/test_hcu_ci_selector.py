# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable contracts for the HCU CI change selector."""

from __future__ import annotations

import ast
import builtins
from collections import defaultdict
import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
CI_SCRIPTS = REPOSITORY / ".github" / "scripts" / "hcu_ci"
if str(CI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CI_SCRIPTS))

from select_hcu_tests import (  # noqa: E402
    DEFAULT_CONFIG,
    _load_config,
    _write_github_outputs,
    select_jobs,
    validate_config,
)
from hcu_ci_preflight import PreflightError, run_preflight  # noqa: E402
from hcu_ci_preflight import (  # noqa: E402
    _check_environment_lock,
    _check_requirements,
)
from build_hcu_matrix import MatrixError, build_matrix  # noqa: E402
from compile_changed_python import _compile as compile_python_file  # noqa: E402
from compile_changed_python import main as compile_changed_python_main  # noqa: E402
from hcu_ci_register import (  # noqa: E402
    HCURegistry,
    RegistrationError,
    parse_registry,
    partition_registrations,
    validate_registrations,
)


def _config() -> dict:
    return _load_config(DEFAULT_CONFIG)


def _selected_job_ids(*paths: str) -> set[str]:
    jobs, _, _ = select_jobs(_config(), paths)
    return {job["registry_job"] for job in jobs}


def _contains_pytest_hcu_marker(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "hcu"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
        for node in ast.walk(tree)
    )


def test_selector_configuration_is_valid() -> None:
    jobs = validate_config(_config())
    assert "accuracy-gfx936" in jobs
    assert "deepseek-tp-ep" in jobs
    assert "single-node-topology" in jobs


def test_full_matrix_contains_every_configured_job() -> None:
    config = _config()
    matrix = build_matrix(config, profile="full")
    assert {job["registry_job"] for job in matrix} == set(config["jobs"])


def test_static_hcu_registry_covers_every_configured_job() -> None:
    jobs = validate_config(_config())
    registrations = parse_registry()
    validate_registrations(registrations, jobs)
    assert {registration.job for registration in registrations} == set(jobs)


def test_static_hcu_registry_rejects_runtime_generated_metadata(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.py"
    registry.write_text(
        "duration = 10\n"
        "register_hcu_ci(job='example', target='tests/test_example.py', "
        "est_time=duration)\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistrationError, match="est_time must be a Python literal"):
        parse_registry(registry)


def test_changed_python_checker_compiles_critical_sources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert compile_changed_python_main([]) == 0
    assert "Python syntax check passed" in capsys.readouterr().out


def test_changed_python_checker_rejects_syntax_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n    pass\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        compile_python_file(broken)


def test_changed_python_checker_fails_closed_when_diff_is_unavailable() -> None:
    assert (
        compile_changed_python_main(
            ["--base", "not-a-real-commit", "--head", "HEAD"]
        )
        == 2
    )


def test_lpt_partitioning_is_deterministic_and_balances_longest_first() -> None:
    registrations = [
        HCURegistry(job="example", target=f"tests/test_{index}.py", est_time=seconds)
        for index, seconds in enumerate((10, 9, 8, 1))
    ]
    first = partition_registrations(registrations, 0, 2)
    second = partition_registrations(registrations, 1, 2)
    assert [item.est_time for item in first] == [10, 1]
    assert [item.est_time for item in second] == [9, 8]
    assert partition_registrations(registrations, 0, 2) == first


def test_configured_multi_target_jobs_expand_into_lpt_partitions() -> None:
    matrix = build_matrix(_config(), profile="full")
    qwen25 = [item for item in matrix if item["registry_job"] == "qwen25-models"]
    assert [item["partition_id"] for item in qwen25] == [0, 1]
    assert {item["partition_size"] for item in qwen25} == {2}
    assert sum(item["estimated_seconds"] for item in qwen25) == 3600


def test_nightly_matrix_explicitly_covers_extended_models_and_topology() -> None:
    config = _config()
    matrix = build_matrix(config, profile="nightly")
    ids = {job["id"] for job in matrix}
    registry_jobs = {job["registry_job"] for job in matrix}
    assert len(ids) == len(matrix)
    assert registry_jobs == set(config["jobs"])
    assert {
        "accuracy-gfx938",
        "integration-smoke-gfx938",
        "mamba-smoke",
        "qwen25-models",
        "qwen3-pooling",
        "qwen3-protocol",
        "single-node-topology",
    }.issubset(registry_jobs)


def test_every_registered_hcu_test_file_routes_to_one_of_its_jobs() -> None:
    jobs_by_file: dict[str, set[str]] = defaultdict(set)
    for registration in parse_registry():
        jobs_by_file[registration.test_file].add(registration.job)

    missing = {
        test_file: sorted(expected_jobs)
        for test_file, expected_jobs in jobs_by_file.items()
        if not (_selected_job_ids(test_file) & expected_jobs)
    }
    assert missing == {}


def test_every_hcu_marked_test_file_is_registered() -> None:
    hcu_test_files = {
        path.relative_to(REPOSITORY).as_posix()
        for path in (REPOSITORY / "tests").rglob("test_*.py")
        if _contains_pytest_hcu_marker(path)
    }
    registered_files = {registration.test_file for registration in parse_registry()}
    assert hcu_test_files <= registered_files


def test_due_quarantine_builds_a_fail_closed_retest_job(tmp_path: Path) -> None:
    quarantine = tmp_path / "quarantine.json"
    quarantine.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "id": "qwen35-regression",
                        "job": "qwen35-smoke",
                        "owner": "hcu-runtime-owner",
                        "reason": "Tracked model regression",
                        "issue": "#123",
                        "nodeid": (
                            "tests/integration/models/test_qwen35_9b_smoke.py::"
                            "test_qwen35_9b_greedy_generation_smoke"
                        ),
                        "retest_after": "2026-08-01",
                        "expires": "2026-08-31",
                        "pytest_args": ["-k", "qwen35_9b_greedy_generation_smoke"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    matrix = build_matrix(
        _config(),
        profile="quarantine",
        quarantine_path=quarantine,
        today=dt.date(2026, 8, 3),
    )
    assert [job["id"] for job in matrix] == ["quarantine-qwen35-regression"]
    assert matrix[0]["pytest_args"] == [
        "-k",
        "qwen35_9b_greedy_generation_smoke",
    ]


def test_expired_quarantine_requires_maintenance(tmp_path: Path) -> None:
    quarantine = tmp_path / "quarantine.json"
    quarantine.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "id": "expired-case",
                        "job": "qwen35-smoke",
                        "owner": "hcu-runtime-owner",
                        "reason": "Tracked model regression",
                        "issue": "#123",
                        "nodeid": (
                            "tests/integration/models/test_qwen35_9b_smoke.py::"
                            "test_qwen35_9b_greedy_generation_smoke"
                        ),
                        "retest_after": "2026-07-01",
                        "expires": "2026-07-31",
                        "pytest_args": [
                            "-k",
                            "qwen35_9b_greedy_generation_smoke",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="expired on 2026-07-31"):
        build_matrix(
            _config(),
            profile="quarantine",
            quarantine_path=quarantine,
            today=dt.date(2026, 8, 3),
        )


def test_environment_lock_detects_distribution_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "dtk"
    version_file = runtime_root / ".info" / "rocm_version"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("26.04\n", encoding="utf-8")
    monkeypatch.setenv("TEST_HCU_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr("hcu_ci_preflight.platform.python_version", lambda: "3.10.12")
    lock = tmp_path / "environment.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": "3.10.12",
                "torch_hip": "6.3.26113",
                "rocm": {
                    "environment": "TEST_HCU_RUNTIME_ROOT",
                    "version_file": ".info/rocm_version",
                    "version": "26.04",
                },
                "distributions": {
                    "torch": {"match": "exact", "version": "2.11.0+hcu"},
                    "vllm-hcu": {"match": "prefix", "version": "0.25.1+"},
                },
            }
        ),
        encoding="utf-8",
    )
    report = _check_environment_lock(
        lock,
        versions={"torch": "2.11.0+hcu", "vllm-hcu": "0.25.1+build.1"},
        torch_hip="6.3.26113",
    )
    assert report["rocm"] == "26.04"
    with pytest.raises(PreflightError, match="distribution drift for torch"):
        _check_environment_lock(
            lock,
            versions={"torch": "2.12.0+hcu", "vllm-hcu": "0.25.1+build.1"},
            torch_hip="6.3.26113",
        )


def test_environment_lock_allows_compatible_hip_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "dtk"
    version_file = runtime_root / ".info" / "rocm_version"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("26.04\n", encoding="utf-8")
    monkeypatch.setenv("TEST_HCU_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr("hcu_ci_preflight.platform.python_version", lambda: "3.10.12")
    lock = tmp_path / "environment.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": "3.10.12",
                "torch_hip": {"match": "prefix", "version": "6.3."},
                "rocm": {
                    "environment": "TEST_HCU_RUNTIME_ROOT",
                    "version_file": ".info/rocm_version",
                    "version": "26.04",
                },
                "distributions": {
                    "torch": {"match": "prefix", "version": "2.11.0+"},
                    "vllm-hcu": {"match": "prefix", "version": "0.25.1+"},
                    "aiter": {"match": "prefix", "version": "0.1."},
                },
            }
        ),
        encoding="utf-8",
    )
    _check_environment_lock(
        lock,
        versions={
            "torch": "2.11.0+build.1",
            "vllm-hcu": "0.25.1+build.1",
            "aiter": "0.1.5+dtk2604.torch2110.2608211546.g6ddaa7",
        },
        torch_hip="6.3.26093",
    )
    with pytest.raises(PreflightError, match="torch HIP drift"):
        _check_environment_lock(
            lock,
            versions={
                "torch": "2.11.0+build.1",
                "vllm-hcu": "0.25.1+build.1",
                "aiter": "0.1.5+dtk2604.torch2110.2608211546.g6ddaa7",
            },
            torch_hip="6.4.0",
        )


def test_evalscope_is_required_only_by_evalscope_jobs() -> None:
    config = _config()
    evalscope_jobs = {
        "qwen35-gsm8k",
        "qwen3-vl-mmmu",
        "qwen3-8b-gsm8k",
        "deepseek-gsm8k",
        "glm52-pcp",
    }
    expected = {
        "kind": "distribution",
        "name": "evalscope",
        "match": "exact",
        "version": "1.9.1",
    }
    for job_id in evalscope_jobs:
        assert expected in config["jobs"][job_id]["requirements"]
    assert expected not in config["jobs"]["qwen35-smoke"]["requirements"]

    environment_lock = json.loads(
        (
            REPOSITORY
            / ".github/workflows/configs/hcu-runner-environment.json"
        ).read_text(encoding="utf-8")
    )
    assert "evalscope" not in environment_lock["distributions"]


def test_distribution_requirement_is_checked_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = {
        "kind": "distribution",
        "name": "evalscope",
        "match": "exact",
        "version": "1.9.1",
    }
    monkeypatch.setattr(
        "hcu_ci_preflight._distribution_version",
        lambda name: None,
    )
    with pytest.raises(
        PreflightError,
        match="required distribution is missing: evalscope",
    ):
        _check_requirements([requirement], None)

    monkeypatch.setattr(
        "hcu_ci_preflight._distribution_version",
        lambda name: "1.9.1",
    )
    assert _check_requirements([requirement], None) == [
        {
            "kind": "distribution",
            "name": "evalscope",
            "version": "1.9.1",
        }
    ]


def test_hcu_container_uses_checked_out_environment_lock() -> None:
    source = (
        REPOSITORY / "scripts/ci/hcu/hcu_ci_start_container.sh"
    ).read_text(encoding="utf-8")
    assert (
        "HCU_CI_ENVIRONMENT_LOCK=/vllm-plugin-das/.github/workflows/"
        "configs/hcu-runner-environment.json"
    ) in source


def test_docs_only_change_does_not_select_hardware() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["docs/runtime_patch_architecture_v0251.md"],
    )
    assert jobs == []
    assert groups == ["docs-only"]
    assert fallback is False


@pytest.mark.parametrize("jobs, expected", [([], "false"), ([{"id": "a"}], "true")])
def test_pr_selector_uses_shared_has_jobs_output(
    tmp_path: Path,
    jobs: list[dict[str, str]],
    expected: str,
) -> None:
    output = tmp_path / "github-output"
    _write_github_outputs(
        output,
        {
            "jobs": jobs,
            "groups": [],
            "docs_only": not jobs,
            "fallback": False,
        },
    )
    values = dict(
        line.split("=", 1)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert values["has_jobs"] == expected
    assert "has_hcu" not in values


def test_moe_change_selects_kernel_and_tp_ep_jobs() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["vllm_hcu/model_executor/layers/fused_moe/aiter_runtime.py"],
    )
    assert {job["registry_job"] for job in jobs}.issuperset(
        {"accuracy-gfx936", "qwen35-tp-ep"}
    )
    assert "moe" in groups
    assert fallback is False


def test_model_runtime_change_selects_text_vl_and_pooling_models() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/model_runtime.py"],
    )
    assert {job["registry_job"] for job in jobs}.issuperset(
        {
            "qwen35-smoke",
            "qwen25-models",
            "qwen3-pooling",
        }
    )
    assert "model-runtime-helper" in groups
    assert fallback is False


def test_qwen35_smoke_change_does_not_select_unrelated_models() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/models/test_qwen35_9b_smoke.py"],
    )
    assert {job["registry_job"] for job in jobs} == {"qwen35-smoke"}
    assert groups == ["qwen35-smoke-tests"]
    assert fallback is False


def test_deepseek_eval_change_does_not_select_tp_ep() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/test_evalscope_deepseek_r1_gsm8k.py"],
    )
    assert {job["registry_job"] for job in jobs} == {"deepseek-gsm8k"}
    assert groups == ["deepseek-evalscope"]
    assert fallback is False


def test_evalscope_report_change_runs_the_changed_test_job() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/test_evalscope_report_threshold.py"],
    )
    assert {job["registry_job"] for job in jobs} == {
        "integration-smoke-gfx938"
    }
    assert groups == ["evalscope-report-tests"]
    assert fallback is False


def test_kernel_accuracy_change_runs_both_supported_architectures() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/accuracy/test_hcu_kernel_accuracy.py"],
    )
    assert {job["registry_job"] for job in jobs} == {
        "accuracy-gfx936",
        "accuracy-gfx938",
    }
    assert groups == ["kernel-accuracy-tests"]
    assert fallback is False


def test_hcu_container_script_change_selects_ci_smoke() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["scripts/ci/hcu/hcu_ci_start_container.sh"],
    )
    assert {job["registry_job"] for job in jobs} == {
        "accuracy-gfx936",
        "contract-hcu-gfx936",
        "integration-smoke-gfx938",
    }
    assert groups == ["ci"]
    assert fallback is False


def test_classified_and_unclassified_changes_keep_conservative_fallback() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        [
            "tests/integration/lora/test_qwen3_4b_lora_switching.py",
            "vllm_hcu/forward_context_runtime.py",
        ],
    )
    assert {job["registry_job"] for job in jobs} == {
        "accuracy-gfx936",
        "contract-hcu-gfx936",
        "integration-smoke-gfx938",
        "lora",
    }
    assert groups == ["lora", "conservative-fallback"]
    assert fallback is True


def test_non_hcu_document_does_not_add_fallback_to_classified_change() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        [
            "tests/README.md",
            "tests/integration/lora/test_qwen3_4b_lora_switching.py",
        ],
    )
    assert {job["registry_job"] for job in jobs} == {"lora"}
    assert groups == ["lora"]
    assert fallback is False


def test_protocol_change_selects_protocol_server_job() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/test_qwen3_protocol_features.py"],
    )
    assert {job["registry_job"] for job in jobs} == {"qwen3-protocol"}
    assert "qwen3-protocol-tests" in groups
    assert fallback is False


def test_pooling_server_change_selects_pooling_job() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/test_qwen3_pooling_server.py"],
    )
    assert {job["registry_job"] for job in jobs} == {"qwen3-pooling"}
    assert "pooling-tests" in groups
    assert fallback is False


def test_protocol_helper_change_selects_all_server_consumers() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/openai_server.py"],
    )
    assert {job["registry_job"] for job in jobs} == {
        "qwen25-models",
        "qwen3-pooling",
        "qwen3-protocol",
    }
    assert "protocol-tests" in groups
    assert fallback is False


def test_mamba_change_selects_real_mamba_smoke() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["vllm_hcu/model_executor/layers/mamba_runtime.py"],
    )
    assert {job["registry_job"] for job in jobs} == {"mamba-smoke"}
    assert "mamba" in groups
    assert fallback is False


def test_unknown_production_change_uses_conservative_fallback() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["vllm_hcu/new_runtime_area.py"],
    )
    assert {job["registry_job"] for job in jobs} == {
        "accuracy-gfx936",
        "contract-hcu-gfx936",
        "integration-smoke-gfx938",
    }
    assert groups == ["conservative-fallback"]
    assert fallback is True


def test_accuracy_mode_adds_all_accuracy_jobs() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["docs/accuracy-notes.md"],
        accuracy=True,
    )
    assert {job["registry_job"] for job in jobs}.issuperset(
        {
            "qwen35-gsm8k",
            "qwen3-8b-gsm8k",
            "qwen3-vl-mmmu",
            "deepseek-gsm8k",
        }
    )
    assert "docs-only" in groups
    assert "accuracy-hcu" in groups
    assert fallback is False


def test_preflight_hides_runtime_dependency_import_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    private_backend_name = "".join(("A", "M", "D"))

    def fail_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError(
                f"{private_backend_name} backend package is unavailable"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_torch_import)

    with pytest.raises(
        PreflightError,
        match="^HCU runtime dependency initialization failed\\.$",
    ) as error:
        run_preflight(
            expected_arch="gfx936",
            required_cards=1,
            requirements=[],
        )

    assert error.value.__suppress_context__ is True
    assert private_backend_name not in str(error.value)


def test_preflight_hides_device_inspection_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    private_link_name = "".join(("XG", "MI"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: (_ for _ in ()).throw(
            RuntimeError(f"{private_link_name} backend query failed")
        ),
    )

    with pytest.raises(
        PreflightError,
        match="^HCU device inspection failed\\.$",
    ) as error:
        run_preflight(
            expected_arch="gfx936",
            required_cards=1,
            requirements=[],
        )

    assert error.value.__suppress_context__ is True
    assert private_link_name not in str(error.value)
