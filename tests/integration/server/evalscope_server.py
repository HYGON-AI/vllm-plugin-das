# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared OpenAI server + EvalScope integration-test runner."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import pytest
import psutil
import yaml


ROOT = Path(__file__).resolve().parents[3]
EVALSCOPE_OWNED_ROOT = Path("/tmp/vllm-hcu-evalscope")
EVALSCOPE_OWNER_MARKER = ".vllm-hcu-evalscope-owned"
EVALSCOPE_OWNER_SIGNATURE = "vllm-plugin-das evalscope artifacts\n"
EVALSCOPE_PROCESS_OWNER_ENV = "VLLM_HCU_EVAL_PROCESS_OWNER"
_DIRECT_URL_OPENER = build_opener(ProxyHandler({}))


def _direct_urlopen(url: str, *, timeout: int):
    """Open a local service URL without inheriting host proxy settings."""

    return _DIRECT_URL_OPENER.open(url, timeout=timeout)


def load_config(default_path: Path, config_env: str) -> dict[str, Any]:
    config_path = Path(os.environ.get(config_env, str(default_path)))
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"invalid eval config: {config_path}")
    return config


def load_profiled_config(
    default_path: Path,
    config_env: str,
    *,
    profile: str,
) -> dict[str, Any]:
    """Load one named profile while preserving the existing config schema."""

    config = load_config(default_path, config_env)
    profiles = config.pop("profiles", None)
    if not isinstance(profiles, dict):
        raise TypeError("profiled eval config requires a profiles mapping")
    selected = profiles.get(profile)
    if not isinstance(selected, dict):
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(
            f"unknown eval profile {profile!r}; available profiles: {available}"
        )

    merged = copy.deepcopy(config)
    for section, value in selected.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section].update(copy.deepcopy(value))
        else:
            merged[section] = copy.deepcopy(value)
    merged["profile"] = profile
    return merged


def _available_hcu_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _model_path(config: dict[str, Any], model_env: str) -> str:
    return os.environ.get(model_env, str(config["model"]))


def _require_runtime(
    config: dict[str, Any],
    *,
    model_env: str,
    model_label: str,
    required_hcu_count: int,
) -> None:
    model = Path(_model_path(config, model_env))
    if not model.exists():
        pytest.skip(f"{model_label} model path is unavailable: {model}")
    if shutil.which("vllm") is None:
        pytest.skip("vllm CLI is unavailable")
    if shutil.which("evalscope") is None:
        pytest.skip("evalscope CLI is unavailable")
    hcu_count = _available_hcu_count()
    if hcu_count < required_hcu_count:
        pytest.skip(
            f"{model_label} test requires {required_hcu_count} HCU devices, "
            f"got {hcu_count}"
        )


def _maybe_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def server_command(
    config: dict[str, Any], *, model_env: str
) -> tuple[list[str], str, int]:
    server = config["server"]
    model = _model_path(config, model_env)
    host = os.environ.get("VLLM_HCU_SERVER_HOST", str(server.get("host", "127.0.0.1")))
    port = _maybe_int_env("VLLM_HCU_SERVER_PORT", int(server.get("port", 10128)))
    args = [str(item) for item in server.get("args", [])]
    served_model_name = server.get("served_model_name")
    if served_model_name is not None and "--served-model-name" not in args:
        args.extend(["--served-model-name", str(served_model_name)])
    if "--port" not in args and "-p" not in args:
        args.extend(["--port", str(port)])
    return ["vllm", "serve", model, *args], host, port


def evalscope_command(
    config: dict[str, Any],
    *,
    model_env: str,
    host: str,
    port: int,
    work_dir: Path,
) -> list[str]:
    evalscope = config["evalscope"]
    model = str(
        config.get("server", {}).get(
            "served_model_name",
            _model_path(config, model_env),
        )
    )
    generation = evalscope["generation_config"]
    dataset_args = json.dumps(evalscope["dataset_args"], separators=(",", ":"))
    command = [
        "evalscope",
        "eval",
        "--model",
        model,
        "--api-url",
        f"http://{host}:{port}/v1",
        "--api-key",
        str(evalscope.get("api_key", "EMPTY")),
        "--eval-type",
        str(evalscope.get("eval_type", "openai_api")),
        "--generation-config",
        json.dumps(generation, separators=(",", ":")),
    ]
    if evalscope.get("stream", False):
        command.append("--stream")
    command.extend(
        [
            "--eval-batch-size",
            str(os.environ.get("VLLM_HCU_EVAL_BATCH_SIZE", evalscope["eval_batch_size"])),
            "--timeout",
            str(evalscope["timeout"]),
            "--limit",
            str(os.environ.get("VLLM_HCU_EVAL_LIMIT", evalscope["limit"])),
            "--datasets",
            ",".join(str(item) for item in evalscope["datasets"]),
            "--dataset-args",
            dataset_args,
            "--work-dir",
            str(work_dir),
        ]
    )
    if evalscope.get("no_timestamp", True):
        command.append("--no-timestamp")
    return command


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("ab")


def _reset_evalscope_artifacts(work_dir: Path) -> None:
    """Remove outputs that EvalScope reuses when ``--no-timestamp`` is set."""

    if work_dir.is_symlink():
        raise ValueError(f"EvalScope work directory must not be a symlink: {work_dir}")
    root = work_dir.resolve()
    if root == Path(root.anchor):
        raise ValueError(f"refusing to reset EvalScope artifacts under {root}")
    marker = root / EVALSCOPE_OWNER_MARKER
    if marker.is_symlink():
        raise ValueError(f"EvalScope ownership marker must not be a symlink: {marker}")
    if not marker.exists():
        owned_root = EVALSCOPE_OWNED_ROOT.resolve()
        if root.parent != owned_root:
            raise ValueError(
                "refusing to reset EvalScope artifacts without an ownership "
                f"marker under {root}"
            )
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(EVALSCOPE_OWNER_SIGNATURE, encoding="utf-8")
    elif not marker.is_file() or marker.read_text(
        encoding="utf-8"
    ) != EVALSCOPE_OWNER_SIGNATURE:
        raise ValueError(f"invalid EvalScope ownership marker under {root}")
    for name in ("configs", "predictions", "reviews", "reports"):
        path = root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _wait_for_server(proc: subprocess.Popen, url: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(f"vLLM server exited before ready, rc={returncode}")
        try:
            with _direct_urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except URLError as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def _owned_process_environment(
    env: dict[str, str],
) -> tuple[dict[str, str], str]:
    owner_token = uuid.uuid4().hex
    owned_env = env.copy()
    owned_env[EVALSCOPE_PROCESS_OWNER_ENV] = owner_token
    return owned_env, owner_token


def _terminate_process_group(
    proc: subprocess.Popen,
    timeout_s: int,
    *,
    owner_token: str | None = None,
) -> None:
    descendants: list[psutil.Process] = []
    if proc.poll() is None:
        try:
            descendants = psutil.Process(proc.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if owner_token is not None:
        known_pids = {child.pid for child in descendants}
        for child in psutil.process_iter():
            if child.pid == os.getpid() or child.pid in known_pids:
                continue
            try:
                if child.environ().get(EVALSCOPE_PROCESS_OWNER_ENV) == owner_token:
                    descendants.append(child)
                    known_pids.add(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    try:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=timeout_s)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)

    # DP engine cores may create workers in their own sessions. Capture the
    # complete tree before stopping the API server, then explicitly reap any
    # descendants that were outside the server's process group.
    alive: list[psutil.Process] = []
    for child in descendants:
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                child.terminate()
                alive.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(alive, timeout=timeout_s)
    for child in alive:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(alive, timeout=10)


def _server_environment(config: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VLLM_PLUGINS", None)
    env["VLLM_HCU_USE_FLASH_ATTN_UNIFIED"] = "1"
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if config is not None:
        configured = config.get("server", {}).get("environment", {})
        if not isinstance(configured, dict):
            raise TypeError("server.environment must be a mapping")
        env.update({str(name): str(value) for name, value in configured.items()})
    # The server and EvalScope client share this environment. Preserve any
    # configured outbound proxy while ensuring local OpenAI-compatible requests
    # never route through it.
    no_proxy_hosts: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        for host in env.get(name, "").split(","):
            host = host.strip()
            if host and host not in no_proxy_hosts:
                no_proxy_hosts.append(host)
    for host in ("localhost", "127.0.0.1", "::1"):
        if host not in no_proxy_hosts:
            no_proxy_hosts.append(host)
    no_proxy = ",".join(no_proxy_hosts)
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env


def _report_metric(
    work_dir: Path,
    *,
    model: str,
    dataset: str,
    metric: str,
) -> tuple[float, int, Path]:
    reports_dir = work_dir / "reports"
    report_paths = sorted(
        reports_dir.rglob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    expected_model_names = {model, Path(model).name}
    available: list[str] = []
    for report_path in report_paths:
        try:
            with report_path.open(encoding="utf-8") as stream:
                report = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        report_dataset = str(report.get("dataset_name", ""))
        report_model = str(report.get("model_name", ""))
        metrics = report.get("metrics")
        if not isinstance(metrics, list):
            continue
        for report_metric in metrics:
            if not isinstance(report_metric, dict):
                continue
            metric_name = _report_metric_name(report_metric)
            available.append(f"{report_model}/{report_dataset}/{metric_name}")
            if (
                report_dataset.casefold() != dataset.casefold()
                or report_model not in expected_model_names
                or metric_name.casefold() != metric.casefold()
            ):
                continue
            score = report_metric.get("score")
            num = report_metric.get("num")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise AssertionError(
                    f"invalid {dataset} {metric} score in {report_path}: {score!r}"
                )
            if not isinstance(num, int) or isinstance(num, bool):
                raise AssertionError(
                    f"invalid {dataset} {metric} sample count in "
                    f"{report_path}: {num!r}"
                )
            return float(score), num, report_path
    rendered = ", ".join(available) if available else "none"
    raise AssertionError(
        f"missing EvalScope report metric model={Path(model).name!r}, "
        f"dataset={dataset!r}, metric={metric!r} under {reports_dir}; "
        f"available={rendered}"
    )


def _report_metric_name(report_metric: dict[str, Any]) -> str:
    """Normalize EvalScope v1 and v2 metric identities."""

    direct_name = report_metric.get("name") or report_metric.get("legacy_name")
    if direct_name:
        return str(direct_name)

    identity = report_metric.get("identity")
    if not isinstance(identity, dict):
        return ""
    name = str(identity.get("name", ""))
    aggregation = str(identity.get("aggregation", ""))
    dimensions = identity.get("dimensions")
    if name == "accuracy" and aggregation == "mean":
        return "mean_acc"
    if (
        name == "accuracy"
        and aggregation == "pass_at_k"
        and isinstance(dimensions, dict)
        and "k" in dimensions
    ):
        return f"mean_acc_pass@{dimensions['k']}"
    return name


def _assert_pass_criteria(
    config: dict[str, Any],
    *,
    model_env: str,
    work_dir: Path,
    eval_log_path: Path,
) -> None:
    criteria = config["evalscope"].get("pass_criteria")
    if criteria is None:
        return
    if not isinstance(criteria, dict):
        raise TypeError("evalscope.pass_criteria must be a mapping")
    if "metric" not in criteria:
        _assert_exact_pass_criteria(
            config,
            criteria=criteria,
            model_env=model_env,
            work_dir=work_dir,
            eval_log_path=eval_log_path,
        )
        return
    dataset = str(criteria["dataset"])
    metric = str(criteria["metric"])
    display_name = str(criteria.get("display_name", metric))
    minimum_score = float(criteria["minimum_score"])
    model = _model_path(config, model_env)
    score, num, report_path = _report_metric(
        work_dir,
        model=model,
        dataset=dataset,
        metric=metric,
    )
    verdict = (
        f"pass criterion: {dataset} {display_name}={score:.4f}, "
        f"required>={minimum_score:.4f}, samples={num}, report={report_path}"
    )
    with _open_log(eval_log_path) as eval_log:
        eval_log.write((verdict + "\n").encode())
    assert score >= minimum_score, verdict


def _artifact_records(
    work_dir: Path,
    *,
    artifact: str,
    model: str,
    dataset: str,
) -> tuple[list[dict[str, Any]], Path]:
    root = work_dir / artifact
    expected_model_names = {model, Path(model).name}
    paths = sorted(
        root.rglob("*.jsonl") if root.is_dir() else (),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in paths:
        if path.parent.name not in expected_model_names:
            continue
        if not path.name.casefold().startswith(dataset.casefold()):
            continue
        encoded_records = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(encoded_records, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"invalid {artifact} JSONL record "
                    f"{path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise AssertionError(
                    f"invalid {artifact} JSONL record "
                    f"{path}:{line_number}: expected object"
                )
            records.append(record)
        return records, path
    raise AssertionError(
        f"missing EvalScope {artifact} JSONL for model={Path(model).name!r}, "
        f"dataset={dataset!r} under {root}"
    )


def _artifact_record_count(
    work_dir: Path,
    *,
    artifact: str,
    model: str,
    dataset: str,
) -> tuple[int, Path]:
    records, path = _artifact_records(
        work_dir,
        artifact=artifact,
        model=model,
        dataset=dataset,
    )
    return len(records), path


def _normalize_humaneval_completion(completion: str) -> str:
    """Remove one complete or truncated Markdown fence around Python code."""

    opening = re.search(
        r"(?m)^[ \t]*```(?:python|py)?[ \t]*\r?\n",
        completion,
    )
    if opening is None:
        return completion.strip()
    code = completion[opening.end() :]
    closing = re.search(r"(?m)^[ \t]*```[ \t]*$", code)
    if closing is not None:
        code = code[: closing.start()]
    return code.strip()


def _check_humaneval_completion(
    problem: dict[str, Any],
    completion: str,
    timeout: int,
) -> dict[str, Any]:
    from evalscope.benchmarks.humaneval.utils import check_correctness

    return check_correctness(
        problem=problem,
        completion=completion,
        timeout=timeout,
    )


def _normalized_humaneval_score(
    work_dir: Path,
    *,
    model: str,
    dataset: str,
    expected_reviews: int,
) -> tuple[float, Path]:
    records, review_path = _artifact_records(
        work_dir,
        artifact="reviews",
        model=model,
        dataset=dataset,
    )
    assert len(records) == expected_reviews, (
        f"expected {expected_reviews} reviews, got {len(records)}; "
        f"path={review_path}"
    )

    passed = 0
    failed_task_ids: list[str] = []
    for record in records:
        sample_score = record.get("sample_score")
        if not isinstance(sample_score, dict):
            raise AssertionError(f"missing sample_score in {review_path}")
        score = sample_score.get("score")
        problem = sample_score.get("sample_metadata")
        if not isinstance(score, dict) or not isinstance(problem, dict):
            raise AssertionError(
                f"invalid HumanEval sample_score in {review_path}"
            )
        prediction = score.get("prediction")
        if not isinstance(prediction, str):
            raise AssertionError(
                f"missing HumanEval prediction in {review_path}"
            )
        task_id = str(problem.get("task_id", record.get("index", "unknown")))
        completion = _normalize_humaneval_completion(prediction)
        result = _check_humaneval_completion(problem, completion, timeout=4)
        if bool(result.get("passed", False)):
            passed += 1
        else:
            failed_task_ids.append(task_id)

    score = passed / len(records) if records else 0.0
    report_path = work_dir / "reports/normalized_humaneval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "model": model,
                "num": len(records),
                "passed": passed,
                "score": score,
                "failed_task_ids": failed_task_ids,
                "source_review": str(review_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert passed == expected_reviews, (
        f"normalized HumanEval expected {expected_reviews} passed, got "
        f"{passed}; failed task_ids={failed_task_ids}; report={report_path}"
    )
    return score, report_path


def _assert_exact_pass_criteria(
    config: dict[str, Any],
    *,
    criteria: dict[str, Any],
    model_env: str,
    work_dir: Path,
    eval_log_path: Path,
) -> None:
    dataset = str(criteria["dataset"])
    model = str(
        config.get("server", {}).get(
            "served_model_name",
            _model_path(config, model_env),
        )
    )
    expected_predictions = int(criteria["num_predictions"])
    expected_reviews = int(criteria["num_reviews"])
    metric_expectations = {
        metric: float(criteria[metric])
        for metric in ("mean_acc", "mean_acc_pass@1")
        if metric in criteria
    }
    if not metric_expectations:
        raise TypeError("exact pass criteria requires at least one score metric")

    raw_metrics: dict[str, tuple[float, int, Path]] = {}
    for metric in metric_expectations:
        score, num, report_path = _report_metric(
            work_dir,
            model=model,
            dataset=dataset,
            metric=metric,
        )
        assert num == expected_predictions, (
            f"{dataset} {metric} expected {expected_predictions} samples, "
            f"got {num}; report={report_path}"
        )
        raw_metrics[metric] = (score, num, report_path)

    prediction_count, prediction_path = _artifact_record_count(
        work_dir,
        artifact="predictions",
        model=model,
        dataset=dataset,
    )
    review_count, review_path = _artifact_record_count(
        work_dir,
        artifact="reviews",
        model=model,
        dataset=dataset,
    )
    assert prediction_count == expected_predictions, (
        f"expected {expected_predictions} predictions, got {prediction_count}; "
        f"path={prediction_path}"
    )
    assert review_count == expected_reviews, (
        f"expected {expected_reviews} reviews, got {review_count}; "
        f"path={review_path}"
    )

    normalized_score: float | None = None
    normalized_report: Path | None = None
    if bool(criteria.get("normalize_code_fences", False)):
        if dataset.casefold() != "humaneval":
            raise TypeError(
                "normalize_code_fences is supported only for HumanEval"
            )
        normalized_score, normalized_report = _normalized_humaneval_score(
            work_dir,
            model=model,
            dataset=dataset,
            expected_reviews=expected_reviews,
        )

    verdicts = []
    for metric, expected_score in metric_expectations.items():
        raw_score, num, report_path = raw_metrics[metric]
        effective_score = (
            normalized_score if normalized_score is not None else raw_score
        )
        if normalized_report is None:
            verdict = (
                f"pass criterion: {dataset} {metric}={raw_score:.4f}, "
                f"required=={expected_score:.4f}, samples={num}, "
                f"report={report_path}"
            )
        else:
            verdict = (
                f"pass criterion: {dataset} raw_{metric}={raw_score:.4f}, "
                f"normalized_{metric}={effective_score:.4f}, "
                f"required=={expected_score:.4f}, samples={num}, "
                f"report={report_path}, normalized_report={normalized_report}"
            )
        assert effective_score == expected_score, verdict
        verdicts.append(verdict)

    verdicts.append(
        f"artifact counts: predictions={prediction_count}, "
        f"reviews={review_count}, prediction_path={prediction_path}, "
        f"review_path={review_path}"
    )
    with _open_log(eval_log_path) as eval_log:
        eval_log.write(("\n".join(verdicts) + "\n").encode())


def run_evalscope_server_test(
    config: dict[str, Any],
    *,
    model_env: str,
    model_label: str,
    required_hcu_count: int,
) -> None:
    _require_runtime(
        config,
        model_env=model_env,
        model_label=model_label,
        required_hcu_count=required_hcu_count,
    )
    command, host, port = server_command(config, model_env=model_env)
    startup_timeout = _maybe_int_env(
        "VLLM_HCU_SERVER_STARTUP_TIMEOUT",
        int(config["server"].get("startup_timeout_s", 3600)),
    )
    shutdown_timeout = int(config["server"].get("shutdown_timeout_s", 60))
    work_dir = Path(
        os.environ.get("VLLM_HCU_EVAL_WORK_DIR", config["evalscope"]["work_dir"])
    )
    _reset_evalscope_artifacts(work_dir)
    log_dir = work_dir / "logs"
    server_log_path = log_dir / "vllm_server.log"
    eval_log_path = log_dir / "evalscope.log"

    env, owner_token = _owned_process_environment(_server_environment(config))

    with _open_log(server_log_path) as server_log:
        server_log.write(("server command: " + " ".join(command) + "\n").encode())
        server_log.write(b"server environment: VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1\n")
        server_log.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(proc, f"http://{host}:{port}/health", startup_timeout)
            eval_command = evalscope_command(
                config,
                model_env=model_env,
                host=host,
                port=port,
                work_dir=work_dir,
            )
            with _open_log(eval_log_path) as eval_log:
                eval_log.write(("eval command: " + " ".join(eval_command) + "\n").encode())
                eval_log.flush()
                result = subprocess.run(
                    eval_command,
                    cwd=ROOT,
                    env=env,
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            assert result.returncode == 0, (
                f"evalscope failed with rc={result.returncode}; "
                f"server_log={server_log_path}; eval_log={eval_log_path}"
            )
            _assert_pass_criteria(
                config,
                model_env=model_env,
                work_dir=work_dir,
                eval_log_path=eval_log_path,
            )
        finally:
            if (
                os.environ.get("VLLM_HCU_KEEP_SERVER_ON_FAILURE") == "1"
                and sys.exc_info()[0] is not None
            ):
                print(f"keeping vLLM server alive for debugging: pid={proc.pid}")
            else:
                _terminate_process_group(
                    proc,
                    shutdown_timeout,
                    owner_token=owner_token,
                )
