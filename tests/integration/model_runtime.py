# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared helpers for real-model integration tests.

This module is intentionally safe to import on CPU-only hosts. vLLM is imported
only inside the subprocess cases after pytest has checked model and HCU
availability.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
if os.environ.get("VLLM_HCU_RELEASE_WHEEL") == "1":
    repository_text = str(REPOSITORY)
    if repository_text not in sys.path:
        sys.path.append(repository_text)

from tests.fixtures.resources import TestResources as HcuTestResources


DEFAULT_MODEL_ROOT = Path("/models/llm-models")
DEFAULT_LOG_DIR = Path("/tmp/vllm-hcu-integration/logs")
RESULT_PREFIX = "VLLM_HCU_RESULT="
UNIFIED_ATTENTION_HEAD_DIMS = {128, 192, 256, 512}


class _DataParallelTermination(BaseException):
    pass


class _DataParallelTerminationHandler:
    def __init__(self) -> None:
        self.defer_termination = False
        self.deferred_signum: int | None = None

    def __call__(self, signum: int, frame: Any) -> None:
        if self.defer_termination:
            self.deferred_signum = signum
            return
        _raise_data_parallel_signal(signum, frame)

    def raise_if_deferred(self) -> None:
        signum = self.deferred_signum
        self.deferred_signum = None
        if signum is not None:
            _raise_data_parallel_signal(signum, None)


def _raise_data_parallel_signal(signum: int, frame: Any) -> None:
    if signum == signal.SIGINT:
        signal.default_int_handler(signum, frame)
    _raise_data_parallel_termination(signum, frame)


def _raise_data_parallel_termination(signum: int, frame: Any) -> None:
    del frame
    raise _DataParallelTermination(
        f"data-parallel launcher received signal {signum}"
    )


def available_hcu_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def require_gfx_arch(required_arch: str, label: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip(f"{label} requires {required_arch}, but no live HCU is available")
        properties = torch.cuda.get_device_properties(0)
    except Exception as exc:
        pytest.skip(f"{label} requires {required_arch}, but HCU arch is unavailable: {exc}")
    gcn_arch = getattr(properties, "gcnArchName", None)
    if not isinstance(gcn_arch, str):
        pytest.skip(f"{label} requires {required_arch}, but HCU arch is unavailable")
    current_arch = gcn_arch.split(":", 1)[0]
    if current_arch != required_arch:
        pytest.skip(f"{label} requires {required_arch}, got {current_arch}")


def resolve_model_path(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    if resources.model_root is not None:
        return resources.resolve_model(relative_path)
    default_path = DEFAULT_MODEL_ROOT / relative_path
    if default_path.exists():
        return default_path.resolve()
    return Path(relative_path)


def require_model_runtime(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
    label: str,
    hcu_count: int = 1,
) -> Path:
    model_path = resolve_model_path(
        resources,
        env_name=env_name,
        relative_path=relative_path,
    )
    if not model_path.exists():
        message = f"{label} model path is unavailable: {model_path}"
        if resources.strict:
            pytest.fail(message)
        pytest.skip(message)
    _require_loadable_local_model(resources, model_path, label)
    _require_unified_attention_compatible(resources, model_path, label)
    actual_hcu_count = available_hcu_count()
    if actual_hcu_count < hcu_count:
        pytest.skip(
            f"{label} test requires {hcu_count} HCU devices, got {actual_hcu_count}"
        )
    return model_path


def _require_loadable_local_model(
    resources: HcuTestResources,
    model_path: Path,
    label: str,
) -> None:
    if model_path.is_dir() and (
        (model_path / "config.json").is_file()
        or (model_path / "params.json").is_file()
    ):
        return
    message = (
        f"{label} model path is not a loadable local model directory: "
        f"{model_path}; expected config.json or params.json"
    )
    if resources.strict:
        pytest.fail(message)
    pytest.skip(message)


def require_non_hybrid_model(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
    label: str,
    hcu_count: int = 1,
) -> Path:
    model_path = resolve_model_path(
        resources,
        env_name=env_name,
        relative_path=relative_path,
    )
    if not model_path.exists():
        message = f"{label} model path is unavailable: {model_path}"
        if resources.strict:
            pytest.fail(message)
        pytest.skip(message)
    _require_loadable_local_model(resources, model_path, label)
    _require_unified_attention_compatible(resources, model_path, label)
    if _is_hybrid_kv_model(model_path):
        message = (
            f"{label} model {model_path} is a hybrid KV-cache model and is "
            "incompatible with ExampleConnector, which does not support HMA"
        )
        if resources.strict:
            pytest.fail(message)
        pytest.skip(message)
    actual_hcu_count = available_hcu_count()
    if actual_hcu_count < hcu_count:
        pytest.skip(
            f"{label} test requires {hcu_count} HCU devices, got {actual_hcu_count}"
        )
    return model_path


def require_resource_path(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
    label: str,
) -> Path:
    path = resolve_model_path(
        resources,
        env_name=env_name,
        relative_path=relative_path,
    )
    if path.exists():
        return path
    message = f"{label} path is unavailable: {path}"
    if resources.strict:
        pytest.fail(message)
    pytest.skip(message)


def require_model_architecture(
    resources: HcuTestResources,
    model_path: Path,
    *,
    label: str,
    supported_architectures: set[str],
) -> None:
    architectures = _model_architectures(model_path)
    if architectures and any(item in supported_architectures for item in architectures):
        return
    rendered = ", ".join(architectures) if architectures else "unknown"
    message = (
        f"{label} architectures [{rendered}] are unsupported for this "
        f"integration test; expected one of {sorted(supported_architectures)}"
    )
    if resources.strict:
        pytest.fail(message)
    pytest.skip(message)


def _require_unified_attention_compatible(
    resources: HcuTestResources,
    model_path: Path,
    label: str,
) -> None:
    if _is_mla_attention_model(model_path):
        return
    head_dim = _model_attention_head_dim(model_path)
    if head_dim is None or head_dim in UNIFIED_ATTENTION_HEAD_DIMS:
        return
    message = (
        f"{label} attention head_dim={head_dim} is incompatible with "
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1"
    )
    if resources.strict:
        pytest.fail(message)
        pytest.skip(message)


def _is_mla_attention_model(model_path: Path) -> bool:
    config = _read_model_config(model_path)
    if config is None:
        return False
    model_type = str(config.get("model_type", "")).casefold()
    architectures = _model_architectures(model_path)
    if model_type.startswith("deepseek"):
        return True
    if any("deepseek" in item.casefold() for item in architectures):
        return True
    mla_fields = ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
    return any(isinstance(config.get(name), int) for name in mla_fields)


def _model_attention_head_dim(model_path: Path) -> int | None:
    config = _read_model_config(model_path)
    if config is None:
        return None
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        head_dim = _attention_head_dim_from_config(text_config)
        if head_dim is not None:
            return head_dim
    return _attention_head_dim_from_config(config)


def _model_architectures(model_path: Path) -> list[str]:
    config = _read_model_config(model_path)
    if config is None:
        return []
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        return []
    return [str(item) for item in architectures]


def _read_model_config(model_path: Path) -> dict[str, Any] | None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        return None
    return config


def _attention_head_dim_from_config(config: dict[str, Any]) -> int | None:
    explicit_head_dim = config.get("head_dim")
    if isinstance(explicit_head_dim, int) and not isinstance(
        explicit_head_dim, bool
    ):
        return explicit_head_dim
    hidden_size = config.get("hidden_size")
    attention_heads = config.get("num_attention_heads")
    if (
        isinstance(hidden_size, int)
        and isinstance(attention_heads, int)
        and not isinstance(hidden_size, bool)
        and not isinstance(attention_heads, bool)
        and attention_heads > 0
    ):
        return hidden_size // attention_heads
    return None


def _is_hybrid_kv_model(model_path: Path) -> bool:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        return False
    return _config_has_multiple_kv_layer_types(config)


def _config_has_multiple_kv_layer_types(config: dict[str, Any]) -> bool:
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and len({str(item) for item in layer_types}) > 1:
        return True
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return _config_has_multiple_kv_layer_types(text_config)
    return False


def run_vllm_case(
    case: str,
    model_path: Path,
    *,
    timeout_s: int = 900,
    gpu_memory_utilization: float | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    log_label: str | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("VLLM_PLUGINS", None)
    env["VLLM_HCU_USE_FLASH_ATTN_UNIFIED"] = "1"
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu_memory_utilization is not None:
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError(
                "gpu_memory_utilization must be greater than 0 and at most 1"
            )
        env["VLLM_HCU_TEST_GPU_MEMORY_UTILIZATION"] = str(
            gpu_memory_utilization
        )
    if extra_env:
        env.update(extra_env)
    log_path = _case_log_path(log_label or case, model_path)
    release_wheel = os.environ.get("VLLM_HCU_RELEASE_WHEEL") == "1"
    if release_wheel:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            case,
            "--model",
            str(model_path),
        ]
        child_cwd = Path(
            os.environ.get("HCU_CI_JOB_ROOT", "/tmp/vllm-hcu-release-wheel")
        ) / "model-subprocess"
        child_cwd.mkdir(parents=True, exist_ok=True)
    else:
        command = [
            sys.executable,
            "-m",
            "tests.integration.model_runtime",
            case,
            "--model",
            str(model_path),
        ]
        child_cwd = REPOSITORY
    if extra_args:
        command.extend(extra_args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(_case_log_header(command, env=env, extra_env=extra_env))
            log.flush()
            proc = subprocess.Popen(
                command,
                cwd=child_cwd,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _terminate_case_process_group(proc)
                output = log_path.read_text(encoding="utf-8", errors="replace")
                raise AssertionError(
                    f"vLLM integration case {case!r} timed out after "
                    f"{timeout_s}s\n"
                    f"command={' '.join(command)}\n"
                    f"log={log_path}\n"
                    f"{output}"
                ) from None
        output = log_path.read_text(encoding="utf-8", errors="replace")
        if returncode != 0:
            raise AssertionError(
                f"vLLM integration case {case!r} failed with rc={returncode}\n"
                f"command={' '.join(command)}\n"
                f"log={log_path}\n"
                f"{output}"
            )
        for line in reversed(output.splitlines()):
            if line.startswith(RESULT_PREFIX):
                payload = line.removeprefix(RESULT_PREFIX)
                parsed = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise AssertionError(
                        f"invalid result payload for {case!r}: {payload}"
                    )
                return parsed
        raise AssertionError(
            f"vLLM integration case {case!r} did not emit {RESULT_PREFIX!r}\n"
            f"log={log_path}\n"
            f"{output}"
        )
    except BaseException:
        if proc is not None:
            _terminate_case_process_group(proc)
        raise


def _case_log_path(case: str, model_path: Path) -> Path:
    log_dir = Path(os.environ.get("VLLM_HCU_INTEGRATION_LOG_DIR", DEFAULT_LOG_DIR))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_path.name)
    case_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case)
    return log_dir / f"{timestamp}_{model_name}_{case_name}.log"


def _case_log_header(
    command: list[str],
    *,
    env: dict[str, str],
    extra_env: dict[str, str] | None,
) -> str:
    header = "command: " + " ".join(command) + "\n"
    env_names = [
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED",
        "VLLM_HCU_TEST_GPU_MEMORY_UTILIZATION",
    ]
    if extra_env:
        env_names.extend(sorted(extra_env))
    header += "environment: " + " ".join(
        f"{name}={env[name]}" for name in env_names if name in env
    ) + "\n"
    return header


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_groups(
    process_group_ids: set[int],
    timeout_s: float,
    *,
    process_leaders: list[Any] | None = None,
) -> set[int]:
    remaining = set(process_group_ids)
    deadline = time.monotonic() + timeout_s
    while remaining:
        for process in process_leaders or ():
            if hasattr(process, "poll"):
                process.poll()
            else:
                process.join(timeout=0)
        remaining = {
            process_group_id
            for process_group_id in remaining
            if _process_group_exists(process_group_id)
        }
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    return remaining


def _signal_process_groups(process_group_ids: set[int], sig: signal.Signals) -> None:
    for process_group_id in sorted(process_group_ids):
        try:
            os.killpg(process_group_id, sig)
        except ProcessLookupError:
            pass


def _terminate_owned_process_groups(
    process_group_ids: list[int] | set[int],
    *,
    process_leaders: list[Any] | None = None,
    term_timeout_s: float = 30,
    kill_timeout_s: float = 10,
) -> None:
    owned_groups = {int(process_group_id) for process_group_id in process_group_ids}
    if not owned_groups:
        return
    protected_groups = {os.getpgrp()}
    try:
        protected_groups.add(os.getpgid(os.getppid()))
    except ProcessLookupError:
        pass
    if min(owned_groups) <= 1 or owned_groups & protected_groups:
        raise RuntimeError(
            f"refusing to signal unvalidated process groups {sorted(owned_groups)}"
        )

    wait_kwargs = (
        {"process_leaders": process_leaders}
        if process_leaders is not None
        else {}
    )
    _signal_process_groups(owned_groups, signal.SIGTERM)
    remaining = _wait_for_process_groups(
        owned_groups,
        term_timeout_s,
        **wait_kwargs,
    )
    if remaining:
        _signal_process_groups(remaining, signal.SIGKILL)
        remaining = _wait_for_process_groups(
            remaining,
            kill_timeout_s,
            **wait_kwargs,
        )
    if remaining:
        raise RuntimeError(
            f"task-owned process groups survived cleanup: {sorted(remaining)}"
        )


def _terminate_case_process_group(proc: subprocess.Popen) -> None:
    _terminate_owned_process_groups([proc.pid], process_leaders=[proc])
    try:
        proc.wait(timeout=0)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"case process {proc.pid} survived process-group cleanup"
        ) from exc


def _llm_kwargs(
    model_path: Path,
    *,
    enforce_eager: bool,
    **overrides: Any,
) -> dict[str, Any]:
    kwargs = {
        "model": str(model_path),
        "trust_remote_code": True,
        "enforce_eager": enforce_eager,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 4,
        "gpu_memory_utilization": float(
            os.environ.get("VLLM_HCU_TEST_GPU_MEMORY_UTILIZATION", "0.35")
        ),
        "seed": 0,
    }
    kwargs.update(overrides)
    return kwargs


def _shutdown_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=30)
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _single_completion(record: Any) -> dict[str, Any]:
    output = record.outputs[0]
    token_ids = list(output.token_ids)
    cumulative_logprob = output.cumulative_logprob
    if cumulative_logprob is not None and not math.isfinite(cumulative_logprob):
        raise AssertionError(f"non-finite cumulative logprob: {cumulative_logprob}")
    lora_request = getattr(record, "lora_request", None)
    sample_logprobs = getattr(output, "logprobs", None)
    prompt_logprobs = getattr(record, "prompt_logprobs", None)
    return {
        "prompt_token_count": len(record.prompt_token_ids or []),
        "token_ids": token_ids,
        "text": output.text,
        "finish_reason": output.finish_reason,
        "cumulative_logprob": cumulative_logprob,
        "sample_logprob_count": _logprob_position_count(sample_logprobs),
        "sample_top_logprob_count": _logprob_entry_count(sample_logprobs),
        "prompt_logprob_count": _logprob_position_count(prompt_logprobs),
        "prompt_top_logprob_count": _logprob_entry_count(prompt_logprobs),
        "lora_name": getattr(lora_request, "lora_name", None),
        "lora_int_id": getattr(lora_request, "lora_int_id", None),
    }


def _logprob_position_count(logprobs: Any) -> int:
    if logprobs is None:
        return 0
    return sum(1 for item in logprobs if item is not None)


def _logprob_entry_count(logprobs: Any) -> int:
    if logprobs is None:
        return 0
    total = 0
    for item in logprobs:
        if item is None:
            continue
        try:
            total += len(item)
        except TypeError:
            total += 1
    return total


def _generate_with_llm(
    llm: Any,
    *,
    prompts: list[str] | None = None,
    lora_request: Any = None,
    max_tokens: int = 8,
    logprobs: int | None = 1,
    prompt_logprobs: int | None = None,
) -> list[dict[str, Any]]:
    from vllm.sampling_params import SamplingParams

    if prompts is None:
        prompts = [
            "The capital of France is",
            "Answer with one number: 2 + 2 =",
        ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        seed=0,
    )
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    return [_single_completion(record) for record in outputs]


def _generate(
    model_path: Path,
    *,
    enforce_eager: bool,
    **llm_overrides: Any,
) -> list[dict[str, Any]]:
    from vllm import LLM

    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=enforce_eager,
            **llm_overrides,
        )
    )
    try:
        return _generate_with_llm(llm)
    finally:
        _shutdown_llm(llm)


def _case_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM

    llm = LLM(**_llm_kwargs(model_path, enforce_eager=True))
    try:
        first = _generate_with_llm(llm)
        second = _generate_with_llm(llm)
    finally:
        _shutdown_llm(llm)
    return {
        "first": first,
        "second": second,
    }


def _case_graph_parity(model_path: Path) -> dict[str, Any]:
    eager = _generate(model_path, enforce_eager=True)
    graph = _generate(model_path, enforce_eager=False)
    return {
        "eager": eager,
        "graph": graph,
    }


def _case_lora_switching(
    model_path: Path,
    *,
    lora_a: Path,
    lora_b: Path,
) -> dict[str, Any]:
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    prompt = ["用一句话写一个武侠小说开头："]
    adapter_a = LoRARequest("adapter-a", 1, str(lora_a))
    adapter_b = LoRARequest("adapter-b", 2, str(lora_b))
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            enable_lora=True,
            max_loras=2,
            max_cpu_loras=2,
            max_lora_rank=16,
        )
    )
    try:
        base = _generate_with_llm(llm, prompts=prompt)
        first_a = _generate_with_llm(llm, prompts=prompt, lora_request=adapter_a)
        first_b = _generate_with_llm(llm, prompts=prompt, lora_request=adapter_b)
        second_a = _generate_with_llm(llm, prompts=prompt, lora_request=adapter_a)
    finally:
        _shutdown_llm(llm)
    return {
        "base": base,
        "adapter_a": first_a,
        "adapter_b": first_b,
        "adapter_a_again": second_a,
    }


def _case_spec_decode_parity(
    model_path: Path,
    *,
    draft_model: Path,
) -> dict[str, Any]:
    baseline = _generate(model_path, enforce_eager=True)
    speculative = _generate(
        model_path,
        enforce_eager=True,
        spec_model=str(draft_model),
        spec_tokens=2,
    )
    return {
        "baseline": baseline,
        "speculative": speculative,
    }


def _case_mtp_parity(model_path: Path) -> dict[str, Any]:
    baseline = _generate(model_path, enforce_eager=True)
    speculative = _generate(
        model_path,
        enforce_eager=True,
        spec_method="mtp",
        spec_tokens=1,
    )
    return {
        "baseline": baseline,
        "speculative": speculative,
    }


def _case_kv_transfer_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM
    from vllm.config.kv_transfer import KVTransferConfig

    storage_path = Path(
        os.environ.get(
            "VLLM_HCU_KV_TRANSFER_STORAGE",
            "/tmp/vllm-hcu-integration/kv-transfer",
        )
    )
    storage_path.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            disable_hybrid_kv_cache_manager=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "shared_storage_path": str(storage_path),
                },
            ),
        )
    )
    try:
        output = _generate_with_llm(
            llm,
            prompts=["KV transfer smoke test prompt:"],
            max_tokens=4,
        )
    finally:
        _shutdown_llm(llm)
    return {
        "connector": "ExampleConnector",
        "storage_path": str(storage_path),
        "output": output,
    }


def _case_prefix_caching_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM

    shared_prefix = (
        "You are checking prefix caching on HCU. "
        "Repeatable context: alpha beta gamma delta. "
    ) * 16
    prompts = [
        shared_prefix + "Question A: answer with one short word.",
        shared_prefix + "Question B: answer with one short word.",
    ]
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            enable_prefix_caching=True,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
        )
    )
    try:
        first = _generate_with_llm(llm, prompts=prompts, max_tokens=6)
        second = _generate_with_llm(llm, prompts=prompts, max_tokens=6)
    finally:
        _shutdown_llm(llm)
    return {
        "enable_prefix_caching": True,
        "first": first,
        "second": second,
    }


def _case_chunked_prefill_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM

    long_prompt = (
        "Chunked prefill should process this repeated HCU prompt safely. "
        "The model should continue after a long shared context. "
    ) * 20
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            enable_chunked_prefill=True,
            max_model_len=1024,
            max_num_batched_tokens=256,
            max_num_seqs=2,
        )
    )
    try:
        output = _generate_with_llm(llm, prompts=[long_prompt], max_tokens=4)
    finally:
        _shutdown_llm(llm)
    return {
        "enable_chunked_prefill": True,
        "output": output,
    }


def _case_logprobs_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM

    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=2,
        )
    )
    try:
        output = _generate_with_llm(
            llm,
            prompts=["Return a short deterministic answer for logprob testing."],
            max_tokens=5,
            logprobs=3,
            prompt_logprobs=2,
        )
    finally:
        _shutdown_llm(llm)
    return {
        "output": output,
    }


def _case_batch_mixed_lengths(model_path: Path) -> dict[str, Any]:
    from vllm import LLM

    prompts = [
        "Short prompt:",
        "Medium prompt: " + "count carefully " * 16,
        "Long prompt: " + "mixed length batch scheduling on HCU " * 48,
        "Tiny:",
    ]
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
        )
    )
    try:
        output = _generate_with_llm(llm, prompts=prompts, max_tokens=6)
    finally:
        _shutdown_llm(llm)
    return {
        "prompt_count": len(prompts),
        "output": output,
    }


def _case_embedding_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM

    prompts = [
        "HCU inference runtime",
        "HCU inference runtime",
        "A completely unrelated cooking recipe",
    ]
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            runner="pooling",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=4,
        )
    )
    try:
        outputs = llm.embed(prompts, use_tqdm=False)
        embeddings = [record.outputs.embedding for record in outputs]
    finally:
        _shutdown_llm(llm)

    def cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            raise AssertionError("embedding model returned a zero vector")
        return numerator / (left_norm * right_norm)

    return {
        "count": len(embeddings),
        "hidden_size": len(embeddings[0]),
        "identical_cosine": cosine(embeddings[0], embeddings[1]),
        "unrelated_cosine": cosine(embeddings[0], embeddings[2]),
        "all_finite": all(
            math.isfinite(value)
            for embedding in embeddings
            for value in embedding
        ),
    }


def _case_reranker_smoke(model_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the "
        "requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    query = "What is the capital of China?"
    documents = [
        "The capital of China is Beijing.",
        "Whales are marine mammals that live in oceans.",
    ]
    prompts = [
        prefix
        + "<Instruct>: Given a question, retrieve a passage that answers it\n"
        + f"<Query>: {query}\n<Document>: {document}"
        + suffix
        for document in documents
    ]
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=2,
        logprob_token_ids=[true_token, false_token],
        allowed_token_ids=[true_token, false_token],
    )
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=2,
        )
    )
    try:
        outputs = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=False,
        )
        scores: list[float] = []
        for record in outputs:
            logprobs = record.outputs[0].logprobs
            if not logprobs or not logprobs[0]:
                raise AssertionError("reranker did not return token logprobs")
            first = logprobs[0]
            missing = {true_token, false_token}.difference(first)
            if missing:
                raise AssertionError(
                    "reranker omitted requested label logprobs: "
                    f"missing={sorted(missing)}, returned={sorted(first)}"
                )
            true_logprob = first[true_token].logprob
            false_logprob = first[false_token].logprob
            true_score = math.exp(true_logprob)
            false_score = math.exp(false_logprob)
            scores.append(true_score / (true_score + false_score))
    finally:
        _shutdown_llm(llm)
    return {
        "scores": scores,
        "relevant_index": max(range(len(scores)), key=scores.__getitem__),
    }


def _case_vl_image_smoke(model_path: Path) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    image = Image.new("RGB", (64, 32), color=(220, 30, 30))
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "What is the dominant color? Answer with one word.",
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 1},
        )
    )
    try:
        records = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            SamplingParams(temperature=0, max_tokens=8, seed=0),
            use_tqdm=False,
        )
        output = _single_completion(records[0])
    finally:
        _shutdown_llm(llm)
    return {
        "image_size": list(image.size),
        "prompt_has_image_token": "image" in prompt.casefold(),
        "output": output,
    }


def _parallel_config_summary(llm: Any) -> dict[str, Any]:
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return {}
    return {
        "tensor_parallel_size": getattr(
            parallel_config,
            "tensor_parallel_size",
            None,
        ),
        "pipeline_parallel_size": getattr(
            parallel_config,
            "pipeline_parallel_size",
            None,
        ),
        "data_parallel_size": getattr(
            parallel_config,
            "data_parallel_size",
            None,
        ),
        "all2all_backend": getattr(
            parallel_config,
            "all2all_backend",
            None,
        ),
        "enable_expert_parallel": getattr(
            parallel_config,
            "enable_expert_parallel",
            None,
        ),
        "world_size": getattr(parallel_config, "world_size", None),
    }


def _case_tp_ep_smoke(
    model_path: Path,
    *,
    tensor_parallel_size: int,
    data_parallel_size: int,
    gpu_memory_utilization: float,
    all2all_backend: str | None,
    moe_backend: str,
) -> dict[str, Any]:
    if data_parallel_size > 1:
        return _case_tp_ep_smoke_data_parallel(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            all2all_backend=all2all_backend,
            moe_backend=moe_backend,
        )

    return _case_tp_ep_smoke_rank(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        all2all_backend=all2all_backend,
        moe_backend=moe_backend,
    )


def _case_tp_ep_smoke_rank(
    model_path: Path,
    *,
    tensor_parallel_size: int,
    data_parallel_size: int,
    gpu_memory_utilization: float,
    all2all_backend: str | None,
    moe_backend: str,
) -> dict[str, Any]:
    from vllm import LLM

    max_num_batched_tokens = (
        300 if all2all_backend == "deepep_low_latency" else 512
    )
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size,
            all2all_backend=all2all_backend,
            enable_expert_parallel=True,
            max_model_len=512,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=2,
            gpu_memory_utilization=gpu_memory_utilization,
            moe_backend=moe_backend,
        )
    )
    try:
        output = _generate_with_llm(
            llm,
            prompts=[
                "TP and expert parallel smoke test:",
                "Answer briefly: HCU parallel execution is",
            ],
            max_tokens=4,
        )
        parallel_config = _parallel_config_summary(llm)
    finally:
        _shutdown_llm(llm)
    return {
        "requested_tensor_parallel_size": tensor_parallel_size,
        "requested_data_parallel_size": data_parallel_size,
        "requested_all2all_backend": all2all_backend,
        "requested_enable_expert_parallel": True,
        "requested_gpu_memory_utilization": gpu_memory_utilization,
        "requested_moe_backend": moe_backend,
        "parallel_config": parallel_config,
        "output": output,
    }


def _tp_ep_data_parallel_rank(
    local_dp_rank: int,
    data_parallel_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    model_path: Path,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    all2all_backend: str | None,
    moe_backend: str,
    result_queue: Any,
    process_group_id: Any,
    process_group_lock: Any,
    process_group_ready: Any,
    start_gate: Any,
) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    with process_group_lock:
        if process_group_id.value == 0:
            os.setpgid(0, 0)
            process_group_id.value = os.getpid()
        else:
            os.setpgid(0, process_group_id.value)
    process_group_ready.set()
    start_gate.wait()
    os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(data_parallel_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    result = _case_tp_ep_smoke_rank(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        all2all_backend=all2all_backend,
        moe_backend=moe_backend,
    )
    result_queue.put((local_dp_rank, result))


def _case_tp_ep_smoke_data_parallel(
    model_path: Path,
    *,
    tensor_parallel_size: int,
    data_parallel_size: int,
    gpu_memory_utilization: float,
    all2all_backend: str | None,
    moe_backend: str,
) -> dict[str, Any]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        dp_master_port = int(listener.getsockname()[1])

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process_group_id = context.Value("q", 0)
    process_group_lock = context.Lock()
    process_group_ready = [context.Event() for _ in range(data_parallel_size)]
    start_gate = context.Event()
    processes = [
        context.Process(
            target=_tp_ep_data_parallel_rank,
            args=(
                rank,
                data_parallel_size,
                "127.0.0.1",
                dp_master_port,
                model_path,
                tensor_parallel_size,
                gpu_memory_utilization,
                all2all_backend,
                moe_backend,
                result_queue,
                process_group_id,
                process_group_lock,
                process_group_ready[rank],
                start_gate,
            ),
        )
        for rank in range(data_parallel_size)
    ]
    started_processes = []
    completed = False
    termination_handler = _DataParallelTerminationHandler()
    previous_signal_handlers = {
        sig: signal.signal(sig, termination_handler)
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for process in processes:
            started_processes.append(process)
            termination_handler.defer_termination = True
            try:
                process.start()
            finally:
                termination_handler.defer_termination = False
            termination_handler.raise_if_deferred()

        for process, ready in zip(processes, process_group_ready, strict=True):
            while not ready.wait(timeout=0.1):
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"data-parallel rank process {process.pid} failed "
                        f"before process-group setup with exit code "
                        f"{process.exitcode}"
                    )
        start_gate.set()

        pending = set(processes)
        while pending:
            for process in tuple(pending):
                process.join(timeout=0.1)
                if process.exitcode is None:
                    continue
                pending.remove(process)
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"data-parallel rank process {process.pid} failed with "
                        f"exit code {process.exitcode}"
                    )

        results = dict(result_queue.get(timeout=10) for _ in processes)
        if len(results) != data_parallel_size:
            raise RuntimeError(
                f"expected {data_parallel_size} data-parallel rank results, "
                f"got {len(results)}"
            )
        completed = True
        return results[0]
    finally:
        try:
            if not completed:
                termination_handler.defer_termination = True
                _terminate_data_parallel_process_groups(
                    started_processes,
                    process_group_ready[: len(started_processes)],
                    process_group_id,
                )
        finally:
            for sig, previous_handler in previous_signal_handlers.items():
                signal.signal(sig, previous_handler)
            termination_handler.defer_termination = False
        termination_handler.raise_if_deferred()


def _terminate_data_parallel_process_groups(
    processes: list[Any],
    process_group_ready: list[Any],
    process_group_id: Any,
) -> None:
    if len(processes) != len(process_group_ready):
        raise RuntimeError("rank process/group-ready tracking is inconsistent")

    started_processes = [
        process for process in processes if process.pid is not None
    ]
    unready_processes = [
        process
        for process, ready in zip(processes, process_group_ready, strict=True)
        if process.pid is not None and not ready.is_set()
    ]

    for process in unready_processes:
        if process.is_alive():
            process.terminate()

    owned_group_id = int(process_group_id.value)
    if owned_group_id > 1:
        _terminate_owned_process_groups(
            [owned_group_id],
            process_leaders=started_processes,
            term_timeout_s=10,
            kill_timeout_s=5,
        )

    for process in started_processes:
        process.join(timeout=10)
    survivors = [process for process in started_processes if process.is_alive()]
    for process in survivors:
        process.kill()
    for process in survivors:
        process.join(timeout=10)
    survivors = [
        process.pid for process in started_processes if process.is_alive()
    ]
    if survivors:
        raise RuntimeError(
            f"data-parallel rank parents survived cleanup: {survivors}"
        )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=(
            "smoke",
            "graph-parity",
            "lora-switching",
            "mtp-parity",
            "spec-decode-parity",
            "kv-transfer-smoke",
            "prefix-caching-smoke",
            "chunked-prefill-smoke",
            "logprobs-smoke",
            "batch-mixed-lengths",
            "embedding-smoke",
            "reranker-smoke",
            "vl-image-smoke",
            "tp-ep-smoke",
        ),
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--lora-a", type=Path)
    parser.add_argument("--lora-b", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--all2all-backend", default=None)
    parser.add_argument("--moe-backend", default="auto")
    args = parser.parse_args(argv)

    if args.case == "smoke":
        payload = _case_smoke(args.model)
    elif args.case == "graph-parity":
        payload = _case_graph_parity(args.model)
    elif args.case == "lora-switching":
        if args.lora_a is None or args.lora_b is None:
            raise SystemExit("lora-switching requires --lora-a and --lora-b")
        payload = _case_lora_switching(
            args.model,
            lora_a=args.lora_a,
            lora_b=args.lora_b,
        )
    elif args.case == "spec-decode-parity":
        if args.draft_model is None:
            raise SystemExit("spec-decode-parity requires --draft-model")
        payload = _case_spec_decode_parity(args.model, draft_model=args.draft_model)
    elif args.case == "mtp-parity":
        payload = _case_mtp_parity(args.model)
    elif args.case == "kv-transfer-smoke":
        payload = _case_kv_transfer_smoke(args.model)
    elif args.case == "prefix-caching-smoke":
        payload = _case_prefix_caching_smoke(args.model)
    elif args.case == "chunked-prefill-smoke":
        payload = _case_chunked_prefill_smoke(args.model)
    elif args.case == "logprobs-smoke":
        payload = _case_logprobs_smoke(args.model)
    elif args.case == "batch-mixed-lengths":
        payload = _case_batch_mixed_lengths(args.model)
    elif args.case == "embedding-smoke":
        payload = _case_embedding_smoke(args.model)
    elif args.case == "reranker-smoke":
        payload = _case_reranker_smoke(args.model)
    elif args.case == "vl-image-smoke":
        payload = _case_vl_image_smoke(args.model)
    else:
        payload = _case_tp_ep_smoke(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            all2all_backend=args.all2all_backend,
            moe_backend=args.moe_backend,
        )
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
