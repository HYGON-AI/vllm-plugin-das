# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared P/D-disaggregated vLLM + EvalScope integration-test runner."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.integration.server.evalscope_server import _server_environment


@dataclass(frozen=True)
class PDCommands:
    """Commands, environments, and endpoints for one P/D test topology."""

    prefill: list[str]
    decode: list[str]
    proxy: list[str]
    prefill_env: dict[str, str]
    decode_env: dict[str, str]
    proxy_env: dict[str, str]
    host: str
    proxy_port: int
    prefill_port: int
    decode_port: int
    startup_timeout_s: int
    shutdown_timeout_s: int


def _role_command(
    *,
    model: str,
    served_model_name: str,
    common_args: list[str],
    role: dict[str, Any],
) -> list[str]:
    return [
        "vllm",
        "serve",
        model,
        *common_args,
        *(str(item) for item in role.get("args", [])),
        "--served-model-name",
        served_model_name,
        "--port",
        str(role["port"]),
        "--data-parallel-rpc-port",
        str(role["data_parallel_rpc_port"]),
    ]


def _role_environment(
    base: dict[str, str],
    *,
    host: str,
    role: dict[str, Any],
) -> dict[str, str]:
    env = base.copy()
    env.update(
        {
            "HIP_VISIBLE_DEVICES": str(role["visible_devices"]),
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(role["bootstrap_port"]),
            "VLLM_DP_MASTER_IP": host,
            "VLLM_DP_MASTER_PORT": str(role["data_parallel_master_port"]),
        }
    )
    return env


def pd_commands(config: dict[str, Any], *, model_env: str) -> PDCommands:
    """Build the official-style P, D, and Mooncake proxy commands."""

    pd = config["pd"]
    host = str(pd.get("host", "127.0.0.1"))
    model = os.environ.get(model_env, str(config["model"]))
    served_model_name = str(config["served_model_name"])
    common_args = [str(item) for item in pd.get("common_args", [])]
    prefill = pd["prefill"]
    decode = pd["decode"]

    base_env = _server_environment()
    environment = pd.get("environment", {})
    if not isinstance(environment, dict):
        raise TypeError("pd.environment must be a mapping")
    base_env.update({str(name): str(value) for name, value in environment.items()})

    source_root_value = os.environ.get("VLLM_V0251_SOURCE_ROOT")
    if source_root_value is None:
        raise FileNotFoundError(
            "VLLM_V0251_SOURCE_ROOT must identify the vLLM v0.25.1 source tree"
        )
    proxy_script = (
        Path(source_root_value)
        / "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    if not proxy_script.is_file():
        raise FileNotFoundError(f"Mooncake connector proxy is absent: {proxy_script}")

    proxy_port = int(pd["proxy_port"])
    prefill_port = int(prefill["port"])
    decode_port = int(decode["port"])
    proxy = [
        sys.executable,
        str(proxy_script),
        "--prefill",
        f"http://{host}:{prefill_port}",
        str(prefill["bootstrap_port"]),
        "--decode",
        f"http://{host}:{decode_port}",
        "--host",
        host,
        "--port",
        str(proxy_port),
    ]

    return PDCommands(
        prefill=_role_command(
            model=model,
            served_model_name=served_model_name,
            common_args=common_args,
            role=prefill,
        ),
        decode=_role_command(
            model=model,
            served_model_name=served_model_name,
            common_args=common_args,
            role=decode,
        ),
        proxy=proxy,
        prefill_env=_role_environment(base_env, host=host, role=prefill),
        decode_env=_role_environment(base_env, host=host, role=decode),
        proxy_env=base_env.copy(),
        host=host,
        proxy_port=proxy_port,
        prefill_port=prefill_port,
        decode_port=decode_port,
        startup_timeout_s=int(pd.get("startup_timeout_s", 3600)),
        shutdown_timeout_s=int(pd.get("shutdown_timeout_s", 60)),
    )
