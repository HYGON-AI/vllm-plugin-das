"""CPU checks for the standard-plugin Step5 W4A8 P/D recipes."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "benchmarks/step5_w4a8_pd"
PLUGIN = REPO / "vllm_hcu/model_executor/layers/fused_moe"
spec = importlib.util.spec_from_file_location("step5_launch", ROOT / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def arguments(tmp_path, role="prefill"):
    model = tmp_path / "model"
    model.mkdir()
    return SimpleNamespace(
        role=role,
        model=model,
        tokenizer=None,
        work_dir=tmp_path / "work",
        host="127.0.0.1",
        port=8000,
    )


def profile_command(tmp_path, role):
    """Execute the real Bash argument assembly with an inert Python executable."""
    fake = tmp_path / "bin"
    fake.mkdir()
    python = fake / "python3"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'arg:%s\\n' \"$@\"\n"
        "printf 'env:HCU_DBO_MIN_TOKENS=%s\\n' \"$HCU_DBO_MIN_TOKENS\"\n"
        "printf 'env:HCU_EP_DIRECT_SUM=%s\\n' \"${HCU_EP_DIRECT_SUM-}\"\n"
        "printf 'env:VLLM_HCU_STEP5_W4A8_PROFILE=%s\\n' \"$VLLM_HCU_STEP5_W4A8_PROFILE\"\n"
    )
    python.chmod(0o755)
    run = tmp_path / "run"
    run.mkdir()
    (run / "guard-identity.json").write_text("{}")
    environment = dict(
        os.environ,
        PATH=str(fake) + os.pathsep + os.environ["PATH"],
        STEP5_W4_BUDGET_GUARD="CPU-test",
        MODEL_PATH="model with spaces",
        TOKENIZER_PATH="tokenizer with spaces",
        STEP5_EXP_RUN=str(run),
        PLACEMENT_PATH=str(ROOT / "assets/expert_placement_prefill.pt"),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "profiles" / (role + ".sh"))],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    tokens = [
        line[4:] for line in result.stdout.splitlines() if line.startswith("arg:")
    ]
    exported = dict(
        line[4:].split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith("env:")
    )
    return tokens, exported


@pytest.mark.parametrize(
    "role,tp,dp,budget,sequences",
    [("prefill", "1", "8", "16384", "64"), ("decode", "4", "2", "4096", "32")],
)
def test_profiles_preserve_measured_settings(tmp_path, role, tp, dp, budget, sequences):
    tokens, environment = profile_command(tmp_path, role)

    def value(flag):
        return tokens[tokens.index(flag) + 1]

    assert value("--tensor-parallel-size") == tp and value("--data-parallel-size") == dp
    assert (
        value("--max-num-batched-tokens") == budget
        and value("--max-num-seqs") == sequences
    )
    assert value("--all2all-backend") == "allgather_reducescatter"
    assert "--worker-cls" not in tokens
    assert json.loads(value("--speculative-config"))["synthetic_acceptance_rates"] == [
        0.838992,
        0.598322,
        0.362686,
    ]
    assert (
        json.loads(value("--compilation-config"))["cudagraph_mode"]
        == "FULL_AND_PIECEWISE"
    )
    assert environment["VLLM_HCU_STEP5_W4A8_PROFILE"] == role


def test_plan_has_no_external_library_or_reservation(tmp_path):
    command, env, run = launch.plan(arguments(tmp_path))
    assert not any(
        "native-manifest" in item or "build_native" in item for item in command
    )
    assert not any(k.startswith("STEP5_NATIVE_") for k in env)
    assert env["STEP5_RESERVED_BYTES_PER_CARD"] == "0" and not run.exists()


def test_dry_run_needs_no_build_manifest(tmp_path):
    args = arguments(tmp_path, "decode")
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "launch.py"),
            "--role",
            "decode",
            "--model",
            str(args.model),
            "--work-dir",
            str(args.work_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not json.loads(p.stdout)["model_started"] and not args.work_dir.exists()


@pytest.mark.parametrize(
    "guard,child,expected", [(0, 0, 0), (0, 7, 7), (0, -15, 143), (1, 0, 1)]
)
def test_child_exit_is_preserved(tmp_path, guard, child, expected):
    (tmp_path / "guard-result.json").write_text(json.dumps({"returncode": child}))
    assert launch.service_exit_code(guard, tmp_path) == expected


def test_no_jit_extension_or_custom_worker_remains():
    assert not (ROOT / "native").exists() and not (ROOT / "runtime").exists()
    assert not (ROOT / "build_native.py").exists()
    for p in ROOT.rglob("*.py"):
        assert (
            "cpp_extension" not in p.read_text()
            and "spec_from_file_location" not in p.read_text()
        )
    for p in ROOT.rglob("*.sh"):
        assert "LOCAL_EXT" not in p.read_text() and "--worker-cls" not in p.read_text()


def test_placement_asset():
    assert (
        hashlib.sha256(
            (ROOT / "assets/expert_placement_prefill.pt").read_bytes()
        ).hexdigest()
        == "67b40ac51bc1828a6ba1a0012de31892f7966d608d754d213b64ec5015e38873"
    )


def test_numeric_kernels_are_unchanged_by_deduplication():
    expected = {
        "_row_value": (
            "kernels.py",
            "c2a327c70ad49d8dc7ad453f118fdabc251d640bcf1730294c62fcf1cbe73aec",
        ),
        "activation_persistent": (
            "kernels.py",
            "ecf409a97ee9eff20795ed1bf54d0292835c1069bd98e5b5fb4273b0dd333489",
        ),
        "_small_route_meta": (
            "kernels.py",
            "c8bc2cbcf0081f8f32ca3b3ed078c773e5c56847fddb26c4933a78ccf996068e",
        ),
        "_small_route_gather": (
            "kernels.py",
            "61bd1bf201b20000f22136df7153c66c8fbf313c010871086c7fa8550cefb874",
        ),
        "_activation": (
            "kernels.py",
            "d6b24feca8aa4a84eda81b379610dded647868adf3ccdca38f68208da7bec15a",
        ),
        "_weighted_output": (
            "kernels.py",
            "e23b4ff3c0a1e33454211493f2f84b43edf407533736f17fd398343173500c09",
        ),
        "_histogram": (
            "routing.py",
            "11cc39a92afde8b0923ae713306b44d3363e9c96f722cf01f52390c05dccacd4",
        ),
        "_block_prefixes": (
            "routing.py",
            "8d3836416ee3e925e2d4e417ef0b410e9f119f5c6fac92f4b123be9a08032d6c",
        ),
        "_assign_positions": (
            "routing.py",
            "0382b7be1a616975ec7bc03dfd1a0a32792cae9a25c9f9cb3d5ab0bb24853a20",
        ),
        "_gather_tokens": (
            "routing.py",
            "55108bbe92c172ab9b6a7dbc68727489af7a5eeca21a4d2c69a89bf757ff16a5",
        ),
    }
    found = {}
    for filename in ("step5_w4a8_kernels.py", "step5_w4a8_routing.py"):
        for node in ast.parse((PLUGIN / filename).read_text(encoding="utf8")).body:
            if isinstance(node, ast.FunctionDef):
                found[node.name] = hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode()
                ).hexdigest()
    for name, (filename, digest) in expected.items():
        assert found[name] == digest, (filename, name)
