from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "validation" / "v0.32.0" / "compat-lock.json"
OLD_ADAPTER_DISTRIBUTION = "hcu-vllm" + "-adapter"
OLD_ADAPTER_NAMESPACE = "hcu_vllm" + "_adapter"


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"final_commit", "peer_final_commit"}
            or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def test_plugin_compat_lock_pins_vllm032_pair_without_circular_commits() -> None:
    document = json.loads(LOCK.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert (
        document["validation_id"]
        == "vllm-v0.32.0-upstream-first-plugin-20261030"
    )
    assert document["status"] in {"pending", "passed"}
    assert document["repository"] == {
        "name": "vllm-plugin-das",
        "url": "https://github.com/alexanderbin123/vllm-plugin-das.git",
        "branch": "feat/vllm032-plugin-onoff-gates",
        "source_base": "e76220d9c0006cc3c9d886587a4771a31e7d83e8",
        "test_policy_commit": "a7f5c2bd15c19296ab00219f188b3b7d65b45f50",
    }
    assert document["peer"] == {
        "name": "adapter-das",
        "url": "https://github.com/alexanderbin123/adapter-das.git",
        "branch": "feat/vllm032-adapter-delivery-gates",
        "source_base": "c99886d41f435e594050d47663e03ea94081fdc6",
    }
    assert document["vllm"] == {
        "version": "0.32.0",
        "commit": "5df63a7203cdb9328d824a27b682e77c7b9928be",
    }
    assert document["human_eval"]["commit"] == (
        "6d43fb980f9fee3c892a914eda09951f772ad10d"
    )
    assert document["hardware"]["architecture"] == "gfx936"
    assert document["artifacts"]["plugin_wheel"] == {
        "filename": "hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl",
        "sha256": (
            "67db8901f289e5e9cc7ee8476be2e9c13980225d185cfe32665d9af8873893b9"
        ),
        "build_commit": "a7f5c2bd15c19296ab00219f188b3b7d65b45f50",
        "capabilities": ["aiter-int8-moe", "lightop-rms-norm"],
    }
    assert document["gates"]["static_portable_tests"] == {
        "status": "pass",
        "result": "100 passed, 6 live-HCU tests skipped by default",
    }
    assert document["gates"]["build"]["status"] == "pass"
    assert document["gates"]["installed_integration"]["status"] == "pass"
    assert document["gates"]["live_hardware_operators"] == {
        "status": "pass",
        "result": (
            "exact a7f5c2bd/67db8901 wheel on gfx936 card 0: RMSNorm "
            "eager/compile, ON/OFF graph, and AITER INT8 M=1/16/128 "
            "numerics plus graph replay passed; 6 passed, 14 warnings in 112.02s"
        ),
    }
    assert document["gates"]["serving"]["status"] == "pending"
    assert document["gates"]["model_parity"]["status"] == "pending"
    assert document["gates"]["humaneval_prefix"]["status"] == "pending"
    assert document["human_eval"]["plugin_on"]["passed"] == 61
    assert document["human_eval"]["plugin_off"]["passed"] == 60
    assert not _contains_forbidden_key(document)


def test_runtime_build_verifier_and_vllm032_contracts_drop_old_adapter() -> None:
    targets = (
        ROOT / "adapter_extension",
        ROOT / "tests" / "vllm032",
        LOCK,
    )
    offenders: list[str] = []
    for target in targets:
        paths = (target,) if target.is_file() else target.rglob("*")
        for path in paths:
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix not in {".json", ".md", ".py", ".toml"}
            ):
                continue
            text = path.read_text(encoding="utf-8")
            if OLD_ADAPTER_DISTRIBUTION in text or OLD_ADAPTER_NAMESPACE in text:
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
