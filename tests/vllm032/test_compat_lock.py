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
    assert document["validation_id"] == "vllm-v0.32.0-adapter-das-20261028"
    assert document["status"] in {"pending", "passed"}
    assert document["repository"] == {
        "name": "vllm-plugin-das",
        "url": "https://github.com/alexanderbin123/vllm-plugin-das.git",
        "branch": "compat/vllm-v0.32.0",
        "source_base": "1be8d694edff552eebe8b3b9a9c3b2e44235d486",
        "test_policy_commit": "af84424cf652893fbbf392f9e1759ff19241ff96",
    }
    assert document["peer"] == {
        "name": "adapter-das",
        "url": "https://github.com/alexanderbin123/adapter-das.git",
        "branch": "feat/vllm032-runtime-integration",
        "source_base": "81361880e203cb33cf631f77716ac8244f5f299f",
    }
    assert document["vllm"] == {
        "version": "0.32.0",
        "commit": "5df63a7203cdb9328d824a27b682e77c7b9928be",
    }
    assert document["human_eval"]["commit"] == (
        "6d43fb980f9fee3c892a914eda09951f772ad10d"
    )
    assert document["hardware"]["architecture"] == "gfx936"
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
