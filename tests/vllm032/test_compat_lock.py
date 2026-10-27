from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "validation" / "v0.32.0" / "compat-lock.json"


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
    assert document["validation_id"] == "vllm-v0.32.0-gfx936-20261027"
    assert document["status"] in {"pending", "passed"}
    assert document["repository"] == {
        "name": "vllm-plugin-das",
        "url": "https://github.com/alexanderbin123/vllm-plugin-das.git",
        "branch": "compat/vllm-v0.32.0",
        "source_base": "1be8d694edff552eebe8b3b9a9c3b2e44235d486",
        "test_policy_commit": "af84424cf652893fbbf392f9e1759ff19241ff96",
    }
    assert document["peer"] == {
        "name": "hcu-vllm-adapter",
        "url": "https://github.com/alexanderbin123/hcu-vllm-adapter.git",
        "branch": "compat/vllm-v0.32.0",
        "source_base": "7f9fa4d5a2b38d2e6d848bbfaa9bb2679fe86ff1",
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
