"""The EP16 recipe must reject the exact E44-only deployment regression."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "step5_ep_table_check", ROOT / "benchmarks/step5/check_ep_table.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
@pytest.fixture(autouse=True)
def external_table(tmp_path, monkeypatch):
    global TABLE
    TABLE = tmp_path / "ep_table.json"
    pair = [{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}] * 2
    seeds = {"256": pair, "32768": pair}
    TABLE.write_text(json.dumps({f"E={e},N=3072,K=4096,topk=8":
                               dict(seeds, **({"65536": pair} if e == 23 else {}))
                               for e in (22,23,44)}))


@pytest.mark.parametrize("ep_size", [8, 16])
def test_shipped_table_covers_decode_and_prefill(ep_size):
    result = CHECK.check_ep_table(TABLE, ep_size)
    assert result["key"] == f"E={352 // ep_size},N=3072,K=4096,topk=8"
    assert result["selected"]["256"]["table_m"] == 256
    assert result["selected"]["32768"]["table_m"] == 32768


def test_e44_only_package_is_rejected(tmp_path):
    data = json.loads(TABLE.read_text())
    del data[CHECK.KEY]
    path = tmp_path / "ep_table.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="EP16 requires"):
        CHECK.check_ep_table(path)

def test_ep8_requires_e44_even_when_ep16_is_present(tmp_path):
    data = json.loads(TABLE.read_text())
    del data["E=44,N=3072,K=4096,topk=8"]
    path = tmp_path / "ep_table.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="EP8 requires"):
        CHECK.check_ep_table(path, 8)


@pytest.mark.parametrize("ep_size", [8, 16])
@pytest.mark.parametrize("pair", [[], [{}, {}], [
    {"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":256},
    {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256},
]])
def test_invalid_config_pairs_fail_before_model_start(tmp_path, pair, ep_size):
    path = tmp_path / "bad.json"
    key = f"E={352 // ep_size},N=3072,K=4096,topk=8"
    path.write_text(json.dumps({key: {"256": pair}}))
    with pytest.raises(ValueError):
        CHECK.check_ep_table(path, ep_size)


def test_e23_requires_its_own_table_and_explicit_hot_row(tmp_path):
    data = json.loads(TABLE.read_text())
    data.pop("E=23,N=3072,K=4096,topk=8", None)
    path = tmp_path / "e23.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="E=23"):
        CHECK.check_ep_table(path, 16, 368, 65536)
    data["E=23,N=3072,K=4096,topk=8"] = data[CHECK.KEY].copy()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="explicit tuned row at M=65536"):
        CHECK.check_ep_table(path, 16, 368, 65536)
    data["E=23,N=3072,K=4096,topk=8"]["65536"] = data[CHECK.KEY]["32768"]
    path.write_text(json.dumps(data))
    result = CHECK.check_ep_table(path, 16, 368, 65536)
    assert result["physical_experts"] == 368 and result["local_experts"] == 23
    assert result["selected"]["65536"]["table_m"] == 65536
    assert CHECK.check_ep_table(path)["key"] == CHECK.KEY


def test_no_machine_table_is_shipped_in_the_source_tree():
    assert not (ROOT / "vllm_hcu/model_executor/layers/fused_moe/ep_table.json").exists()
    assert not (ROOT / "benchmarks/step5/ep_table_decode_candidate.json").exists()


@pytest.mark.parametrize("physical_experts,require_m", [(True, None), (353, None), (368, 0), (368, True)])
def test_invalid_physical_count_or_hot_row_is_rejected(physical_experts, require_m):
    with pytest.raises(ValueError):
        CHECK.check_ep_table(TABLE, 16, physical_experts, require_m)
