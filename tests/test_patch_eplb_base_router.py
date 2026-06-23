from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = (
    ROOT
    / "vllm_hcu/patches/"
    / "vllm__model_executor__layers__fused_moe__router__base_router.patch.py"
)
ENVS_FILE = ROOT / "vllm_hcu/platforms/envs.py"


def test_eplb_base_router_patch_selects_torch_fallback_by_env() -> None:
    source = PATCH_FILE.read_text(encoding="utf-8")

    assert "Patch for vllm.model_executor.layers.fused_moe.router.base_router" in source
    assert "safe_offs = tl.minimum(offs, numel - 1)" not in source
    assert "tl.load(topk_ids_ptr + safe_offs" not in source
    assert "tl.store(out_ids_ptr + safe_offs" not in source
    assert "safe_physical_id = tl.where(valid, physical_id, 0)" not in source
    assert "from vllm_hcu.platforms import envs as henvs" in source
    assert "henvs.VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD" in source
    assert "return _eplb_map_and_record_triton(" in source
    assert "HCU: optionally use torch ops instead of the fused Triton EPLB kernel" in source
    assert "expert_load_view.scatter_add_" in source
    assert "return physical_flat.reshape(topk_shape)" in source


def test_env_registers_torch_eplb_map_record_default_disabled() -> None:
    source = ENVS_FILE.read_text(encoding="utf-8")

    assert "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD: bool = False" in source
    assert '"VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD"' in source
    assert 'os.environ.get("VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD", "False")' in source
