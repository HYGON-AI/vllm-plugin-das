"""Execute production derivation and factory contracts without accelerator imports."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "adapter_extension/src/hcu_vllm_das/models"


def test_standard_and_flash_have_distinct_attention_factories():
    path = MODELS / "hy4_dfly.py"
    assert path.exists()
    tree = ast.parse(path.read_text(encoding="utf8"))
    attention = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HYV4StandardDFlyAttention"
    )
    assert [ast.unparse(b) for b in attention.bases] == ["HYV4StandardAttention"]
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name == "forward" for n in attention.body
    )
    flash = ast.parse((MODELS / "hy_v4_flash/dfly.py").read_text(encoding="utf8"))
    decoder = next(
        n
        for n in flash.body
        if isinstance(n, ast.ClassDef) and n.name == "HYV4DFlyDecoderLayer"
    )
    assert any(
        isinstance(n, ast.Attribute) and n.attr == "attention_cls"
        for n in ast.walk(decoder)
    )


def test_standard_config_routes_without_weakening_flash_mm_gate():
    source = (MODELS / "hy_v4_flash/dfly_config.py").read_text(encoding="utf8")
    assert "derive_standard_dfly_config" in source
    assert 'family == "hy_v4_flash" and target.supports_multimodal_inputs' in source
    assert '"hcu_hy4_ct_w4a8"' in source


def test_standard_draft_retains_hcu_int8_quant_method():
    tree = ast.parse((MODELS / "hy4_dfly.py").read_text(encoding="utf8"))
    quant = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "HYV4StandardDFlyQuantizationConfig"
    )
    assert [ast.unparse(b) for b in quant.bases] == [
        "HYV4DFlyQuantizationConfig",
        "HcuHY4CTW4A8Config",
    ]
