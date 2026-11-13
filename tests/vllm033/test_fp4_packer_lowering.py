# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU/mock contracts for the opt-in E2M1 packer lowering.

Run with hidden devices (no GPU initialisation):
  HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= python3 -B -m pytest \
      tests/vllm033/test_fp4_packer_lowering.py -q

Evidence level: spec-consistent, hardware-unverified. Expected codes below are
literals derived from the E2M1 grid, not produced by the implementation. The
Triton kernel is audited structurally and its arithmetic is transcribed for
ordering checks; it is NOT compiled or executed here. Native compilation,
kernel hit, Graph replay and model behaviour remain unverified.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import struct
import types

import pytest

from hcu_adapter.extensions import ExtensionContext

from hcu_vllm_das import fp4_packer as packer

FLT_MAX = 3.4028234663852886e38
CORE = "vllm.models.deepseek_v4.common.ops.fused_indexer_q"


# ---------------------------------------------------------------------------
# helpers: bit-exact FP32 neighbours and a mechanical transcription
# ---------------------------------------------------------------------------

def _raw(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _from_raw(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _neighbour(value: float, step: int) -> float:
    """Adjacent FP32 value; step=-1 toward zero, +1 away from it."""
    return _from_raw(_raw(value) + step)


def _simulate(x: float) -> int:
    """Line-by-line transcription of fp4_packer_triton.e2m1_code.

    A transcription for ordering checks only: every asserted value in this
    file is a literal, so this can never be its own oracle.
    """
    bits = _raw(x)
    sign = (bits >> 31) & 1
    magnitude = abs(x)
    magnitude = 0.0 if magnitude != magnitude else magnitude   # NaN policy
    magnitude = min(magnitude, 6.0)                            # saturation
    code = 0
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += 1 if magnitude >= boundary else 0
    if magnitude in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) and (code & 1):
        code -= 1                                              # tie to even code
    return code | (sign << 3)


def _pack(low: float, high: float) -> int:
    return _simulate(low) | (_simulate(high) << 4)


# ---------------------------------------------------------------------------
# 1. explicit E2M1 code table (low-nibble slot, high nibble held at zero)
# ---------------------------------------------------------------------------

GRID = (
    (0.0, 0x00), (-0.0, 0x08),
    (0.5, 0x01), (-0.5, 0x09),
    (1.0, 0x02), (-1.0, 0x0A),
    (1.5, 0x03), (-1.5, 0x0B),
    (2.0, 0x04), (-2.0, 0x0C),
    (3.0, 0x05), (-3.0, 0x0D),
    (4.0, 0x06), (-4.0, 0x0E),
    (6.0, 0x07), (-6.0, 0x0F),
)


@pytest.mark.parametrize(("value", "expected"), GRID)
def test_grid_value_codes(value: float, expected: int) -> None:
    assert _pack(value, 0.0) == expected


# Ties round to the even magnitude code: 0, 2, 4, 6 (never 1, 3, 5, 7).
TIES = (
    (0.25, 0x00), (-0.25, 0x08),
    (0.75, 0x02), (-0.75, 0x0A),
    (1.25, 0x02), (-1.25, 0x0A),
    (1.75, 0x04), (-1.75, 0x0C),
    (2.5, 0x04), (-2.5, 0x0C),
    (3.5, 0x06), (-3.5, 0x0E),
    (5.0, 0x06), (-5.0, 0x0E),
)


@pytest.mark.parametrize(("value", "expected"), TIES)
def test_tie_rounds_to_even_magnitude_code(value: float, expected: int) -> None:
    assert _pack(value, 0.0) == expected


BELOW_TIE = (
    (0.25, 0x00), (0.75, 0x01), (1.25, 0x02), (1.75, 0x03),
    (2.5, 0x04), (3.5, 0x05), (5.0, 0x06),
)


@pytest.mark.parametrize(("tie", "expected"), BELOW_TIE)
def test_just_below_tie_keeps_lower_code(tie: float, expected: int) -> None:
    below = _neighbour(tie, -1)
    assert _raw(below) != _raw(tie), "neighbour must differ in FP32 bits"
    assert _pack(below, 0.0) == expected
    assert _pack(-below, 0.0) == expected | 0x08


@pytest.mark.parametrize(("tie", "expected"), BELOW_TIE)
def test_just_above_tie_takes_upper_code(tie: float, expected: int) -> None:
    above = _neighbour(tie, +1)
    assert _raw(above) != _raw(tie), "neighbour must differ in FP32 bits"
    upper = expected + 1
    assert _pack(above, 0.0) == upper
    assert _pack(-above, 0.0) == upper | 0x08


def test_nibble_order_low_is_the_first_argument() -> None:
    # A swapped nibble is a silent value change, so pin both directions.
    assert _pack(0.0, 0.0) == 0x00
    assert _pack(0.5, 1.0) == 0x21
    assert _pack(1.0, 0.5) == 0x12
    assert _pack(-0.5, 0.5) == 0x19
    assert _pack(0.5, -0.5) == 0x91
    assert _pack(-0.0, 0.0) == 0x08
    assert _pack(0.0, -0.0) == 0x80
    assert _pack(6.0, -6.0) == 0xF7


# ---------------------------------------------------------------------------
# 2. saturation, special values, subnormals
# ---------------------------------------------------------------------------

SATURATION = (
    (7.0, 0x07), (-7.0, 0x0F),
    (2.0**28, 0x07), (-(2.0**28), 0x0F),
    (FLT_MAX, 0x07), (-FLT_MAX, 0x0F),
    (float("inf"), 0x07), (float("-inf"), 0x0F),
)


@pytest.mark.parametrize(("value", "expected"), SATURATION)
def test_out_of_range_saturates_to_maxnorm_with_sign(value: float, expected: int) -> None:
    assert _pack(value, 0.0) == expected


def test_large_magnitudes_do_not_collapse_to_zero() -> None:
    # The argmin helper maps 2**28 and FLT_MAX to code 0; that defect must not
    # be inherited by the lowering.
    for magnitude in (2.0**28, FLT_MAX):
        assert _pack(magnitude, 0.0) == 0x07
        assert _pack(-magnitude, 0.0) == 0x0F


def test_nan_magnitude_is_zero_and_sign_is_kept() -> None:
    # Legacy 029 CPU policy, not a claim about NVIDIA NaN handling.
    assert _pack(float("nan"), 0.0) == 0x00
    assert _pack(float("-nan"), 0.0) == 0x08


def test_nan_policy_does_not_depend_on_minimum_mode() -> None:
    from hcu_vllm_das import fp4_packer_triton as impl

    body = inspect.getsource(impl.e2m1_code.fn)
    assert "x != x" in body, "NaN handling must be explicit, not tl.minimum's default"


def test_subnormal_and_tiny_values_round_to_signed_zero() -> None:
    for tiny in (5e-324, 1.0e-38, 1.0e-8, 0.1, 0.2499999):
        assert _pack(tiny, 0.0) == 0x00
        assert _pack(-tiny, 0.0) == 0x08
    assert _pack(0.25, 0.0) == 0x00      # tie between zero and 0.5 keeps zero
    assert _pack(-0.25, 0.0) == 0x08
    assert _pack(_neighbour(0.25, +1), 0.0) == 0x01


def test_negative_zero_keeps_its_sign_bit() -> None:
    assert _pack(-0.0, 0.0) == 0x08
    assert _pack(0.0, 0.0) == 0x00
    assert _simulate(-0.0) & 0x08 == 0x08


# ---------------------------------------------------------------------------
# 3. the Triton body carries the required operations with the right constants
# ---------------------------------------------------------------------------

def test_triton_body_has_saturation_ties_and_ieee_sign() -> None:
    from hcu_vllm_das import fp4_packer_triton as impl

    body = inspect.getsource(impl.e2m1_code.fn)
    assert "tl.minimum(tl.abs(x), 6.0)" in body, "min(|x|, 6) saturation"
    for boundary in ("0.25", "0.75", "1.25", "1.75", "2.5", "3.5", "5.0"):
        assert f"ax >= {boundary}" in body, boundary
        assert f"ax == {boundary}" in body, boundary
    assert "(idx & 1) == 1" in body and "idx - 1" in body, "tie to even code"
    assert "bitcast=True" in body and ">> 31" in body, "IEEE sign bit"

    packed = inspect.getsource(impl.fp32x2_to_fp4x2.fn)
    assert "e2m1_code(x_lo)" in packed
    assert "e2m1_code(x_hi) << 4" in packed
    assert "inline_asm_elementwise" not in packed, "replacement must not emit PTX asm"


def test_replacement_is_not_the_audited_inline_asm() -> None:
    # Guards the audit direction: the replacement must never be mistaken for
    # the object it is allowed to replace.
    from hcu_vllm_das import fp4_packer_triton as impl

    assert packer._is_audited_inline_asm(impl.fp32x2_to_fp4x2) is False
    assert packer._is_audited_source(
        inspect.getsource(impl.fp32x2_to_fp4x2.fn)
    ) is False


# ---------------------------------------------------------------------------
# 4. audit gate: only the exact upstream definition may be rebound
# ---------------------------------------------------------------------------

def _core_source() -> str:
    module = __import__(CORE, fromlist=["x"])
    return inspect.getsource(module._fp32x2_to_fp4x2.fn)


def test_audit_accepts_the_real_upstream_definition() -> None:
    module = __import__(CORE, fromlist=["x"])
    upstream = module._fp32x2_to_fp4x2
    assert packer._is_audited_inline_asm(upstream) is True
    assert packer._is_audited_source(_core_source()) is True


def test_audit_expects_the_three_float_operand_constraint() -> None:
    source = _core_source()
    kwargs = _asm_keywords(source)
    assert ast.literal_eval(kwargs["constraints"]) == "=r,f,f"
    # $1 is the high nibble, $2 the low one: args order is [x_hi, x_lo].
    assert [ast.unparse(node) for node in kwargs["args"].elts] == ["x_hi", "x_lo"]
    assert ast.unparse(kwargs["pack"]) == "1"
    assert ast.unparse(kwargs["is_pure"]) == "True"
    assert "cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;" in source
    assert "cvt.u32.u8 $0, tmp;" in source


def _asm_keywords(source: str):
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "inline_asm_elementwise":
                return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError("inline_asm_elementwise call is absent")


TAMPERED = (
    ("args=[x_hi, x_lo]", "args=[x_lo, x_hi]"),          # operand order
    ("cvt.rn.satfinite", "cvt.rz.satfinite"),            # rounding mode
    ('constraints="=r,f,f"', 'constraints="=r,f"'),       # operand count
    ("is_pure=True", "is_pure=False"),                   # purity
    ("pack=1", "pack=2"),                                # packing
    ("tmp, $1, $2", "tmp, $2, $1"),                      # asm operand order
    ("cvt.u32.u8 $0, tmp;", "cvt.u32.u16 $0, tmp;"),     # result width
)


@pytest.mark.parametrize(("needle", "replacement"), TAMPERED)
def test_audit_rejects_every_tampered_variant(needle: str, replacement: str) -> None:
    source = _core_source()
    assert needle in source, needle
    mutated = source.replace(needle, replacement)
    assert mutated != source
    assert packer._is_audited_source(mutated) is False


def test_audit_rejects_a_non_inline_asm_object() -> None:
    assert packer._is_audited_source("def _fp32x2_to_fp4x2(x_lo, x_hi):\n    return 0\n") is False
    assert packer._is_audited_source("not python(") is False


def test_audit_accepts_comment_and_whitespace_changes() -> None:
    source = _core_source()
    relaxed = source.replace(
        "# NOTE: $1 is high nibble, $2 is low nibble", "# NOTE: reformatted comment"
    )
    assert relaxed != source
    assert packer._is_audited_source(relaxed) is True


# ---------------------------------------------------------------------------
# 5. policy: default stock, kill switch wins, unsafe states refuse
# ---------------------------------------------------------------------------

def test_default_is_stock_with_no_capability(monkeypatch) -> None:
    monkeypatch.delenv(packer.VARIABLE, raising=False)
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    assert packer._mode() == "stock"
    assert packer.extension_spec().capabilities == frozenset()


def test_kill_switch_wins_over_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv(packer.VARIABLE, "triton")
    monkeypatch.setenv("HCU_DISABLE_OPS", "vllm_fp4_packer")
    assert packer._mode() == "stock"
    assert packer.extension_spec().capabilities == frozenset()


def test_explicit_opt_in_advertises_capability(monkeypatch) -> None:
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setenv(packer.VARIABLE, "triton")
    spec = packer.extension_spec()
    assert spec.capabilities == frozenset({packer.CAPABILITY})
    assert spec.name == "das-hcu-fp4-packer"
    assert spec.api_version == 1
    assert callable(spec.check_health)


def test_invalid_value_is_refused(monkeypatch) -> None:
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setenv(packer.VARIABLE, "aiter")
    with pytest.raises(packer.Fp4PackerCompatibilityError):
        packer._mode()


def test_stock_activation_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr(packer, "_state", None, raising=False)
    monkeypatch.setattr(packer, "_failure", None, raising=False)
    monkeypatch.delenv(packer.VARIABLE, raising=False)
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    packer.activate_extension(ExtensionContext("0.1.0", "0.33.0"))
    assert packer._state == {"mode": "stock", "pid": __import__("os").getpid()}
    packer.check_health()


def test_pending_opt_in_is_refused_before_activation(monkeypatch) -> None:
    monkeypatch.setattr(packer, "_state", None, raising=False)
    monkeypatch.setattr(packer, "_failure", None, raising=False)
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setenv(packer.VARIABLE, "triton")
    with pytest.raises(packer.Fp4PackerCompatibilityError):
        packer.check_health()


def test_release_gate_rejects_other_release_line(monkeypatch) -> None:
    monkeypatch.setattr(packer, "_state", None, raising=False)
    monkeypatch.setattr(packer, "_failure", None, raising=False)
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setenv(packer.VARIABLE, "triton")
    with pytest.raises(Exception):
        packer.activate_extension(ExtensionContext("0.1.0", "0.32.1"))


def test_non_hcu_or_non_gfx938_platform_refuses(monkeypatch) -> None:
    monkeypatch.setattr(packer, "_state", None, raising=False)
    monkeypatch.setattr(packer, "_failure", None, raising=False)
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setenv(packer.VARIABLE, "triton")

    fake = types.SimpleNamespace(
        is_rocm=lambda: True, device_name="cuda"
    )
    real_import = packer.import_module
    monkeypatch.setattr(
        packer,
        "import_module",
        lambda name: types.SimpleNamespace(current_platform=fake)
        if name == "vllm.platforms"
        else real_import(name),
    )
    with pytest.raises(packer.Fp4PackerCompatibilityError):
        packer.activate_extension(ExtensionContext("0.1.0", "0.33.0"))


def _packer_stub(module, *, hashed=None):
    """A JITFunction stand-in whose fn resolves the audited symbol.

    Executes directly into the module namespace so `fn.__globals__` is that
    module's dict, which is what the cold check keys on.
    """
    module.__dict__.setdefault(packer.SYMBOL, object())
    module.__dict__.setdefault("_quantize_mxfp4_pair", object())
    exec(
        "def caller(x_lo, x_hi):\n"
        f"    return {packer.SYMBOL}(x_lo, x_hi)\n",
        module.__dict__,
    )
    return types.SimpleNamespace(
        fn=module.__dict__["caller"], hash=hashed, device_caches={}
    )


def test_cold_check_allows_a_fresh_packer() -> None:
    module = types.ModuleType("fake_owner_fresh")
    module.__dict__["_fp32x2_to_fp4x2"] = _packer_stub(module)
    packer._require_cold((module,))


@pytest.mark.parametrize("hashed", ["578a2929", ""])
def test_cold_check_refuses_an_already_hashed_packer(hashed) -> None:
    module = types.ModuleType("fake_owner_stale")
    module.__dict__["_fp32x2_to_fp4x2"] = _packer_stub(module, hashed=hashed)
    with pytest.raises(packer.Fp4PackerCompatibilityError):
        packer._require_cold((module,))


def test_cold_check_refuses_a_compiled_device_cache() -> None:
    module = types.ModuleType("fake_owner_compiled")
    stub = _packer_stub(module)
    stub.device_caches = {0: {"kernel": object()}}
    module.__dict__["_fp32x2_to_fp4x2"] = stub
    with pytest.raises(packer.Fp4PackerCompatibilityError):
        packer._require_cold((module,))
