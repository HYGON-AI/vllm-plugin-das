from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PLUGIN_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))

from hcu_vllm_das.moe import fused_moe_aiter as provider


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], *, dtype: object = "bf16") -> None:
        self.shape = shape
        self.dtype = dtype


def inputs() -> dict[str, object]:
    return {
        "hidden_states": FakeTensor((3, 8)),
        "w1": FakeTensor((4, 12, 8), dtype="int8"),
        "w2": FakeTensor((4, 8, 6), dtype="int8"),
        "topk_weights": FakeTensor((3, 2), dtype="float32"),
        "topk_ids": FakeTensor((3, 2), dtype="int32"),
        "moe_config": SimpleNamespace(num_experts=8),
        "quant_config": SimpleNamespace(
            use_int8_w8a8=True,
            w1_scale="w1-scale",
            w2_scale="w2-scale",
            w1_zp="w1-zp",
            w2_zp="w2-zp",
            a1_scale="a1-scale",
            a2_scale="a2-scale",
        ),
        "activation": SimpleNamespace(value="silu"),
        "expert_map": None,
        "output_dtype": "bf16-output",
    }


def native_config(**overrides: object) -> SimpleNamespace:
    values = {
        "solution_type": "asm",
        "config": {"SOL_ID1": 10010, "SOL_ID2": 20000, "BLOCK_SIZE_M": 16},
        "need_shuffle": False,
        "need_shuffle_scale": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("bias", ("w1_bias", "w2_bias"))
def test_direct_provider_rejects_expert_bias_before_config_or_execution(monkeypatch, bias):
    arguments = inputs()
    setattr(arguments["quant_config"], bias, object())
    monkeypatch.setattr(provider, "_load_aiter_api", lambda: pytest.fail("vendor reached"))
    with pytest.raises(ValueError, match="expert bias"):
        provider.apply_aiter_int8_moe(**arguments)


@pytest.mark.parametrize("activation", ("situ", "swigluoai", "swigluoai_uninterleave"))
def test_direct_provider_rejects_unforwarded_activation_parameters(monkeypatch, activation):
    arguments = inputs()
    arguments["activation"] = SimpleNamespace(value=activation)
    monkeypatch.setattr(provider, "_load_aiter_api", lambda: pytest.fail("vendor reached"))
    with pytest.raises(ValueError, match="activation"):
        provider.apply_aiter_int8_moe(**arguments)


def test_shape_evidence_uses_vllm_runtime_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str, tuple[object, ...]]] = []
    logger_names: list[str] = []
    runtime_logger = SimpleNamespace(
        log=lambda level, message, *args: calls.append((level, message, args))
    )

    def init_logger(name: str):
        logger_names.append(name)
        return runtime_logger

    def load_module(name: str):
        assert name == "vllm.logger"
        return SimpleNamespace(init_logger=init_logger)

    monkeypatch.setattr(provider, "import_module", load_module)
    provider._record_once(
        provider.logging.INFO,
        "selected",
        (7, 4, 12, 8, 8, 2),
        "asm",
        native_config(),
    )

    assert logger_names == ["vllm.hcu_vllm_das.moe.fused_moe_aiter"]
    assert calls == [
        (
            provider.logging.INFO,
            "HCU AITER INT8 MoE %s: M=%d E=%d N1=%d N2=%d K=%d top_k=%d "
            "solution=%s config=%s",
            (
                "selected",
                7,
                4,
                12,
                8,
                8,
                2,
                "asm",
                {"SOL_ID1": 10010, "SOL_ID2": 20000, "BLOCK_SIZE_M": 16},
            ),
        )
    ]


def test_native_config_invokes_public_aiter_w8a8_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    expected = object()
    arguments = inputs()
    selected_config = native_config()

    def get_config(*args: object, **kwargs: object):
        calls["config"] = (args, kwargs)
        return True, selected_config

    def aiter_moe(*args: object, **kwargs: object):
        calls["aiter"] = (args, kwargs)
        return expected

    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    monkeypatch.setattr(
        provider,
        "_load_triton_fallback",
        lambda: pytest.fail("native AITER config must not use Triton fallback"),
    )

    result = provider.apply_aiter_int8_moe(**arguments)

    assert result is expected
    assert calls["config"] == (
        (3, 4, 12, 8, 8, 2, 0, "bf16", "int8_w8a8"),
        {"activation": "silu", "use_shuffle": 0},
    )
    assert calls["aiter"] == (
        (),
        {
            "hidden_states": arguments["hidden_states"],
            "w1": arguments["w1"],
            "w2": arguments["w2"],
            "topk_weights": arguments["topk_weights"],
            "topk_ids": arguments["topk_ids"],
            "moe_config": selected_config,
            "inplace": False,
            "activation": "silu",
            "w1_scale": "w1-scale",
            "w2_scale": "w2-scale",
            "w1_zp": "w1-zp",
            "w2_zp": "w2-zp",
            "a1_scale": "a1-scale",
            "a2_scale": "a2-scale",
            "block_shape": None,
            "global_num_experts": 8,
            "expert_map": None,
            "routed_scaling_factor": 1.0,
            "use_weight_shuffle": False,
            "output_dtype": "bf16-output",
        },
    )


@pytest.mark.parametrize(
    ("found", "config"),
    (
        (False, native_config(solution_type=None, config=None)),
        (True, native_config(need_shuffle=True)),
        (True, native_config(need_shuffle_scale=True)),
    ),
    ids=("config-miss", "weight-shuffle", "scale-shuffle"),
)
@pytest.mark.parametrize(
    "expert_map",
    (
        torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        torch.tensor([-1, -1, 0, 1], dtype=torch.int32),
        torch.tensor([0, -1, 1, -1], dtype=torch.int32),
        torch.tensor([-1, 0, -1, 1], dtype=torch.int32),
    ),
    ids=(
        "linear-rank0",
        "linear-rank1",
        "round-robin-rank0",
        "round-robin-rank1",
    ),
)
def test_unsupported_aiter_layout_uses_official_int8_triton_fallback(
    monkeypatch: pytest.MonkeyPatch,
    found: bool,
    config: SimpleNamespace,
    expert_map: torch.Tensor,
) -> None:
    calls: list[dict[str, object]] = []
    expected = object()
    arguments = inputs()
    arguments["expert_map"] = expert_map
    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=lambda *args, **kwargs: (found, config),
            aiter_moe=lambda **kwargs: pytest.fail("unsupported config must fallback"),
        ),
    )

    def fallback(**kwargs: object):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(provider, "_load_triton_fallback", lambda: fallback)

    result = provider.apply_aiter_int8_moe(**arguments)

    assert result is expected
    assert len(calls) == 1
    fallback_call = calls[0]
    assert fallback_call.pop("expert_map") is expert_map
    assert fallback_call == {
        "hidden_states": arguments["hidden_states"],
        "w1": arguments["w1"],
        "w2": arguments["w2"],
        "topk_weights": arguments["topk_weights"],
        "topk_ids": arguments["topk_ids"],
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": False,
        "use_int8_w8a8": True,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "per_channel_quant": True,
        "global_num_experts": 8,
        "w1_scale": "w1-scale",
        "w2_scale": "w2-scale",
        "w1_zp": "w1-zp",
        "w2_zp": "w2-zp",
        "a1_scale": "a1-scale",
        "a2_scale": "a2-scale",
        "block_shape": None,
    }


@pytest.mark.parametrize(
    "expert_map",
    ([0, 1, -1, -1], [-1, -1, 0, 1], [0, -1, 1, -1], [-1, 0, -1, 1]),
    ids=("linear-rank0", "linear-rank1", "round-robin-rank0", "round-robin-rank1"),
)
def test_native_asm_receives_unchanged_canonical_expert_map(
    monkeypatch: pytest.MonkeyPatch,
    expert_map: list[int],
) -> None:
    arguments = inputs()
    arguments["expert_map"] = torch.tensor(expert_map, dtype=torch.int32)
    calls: list[dict[str, object]] = []

    from hcu_vllm_das.moe import int8_asm

    vendor = SimpleNamespace(
        MoeQuantType=SimpleNamespace(W8A8="int8_w8a8"),
        get_aiter_moe_config=lambda *a, **kw: (True, native_config()),
        aiter_moe=lambda **kw: pytest.fail("ASM must use explicit stages"),
    )
    native = SimpleNamespace()
    original_import = provider.import_module
    monkeypatch.setattr(
        provider,
        "import_module",
        lambda name: vendor
        if name == "aiter.moe"
        else native
        if name == "aiter"
        else original_import(name),
    )
    monkeypatch.setattr(int8_asm, "unsupported_reason", lambda a, n: None)
    monkeypatch.setattr(int8_asm, "execute", lambda **kw: calls.append(kw) or object())

    provider.apply_aiter_int8_moe(**arguments)

    assert len(calls) == 1
    assert calls[0]["expert_map"] is arguments["expert_map"]
    assert torch.equal(calls[0]["expert_map"], torch.tensor(expert_map, dtype=torch.int32))


def test_vendor_failure_after_config_selection_does_not_change_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_vendor(**kwargs: object):
        raise RuntimeError("vendor kernel failed")

    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=lambda *args, **kwargs: (True, native_config()),
            aiter_moe=fail_vendor,
        ),
    )
    monkeypatch.setattr(
        provider,
        "_load_triton_fallback",
        lambda: pytest.fail("vendor execution errors must propagate"),
    )

    with pytest.raises(RuntimeError, match="vendor kernel failed"):
        provider.apply_aiter_int8_moe(**inputs())


@pytest.mark.parametrize("solution", ("triton", "moe_c"))
@pytest.mark.parametrize(
    "mapping", ([0, 1, -1, -1], [-1, -1, 0, 1], [0, -1, 1, -1], [-1, 0, -1, 1])
)
def test_index_map_solutions_preserve_canonical_ep_mapping(
    monkeypatch, solution, mapping
):
    arguments = inputs()
    canonical = torch.tensor(mapping, dtype=torch.int32)
    arguments["expert_map"] = canonical
    calls = []
    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=lambda *a, **k: (True, native_config(solution_type=solution)),
            aiter_moe=lambda **k: calls.append(k) or object(),
        ),
    )
    provider.apply_aiter_int8_moe(**arguments)
    assert len(calls) == 1
    assert calls[0]["expert_map"] is canonical
    assert canonical.tolist() == mapping


def test_unknown_ep_solution_uses_official_fallback_before_vendor(monkeypatch):
    arguments = inputs()
    canonical = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    arguments["expert_map"] = canonical
    calls = []
    expected = object()
    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8="int8_w8a8",
            get_config=lambda *a, **k: (
                True,
                native_config(solution_type="new-vendor-abi"),
            ),
            aiter_moe=lambda **k: pytest.fail(
                "unknown EP map ABI must not launch vendor"
            ),
        ),
    )
    monkeypatch.setattr(
        provider,
        "_load_triton_fallback",
        lambda: lambda **k: calls.append(k) or expected,
    )
    assert provider.apply_aiter_int8_moe(**arguments) is expected
    assert len(calls) == 1 and calls[0]["expert_map"] is canonical


def test_ck_keeps_binary_ownership_mask_abi(monkeypatch):
    canonical = torch.tensor([1, -1, 0, -1], dtype=torch.int32)
    seen = []
    vendor = SimpleNamespace(
        MoeQuantType=SimpleNamespace(W8A8="int8_w8a8"),
        get_aiter_moe_config=lambda *a, **kw: None,
        aiter_moe=lambda **kw: seen.append(kw) or object(),
    )
    original_import = provider.import_module
    monkeypatch.setattr(provider, "import_module", lambda n: vendor if n == "aiter.moe" else original_import(n))
    provider._load_aiter_api().aiter_moe(moe_config=SimpleNamespace(solution_type="ck"), expert_map=canonical)
    assert torch.equal(seen[0]["expert_map"], torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32))
    assert torch.equal(canonical, torch.tensor([1, -1, 0, -1], dtype=torch.int32))


def ep_inputs(tokens=1):
    result = inputs()
    result.update(hidden_states=FakeTensor((tokens, 2048), dtype=torch.bfloat16),
                  w1=FakeTensor((128, 1024, 2048), dtype=torch.int8),
                  w2=FakeTensor((128, 2048, 512), dtype=torch.int8),
                  topk_weights=FakeTensor((tokens, 8), dtype=torch.float32),
                  topk_ids=FakeTensor((tokens, 8), dtype=torch.int32),
                  moe_config=SimpleNamespace(num_experts=256),
                  expert_map=torch.cat((torch.arange(128, dtype=torch.int32),
                                        torch.full((128,), -1, dtype=torch.int32))),
                  output_dtype=torch.bfloat16, asm_only=True)
    for name in ('w1_zp', 'w2_zp', 'a1_scale', 'a2_scale'):
        setattr(result['quant_config'], name, None)
    return result


# Expected plans come from the independently audited native stage.
EP2_QUALIFIED_PROFILES = json.loads(
    (Path(__file__).parent / 'data/int8_ep2_gfx936_qualified_profiles.json').read_text()
)['profiles']


@pytest.mark.parametrize(('tokens', 'config'), [
    (int(m), config) for m, config in sorted(EP2_QUALIFIED_PROFILES.items(), key=lambda item: int(item[0]))
])
def test_ep_uses_explicit_qualified_global_profile_after_real_local_miss(monkeypatch, tokens, config):
    args = ep_inputs(tokens)
    profile = native_config(config=dict(config))
    lookups, native_calls, events = [], [], []
    def lookup(*values, **kwargs):
        lookups.append((values, kwargs))
        return (False, SimpleNamespace(solution_type=None, config=None)) if values[1] == 128 else (True, profile)
    api = SimpleNamespace(quant_type_w8a8='int8_w8a8', get_config=lookup,
                          aiter_moe=lambda **kw: native_calls.append(kw) or 'native')
    monkeypatch.setattr(provider, '_load_aiter_api', lambda: api)
    monkeypatch.setattr(provider, '_record_once', lambda level, event, *a: events.append(event))
    monkeypatch.setattr(provider, '_load_triton_fallback', lambda: pytest.fail('qualified profile should execute'))
    assert provider.apply_aiter_int8_moe(**args, allow_ep_global_profile=True) == 'native'
    assert [values[0][1] for values in lookups] == [128, 256]
    assert all(kwargs['spec_sol_type'] == 'asm' for _, kwargs in lookups)
    assert native_calls[0]['moe_config'] is profile
    assert native_calls[0]['expert_map'] is args['expert_map']
    assert any('ep-global-profile' in event for event in events)


@pytest.mark.parametrize('kind', ['not-opted-in', 'tokens5', 'fp16', 'unknown-sol', 'shuffle', 'padded'])
def test_unqualified_ep_profile_uses_complete_official_fallback(monkeypatch, kind):
    args = ep_inputs(5 if kind == 'tokens5' else 1)
    if kind == 'fp16':
        args['hidden_states'].dtype = torch.float16
        args['output_dtype'] = torch.float16
    profile = native_config()
    if kind == 'unknown-sol':profile.config['SOL_ID1'] = 99999
    if kind == 'shuffle':profile.need_shuffle = True
    if kind == 'padded':profile.config['PADDED_K'] = 4096
    lookups, fallbacks = [], []
    def lookup(*values, **kwargs):
        lookups.append(values[1])
        return (False, SimpleNamespace(solution_type=None, config=None)) if values[1] == 128 else (True, profile)
    api = SimpleNamespace(quant_type_w8a8='int8_w8a8', get_config=lookup,
                          aiter_moe=lambda **kw: pytest.fail('unqualified native execution'))
    monkeypatch.setattr(provider, '_load_aiter_api', lambda: api)
    monkeypatch.setattr(provider, '_load_triton_fallback', lambda: lambda **kw: fallbacks.append(kw) or 'fallback')
    assert provider.apply_aiter_int8_moe(**args, allow_ep_global_profile=kind != 'not-opted-in') == 'fallback'
    assert lookups == ([128] if kind in ('not-opted-in', 'tokens5', 'fp16') else [128, 256])
    assert fallbacks[0]['expert_map'] is args['expert_map']
    assert fallbacks[0]['w1'] is args['w1'] and fallbacks[0]['hidden_states'] is args['hidden_states']


@pytest.mark.parametrize('tokens', [1, 5, 512, 1024])
def test_real_local_vendor_profile_takes_precedence_over_compatibility(monkeypatch, tokens):
    args = ep_inputs(tokens)
    profile = native_config()
    lookups = []
    api = SimpleNamespace(quant_type_w8a8='int8_w8a8',
                          get_config=lambda *a, **kw: (lookups.append(a[1]) or True, profile),
                          aiter_moe=lambda **kw: 'native')
    monkeypatch.setattr(provider, '_load_aiter_api', lambda: api)
    assert provider.apply_aiter_int8_moe(**args, allow_ep_global_profile=True) == 'native'
    assert lookups == [128]


@pytest.mark.parametrize('tokens', [512, 1024])
@pytest.mark.parametrize('field', ['SOL_ID1', 'SOL_ID2', 'BLOCK_SIZE_M'])
def test_ep_compatibility_rejects_any_changed_stage_or_block(monkeypatch, tokens, field):
    args = ep_inputs(tokens)
    config = dict(EP2_QUALIFIED_PROFILES[str(tokens)])
    config[field] += 1
    profile = native_config(config=config)
    lookups, fallbacks = [], []
    def lookup(*values, **kwargs):
        lookups.append(values[1])
        return (False, SimpleNamespace(solution_type=None, config=None)) if values[1] == 128 else (True, profile)
    monkeypatch.setattr(provider, '_load_aiter_api', lambda: SimpleNamespace(
        quant_type_w8a8='int8_w8a8', get_config=lookup,
        aiter_moe=lambda **kw: pytest.fail('unqualified stage/block launched'),
    ))
    monkeypatch.setattr(provider, '_load_triton_fallback', lambda: lambda **kw: fallbacks.append(kw) or 'fallback')
    assert provider.apply_aiter_int8_moe(**args, allow_ep_global_profile=True) == 'fallback'
    assert lookups == [128, 256]
    assert fallbacks[0]['hidden_states'] is args['hidden_states']
    assert fallbacks[0]['w1'] is args['w1'] and fallbacks[0]['expert_map'] is args['expert_map']


@pytest.mark.parametrize('tokens', [5, 7, 15, 33, 63, 65, 127, 129, 255, 257, 511, 513, 1023, 1025])
def test_untested_token_counts_do_not_borrow_a_global_profile(monkeypatch, tokens):
    args = ep_inputs(tokens)
    lookups, fallbacks = [], []
    def lookup(*values, **kwargs):
        lookups.append(values[1])
        return False, SimpleNamespace(solution_type=None, config=None)
    monkeypatch.setattr(provider, '_load_aiter_api', lambda: SimpleNamespace(
        quant_type_w8a8='int8_w8a8', get_config=lookup,
        aiter_moe=lambda **kw: pytest.fail('untested token count launched'),
    ))
    monkeypatch.setattr(provider, '_load_triton_fallback', lambda: lambda **kw: fallbacks.append(kw) or 'fallback')
    assert provider.apply_aiter_int8_moe(**args, allow_ep_global_profile=True) == 'fallback'
    assert lookups == [128]
    assert fallbacks[0]['hidden_states'] is args['hidden_states']
    assert fallbacks[0]['w1'] is args['w1'] and fallbacks[0]['expert_map'] is args['expert_map']
    assert fallbacks[0]['use_int8_w8a8'] and fallbacks[0]['per_channel_quant']
