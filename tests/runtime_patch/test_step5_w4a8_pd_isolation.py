"""Model selection and allocator policy stay inert for other models by default."""

import ast
import copy
import importlib.util
import itertools
import os
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "vllm_hcu/model_executor/layers/fused_moe"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


owner = load_file("step5_owner", REPO / "vllm_hcu/runtime_compat/step5.py")
namespace = {
    "os": os,
    "is_step5_owner": owner.is_step5_owner,
    "is_step5_config": owner.is_step5_config,
}
tree = ast.parse((PLUGIN / "step5_w4a8_policy.py").read_text())
exec(
    compile(
        ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
            type_ignores=[],
        ),
        "policy",
        "exec",
    ),
    namespace,
)
common = NS(
    **{
        name: value
        for name, value in namespace.items()
        if name in ("resolve_profile", "padded_route_rows")
    }
)
allocator = load_file(
    "allocator_limit", REPO / "vllm_hcu/runtime_compat/allocator_limit.py"
)


def moe(dp=8, step5=True, activation="swiglustep"):
    return NS(
        _hcu_step5_owner=step5,
        activation=activation,
        moe_parallel_config=NS(
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            ep_size=8,
            dp_size=dp,
        ),
    )


def test_default_selection_is_inert(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_STEP5_W4A8_PROFILE", raising=False)
    assert common.resolve_profile(object()) is None


@pytest.mark.parametrize("phase,dp", [("prefill", 8), ("decode", 2)])
def test_profile_selects_step5_only(monkeypatch, phase, dp):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    assert common.resolve_profile(moe(dp)) == phase
    assert common.resolve_profile(moe(dp, False)) is None
    assert common.resolve_profile(moe(dp, True, "silu")) is None


def test_bad_step5_profile_rejected_but_foreign_ignored(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", "bad")
    assert common.resolve_profile(moe(step5=False)) is None
    with pytest.raises(ValueError):
        common.resolve_profile(moe())


def model_config(phase="prefill", model_type="step5", quantization="slimquant_w4a8"):
    return NS(
        model_config=NS(hf_config=NS(model_type=model_type), quantization=quantization),
        parallel_config=NS(
            tensor_parallel_size=1 if phase == "prefill" else 4,
            data_parallel_size=8 if phase == "prefill" else 2,
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            enable_eplb=False,
        ),
        lora_config=None,
    )


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_model_selection_requires_matching_w4a8_parallelism(monkeypatch, phase):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    config = model_config(phase)
    assert namespace["profile_for_model"](config) == phase
    assert namespace["profile_for_model"](model_config(phase, "qwen3_moe")) is None
    with pytest.raises(ValueError):
        namespace["profile_for_model"](model_config(phase, quantization="fp8"))
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError):
        namespace["profile_for_model"](config)


def test_model_selection_default_is_inert(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_STEP5_W4A8_PROFILE", raising=False)
    assert namespace["profile_for_model"](object()) is None


def test_wrong_ep_or_eplb_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", "prefill")
    value = moe()
    value.moe_parallel_config.ep_size = 4
    with pytest.raises(ValueError):
        common.resolve_profile(value)
    value = moe()
    value.moe_parallel_config.enable_eplb = True
    with pytest.raises(ValueError):
        common.resolve_profile(value)


def test_allocator_default_does_not_touch_torch(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_ALLOCATOR_LIMIT_GIB", raising=False)
    assert allocator.apply_allocator_limit("cuda:0", object()) is None


@pytest.mark.parametrize("raw", ["nan", "inf", "0", "-1"])
def test_invalid_allocator_caps_rejected(monkeypatch, raw):
    monkeypatch.setenv("VLLM_HCU_ALLOCATOR_LIMIT_GIB", raw)
    with pytest.raises(ValueError):
        allocator.apply_allocator_limit("cuda:0", object())


def test_allocator_cap_matches_old86gib_recipe(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_ALLOCATOR_LIMIT_GIB", "86")
    calls = []
    fake = NS(
        cuda=NS(
            get_device_properties=lambda dev: NS(total_memory=144 * 2**30),
            set_per_process_memory_fraction=lambda fraction, device: calls.append(
                (fraction, device)
            ),
        )
    )
    assert allocator.apply_allocator_limit("cuda:0", fake) == 86 * 2**30
    assert calls == [(86 / 144, "cuda:0")]


def test_shared_classes_are_not_monkeypatched():
    source = (PLUGIN / "step5_w4a8.py").read_text()
    assert (
        "scoped_post_load" not in source
        and "process_weights_after_loading =" not in source
    )
    assert "spec_from_file_location" not in source and "LOCAL_EXT" not in source


def test_slimquant_native_branch_exits_before_generic_layout():
    source = (
        REPO / "vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py"
    ).read_text()
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SlimQuantW4A8Int8AiterMoEMethod"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading"
    )
    assert "prepare_step5_w4a8" in ast.unparse(fn.body[1])
    assert any(isinstance(n, ast.Return) for n in ast.walk(fn.body[1]))


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("rows", [1, 32, 256, 4096, 16384])
def test_phase_specific_workspace_layout_is_preserved(prefill, rows):
    tree = ast.parse((PLUGIN / "step5_w4a8.py").read_text(encoding="utf8"))
    original = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Step5W4A8Experts"
    )
    cls = copy.deepcopy(original)
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "workspace_shapes"
    ]

    class Base:
        ALIGNMENT = 256

        def workspace_shapes(self, *args):
            return ((1, 2), (3, 4), (5, 6))

    namespace = {
        "Base": Base,
        "is_prefill": prefill,
        "padded_route_rows": common.padded_route_rows,
        "triton": NS(cdiv=lambda a, b: (a + b - 1) // b),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "workspace",
            "exec",
        ),
        namespace,
    )
    actual = namespace[cls.name]().workspace_shapes(
        rows, 3072, 4096, 8, 352, 44, None, "swiglustep"
    )
    if not prefill:
        assert actual == ((1, 2), (3, 4), (5, 6))
    else:
        aligned = common.padded_route_rows(rows, 8, 44)
        elements = max(aligned * 4096 // 2, rows * 4096)
        assert actual == (
            ((elements + 4095) // 4096, 4096),
            (aligned, 4096),
            (rows, 4096),
        )


def test_route_bound_covers_distributions():
    for routes in range(9):
        for counts in itertools.product(range(routes + 1), repeat=3):
            if sum(counts) == routes:
                assert sum(
                    (c + 63) // 64 * 64 for c in counts
                ) <= common.padded_route_rows(routes, 1, 3)
