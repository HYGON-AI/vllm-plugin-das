from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_hcu/model_executor/layers/fused_moe/static_ht_scratch.py"
)


def load():
    assert PATH.exists(), "static HT scratch contract is not implemented"
    import importlib

    return importlib.import_module(
        "vllm_hcu.model_executor.layers.fused_moe.static_ht_scratch"
    )


def test_canonical_map_requires_correct_rank_and_detects_mutation():
    mod = load()
    expert_map = torch.tensor([-1, -1, 0, 1, -1, -1], dtype=torch.int32)
    assert mod.is_linear_map(expert_map, 6, 2, 2)
    assert not mod.is_linear_map(expert_map, 6, 2, 0)
    expert_map[2] = 1
    assert not mod.is_linear_map(expert_map, 6, 2, 2)


def test_fresh_counts_create_rounded_plan_only_for_valid_nonzero_receives():
    mod = load()
    assert mod.make_plan([255, 256, 257], 3, 1) == (256, 256, 512)
    for counts, experts, received in [
        ([0, 0], 2, 1),
        ([1], 2, 1),
        ([-1, 1], 2, 1),
        ([True, 1], 2, 1),
        ([1.0, 1], 2, 1),
        ([1, 1], 2, 0),
    ]:
        assert mod.make_plan(counts, experts, received) is None


def test_diagnostic_rejects_per_expert_overflow_even_when_total_is_safe():
    mod = load()
    mod.check_recount((256, 512), torch.tensor([256, 257]))
    with pytest.raises(RuntimeError):
        mod.check_recount((256, 512), torch.tensor([257, 1]))


def test_untrusted_metadata_keeps_conservative_workspace():
    import ast

    source = (PATH.parent / "deep_gemm_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"expert_num_tokens_round_up_and_sum", "compute_aligned_M_and_alignment"}
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    isolated = ast.Module(body=functions, type_ignores=[])
    import __future__

    namespace = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_HCU_TOKEN_ALIGNMENT": 256,
        "round_up": lambda n, a: (n + a - 1) // a * a,
    }
    exec(
        compile(
            isolated, str(PATH), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    compute = namespace["compute_aligned_M_and_alignment"]
    meta = SimpleNamespace(expert_num_tokens_cpu=torch.tensor([1, 0]))
    assert compute(100, 8, 2, 256, meta) == (1536, 256)
    mod = load()
    mod.attach_plan(meta, [1, 0], 2, 100, True)
    assert compute(100, 8, 2, 256, meta) == (256, 256)


def extract_functions(path, names, namespace):
    import __future__

    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def test_prepare_tags_only_validated_static_owner_and_map(monkeypatch):
    import weakref

    from vllm_hcu.platforms import envs

    monkeypatch.setenv("VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH", "1")
    monkeypatch.setenv("VLLM_HCU_STATIC_HT_W4A8_COUNT_DIAGNOSTIC", "1")
    ns = extract_functions(
        PATH.parent / "deepep_runtime.py",
        {"ht_prepare_async"},
        {
            "current_platform": SimpleNamespace(is_rocm=lambda: True),
            "henvs": envs,
            "_uses_post_dispatch_channel_fp8_quant": lambda _: False,
        },
    )
    owner = SimpleNamespace(
        _vllm_hcu_static_eplb_row=(1, 0),
        quant_method=SimpleNamespace(_hcu_static_eplb=True),
    )

    class Owner:
        pass

    real_owner = Owner()
    real_owner.__dict__.update(owner.__dict__)
    meta = SimpleNamespace(_hcu_static_ht_native_counts=[1, 0])
    receiver = lambda: (torch.ones((1, 2)), None, meta, None, None)
    dispatcher = SimpleNamespace(
        _hcu_static_ht_owner=weakref.ref(real_owner),
        rank_expert_offset=2,
        _do_dispatch=lambda **kwargs: receiver,
    )
    quant = SimpleNamespace(
        is_block_quantized=False,
        is_per_act_token=False,
        quant_dtype="int8",
        a1_scale=None,
    )
    expert_map = torch.full((32,), -1, dtype=torch.int32)
    expert_map[2:4] = torch.tensor([0, 1], dtype=torch.int32)

    def prepare():
        return ns["ht_prepare_async"](
            SimpleNamespace(),
            dispatcher,
            torch.ones((1, 2)),
            torch.ones((1, 1)),
            torch.zeros((1, 1)),
            32,
            expert_map,
            False,
            quant,
        )

    prepare()()
    plan = load().get_plan(meta, 2, 256)
    assert plan.bounds == (256, 0) and plan.diagnostic
    expert_map[2] = 1
    assert load().get_plan(meta, 2, 256) is None
    meta._hcu_static_ht_scratch = None
    prepare()()
    assert load().get_plan(meta, 2, 256) is None
    expert_map[2] = 0
    monkeypatch.delenv("VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH")
    prepare()()
    assert load().get_plan(meta, 2, 256) is None


def test_permute_diagnostic_rejects_overflow_before_scatter():
    mod = load()
    meta = SimpleNamespace(expert_num_tokens_cpu=torch.tensor([1, 0]))
    mod.attach_plan(meta, [1, 0], 2, 1, True)
    writes = []
    namespace = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_HCU_TOKEN_ALIGNMENT": 256,
        "round_up": lambda n, a: (n + a - 1) // a * a,
        "count_expert_num_tokens": lambda *args: torch.tensor([257, 0]),
        "ep_scatter": lambda **kwargs: writes.append(kwargs),
    }
    extract_functions(
        PATH.parent / "deep_gemm_utils.py",
        {
            "expert_num_tokens_round_up_and_sum",
            "compute_aligned_M_and_alignment",
            "deepgemm_moe_permute",
        },
        namespace,
    )
    with pytest.raises(RuntimeError, match="post-map"):
        namespace["deepgemm_moe_permute"](
            torch.ones((1, 2), dtype=torch.int8),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int64),
            2,
            None,
            meta,
            torch.empty((256, 2), dtype=torch.int8),
        )
    assert writes == []


def test_permute_rejects_different_map_after_small_allocation():
    mod = load()
    original_map = torch.tensor([0, -1], dtype=torch.int32)
    meta = SimpleNamespace(expert_num_tokens_cpu=torch.tensor([1, 0]))
    mod.attach_plan(meta, [1, 0], 2, 1, False, original_map, original_map._version)
    writes = []
    ns = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_HCU_TOKEN_ALIGNMENT": 256,
        "round_up": lambda n, a: (n + a - 1) // a * a,
        "count_expert_num_tokens": lambda *args: torch.tensor([1, 0]),
        "ep_scatter": lambda **kwargs: writes.append(kwargs),
    }
    extract_functions(
        PATH.parent / "deep_gemm_utils.py",
        {
            "expert_num_tokens_round_up_and_sum",
            "compute_aligned_M_and_alignment",
            "deepgemm_moe_permute",
        },
        ns,
    )
    with pytest.raises(RuntimeError, match="expert map"):
        ns["deepgemm_moe_permute"](
            torch.ones((1, 2), dtype=torch.int8),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int64),
            2,
            original_map.clone(),
            meta,
            torch.empty((256, 2), dtype=torch.int8),
        )
    assert writes == []


def test_permute_rejects_map_mutated_during_recount():
    mod = load()
    expert_map = torch.tensor([0, -1], dtype=torch.int32)
    meta = SimpleNamespace(expert_num_tokens_cpu=torch.tensor([1, 0]))
    mod.attach_plan(meta, [1, 0], 2, 1, False, expert_map, expert_map._version)
    writes = []

    def recount(*args):
        expert_map[0] = -1
        return torch.tensor([1, 0])

    ns = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_HCU_TOKEN_ALIGNMENT": 256,
        "round_up": lambda n, a: (n + a - 1) // a * a,
        "count_expert_num_tokens": recount,
        "ep_scatter": lambda **kwargs: writes.append(kwargs),
    }
    extract_functions(
        PATH.parent / "deep_gemm_utils.py",
        {
            "expert_num_tokens_round_up_and_sum",
            "compute_aligned_M_and_alignment",
            "deepgemm_moe_permute",
        },
        ns,
    )
    with pytest.raises(RuntimeError, match="expert map"):
        ns["deepgemm_moe_permute"](
            torch.ones((1, 2), dtype=torch.int8),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int64),
            2,
            expert_map,
            meta,
            torch.empty((256, 2), dtype=torch.int8),
        )
    assert writes == []


def test_map_cache_is_scoped_to_rank_offset(monkeypatch):
    import weakref

    from vllm_hcu.platforms import envs

    monkeypatch.setenv("VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH", "1")
    ns = extract_functions(
        PATH.parent / "deepep_runtime.py",
        {"ht_prepare_async"},
        {
            "current_platform": SimpleNamespace(is_rocm=lambda: True),
            "henvs": envs,
            "_uses_post_dispatch_channel_fp8_quant": lambda _: False,
        },
    )

    class Owner:
        pass

    owner = Owner()
    owner._vllm_hcu_static_eplb_row = (1, 0)
    owner.quant_method = SimpleNamespace(_hcu_static_eplb=True)
    meta = SimpleNamespace(_hcu_static_ht_native_counts=[1, 0])
    receiver = lambda: (torch.ones((1, 2)), None, meta, None, None)
    dispatcher = SimpleNamespace(
        _hcu_static_ht_owner=weakref.ref(owner),
        rank_expert_offset=2,
        _do_dispatch=lambda **kwargs: receiver,
    )
    quant = SimpleNamespace(
        is_block_quantized=False,
        is_per_act_token=False,
        quant_dtype="int8",
        a1_scale=None,
    )
    expert_map = torch.full((32,), -1, dtype=torch.int32)
    expert_map[2:4] = torch.tensor([0, 1], dtype=torch.int32)

    def prepare():
        return ns["ht_prepare_async"](
            SimpleNamespace(),
            dispatcher,
            torch.ones((1, 2)),
            torch.ones((1, 1)),
            torch.zeros((1, 1)),
            32,
            expert_map,
            False,
            quant,
        )

    prepare()()
    assert load().get_plan(meta, 2, 256) is not None
    meta._hcu_static_ht_scratch = None
    dispatcher.rank_expert_offset = 0
    prepare()()
    assert load().get_plan(meta, 2, 256) is None


def test_float_map_is_not_a_native_physical_map():
    mod = load()
    assert not mod.is_linear_map(torch.tensor([0.0, 1.0, -1.0, -1.0]), 4, 2, 0)


def test_prepare_accepts_static_owner_inference_map_and_detects_replacement(
    monkeypatch,
):
    import weakref

    from vllm_hcu.platforms import envs

    monkeypatch.setenv("VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH", "1")
    ns = extract_functions(
        PATH.parent / "deepep_runtime.py",
        {"ht_prepare_async"},
        {
            "current_platform": SimpleNamespace(is_rocm=lambda: True),
            "henvs": envs,
            "_uses_post_dispatch_channel_fp8_quant": lambda _: False,
        },
    )

    class Owner:
        pass

    owner = Owner()
    owner._vllm_hcu_static_eplb_row = (1, 0)
    owner.quant_method = SimpleNamespace(_hcu_static_eplb=True)
    meta = SimpleNamespace(_hcu_static_ht_native_counts=[1, 0])
    receiver = lambda: (torch.ones((1, 2)), None, meta, None, None)
    dispatcher = SimpleNamespace(
        _hcu_static_ht_owner=weakref.ref(owner),
        rank_expert_offset=0,
        _do_dispatch=lambda **kwargs: receiver,
    )
    quant = SimpleNamespace(
        is_block_quantized=False,
        is_per_act_token=False,
        quant_dtype="int8",
        a1_scale=None,
    )
    with torch.inference_mode():
        expert_map = torch.full((32,), -1, dtype=torch.int32)
        expert_map[:2] = torch.tensor([0, 1], dtype=torch.int32)
    owner.expert_map_manager = SimpleNamespace(expert_map=expert_map)
    result = ns["ht_prepare_async"](
        SimpleNamespace(),
        dispatcher,
        torch.ones((1, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 1)),
        32,
        expert_map,
        False,
        quant,
    )
    result()
    assert load().get_plan(meta, 2, 256) is not None
    owner.expert_map_manager.expert_map = expert_map.clone()
    assert load().get_plan(meta, 2, 256) is None


def test_workspace_and_apply_use_same_tagged_bound():
    import __future__

    import ast
    import math

    meta = SimpleNamespace(expert_num_tokens_cpu=torch.tensor([1, 0]))
    load().attach_plan(meta, [1, 0], 2, 100, False)
    ns = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_HCU_TOKEN_ALIGNMENT": 256,
        "round_up": lambda n, a: (n + a - 1) // a * a,
    }
    extract_functions(
        PATH.parent / "deep_gemm_utils.py",
        {
            "expert_num_tokens_round_up_and_sum",
            "compute_aligned_M_and_alignment",
            "compute_aligned_M",
        },
        ns,
    )
    observed = []

    def permute(**kwargs):
        observed.append(("input", kwargs["aq_out"].shape[0]))
        return (
            kwargs["aq_out"],
            torch.ones((256, 1)),
            torch.zeros(256),
            torch.zeros((100, 8)),
            256,
        )

    def gemm(a, b, out, indices):
        observed.append(("gemm", out.shape[0]))

    ns.update(
        {
            "_resize_cache": lambda x, shape: x.reshape(-1)[: math.prod(shape)].view(
                shape
            ),
            "deepgemm_moe_permute": permute,
            "m_grouped_w4a8_gemm_nt_contiguous_hipc": gemm,
            "deepgemm_unpermute_and_reduce": lambda **kwargs: None,
        }
    )
    runtime = PATH.parent.parent / "quantization/slimquant_w4a8_deepgemm_runtime.py"
    tree = ast.parse(runtime.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "DeepEPDeepGemmW4A8ContiguousExperts"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("workspace_shapes", "apply")
    ]
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(runtime),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    owner = SimpleNamespace(
        ALIGNMENT=256,
        adjust_N_for_activation=lambda n, a: n // 2,
        _validate_activation=lambda a: None,
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
        _deepgemm_w13=torch.ones(1),
        _deepgemm_w2=torch.ones(1),
        moe_problem_size=lambda *args: (2, 100, 4, 4, 8),
        _ensure_2d_scale=lambda x: x,
        _permute_scale_kwargs=lambda k: {},
        _quantize_activation=lambda gate, q, idx: (q, torch.ones((256, 1))),
    )
    shapes = ns["workspace_shapes"](owner, 100, 4, 4, 8, 32, 2, meta, None)
    assert shapes == ((256, 4), (256, 4), (100, 4))
    ns["apply"](
        owner,
        torch.empty((100, 4)),
        torch.ones((100, 4), dtype=torch.int8),
        torch.ones(1),
        torch.ones(1),
        torch.ones((100, 8)),
        torch.zeros((100, 8), dtype=torch.int64),
        None,
        32,
        None,
        torch.ones((100, 1)),
        None,
        torch.empty(shapes[0], dtype=torch.bfloat16),
        torch.empty(shapes[1], dtype=torch.bfloat16),
        meta,
        False,
    )
    assert observed == [("input", 256), ("gemm", 256), ("gemm", 256)]


def test_receiver_preserves_native_list_for_later_validation():
    ns = extract_functions(
        PATH.parent / "deepep_runtime.py",
        {"ht_receiver"},
        {
            "_uses_post_dispatch_channel_fp8_quant": lambda _: False,
        },
    )

    class Metadata:
        @staticmethod
        def make_from_list(values, device):
            return SimpleNamespace(
                expert_num_tokens_cpu=torch.tensor(values, dtype=torch.int32)
            )

    module = SimpleNamespace(
        torch=torch, mk=SimpleNamespace(ExpertTokensMetadata=Metadata)
    )
    owner = SimpleNamespace(rank_expert_offset=0, _hcu_static_ht_owner=object())
    counts = [1, 0]
    output = ns["ht_receiver"](
        module,
        owner,
        SimpleNamespace(event=None),
        False,
        torch.ones((1, 2)),
        torch.tensor([[0, -1]]),
        32,
        counts,
        torch.ones((1, 2)),
        None,
        SimpleNamespace(is_block_quantized=False, is_per_act_token=True),
        True,
    )
    assert output[2]._hcu_static_ht_native_counts is counts
    assert output[3].tolist() == [[0, 31]]


@pytest.mark.parametrize(
    ("eager", "expects_owner"),
    [(True, True), (False, False), (None, False)],
    ids=["eager", "not-eager", "missing-eager"],
)
@pytest.mark.parametrize("model_type", ["hy_v4", "hy_v4_mtp", "hy_v4_vl", "unknown"])
@pytest.mark.parametrize("scratch_enabled", [True, False])
@pytest.mark.parametrize("static_eplb", [True, False])
@pytest.mark.parametrize("route", ["ht", "ll", "auto"])
def test_post_load_caller_requires_explicit_eager_for_static_owner(
    monkeypatch, eager, expects_owner, model_type, scratch_enabled, static_eplb, route
):
    import sys
    import types

    root = PATH.parent.parent / "quantization/slimquant_w4a8.py"
    runtime_name = "vllm_hcu.model_executor.layers.fused_moe.deepep_runtime"
    factory_name = (
        "vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe"
    )
    runtime = types.ModuleType(runtime_name)
    runtime.slimquant_w4a8_uses_deepep_ht = lambda moe: route == "ht"
    runtime.slimquant_w4a8_uses_deepep_ll = lambda moe: route == "ll"
    runtime.slimquant_w4a8_uses_deepep_auto = lambda moe: route == "auto"
    monkeypatch.setitem(sys.modules, runtime_name, runtime)
    observed = []

    def factory(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(
            fused_experts=SimpleNamespace(
                process_weights_after_loading=lambda layer: None
            )
        )

    factories = types.ModuleType(factory_name)
    factories.make_deepep_ht_deepgemm_w4a8_moe_kernel = factory
    factories.make_deepep_ll_deepgemm_w4a8_moe_kernel = factory
    factories.make_deepep_auto_deepgemm_w4a8_moe_kernel = factory
    monkeypatch.setitem(sys.modules, factory_name, factories)
    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    if eager is not None:
        model_config.enforce_eager = eager
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.get_current_vllm_config_or_none = lambda: SimpleNamespace(
        model_config=model_config
    )
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    ns = {
        "torch": torch,
        "Parameter": torch.nn.Parameter,
        "_SLIMQUANT_MOE_LAYOUT_TENSORS": ("weight",),
        "_SLIMQUANT_MOE_WEIGHT_TENSORS": ("weight",),
        "_slimquant_moe_tensor_generation": lambda layer, names: 1,
        "henvs": SimpleNamespace(VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH=scratch_enabled),
    }
    import __future__

    import ast

    tree = ast.parse(root.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SlimQuantW4A8Int8AiterMoEMethod"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading"
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(root),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(1)), _expert_routing_tables=lambda: None
    )
    owner = SimpleNamespace(
        moe=object(),
        _hcu_static_eplb=static_eplb,
        get_fused_moe_quant_config=lambda layer: object(),
    )
    ns["process_weights_after_loading"](owner, layer)
    assert len(observed) == 1
    if (expects_owner and model_type != "unknown" and scratch_enabled
            and static_eplb and route == "ht"):
        assert observed[0]["static_owner"] is layer
    else:
        assert "static_owner" not in observed[0]
