# SPDX-License-Identifier: Apache-2.0
"""Review regressions using production AST, real Torch tensors and CPU Gloo.

Framework constructors are isolated; these tests do not qualify device kernels,
installed dispatch, checkpoint loading or full-model inference.
"""
import ast
from copy import deepcopy
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_hcu/models"


def _node(path, name):
    tree = ast.parse(path.read_text())
    return deepcopy(next(n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef) and n.name == name))


def _compile(node, namespace):
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
                 "<production-ast>", "exec"), namespace)
    return namespace[node.name]


def _processor_functions():
    ns = {"math": math}
    for name in ("adaptive_resize", "vision_token_grid", "vision_token_count"):
        _compile(_node(MODEL / "hy_v4/processor.py", name), ns)
    return ns


@pytest.mark.parametrize("spatial", [1, 2])
@pytest.mark.parametrize("budget", [1, 32, 8192])
def test_dummy_video_search_uses_supported_shapes(spatial, budget):
    ns = _processor_functions()
    fn = _compile(_node(MODEL / "hy_v4/mm_preprocess.py", "get_dummy_video_shape"), ns)
    vision = NS(patch_size=16, spatial_merge_size=2, spatial_patch_size=spatial,
                temporal_patch_size=2, cat_extra_token=True, use_image_newline_token=True,
                use_2d_rope=False)
    processor = NS(size={"longest_edge": 1024**2}, max_frames=4, do_resize=True,
                   video_max_image_size=128, timestamp_token_per_frame=2,
                   timestamp_token_reserve=0)
    info = NS(get_hf_processor=lambda: NS(video_processor=processor,
                  tokenizer=NS(encode=lambda text, **kw: list(text))),
              get_hf_config=lambda: NS(vision_config=vision),
              _merge_and_resolve_mm_processor_kwargs=lambda _: {})
    frames, height, width = fn(info, budget, {"video": 1})
    assert frames >= (2 if spatial == 2 else 1)
    assert ns["vision_token_count"]((frames, height // 16, width // 16), vision, "video") > 0


@pytest.mark.parametrize("images", [0, 1])
def test_disabled_video_does_not_search_or_build_video(images):
    fn = _compile(_node(MODEL / "hy_v4/mm_preprocess.py", "get_dummy_mm_data"), {})
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled video was accessed")
    builder = NS(info=NS(get_image_size_with_most_features=lambda: NS(width=32, height=32),
                         get_dummy_video_shape=forbidden),
                 _get_dummy_videos=forbidden,
                 _get_dummy_images=lambda **kw: ["image"] * kw["num_images"])
    actual = fn(builder, 8192, {"image": images, "video": 0}, {})
    assert actual == {"image": ["image"] * images, "video": []}


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("frames", [2, 3])
@pytest.mark.parametrize("trailing", [0, 4])
def test_video_extras_receive_rope_and_decode_delta(extra, frames, trailing):
    ns = _processor_functions()
    class Features:
        @staticmethod
        def gather_kwargs(features, names):
            return {"video_grid_thw": [torch.tensor([frames, 4, 4])]}
    ns.update(torch=torch, MultiModalFeatureSpec=Features)
    fn = _compile(_node(MODEL / "hy_v4/vl_model.py", "get_mrope_input_positions"), ns)
    vision = NS(spatial_merge_size=1, spatial_patch_size=1, temporal_patch_size=1,
                use_image_newline_token=False, cat_extra_token=extra)
    cfg = NS(vision_config=vision, image_start_token_id=10, image_end_token_id=11,
             video_start_token_id=12, video_end_token_id=13, video_token_id=14,
             image_token_id=15)
    tokens = [1, 12]
    grid_spans = []
    for _ in range(frames):
        tokens += [2, 10] + [14] * int(extra)
        grid_spans.append((len(tokens), len(tokens) + 16))
        tokens += [14] * (16 + int(extra)) + [11]
    tokens += [13] + [3] * trailing
    actual, delta = fn(NS(config=cfg), tokens, [object()])
    # Independent sequential boundary reference: every non-grid token,
    # including the extra embeddings, advances all three axes together.
    expected = torch.empty((3, len(tokens)), dtype=torch.long)
    cursor = position = 0
    for begin, end in grid_spans:
        expected[:, cursor:begin] = torch.arange(begin - cursor) + position
        position += begin - cursor
        expected[0, begin:end] = position
        expected[1, begin:end] = torch.arange(4).repeat_interleave(4) + position
        expected[2, begin:end] = torch.arange(4).repeat(4) + position
        position += 4
        cursor = end
    expected[:, cursor:] = torch.arange(len(tokens) - cursor) + position
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert delta == int(expected.max()) + 1 - len(tokens)
    assert delta == -12 * frames
    # Scheduler's next decode position must start after the final prompt max.
    assert len(tokens) + delta == int(expected.max()) + 1


@pytest.mark.parametrize("target_revision,explicit,valid", [
    ("target-sha", None, True), ("target-sha", "target-sha", True),
    ("target-sha", "other-sha", False), (None, None, True),
    (None, "draft-only-sha", False),
])
@pytest.mark.parametrize("field", ["revision", "code_revision"])
def test_dfly_revision_tracks_target(field, target_revision, explicit, valid):
    ns = {"torch": torch, "deepcopy": deepcopy,
          "derive_flash_dfly_config": lambda _: NS(num_attention_heads=32,
              dfly_intermediate_size=64, hidden_correction_intermediate_size=64),
          "ModelConfig": lambda **kw: NS(**kw), "validate_flash_quant_config": lambda _: None}
    fn = _compile(_node(MODEL / "hy_v4_flash/dfly_config.py", "__post_init__"), ns)
    target = NS(hf_text_config=NS(model_type="hy_v4_flash"), dtype=torch.bfloat16,
                supports_multimodal_inputs=False, enforce_eager=True, model="repo/model",
                model_weights=None, quantization=None, hf_config=NS(), revision=None,
                code_revision=None, tokenizer="repo/model", tokenizer_mode="auto",
                trust_remote_code=False, allowed_local_media_path="", allowed_media_domains=None,
                seed=0, tokenizer_revision="tokenizer-sha", max_model_len=8192,
                max_logprobs=20, config_format="auto")
    setattr(target, field, target_revision)
    parallel = NS(data_parallel_size=1, pipeline_parallel_size=1,
                  prefill_context_parallel_size=1, decode_context_parallel_size=1,
                  tensor_parallel_size=8)
    spec = NS(target_model_config=target, target_parallel_config=parallel, enforce_eager=None,
              parallel_drafting=False, enable_adaptive_verification=False,
              rejection_sample_method="standard", use_local_argmax_reduction=False,
              draft_lm_head_quantization=None, use_heterogeneous_vocab=False,
              num_speculative_tokens_per_batch_size=None, attention_backend=None,
              kv_cache_dtype=None, num_speculative_tokens=5, model=None, quantization=None,
              draft_tensor_parallel_size=None, revision=None, code_revision=None,
              max_model_len=None, create_draft_parallel_config=lambda *args: parallel)
    setattr(spec, field, explicit)
    if not valid:
        with pytest.raises(ValueError, match=field):
            fn(spec)
    else:
        fn(spec)
        assert getattr(spec, field) == target_revision
        assert getattr(spec.draft_model_config, field) == target_revision
        assert spec.draft_model_config.tokenizer_revision == "tokenizer-sha"


def _up_projection(hcu=True):
    node = _node(MODEL / "hy_v4_flash/nvidia/moe.py", "__init__")
    # First __init__ is the feed-forward constructor; select the MoE class.
    tree = ast.parse((MODEL / "hy_v4_flash/nvidia/moe.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HYV4FlashMoE")
    node = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
    cut = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "__init__")
    node.body = node.body[:cut] + [ast.Return(value=ast.Name(id="up", ctx=ast.Load()))]
    ns = {"nn": torch.nn, "is_hcu_platform": lambda: hcu,
          "ReplicatedLinear": lambda a, b, **kw: torch.nn.Linear(a, b, bias=False)}
    fn = _compile(node, ns)
    return fn(None, NS(routed_expert_hidden_size=2, hidden_size=2, flash_w4a8_backend="disabled"), None, "model.mlp")


def _runner(collective):
    path = ROOT / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    ns = {"torch": torch, "os": os, "tensor_model_parallel_all_reduce": collective,
          "_unpack": lambda result: result}
    methods = ["apply_routed_output_transform", "_maybe_reduce_shared_expert_output",
               "_maybe_reduce_routed_output_before_transform", "_can_fuse_kimi_k3_latent_all_reduce",
               "_fuse_kimi_k3_latent_all_reduce", "_maybe_reduce_final_output",
               "_maybe_apply_routed_scale_to_output"]
    attrs = {name: _compile(_node(path, name), ns) for name in methods}
    node = _node(path, "forward")
    cut = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Tuple)
               and [x.id for x in n.targets[0].elts] == ["shared_output", "fused_output"])
    stub = ast.parse("def post(self, result, og_hidden_dim_pre_xform=None, og_hidden_dim_post_xform=None): pass").body[0]
    stub.body = node.body[cut:]
    attrs["post"] = _compile(stub, ns)
    attrs["_maybe_add_zero_expert_output"] = lambda self, value: value
    return type("ProductionRunnerTail", (), attrs)()


def _dynamic_quant(x, kind):
    bound = 127 if kind == "int8" else 448
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-20) / bound
    scaled = x / scale
    quantized = (scaled.round().clamp(-127, 127).to(torch.int8) if kind == "int8"
                 else scaled.to(torch.float8_e4m3fn))
    return quantized.float() * scale


def _check_runner(rank, world):
    import torch.distributed as dist
    for kind in ("int8", "fp8"):
        for shared in (False, True):
            for reduced in (False, True):
                for fuse in (False, True):
                    os.environ["VLLM_HCU_KIMI_LATENT_MOE_FUSE_ALLREDUCE"] = str(int(fuse))
                    calls = []
                    def collective(value):
                        calls.append(value.shape[-1])
                        value = value.clone()
                        if world > 1:
                            dist.all_reduce(value)
                        return value
                    runner = _runner(collective)
                    runner.routed_output_transform = _up_projection()
                    runner.routed_output_transform.forward = lambda x: _dynamic_quant(x, kind)
                    runner._shared_experts = object() if shared else None
                    runner._fused_output_is_reduced = reduced
                    runner._ep_fused_combine = False
                    runner.routed_scaling_factor = 1.0
                    runner.moe_config = NS(tp_size=world, ep_size=1, is_sequence_parallel=False,
                                           skip_final_all_reduce=False)
                    partial = torch.tensor([[128. if rank == 0 else -128., .1]])
                    routed_sum = torch.tensor([[0., .2]]) if world == 2 else partial.clone()
                    shared_partial = torch.tensor([[1., 2.]]) if rank == 0 else torch.tensor([[3., 4.]])
                    shared_sum = torch.tensor([[4., 6.]]) if world == 2 else shared_partial.clone()
                    result = runner.post((shared_partial if shared else None,
                                          routed_sum.clone() if reduced else partial))
                    expected = _dynamic_quant(routed_sum, kind) + (shared_sum if shared else 0)
                    torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)
                    if world == 2:
                        expected_calls = (int(shared) if reduced else
                                          (1 if not shared or fuse else 2))
                        assert len(calls) == expected_calls


def _gloo_worker(rank, rendezvous):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        _check_runner(rank, 2)
    finally:
        dist.destroy_process_group()


def test_quantized_up_projection_tp2_real_cpu_collectives(tmp_path):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires CPU Gloo")
    torch.multiprocessing.spawn(_gloo_worker,
        args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


def test_up_projection_marker_is_hcu_only():
    assert getattr(_up_projection(True), "_hcu_fuse_shared_and_routed_tp_all_reduce", False)
    assert not getattr(_up_projection(False), "_hcu_fuse_shared_and_routed_tp_all_reduce", False)


def test_quantized_up_projection_tp1(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_KIMI_LATENT_MOE_FUSE_ALLREDUCE", "0")
    _check_runner(0, 1)
