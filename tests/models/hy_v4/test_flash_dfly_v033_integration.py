# SPDX-License-Identifier: Apache-2.0
"""Real Core v0.33 configuration and lifecycle tests; no target weights loaded.

The integration lane supplies HY4_DFLY_TEST_MODEL explicitly. CPU orchestration
checks below do not claim GPU forward/accuracy qualification.
"""
from copy import copy, deepcopy
from dataclasses import replace
import os

import pytest
import torch


@pytest.fixture(scope="module")
def engine():
    from vllm.config.speculative import SpeculativeConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm_hcu.models.hy_v4_flash import dfly_compat
    fields = SpeculativeConfig.__pydantic_fields__
    original = EngineArgs.create_speculative_config
    model = os.environ.get("HY4_DFLY_TEST_MODEL")
    if not model:
        pytest.skip("requires explicit checkpoint metadata path")
    dfly_compat.install_v033_dfly()
    args = EngineArgs(
        model=model, tensor_parallel_size=8, dtype="bfloat16",
        max_model_len=8192, max_num_seqs=2, max_num_batched_tokens=512,
        enforce_eager=True,
        speculative_config={"num_speculative_tokens": 5, "method": "dfly"},
        attention_backend="FLASHMLA_SPARSE", disable_custom_all_reduce=True,
        hf_overrides={"speculative_config": None},
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    cfg = args.create_engine_config()
    return cfg, fields, original


def test_real_requested_config(engine):
    cfg, fields, _ = engine
    from vllm.config.speculative import SpeculativeConfig
    from vllm_hcu.models.hy_v4_flash.dfly_config import HCUDFlySpeculativeConfig
    spec = cfg.speculative_config
    assert isinstance(spec, HCUDFlySpeculativeConfig)
    assert SpeculativeConfig.__pydantic_fields__ is fields
    assert "dfly" not in str(fields["method"].annotation)
    assert spec.method == "dfly"
    assert spec.num_speculative_tokens == 5
    assert cfg.num_lookahead_tokens == 6
    assert cfg.model_config.max_model_len == 8192
    assert spec.draft_model_config.architectures == ["HYV4DFlyModel"]
    assert spec.draft_model_config.hf_config.model_type == "hy_v4_dfly"
    assert spec.draft_model_config.hf_config.num_hidden_layers == 5
    assert spec.draft_parallel_config.tensor_parallel_size == 8
    assert not spec.draft_parallel_config.enable_expert_parallel
    assert spec.draft_model_config.quantization == cfg.model_config.quantization
    assert not spec.parallel_drafting  # Core's P-EAGLE-only padding flag.
    assert spec.use_eagle()  # Core v0.33's target-hidden consumer predicate.
    assert not spec.use_dflash() and not spec.use_dspark()
    assert not spec.use_eagle_block_drop()
    assert spec.max_num_new_slots_for_drafting == 0


def test_installer_idempotence_and_ownership(engine, monkeypatch):
    from vllm.engine.arg_utils import EngineArgs
    from vllm_hcu.models.hy_v4_flash import dfly_compat
    assert dfly_compat.install_v033_dfly() is False
    monkeypatch.setattr(EngineArgs, "create_speculative_config", object())
    with pytest.raises(RuntimeError, match="owner changed"):
        dfly_compat.install_v033_dfly()


def test_non_dfly_constructor_unchanged(engine):
    cfg, _, original = engine
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config.speculative import SpeculativeConfig
    args = EngineArgs(model=cfg.model_config.model,
                      speculative_config={"method": "ngram", "num_speculative_tokens": 5})
    before = original(args, cfg.model_config, cfg.parallel_config)
    after = args.create_speculative_config(cfg.model_config, cfg.parallel_config)
    assert type(before) is type(after) is SpeculativeConfig
    assert before.method == after.method == "ngram"
    assert before.num_speculative_tokens == after.num_speculative_tokens == 5
    no_spec = copy(cfg)
    no_spec.speculative_config = None
    assert no_spec.num_lookahead_tokens == 0


@pytest.mark.parametrize("changes, message", [
    ({"num_speculative_tokens": 0}, "num_speculative_tokens"),
    ({"num_speculative_tokens": 17}, "1..16"),
    ({"model": "not-the-target-checkpoint"}, "target checkpoint"),
    ({"quantization": "awq"}, "preserve checkpoint quantization"),
    ({"draft_tensor_parallel_size": 1}, "target TP"),
    ({"enforce_eager": False}, "enforce_eager=True"),
    ({"parallel_drafting": True}, "Unsupported DFly"),
    ({"enable_adaptive_verification": True}, "Unsupported DFly"),
    ({"use_local_argmax_reduction": True}, "Unsupported DFly"),
    ({"use_heterogeneous_vocab": True}, "Unsupported DFly"),
    ({"draft_lm_head_quantization": "fp8"}, "Unsupported DFly"),
    ({"kv_cache_dtype": "fp8"}, "KV cache"),
    ({"rejection_sample_method": "synthetic"}, "Unsupported DFly"),
    ({"num_speculative_tokens_per_batch_size": [(1, 2, 3)]}, "Unsupported DFly"),
])
def test_unsupported_options_fail_closed(engine, changes, message):
    cfg, _, _ = engine
    with pytest.raises((ValueError, NotImplementedError), match=message):
        replace(cfg.speculative_config, **changes)


def test_constructed_config_can_be_revalidated(engine):
    cfg, _, _ = engine
    spec = replace(cfg.speculative_config)
    assert spec.method == "dfly" and spec.num_speculative_tokens == 5
    assert spec.prompt_lookup_min is None and spec.prompt_lookup_max is None


@pytest.mark.parametrize("field", ["data_parallel_size", "pipeline_parallel_size",
                                   "prefill_context_parallel_size", "decode_context_parallel_size"])
def test_unsupported_parallelism_fail_closed(engine, field):
    cfg, _, _ = engine
    parallel = copy(cfg.parallel_config)
    setattr(parallel, field, 2)
    with pytest.raises(ValueError, match=field):
        replace(cfg.speculative_config, target_parallel_config=parallel)


def test_real_speculator_does_not_mutate_target(engine):
    cfg, _, _ = engine
    from vllm.v1.worker.gpu.spec_decode import init_speculator
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_speculator import HYV4DFlySpeculator
    model_config, attn_config = cfg.model_config, cfg.attention_config
    attn_before = copy(vars(attn_config))
    draft = init_speculator(cfg, torch.device("cpu"), None)
    assert isinstance(draft, HYV4DFlySpeculator)
    assert cfg.model_config is model_config
    assert cfg.attention_config is attn_config
    assert vars(attn_config) == attn_before
    assert draft.attn_vllm_config is not cfg
    assert draft.attn_vllm_config.model_config is not model_config
    assert draft.attn_vllm_config.attention_config is not attn_config
    assert draft.attn_vllm_config.attention_config.sparse_mla_force_mqa
    assert draft.num_query_per_req == 6
    assert draft.mrope_positions.shape == (3, 512)
    assert draft.max_model_len_gpu.item() == 8192
    assert cfg.model_config.hf_text_config.model_type == "hy_v4_flash"


def test_derivation_preserves_target_metadata(engine):
    cfg, _, _ = engine
    from vllm_hcu.models.hy_v4_flash.dfly_config import derive_flash_dfly_config
    target = cfg.model_config.hf_config
    before = deepcopy(target.to_dict())
    draft = derive_flash_dfly_config(target)
    assert target.to_dict() == before
    assert draft.target_index_layer == [25, 29, 33, 37, 41]
    assert draft.target_kv_layers == [25, 29, 33, 37, 41]
    assert draft.target_index_topk == 2048
    assert draft.n_routed_experts == 0


@pytest.mark.parametrize("different_axes", [False, True])
def test_real_draft_mrope_matches_native(engine, different_axes):
    cfg, _, _ = engine
    from vllm.model_executor.layers.rotary_embedding import get_rope
    config = cfg.speculative_config.draft_model_config.hf_config
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(cfg):
        rope = get_rope(config.qk_rope_head_dim, max_position=8192,
                        is_neox_style=False, rope_parameters=config.rope_parameters,
                        dtype=torch.bfloat16).to("cuda")
    pos = torch.tensor([0, 1, 127, 128, 8186, 8190], device="cuda")
    axes = pos.repeat(3, 1)
    if different_axes:
        axes[1] = (axes[1] - 1).clamp_min(0)
        axes[2] = (axes[2] - 3).clamp_min(0)
    torch.manual_seed(123)
    q = torch.randn(6, 8, config.qk_rope_head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(6, 1, config.qk_rope_head_dim, device="cuda", dtype=torch.bfloat16)
    expected = rope.forward_native(axes, q.clone(), k.clone())
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly import HYV4DFlyRotaryEmbedding
    wrapped = HYV4DFlyRotaryEmbedding(rope)
    # Match the MLA caller: q_pe is a non-contiguous slice of a wider head.
    padded = torch.empty(6, 8, config.qk_head_dim, device="cuda", dtype=q.dtype)
    padded[..., -config.qk_rope_head_dim:] = q
    query_view = padded[..., -config.qk_rope_head_dim:]
    assert not query_view.is_contiguous()
    actual = wrapped(axes, query_view, k.clone())
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02)
    if not different_axes:
        one_d = wrapped(pos, q.clone(), k.clone())
        for result, reference in zip(actual, one_d):
            torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("tp_rank", range(8))
def test_real_autoloader_preserves_v033_split_qkv_and_sinks(monkeypatch, tp_rank):
    from types import SimpleNamespace
    from torch import nn
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly import HYV4DFlyForCausalLM
    from vllm_hcu.models.hy_v4_flash.nvidia import dfly as dfly_module
    monkeypatch.setattr(dfly_module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(dfly_module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    draft = HYV4DFlyForCausalLM.__new__(HYV4DFlyForCausalLM)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    for name in ("q_a_proj", "kv_a_proj_with_mqa"):
        setattr(layer.self_attn, name, nn.Linear(4, 3, bias=False, dtype=torch.bfloat16))
    layer.self_attn.learnable_sink_param = nn.Parameter(torch.empty(4, dtype=torch.float32))
    draft.model.layers = nn.ModuleList([layer])
    draft.model.ffn_width = 8
    draft.model.resolve_post_proj_norm = lambda seen: set()
    draft.config = SimpleNamespace(hidden_size=4, num_attention_heads=32)
    tensors = {name: torch.full((3, 4), value, dtype=torch.bfloat16)
               for name, value in (("q_a_proj", 2), ("kv_a_proj_with_mqa", 3))}
    weights = [(f"model.dfly_layers.0.self_attn.{name}.weight", tensor)
               for name, tensor in tensors.items()]
    weights.append(("model.dfly_layers.0.self_attn.learnable_sink_param",
                    torch.arange(32, dtype=torch.bfloat16)))
    with torch.no_grad():
        loaded = draft.load_weights(weights)
    assert loaded == {f"model.layers.0.self_attn.{name}.weight" for name in tensors} | {
        "model.layers.0.self_attn.learnable_sink_param"}
    torch.testing.assert_close(layer.self_attn.learnable_sink_param,
                               torch.arange(tp_rank * 4, (tp_rank + 1) * 4, dtype=torch.float32))
    for name, tensor in tensors.items():
        torch.testing.assert_close(getattr(layer.self_attn, name).weight, tensor)
    assert "fused_qkv_a_proj" not in draft.packed_modules_mapping

@pytest.fixture
def kv_alias_case(engine):
    """Real MLA module types and Core binding, without loading model weights."""
    from torch import nn
    from vllm.model_executor.layers.attention import MLAAttention
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, KVCacheGroupSpec
    cfg = copy(engine[0])
    cfg.compilation_config = copy(cfg.compilation_config)
    cfg.kernel_config = copy(cfg.kernel_config)
    cfg.kernel_config.enable_jit_warmup = False
    cfg.cache_config = copy(cfg.cache_config)
    from vllm.v1.kv_cache_layout import KVCacheLayout
    cfg.cache_config.kv_cache_layout = KVCacheLayout.LBHNC.name
    names = ("language_model.model.layers.25.self_attn.attn",
             "dfly.model.layers.0.self_attn.attn")
    layers = []
    for name, target in zip(names, (None, names[0]), strict=True):
        layer = MLAAttention.__new__(MLAAttention)
        nn.Module.__init__(layer)
        layer.layer_name = name
        layer.kv_sharing_target_layer_name = target
        layer._vllm_config = cfg
        layers.append(layer)
    unrelated = nn.Linear(1, 1)
    unrelated.kv_sharing_target_layer_name = names[0]
    cfg.compilation_config.static_forward_context = dict(zip(names, layers, strict=True))
    cfg.compilation_config.static_forward_context["not_attention"] = unrelated
    spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=576,
                            dtype=torch.bfloat16)
    group = KVCacheGroupSpec(layer_names=[names[0]], kv_cache_spec=spec)
    return cfg, names, layers, group


def test_real_mla_alias_discovery_scoped_to_dfly(kv_alias_case):
    from vllm.v1.worker.gpu import attn_utils
    cfg, (target, draft), _, _ = kv_alias_case
    assert attn_utils.get_shared_kv_cache_layers(cfg) == {draft: target}
    no_spec = copy(cfg)
    no_spec.speculative_config = None
    # The old dense-only behavior must remain unchanged outside owned DFly.
    assert attn_utils.get_shared_kv_cache_layers(no_spec) == {}


@pytest.mark.parametrize("uniform", [False, True])
def test_real_mla_alias_groups_idempotent(kv_alias_case, uniform):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    from vllm.v1.worker.gpu import attn_utils
    cfg, (target, draft), _, group = kv_alias_case
    if uniform:
        group.kv_cache_spec = UniformTypeKVCacheSpecs(
            block_size=64, kv_cache_specs={target: group.kv_cache_spec})
    aliases = attn_utils.get_shared_kv_cache_layers(cfg)
    runner_layers = set()
    for _ in range(2):
        attn_utils.add_kv_sharing_layers_to_kv_cache_groups(
            aliases, [group], runner_layers)
    assert group.layer_names == [target, draft]
    assert runner_layers == {draft}
    if uniform:
        specs = group.kv_cache_spec.kv_cache_specs
        assert specs[draft] is specs[target]


def test_real_mla_alias_group_mismatch_rejected(kv_alias_case):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
    from vllm.v1.worker.gpu import attn_utils
    cfg, (target, draft), _, group = kv_alias_case
    other = KVCacheGroupSpec(layer_names=[draft], kv_cache_spec=group.kv_cache_spec)
    aliases = attn_utils.get_shared_kv_cache_layers(cfg)
    with pytest.raises(ValueError, match="share its target KV group"):
        attn_utils.add_kv_sharing_layers_to_kv_cache_groups(aliases, [group, other])
    assert group.layer_names == [target]


def test_real_mla_alias_allocator_binds_identical_storage(kv_alias_case, monkeypatch):
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu import attn_utils
    cfg, (target, draft), layers, group = kv_alias_case
    aliases = attn_utils.get_shared_kv_cache_layers(cfg)
    attn_utils.add_kv_sharing_layers_to_kv_cache_groups(aliases, [group])
    allocation = torch.zeros((2, 1, 64, 576), dtype=torch.bfloat16)
    # Stub only allocation; use Core's actual alias map and MLA cache binder.
    monkeypatch.setattr(attn_utils, "allocate_kv_cache",
                        lambda *args, **kwargs: {target: allocation})
    caches = attn_utils.init_kv_cache(cfg.compilation_config.static_forward_context,
        KVCacheConfig(num_blocks=2, kv_cache_tensors=[], kv_cache_groups=[group]),
        torch.device("cpu"), [64], cfg)
    assert caches[draft] is caches[target]
    assert layers[0].kv_cache.data_ptr() == layers[1].kv_cache.data_ptr()
    assert layers[0].kv_cache.shape == layers[1].kv_cache.shape == (2, 64, 576)
    layers[1].kv_cache[:, 1:7].fill_(23)
    assert torch.all(layers[0].kv_cache[:, 1:7] == 23)
    assert torch.all(layers[0].kv_cache[:, 7:] == 0)


def test_dfly_kv_discovery_owner_guard(engine, monkeypatch):
    from vllm.v1.worker.gpu import attn_utils
    from vllm_hcu.models.hy_v4_flash import dfly_compat
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", object())
    with pytest.raises(RuntimeError, match="owner changed"):
        dfly_compat.install_v033_dfly()




def test_real_eager_draft_manager_uses_v033_api(engine, monkeypatch):
    from types import SimpleNamespace
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm.v1.worker.gpu.spec_decode import init_speculator
    cfg = engine[0]
    compilation_before = copy(vars(cfg.compilation_config))
    # Only distributed PP rank is stubbed; exercise the real old-Core manager.
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: SimpleNamespace(
        is_first_rank=True, is_last_rank=True))
    draft = init_speculator(cfg, torch.device("cpu"), None)
    draft.init_cudagraph_manager(CUDAGraphMode.NONE)
    manager = draft.cudagraph_manager
    assert manager.decode_query_len == 6
    assert not manager.needs_capture()
    desc = manager.dispatch(num_reqs=1, num_tokens=6, uniform_token_count=6,
                            num_active_loras=0)
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_tokens == 6 and desc.num_reqs == 1
    assert vars(cfg.compilation_config) == compilation_before


@pytest.mark.parametrize("mode_name", ["FULL", "PIECEWISE"])
def test_draft_manager_rejects_unqualified_graph_modes(engine, mode_name):
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode import init_speculator
    draft = init_speculator(engine[0], torch.device("cpu"), None)
    with pytest.raises(NotImplementedError, match="graphs are not qualified"):
        draft.init_cudagraph_manager(getattr(CUDAGraphMode, mode_name))
