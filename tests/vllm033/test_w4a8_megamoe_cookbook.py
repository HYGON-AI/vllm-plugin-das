"""CPU composition of MegaMoE with native replay and DSpark ownership."""
from types import SimpleNamespace as NS
import pytest
from test_dsv41_megamoe_cpu import mega_config
from hcu_vllm_das.moe import dsv41_megamoe as common
from hcu_vllm_das.moe import dsv41_w4a8_megamoe as w4
from hcu_vllm_das.dsv41.profile_worker import profile_token_capacity, WORKER


@pytest.mark.parametrize("replay,speculative", [(False, False), (True, False), (False, True), (True, True)])
def test_real_w4a8_constructor_bounds_tile_capacity(monkeypatch, replay, speculative):
    config, moe = mega_config()
    config.model_config.quantization = "hcu_dsv41_ct_w4a8"
    config.parallel_config.worker_cls = WORKER
    config.scheduler_config = NS(max_num_batched_tokens=32768, max_num_seqs=128)
    config.cache_config.swa_bounded_replay = replay
    if speculative:
        config.speculative_config = NS(method="dspark", moe_backend="deep_gemm",
                                       draft_model_config=NS(enforce_eager=True))
    moe.max_num_tokens = 32768
    monkeypatch.setattr(common, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(w4, "w4a8_provider", lambda: NS())
    method = w4.HcuDSV41W4A8MegaMoEMethod(None, None, moe, "experts")
    assert method.buffer_capacity == 4352


@pytest.mark.parametrize("method,backend,eager", [
    ("eagle", "deep_gemm", True), ("dspark", "deep_gemm_mega_moe", True),
    ("dspark", "deep_gemm", False)])
def test_unsupported_draft_fails_before_provider(monkeypatch, method, backend, eager):
    config, moe = mega_config()
    config.speculative_config = NS(method=method, moe_backend=backend,
                                   draft_model_config=NS(enforce_eager=eager))
    monkeypatch.setattr(common, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(w4, "w4a8_provider", lambda: pytest.fail("provider must remain unopened"))
    with pytest.raises(ValueError, match="eager DSpark with deep_gemm draft"):
        w4.HcuDSV41W4A8MegaMoEMethod(None, None, moe, "experts")


def test_native_dspark_keeps_deepgemm_draft_for_megamoe_target(monkeypatch):
    import test_pcp_dspark_draft as draft
    original = draft.config
    def target_config():
        cfg = original()
        cfg.kernel_config.moe_backend = "deep_gemm_mega_moe"
        cfg.speculative_config.moe_backend = "deep_gemm"
        return cfg
    monkeypatch.setattr(draft, "config", target_config)
    cfg = draft.native_draft_config()
    assert cfg.kernel_config.moe_backend == "deep_gemm_mega_moe"
    owned = draft.draft_config(cfg)
    assert owned.kernel_config.moe_backend == "deep_gemm"
    assert owned.kernel_config is not cfg.kernel_config
    assert cfg.kernel_config.moe_backend == "deep_gemm_mega_moe"
    assert owned.parallel_config.prefill_context_parallel_size == 1
    assert draft.moe_pcp_size(owned) == 8
    assert "hcu_static_eplb" not in owned.additional_config


from test_fp4_full_chain import chain
from test_v41_static_eplb import static_config
from hcu_vllm_das import static_eplb_runtime as static


def test_real_megamoe_constructor_admits_416_physical_experts(chain, monkeypatch):
    monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "native")
    cfg = static_config(chain.cfg)
    cfg.model_config.quantization = "hcu_dsv41_ct_w4a8"
    cfg.kernel_config.moe_backend = "deep_gemm_mega_moe"
    _, moe = mega_config()
    moe.num_experts = 416
    moe.max_num_tokens = cfg.scheduler_config.max_num_batched_tokens
    monkeypatch.setattr(common, "get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr(w4, "w4a8_provider", lambda: NS())
    method = w4.HcuDSV41W4A8MegaMoEMethod(None, None, moe, "model.layers.0.ffn.experts")
    assert method.supports_eplb
    assert method.buffer_capacity == min(moe.max_num_tokens, (moe.max_num_tokens + 7) // 8 + 2 * cfg.scheduler_config.max_num_seqs)
    assert static.moe_eplb_kwargs(cfg, "model.layers.0.ffn.experts") == {
        "enable_eplb": True, "num_redundant_experts": 32}
    moe.num_experts = 384
    with pytest.raises(ValueError, match="expert counts"):
        w4.HcuDSV41W4A8MegaMoEMethod(None, None, moe, "experts")


@pytest.mark.parametrize("quant", ["hcu_dsv41_channel_fp8", "fp8", None])
def test_static_megamoe_does_not_expand_other_quantization(chain, monkeypatch, quant):
    monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "native")
    cfg = static_config(chain.cfg)
    cfg.kernel_config.moe_backend = "deep_gemm_mega_moe"
    cfg.model_config.quantization = quant
    with pytest.raises(static.StaticEplbError, match="owned DeepGEMM"):
        static.validate_config(cfg)


@pytest.mark.parametrize("rank", [0, 1])
def test_megamoe_packed_checkpoint_fanout_uses_native_loader(chain, monkeypatch, tmp_path, rank):
    import json
    import torch
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from replay_v41_static_eplb import cfg as placement_config

    experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(experts)
    experts.quant_config = NS(get_name=lambda: "hcu_dsv41_ct_w4a8")
    method = w4.HcuDSV41W4A8MegaMoEMethod.__new__(w4.HcuDSV41W4A8MegaMoEMethod)
    method.moe_kernel = None
    experts.quant_method = method
    experts.local_num_experts = 3
    experts.moe_config = NS(num_logical_experts=4, num_experts=6, num_local_experts=3,
        tp_rank=0, is_act_and_mul=True, tp_shard_with_padding=False,
        moe_parallel_config=NS(tp_size=1))
    experts.expert_map_manager = NS(num_fused_shared_experts=0,
        map_global_to_local=lambda e: e-rank*3 if rank*3 <= e < (rank+1)*3 else -1)
    experts._fused_shared_expert_quantizer = None
    method.create_weights(experts, 3, 512, 256, torch.bfloat16, weight_loader=experts.weight_loader)
    cfg = placement_config(width=2, num_redundant_experts=2)
    cfg.model_config.hf_config.n_routed_experts = 4
    row = [2, 0, 3, 1, 0, 1]
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"version": 2,
        "format": "vllm_offline_eplb_physical_to_logical_by_model",
        "model_maps": {"CTModel": {"record_kind": "proposal", "model_class": "CTModel",
            "model_name": cfg.model_config.model, "num_moe_layers": 1,
            "num_logical_experts": 4, "num_physical_experts": 6,
            "num_redundant_experts": 2, "physical_to_logical_map": [row]}}}))
    cfg.additional_config["hcu_static_eplb"].update(expert_map_path=str(path))
    model = NS(moe_mlp_layers=[NS(experts=NS(routed_experts=experts))],
        num_moe_layers=1, num_redundant_experts=2, num_physical_experts=6, num_logical_experts=4)
    monkeypatch.setattr(static, "resolve_model_key", lambda m: "CTModel")
    monkeypatch.setattr("vllm.model_executor.models.interfaces.get_mixture_of_experts_model", lambda m: m)
    static.bind_placement(model, cfg)
    for logical in range(4):
        for shard, rows, cols in [("w1", 256, 512), ("w3", 256, 512), ("w2", 512, 256)]:
            stem = "w2" if shard == "w2" else "w13"
            for suffix, data in [
                ("weight_packed", torch.full((rows, cols//8), logical*0x11111111, dtype=torch.int32)),
                ("weight_scale", torch.full((rows, 1), float(logical+1))),
                ("weight_shape", torch.tensor([rows, cols]))]:
                name = stem+"_"+suffix
                param = getattr(experts, name)
                loaded = param.weight_loader(param, data, name, shard_id=shard,
                    expert_id=logical, return_success=True)
                assert bool(loaded) == (logical in row[rank*3:(rank+1)*3])
    for local, logical in enumerate(row[rank*3:(rank+1)*3]):
        for stem in ("w13", "w2"):
            assert torch.all(getattr(experts, stem+"_weight_packed")[local] == logical*0x11111111)
            assert torch.all(getattr(experts, stem+"_weight_scale")[local] == logical+1)
        assert torch.equal(experts.w13_weight_shape[local], torch.tensor([256, 512]))
        assert torch.equal(experts.w2_weight_shape[local], torch.tensor([512, 256]))


def test_megamoe_static_dspark_replay_uses_existing_layer_split(monkeypatch):
    from test_pcp_dspark import config
    from hcu_vllm_das import pcp_dspark, v41_pcp_layer_split
    from hcu_vllm_das.models import v41_pcp as models
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1")
    monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "native")
    cfg = config()
    cfg.model_config.quantization = "hcu_dsv41_ct_w4a8"
    cfg.kernel_config.moe_backend = "deep_gemm_mega_moe"
    cfg.speculative_config.moe_backend = "deep_gemm"
    cfg.additional_config["hcu_layer_split"] = {"enabled": True}
    cfg.model_config.hf_config.kv_source_layer_ids = [2]
    cfg.cache_config.swa_bounded_replay = True
    assert static.validate_config(cfg)
    selected = models.select_attention(cfg)
    assert issubclass(selected, pcp_dspark.HcuV41PcpDSparkAttention)
    assert selected.forward_mqa is v41_pcp_layer_split.LayerSplitAttentionMixin.forward_mqa
    assert selected._indexer_cls(None, None) is v41_pcp_layer_split.HcuV41PcpIndexer


def test_forward_pool_profiles_full_batch_and_releases_segments(monkeypatch):
    from contextlib import contextmanager
    import sys
    import torch
    from hcu_vllm_das.moe import dsv41_forward_pool as fp

    events = []
    pools = []
    def new_pool(allocator, use_on_oom):
        assert allocator == "segments" and use_on_oom
        pools.append(len(pools))
        return pools[-1]

    @contextmanager
    def enter(pool):
        events.append(("enter", pool))
        yield
        events.append(("exit", pool))

    monkeypatch.setitem(sys.modules, "deepgemm.mega", NS(_C=NS(__file__="provider.so")))
    monkeypatch.setattr(torch.cuda.memory, "CUDAPluggableAllocator",
                        lambda *a: NS(allocator=lambda: "segments"))
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "MemPool", new_pool)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", enter)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(fp, "_reserve_attention_workspace", lambda runner: events.append("reserve"))
    runner = NS(execute_model=lambda n: n)
    runner.profile_run = lambda: runner.execute_model(32768)
    config = NS(model_config=NS(quantization="hcu_dsv41_ct_w4a8"),
                kernel_config=NS(moe_backend="deep_gemm"))
    ctx = NS(worker=NS(model_runner=runner), vllm_config=config)
    fp.configure_forward_pool(ctx)
    assert not pools
    config.kernel_config.moe_backend = "deep_gemm_mega_moe"
    fp.configure_forward_pool(ctx)
    fp.configure_forward_pool(ctx)
    assert runner.profile_run() == 32768
    assert runner.execute_model(8) == 8
    assert pools == [0, 1]
    assert events == ["reserve", ("enter", 0), ("exit", 0), "sync", ("enter", 1), ("exit", 1)]
