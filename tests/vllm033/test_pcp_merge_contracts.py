"""Merge contracts for PCP providers; CPU wiring checks, no GPU acceptance."""
from types import SimpleNamespace as NS

import pytest
import torch


def config(pcp=8, *, worker="auto", quantization="hcu_dsv41_channel_fp8"):
    return NS(
        model_config=NS(enforce_eager=True, quantization=quantization,
                        hf_config=NS(model_type="deepseek_v41", engram_layer_ids=[])),
        parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1,
            pipeline_parallel_size=1, prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1, enable_expert_parallel=True,
            enable_eplb=False, enable_dbo=False, worker_cls=worker),
        kernel_config=NS(moe_backend="deep_gemm", monokernel=False),
        scheduler_config=NS(max_num_batched_tokens=32768, max_num_seqs=128),
        use_v2_model_runner=True, speculative_config=None, lora_config=None,
        kv_transfer_config=None, cache_config=NS(swa_bounded_replay=False),
        attention_config=NS(backend=None, indexer_sparse_logits=False,
                            resolve_indexer_kv_dtype=lambda _: "fp8"),
    )


@pytest.mark.parametrize('decoder_init', ['_DECODER_INIT', '_FP4_PCP_DECODER_INIT'])
def test_pcp_expert_protocol_matches_real_expert_partition(monkeypatch, decoder_init):
    from hcu_vllm_das.models import v41_pcp as model

    def native_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.n_local_experts = self.n_local_physical_experts = 384
        self.experts = NS(moe_config=NS(num_local_experts=48))

    monkeypatch.setattr(model.native.DeepseekV4MoE, "__init__", native_init)
    constructor = getattr(model, decoder_init)
    owned = model.HcuV41PcpMoE(config())
    assert owned.n_local_physical_experts == 48
    assert owned.n_local_experts == 384
    assert constructor.__globals__["DeepseekV4MoE"] is type(owned)
    assert model.native.DeepseekV4DecoderLayer.__init__.__globals__["DeepseekV4MoE"] is model.native.DeepseekV4MoE


def test_pcp_model_keeps_native_names_for_global_mhc_oot():
    from hcu_vllm_das.models import v41_pcp as model
    from vllm.model_executor.layers import mhc

    assert model._DECODER_INIT.__globals__["MHCPreDelayedOp"] is mhc.MHCPreDelayedOp
    assert model._DECODER_INIT.__globals__["MHCPostOp"] is mhc.MHCPostOp
    assert model._BACKBONE_INIT.__globals__["MHCPostOp"] is mhc.MHCPostOp


@pytest.mark.parametrize("pcp,ep,ht,ll,backend,expected", [
    (8, 8, True, False, "deep_gemm", "pcp-deepgemm"),
    (8, 8, True, False, "deep_gemm_mega_moe", "mega"),
    (1, 8, False, True, "deep_gemm", "existing"),
    (1, 8, True, False, "deep_gemm", "existing"),
    (1, 8, False, False, "deep_gemm", "existing"),
    (4, 4, True, False, "deep_gemm", "existing"),
    (8, 8, True, False, "auto", "existing"),
])
def test_provider_dispatch_does_not_replace_decode_or_draft(monkeypatch, pcp, ep, ht, ll, backend, expected):
    from hcu_vllm_das import _seams
    from hcu_vllm_das.quantization import channel_fp8_moe as existing
    from hcu_vllm_das.moe import dsv41_deepgemm, dsv41_megamoe

    calls = []
    def constructor(name):
        def call(*args):
            calls.append(args)
            return name
        return call
    monkeypatch.setattr(existing, "HcuDSV41ChannelFP8MoEMethod", constructor("existing"))
    monkeypatch.setattr(dsv41_deepgemm, "HcuDSV41DeepGemmMethod", constructor("pcp-deepgemm"))
    monkeypatch.setattr(dsv41_megamoe, "HcuDSV41MegaMoEMethod", constructor("mega"))
    parallel = NS(tp_size=1, dp_size=1 if pcp > 1 else 8,
                  use_deepep_ht_kernels=ht, use_deepep_ll_kernels=ll,
                  all2all_backend="allgather_reducescatter" if not ht and not ll else "deepep")
    moe = NS(pcp_size=pcp, ep_size=ep, moe_backend=backend,
             moe_parallel_config=parallel)
    weights, inputs = object(), object()
    assert _seams._moe_provider()(weights, inputs, moe, "layer") == expected
    assert calls == [(weights, inputs, moe, "layer")]


@pytest.mark.parametrize("pcp,ep,ht", [(1, 8, True), (4, 4, True), (8, 8, False)])
def test_explicit_mega_cannot_fall_back_to_existing(monkeypatch, pcp, ep, ht):
    from hcu_vllm_das import _seams
    from hcu_vllm_das.quantization import channel_fp8_moe as existing

    monkeypatch.setattr(existing, "HcuDSV41ChannelFP8MoEMethod",
                        lambda *args: pytest.fail("explicit MegaMoE fell back"))
    moe = NS(pcp_size=pcp, ep_size=ep, moe_backend="deep_gemm_mega_moe",
        moe_parallel_config=NS(tp_size=1, dp_size=1,
            use_deepep_ht_kernels=ht, use_deepep_ll_kernels=not ht))
    with pytest.raises(ValueError, match="PCP8|HT"):
        _seams._moe_provider()(object(), object(), moe, "layer")


@pytest.mark.parametrize("worker,budget,expected", [
    ("hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker", 32768, 4352),
    ("hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker", 32769, 4353),
    ("example.CustomWorker", 32768, 32768),
])
def test_existing_host_engram_uses_same_capacity_as_profile_worker(monkeypatch, worker, budget, expected):
    from hcu_vllm_das import v41_engram as engram
    cfg = config(worker=worker)
    cfg.scheduler_config.max_num_batched_tokens = budget
    monkeypatch.setattr(engram, "get_current_vllm_config", lambda: cfg)
    obj = object.__new__(engram.HcuV41PcpEngram)
    torch.nn.Module.__init__(obj)
    obj.embed_tokens = NS(_hcu_pcp_size=8, part_n_hash_cols=3)
    with torch.device("meta"):
        obj._init_staging(budget, 256)
    assert obj.staged_rows.shape == (expected * 8, 3, 256)
    assert obj._prefetch_stream is None and obj._prefetch_done is None


@pytest.mark.parametrize("pcp,quant,expected", [
    (8, "hcu_dsv41_channel_fp8", True),
    (1, "hcu_dsv41_channel_fp8", False),
    (4, "hcu_dsv41_channel_fp8", False),
    (8, "hcu_dsv41_ct_w4a8", False),
    (8, "fp8", False),
])
def test_profile_worker_is_scoped_to_integrated_channel_fp8_route(pcp, quant, expected):
    from hcu_vllm_das.dsv41.worker import uses_pcp_channel_fp8
    assert uses_pcp_channel_fp8(config(pcp, quantization=quant)) is expected


@pytest.fixture
def publication(monkeypatch):
    from hcu_vllm_das import v41_pcp_activation as activation, v41_pcp
    from hcu_adapter.extensions import ExtensionContext
    from vllm import platforms, envs
    from vllm.model_executor.models import ModelRegistry
    selected = []
    monkeypatch.setattr(platforms, "current_platform", NS(
        is_rocm=lambda: True, device_name="hcu",
        register_worker_selector=lambda fn: selected.append(fn)))
    monkeypatch.setattr(activation, "_state", None)
    monkeypatch.setattr(activation, "_failure", None)
    monkeypatch.setattr(activation, "_publishing", False)
    monkeypatch.setattr(activation, "_adapter_bindings_valid", lambda: True)
    monkeypatch.setattr(v41_pcp, "bind_adapter_classes", lambda: None)
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    monkeypatch.setattr(envs, "environment_variables", dict(envs.environment_variables))
    envs.environment_variables.pop(activation._FACTOR, None)
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1")
    monkeypatch.setenv("VLLM_HCU_DSV41_ENGRAM_HOST", "0")
    monkeypatch.setenv("HCU_VLLM_FP4_INDEXER_IMPL", "stock")
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", "1")
    return activation, ExtensionContext("0.5.22", "0.33.0"), selected, ModelRegistry


def test_pcp_activation_installs_selector_once_before_model_publication(publication, monkeypatch):
    activation, context, selected, models = publication
    register = models.register_model
    def publish(model_arch, model_cls):
        assert len(selected) == 1
        return register(model_arch, model_cls)
    monkeypatch.setattr(models, "register_model", publish)
    activation.activate_extension(context)
    activation.activate_extension(context)
    assert len(selected) == 1
    assert models.models[activation._ARCH].module_name == activation._PLUGIN[0]


def test_profile_policy_change_after_publication_is_latched(publication, monkeypatch):
    from hcu_adapter.extensions import ExtensionCommitError
    activation, context, selected, models = publication
    activation.activate_extension(context)
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", "0")
    with pytest.raises(ExtensionCommitError):
        activation.check_health()
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", "1")
    with pytest.raises(ExtensionCommitError):
        activation.activate_extension(context)


def test_compatibility_paths_resolve_to_the_same_attention_and_model():
    from hcu_vllm_das.models import v41_pcp, dsv41_pcp
    from hcu_vllm_das import v41_pcp as binding
    from hcu_vllm_das.dsv41 import pcp_attention
    assert dsv41_pcp.HcuDSV41PcpForCausalLM is v41_pcp.HcuV41PcpForCausalLM
    assert pcp_attention.HcuDSV41PcpAttention is binding.HcuV41PcpAttention


def test_real_mhc_oot_allocation_is_used_by_the_pcp_constructor(monkeypatch):
    from hcu_vllm_das.models import v41_pcp as model
    from hcu_adapter.bridges import vllm_mhc as owner
    from vllm.model_executor.layers import mhc
    from vllm.model_executor import custom_op
    from vllm.model_executor.kernels.mhc import torch as kernels
    from vllm import ir

    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    # Only the runtime constructor is doubled; real declaration, register_oot
    # and CustomOp.__new__ execute. No kernel/device initialization is needed.
    backend = NS(MhcAiterRuntime=lambda *args: NS())
    declarations, _ = owner._declare_aiter(mhc, kernels, ir, backend, None, "1")
    for name, declaration in zip(("MHCPreDelayedOp", "MHCPostOp"), declarations[2:4]):
        original = model._DECODER_INIT.__globals__[name]
        original.register_oot(declaration[2])
        assert type(original.__new__(original)) is declaration[2]
    original = model._BACKBONE_INIT.__globals__["MHCPostOp"]
    assert type(original.__new__(original)) is declarations[3][2]


@pytest.mark.parametrize("enabled", [False, True])
def test_required_pcp_capability_is_owned_only_by_enabled_pcp(publication, monkeypatch, enabled):
    from hcu_adapter.extensions import ExtensionContext, RequiredExtensionError, activate_extensions
    activation, context, selected, models = publication
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1" if enabled else "0")
    point = NS(name="das_hcu_v41_pcp", load=lambda: activation.extension_spec)
    requested = ExtensionContext("0.5.22", "0.33.0",
        requested_capabilities={"dsv41-pcp-megamoe"})
    if enabled:
        reports = activate_extensions(requested, [point])
        assert reports[0].status.value == "active"
        assert models.models[activation._ARCH].module_name == activation._PLUGIN[0]
    else:
        with pytest.raises(RequiredExtensionError):
            activate_extensions(requested, [point])
        assert "dsv41-pcp-megamoe" not in activation.extension_spec().capabilities
        assert models.models[activation._ARCH].module_name == activation._NATIVE[0]
        assert not selected


def test_selector_publication_failure_latches_before_model_registration(publication, monkeypatch):
    from hcu_adapter.extensions import ExtensionCommitError
    activation, context, selected, models = publication
    def broken():
        selected.append("partially published selector")
        raise RuntimeError("worker selector failed")
    monkeypatch.setattr(activation, "_install_profile_worker", broken)
    with pytest.raises(ExtensionCommitError):
        activation.activate_extension(context)
    assert models.models[activation._ARCH].module_name == activation._NATIVE[0]
    monkeypatch.setattr(activation, "_install_profile_worker", lambda: None)
    with pytest.raises(ExtensionCommitError):
        activation.activate_extension(context)


@pytest.mark.parametrize("pcp,quant,expected", [
    (8, "hcu_dsv41_channel_fp8", 4352),
    (8, "hcu_dsv41_ct_w4a8", 32768),
    (4, "hcu_dsv41_channel_fp8", 32768),
])
def test_explicit_profile_worker_and_engram_reservation_use_same_scope(monkeypatch, pcp, quant, expected):
    from hcu_vllm_das.dsv41.worker import HcuDSV41PcpWorker
    from hcu_vllm_das.dsv41.profile_worker import WORKER, profile_token_capacity
    from hcu_adapter.bridges.vllm_worker import Worker
    from hcu_vllm_das import v41_engram as engram

    cfg = config(pcp, worker=WORKER, quantization=quant)
    worker = object.__new__(HcuDSV41PcpWorker)
    worker.vllm_config, worker.parallel_config, worker.rank = cfg, cfg.parallel_config, 0
    worker.model_runner = NS(max_num_tokens=32768, max_num_reqs=128, pcp_manager=object())
    profile_rows = []
    def core_profile(self):
        profile_rows.append(self.model_runner.max_num_tokens)
        return 123
    monkeypatch.setattr(Worker, "determine_available_memory", core_profile)
    assert worker.determine_available_memory() == 123
    assert profile_rows == [expected]
    assert worker.model_runner.max_num_tokens == 32768
    assert profile_token_capacity(cfg, pcp, 32768) == profile_rows[0]

    monkeypatch.setattr(engram, "get_current_vllm_config", lambda: cfg)
    obj = object.__new__(engram.HcuV41PcpEngram)
    torch.nn.Module.__init__(obj)
    obj.embed_tokens = NS(_hcu_pcp_size=pcp, part_n_hash_cols=3)
    with torch.device("meta"):
        obj._init_staging(32768, 256)
    assert obj.staged_rows.shape == (expected * pcp, 3, 256)
