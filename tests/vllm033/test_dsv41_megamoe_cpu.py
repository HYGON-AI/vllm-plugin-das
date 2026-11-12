"""CPU contracts for PCP MegaMoE; native GPU execution is not tested here."""
from types import SimpleNamespace as NS
import pytest
import torch
from hcu_vllm_das.moe.dsv41_megamoe import MegaExperts, MegaPrepare, validate_config, provider

@pytest.mark.parametrize("rows",[0,1,7])
def test_collective_empty_rank_and_no_weight_rescaling(monkeypatch,rows):
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    calls=[]
    class Vendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,n):
            calls.append(("pre",n,ids.clone(),weights.clone()))
        def fp8_mega_moe(self,out,l1,l2,buf,**kw):
            calls.append(("launch",out.shape[0],kw));out.fill_(3)
    expert=object.__new__(MegaExperts)
    expert.moe_config=NS(hidden_dim=8,experts_per_token=2)
    expert.quant_config=NS(gemm1_clamp_limit=7.0)
    expert.buffer=NS(num_max_tokens_per_rank=16)
    expert.mega=Vendor();expert.l1=expert.l2=None
    x=torch.zeros(rows,8,dtype=torch.bfloat16)
    ids=torch.zeros(rows,2,dtype=torch.int64);weights=torch.full((rows,2),0.25)
    out=torch.empty_like(x);expert.run(out,x,ids,weights)
    assert calls[-1][0:2]==("launch",rows)
    assert calls[-1][2]["activation_clamp"]==7.0
    assert len(calls)==(2 if rows else 1)
    if rows:assert torch.equal(calls[0][3],weights)
    assert torch.equal(out,torch.full_like(out,3))


def test_prepare_reuses_core_passthrough():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import PassThroughPrepareAndFinalize
    prepare = MegaPrepare()
    assert MegaPrepare.prepare is PassThroughPrepareAndFinalize.prepare
    assert MegaPrepare.finalize is PassThroughPrepareAndFinalize.finalize
    assert prepare.output_is_reduced() and not prepare.supports_deferred_moe_finalize()
    assert prepare.topk_indices_dtype() is torch.int64


def test_native_provider_signature():
    assert provider().__name__=="deepgemm.mega"


@pytest.mark.parametrize("write_outputs",[False,True])
def test_startup_probe_rejects_nonexecuting_vendor(write_outputs):
    from hcu_vllm_das.moe.dsv41_megamoe import preflight_staging
    class Vendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,rows):
            if write_outputs:
                buf.topk_idx.copy_(ids);buf.topk_weights.copy_(weights)
                sf=x.float().abs().amax(-1)*torch.tensor(1/448,dtype=torch.float32)
                buf.x.copy_((x.float()/sf[:,None]).to(buf.x.dtype));buf.x_sf.copy_(sf)
    vendor=Vendor()
    if write_outputs:
        preflight_staging(vendor,"cpu")
    else:
        with pytest.raises(RuntimeError,match="startup execution probe"):
            preflight_staging(vendor,"cpu")


def test_rejected_reload_latches_forward_failure():
    from hcu_vllm_das.moe.dsv41_megamoe import HcuDSV41MegaMoEMethod
    method=object.__new__(HcuDSV41MegaMoEMethod)
    method._processed=True;method._failure=None
    with pytest.raises(RuntimeError,match="postprocessed twice"):
        method.process_weights_after_loading(None)
    with pytest.raises(RuntimeError,match="postprocessed twice"):
        method.apply(None,None,None,None,None,None)


def test_staging_routing_dtype_ignores_bf16_default():
    from hcu_vllm_das.moe.dsv41_megamoe import preflight_staging
    class Vendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,rows):
            assert weights.dtype == buf.topk_weights.dtype == torch.float32
            assert buf.x_sf.dtype == torch.float32
            buf.topk_idx.copy_(ids);buf.topk_weights.copy_(weights)
            sf=x.float().abs().amax(-1)*torch.tensor(1/448,dtype=torch.float32)
            buf.x.copy_((x.float()/sf[:,None]).to(buf.x.dtype));buf.x_sf.copy_(sf)
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        preflight_staging(Vendor(),"cpu")
    finally:
        torch.set_default_dtype(previous)


def test_incompatible_native_pool_is_destroyed_before_rejection():
    from hcu_vllm_das.moe.dsv41_megamoe import make_buffer
    destroyed=[]
    class Vendor:
        def get_symm_buffer_for_mega_moe(self,*args,**kwargs):
            return NS(l1_acts=NS(shape=(61441,5120)),l2_acts=NS(shape=(128,2304)),
                      destroy=lambda:destroyed.append(True))
    with pytest.raises(RuntimeError,match="pool capacity"):
        make_buffer(Vendor(),NS(size=lambda:8),392,2048,6,5120,2304)
    assert destroyed==[True]


def test_staging_rejects_division_at_fp8_midpoint():
    from hcu_vllm_das.moe.dsv41_megamoe import preflight_staging
    class DividingVendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,rows):
            scale=x.float().abs().amax(-1)/448
            buf.x.copy_((x.float()/scale[:,None]).to(buf.x.dtype))
            buf.x_sf.copy_(scale)
            buf.topk_idx.copy_(ids);buf.topk_weights.copy_(weights)
    with pytest.raises(RuntimeError,match="rounding/scale contract"):
        preflight_staging(DividingVendor(),"cpu")


def test_staging_checks_known_bf16_fp8_midpoint_bytes():
    from hcu_vllm_das.moe.dsv41_megamoe import preflight_staging
    class Vendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,rows):
            scale=x.float().abs().amax(-1)*torch.tensor(1/448,dtype=torch.float32)
            q=(x.float()/scale[:,None]).to(buf.x.dtype)
            assert int(scale[0].view(torch.int32)) == 0x3be00001
            assert int(q.view(torch.uint8)[0,1]) == 0xdf
            buf.x.copy_(q);buf.x_sf.copy_(scale)
            buf.topk_idx.copy_(ids);buf.topk_weights.copy_(weights)
    preflight_staging(Vendor(),"cpu")



def mega_config():
    config = NS(
        parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1,
            pipeline_parallel_size=1, enable_expert_parallel=True,
            enable_dbo=False, enable_eplb=False, expert_placement_strategy="linear"),
        model_config=NS(enforce_eager=True, enable_sleep_mode=False),
        speculative_config=None, offload_config=None)
    moe = NS(pcp_size=8, ep_size=8, num_experts=384, num_logical_experts=384,
        moe_parallel_config=NS(use_deepep_ht_kernels=True),
        in_dtype=torch.bfloat16, is_sequence_parallel=False, has_bias=False,
        is_lora_enabled=False, is_act_and_mul=True, activation=NS(value="silu"),
        experts_per_token=6, hidden_dim=5120, intermediate_size_per_partition=2304,
        swiglu_alpha=None, swiglu_beta=None, swiglu_limit=None, activation_situ_beta=None,
        activation_situ_linear_beta=None, skip_final_all_reduce=False)
    return config, moe


def test_target_contract_and_excluded_features():
    config, moe = mega_config()
    validate_config(config, moe)
    for owner, name, value, message in [
        (config, "speculative_config", object(), "speculative"),
        (config.parallel_config, "enable_eplb", True, "EPLB"),
        (moe, "num_experts", 392, "redundant"),
        (moe, "experts_per_token", 3, "target shape"),
        (config.model_config, "enforce_eager", False, "eager"),
        (config.parallel_config, "tensor_parallel_size", 2, "TP1"),
    ]:
        previous = getattr(owner, name)
        setattr(owner, name, value)
        with pytest.raises(ValueError, match=message):
            validate_config(config, moe)
        setattr(owner, name, previous)


def test_identity_routes_and_no_redundant_factory():
    from hcu_vllm_das.models import dsv41_pcp as model
    from vllm.models.deepseek_v41.amd import model as native
    assert model._DECODER_INIT.__globals__["DeepseekV4MoE"] is model.HcuDSV41PcpMoE
    assert model.HcuDSV41PcpMoE.__mro__[1] is native.DeepseekV4MoE
    ids = torch.tensor([[1, 5], [7, 2]])
    x = torch.ones(2, 8)
    weights = torch.full((2, 2), .25)
    result = MegaPrepare().prepare(x, weights, ids, 384, None, False, None, True)
    assert result[0] is x and result[3:] == (None, None)


def test_mega_native_modular_instance():
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    _, moe = mega_config()
    quant = FusedMoEQuantConfig.make(
        torch.float8_e4m3fn, per_act_token_quant=True, per_out_ch_quant=True)
    experts = MegaExperts(moe, quant, NS(), None, None, NS())
    assert experts.expects_unquantized_inputs
    assert experts._supports_parallel_config(NS(use_deepep_ht_kernels=True,
                                                use_deepep_ll_kernels=False))
    assert experts.supports_expert_map()


def test_pcp_reports_ep_local_expert_count(monkeypatch):
    from hcu_vllm_das.models import dsv41_pcp as model
    from vllm.models.deepseek_v41.amd import model as native
    def initialize(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.n_local_physical_experts = 384
        self.experts = NS(moe_config=NS(num_local_experts=48))
    monkeypatch.setattr(native.DeepseekV4MoE, "__init__", initialize)
    assert model.HcuDSV41PcpMoE().n_local_physical_experts == 48


@pytest.mark.parametrize("rows", [0, 2])
def test_kernel_keeps_native_routes_and_empty_rank_collective(rows):
    from hcu_vllm_das.moe.dsv41_megamoe import MegaKernel
    calls = []
    def run(output, x, ids, weights):
        calls.append((x, ids, weights))
        output.fill_(3)
    experts = NS(quant_config=None, run=run)
    kernel = NS(prepare_finalize=MegaPrepare(), fused_experts=experts)
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    ids = torch.zeros(rows, 6, dtype=torch.int64)
    weights = torch.full((rows, 6), .25)
    out = MegaKernel.apply(kernel, x, None, None, weights, ids, None, 384, None, False)
    assert len(calls) == 1 and calls[0][1] is ids and calls[0][2] is weights
    assert torch.equal(out, torch.full_like(x, 3))
    with pytest.raises(ValueError, match="after GEMM2"):
        MegaKernel.apply(kernel, x, None, None, weights, ids, None, 384, None, True)
