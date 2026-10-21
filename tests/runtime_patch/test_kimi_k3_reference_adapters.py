# SPDX-License-Identifier: Apache-2.0
"""CPU adapter/loader regressions; device kernels are explicit test doubles."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_hcu.model_executor.layers import kimi_k3_merged_mla_gate, kimi_k3_mla
from vllm_hcu.model_executor.layers.kimi_k3_prefill import LENGTHS_KEY, PREFILL_LENGTHS
from vllm_hcu.model_executor.layers.quantization.kimi_k3_situ import SituAndMul
from vllm_hcu.patch.worker.op_opt import (
    patch_kimi_k3_cpu_metadata,
    patch_kimi_k3_kda_gemm_warmup,
    patch_kimi_k3_kda_reference_paths,
    patch_kimi_k3_mla_gate,
    patch_kimi_k3_reference_paths,
)


class _Projection(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.weight = nn.Parameter(torch.full((sum(sizes), 2), -777.0))
        self.weight.weight_loader = self.load
        self.quant_method = type("UnquantizedLinearMethod", (), {})()

    def load(self, param, value, shard=None):
        with torch.no_grad():
            if shard is None:
                param.copy_(value)
            else:
                start = sum(self.sizes[:shard])
                param[start:start + self.sizes[shard]].copy_(value)


def _mla_target(monkeypatch):
    class Attention(nn.Module):
        def __init__(self, prefix=""):
            super().__init__()
            self.hidden_size = self.q_lora_rank = self.num_heads = 2
            self.kv_lora_rank = self.qk_rope_head_dim = self.v_head_dim = 1
            self.use_output_gate = True
            self.fused_qkv_a_proj = _Projection([2, 2])
            self.g_proj = _Projection([2])
            self.mla_attn = nn.Module()

    class Model(nn.Module):
        def __init__(self, fail=False):
            super().__init__()
            self.attn = Attention(prefix="attn")
            if fail:
                raise RuntimeError("construction failed")

        def load_weights(self, weights):
            return {name for name, _ in weights}

    class Merged(_Projection):
        def __init__(self, *args, **kwargs):
            super().__init__([2, 2, 2])

    target = ModuleType(patch_kimi_k3_mla_gate.TARGET_MODULE)
    target.KimiMLAAttention, target.KimiLinearModel = Attention, Model
    monkeypatch.setattr(kimi_k3_merged_mla_gate, "KimiK3MergedQKVGateLinear", Merged)
    monkeypatch.setattr(kimi_k3_mla, "HcuKimiK3MergedMLAWrapper", type("Wrapper", (nn.Module,), {}))
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_KIMI_FUSED_MLA_GATE", "1")
    assert patch_kimi_k3_mla_gate.apply_to_module(target)
    assert not patch_kimi_k3_mla_gate.apply_to_module(target)
    return target


@pytest.mark.parametrize("checkpoint_prefix", ["", "language_model."])
def test_mtp_native_loader_keeps_and_loads_all_projection_shards(monkeypatch, checkpoint_prefix):
    # Execute the installed MTP loader and its real spec-layer name rewrite.
    # Only projection allocation is replaced with small CPU tensors.
    from vllm.models.kimi_k3.amd.mtp import KimiK3MTP
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    target = _mla_target(monkeypatch)
    main = target.KimiLinearModel()
    assert not hasattr(main.attn, "g_proj")
    main.load_weights(iter([("attn.g_proj.weight", torch.full((2, 2), 3.0))]))
    torch.testing.assert_close(
        main.attn.fused_qkv_a_proj.weight[4:], torch.full((2, 2), 3.0)
    )
    attention = target.KimiMLAAttention(prefix="model.layers.1.self_attn")
    assert hasattr(attention, "g_proj")
    draft = KimiK3MTP.__new__(KimiK3MTP)
    nn.Module.__init__(draft)
    draft.config = KimiLinearConfig(
        num_hidden_layers=1, num_nextn_predict_layers=1, q_lora_rank=2,
        linear_attn_config=None,
    )
    draft.model = nn.Module()
    draft.model.mtp_start_layer_idx = 1
    draft.model.num_mtp_layers = 1
    block = nn.Module()
    block.self_attn = attention
    layer = nn.Module()
    layer.mtp_block = block
    draft.model.layers = nn.ModuleDict({"1": layer})
    weights = [
        (checkpoint_prefix + "model.layers.1.self_attn." + name + ".weight",
         torch.full((2, 2), value))
        for name, value in (("q_a_proj", 1.0), ("kv_a_proj_with_mqa", 2.0), ("g_proj", 3.0))
    ]
    loaded = draft.load_weights(iter(weights))
    assert "model.layers.1.mtp_block.self_attn.g_proj.weight" in loaded
    torch.testing.assert_close(attention.fused_qkv_a_proj.weight[:2], torch.ones(2, 2))
    torch.testing.assert_close(attention.fused_qkv_a_proj.weight[2:], torch.full((2, 2), 2.0))
    torch.testing.assert_close(attention.g_proj.weight, torch.full((2, 2), 3.0))


def test_main_loader_scope_restores_after_constructor_failure(monkeypatch):
    target = _mla_target(monkeypatch)
    with pytest.raises(RuntimeError, match="construction failed"):
        target.KimiLinearModel(fail=True)
    assert hasattr(target.KimiMLAAttention(), "g_proj")


def test_cpu_metadata_owns_decode_first_lengths(monkeypatch):
    patch = patch_kimi_k3_cpu_metadata
    class Builder:
        def build(self, *args):
            return SimpleNamespace(num_prefills=2, num_decodes=1,
                                   spec_sequence_masks=None, nums_dict={})

        def _build_chunk_metadata(self, *args):
            return "original"

    target = ModuleType(patch.TARGET_MODULE)
    target.KimiK3ROCmKDAMetadataBuilder = Builder
    assert patch.apply_to_module(target)
    assert not patch.apply_to_module(target)
    boundaries = torch.tensor([0, 1, 129, 385])
    metadata = Builder().build(0, SimpleNamespace(query_start_loc_cpu=boundaries))
    boundaries.zero_()
    assert metadata.nums_dict[LENGTHS_KEY] == [1, 128, 256]


def test_kda_adapter_preserves_v_and_resets_length_scope(monkeypatch):
    patch = patch_kimi_k3_kda_reference_paths
    calls = []
    target = ModuleType(patch.TARGET_MODULE)
    def original(**kwargs):
        calls.append("native")
        return kwargs["v"], kwargs["initial_state"]

    class Attention:
        prefix = "layer"
        def _forward(self, **kwargs):
            return target.chunk_kda_with_fused_gate(**kwargs)

    metadata = SimpleNamespace(nums_dict={LENGTHS_KEY: [128]}, num_decodes=0)
    target.causal_conv1d_fn = target.causal_conv1d_update = lambda *a, **k: None
    target.chunk_kda_with_fused_gate = original
    target.KimiK3DeltaAttention = Attention
    target.get_forward_context = lambda: SimpleNamespace(attn_metadata={"layer": metadata})
    assert patch.apply_to_module(target)
    assert not patch.apply_to_module(target)
    external = ModuleType("flash_kda")
    def fwd(q, k, v, g, beta, scale, out, A, bias, lower, **kwargs):
        calls.append("flash")
        out.copy_(v)
        kwargs["final_state"].copy_(kwargs["initial_state"])
        v.zero_()
    external.fwd = fwd
    monkeypatch.setitem(sys.modules, "flash_kda", external)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv("VLLM_KIMI_FLASHKDA_BACKEND", raising=False)
    kwargs = dict(q=torch.ones(1, 128, 1, 128), k=torch.ones(1, 128, 1, 128),
                  v=torch.ones(1, 128, 1, 128), raw_g=torch.ones(1, 128, 1, 128),
                  raw_beta=torch.ones(1, 128, 1), A_log=torch.ones(1),
                  g_bias=torch.ones(128), lower_bound=-5.0,
                  initial_state=torch.ones(1, 1, 128, 128),
                  cu_seqlens=torch.tensor([0, 128], dtype=torch.int32))
    output, _ = Attention()._forward(**kwargs)
    assert calls == ["flash"]
    torch.testing.assert_close(output, kwargs["v"])
    assert torch.all(kwargs["v"] == 1)
    assert PREFILL_LENGTHS.get() is None
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    Attention()._forward(**kwargs)
    assert calls == ["flash", "native"]


def test_kda_constructor_warmup_does_not_launch_for_cpu_weights(monkeypatch):
    patch = patch_kimi_k3_kda_gemm_warmup
    target = ModuleType(patch.TARGET_MODULE)
    class Attention:
        def __init__(self):
            self.in_proj_qkvgfab = _Projection([2])
    target.KimiK3DeltaAttention = Attention
    monkeypatch.setattr(patch, "_warmup_done", False)
    assert patch.apply_to_module(target)
    assert not patch.apply_to_module(target)
    assert Attention().in_proj_qkvgfab.weight.device.type == "cpu"
    assert not patch._warmup_done


@pytest.mark.parametrize("nn_setting", ["0", "1", None])
@pytest.mark.parametrize("failure", [None, "projection", "synchronize"])
def test_kda_device_warmup_uses_logical_input_width_and_retries(
    monkeypatch, nn_setting, failure
):
    import importlib.util
    from pathlib import Path

    from vllm.model_executor import custom_op, parameter
    from vllm_hcu.platforms import envs as henvs

    # Load the actual HCU methods with an isolated registration table: native
    # linear classes have already registered the same layer names in pytest.
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/linear.py"
    )
    spec = importlib.util.spec_from_file_location("kda_warmup_hcu_linear", path)
    hcu_linear = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as registration:
        registration.setattr(custom_op, "op_registry", {})
        spec.loader.exec_module(hcu_linear)

    if nn_setting is None:
        monkeypatch.delenv("VLLM_USE_NN", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_NN", nn_setting)
    nn_layout = nn_setting != "0"
    assert henvs.VLLM_USE_NN is nn_layout
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 2)
    # Real HCU allocation/apply methods and real CPU GEMMs; TP metadata, device
    # identity, scratch allocation and synchronization are explicit doubles.
    monkeypatch.setattr(
        hcu_linear, "dispatch_unquantized_gemm",
        lambda: lambda layer, x, weight, bias: torch.nn.functional.linear(
            x, weight, bias
        ),
    )
    calls, messages = [], []
    pending_failure = [failure]
    real_zeros = torch.zeros

    def scratch_zeros(shape, *, device, dtype):
        assert device == torch.device("cuda:0")
        return real_zeros(shape, device="cpu", dtype=dtype)

    def synchronize(device):
        assert device == torch.device("cuda:0")
        if pending_failure[0] == "synchronize":
            pending_failure[0] = None
            raise RuntimeError("injected synchronize failure")

    class Projection:
        input_size = 4
        output_partition_sizes = [5, 7]

        def __init__(self):
            self.layer = nn.Module()
            self.method = hcu_linear.UnquantizedLinearMethod()
            self.method.create_weights(
                self.layer, input_size_per_partition=self.input_size,
                output_partition_sizes=self.output_partition_sizes,
                input_size=self.input_size, output_size=24,
                params_dtype=torch.bfloat16, weight_loader=lambda *a: None,
            )
            with torch.no_grad():
                self.layer.weight.fill_(1)
            expected_shape = (4, 12) if nn_layout else (12, 4)
            assert self.layer.weight.shape == expected_shape
            self.weight = SimpleNamespace(
                device=torch.device("cuda:0"), dtype=self.layer.weight.dtype,
                shape=self.layer.weight.shape,
            )

        def __call__(self, x):
            calls.append(tuple(x.shape))
            if pending_failure[0] == "projection":
                pending_failure[0] = None
                raise RuntimeError("injected projection failure")
            result = self.method.apply(self.layer, x)
            assert result.shape == (patch._PROFILE_M, 12)
            assert result.dtype == torch.bfloat16
            assert torch.count_nonzero(result) == 0
            return result, None

    class Attention:
        def __init__(self):
            self.in_proj_qkvgfab = Projection()

    patch = patch_kimi_k3_kda_gemm_warmup
    target = ModuleType(patch.TARGET_MODULE)
    target.KimiK3DeltaAttention = Attention
    target.logger = SimpleNamespace(
        info=lambda message, *args: messages.append(message % args)
    )
    monkeypatch.setattr(patch, "_warmup_done", False)
    monkeypatch.setattr(torch, "zeros", scratch_zeros)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    assert patch.apply_to_module(target)
    assert not patch.apply_to_module(target)
    if failure is not None:
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            Attention()
        assert not patch._warmup_done
    Attention()
    assert patch._warmup_done
    assert calls and all(shape == (patch._PROFILE_M, 4) for shape in calls)
    assert "packed_output=12, K=4" in messages[-2]
    count = len(calls)
    Attention()
    assert len(calls) == count


def test_residual_master_switch_restores_original_forward(monkeypatch):
    patch = patch_kimi_k3_reference_paths
    target = ModuleType(patch.TARGET_MODULE)
    class Decoder:
        def forward_attn_residual(self, *args):
            return "upstream"
    class Transform:
        pass
    class MoE:
        def __init__(self):
            pass
    class Wrapper:
        def forward(self, *args):
            pass
    target.KimiDecoderLayer = Decoder
    target.KimiRoutedOutputTransform = Transform
    target.KimiMoE = MoE
    target.KimiK3MultiHeadLatentAttentionWrapper = Wrapper
    runner_module = ModuleType("vllm_hcu.model_executor.layers.fused_moe.moe_runner")
    runner_module.MoERunner = type("Runner", (), {})
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    assert patch.apply_to_module(target)
    assert not patch.apply_to_module(target)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert Decoder().forward_attn_residual(None, None, None) == "upstream"
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    external = ModuleType("boltops.generic")
    calls = []
    def attn_res(prefix, *args):
        calls.append(True)
        return prefix
    external.attn_res = attn_res
    monkeypatch.setitem(sys.modules, "boltops.generic", external)
    decoder = Decoder()
    decoder.prev_valid_blocks = 1
    decoder.is_block_write_layer = False
    norm = nn.Identity()
    norm.weight = torch.ones(2)
    norm.variance_epsilon = 1e-5
    decoder.self_attention_res_norm = decoder.mlp_res_norm = norm
    decoder.self_attention_res_proj = decoder.mlp_res_proj = SimpleNamespace(weight=torch.ones(1, 2))
    decoder.input_layernorm = decoder.post_attention_layernorm = norm
    decoder._run_self_attn = lambda positions, hidden: hidden
    decoder.mlp = nn.Identity()
    result, _ = decoder.forward_attn_residual(None, torch.ones(1, 2), torch.ones(1, 1, 2))
    torch.testing.assert_close(result, torch.full((1, 2), 4.0))
    assert len(calls) == 2


def test_situ_master_switch_avoids_device_kernel_for_cuda_dispatch(monkeypatch):
    class CudaDispatchTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True
    x = torch.tensor([[0.5, -0.25, 0.1, 0.2]]).as_subclass(CudaDispatchTensor)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    external = ModuleType("boltops.fused_moe.triton.moe_activation")
    def forbidden(*args):
        raise AssertionError("device SiTU must not run with master switch off")
    external.triton_situ_and_mul = forbidden
    monkeypatch.setitem(sys.modules, external.__name__, external)
    actual = SituAndMul(1.25, 2.0)(x)
    gate, up = x.chunk(2, dim=-1)
    expected = 1.25 * torch.tanh(gate / 1.25) * torch.sigmoid(gate) * 2 * torch.tanh(up / 2)
    torch.testing.assert_close(actual, expected)


def test_situ_fallback_preserves_fp16_output_after_small_sigmoid(monkeypatch):
    from vllm.model_executor.layers.activation import SituAndMul as NativeSitu

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    activation = SituAndMul()
    x = torch.tensor([[-18.0, 8192.0]], dtype=torch.float16)
    expected = NativeSitu.forward_native(activation, x)
    assert expected.item() != 0.0
    torch.testing.assert_close(activation(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("beta,linear_beta", [(1.0, None), (1.25, 2.0), (0.5, 8192.0)])
@pytest.mark.parametrize("strided", [False, True])
def test_situ_fallback_matches_native_dynamic_range(
    monkeypatch, dtype, beta, linear_beta, strided
):
    from vllm.model_executor.layers.activation import SituAndMul as NativeSitu

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    values = torch.tensor(
        [
            [-18.0, 8192.0],
            [-20.0, 65504.0],
            [-8.0, 6.0],
            [8.0, 2048.0],
            [0.0, 1.0],
            [1e-4, 1e-4],
            [-1e-4, -1e-4],
        ],
        dtype=dtype,
    )
    if strided:
        storage = torch.empty((values.shape[0], 4), dtype=dtype)
        storage[:, ::2] = values
        values = storage[:, ::2]
        assert not values.is_contiguous()
    activation = SituAndMul(beta, linear_beta)
    expected = NativeSitu.forward_native(activation, values)
    actual = activation(values)
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
