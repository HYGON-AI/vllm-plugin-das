# SPDX-License-Identifier: Apache-2.0
"""Load and replay each declared/retained dense and PLE storage combination."""

import pytest

from tests.vllm033.test_slimquant import (
    checkpoint_config,
    native,
    native_parallel,  # noqa: F401 -- imported pytest fixture
)


@pytest.mark.hcu
@native
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("ngram", [False, True])
def test_optional_groups_load_storage_and_changed_input_graph(
    native_parallel,  # noqa: F811 -- pytest injects the imported fixture
    dense,
    ngram,
):
    import torch
    from torch.nn import functional as F

    from hcu_vllm_das.slimquant import SlimQuantConfig
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLEDeviceEmbedding,
        Qwen4ExpPLEEmbeddingMethod,
    )

    raw = checkpoint_config()
    if not dense:
        raw["mixed_precision"].pop("w8a8_include")
    if not ngram:
        raw["mixed_precision"].pop("ngram_embedding")
    config = SlimQuantConfig.from_config(raw)
    prefix = "model.layers.0.ple.ple_embedding.ngram_embedding"
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(config, prefix)
    with torch.device("cuda:0"):
        linear = ReplicatedLinear(
            256,
            128,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="model.layers.0.self_attn.q_proj",
        )
        table = Qwen4ExpPLEDeviceEmbedding(
            16,
            3,
            params_dtype=torch.bfloat16,
            padding_size=8,
            prefix=prefix,
            embedding_method=method,
            num_ngram_heads=2,
            max_total_tokens=4,
        )
    weight = (torch.arange(128 * 256).reshape(128, 256) % 15 - 7).to(
        torch.int8 if dense else torch.bfloat16
    )
    if not dense:
        weight /= 128
    linear.weight.weight_loader(linear.weight, weight)
    if dense:
        scales = (torch.arange(128).reshape(128, 1) % 4 + 1).float() / 256
        linear.weight_scale.weight_loader(linear.weight_scale, scales)
    else:
        assert not hasattr(linear, "weight_scale")
    linear.quant_method.process_weights_after_loading(linear)

    table_weight = (torch.arange(48).reshape(16, 3) % 17 - 8).to(
        torch.int8 if ngram else torch.bfloat16
    )
    table.weight.weight_loader(table.weight, table_weight, checkpoint_start=0)
    if ngram:
        table_scales = (torch.arange(16).reshape(16, 1).float() + 1) / 128
        table.weight_scale.weight_loader(
            table.weight_scale, table_scales, checkpoint_start=0
        )
    else:
        assert not hasattr(table, "weight_scale")
    method.process_weights_after_loading(table)
    assert linear.weight.dtype == (torch.int8 if dense else torch.bfloat16)
    assert table.weight.dtype == (torch.int8 if ngram else torch.bfloat16)

    x = (
        torch.linspace(-0.5, 0.5, 256, device="cuda:0")
        .to(torch.bfloat16)[None]
        .repeat(3, 1)
    )
    ids = torch.tensor([[0, 15], [1, 2]], device="cuda:0")
    linear(x)
    table(ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = linear(x)
        embeddings = table(ids)
    addresses = (output.data_ptr(), embeddings.data_ptr())
    for shift, values in ((0.0, [[0, 15], [1, 2]]), (0.25, [[3, 4], [15, 0]])):
        x.copy_(torch.linspace(-0.5, 0.5, 256, device="cuda:0") + shift)
        ids.copy_(torch.tensor(values, device="cuda:0"))
        graph.replay()
        inputs = x.cpu().double()
        if dense:
            input_scales = inputs.abs().amax(dim=-1, keepdim=True) / 127
            inputs = (inputs / input_scales).round().clamp(-127, 127) * input_scales
            reference = inputs @ (weight.double() * scales.double()).T
        else:
            reference = inputs @ weight.double().T
        torch.testing.assert_close(
            output.cpu().double(), reference, rtol=0.01, atol=0.002
        )
        reference_table = table_weight.float()
        if ngram:
            reference_table *= table_scales
        expected = F.embedding(torch.tensor(values), reference_table).bfloat16()
        torch.testing.assert_close(embeddings.cpu(), expected, rtol=0, atol=0)
        assert (output.data_ptr(), embeddings.data_ptr()) == addresses
