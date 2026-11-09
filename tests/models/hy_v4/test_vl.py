# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.models.hy_v4.vl import (
    HYV4PatchProjection,
    HYV4VisionBlock,
    HYV4VisionMerger,
    HYV4VisionPatchEmbed,
    HYV4VLForConditionalGeneration,
    HYV4VLProcessor,
)
from vllm_hcu.models.hy_v4.vl_config import HYV4VLConfig


@pytest.mark.parametrize(
    "case,answer,expected",
    [
        ("red", "红色", True),
        ("red", "蓝色", False),
        ("blue", "blue", True),
        ("count", "3", True),
        ("count", "2", False),
        ("ocr", "HY4 7285", True),
        ("ocr", "HY4 7286", False),
        ("multi", "第一张红色，第二张蓝色", True),
        ("multi", "第一张蓝色，第二张红色", False),
        ("red_repeat", "", False),
    ],
)
def test_image_smoke_checks_answers(case, answer, expected):
    from tools.hyv4_vl_image_smoke import check_answer

    assert check_answer(case, answer) == expected


def test_flat_config_round_trip():
    config = HYV4VLConfig(
        hidden_size=16, vision_config={"hidden_size": 8, "out_hidden_size": 16}
    )
    copy = HYV4VLConfig.from_dict(config.to_dict())
    assert copy.model_type == "hy_v4_vl"
    assert copy.hidden_size == 16
    assert copy.vision_config.hidden_size == 8
    assert copy.kv_lora_rank == 512


@pytest.mark.parametrize("layers", [1, 78])
@pytest.mark.parametrize("null_lists", [False, True])
def test_language_defaults_follow_final_layer_count(layers, null_lists):
    from vllm_hcu.models.hy_v4.attention import compute_skip_topk_layers
    from vllm_hcu.models.hy_v4.config import HYV4Config

    kwargs = {"num_hidden_layers": layers}
    if null_lists:
        kwargs.update(mlp_layer_types=None, indexer_types=None)
    config = HYV4VLConfig(**kwargs)
    reference = HYV4Config(**kwargs)
    assert config.mlp_layer_types == reference.mlp_layer_types
    assert config.indexer_types == reference.indexer_types
    assert len(config.mlp_layer_types) == len(config.indexer_types) == layers
    assert compute_skip_topk_layers(config) == compute_skip_topk_layers(reference)
    copy = HYV4VLConfig.from_dict(config.to_dict())
    assert copy.mlp_layer_types == config.mlp_layer_types
    assert copy.indexer_types == config.indexer_types


def test_explicit_language_lists_and_rotary_dimensions_are_preserved():
    config = HYV4VLConfig(
        num_hidden_layers=2,
        mlp_layer_types=["sparse", "dense"],
        indexer_types=["full", "full"],
        qk_nope_head_dim=80,
        qk_rope_head_dim=32,
    )
    assert config.mlp_layer_types == ["sparse", "dense"]
    assert config.indexer_types == ["full", "full"]
    assert config.qk_head_dim == 112
    assert config.head_dim == 32


def test_encoder_only_is_rejected_before_backbone_allocation(monkeypatch):
    from vllm_hcu.models.hy_v4.model import HYV4ForCausalLM

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("encoder-only must be rejected before allocating the backbone")

    monkeypatch.setattr(HYV4ForCausalLM, "__init__", unexpected_allocation)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(mm_encoder_only=True)
        )
    )
    with pytest.raises(ValueError, match="encoder-only"):
        HYV4VLForConditionalGeneration(vllm_config=config)


@pytest.mark.parametrize("tower_lora", [True, False, None])
def test_tower_lora_is_rejected_but_language_lora_reaches_backbone(
    monkeypatch, tower_lora
):
    from vllm_hcu.models.hy_v4.model import HYV4ForCausalLM

    class BackboneReached(Exception):
        pass

    def backbone(*args, **kwargs):
        raise BackboneReached

    monkeypatch.setattr(HYV4ForCausalLM, "__init__", backbone)
    config = SimpleNamespace(
        lora_config=None
        if tower_lora is None
        else SimpleNamespace(enable_tower_connector_lora=tower_lora),
        speculative_config=None,
        model_config=SimpleNamespace(
            multimodal_config=None,
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(projector_type="rmsnorm_mlp_adapter")
            ),
        ),
    )
    if tower_lora:
        with pytest.raises(ValueError, match="vision tower/connector LoRA"):
            HYV4VLForConditionalGeneration(vllm_config=config)
    else:
        with pytest.raises(BackboneReached):
            HYV4VLForConditionalGeneration(vllm_config=config)


@pytest.mark.parametrize("samples", [1, 4, 7])
def test_perf_rejects_small_samples_before_io_or_warmup(
    monkeypatch, tmp_path, capsys, samples
):
    from tools import hyv4_vl_perf

    monkeypatch.setattr(
        "sys.argv",
        [
            "hyv4_vl_perf",
            "--data-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "output"),
            "--samples",
            str(samples),
        ],
    )

    def unexpected(*args, **kwargs):
        pytest.fail("invalid sample count must fail before data loading or HTTP")

    monkeypatch.setattr(hyv4_vl_perf.pq, "read_table", unexpected)
    monkeypatch.setattr(hyv4_vl_perf, "request", unexpected)
    with pytest.raises(SystemExit) as error:
        hyv4_vl_perf.main()
    assert error.value.code == 2
    assert "--samples must be at least 8" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_perf_eight_samples_completes_all_suites(monkeypatch, tmp_path):
    import io
    import json

    from PIL import Image

    from tools import hyv4_vl_perf

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    rows = [
        {"image": {"bytes": buffer.getvalue()}, "question": "Q?", "answer": "A"}
    ] * 8
    (tmp_path / "test-0.parquet").touch()
    monkeypatch.setattr(
        hyv4_vl_perf.pq, "read_table", lambda p: SimpleNamespace(to_pylist=lambda: rows)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "hyv4_vl_perf",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--samples",
            "8",
            "--rounds",
            "1",
        ],
    )
    calls = []

    def request(body, url):
        calls.append(body)
        return {
            "ttft_seconds": 1.0,
            "latency_seconds": 2.0,
            "content_chunk_interval_seconds": None,
            "usage": {"completion_tokens": 2},
        }

    monkeypatch.setattr(hyv4_vl_perf, "request", request)
    hyv4_vl_perf.main()
    expected = {"qa_c1": 8, "qa_c2": 8, "caption32_c1": 4, "two_images32_c1": 4}
    for suite, count in expected.items():
        results = json.loads((tmp_path / "output" / f"{suite}.json").read_text())
        assert len(results) == count
    assert len(calls) == 26  # two warmups plus all measured requests
    assert sum(len(b["messages"][0]["content"]) == 3 for b in calls) == 4


@pytest.mark.parametrize(
    "deltas,completion_tokens,expected_interval",
    [
        (["A"], 2, None),
        (["multiple tokens in one delta"], 7, None),
        (["a", "b", "c"], 9, 1.0),
    ],
)
def test_streaming_metric_measures_content_chunks(
    monkeypatch, deltas, completion_tokens, expected_interval
):
    import json
    from unittest.mock import MagicMock

    from tools import hyv4_vl_perf

    chunks = [{"choices": [{"delta": {"content": ""}}]}]
    chunks.extend({"choices": [{"delta": {"content": s}}]} for s in deltas)
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    chunks.append({"choices": [], "usage": {"completion_tokens": completion_tokens}})
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = [
        b"data: " + json.dumps(chunk).encode() for chunk in chunks
    ] + [b"data: [DONE]"]
    monkeypatch.setattr(hyv4_vl_perf.requests, "post", lambda *a, **k: response)
    ticks = iter(range(10))
    monkeypatch.setattr(hyv4_vl_perf.time, "perf_counter", lambda: next(ticks))
    result = hyv4_vl_perf.request({}, "http://example.invalid")
    assert result["content_chunks"] == len(deltas)
    assert result["content_chunk_interval_seconds"] == expected_interval
    assert result["ttft_seconds"] == 1
    assert result["latency_seconds"] == len(deltas) + 1
    assert result["answer"] == "".join(deltas)
    assert result["usage"]["completion_tokens"] == completion_tokens
    assert "decode_seconds_per_token" not in result


def test_text_only_cached_prompt_preserves_image_placeholders():
    def tokenize(texts, **kwargs):
        assert texts == ["before<image>after"]
        return {"input_ids": [[1, 120120, 2]]}

    processor = SimpleNamespace(image_token="<image>", tokenizer=tokenize)
    result = HYV4VLProcessor.__call__(processor, text="before<image>after")
    assert result["input_ids"] == [[1, 120120, 2]]


def test_positions_two_rectangular_images_include_row_separators():
    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))

    def feature(offset, h, w):
        return SimpleNamespace(
            mm_position=SimpleNamespace(offset=offset),
            data={"image_grid_thw": SimpleNamespace(data=torch.tensor([1, h, w]))},
        )

    positions, delta = model.get_mrope_input_positions(
        list(range(16)), [feature(2, 4, 4), feature(10, 2, 4)]
    )
    assert positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 10, 11, 12],
        [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 7, 7, 10, 11, 12],
        [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    assert delta == -3


def test_no_image_positions_are_sequential():
    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    positions, delta = model.get_mrope_input_positions([1, 2, 3], [])
    assert torch.equal(positions, torch.arange(3).repeat(3, 1))
    assert delta == 0


def test_merger_inserts_checkpoint_newline_at_each_row():
    config = SimpleNamespace(
        hidden_size=2, out_hidden_size=8, rms_norm_eps=1e-5, spatial_merge_size=2
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        merger = HYV4VisionMerger(config)
        merger.norm._forward_method = merger.norm.forward_native
    with torch.no_grad():
        merger.image_newline.copy_(torch.arange(8))
    output = merger(torch.randn(1, 24, 2), size=(4, 6))
    assert output.shape == (1, 8, 8)
    torch.testing.assert_close(output[0, 3], torch.arange(8, dtype=output.dtype))
    torch.testing.assert_close(output[0, 7], torch.arange(8, dtype=output.dtype))


@pytest.mark.parametrize("bias", [False, True])
def test_patch_projection_matches_checkpoint_convolution(bias):
    projection = HYV4PatchProjection(3, 16, 4, stride=4, bias=bias)
    patches = torch.randn(11, 3, 4, 4)
    reference = torch.nn.functional.conv2d(
        patches, projection.weight, projection.bias, stride=4
    )
    torch.testing.assert_close(projection(patches), reference)
    assert set(projection.state_dict()) == ({"weight", "bias"} if bias else {"weight"})


@pytest.mark.hcu
@pytest.mark.parametrize("patches", [384, 1024, 2048, 3900, 7788, 8192, 8196])
@pytest.mark.parametrize("seed", [0, 338])
def test_patch_projection_preserves_hcu_bf16_rounding(patches, seed):
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    with torch.device("cuda"), torch.inference_mode():
        torch.manual_seed(seed)
        projection = HYV4PatchProjection(3, 1536, 16, stride=16).bfloat16()
        inputs = torch.randn(patches, 3, 16, 16, dtype=torch.bfloat16)
        reference = torch.nn.functional.conv2d(
            inputs, projection.weight, projection.bias, stride=16
        )
        torch.testing.assert_close(projection(inputs), reference, rtol=0, atol=0)


def test_packed_vision_attention_preserves_image_boundaries(monkeypatch):
    from types import MethodType

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    from vllm.model_executor.models.hunyuan_vision import (
        HunYuanVisionAttention,
        HunYuanVisionBlock,
    )
    from vllm.v1.attention.ops.vit_attn_wrappers import torch_sdpa_wrapper

    # This HCU build registers the SDPA custom op only for CUDA. Exercise its
    # real implementation directly so this boundary test remains CPU-only.
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mm_encoder_attention.vit_torch_sdpa_wrapper",
        torch_sdpa_wrapper,
    )

    class TupleLinear(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    block = HYV4VisionBlock.__new__(HYV4VisionBlock)
    torch.nn.Module.__init__(block)
    block.input_layernorm = torch.nn.LayerNorm(8)
    block.post_attention_layernorm = torch.nn.LayerNorm(8)
    block.mlp = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8)
    )
    block.self_attn = torch.nn.Module()
    block.self_attn.qkv = TupleLinear(8, 24)
    block.self_attn.o_proj = TupleLinear(8, 8)
    with set_current_vllm_config(VllmConfig()):
        block.self_attn.attn = MMEncoderAttention(2, 4, 0.5)
        block.self_attn.attn._forward_method = block.self_attn.attn.forward_native
    block.self_attn.forward = MethodType(
        HunYuanVisionAttention.forward, block.self_attn
    )
    images = [torch.randn(1, length, 8) for length in (6, 10, 4)]
    reference = torch.cat([HunYuanVisionBlock.forward(block, x) for x in images], dim=1)
    actual = block(
        torch.cat(images, dim=1),
        torch.tensor([0, 6, 16, 20], dtype=torch.int32),
        torch.tensor(10),
    )
    torch.testing.assert_close(actual, reference)
    # Altering the middle image must not affect either neighbouring image.
    images[1] = images[1] * 10 + 5
    changed = block(
        torch.cat(images, dim=1),
        torch.tensor([0, 6, 16, 20], dtype=torch.int32),
        torch.tensor(10),
    )
    torch.testing.assert_close(changed[:, :6], actual[:, :6])
    torch.testing.assert_close(changed[:, 16:], actual[:, 16:])


def test_multiaxis_rope_matches_interleaved_pair_rotation():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    with set_current_vllm_config(VllmConfig()):
        rope = MRotaryEmbedding(
            64,
            64,
            32,
            10000,
            False,
            torch.float32,
            mrope_section=[12, 10, 10],
            mrope_interleaved=True,
        )
    positions = torch.tensor([[1, 2], [3, 4], [5, 6]])
    query = torch.randn(2, 2, 64)
    key = torch.randn(2, 1, 64)
    actual_q, actual_k = rope.forward_native(positions, query, key)
    axes = torch.zeros(32, dtype=torch.long)
    axes[1:30:3] = 1
    axes[2:30:3] = 2
    frequencies = 10000 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    angles = positions[axes].T * frequencies

    def rotate(x):
        pairs = x.reshape(2, -1, 32, 2)
        even, odd = pairs.unbind(-1)
        c, s = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        return torch.stack([even * c - odd * s, odd * c + even * s], -1).reshape_as(x)

    torch.testing.assert_close(actual_q, rotate(query))
    torch.testing.assert_close(actual_k, rotate(key))


def _position_reload_fixture(dtype, device="cpu"):
    config = SimpleNamespace(
        hidden_size=4,
        patch_size=2,
        num_channels=1,
        spatial_merge_size=2,
        interpolate_mode="bicubic",
        max_image_size=4,
    )
    embed = HYV4VisionPatchEmbed(config).to(device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in embed.parameters():
            parameter.zero_()
        embed.position_embedding.weight.fill_(1)
    return embed, torch.zeros(4, 4, device=device, dtype=dtype), [[1, 2, 2]]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("update", ["copy", "replace", "kernel_reload"])
def test_vision_position_cache_follows_weight_updates(dtype, update):
    embed, pixels, grids = _position_reload_fixture(dtype)
    with torch.inference_mode():
        before = embed(pixels, grids)
        torch.testing.assert_close(
            before, torch.ones_like(before), rtol=1e-6, atol=1e-6
        )
        # Existing consumers, including captured work, retain this storage.
        retained_positions = embed.patch_pos_embed
        if update == "copy":
            embed.position_embedding.weight.fill_(7)
        elif update == "replace":
            embed.position_embedding.weight = torch.nn.Parameter(
                torch.full_like(embed.position_embedding.weight, 7),
                requires_grad=False,
            )
        else:
            from vllm_hcu.v1.hcu_model_runner import GPUModelRunner

            model = torch.nn.Module()
            model.embeddings = embed
            runner = SimpleNamespace(
                get_model=lambda: model,
                model_config=SimpleNamespace(quantization=None),
            )
            weights = [
                (
                    name,
                    torch.full_like(parameter, 7)
                    if name.endswith("position_embedding.weight")
                    else parameter.clone(),
                )
                for name, parameter in model.named_parameters()
            ]
            # This actual runner path bypasses model.load_weights entirely.
            GPUModelRunner.reload_weights(
                runner, weights_iterator=weights, is_checkpoint_format=False
            )
        after = embed(pixels, grids)
        torch.testing.assert_close(
            after, torch.full_like(after, 7), rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            retained_positions, torch.full_like(retained_positions, 7), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_vision_position_refresh_preserves_unchanged_weight_outputs(dtype):
    embed, pixels, grids = _position_reload_fixture(dtype)
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(338)
        for parameter in embed.parameters():
            parameter.normal_(generator=generator)
        pixels.copy_(torch.randn(pixels.shape, dtype=dtype, generator=generator))
        reference = embed(pixels, grids).clone()
        for _ in range(3):
            torch.testing.assert_close(embed(pixels, grids), reference, rtol=0, atol=0)


@pytest.mark.hcu
def test_captured_vision_positions_follow_updated_weights():
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    embed, pixels, grids = _position_reload_fixture(torch.bfloat16, "cuda")
    with torch.inference_mode():
        for _ in range(3):
            embed(pixels, grids)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = embed(pixels, grids)
        embed.position_embedding.weight.fill_(7)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 7), rtol=0, atol=0)
