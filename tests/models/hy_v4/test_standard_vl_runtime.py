# SPDX-License-Identifier: Apache-2.0
"""Stock-Core CPU contracts and explicitly enabled native HCU vision tests."""

import asyncio
import json
import os
from copy import copy
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def _use_v2_model_runner(monkeypatch):
    # Native DFly uses Core's actual V2 runner admission, as the service does.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def _cpu_config(
    tmp_path,
    *,
    use_2d_rope=False,
    parallel=None,
    dtype="float32",
    with_dfly=False,
    device="cpu",
):
    from hcu_adapter.extensions import ExtensionContext
    from hcu_vllm_das.hy4_vl_activation import activate
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        VllmConfig,
    )

    activate(ExtensionContext(adapter_version="test", vllm_version="0.33.0"))
    HYV4VLConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        indexer_types=["full"] * 6,
        dfly_num_hidden_layers=5,
        dfly_block_size=7,
        dfly_target_kv_layers=[1, 2, 3, 4, 5],
        dfly_target_index_layer=[1, 2, 3, 4, 5],
        dfly_mask_token_id=4,
        image_start_token_id=1,
        image_end_token_id=2,
        image_token_id=3,
        image_newline_token_id=4,
        video_start_token_id=5,
        video_end_token_id=6,
        video_token_id=7,
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_image_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "perceive_intermediate_size": 64,
            "cat_extra_token": 1,
            "use_2d_rope": use_2d_rope,
        },
    ).save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype=dtype,
        max_model_len=128,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_encoder_attn_backend="TORCH_SDPA",
    )
    parallel = parallel or ParallelConfig()
    speculative = None
    if with_dfly:
        from hcu_vllm_das.models.hy_v4_flash.dfly_config import HCUDFlySpeculativeConfig

        speculative = HCUDFlySpeculativeConfig(
            target_model_config=model,
            target_parallel_config=parallel,
            method="dfly",
        )
    return VllmConfig(
        model_config=model,
        device_config=DeviceConfig(device) if device is not None else DeviceConfig(),
        parallel_config=parallel,
        speculative_config=speculative,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=["none" if device == "cpu" else "all"],
        ),
    )


@pytest.mark.parametrize("use_2d_rope", [False, True])
@pytest.mark.skipif(
    os.environ.get("HCU_HY4_VL_TINY_VISION") != "1",
    reason="native HCU vision requires HCU_HY4_VL_TINY_VISION=1 and a visible GPU",
)
def test_tiny_real_vision_tower_hcu_sdpa(tmp_path, use_2d_rope):
    """Exercise the real linear/RMSNorm/MMEncoderAttention/rotary HCU path."""
    from hcu_vllm_das.models.hy4_standard.vision import HYV4VisionTransformer
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert torch.cuda.is_available()
    assert torch.version.hip is not None
    config = _cpu_config(tmp_path, use_2d_rope=use_2d_rope, device=None)
    assert config.device_config.device_type == "cuda"
    assert not torch.distributed.is_initialized()
    assert not model_parallel_is_initialized()
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        with set_current_vllm_config(config), torch.device(device):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                backend="gloo",
                distributed_init_method=(tmp_path / "gloo-init").as_uri(),
            )
            initialize_model_parallel(backend="gloo")
            torch.manual_seed(19)
            tower = HYV4VisionTransformer(config.model_config.hf_config.vision_config)
            for parameter in tower.parameters():
                torch.nn.init.uniform_(parameter, -0.05, 0.05)
            tower.eval()
            patches = torch.randn(16, 3 * 16 * 16)
            with torch.inference_mode():
                output = tower(patches, [[1, 4, 4]])
                repeated = tower(patches.clone(), [[1, 4, 4]])
                changed = tower(patches + 0.2, [[1, 4, 4]])
            assert tower.attn_backend == AttentionBackendEnum.TORCH_SDPA
            assert len(output) == 1
            assert output[0].shape == (8, 256)
            assert output[0].device == device
            assert torch.isfinite(output[0]).all()
            # HIP FP32 GEMM/SDPA reductions need not repeat bit-for-bit.
            torch.testing.assert_close(output[0], repeated[0], rtol=1e-5, atol=1e-7)
            assert not torch.allclose(output[0], changed[0])
            with pytest.raises(ValueError, match="length"):
                tower(patches[:-1], [[1, 4, 4]])
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("backend", ["deepep_high_throughput", "deepep_low_latency"])
@pytest.mark.parametrize("with_dfly", [False, True])
def test_vl_runtime_admits_real_tp1_dp16_ep16_config(tmp_path, backend, with_dfly):
    from hcu_adapter.extensions import ExtensionContext
    from hcu_vllm_das.hy4_ct_model_activation import activate
    from hcu_vllm_das.models.hy4_standard.vl_model import validate_vl_runtime
    from vllm.config import ParallelConfig

    activate(
        ExtensionContext(
            adapter_version="test",
            vllm_version="0.33.0",
            requested_capabilities=frozenset({"hy4-ct-w4a8"}),
        )
    )
    if with_dfly:
        from hcu_vllm_das.hy4_dfly_activation import activate as activate_dfly

        activate_dfly(
            ExtensionContext(
                adapter_version="test",
                vllm_version="0.33.0",
                requested_capabilities=frozenset({"hy4-dfly"}),
            )
        )
    parallel = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=16,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        enable_expert_parallel=True,
        all2all_backend=backend,
    )
    config = _cpu_config(
        tmp_path, parallel=parallel, dtype="bfloat16", with_dfly=with_dfly
    )
    validate_vl_runtime(config)
    if with_dfly:
        assert config.speculative_config.num_speculative_tokens == 6
        assert (
            config.speculative_config.draft_model_config.hf_config.num_hidden_layers
            == 5
        )
    for changes in (
        {"data_parallel_size": 1},
        {"tensor_parallel_size": 2},
        {"enable_expert_parallel": False},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"enable_dbo": True},
        {"all2all_backend": "allgather_reducescatter"},
    ):
        # Real dataclass objects remain real; use public field changes to test
        # the guard independently of earlier Core configuration validation.
        original = config.parallel_config
        config.parallel_config = copy(original)
        for field, value in changes.items():
            setattr(config.parallel_config, field, value)
        try:
            with pytest.raises(ValueError):
                validate_vl_runtime(config)
        finally:
            config.parallel_config = original


def test_owned_renderer_defaults_are_scoped_and_params_are_immutable(tmp_path):
    from hcu_vllm_das.models.hy4_standard.processor import HYV4VLImageProcessor
    from hcu_vllm_das.models.hy4_standard.renderer import HY4VLHfRenderer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from vllm.renderers.params import ChatParams
    from vllm.tokenizers.hf import CachedHfTokenizer

    config = _cpu_config(tmp_path)
    special = [
        "<unk>",
        "<image_start>",
        "<image_end>",
        "<image>",
        "<image_newline>",
        "<video_start>",
        "<video_end>",
        "<video>",
    ]
    backend = Tokenizer(
        WordLevel(
            {word: index for index, word in enumerate([*special, "hello"])},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=special[1:],
        chat_template="{{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(tmp_path)
    HYV4VLImageProcessor().save_pretrained(tmp_path)
    renderer = HY4VLHfRenderer(config, CachedHfTokenizer.from_pretrained(tmp_path))
    try:
        defaults = ChatParams()
        assert renderer._hy4_params(defaults).chat_template_content_format == "string"
        assert defaults.chat_template_content_format == "auto"
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        conversation, prompt = renderer.render_messages(messages, defaults)
        assert conversation[0]["content"] == "hello"
        assert asyncio.run(renderer.render_messages_async(messages, defaults)) == (
            conversation,
            prompt,
        )
        for params in (
            ChatParams(chat_template="{{ messages }}"),
            ChatParams(chat_template_content_format="openai"),
            ChatParams(chat_template_content_format="string"),
        ):
            assert renderer._hy4_params(params) is params
        original = config.model_config.hf_config
        config.model_config.hf_config = original.text_config
        assert renderer._hy4_params(defaults) is defaults
        config.model_config.hf_config = original
    finally:
        renderer.shutdown()


def test_checkpoint_visual_ignore_survives_real_core_ct_mapper():
    """The actual checkpoint recipe excludes BF16 vision before CT schemes."""
    from hcu_vllm_das.models.hy4_standard.vl_model import HYV4VLForConditionalGeneration
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    recipe = json.loads(
        (Path(__file__).parent / "fixtures" / "hy4_vl_quantization.json").read_text(
            encoding="utf-8"
        )
    )
    assert r"re:.*vit\..*" in recipe["ignore"]
    assert r"re:.*visual\..*" in recipe["ignore"]
    parsed = CompressedTensorsConfig.from_config(recipe)
    parsed.packed_modules_mapping = (
        HYV4VLForConditionalGeneration.packed_modules_mapping
    )
    mapper = HYV4VLForConditionalGeneration.hf_to_vllm_mapper
    parsed.apply_vllm_mapper(mapper)
    for checkpoint_name in (
        "vit.vit.layers.0.self_attn.q_proj",
        "vit.vit.layers.0.self_attn.k_proj",
        "vit.vit.layers.0.self_attn.v_proj",
        "vit.vit.layers.0.self_attn.o_proj",
        "vit.vit.layers.0.mlp.fc1",
    ):
        prefix = mapper._map_name(checkpoint_name)
        assert prefix.startswith("visual.")
        assert parsed.get_scheme_dict(torch.nn.Linear(32, 32), prefix) is None
    assert (
        parsed.get_scheme_dict(torch.nn.Linear(32, 32), "visual.layers.0.self_attn.qkv")
        is None
    )
