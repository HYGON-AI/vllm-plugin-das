# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exercise the v0.25.1 entry points used by K3 drafting and serving."""

import copy
import dataclasses
from types import SimpleNamespace
from typing import get_args

import pytest
import torch
from torch import nn

from vllm_hcu.patch.platform.core_fix import patch_kimi_k3_mtp_config
from vllm_hcu.transformers_utils.configs.kimi_k3 import KimiK3Config, KimiK3VisionConfig


@pytest.mark.parametrize("count", [1, 2])
def test_kimi_mtp_override_reads_nested_config_and_preserves_target(count):
    import vllm.config.speculative as module
    import vllm.transformers_utils.model_arch_config_convertor as arch_module
    from vllm_hcu.patch.kimi_k3_callbacks import apply_kimi_k3_mtp_arch_config

    patch_kimi_k3_mtp_config.apply_to_module(module)
    config = KimiK3Config(
        text_config={"num_nextn_predict_layers": count, "kv_lora_rank": 512},
        architectures=["KimiK3ForConditionalGeneration"],
    )
    config.num_nextn_predict_layers = 99
    before = copy.deepcopy(config.to_dict())
    draft = module.SpeculativeConfig.hf_config_override(copy.deepcopy(config))
    assert draft.model_type == "kimi_k3_mtp"
    assert draft.architectures == ["KimiK3MTPModel"]
    assert draft.n_predict == count
    assert draft.text_config.num_nextn_predict_layers == count
    assert module.SpeculativeConfig.hf_config_override(draft) is draft
    assert config.to_dict() == before
    assert "kimi_k3_mtp" in get_args(module.MTPModelTypes)
    apply_kimi_k3_mtp_arch_config(arch_module)
    convertor = arch_module.MODEL_ARCH_CONFIG_CONVERTORS[draft.model_type](
        draft, draft.get_text_config())
    assert convertor.get_num_hidden_layers() == count
    assert convertor.get_num_hidden_layers() != draft.text_config.num_hidden_layers
    assert convertor.is_deepseek_mla()


@pytest.mark.parametrize("count", [0, -1, None])
def test_kimi_mtp_override_rejects_missing_checkpoint_draft_layers(count):
    import vllm.config.speculative as module

    patch_kimi_k3_mtp_config.apply_to_module(module)
    config = KimiK3Config(text_config={"num_nextn_predict_layers": count})
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        module.SpeculativeConfig.hf_config_override(config)


def test_standard_mtp_initialization_detects_kimi_draft(monkeypatch):
    import vllm.config.speculative as module
    from pydantic.fields import FieldInfo

    patch_kimi_k3_mtp_config.apply_to_module(module)
    target_hf = KimiK3Config(text_config={"num_nextn_predict_layers": 1},
                            architectures=["KimiK3ForConditionalGeneration"])
    target = SimpleNamespace(
        model="local/kimi", tokenizer="local/kimi", tokenizer_mode="auto",
        quantization=None, trust_remote_code=False, allowed_local_media_path="",
        allowed_media_domains=None, dtype=torch.bfloat16, seed=0,
        tokenizer_revision=None, max_model_len=128, enforce_eager=True,
        max_logprobs=20, hf_overrides={}, config_format="auto",
        hf_text_config=target_hf.text_config,
    )
    parallel = SimpleNamespace(tensor_parallel_size=1)

    def load_draft_config(**kwargs):
        # Replace checkpoint I/O only; run the real override and method detection.
        hf = kwargs["hf_overrides"](copy.deepcopy(target_hf))
        return SimpleNamespace(model=kwargs["model"], hf_config=hf,
                               architectures=hf.architectures, max_model_len=128)

    monkeypatch.setattr(module, "ModelConfig", load_draft_config)
    monkeypatch.setattr(module.SpeculativeConfig, "create_draft_parallel_config",
                        staticmethod(lambda *args: parallel))
    spec = object.__new__(module.SpeculativeConfig)
    for field in dataclasses.fields(module.SpeculativeConfig):
        if field.default is not dataclasses.MISSING:
            default = field.default
            if isinstance(default, FieldInfo):
                default = default.get_default(call_default_factory=True)
            setattr(spec, field.name, copy.deepcopy(default))
        elif field.default_factory is not dataclasses.MISSING:
            setattr(spec, field.name, field.default_factory())
    spec.target_model_config = target
    spec.target_parallel_config = parallel
    spec.method = "mtp"
    spec.num_speculative_tokens = 1
    spec.__post_init__()
    assert spec.method == "mtp"
    assert spec.draft_model_config.hf_config.architectures == ["KimiK3MTPModel"]
    assert spec.draft_model_config.hf_config.n_predict == 1
    assert target_hf.model_type == "kimi_k3"


@pytest.mark.parametrize("method,arch,expected", [
    ("mtp", "KimiK3MTPModel", True),
    ("mtp", "DeepSeekMTPModel", True),
    ("mtp", "Qwen3_5MTP", False),
    ("draft_model", "KimiK3MTPModel", False),
])
def test_real_proposer_tuple_detection_preserves_other_methods(method, arch, expected):
    import vllm.v1.spec_decode.llm_base_proposer as module
    from vllm_hcu.patch.worker.framework_opt import patch_llm_base_proposer

    patch_llm_base_proposer.apply_to_module(module)
    proposer = object.__new__(module.SpecDecodeBaseProposer)
    proposer.method = method
    proposer.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
    assert proposer.model_returns_tuple() is expected


class _Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(map(ord, text))


def _stream_parser(thinking):
    import vllm.parser.abstract_parser as module
    from vllm_hcu.patch.kimi_k3_callbacks import apply_kimi_k3_streaming_content
    from vllm_hcu.runtime_compat.kimi_k3_reasoning_parser import KimiK3ReasoningParser

    apply_kimi_k3_streaming_content(module)

    class K3Parser(module.DelegatingParser):
        reasoning_parser_cls = KimiK3ReasoningParser

    return K3Parser(_Tokenizer(), chat_template_kwargs={"enable_thinking": thinking})


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("width", [1, 2, 7, 1000])
@pytest.mark.parametrize("spaced", [False, True])
def test_real_delegating_parser_strips_split_xtml(thinking, width, spaced):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    parser = _stream_parser(thinking)
    request = ChatCompletionRequest(model="kimi", messages=[], tool_choice="none")
    body = "<|open|>response<|sep|>answer<|close|>response<|sep|><|close|>message<|sep|>"
    if spaced:
        body = body.replace("response", " response ").replace("message", " message ")
    text = ("<|open|>think<|sep|>reason<|close|>think<|sep|>" if thinking else "") + body
    chunks = [text[i:i + width] for i in range(0, len(text), width)]
    contents, reasons = [], []
    for index, chunk in enumerate(chunks):
        delta = parser.parse_delta(chunk, list(map(ord, chunk)), request,
                                   prompt_token_ids=[], finished=index == len(chunks) - 1)
        if delta:
            contents.append(delta.content or "")
            reasons.append(delta.reasoning or "")
    expected_reason, expected_content = parser._reasoning_parser.extract_reasoning(text, request)
    assert "".join(contents) == expected_content == "answer"
    assert "".join(reasons) == (expected_reason or "")


@pytest.mark.parametrize("width", [1, 2, 7, 1000])
@pytest.mark.parametrize("think_open", ["", "<|open|>think<|sep|>"])
def test_disabled_thinking_stream_discards_residual_analysis(width, think_open):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    parser = _stream_parser(False)
    request = ChatCompletionRequest(model="kimi", messages=[], tool_choice="none")
    prefix = think_open + "analysis that must not reach the caller<|close|>think<|sep|>"
    body = "<|open|>response<|sep|>answer<|close|>response<|sep|><|close|>message<|sep|>"
    contents = []
    # Finish the residual analysis before sending any response bytes: checking
    # only the final concatenation would miss premature disclosure.
    for start in range(0, len(prefix), width):
        chunk = prefix[start:start + width]
        delta = parser.parse_delta(chunk, list(map(ord, chunk)), request,
                                   prompt_token_ids=[], finished=False)
        assert delta is None or not delta.content
    for start in range(0, len(body), width):
        chunk = body[start:start + width]
        delta = parser.parse_delta(chunk, list(map(ord, chunk)), request,
                                   prompt_token_ids=[], finished=False)
        if delta:
            assert not delta.reasoning
            contents.append(delta.content or "")
    # Wrapped responses must still stream before the final empty event.
    assert "".join(contents) == "answer"
    delta = parser.parse_delta("", [], request, prompt_token_ids=[], finished=True)
    if delta:
        contents.append(delta.content or "")
    assert "".join(contents) == parser._reasoning_parser.extract_reasoning(
        prefix + body, request
    )[1] == "answer"


@pytest.mark.parametrize("width", [1, 2, 7, 1000])
def test_prompt_open_response_streams_before_finish_and_stays_closed(width):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    parser = _stream_parser(False)
    request = ChatCompletionRequest(model="kimi", messages=[], tool_choice="none")
    prompt_ids = list(map(ord, '<|open|>message role="assistant"<|sep|>'
                              '<|open|>response<|sep|>'))
    # The response opener is in the prompt, never in the generated suffix.
    for chunk in ("answer", " continues"):
        delta = parser.parse_delta(chunk, list(map(ord, chunk)), request,
                                   prompt_token_ids=prompt_ids, finished=False)
        assert delta is not None
        assert delta.content == chunk
        assert not delta.reasoning
    close = "<|close|>response<|sep|><|close|>message<|sep|>"
    for start in range(0, len(close), width):
        chunk = close[start:start + width]
        delta = parser.parse_delta(chunk, list(map(ord, chunk)), request,
                                   prompt_token_ids=prompt_ids, finished=False)
        assert delta is None or not delta.content
    # Repeated prompt ids must not reopen a response closed by generation.
    delta = parser.parse_delta("", [], request, prompt_token_ids=prompt_ids,
                               finished=True)
    assert delta is None or not delta.content


@pytest.mark.parametrize("suffix", [
    "<|close|>response<|sep|><|close|>message<|sep|>",
    '<|close|>response<|sep|><|open|>message role="assistant"<|sep|>'
    '<|open|>think<|sep|>',
    "<|close|>response<|sep|><|open|>tool<|sep|>",
])
def test_old_prompt_response_does_not_release_residual_analysis(suffix):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    parser = _stream_parser(False)
    request = ChatCompletionRequest(model="kimi", messages=[], tool_choice="none")
    prompt_ids = list(map(ord, "<|open|>response<|sep|>old answer" + suffix))
    for chunk in ("analysis that must not reach the caller", "<|close|>think<|sep|>"):
        delta = parser.parse_delta(chunk, list(map(ord, chunk)), request,
                                   prompt_token_ids=prompt_ids, finished=False)
        assert delta is None or not delta.content
    chunk = "<|open|>response<|sep|>answer"
    delta = parser.parse_delta(chunk, list(map(ord, chunk)), request,
                               prompt_token_ids=prompt_ids, finished=False)
    assert delta.content == "answer"


def test_stream_preserves_plain_text_partial_prefix_at_finish_and_tool_channels():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    parser = _stream_parser(False)
    request = ChatCompletionRequest(model="kimi", messages=[], tool_choice="none")
    parts = ["a <", " b <|op"]
    deltas = [parser.parse_delta(part, [], request, prompt_token_ids=[], finished=i == 1)
              for i, part in enumerate(parts)]
    assert "".join(d.content or "" for d in deltas if d) == "".join(parts)

    # Active tool channels are left for the tool parser, as in non-streaming.
    parser = _stream_parser(False)
    request.tools = [SimpleNamespace()]
    request.tool_choice = "auto"
    raw = "<|open|>response<|sep|>tool payload"
    delta = parser.parse_delta(raw, [], request, prompt_token_ids=[], finished=True)
    assert delta.content == raw


@pytest.fixture
def vision_module(monkeypatch):
    import vllm.model_executor.models.kimi_k25_vit as module
    from vllm_hcu.runtime_compat.kimi_k25_vit import install_kimi_k25_qkv_layout_compat

    class Linear(nn.Linear):
        def __init__(self, input_size, output_size, bias=True, **kwargs):
            super().__init__(input_size, output_size, bias=bias)

    class QKV(Linear):
        def __init__(self, hidden_size, head_size, total_num_heads, bias, **kwargs):
            super().__init__(hidden_size, 3 * head_size * total_num_heads, bias=bias)

    monkeypatch.setattr(module, "is_vit_use_data_parallel", lambda: True)
    monkeypatch.setattr(module, "ColumnParallelLinear", Linear)
    monkeypatch.setattr(module, "RowParallelLinear", Linear)
    monkeypatch.setattr(module, "QKVParallelLinear", QKV)
    monkeypatch.setattr(module, "MMEncoderAttention", lambda **kwargs: nn.Identity())
    install_kimi_k25_qkv_layout_compat(module)
    return module


@pytest.mark.parametrize("norm,bias", [("rmsnorm", False), ("layernorm", True)])
def test_k3_vision_tower_honors_config_and_checkpoint_parameter_set(vision_module, norm, bias):
    config = KimiK3VisionConfig(
        vt_hidden_size=16, vt_num_attention_heads=2, vt_num_hidden_layers=1,
        vt_intermediate_size=32, qkv_hidden_size=24, norm_type=norm,
        attn_bias=bias, linear_bias=bias, patch_embed_proj_bias=bias,
        init_pos_emb_height=2, init_pos_emb_width=2,
    )
    tower = vision_module.MoonViT3dPretrainedModel(config)
    block = tower.encoder.blocks[0]
    expected_norm = nn.RMSNorm if norm == "rmsnorm" else nn.LayerNorm
    assert isinstance(block.norm0, expected_norm)
    assert isinstance(block.norm1, expected_norm)
    assert isinstance(tower.encoder.final_layernorm, expected_norm)
    assert block.wqkv.weight.shape == (72, 16)
    assert block.wo.weight.shape == (16, 24)
    for layer in (block.wqkv, block.wo, block.mlp.fc0, block.mlp.fc1, tower.patch_embed.proj):
        assert (layer.bias is not None) == bias
    if not bias:
        assert not any(name.endswith(".bias") for name in tower.state_dict())


def test_k25_vision_defaults_are_preserved(vision_module):
    from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig

    config = KimiK25VisionConfig(hidden_size=16, num_attention_heads=2,
                               num_hidden_layers=1, intermediate_size=32)
    tower = vision_module.MoonViT3dPretrainedModel(config)
    block = tower.encoder.blocks[0]
    assert isinstance(block.norm0, nn.LayerNorm)
    assert block.wqkv.weight.shape == (48, 16)
    assert all(layer.bias is not None for layer in (
        block.wqkv, block.wo, block.mlp.fc0, block.mlp.fc1, tower.patch_embed.proj))
