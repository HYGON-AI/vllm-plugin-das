# SPDX-License-Identifier: Apache-2.0
"""CPU callback contracts using the real tokenizers DecodeStream."""
import importlib
import importlib.util
from types import ModuleType, SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

ADAPTER = "vllm_hcu.patch.platform.framework_opt.patch_detokenizer"
FLAG = "VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL"


def load_adapter():
    assert importlib.util.find_spec(ADAPTER) is not None, "frontend callback is missing"
    return importlib.import_module(ADAPTER)


def target_module(decoder=None, substitute_stream=False):
    backend = Tokenizer(models.BPE(
        {c:i for i,c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoder if decoder is not None else decoders.ByteLevel()
    module = ModuleType("vllm.v1.engine.detokenizer")

    class FastIncrementalDetokenizer:
        def __init__(self, tokenizer, request):
            self.tokenizer = tokenizer._tokenizer
            self.skip_special_tokens = request.sampling_params.skip_special_tokens
            self.original_ids = request.prompt_token_ids
            self.stream = (object() if substitute_stream else decoders.DecodeStream(
                ids=request.prompt_token_ids, skip_special_tokens=self.skip_special_tokens))

    module.FastIncrementalDetokenizer = FastIncrementalDetokenizer
    ids = backend.encode("ordinary prefix " * 100).ids
    return module, SimpleNamespace(_tokenizer=backend), SimpleNamespace(
        prompt_token_ids=ids, sampling_params=SimpleNamespace(skip_special_tokens=True))


def test_disabled_callback_keeps_constructor_identity(monkeypatch):
    adapter = load_adapter()
    monkeypatch.delenv(FLAG, raising=False)
    module, _, _ = target_module()
    before = module.FastIncrementalDetokenizer.__init__
    assert adapter.apply_to_module(module) is False
    assert module.FastIncrementalDetokenizer.__init__ is before


def test_enabled_callback_keeps_request_and_first_deltas(monkeypatch):
    adapter = load_adapter()
    monkeypatch.setenv(FLAG, "1")
    module, tokenizer, request = target_module()
    before = module.FastIncrementalDetokenizer(tokenizer, request)
    original_ids = request.prompt_token_ids[:]
    original_init = module.FastIncrementalDetokenizer.__init__
    original_streams = []
    def record_original(self, tokenizer, request):
        original_init(self, tokenizer, request)
        original_streams.append(self.stream)
    module.FastIncrementalDetokenizer.__init__ = record_original
    assert adapter.apply_to_module(module) is True
    after = module.FastIncrementalDetokenizer(tokenizer, request)
    assert after.stream is not original_streams[-1]
    assert after.original_ids is request.prompt_token_ids
    assert request.prompt_token_ids == original_ids
    output = tokenizer._tokenizer.encode("你好 😀 abc").ids
    assert [before.stream.step(tokenizer._tokenizer,t) for t in output] == [
        after.stream.step(tokenizer._tokenizer,t) for t in output]


@pytest.mark.parametrize("decoder,substitute_stream", [(decoders.BPEDecoder(),False),(None,True)])
def test_unsupported_backend_keeps_original_stream(monkeypatch, decoder, substitute_stream):
    adapter = load_adapter()
    monkeypatch.setenv(FLAG, "1")
    module, tokenizer, request = target_module(decoder,substitute_stream)
    original = module.FastIncrementalDetokenizer.__init__
    objects=[]
    def recording(self, tokenizer, request):
        original(self,tokenizer,request);objects.append(self.stream)
    module.FastIncrementalDetokenizer.__init__=recording
    assert adapter.apply_to_module(module) is True
    instance=module.FastIncrementalDetokenizer(tokenizer,request)
    assert instance.stream is objects[-1]


def test_registration_is_idempotent_and_stale_wrapper_is_rejected(monkeypatch):
    adapter = load_adapter()
    monkeypatch.setenv(FLAG, "1")
    module, _, _ = target_module()
    assert adapter.apply_to_module(module) is True
    assert adapter.apply_to_module(module) is False
    module.FastIncrementalDetokenizer.__init__ = lambda self,tokenizer,request: None
    with pytest.raises(RuntimeError, match="stale"):
        adapter.apply_to_module(module)


@pytest.mark.parametrize("patch_before_apply", [True, False])
def test_decode_stream_monkeypatch_before_or_after_callback_is_preserved(monkeypatch, patch_before_apply):
    adapter = load_adapter()
    monkeypatch.setenv(FLAG, "1")
    module, tokenizer, request = target_module()
    constructed = []

    class ReplacementStream:
        def __init__(self, *, ids, skip_special_tokens):
            self.ids = ids
            self.skip_special_tokens = skip_special_tokens
            constructed.append(self)

    if patch_before_apply:
        monkeypatch.setattr(decoders, "DecodeStream", ReplacementStream)
    assert adapter.apply_to_module(module) is True
    if not patch_before_apply:
        monkeypatch.setattr(decoders, "DecodeStream", ReplacementStream)
    instance = module.FastIncrementalDetokenizer(tokenizer, request)
    assert len(constructed) == 1
    assert instance.stream is constructed[0]
    assert instance.stream.ids is request.prompt_token_ids
    assert instance.stream.skip_special_tokens is True
