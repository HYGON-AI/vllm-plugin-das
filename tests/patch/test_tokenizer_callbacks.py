# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import builtins
import importlib.abc
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import vllm_hcu.patch.tokenizer_callbacks as tokenizer_callbacks
from vllm_hcu.patch._stage3_common import Stage3CompatibilityError
from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.runtime_state import (
    LatchedPatchError,
    PatchRegistry,
    PatchStatus,
)
from vllm_hcu.patch.tokenizer_callbacks import (
    DEEPSEEK_PATCH_ID,
    KIMI_PATCH_ID,
    TARGET_MODULE,
    apply_deepseek_distill_tokenizer,
    apply_kimi_k25_tokenizer,
    register_tokenizer_callbacks,
)


def _tokenizer_module():
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    class CachedHfTokenizer:
        @classmethod
        def from_pretrained(
            cls,
            path_or_repo_id,
            *args,
            trust_remote_code=False,
            revision=None,
            download_dir=None,
            **kwargs,
        ):
            all_kwargs = {
                "trust_remote_code": trust_remote_code,
                "revision": revision,
                "download_dir": download_dir,
                **kwargs,
            }
            calls.append((path_or_repo_id, args, all_kwargs))
            return "upstream-tokenizer", path_or_repo_id, all_kwargs

    module = ModuleType(TARGET_MODULE)
    module.CachedHfTokenizer = CachedHfTokenizer
    module.get_cached_tokenizer = lambda tokenizer: ("cached", tokenizer)
    return module, CachedHfTokenizer, calls


def test_kimi_callback_sets_regex_only_for_target_and_is_marker_idempotent():
    module, tokenizer_class, calls = _tokenizer_module()
    assert apply_kimi_k25_tokenizer(module) is True
    assert apply_kimi_k25_tokenizer(module) is False

    other = tokenizer_class.from_pretrained("Qwen3-8B", custom=1)
    kimi = tokenizer_class.from_pretrained("moonshot/Kimi-K2.5", custom=2)
    kimi_k26 = tokenizer_class.from_pretrained("/models/Kimi-K2.6", custom=3)

    assert other[2]["custom"] == 1
    assert "fix_mistral_regex" not in other[2]
    assert kimi[2]["custom"] == 2
    assert kimi[2]["fix_mistral_regex"] is True
    assert kimi_k26[2]["custom"] == 3
    assert kimi_k26[2]["fix_mistral_regex"] is True
    assert len(calls) == 3


def test_kimi_callback_retries_only_the_known_backend_tokenizer_error():
    class CachedHfTokenizer:
        calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(
            cls,
            path_or_repo_id,
            *args,
            trust_remote_code=False,
            revision=None,
            download_dir=None,
            **kwargs,
        ):
            cls.calls.append(dict(kwargs))
            if kwargs.get("fix_mistral_regex"):
                raise AttributeError(
                    "'tokenizers.Tokenizer' object has no attribute 'backend_tokenizer'"
                )
            return "fallback-ok"

    module = ModuleType(TARGET_MODULE)
    module.CachedHfTokenizer = CachedHfTokenizer
    module.get_cached_tokenizer = lambda tokenizer: tokenizer
    apply_kimi_k25_tokenizer(module)

    assert CachedHfTokenizer.from_pretrained("Kimi_K25") == "fallback-ok"
    assert CachedHfTokenizer.calls == [{"fix_mistral_regex": True}, {}]


def test_deepseek_callback_builds_local_llama_tokenizer_and_composes_with_kimi(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    module, tokenizer_class, calls = _tokenizer_module()
    apply_kimi_k25_tokenizer(module)
    assert apply_deepseek_distill_tokenizer(module) is True
    assert apply_deepseek_distill_tokenizer(module) is False

    model_path = tmp_path / "DeepSeek-R1-Distill-Llama-70B"
    model_path.mkdir()
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = {
        "legacy": False,
        "model_max_length": 32768,
        "bos_token": {"content": "<s>"},
        "eos_token": "</s>",
        "chat_template": "{{ messages }}",
    }
    (model_path / "tokenizer_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    created: list[dict[str, object]] = []

    def tokenizer_factory(**kwargs):
        created.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        tokenizer_callbacks,
        "_load_llama_tokenizer_fast",
        lambda: tokenizer_factory,
    )
    result = tokenizer_class.from_pretrained(model_path)

    assert result[0] == "cached"
    tokenizer = result[1]
    assert tokenizer.name_or_path == str(model_path)
    assert tokenizer.chat_template == "{{ messages }}"
    assert created == [
        {
            "tokenizer_file": str(model_path / "tokenizer.json"),
            "legacy": False,
            "model_max_length": 32768,
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
    ]
    assert calls == []

    tokenizer_class.from_pretrained("moonshot/Kimi-K2.5")
    assert calls[-1][2]["fix_mistral_regex"] is True


def test_composed_wrappers_preserve_subclass_cls_dispatch():
    class CachedHfTokenizer:
        @classmethod
        def from_pretrained(
            cls,
            path_or_repo_id,
            *args,
            trust_remote_code=False,
            revision=None,
            download_dir=None,
            **kwargs,
        ):
            return cls

    class DerivedTokenizer(CachedHfTokenizer):
        pass

    module = ModuleType(TARGET_MODULE)
    module.CachedHfTokenizer = CachedHfTokenizer
    module.get_cached_tokenizer = lambda tokenizer: tokenizer
    apply_kimi_k25_tokenizer(module)
    apply_deepseek_distill_tokenizer(module)

    assert DerivedTokenizer.from_pretrained("Qwen3-8B") is DerivedTokenizer
    assert DerivedTokenizer.from_pretrained("moonshot/Kimi-K2.5") is DerivedTokenizer


def test_llama_tokenizer_fast_import_path_is_available():
    from transformers.models.llama.tokenization_llama import LlamaTokenizerFast

    tokenizer_class = tokenizer_callbacks._load_llama_tokenizer_fast()
    assert tokenizer_class is LlamaTokenizerFast
    assert callable(tokenizer_class)


def test_callbacks_apply_immediately_to_loaded_module_and_repeat_registration(
    monkeypatch: pytest.MonkeyPatch,
):
    module, tokenizer_class, _ = _tokenizer_module()
    monkeypatch.setitem(sys.modules, TARGET_MODULE, module)
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    first = register_tokenizer_callbacks(coordinator)
    second = register_tokenizer_callbacks(coordinator)

    assert [item.status for item in first] == ["applied", "applied"]
    assert [item.status for item in second] == ["applied", "applied"]
    assert len(coordinator.registrations()) == 2
    assert getattr(tokenizer_class, "_hcu_kimi_k25_regex_patch_applied")
    assert getattr(
        tokenizer_class,
        "_hcu_deepseek_r1_distill_llama_70b_patch_applied",
    )


class _TokenizerLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        populated, tokenizer_class, _ = _tokenizer_module()
        module.CachedHfTokenizer = tokenizer_class
        module.get_cached_tokenizer = populated.get_cached_tokenizer


class _TokenizerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET_MODULE:
            return None
        return importlib.util.spec_from_loader(fullname, _TokenizerLoader())


def _fake_package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = []
    package.__spec__ = importlib.util.spec_from_loader(
        name, loader=None, is_package=True
    )
    return package


def test_callbacks_arm_before_module_load_and_run_on_exact_import(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "vllm", _fake_package("vllm"))
    monkeypatch.setitem(
        sys.modules, "vllm.tokenizers", _fake_package("vllm.tokenizers")
    )
    monkeypatch.delitem(sys.modules, TARGET_MODULE, raising=False)
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)
    registrations = register_tokenizer_callbacks(coordinator)
    assert [item.status for item in registrations] == ["armed", "armed"]

    coordinator.install()
    finder = _TokenizerFinder()
    sys.meta_path.insert(1, finder)
    try:
        module = builtins.__import__(TARGET_MODULE, fromlist=["*"])
    finally:
        while finder in sys.meta_path:
            sys.meta_path.remove(finder)
        coordinator.reset_for_tests(reset_registry=False)
        # The module was created by importlib after ``monkeypatch.delitem``;
        # remove it explicitly so it cannot poison later tests that import the
        # real ``vllm.tokenizers.hf`` module.
        sys.modules.pop(TARGET_MODULE, None)

    tokenizer_class = module.CachedHfTokenizer
    assert getattr(tokenizer_class, "_hcu_kimi_k25_regex_patch_applied")
    assert getattr(
        tokenizer_class,
        "_hcu_deepseek_r1_distill_llama_70b_patch_applied",
    )
    assert registry.get(KIMI_PATCH_ID).status is PatchStatus.APPLIED
    assert registry.get(DEEPSEEK_PATCH_ID).status is PatchStatus.APPLIED


def test_callback_failure_is_latched_and_not_silently_retried(
    monkeypatch: pytest.MonkeyPatch,
):
    broken_module = ModuleType(TARGET_MODULE)
    monkeypatch.setitem(sys.modules, TARGET_MODULE, broken_module)
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    with pytest.raises(Stage3CompatibilityError, match="CachedHfTokenizer"):
        coordinator.register_callback(
            KIMI_PATCH_ID,
            TARGET_MODULE,
            apply_kimi_k25_tokenizer,
        )
    record = registry.get(KIMI_PATCH_ID)
    assert record is not None and record.status is PatchStatus.FAILED

    with pytest.raises(LatchedPatchError, match="previously failed"):
        coordinator.register_callback(
            KIMI_PATCH_ID,
            TARGET_MODULE,
            apply_kimi_k25_tokenizer,
        )


@pytest.fixture(autouse=True)
def no_tokenizer_hub_network(monkeypatch):
    # Unit tests do not download metadata for fake non-Step repository IDs.
    import transformers.utils.hub

    original = transformers.utils.hub.cached_file
    monkeypatch.setattr(transformers.utils.hub, "cached_file", lambda *a, **k: None)
    return original


@pytest.mark.parametrize("model_type", ["step4", "step5_vl"])
@pytest.mark.parametrize("remote", [False, True])
def test_step_bytelevel_local_and_hub_preserve_backend(
    tmp_path, monkeypatch, model_type, remote
):
    import transformers
    import transformers.utils.hub
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    snapshot = tmp_path / "snapshots" / "revision-a" / "tokenizer"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    samples = ["hello world", "中文 有空格", " leading  spaces"]
    tokenizer.train_from_iterator(samples, trainer=trainer)
    tokenizer.save(str(snapshot / "tokenizer.json"))
    real_backend = transformers.TokenizersBackend
    hub_calls = []
    backend_calls = []

    def cached_file(repo_id, filename, **kwargs):
        hub_calls.append((repo_id, filename, kwargs))
        return str(snapshot / filename)

    class Backend:
        @classmethod
        def from_pretrained(cls, model_id, *args, **kwargs):
            backend_calls.append((model_id, args, kwargs))
            return real_backend(tokenizer_file=str(snapshot / "tokenizer.json"))

    monkeypatch.setattr(transformers.utils.hub, "cached_file", cached_file)
    monkeypatch.setattr(transformers, "TokenizersBackend", Backend)
    module, tokenizer_class, calls = _tokenizer_module()
    apply_deepseek_distill_tokenizer(module)
    model_id = "org/arbitrarily-named-checkpoint" if remote else snapshot.parent
    result = tokenizer_class.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision="revision-a",
        download_dir=str(tmp_path / "cache"),
        token="test-token",
        subfolder="tokenizer",
        local_files_only=True,
    )
    assert result[0] == "cached"
    assert calls == []
    for sample in samples:
        assert result[1].decode(result[1].encode(sample)) == sample
    assert backend_calls == [
        (
            model_id,
            (),
            {
                "trust_remote_code": True,
                "revision": "revision-a",
                "cache_dir": str(tmp_path / "cache"),
                "token": "test-token",
                "subfolder": "tokenizer",
                "local_files_only": True,
            },
        )
    ]
    if remote:
        assert [item[1] for item in hub_calls] == ["config.json", "tokenizer.json"]
        for repo_id, _, options in hub_calls:
            assert repo_id == model_id
            assert options["revision"] == "revision-a"
            assert options["cache_dir"] == str(tmp_path / "cache")
            assert options["token"] == "test-token"
            assert options["local_files_only"] is True
            assert options["subfolder"] == "tokenizer"
    else:
        assert hub_calls == []


@pytest.mark.parametrize(
    "config_kind", ["missing", "other", "malformed", "no_tokenizer"]
)
def test_step_hub_detection_preserves_upstream_for_other_or_missing_metadata(
    tmp_path, monkeypatch, config_kind
):
    import transformers.utils.hub

    config_file = tmp_path / "config.json"
    config_file.write_text(
        "[]"
        if config_kind == "malformed"
        else json.dumps(
            {"model_type": "qwen3" if config_kind == "other" else "step5_vl"}
        )
    )
    filenames = []

    def cached_file(repo_id, filename, **kwargs):
        filenames.append(filename)
        if config_kind == "missing" or filename == "tokenizer.json":
            return None
        return str(config_file)

    monkeypatch.setattr(transformers.utils.hub, "cached_file", cached_file)
    module, tokenizer_class, calls = _tokenizer_module()
    apply_deepseek_distill_tokenizer(module)
    result = tokenizer_class.from_pretrained("org/step5-lookalike", revision="v1")
    assert result[0] == "upstream-tokenizer"
    assert len(calls) == 1
    assert calls[0][2]["revision"] == "v1"
    assert filenames == (
        ["config.json", "tokenizer.json"]
        if config_kind == "no_tokenizer"
        else ["config.json"]
    )


def test_step_repo_id_resolves_real_offline_hub_snapshot(
    tmp_path, monkeypatch, no_tokenizer_hub_network
):
    import transformers.utils.hub

    cache = tmp_path / "hub"
    repo_cache = cache / "models--org--metadata-checkpoint"
    sha = "a" * 40
    snapshot = repo_cache / "snapshots" / sha
    snapshot.mkdir(parents=True)
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "revision-a").write_text(sha)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "step5_vl"}))
    (snapshot / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(transformers.utils.hub, "cached_file", no_tokenizer_hub_network)
    assert tokenizer_callbacks._is_step_bytelevel(
        "org/metadata-checkpoint",
        revision="revision-a",
        download_dir=str(cache),
        local_files_only=True,
    )
    assert not tokenizer_callbacks._is_step_bytelevel(
        "org/metadata-checkpoint",
        revision="missing-revision",
        download_dir=str(cache),
        local_files_only=True,
    )
