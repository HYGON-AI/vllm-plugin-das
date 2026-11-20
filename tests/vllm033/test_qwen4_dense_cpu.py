# SPDX-License-Identifier: Apache-2.0
"""Real official constructors/loaders with CPU leaf doubles; no device claims."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


CASES = ("discovery", "default-off", "versions", "platform", "registry", "foreign",
         "partial", "drift", "fork", "fork-partial", "spawn", "constructors",
         "config", "loaders", "streaming")


@pytest.mark.parametrize("case", CASES)
def test_installed_qwen4_dense_contract(case):
    env = dict(os.environ, HCU_ADAPTER_HOST="none", VLLM_PLUGINS="none",
               HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    script = str(Path(__file__).resolve())
    command = [sys.executable, "-I", "-B", script, case]
    if case in {"discovery", "default-off", "versions"}:
        code = (f"import sys,runpy; sys.path.insert(0,{sysconfig.get_path('purelib')!r});"
                f"sys.argv=[{script!r},{case!r}];runpy.run_path({script!r},run_name='__main__')")
        command = [sys.executable, "-I", "-S", "-B", "-c", code]
    run = subprocess.run(command, env=env, cwd="/", capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "QWEN4_DENSE_OK=" + case in run.stdout


def _point():
    from importlib.metadata import entry_points
    points = [p for p in entry_points(group="hcu_adapter.extensions")
              if p.name == "das_hcu_qwen4_dense"]
    assert len(points) == 1
    return points[0]


def _activate():
    from hcu_adapter.extensions import ExtensionContext, ExtensionStatus, activate_extensions
    from hcu_vllm_das import qwen4_dense_activation as owner
    # CPU contract: only the platform admission predicate is doubled. No device
    # capability or native execution is inferred from the result.
    owner._is_hcu = lambda: True
    reports = activate_extensions(ExtensionContext("0.5.22", "0.33.0",
        requested_capabilities={owner.CAPABILITY}), entry_points=[_point()])
    assert len(reports) == 1 and reports[0].status is ExtensionStatus.ACTIVE
    owner.check_health()


def _spawn(queue):
    _activate()
    from vllm.model_executor.models import ModelRegistry
    queue.put(ModelRegistry.models["Qwen4ExpMTP"].module_name)


def _scenario(case):
    before = set(sys.modules)
    from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
    from hcu_vllm_das import qwen4_dense_activation as owner
    context = ExtensionContext("0.5.22", "0.33.0", requested_extensions={owner.ENTRY})
    if case in {"discovery", "default-off", "versions"}:
        spec = _point().load()()
        assert spec.check_health is owner.check_health
        assert spec.capabilities == {owner.CAPABILITY}
        if case == "default-off":
            with pytest.raises(RuntimeError, match="explicit"):
                owner.activate(ExtensionContext("0.5.22", "0.33.0"))
        if case == "versions":
            for version in (None, "bad", "0.32.9", "0.34.0", "1!0.33.0"):
                with pytest.raises(RuntimeError, match="vLLM 0.33"):
                    owner.activate(ExtensionContext("0.5.22", version,
                        requested_extensions={owner.ENTRY}))
        assert not any(n.split('.')[0] in {"torch", "vllm", "transformers", "lightop"}
                       for n in set(sys.modules) - before)
        return
    if case == "platform":
        with pytest.raises(RuntimeError, match="requires HCU"):
            owner.activate(context)
        return

    from vllm.model_executor.models import ModelRegistry
    if case == "foreign":
        ModelRegistry.register_model("Qwen4ExpMTP", "foreign.model:Other")
        original = dict(ModelRegistry.models)
        owner._is_hcu = lambda: True
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        assert ModelRegistry.models == original
        return
    if case in {"partial", "fork-partial"}:
        register = ModelRegistry.register_model
        count = 0
        child_status = []
        def publish(*args, **kwargs):
            nonlocal count
            count += 1
            register(*args, **kwargs)
            if count == 1 and case == "fork-partial":
                pid = os.fork()
                if pid == 0:
                    try:
                        with pytest.raises(ExtensionCommitError):
                            owner.activate(context)
                    except BaseException:
                        os._exit(1)
                    os._exit(0)
                child_status.append(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
            if count == 2 and case == "partial":
                raise KeyboardInterrupt("injected after public registration")
        ModelRegistry.register_model = publish
        if case == "partial":
            with pytest.raises(ExtensionCommitError):
                _activate()
            ModelRegistry.register_model = register
            assert ModelRegistry.models[owner._ARCHES[0]].module_name == owner._MODULE
            for fn in (lambda: owner.activate(context), owner.check_health):
                with pytest.raises(ExtensionCommitError):
                    fn()
        else:
            _activate()
            assert child_status == [0]
        return

    _activate()
    if case == "drift":
        original = ModelRegistry.models[owner._ARCHES[0]]
        ModelRegistry.register_model(owner._ARCHES[0], "foreign.model:Other")
        with pytest.raises(ExtensionCommitError):
            owner.check_health()
        assert ModelRegistry.models[owner._ARCHES[0]].module_name == "foreign.model"
        ModelRegistry.models[owner._ARCHES[0]] = original
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        return
    if case == "fork":
        import threading
        ready, release = threading.Event(), threading.Event()
        def hold():
            with owner._lifecycle._lock:
                ready.set()
                assert release.wait(45)
        thread = threading.Thread(target=hold)
        thread.start()
        assert ready.wait(5)
        pid = os.fork()
        if pid == 0:
            try:
                with pytest.raises(RuntimeError, match="after fork"):
                    owner.check_health()
                owner.activate(context)
                owner.check_health()
            except BaseException:
                import traceback
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        release.set()
        thread.join(5)
        assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0
        owner.check_health()
        return
    if case == "spawn":
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        child = ctx.Process(target=_spawn, args=(queue,))
        child.start()
        child.join(60)
        assert child.exitcode == 0
        assert queue.get(timeout=5) == owner._MODULE
        queue.close()
        queue.join_thread()
        return

    import torch
    from torch import nn
    from types import SimpleNamespace as NS
    assert owner._MODULE not in sys.modules  # Publication is genuinely lazy.
    from vllm.models.qwen4_exp.amd import model as official
    from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP, Qwen4ExpMultiTokenPredictor
    from hcu_vllm_das.models import qwen4_dense as candidate
    from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

    def text_config(heads=None, **kwargs):
        if heads is not None:
            kwargs.update(indexer_n_heads=heads, indexer_kv_heads=1,
                          indexer_head_dim=256, indexer_budget=2048,
                          indexer_compress_ratio=4)
        return Qwen4ExpTextConfig(**kwargs)

    pairs = [(candidate.HcuQwen4ExpForCausalLM, official.Qwen4ExpForCausalLM),
             (candidate.HcuQwen4ExpForConditionalGeneration, official.Qwen4ExpForConditionalGeneration),
             (candidate.HcuQwen4ExpMTP, Qwen4ExpMTP)]
    if case == "registry":
        assert owner._MODULE not in before
        for derived, base in pairs:
            resolved, arch = ModelRegistry.resolve_model_cls(
                [base.__name__], NS(model_impl="vllm"))
            assert resolved is derived and arch == base.__name__
            assert derived.__init__ is base.__init__
            assert derived.forward is base.forward
        assert (candidate.HcuQwen4ExpForConditionalGeneration._processor_factory
                is official.Qwen4ExpForConditionalGeneration._processor_factory)
    elif case == "constructors":
        with pytest.MonkeyPatch.context() as mp:
            class Leaf(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.args, self.kwargs = args, kwargs
            class Dense(Leaf): pass
            class Sparse(Leaf): pass
            mp.setattr(official, "Qwen3NextAttention", Dense)
            mp.setattr(official, "Qwen4ExpQSAAttention", Sparse)
            for name in ("Qwen4ExpSparseMoeBlock", "GatedResidual", "QwenGatedDeltaNetAttention"):
                mp.setattr(official, name, Leaf)
            mp.setattr(official, "is_hc_sequence_parallel_enabled", lambda _: False)
            for heads in (None, 4):
                text = text_config(heads, ple_layer_ids=[])
                before_config = text.to_dict()
                config = NS(model_config=NS(hf_text_config=text), quant_config=None,
                            cache_config=None, parallel_config=NS(use_sequence_parallel_moe=False))
                for layer_type in ("indexed_attention", "qwen_sparse_attention"):
                    layer = official.Qwen4ExpDecoderLayer(config, layer_type, "model.layers.0")
                    assert type(layer.self_attn) is (Dense if heads is None else Sparse)
                    assert text.to_dict() == before_config
                linear = official.Qwen4ExpDecoderLayer(config, "linear_attention", "model.layers.0")
                assert type(linear.linear_attn) is Leaf
                with pytest.raises(ValueError, match="Invalid layer_type"):
                    official.Qwen4ExpDecoderLayer(config, "full_attention", "model.layers.0")
    elif case == "config":
        import json
        import tempfile
        from vllm.config import ModelConfig
        with tempfile.TemporaryDirectory() as directory:
            text = text_config(4, num_hidden_layers=2,
                layer_types=["indexed_attention", "linear_attention"])
            Qwen4ExpConfig(text_config=text, architectures=["Qwen4ExpForConditionalGeneration"]).save_pretrained(directory)
            configs = [ModelConfig(model=directory, tokenizer_mode="skip", dtype="bfloat16",
                max_model_len=4096, hf_overrides=override) for override in (
                    {}, {"text_config": {"indexer_n_heads": None}}, {"indexer_n_heads": None})]
            assert [c.hf_text_config.indexer_n_heads for c in configs] == [4, None, 4]
            assert all(isinstance(c.hf_text_config, Qwen4ExpTextConfig) for c in configs)
            assert all(c.hf_text_config.layer_types == text.layer_types for c in configs)
            assert configs[0].compute_hash() != configs[1].compute_hash()
    elif case == "streaming":
        text = text_config()
        value = torch.ones(2, 2)
        seen = []
        def weights():
            for name in ("model.layers.0.self_attn.indexer.index_qk_proj.weight",
                         "model.layers.0.self_attn.qkv_proj.weight", "visual.indexer.weight"):
                seen.append(name)
                yield name, value
        filtered = candidate._dense_weights(text, weights())
        assert not seen
        assert next(filtered) == ("model.layers.0.self_attn.qkv_proj.weight", value)
        assert len(seen) == 2
        assert next(filtered) == ("visual.indexer.weight", value)
        stream = weights()
        assert candidate._dense_weights(text_config(4), stream) is stream
    elif case == "loaders":
        def inner(cls, heads):
            model = cls.__new__(cls)
            nn.Module.__init__(model)
            model.config = text_config(heads)
            model.is_fused_shared_expert_enabled = False
            model._qsa_layer_ids = frozenset() if heads is None else frozenset({0})
            model.hyper_connection_mixer = None
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.qkv_proj = nn.Linear(2, 2, bias=False)
            if heads is not None:
                layer.self_attn.indexer = nn.Module()
                layer.self_attn.indexer.index_qk_proj = nn.Linear(2, 2, bias=False)
            model.layers = nn.ModuleList([layer])
            return model
        def root(cls, heads):
            model = cls.__new__(cls)
            nn.Module.__init__(model)
            text = text_config(heads)
            if issubclass(cls, official.Qwen4ExpForConditionalGeneration):
                model.config = Qwen4ExpConfig(text_config=text)
                model.language_model_only = False
                model.language_model = root(official.Qwen4ExpForCausalLM, heads)
                model.visual = nn.Linear(2, 2, bias=False)
            else:
                model.config = text
                model.model = inner(Qwen4ExpMultiTokenPredictor if issubclass(cls, Qwen4ExpMTP)
                                    else official.Qwen4ExpModel, heads)
            return model
        for derived, base in pairs:
            prefixes = (["mtp.", "model.mtp.", "model.language_model.mtp.", "language_model.mtp."]
                        if issubclass(base, Qwen4ExpMTP) else ["model.language_model."])
            for prefix in prefixes:
                indexer = prefix + "layers.0.self_attn.indexer.index_qk_proj.weight"
                qkv = prefix + "layers.0.self_attn.qkv_proj.weight"
                value = torch.arange(4, dtype=torch.float32).reshape(2, 2)
                stock = root(base, None)
                with pytest.raises(ValueError, match="indexer"):
                    stock.load_weights(iter([(qkv, value), (indexer, value)]))
                adapted = root(derived, None)
                loaded = adapted.load_weights(iter([(qkv, value), (indexer, value)]))
                assert len(loaded) == 1 and next(iter(loaded)).endswith("qkv_proj.weight")
                inner_model = adapted.language_model.model if hasattr(adapted, "language_model") else adapted.model
                torch.testing.assert_close(inner_model.layers[0].self_attn.qkv_proj.weight, value)
                sparse = root(derived, 4)
                assert len(sparse.load_weights(iter([(qkv, value), (indexer, value)]))) == 2
                with pytest.raises(ValueError, match="other_missing"):
                    adapted.load_weights(iter([(prefix + "layers.0.self_attn.other_missing.weight", value)]))
            if issubclass(base, official.Qwen4ExpForConditionalGeneration):
                model = root(derived, None)
                assert model.load_weights(iter([("visual.weight", value)])) == {"visual.weight"}
                with pytest.raises(ValueError, match="indexer"):
                    model.load_weights(iter([("visual.indexer.weight", value)]))
        dense_name = "layers.0.self_attn.attn._k_scale"
        assert official._remap_qsa_cache_scale_name(dense_name, frozenset()) == dense_name
        assert official._remap_qsa_cache_scale_name(dense_name, frozenset({0})) == "layers.0.self_attn._k_scale"
    else:
        raise AssertionError(case)
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    _scenario(sys.argv[1])
    print("QWEN4_DENSE_OK=" + sys.argv[1])
