# SPDX-License-Identifier: Apache-2.0
"""Real installed config parsing and Adapter lifecycle, without model weights."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import pytest


@pytest.mark.parametrize("case", [
    "discovery", "versions", "roundtrip-text", "roundtrip-vl", "roundtrip-flat",
    "geometry", "foreign-first", "foreign-second", "drift-first", "drift-second",
    "class-drift", "partial-second", "idempotence", "spawn", "fork",
])
def test_installed_hyv4_flash_config(case):
    script = str(Path(__file__).resolve())
    bootstrap = (
        f"import runpy,sys;sys.path.insert(0,{sysconfig.get_path('purelib')!r});"
        f"sys.argv=[{script!r},{case!r}];"
        f"runpy.run_path({script!r},run_name='__main__')"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(HCU_ADAPTER_HOST="none", VLLM_PLUGINS="none",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", bootstrap],
        env=environment, cwd="/", capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"HYV4_FLASH_CONFIG_OK={case}" in result.stdout


def _entry_point():
    from importlib.metadata import entry_points
    points = [point for point in entry_points(group="hcu_adapter.extensions")
              if point.name == "das_hcu_hyv4_flash_config"]
    assert len(points) == 1, points
    return points[0]


def _activate():
    from hcu_adapter.extensions import ExtensionContext, ExtensionStatus, activate_extensions
    reports = activate_extensions(
        ExtensionContext("0.5.22", "0.33.0+flash.config.test",
            requested_capabilities={"hyv4-flash-config"}),
        entry_points=[_entry_point()],
    )
    assert len(reports) == 1 and reports[0].status is ExtensionStatus.ACTIVE
    assert reports[0].capabilities == {"hyv4-flash-config"}


def _tiny():
    return dict(num_hidden_layers=2, hidden_size=64, vocab_size=128,
        intermediate_size=128, moe_intermediate_size=64, num_attention_heads=4,
        n_routed_experts=8, num_experts_per_tok=2, q_lora_rank=16,
        index_topk=128, compressed_indexer_group_topk=32,
        compressed_indexer_candidate_topk=128, sliding_window=32,
        mlp_layer_types=["dense", "sparse"], layer_types=["sparse", "sliding_attention"],
        pad_token_id=0, bos_token_id=1, eos_token_id=2)


def _child(queue, inherited):
    from hcu_vllm_das import hyv4_flash_config_activation as owner
    if inherited:
        with pytest.raises(owner.FlashConfigError, match="current-process"):
            owner.check_health()
    _activate()
    owner.check_health()
    from transformers import AutoConfig
    queue.put(AutoConfig.for_model("hy_v4_flash", **_tiny()).model_type)


def _scenario(case):
    before = set(sys.modules)
    from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
    from hcu_vllm_das import hyv4_flash_config_activation as owner
    context = ExtensionContext("0.5.22", "0.33.0")
    if case == "discovery":
        spec = _entry_point().load()()
        assert spec.capabilities == {"hyv4-flash-config"}
        assert spec.check_health is owner.check_health
        assert not any(name.split(".")[0] in {"torch", "vllm", "transformers", "lightop"}
                       for name in set(sys.modules) - before)
        return
    if case == "versions":
        for version in (None, "bad", "1!0.33.0", "0.32.9", "0.34.0"):
            with pytest.raises(owner.FlashConfigError, match="release line"):
                owner.activate(ExtensionContext("0.5.22", version))
        assert "torch" not in sys.modules and "vllm" not in sys.modules
        return

    from transformers import AutoConfig, CONFIG_MAPPING, PreTrainedConfig
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config
    from vllm.model_executor.models.registry import ModelRegistry
    registry_before = dict(ModelRegistry.models)
    keys = ("hy_v4_flash", "hy_v4_flash_vl")
    if case.startswith("foreign"):
        key = keys[case.endswith("second")]
        foreign = type("Foreign", (PreTrainedConfig,), {"model_type": key})
        AutoConfig.register(key, foreign)
        with pytest.raises(owner.FlashConfigError, match="already has a config owner"):
            owner.activate(context)
        assert CONFIG_MAPPING[key] is foreign
        assert keys[not case.endswith("second")] not in CONFIG_MAPPING
        return
    if case == "partial-second":
        original = AutoConfig.register
        calls = []
        def fail_second(key, cls, **kwargs):
            calls.append(key)
            if len(calls) == 2:
                raise RuntimeError("injected before second publication")
            return original(key, cls, **kwargs)
        AutoConfig.register = fail_second
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        AutoConfig.register = original
        assert calls == list(keys) and keys[0] in CONFIG_MAPPING and keys[1] not in CONFIG_MAPPING
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        with pytest.raises(ExtensionCommitError):
            owner.check_health()
        assert keys[1] not in CONFIG_MAPPING
        return

    _activate()
    from hcu_vllm_das.models.hy_v4_flash.configs import HYV4FlashConfig, HYV4FlashVLConfig, SPARSE_ATTENTION
    classes = (HYV4FlashConfig, HYV4FlashVLConfig)
    assert all(CONFIG_MAPPING[key] is cls for key, cls in zip(keys, classes))
    assert HYV4Config.model_type == "hy_v4"
    assert ModelRegistry.models == registry_before  # Metadata capability only.
    if case.startswith("drift"):
        key = keys[case.endswith("second")]
        foreign = type("Foreign", (PreTrainedConfig,), {"model_type": key})
        AutoConfig.register(key, foreign)
        with pytest.raises(ExtensionCommitError):
            owner.check_health()
        assert CONFIG_MAPPING[key] is foreign
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        return
    if case == "class-drift":
        HYV4FlashConfig.model_type = "foreign"
        with pytest.raises(ExtensionCommitError):
            owner.check_health()
        HYV4FlashConfig.model_type = keys[0]
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        return
    if case == "idempotence":
        original = AutoConfig.register
        def forbid(*args, **kwargs):
            raise AssertionError("second publication")
        AutoConfig.register = forbid
        _activate()
        owner.check_health()
        AutoConfig.register = original
        return
    if case in {"fork", "spawn"}:
        import multiprocessing
        ctx = multiprocessing.get_context(case)
        queue = ctx.Queue()
        process = ctx.Process(target=_child, args=(queue, case == "fork"))
        process.start()
        try:
            assert queue.get(timeout=40) == keys[0]
            process.join(10)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
            queue.close()
            queue.join_thread()
        owner.check_health()
        return
    if case == "geometry":
        values = _tiny()
        values["compressed_indexer_candidate_topk"] = 127
        with pytest.raises(ValueError, match="group_topk"):
            AutoConfig.for_model(keys[0], **values)
        text = {**_tiny(), "architectures": ["HYV4FlashForCausalLM"]}
        with pytest.raises(ValueError, match="projector output"):
            AutoConfig.for_model(keys[1], text_config=text,
                                 vision_config={"out_hidden_size": 63})
        return

    text = {**_tiny(), "architectures": ["HYV4FlashForCausalLM"]}
    if case == "roundtrip-text":
        raw = {"model_type": keys[0], **text}
        expected = HYV4FlashConfig
    elif case == "roundtrip-vl":
        raw = dict(model_type=keys[1], text_config=text,
                   vision_config={"hidden_size": 32, "num_attention_heads": 4})
        expected = HYV4FlashVLConfig
    elif case == "roundtrip-flat":
        raw = dict(model_type=keys[0], vision_config={"hidden_size": 32}, **text)
        expected = HYV4FlashVLConfig
    else:
        raise AssertionError(case)
    original = deepcopy(raw)
    direct = AutoConfig.for_model(raw["model_type"], **{k: v for k, v in raw.items() if k != "model_type"}) if case != "roundtrip-flat" else HYV4FlashConfig.from_dict(raw)
    assert isinstance(direct, expected) and raw == original
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text(json.dumps(raw))
        parsed = get_config(root, trust_remote_code=False)
        assert isinstance(parsed, expected), type(parsed)
        parsed.save_pretrained(root)
        again = AutoConfig.from_pretrained(root, trust_remote_code=False)
        assert isinstance(again, expected)
        actual = again.text_config if isinstance(again, HYV4FlashVLConfig) else again
        assert actual.hidden_size == 64 and actual.index_topk == 128
        assert actual.layer_types == [SPARSE_ATTENTION, "sliding_attention"]
        assert actual.flash_w4a8_backend == "disabled"
        if isinstance(again, HYV4FlashVLConfig):
            assert again.architectures == ["HYV4FlashVLForConditionalGeneration"]
            assert again.vision_config.out_hidden_size == 64
    owner.check_health()
    assert HYV4Config.model_type == "hy_v4" and ModelRegistry.models == registry_before


if __name__ == "__main__":
    _scenario(sys.argv[1])
    print("HYV4_FLASH_CONFIG_OK=" + sys.argv[1])
