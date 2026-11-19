# SPDX-License-Identifier: Apache-2.0
"""Installed config parsing and lifecycle; no weights, kernels or GPU claims."""

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


def _run(case):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        HCU_ADAPTER_HOST="none",
        VLLM_PLUGINS="none",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ROCR_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    script = str(Path(__file__).resolve())
    command = [sys.executable, "-I", "-B", script, case]
    if case in {"discovery", "versions"}:
        # Adapter's installed sitecustomize can import Torch before this file.
        # Skip startup hooks to measure imports made by discovery itself.
        bootstrap = (
            f"import runpy,sys; sys.path.insert(0, {sysconfig.get_path('purelib')!r}); "
            f"sys.argv = [{script!r}, {case!r}]; "
            f"runpy.run_path({script!r}, run_name='__main__')"
        )
        command = [sys.executable, "-I", "-S", "-B", "-c", bootstrap]
    result = subprocess.run(
        command,
        env=environment,
        cwd="/",
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"STEP5_CONFIG_OK={case}" in result.stdout


@pytest.mark.parametrize(
    "case",
    [
        "roundtrip-default",
        "roundtrip-explicit",
        "foreign",
        "drift",
        "class-drift",
        "in-checkpoint",
        "partial-commit",
        "versions",
        "discovery",
        "spawn",
        "fork",
    ],
)
def test_installed_step5_config_contract(case):
    _run(case)


def _entry_point():
    from importlib.metadata import entry_points

    points = [
        point
        for point in entry_points(group="hcu_adapter.extensions")
        if point.name == "das_hcu_step5_config"
    ]
    assert len(points) == 1, points
    return points[0]


def _activate():
    from hcu_adapter.extensions import (
        ExtensionContext,
        ExtensionStatus,
        activate_extensions,
    )

    reports = activate_extensions(
        ExtensionContext(
            "0.5.22",
            "0.33.0+config.test",
            requested_capabilities={"step5-standalone-mtp-config"},
        ),
        entry_points=[_entry_point()],
    )
    assert len(reports) == 1 and reports[0].status is ExtensionStatus.ACTIVE
    assert reports[0].capabilities == {"step5-standalone-mtp-config"}


def _spawn_check(queue):
    _activate()
    from transformers import AutoConfig

    queue.put(AutoConfig.for_model("step5_mtp").model_type)


def _scenario(case):
    before = set(sys.modules)
    from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
    from hcu_vllm_das import step5_activation as owner

    context = ExtensionContext("0.5.22", "0.33.0")
    if case == "discovery":
        spec = _entry_point().load()()
        assert spec.check_health is owner.check_health
        assert spec.capabilities == {"step5-standalone-mtp-config"}
        introduced = set(sys.modules) - before
        assert not any(
            name.split(".")[0] in {"torch", "vllm", "lightop", "transformers"}
            for name in introduced
        )
        return
    if case == "versions":
        for value in (None, "bad", "0.32.9", "0.34.0", "1!0.33.0"):
            with pytest.raises(owner.Step5ConfigError, match="release line"):
                owner.activate(ExtensionContext("0.5.22", value))
        assert "vllm" not in sys.modules and "torch" not in sys.modules
        return

    from transformers import AutoConfig, CONFIG_MAPPING, PreTrainedConfig
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.configs import Step5Config, Step5VLConfig

    if case == "foreign":

        class Foreign(PreTrainedConfig):
            model_type = "step5_mtp"

        AutoConfig.register("step5_mtp", Foreign)
        with pytest.raises(owner.Step5ConfigError, match="already has a config owner"):
            owner.activate(context)
        assert CONFIG_MAPPING["step5_mtp"] is Foreign
        assert Step5Config.model_type == "step5"
        return
    if case == "partial-commit":
        register = AutoConfig.register

        def publish_then_fail(*args, **kwargs):
            register(*args, **kwargs)
            raise RuntimeError("injected after real registration")

        AutoConfig.register = publish_then_fail
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        AutoConfig.register = register
        assert "step5_mtp" in CONFIG_MAPPING  # No fictitious rollback.
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        with pytest.raises(ExtensionCommitError):
            owner.check_health()
        return

    _activate()
    cls = CONFIG_MAPPING["step5_mtp"]
    assert issubclass(cls, Step5Config) and cls is not Step5Config
    if case == "class-drift":
        cls.model_type = "foreign_mutated_type"
        with pytest.raises(ExtensionCommitError):
            owner.check_health()
        assert cls.model_type == "foreign_mutated_type"  # No repair in health.
        cls.model_type = "step5_mtp"
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        return
    if case == "in-checkpoint":
        from itertools import product
        from vllm.config.speculative import SpeculativeConfig

        for vl, layers, quantized in product((False, True), (0, 1, 3), (False, True)):
            text = Step5Config(
                num_nextn_predict_layers=layers, architectures=["Step5ForCausalLM"]
            )
            target = Step5VLConfig(text_config=text) if vl else text
            if quantized:
                target.quantization_config = {"quant_method": "compressed-tensors"}
            draft = SpeculativeConfig.hf_config_override(target)
            assert draft.model_type == "step5_mtp"
            assert draft.architectures == ["Step5MTP"]
            assert draft.n_predict == layers
            if quantized:
                assert draft.quantization_config == target.quantization_config
            assert Step5Config.model_type == "step5"
            assert Step5VLConfig.model_type == "step5_vl"
            owner.check_health()
        return
    if case == "drift":

        class Foreign(PreTrainedConfig):
            model_type = "step5_mtp"

        AutoConfig.register("step5_mtp", Foreign)
        with pytest.raises(ExtensionCommitError):
            owner.check_health()
        assert CONFIG_MAPPING["step5_mtp"] is Foreign
        AutoConfig.register("step5_mtp", cls)
        with pytest.raises(ExtensionCommitError):
            owner.activate(context)
        return
    if case == "spawn":
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        child = ctx.Process(target=_spawn_check, args=(queue,))
        child.start()
        child.join(45)
        assert child.exitcode == 0
        assert queue.get(timeout=5) == "step5_mtp"
        queue.close()
        queue.join_thread()
        return
    if case == "fork":
        import threading

        ready, release = threading.Event(), threading.Event()

        def hold_parent_lock():
            with owner._lock:
                ready.set()
                assert release.wait(45)

        thread = threading.Thread(target=hold_parent_lock)
        thread.start()
        assert ready.wait(5)
        child = os.fork()
        if child == 0:
            try:
                with pytest.raises(owner.Step5ConfigError, match="after fork"):
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
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        owner.check_health()
        return

    import pickle
    import tempfile

    layers = 3 if case == "roundtrip-explicit" else 1
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = dict(
            model_type="step5_mtp",
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_attention_groups=2,
            num_hidden_layers=2,
            vocab_size=258,
            layer_types=["full_attention"] * (2 + layers),
            quantization_config={"quant_method": "compressed-tensors"},
        )
        if case == "roundtrip-explicit":
            payload.update(num_nextn_predict_layers=layers, architectures=["Step5MTP"])
        (root / "config.json").write_text(json.dumps(payload))
        for config_format in ("hf", "auto"):
            draft = get_config(
                root, trust_remote_code=False, config_format=config_format
            )
            assert type(draft) is cls and draft.model_type == "step5_mtp"
            assert draft.architectures == ["Step5MTP"]
            assert draft.num_nextn_predict_layers == layers
            assert draft.layer_types == ["full_attention"] * 2
            assert draft.layer_types_with_mtp == ["full_attention"] * (2 + layers)
            assert draft.quantization_config == payload["quantization_config"]
            saved = root / config_format
            draft.save_pretrained(saved)
            for restored in (
                AutoConfig.from_pretrained(saved, trust_remote_code=False),
                pickle.loads(pickle.dumps(draft)),
            ):
                assert type(restored) is cls
                assert restored.layer_types_with_mtp == draft.layer_types_with_mtp
                assert restored.quantization_config == draft.quantization_config
                assert restored.num_nextn_predict_layers == layers
        for target_cls in (Step5Config, Step5VLConfig):
            target_path = root / target_cls.model_type
            target_cls().save_pretrained(target_path)
            target = get_config(target_path, False, config_format="auto")
            assert type(target) is target_cls
            assert target.model_type == target_cls.model_type
        for kwargs in (
            dict(num_nextn_predict_layers=-1),
            dict(max_position_embeddings=0),
            dict(num_hidden_layers=2, layer_types=["full_attention"]),
        ):
            with pytest.raises(ValueError):
                cls(**kwargs)
    owner.activate(context)
    owner.check_health()
    assert Step5Config.model_type == "step5" and Step5VLConfig.model_type == "step5_vl"


if __name__ == "__main__":
    _scenario(sys.argv[1])
    print("STEP5_CONFIG_OK=" + sys.argv[1])
