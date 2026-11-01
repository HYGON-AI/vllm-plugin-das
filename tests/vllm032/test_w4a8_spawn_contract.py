"""Real spawn/quant-config deserialization before EngineCore plugin activation.

CPU only. No vendor, architecture or oracle doubles; the child exercises the
actual oracle installer and native/fallback constructor boundaries, no kernels.
On official CPU target the GPU Triton fallback must reject CPU; that receipt
proves import/admission ordering, not that a GPU expert executed on CPU.
"""
from __future__ import annotations

import importlib
import importlib.abc
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONSUMER = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
IMPORT_STACKS = []


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, starttime=int(fields[19]))


def _cleanup_spawn_child(process, identity, *, term_s=5, kill_s=5):
    """Signal only this multiprocessing child and prove its identity disappeared."""
    terminated = killed = False
    if process.is_alive():
        process.terminate()
        terminated = True
        process.join(term_s)
    if process.is_alive():
        process.kill()
        killed = True
        process.join(kill_s)
    process.join(0)
    assert not process.is_alive() and process.exitcode is not None, (
        "owned spawn child survived bounded TERM/KILL", identity
    )
    observed = _process_identity(process.pid)
    assert observed is None or observed != identity, (
        "owned spawn identity remains after reap", identity, observed
    )
    return dict(identity=identity, terminated=terminated, killed=killed,
                exitcode=process.exitcode, all_owned_absent=True)


def _ignore_term_child(writer):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    writer.send(True)
    writer.close()
    while True:
        signal.pause()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX owned fork")
def test_spawn_cleanup_escalates_only_its_term_resistant_child():
    """Real stdlib child, no framework/provider imports or device access."""
    import multiprocessing as mp

    context = mp.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_ignore_term_child, args=(writer,))
    process.start()
    identity = _process_identity(process.pid)
    writer.close()
    try:
        assert reader.poll(5) and reader.recv() is True
    finally:
        receipt = _cleanup_spawn_child(process, identity, term_s=.1, kill_s=2)
        reader.close()
    assert identity is not None
    assert receipt["terminated"] and receipt["killed"]
    assert receipt["exitcode"] == -signal.SIGKILL


class ObserveConsumer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == CONSUMER:
            IMPORT_STACKS.append([
                dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_stack()
            ])
        return None


# The helper's main module is imported again by multiprocessing.spawn before
# argument unpickling. Install observation before quant_config's class resolves.
# Ordinary pytest collection neither imports the provider nor changes Torch.
if os.environ.get("QNT009_SPAWN_HELPER") == "1":
    sys.path.insert(0, os.environ["QNT009_SPAWN_PACKAGE_ROOT"])
    sys.meta_path.insert(0, ObserveConsumer())
    import torch

    def no_gpu(*args, **kwargs):
        raise AssertionError("spawn CPU contract queried/initialized a GPU")

    for _name in ("init", "_lazy_init", "current_device", "get_device_properties"):
        setattr(torch.cuda, _name, no_gpu)


@pytest.mark.parametrize("case", ["native", "fallback", "unsupported-fallback", "foreign-consumer"])
def test_spawn_config_does_not_freeze_int8_consumer_before_oracle(tmp_path, case):
    env = dict(os.environ)
    env.update(
        QNT009_SPAWN_HELPER="1",
        QNT009_SPAWN_PACKAGE_ROOT=os.environ.get(
            "QNT009_WHEEL_ROOT", str(ROOT / "adapter_extension/src")
        ),
        HCU_ADAPTER_HOST="none", VLLM_PLUGINS="__disabled__",
        CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("ROCR_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [sys.executable, "-B", __file__, case], env=env, text=True,
        capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(result.stdout.split("SPAWN_EVIDENCE=", 1)[1])
    destination = Path(os.environ.get("QNT009_SPAWN_EVIDENCE_DIR", str(tmp_path)))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{case}.json").write_text(json.dumps(row, indent=2) + "\n")
    assert row["start_method"] == "spawn" and row["config_roundtrip"]
    assert not row["consumer_at_child_entry"], row
    assert row["GPU"] == "not-run" and row["child_exitcode"] == 0
    assert row["cleanup"]["all_owned_absent"]
    assert row["cleanup"]["identity"] == row["spawn_identity"]
    assert "unexpected_error" not in row, row
    assert row["consumer_alias_matches_oracle"], row
    if case == "foreign-consumer":
        assert row["oracle"] == "refused" and row["oracle_unchanged"], row
    else:
        assert row["oracle"] == "installed", row
        assert row["constructor"] == case, row
        if case == "fallback" and row["official_cpu_platform"]:
            assert row["fallback_device_admission"] == "rejected-cpu", row
            assert "current device cpu" in row["fallback_device_error"], row
        assert row["consumer_after_constructor"] == (case == "fallback"), row


def _child(config, expected, case, writer):
    import torch

    row = dict(
        pid=os.getpid(),
        consumer_at_child_entry=CONSUMER in sys.modules,
        config_roundtrip=config.quant_config.config == expected,
        GPU="not-run", start_method="spawn",
        config_source_sha256=hashlib.sha256(
            Path(inspect.getfile(type(config.quant_config))).read_bytes()
        ).hexdigest(),
    )
    stat=Path('/proc/self/stat').read_text()
    row['starttime']=int(stat[stat.rfind(')')+2:].split()[19])
    try:
        # Same real module that EngineCoreProc.run_engine_core targets. Its
        # constructor invokes general plugins after these arguments unpickle.
        importlib.import_module("vllm.v1.engine.core")
        from hcu_vllm_das.moe import int8_oracle
        oracle = importlib.import_module(int8_oracle.TARGET_MODULE)
        original = (oracle.Int8MoeBackend, oracle.map_int8_backend)
        if case == "foreign-consumer":
            importlib.import_module(CONSUMER)
        try:
            int8_oracle.install()
            row["oracle"] = "installed"
        except int8_oracle.Int8OracleInstallError as error:
            row.update(oracle="refused", error=str(error))
        row["oracle_unchanged"] = (
            oracle.Int8MoeBackend is original[0]
            and oracle.map_int8_backend is original[1]
        )
        if case != "foreign-consumer" and row["oracle"] == "installed":
            from hcu_vllm_das.w4a8.config import SlimQuantW4A8MoEMethod
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation
            from vllm.model_executor.layers.fused_moe.config import (
                FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
            )
            moe = FusedMoEConfig(
                num_experts=2, experts_per_token=1, hidden_dim=128,
                intermediate_size=64, num_local_experts=2,
                num_logical_experts=2, activation=MoEActivation.SILU,
                device="cpu", in_dtype=torch.bfloat16, max_num_tokens=4,
                routing_method=RoutingMethodType.Default,
                moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            )
            if case == "native":
                selected = object()
                method = SlimQuantW4A8MoEMethod(moe, selected)
                assert method.native is selected and method.delegate is None
            elif case == "unsupported-fallback":
                from hcu_vllm_das.w4a8.runtime import W4A8Error
                moe.moe_backend = "aiter"
                with pytest.raises(W4A8Error, match="cannot be silently replaced"):
                    SlimQuantW4A8MoEMethod(moe, None)
            else:
                from vllm.platforms import current_platform

                row["official_cpu_platform"] = current_platform.is_cpu()
                if row["official_cpu_platform"]:
                    # The real official GPU expert rejects CPU. Keep that gate
                    # intact: this test is about spawn/import/alias identity.
                    with pytest.raises(ValueError, match="current device cpu") as rejection:
                        SlimQuantW4A8MoEMethod(moe, None)
                    row["fallback_device_admission"] = "rejected-cpu"
                    row["fallback_device_error"] = str(rejection.value)
                else:
                    method = SlimQuantW4A8MoEMethod(moe, None)
                    assert type(method.delegate).__name__ == "CompressedTensorsW8A8Int8MoEMethod"
                    assert method.delegate.moe.moe_backend == "triton"
                    row["fallback_device_admission"] = "constructor-completed-no-kernel"
            row["constructor"] = case
        row["consumer_after_constructor"] = CONSUMER in sys.modules
        consumer = sys.modules.get(CONSUMER)
        row["consumer_alias_matches_oracle"] = (
            consumer is None or consumer.convert_to_int8_moe_kernel_format
            is oracle.convert_to_int8_moe_kernel_format
        )
    except Exception as error:
        row.update(unexpected_error=f"{type(error).__name__}: {error}",
                   unexpected_traceback=traceback.format_exc())
    row["import_stacks"] = IMPORT_STACKS
    assert not torch.cuda.is_initialized()
    writer.send(row)
    writer.close()


def _run(case):
    import multiprocessing as mp
    import torch
    from hcu_vllm_das.w4a8.config import SlimQuantW4A8Config
    from vllm.config import DeviceConfig, VllmConfig

    checkpoint = os.environ.get("QNT009_SPAWN_CHECKPOINT")
    if checkpoint:
        expected = json.loads((Path(checkpoint) / "config.json").read_text())["compression_config"]
    else:
        expected = dict(
            quant_method="slimquant_w4a8", activation_scheme="dynamic", ignore=[],
            mixed_precision=dict(experts=dict(
                format="pack-int4-nibble", weight_bits=4, scope="per_channel",
                modules=["model.layers.*.mlp.experts.gate_up_proj",
                         "model.layers.*.mlp.experts.down_proj"],
            )),
        )
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.quant_config = SlimQuantW4A8Config.from_config(expected)
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(config, expected, case, writer))
    process.start()
    identity = _process_identity(process.pid)
    writer.close()
    destination = Path(os.environ.get("QNT009_SPAWN_EVIDENCE_DIR", "."))
    if "QNT009_SPAWN_EVIDENCE_DIR" in os.environ:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{case}-birth.json").write_text(json.dumps(identity) + "\n")
    try:
        assert identity is not None, "missing owned spawn birth identity"
        assert reader.poll(75), "spawn child failed to report"
        row = reader.recv()
        if "QNT009_SPAWN_EVIDENCE_DIR" in os.environ:
            (destination / f"{case}-reported.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
        process.join(10)
        row["child_exitcode"] = process.exitcode
        row["spawn_pid"] = process.pid
        row["spawn_identity"] = identity
        assert process.exitcode == 0, ("spawn child did not exit cleanly", row)
        assert not torch.cuda.is_initialized()
    finally:
        try:
            cleanup = _cleanup_spawn_child(process, identity)
            if "QNT009_SPAWN_EVIDENCE_DIR" in os.environ:
                (destination / f"{case}-cleanup.json").write_text(
                    json.dumps(cleanup, indent=2) + "\n"
                )
        finally:
            reader.close()
    row["cleanup"] = cleanup
    print("SPAWN_EVIDENCE=" + json.dumps(row), flush=True)


if __name__ == "__main__":
    _run(sys.argv[1])
