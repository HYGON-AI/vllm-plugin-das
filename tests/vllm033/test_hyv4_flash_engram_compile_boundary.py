# SPDX-License-Identifier: Apache-2.0
"""Default installed-CPU coverage for Engram's live compilation boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash"
MASKS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
         "GPU_DEVICE_ORDINAL", "NVIDIA_VISIBLE_DEVICES")
OP = "hcu_hyv4_flash_engram_hashes"


class NativeEngramBoundaryContracts(unittest.TestCase):
    def test_profile_compile_replay_uses_live_owner_and_metadata(self):
        env = dict(os.environ, HCU_ADAPTER_HOST="none", VLLM_PLUGINS="__disabled__",
                   PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        env.update({key: "-1" for key in MASKS})
        env["HCU_ENGRAM_CPU_BOUNDARY_CHILD"] = "1"
        run = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                              "--native-cpu-child"], env=env, capture_output=True,
                             text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        lines = [line for line in run.stdout.splitlines()
                 if line.startswith("ENGRAM_CPU_RESULT=")]
        self.assertEqual(len(lines), 1, run.stdout)
        result = json.loads(lines[0].split("=", 1)[1])
        self.assertEqual(result["graphs"], 1)
        self.assertFalse(result["cuda_initialized"])
        self.assertEqual(result["live_replays"], 3)
        print(lines[0])


def _native_cpu_child():
    assert os.environ.get("HCU_ENGRAM_CPU_BOUNDARY_CHILD") == "1"
    assert all(os.environ.get(key) == "-1" for key in MASKS)
    import gc
    from types import SimpleNamespace as NS
    from unittest.mock import patch
    from weakref import ref
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from vllm.config import CompilationConfig, CompilationMode
    import hcu_vllm_das.models.hy_v4_flash.engram as production

    assert Path(production.__file__).read_bytes() == (MODEL / "engram.py").read_bytes()
    assert not torch.cuda.is_initialized()

    class LayerDouble:
        pass

    class HashArithmeticWitness(production.FlashNgramHashState):
        """Only the inner hash arithmetic is doubled; wrapper/op are production."""
        def __init__(self, bias=0):
            torch.nn.Module.__init__(self)
            self.bias = bias

        def prepare_hashes(self, ids, lookback, metadata):
            if metadata is None:
                return ids.new_full((1, ids.numel(), 1), -1, dtype=torch.int32)
            return (ids + metadata["bias"] + self.bias + lookback[0, 0]).reshape(1, -1, 1).int()

    owner = HashArithmeticWitness()
    layer = LayerDouble()
    # Explicit Core registry/context doubles. Same fixture works on the old
    # wrapper so its RED is the frozen profile value, not a missing new API.
    layer._hcu_flash_engram_hash_owner = ref(owner)
    owner._engram_attention_owner_ref = ref(layer)
    ctx = NS(attn_metadata=None, no_compile_layers={"probe.attn": layer})
    operation = getattr(torch.ops.vllm, OP, None)
    cpu_library = None
    if operation is not None:
        assert not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::" + OP, "CPU")
        cpu_library = torch.library.Library("vllm", "IMPL", "CPU")
        cpu_library.impl(OP, production._flash_engram_hashes_impl)

    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    lookback = torch.zeros((1, 2), dtype=torch.int32)
    out = torch.empty((1, 4, 1), dtype=torch.int32)
    graphs, frames = [], []

    def backend(graph, example_inputs):
        graphs.append(graph)
        def execute(*args):
            frames.append(args)
            return graph.forward(*args)
        return execute

    def invoke(ids, lookback, out):
        owner.prepare_hashes_with_output(ids, lookback, "probe.attn", out)
        return out

    with patch.object(production, "get_forward_context", lambda: ctx):
        compiled = torch.compile(invoke, backend=backend, fullgraph=True)
        compiled(ids, lookback, out)
        assert len(graphs) == 1 and frames
        saved = frames[-1]
        owners = [owner]
        for bias, owner_bias in [(17, 0), (23, 0), (5, 100)]:
            if owner_bias:
                current = HashArithmeticWitness(owner_bias)
                other_layer = LayerDouble()
                other_layer._hcu_flash_engram_hash_owner = ref(current)
                current._engram_attention_owner_ref = ref(other_layer)
                ctx.no_compile_layers = {"probe.attn": other_layer}
                owners.append(current)
            else:
                current = owner
            ctx.attn_metadata = {"probe.attn": {"bias": bias}}
            ids.add_(5)
            lookback.add_(2)
            expected = current.prepare_hashes(ids, lookback, ctx.attn_metadata["probe.attn"])
            out.fill_(-99)
            # Core executes a cached compiled graph directly; do not let Dynamo
            # recompile against the changed Python context and hide the defect.
            graphs[0](*saved)
            assert torch.equal(out, expected), (
                "Engram compiled replay froze profile metadata instead of reading live context",
                out.flatten().tolist(), expected.flatten().tolist(),
            )

        assert operation is not None
        schema = operation.default._schema
        assert [a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write] == ["output"]
        assert not schema.returns
        with patch.object(production, "get_forward_context",
                          side_effect=AssertionError("Meta kernel read runtime context")):
            with FakeTensorMode() as mode:
                operation(mode.from_tensor(ids), mode.from_tensor(lookback),
                          "probe.attn", mode.from_tensor(out))

        for mode in (CompilationMode.NONE, CompilationMode.VLLM_COMPILE):
            config = CompilationConfig(mode=mode)
            preserved = ["custom::existing_split"]
            config.splitting_ops = preserved.copy()
            vcfg = NS(compilation_config=config, parallel_config=NS(
                all2all_backend="allgather_reducescatter", data_parallel_size=1))
            owner.bind_attention_owner(layer, vcfg)
            owner.bind_attention_owner(layer, vcfg)
            assert config.splitting_ops[:1] == preserved
            assert config.splitting_ops.count("vllm::" + OP) == (mode == CompilationMode.VLLM_COMPILE)
            assert not list(owner.named_parameters())

        default = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
        default.splitting_ops = None
        owner.bind_attention_owner(layer, NS(compilation_config=default, parallel_config=NS(
            all2all_backend="allgather_reducescatter", data_parallel_size=1)))
        assert default.splitting_ops and default.splitting_ops.count("vllm::" + OP) == 1

        foreign = HashArithmeticWitness()
        try:
            foreign.bind_attention_owner(layer, NS(compilation_config=default))
        except RuntimeError:
            pass
        else:
            raise AssertionError("foreign hash owner replaced a live binding")

        dead_layer = LayerDouble()
        dead = HashArithmeticWitness()
        dead.bind_attention_owner(dead_layer, NS(compilation_config=default))
        ctx.no_compile_layers = {"probe.attn": dead_layer}
        del dead
        gc.collect()
        try:
            operation(ids, lookback, "probe.attn", out)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dead hash owner accepted")
    assert not torch.cuda.is_initialized()
    print("ENGRAM_CPU_RESULT=" + json.dumps(dict(
        graphs=len(graphs), live_replays=3, cuda_initialized=False,
        scope="actual production CUDA/Meta registration and FX replay; CPU inner hash/context doubles",
    )), flush=True)


if __name__ == "__main__" and "--native-cpu-child" in sys.argv:
    _native_cpu_child()
