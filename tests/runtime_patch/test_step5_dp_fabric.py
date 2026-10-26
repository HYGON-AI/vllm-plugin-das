# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing, fallback and lifecycle checks for Decode Fabric."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.dict(
    sys.modules, {"torch": types.SimpleNamespace(uint8="uint8", bfloat16="bf16")}
):
    spec = importlib.util.spec_from_file_location(
        "step5_fabric_subject",
        Path(__file__).parents[2]
        / "vllm_hcu/distributed/device_communicators/step5_dp_fabric.py",
    )
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)


class Tensor:
    def __init__(self, dtype, shape, contiguous=True):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
        self.contiguous = contiguous

    def numel(self):
        result = 1
        for n in self.shape:
            result *= n
        return result

    def is_contiguous(self):
        return self.contiguous


class Native:
    disabled = False

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append(name)


class Fabric:
    rank = 0
    transport = "fabric"

    def __init__(self):
        self.calls = []

    def should_custom_ag(self, *a):
        return True

    def should_custom_rs(self, *a):
        return True

    def all_gather_unreg(self, *a, **k):
        self.calls.append("gather")

    def reduce_scatter(self, *a, **k):
        self.calls.append("scatter")

    def close(self):
        self.calls.append("close")


class Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"VLLM_HCU_USE_CUSTOM_OPS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="step5")),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=4,data_parallel_size=4))
        subject.torch.cuda = types.SimpleNamespace(get_device_properties=lambda _:types.SimpleNamespace(gcnArchName="gfx938"))
        self.n = Native()
        self.f = Fabric()
        self.p = subject.Step5DPFabricProxy(self.n, self.f)

    def test_dispatch_validated_byte_payload(self):
        for size in [66624, 266496]:
            self.p.all_gather(Tensor("uint8", (size * 4,)), Tensor("uint8", (size,)))
        self.assertEqual(self.f.calls, ["gather", "gather"])
        self.assertEqual(self.n.calls, [])

    def test_other_payloads_and_noncontiguous_inputs_fall_back(self):
        for t in [
            Tensor("uint8", (64, 4096)),
            Tensor("uint8", (1234,)),
            Tensor("uint8", (266496,), False),
            Tensor("bf16", (266496,)),
        ]:
            self.p.all_gather(t, t)
        self.assertEqual(self.n.calls, ["all_gather"] * 4)
        self.assertEqual(self.f.calls, [])

    def test_grouped_or_disabled_collectives_keep_native(self):
        t = Tensor("uint8", (266496,))
        self.p.group_start()
        self.p.all_gather(t, t)
        self.p.group_end()
        self.n.disabled = True
        self.p.all_gather(t, t)
        self.assertEqual(self.f.calls, [])
        self.assertEqual(
            self.n.calls, ["group_start", "all_gather", "group_end", "all_gather"]
        )

    def test_combine_only_two_validated_shapes(self):
        for size in [64, 256]:
            self.p.reduce_scatter(
                Tensor("bf16", (size // 4, 4096)), Tensor("bf16", (size, 4096))
            )
        for t in [
            Tensor("bf16", (128, 4096)),
            Tensor("float32", (256, 4096)),
            Tensor("bf16", (256, 4096), False),
        ]:
            self.p.reduce_scatter(t, t)
        self.assertEqual(self.f.calls, ["scatter", "scatter"])
        self.assertEqual(self.n.calls, ["reduce_scatter"] * 3)

    def test_explicit_stream_keeps_native(self):
        t = Tensor("uint8", (266496,))
        self.p.all_gather(t, t, stream=object())
        t = Tensor("bf16", (256, 4096))
        self.p.reduce_scatter(t, t, stream=object())
        self.assertEqual(self.f.calls, [])
        self.assertEqual(self.n.calls, ["all_gather", "reduce_scatter"])

    def test_destroy_is_idempotent_and_other_methods_delegate(self):
        self.p.all_reduce("input")
        self.p.destroy()
        self.p.destroy()
        self.assertEqual(self.n.calls, ["all_reduce", "destroy"])
        self.assertEqual(self.f.calls, ["close"])

    def test_factory_targets_only_enabled_dp4(self):
        comm = types.SimpleNamespace(world_size=4, pynccl_comm=self.n)
        with patch.dict(os.environ, {"HCU_STEP5_DP_FABRIC": "0"}):
            subject.maybe_install(comm, None, None, "dp:0", vllm_config=self.config)
        with patch.dict(os.environ, {"HCU_STEP5_DP_FABRIC": "1"}):
            subject.maybe_install(comm, None, None, "tp:0")
            comm.world_size = 16
            subject.maybe_install(comm, None, None, "dp:0", vllm_config=self.config)
        self.assertIs(comm.pynccl_comm, self.n)

    def test_factory_rejects_conflicting_legacy_backend(self):
        comm = types.SimpleNamespace(world_size=4, _hcu_dp_packed_ag=True)
        with (
            patch.dict(
                os.environ, {"HCU_STEP5_DP_FABRIC": "1", "VLLM_HCU_DP_CUSTOM_AG": "1"}
            ),
            self.assertRaisesRegex(AssertionError, "Choose one DP dispatch backend"),
        ):
            subject.maybe_install(comm, None, None, "dp:0", vllm_config=self.config)

    def test_factory_requires_packed_dispatch(self):
        comm = types.SimpleNamespace(world_size=4, _hcu_dp_packed_ag=False)
        with (
            patch.dict(
                os.environ, {"HCU_STEP5_DP_FABRIC": "1", "VLLM_HCU_DP_CUSTOM_AG": "0"}
            ),
            self.assertRaisesRegex(AssertionError, "requires packed DP dispatch"),
        ):
            subject.maybe_install(comm, None, None, "dp:0", vllm_config=self.config)

    def test_custom_op_master_off_prevents_communicator_allocation(self):
        comm = types.SimpleNamespace(world_size=4, pynccl_comm=self.n)
        with patch.dict(
            os.environ, {"VLLM_HCU_USE_CUSTOM_OPS": "0", "HCU_STEP5_DP_FABRIC": "1"}
        ):
            subject.maybe_install(comm, None, None, "dp:0", vllm_config=self.config)
        self.assertIs(comm.pynccl_comm, self.n)


if __name__ == "__main__":
    unittest.main()
