# SPDX-License-Identifier: Apache-2.0
"""Optional Fabric fast path for the two validated uniform Step5 Decode shapes."""

from __future__ import annotations

import os

import torch

from vllm_hcu.platforms import envs as henvs
from vllm_hcu.runtime_compat.step5 import is_step5_config


class Step5DPFabricProxy:
    def __init__(self, original, fabric):
        self.original = original
        self.fabric = fabric
        self._group_depth = 0
        self._announced = set()
        self._closed = False

    @property
    def disabled(self):
        return self.original.disabled

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _announce(self, kind):
        if kind not in self._announced:
            self._announced.add(kind)
            print(
                f"[step5_dp_fabric] {kind} active: rank={self.fabric.rank} transport={self.fabric.transport}",
                flush=True,
            )

    def all_gather(self, out, inp, stream=None):
        eligible = (
            stream is None
            and not self.disabled
            and self._group_depth == 0
            and inp.dtype == torch.uint8
            and inp.ndim == 1
            and inp.numel() in (66624, 266496)
            and inp.is_contiguous()
            and out.is_contiguous()
            and self.fabric.should_custom_ag(inp, 0)
        )
        if eligible:
            self._announce("dispatch")
            return self.fabric.all_gather_unreg(inp, out=out, dim=0)
        if stream is None:
            return self.original.all_gather(out, inp)
        return self.original.all_gather(out, inp, stream=stream)

    def reduce_scatter(self, out, inp, stream=None):
        eligible = (
            stream is None
            and not self.disabled
            and self._group_depth == 0
            and inp.dtype == torch.bfloat16
            and tuple(inp.shape) in ((64, 4096), (256, 4096))
            and inp.is_contiguous()
            and out.is_contiguous()
            and self.fabric.should_custom_rs(inp, 0)
        )
        if eligible:
            self._announce("combine")
            return self.fabric.reduce_scatter(inp, out, dim=0)
        if stream is None:
            return self.original.reduce_scatter(out, inp)
        return self.original.reduce_scatter(out, inp, stream=stream)

    def group_start(self):
        self.original.group_start()
        self._group_depth += 1

    def group_end(self):
        try:
            return self.original.group_end()
        finally:
            self._group_depth -= 1

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.fabric.close()
        finally:
            self.original.destroy()


def maybe_install(communicator, cpu_group, device, unique_name, *, vllm_config=None):
    if not henvs.optional_custom_op_enabled(
        os.environ.get("HCU_STEP5_DP_FABRIC", "0") == "1"
    ):
        return
    if unique_name.split(":")[0] != "dp" or communicator.world_size != 4:
        return
    config = vllm_config
    if (config is None or not is_step5_config(config)
            or config.parallel_config.tensor_parallel_size != 4
            or config.parallel_config.data_parallel_size != 4):
        return
    architecture = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if architecture != "gfx938":
        return
    assert communicator._hcu_dp_packed_ag, "Step5 Fabric requires packed DP dispatch"
    assert os.environ.get("VLLM_HCU_DP_CUSTOM_AG", "0") != "1", (
        "Choose one DP dispatch backend"
    )
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    names = {"AITER_AR_TRANSPORT": "fabric", "AITER_AR_ENABLE_REG_CAPTURE": "0"}
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(names)
        fabric = CustomAllreduce(
            cpu_group, communicator.device, max_size=8 * 1024 * 1024
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert not fabric.disabled and fabric.transport == "fabric", (
        "Requested DP Fabric did not initialize"
    )
    communicator.pynccl_comm = Step5DPFabricProxy(communicator.pynccl_comm, fabric)
    print(
        f"[step5_dp_fabric] initialized: rank={fabric.rank} transport={fabric.transport} max_size={fabric.max_size}",
        flush=True,
    )
