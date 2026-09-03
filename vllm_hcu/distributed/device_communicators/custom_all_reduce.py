# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
import vllm_hcu.hcu_ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm_hcu.platforms import envs as henvs

try:
    torch.ops.hcu_ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

logger = init_logger(__name__)

# Explicit opt-in for the PCIe (no XGMI) CustomAllreduce kernel. The default
# is fail-closed to HCCL because that kernel has hard-locked HCU hosts at TP=2.
PCIE_CUSTOM_ALLREDUCE_ENV = "VLLM_HCU_ENABLE_PCIE_CUSTOM_ALLREDUCE"


def allow_custom_allreduce_for_topology(fully_connected: bool) -> bool:
    """Return whether CustomAllreduce may initialize for this device topology.

    Fully-connected (XGMI) topologies keep the existing fast path. PCIe-only
    topologies, including TP=2, fail closed to HCCL unless the operator
    explicitly opts in with ``VLLM_HCU_ENABLE_PCIE_CUSTOM_ALLREDUCE=1``.
    """
    if fully_connected:
        return True
    return bool(henvs.VLLM_HCU_ENABLE_PCIE_CUSTOM_ALLREDUCE)


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=8192 * 1024,
        symm_mem_enabled=False,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        # Native/IPC ownership must be inert before any early return or
        # fallible allocation.  ``__del__`` can run for a partially-created
        # object when communicator construction fails.
        self._ptr = 0
        self.meta_ptrs: list[int] = []
        self.buffer_ptrs: list[int] = []
        self.rank = -1

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES:
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(current_platform.device_count()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        if not allow_custom_allreduce_for_topology(fully_connected):
            # Stricter than upstream vLLM, which still allows TP=2 on PCIe.
            # HCU PCIe CustomAllreduce has hard-locked hosts at TP=2, so this
            # path fails closed to HCCL unless the operator opts in.
            logger.warning(
                "Custom allreduce is disabled because the devices are not "
                "fully connected (PCIe, no XGMI). HCCL will be used instead. "
                "To enable the PCIe custom allreduce kernel anyway, set "
                "%s=1.",
                PCIE_CUSTOM_ALLREDUCE_ENV,
            )
            return
        if not fully_connected:
            max_size = 32 * 8192 * 2
            logger.warning(
                "Custom allreduce is enabled on a PCIe topology because "
                "%s=1. This path can be unstable on some HCU PCIe systems; "
                "prefer HCCL (--disable-custom-all-reduce) if you observe "
                "hangs.",
                PCIE_CUSTOM_ALLREDUCE_ENV,
            )
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(
            torch.ops.hcu_ops.meta_size() + max_size, group=group, uncached=True
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.world_size = world_size
        self.fully_connected = fully_connected
        self._ptr = torch.ops.hcu_ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        torch.ops.hcu_ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = torch.ops.hcu_ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        torch.ops.hcu_ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL. world_size == 2 is only
        # reachable here when init enabled the communicator (XGMI, or an
        # explicit PCIe opt-in).
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            torch.ops.hcu_ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            torch.ops.hcu_ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def close(self):
        native_ptr = getattr(self, "_ptr", 0)
        meta_ptrs = getattr(self, "meta_ptrs", [])
        buffer_ptrs = getattr(self, "buffer_ptrs", [])
        rank = getattr(self, "rank", -1)
        if not native_ptr and not meta_ptrs and not buffer_ptrs:
            self.disabled = True
            return

        # Module globals may already be cleared when ``__del__`` runs during
        # interpreter finalization.  Normal worker teardown drops the
        # communicator while torch is alive and therefore still performs all
        # native releases; at finalization there is no safe Python operator
        # surface left to call.
        torch_module = globals().get("torch")
        torch_ops = getattr(torch_module, "ops", None)
        hcu_ops = getattr(torch_ops, "hcu_ops", None)
        if hcu_ops is None:
            return

        if native_ptr:
            hcu_ops.dispose(native_ptr)
            self._ptr = 0
        if rank >= 0 and meta_ptrs:
            hcu_ops.free_shared_buffer(meta_ptrs[rank])
            self.meta_ptrs = []
        if rank >= 0 and buffer_ptrs:
            hcu_ops.free_shared_buffer(buffer_ptrs[rank])
            self.buffer_ptrs = []
        self.disabled = True

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        pointer, handle = torch.ops.hcu_ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(torch.ops.hcu_ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if torch.ops.hcu_ops is not None:
            torch.ops.hcu_ops.free_shared_buffer(pointers[rank])
