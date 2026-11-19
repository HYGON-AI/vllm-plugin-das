"""Custom-SP communication through Core's public platform class seam.

Core constructs and owns its manager before any MoE execution. This does not
install model token/head layouts or change FusedMoE construction policy.
"""
import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

from .runtime_sp_policy import custom_sp_requested


class RuntimeSPCommunicator(CudaCommunicator):
    def __init__(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
        tcp_store_group=None,
        use_all2all=False,
    ):
        requested = custom_sp_requested(get_current_vllm_config_or_none())
        # Pass admission BEFORE Core builds its manager. Never mutate a runner
        # flag or construct a second manager during dispatch/Graph capture.
        use_all2all = use_all2all or (
            requested and unique_name.split(":", 1)[0] == "ep"
        )
        super().__init__(
            cpu_group=cpu_group, device=device, device_group=device_group,
            unique_name=unique_name, global_ranks=global_ranks,
            global_world_size=global_world_size, tcp_store_group=tcp_store_group,
            use_all2all=use_all2all,
        )

    def all_to_all_single(self, output, input):
        """Exchange equal split buffers over this communicator's device group."""
        if self.world_size == 1:
            output.copy_(input)
        else:
            if self.device_group is None:
                raise RuntimeError("runtime SP exchange requires its device process group")
            torch.distributed.all_to_all_single(output, input, group=self.device_group)
        return output
