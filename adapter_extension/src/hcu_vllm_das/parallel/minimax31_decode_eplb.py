# SPDX-License-Identifier: Apache-2.0
"""Consumer-owned dynamic expert weights for MiniMax strict decode IPC."""
from dataclasses import dataclass
import hashlib
import json
import time

import torch
import torch.distributed as dist
from torch import nn


_WEIGHT_NAMES = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


@dataclass
class DecodeEplbCounters:
    enabled: torch.Tensor
    counts: torch.Tensor


class DecodeEplb:
    """Collect logical routes in graph and migrate at an explicit step boundary."""

    @torch.inference_mode()
    def __init__(self, model, ep, *, window, interval):
        from .minimax31_decode import PermutedDecodeExperts
        from vllm.distributed.eplb.eplb_communicator import TorchDistNcclEplbCommunicator

        if not 1 <= window <= interval or ep.world_size != 8:
            raise ValueError("decode EPLB requires EP8 and 1 <= window <= interval")
        layers = sorted((layer for layer in model.modules()
                         if isinstance(layer, PermutedDecodeExperts)),
                        key=lambda layer: layer.decode_layer_id)
        if [layer.decode_layer_id for layer in layers] != list(range(3, 60)):
            raise ValueError("decode EPLB requires all 57 permuted MiniMax layers")
        self.ep = ep
        self.window = window
        self.interval = interval
        self.weights = []
        self.tables = []
        self.generation = 0
        self.receipts = []
        self.active = False
        self.poisoned = False
        self.device = layers[0].w13_weight.device
        self.step_number = 0
        self.next_update = window
        self.enabled = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.counts = torch.zeros((len(layers), 128), dtype=torch.int32, device=self.device)
        self.table = torch.tensor([layer.decode_permutation for layer in layers],
                                  dtype=torch.int64, device=self.device)
        self.placement = self.table.argsort(dim=1).cpu()
        bytes_copied = 0
        for i, layer in enumerate(layers):
            if getattr(layer, "_minimax31_expert_layout", None) != "signed-high-nibble-first-hipc-split-v1":
                raise ValueError("decode EPLB requires the HIPC packed expert layout")
            tensors = []
            for name in _WEIGHT_NAMES:
                source = getattr(layer, name)
                if source.device != self.device or source.shape[0] != 16 or not source.is_contiguous():
                    raise ValueError("decode EPLB requires 16 contiguous experts per rank")
                private = nn.Parameter(source.detach().clone(), requires_grad=False)
                if private.data_ptr() == source.data_ptr():
                    raise RuntimeError("decode EPLB could not isolate shared expert weights")
                setattr(layer, name, private)
                bytes_copied += private.nbytes
                tensors.append(private)
            layer.quant_method.moe_quant_config = layer.quant_method.get_fused_moe_quant_config(layer)
            layer.decode_gpu_permutation = self.table[i]
            layer.decode_eplb_counters = DecodeEplbCounters(self.enabled, self.counts[i])
            self.weights.append(tensors)
        self.buffer = [torch.empty_like(weight) for weight in self.weights[0]]
        self.communicator = TorchDistNcclEplbCommunicator(ep.device_group)
        torch.cuda.synchronize(self.device)
        self.private_weight_bytes = bytes_copied
        self.migration_buffer_bytes = sum(t.nbytes for t in self.buffer)

    @torch.inference_mode()
    def start_step(self):
        if self.poisoned:
            raise RuntimeError("EPLB migration failed; restart the consumer")
        collect = self.step_number >= self.next_update - self.window
        if collect != self.active:
            self.enabled.fill_(int(collect))
            self.active = collect

    def finish_step(self):
        """Call once per completed real VERIFY, on every EP rank outside capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("decode EPLB migration must run outside graph capture")
        self.step_number += 1
        if self.step_number != self.next_update:
            return
        self.next_update += self.interval
        try:
            self._rebalance()
        except BaseException:
            self.poisoned = True
            raise

    @torch.inference_mode()
    def pause(self):
        """Disable route collection before profiling or running warmups."""
        self.enabled.zero_()
        self.active = False

    @torch.inference_mode()
    def reset_window(self):
        """Start a fresh request sequence on the resident learned placement."""
        self.pause()
        self.counts.zero_()
        self.step_number = 0
        self.next_update = self.window
        self.receipts.clear()

    @torch.inference_mode()
    def _rebalance(self):
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
        from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace

        self.pause()
        begin = time.perf_counter()
        load = self.counts.to(device="cpu", dtype=torch.int64)
        self.counts.zero_()
        dist.all_reduce(load, group=self.ep.cpu_group)
        if (load.sum(dim=1) <= 0).any():
            raise RuntimeError("EPLB window did not collect every target MoE layer")
        old = self.placement
        new = DefaultEplbPolicy.rebalance_experts(load, 128, 1, 1, 8, old_global_expert_indices=old)
        if not torch.equal(new.sort(dim=1).values, torch.arange(128).expand_as(old)):
            raise RuntimeError("EPLB policy returned an invalid 128-slot mapping")
        moved = int((old.argsort(dim=1) // 16 != new.argsort(dim=1) // 16).sum())
        torch.cuda.synchronize(self.device)
        transfer = time.perf_counter()
        if not torch.equal(old, new):
            rearrange_expert_weights_inplace(old, new, self.weights, self.buffer,
                                              self.ep.device_group, self.communicator)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.ep.cpu_group)
            self.table.copy_(new.argsort(dim=1))
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.ep.cpu_group)
            self.placement = new
            self.generation += 1
        done = time.perf_counter()

        def mean_max(p2l):
            sums = load.gather(1, p2l).reshape(-1, 8, 16).sum(dim=2)
            return float(sums.max(dim=1).values.double().mean()) / self.window

        receipt = {
            "step": self.step_number,
            "generation": self.generation,
            "experts_moved_across_ranks": moved,
            "mean_layer_max_rows_before": mean_max(old),
            "mean_layer_max_rows_after": mean_max(new),
            "window_rows": int(load.sum()),
            "map_sha256": hashlib.sha256(new.numpy().tobytes()).hexdigest(),
            "migration_ms": (done - transfer) * 1000,
            "rebalance_ms": (done - begin) * 1000,
        }
        self.receipts.append(receipt)
        print("MINIMAX_DECODE_EPLB " + json.dumps(receipt), flush=True)


def attach_decode_eplb(model, *, window=16, interval=96):
    from vllm.distributed.parallel_state import get_ep_group

    state = DecodeEplb(model, get_ep_group(), window=window, interval=interval)
    model._minimax31_decode_eplb = state
    print("MINIMAX_DECODE_EPLB_PRIVATE " + json.dumps({
        "rank": state.ep.rank_in_group, "window": window, "interval": interval,
        "private_weight_bytes": state.private_weight_bytes,
        "migration_buffer_bytes": state.migration_buffer_bytes,
    }), flush=True)
