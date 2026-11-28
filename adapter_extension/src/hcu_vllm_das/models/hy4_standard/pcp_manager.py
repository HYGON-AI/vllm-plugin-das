# SPDX-License-Identifier: Apache-2.0
"""PCP target layout for the sealed HY4 one-token producer role."""

from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from .pd_policy import classify_standard_hy4_role


@dataclass(frozen=True)
class PCPLocalLayout:
    global_num_tokens: int
    rows: tuple[int, ...]
    valid: tuple[bool, ...]
    epoch: int

    @classmethod
    def from_segments(cls, segments, *, global_num_tokens, padded_tokens, epoch):
        rows = [0] * padded_tokens
        valid = [False] * padded_tokens
        for segment in segments:
            source, local = segment.global_batch_slice, segment.rank_local_batch_slice
            if source.stop - source.start != local.stop - local.start:
                raise ValueError("PCP source/local segment length mismatch")
            if not (
                0 <= source.start <= source.stop <= global_num_tokens
                and 0 <= local.start <= local.stop <= padded_tokens
            ):
                raise ValueError("PCP segment exceeds its actual token extent")
            rows[local.start : local.stop] = range(source.start, source.stop)
            valid[local.start : local.stop] = [True] * (local.stop - local.start)
        return cls(global_num_tokens, tuple(rows), tuple(valid), epoch)

    def select(self, value, *, token_dim=0):
        if value.shape[token_dim] != self.global_num_tokens:
            raise ValueError(
                "PCP global data does not match the current batch token extent"
            )
        shape = list(value.shape)
        shape[token_dim] = len(self.rows)
        if self.global_num_tokens == 0:
            return value.new_zeros(shape)
        index = torch.tensor(self.rows, dtype=torch.int64, device=value.device)
        selected = torch.index_select(value, token_dim, index)
        mask_shape = [1] * value.ndim
        mask_shape[token_dim] = len(self.valid)
        mask = torch.tensor(self.valid, dtype=torch.bool, device=value.device).reshape(
            mask_shape
        )
        return selected.masked_fill(~mask, 0)


class HYV4PCPProducerManager(PCPManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pcp_world_size != 16 or self.dcp_world_size != 1:
            raise ValueError("HY4 producer manager requires PCP16/DCP1")
        self.local_layout = None
        self.local_batch = None
        self._layout_epoch = 0

    @staticmethod
    def validate_config(vllm_config, supports_mm_inputs):
        if classify_standard_hy4_role(vllm_config) != "producer":
            raise ValueError("HY4 PCP manager requires the owned producer role")
        if (
            not vllm_config.use_v2_model_runner
            or not vllm_config.model_config.use_mla
            or vllm_config.model_config.is_encoder_decoder
            or vllm_config.lora_config is not None
            or vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            or vllm_config.scheduler_config.async_scheduling
            or (
                vllm_config.speculative_config is not None
                and vllm_config.speculative_config.method != "dfly"
            )
        ):
            raise ValueError(
                "HY4 PCP producer requires eager synchronous MRV2 MLA and native DFly"
            )
        from .pcp_scheduler import HYV4PCPProducerScheduler

        if (
            vllm_config.scheduler_config.get_scheduler_cls()
            is not HYV4PCPProducerScheduler
        ):
            raise ValueError(
                "HY4 PCP producer requires its owned synchronous scheduler"
            )
        # MM admission is real: the selected owned ModelState gathers global
        # positions/embeddings before selecting this exact partition's rows.

    def _build_batch_layout(self, *args, **kwargs):
        segments, counts = super()._build_batch_layout(*args, **kwargs)
        qsl = kwargs.get("query_start_loc_np", args[3] if len(args) > 3 else None)
        padded = kwargs.get("padded_num_tokens")
        if padded is None:
            padded = max(counts, default=0)
        self._layout_epoch += 1
        self.local_layout = PCPLocalLayout.from_segments(
            segments[self.pcp_rank],
            global_num_tokens=int(qsl[-1]),
            padded_tokens=padded,
            epoch=self._layout_epoch,
        )
        return segments, counts

    def partition_batch(self, input_batch, batch_desc):
        self.local_batch = super().partition_batch(input_batch, batch_desc)
        if (
            self.local_layout is None
            or len(self.local_layout.rows) != self.local_batch.num_tokens_after_padding
        ):
            raise RuntimeError("HY4 PCP live layout differs from the stock local batch")
        return self.local_batch

    def prepare_inputs_for_capture_or_profile(self, input_batch):
        self.local_layout = None
        self.local_batch = None
        return super().prepare_inputs_for_capture_or_profile(input_batch)

    def require_live_batch(self, input_batch):
        if input_batch is not self.local_batch or self.local_layout is None:
            raise RuntimeError(
                "HY4 PCP ModelState received a stale or foreign local batch"
            )
        return self.global_batch, self.local_layout
