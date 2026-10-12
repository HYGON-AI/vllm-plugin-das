# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DecodeBench FP8/startup-fill backport with native packed-cache writers.

Adapted from vLLM PR #58472 (7a9406690ebe2fc47236a81cf7ec33188c6565e2).
The factory installs this connector under the existing DecodeBenchConnector
name. Synthetic history is for performance measurements, not model accuracy.
"""

from __future__ import annotations

import math
import zlib

import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector as UpstreamConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorScheduler as UpstreamScheduler,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode, MLAAttentionSpec

logger = init_logger(__name__)


def layer_spec(group, name):
    return getattr(group.kv_cache_spec, "kv_cache_specs", {}).get(
        name, group.kv_cache_spec
    )


def startup_groups(config):
    """Do not startup-fill groups that the engine subsequently zeroes."""
    groups = set()
    for index, group in enumerate(config.kv_cache_groups):
        specs = [layer_spec(group, name) for name in group.layer_names]
        if any(type(spec).__name__ == "CircularBufferSpec" for spec in specs):
            continue
        if config.needs_kv_cache_zeroing and any(
            isinstance(spec, AttentionSpec) for spec in specs
        ):
            continue
        groups.add(index)
    return groups


def plain_fp8_dtype(spec, cache_dtype):
    if (
        not isinstance(spec, AttentionSpec)
        or spec.dtype != torch.uint8
        or spec.kv_quant_mode != KVQuantMode.FP8_PER_TENSOR
        or getattr(spec, "state_content_bytes", None) is not None
    ):
        return None
    cache_dtype = getattr(spec, "cache_dtype_str", None) or cache_dtype
    if cache_dtype in ("fp8", "fp8_e4m3"):
        return current_platform.fp8_dtype()
    if cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    return None


def make_fill_values(shape, device, dtype, mean, std, generator=None):
    """Sample supported dtypes and convert to the requested representation."""
    if not dtype.is_floating_point:
        return torch.zeros(shape, device=device, dtype=dtype)
    finfo = torch.finfo(dtype)
    if std > 0:
        values = torch.normal(
            mean,
            std,
            size=shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        return values.clamp_(finfo.min, finfo.max).to(dtype)
    return torch.full(
        shape, min(max(mean, finfo.min), finfo.max), device=device, dtype=dtype
    )


def write_mla_cache(values, rope, cache, slots):
    # Use exactly the HCU SparseMLAAttentionImpl cache-update dispatch.
    import vllm_hcu.hcu_ops  # noqa: F401

    torch.ops.hcu_ops.concat_and_cache_mla(
        values,
        rope,
        cache,
        slots,
        "fp8_ds_mla",
        torch.ones((), device=cache.device),
    )


class DecodeBenchConnectorScheduler(UpstreamScheduler):
    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config)
        self.kv_cache_config = kv_cache_config
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.startup_fill = vllm_config.kv_transfer_config.get_from_extra_config(
            "startup_fill", False
        )
        self.startup_group_ids = (
            startup_groups(kv_cache_config) if self.startup_fill else set()
        )

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens == 0:
            return
        start = request.num_computed_tokens
        end = start + num_external_tokens
        selected = []
        for index, (group, ids) in enumerate(
            zip(
                self.kv_cache_config.kv_cache_groups,
                blocks.get_block_ids(),
                strict=True,
            )
        ):
            if index in self.startup_group_ids:
                selected.append([])
                continue
            spec = group.kv_cache_spec
            if type(spec).__name__ == "CircularBufferSpec":
                selected.append(list(ids))
                continue
            # Local DCP tables: a physical attention block spans B * DCP
            # global positions. This also applies to the indexer K cache.
            block_size = spec.block_size
            if isinstance(spec, AttentionSpec):
                block_size *= self.dcp_size
            selected.append(
                list(ids[start // block_size : (end + block_size - 1) // block_size])
            )
        self._filled_requests.add(request.request_id)
        if any(selected):
            self._pending_fills[request.request_id] = (
                tuple(selected),
                num_external_tokens,
            )


class DecodeBenchConnectorWorker:
    def __init__(self, vllm_config, kv_cache_config):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        extra = vllm_config.kv_transfer_config
        self.fill_mean = float(extra.get_from_extra_config("fill_mean", 0.015))
        self.fill_std = float(extra.get_from_extra_config("fill_std", 0.0))
        if (
            not math.isfinite(self.fill_mean)
            or not math.isfinite(self.fill_std)
            or self.fill_std < 0
        ):
            raise ValueError(
                "DecodeBench fill_mean/fill_std must be finite; fill_std >= 0"
            )
        self.fill_seed = int(extra.get_from_extra_config("fill_seed", 0))
        self.startup_fill = extra.get_from_extra_config("startup_fill", False)
        self.chunk_tokens = int(extra.get_from_extra_config("fill_chunk_tokens", 16384))
        if self.chunk_tokens <= 0:
            raise ValueError("DecodeBench fill_chunk_tokens must be positive")
        self.startup_group_ids = (
            startup_groups(kv_cache_config) if self.startup_fill else set()
        )
        self.group_to_layers = {
            i: group.layer_names
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        self.specs = {
            name: layer_spec(group, name)
            for group in kv_cache_config.kv_cache_groups
            for name in group.layer_names
        }
        self.layers = vllm_config.compilation_config.static_forward_context
        self.kv_caches = {}
        self.generators = {}
        self.layouts = {}

    def _generator(self, name, device):
        if name not in self.generators:
            from vllm.distributed import get_dcp_group

            dcp_rank = get_dcp_group().rank_in_group
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            seed = (
                self.fill_seed
                + zlib.crc32(name.encode())
                + 1000003 * dp_rank
                + 9176 * dcp_rank
            ) % (2**63 - 1)
            self.generators[name] = torch.Generator(device=device).manual_seed(seed)
        return self.generators[name]

    def _layout(self, name, cache):
        spec = self.specs[name]
        layer = self.layers.get(name)
        if isinstance(spec, MLAAttentionSpec) and cache.dtype == torch.uint8:
            dtype = getattr(spec, "cache_dtype_str", None) or getattr(
                layer, "kv_cache_dtype", None
            )
            if dtype == "fp8_ds_mla":
                if (
                    cache.ndim != 3
                    or cache.shape[-1] != 656
                    or getattr(spec, "model_version", None) == "deepseek_v4"
                ):
                    raise ValueError(
                        f"Unsupported packed MLA geometry for {name}: {tuple(cache.shape)}"
                    )
                return "ds_mla", torch.float8_e4m3fn
            if (
                layer is not None
                and layer.get_attn_backend().get_name() == "DEEPSEEK_V32_INDEXER"
            ):
                if cache.ndim != 3 or cache.shape[-1] != 132:
                    raise ValueError(
                        f"Unsupported packed indexer geometry for {name}: {tuple(cache.shape)}"
                    )
                return "indexer", current_platform.fp8_dtype()
        dtype = (
            plain_fp8_dtype(spec, self.vllm_config.cache_config.cache_dtype)
            or cache.dtype
        )
        if not dtype.is_floating_point:
            logger.warning_once(
                "DecodeBench: unsupported packed cache %s is zero-filled", name
            )
        return "plain", dtype

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
        for name, cache in kv_caches.items():
            if isinstance(cache, torch.Tensor):
                self.layouts[name] = self._layout(name, cache)
                logger.info(
                    "DecodeBench fill layout: %s = %s, shape=%s",
                    name,
                    self.layouts[name][0],
                    tuple(cache.shape),
                )
        if self.startup_fill:
            for group in self.startup_group_ids:
                for name in self.group_to_layers[group]:
                    self._fill_layer(name, None)
            logger.info(
                "DecodeBench startup fill complete: mean=%s std=%s",
                self.fill_mean,
                self.fill_std,
            )

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata):
        for groups, _ in metadata.reqs_to_fill.values():
            for group, ids in enumerate(groups):
                if ids:
                    for name in self.group_to_layers[group]:
                        self._fill_layer(name, ids)

    def _fill_layer(self, name, block_ids):
        cache = self.kv_caches[name]
        circular = type(self.specs[name]).__name__ == "CircularBufferSpec"
        mean, std = (0.0, 0.0) if circular else (self.fill_mean, self.fill_std)
        if isinstance(cache, (list, tuple)):
            for state in cache:
                rows = max(1, (1 << 22) // max(1, state[0].numel()))
                for chunk in state.split(rows):
                    chunk.copy_(
                        make_fill_values(
                            chunk.shape,
                            chunk.device,
                            chunk.dtype,
                            mean,
                            std,
                            self._generator(name, chunk.device),
                        )
                    )
            return
        layout, dtype = self.layouts[name]
        generator = self._generator(name, cache.device)
        ids = (
            range(cache.shape[0])
            if block_ids is None
            else sorted({i for i in block_ids if 0 <= i < cache.shape[0]})
        )
        block_size = cache.shape[1]
        blocks_per_chunk = max(
            1,
            min(self.chunk_tokens // block_size, (1 << 22) // max(1, cache[0].numel())),
        )
        for start in range(0, len(ids), blocks_per_chunk):
            chunk_ids = torch.tensor(
                ids[start : start + blocks_per_chunk],
                device=cache.device,
                dtype=torch.long,
            )
            if layout == "plain":
                values = make_fill_values(
                    (len(chunk_ids),) + tuple(cache.shape[1:]),
                    cache.device,
                    dtype,
                    mean,
                    std,
                    generator,
                )
                cache[chunk_ids] = values.view(cache.dtype)
                continue
            from vllm import _custom_ops as ops

            slots = (
                chunk_ids[:, None] * block_size
                + torch.arange(block_size, device=cache.device)
            ).reshape(-1)
            dim = 512 if layout == "ds_mla" else 128
            values = make_fill_values(
                (len(slots), dim), cache.device, torch.bfloat16, mean, std, generator
            )
            if layout == "ds_mla":
                rope = make_fill_values(
                    (len(slots), 64), cache.device, torch.bfloat16, mean, std, generator
                )
                write_mla_cache(values, rope, cache, slots)
            else:
                # Native writer handles HCU shuffled payload + block-tail
                # FP32 scales, including UE8M0 scale rounding.
                ops.indexer_k_quant_and_cache(values, cache, slots, 128, "ue8m0")


class DecodeBenchConnector(UpstreamConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = DecodeBenchConnectorScheduler(
                vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = DecodeBenchConnectorWorker(
                vllm_config, kv_cache_config
            )
