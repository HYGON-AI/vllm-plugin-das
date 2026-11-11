# SPDX-License-Identifier: Apache-2.0
"""V41 private Engram tables sharded TP-major across PCP, via owned classes.

Native hashing, vocabulary validation, bucket loader, lookup and residual gate
are retained. PCP hash/head collectives stay on the eager default stream.
"""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType
import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_pcp_group, get_tensor_model_parallel_rank,
                              tensor_model_parallel_all_gather)
from vllm.models.deepseek_v41.common import engram as native
from vllm.logger import init_logger
from .engram_host import allocate_table, check_table_storage

logger = init_logger(__name__)


def validate_engram_config(cfg):
    settings = cfg.engram_config
    if settings is None:
        raise ValueError('HCU PCP Engram requires an explicit resolved Engram config')
    if (not settings.cpu_offload or settings.dp_shared_memory
            or settings.embedding_across_dp or settings.use_thp):
        raise ValueError('HCU PCP Engram requires private CPU offload, no DP sharing/across-DP or THP')
    if cfg.parallel_config.data_parallel_size != 1:
        raise ValueError('HCU PCP Engram requires DP1; DP and PCP shards cannot be mixed')


class HcuV41PcpEmbedding(native.ParallelEngramEmbedding):
    def __init__(self, *args, **kwargs):
        cfg = get_current_vllm_config()
        validate_engram_config(cfg)
        p = cfg.parallel_config.prefill_context_parallel_size
        group = get_pcp_group()
        if p <= 1 or group.world_size != p or not 0 <= group.rank_in_group < p:
            raise RuntimeError('HCU PCP Engram group/config membership mismatch')
        flags = inspect.signature(native.ParallelEngramEmbedding.__init__).bind(self, *args, **kwargs)
        flags.apply_defaults()
        if (not flags.arguments['cpu_offload'] or flags.arguments['dp_shared_memory']
                or flags.arguments['use_thp']):
            raise ValueError('HCU PCP Engram constructor flags do not select private registered storage')
        self._hcu_pcp_size, self._hcu_pcp_rank = p, group.rank_in_group
        super().__init__(*args, **kwargs)
        logger.info('HCU_V41_PCP_ENGRAM: %.2f GiB/rank, heads=%d, shards=%d, shard=%d',
                    self.part_num_embeddings * (self.dim + self.dim // self.block_size) / 1024**3,
                    self.part_n_hash_cols, self.tp_size * p,
                    get_tensor_model_parallel_rank() * p + group.rank_in_group)

    def _get_shard_info(self):
        if self.dp_size != 1:
            raise RuntimeError('HCU PCP Engram does not support Engram-DP shards')
        return (self.tp_size * self._hcu_pcp_size,
                get_tensor_model_parallel_rank() * self._hcu_pcp_size + self._hcu_pcp_rank)

    def _allocate_weights(self):
        if torch.get_default_device().type == 'meta':
            return super()._allocate_weights()
        return allocate_table(self)

    def _storage(self):
        check_table_storage(self)
        # Native op queries hipified HostGetDevicePointer and retains each CPU
        # tensor, rather than assuming host VA == device VA or making a copy.
        return super()._storage()


def _embedding_factory():
    function = native.Engram._create_embedding
    expected = 'ef2909d6afcc990d4a36677768da25e46f8f88bf779830469a143c436f42ae5c'
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if actual != expected or function.__globals__.get('ParallelEngramEmbedding') is not native.ParallelEngramEmbedding:
        raise RuntimeError(f'HCU PCP Engram embedding construction source changed: {actual}')
    namespace = dict(function.__globals__, ParallelEngramEmbedding=HcuV41PcpEmbedding)
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__, clone.__annotations__ = function.__kwdefaults__, function.__annotations__
    return clone


class HcuV41PcpEngram(native.Engram):
    # Exact native argument forwarding through a private class-name substitution,
    # rather than mirroring the embedding constructor's argument list.
    _create_embedding = _embedding_factory()

    def __init__(self, *args, **kwargs):
        bound = inspect.signature(native.Engram.__init__).bind(self, *args, **kwargs)
        if bound.arguments['use_sequence_parallel']:
            raise ValueError('HCU PCP Engram does not support sequence parallelism')
        super().__init__(*args, **kwargs)

    def _init_staging(self, max_tokens, head_dim):
        table = self.embed_tokens
        self.staged_rows = torch.empty(max_tokens * table._hcu_pcp_size,
                                       table.part_n_hash_cols, head_dim, dtype=torch.bfloat16)
        # Synchronous staging keeps eager PCP collectives ordered and avoids
        # a lookup stream retaining temporary hashes past their ownership scope.
        self._prefetch_stream = self._prefetch_done = None
        self._hcu_prepared = None

    def _group(self):
        group = get_pcp_group()
        table = self.embed_tokens
        if (group.world_size != table._hcu_pcp_size
                or group.rank_in_group != table._hcu_pcp_rank):
            raise RuntimeError('HCU PCP Engram group membership changed after construction')
        return group

    def prepare_embeddings(self, hash_ids):
        self._hcu_prepared = None
        table, group = self.embed_tokens, self._group()
        if (hash_ids.ndim != 2 or hash_ids.dtype != torch.int32
                or hash_ids.shape[1] != table.n_hash_cols):
            raise ValueError('HCU PCP Engram requires int32 [local tokens, hash heads]')
        count = hash_ids.shape[0]
        counts = group.all_gather(torch.tensor([count], dtype=torch.int64, device=hash_ids.device), dim=0)
        if counts.numel() != group.world_size or bool((counts < 0).any().item()):
            raise RuntimeError('HCU PCP Engram count collective has incompatible membership')
        slot = int(counts.max().item())
        if slot * group.world_size > self.staged_rows.shape[0]:
            raise ValueError('HCU PCP Engram hash rows exceed persistent staging capacity')
        if slot:
            padded = hash_ids.new_full((slot, table.n_hash_cols), native.DEAD_ID)
            padded[:count].copy_(hash_ids)
            gathered = group.all_gather(padded.contiguous(), dim=0)
            if gathered.shape != (slot * group.world_size, table.n_hash_cols):
                raise RuntimeError('HCU PCP Engram hash collective returned an invalid row layout')
            super().prepare_embeddings(gathered)
        self._hcu_prepared = (count, slot)

    def embed(self, hash_ids):
        state, self._hcu_prepared = self._hcu_prepared, None
        if state is None:
            raise RuntimeError('HCU PCP Engram embed requires successful prepare for this step')
        count, slot = state
        if hash_ids.ndim != 2 or tuple(hash_ids.shape) != (count, self.embed_tokens.n_hash_cols):
            raise ValueError('HCU PCP Engram local row/head extent changed after prepare')
        table, group = self.embed_tokens, self._group()
        if not slot:
            return self.staged_rows.new_empty((0, table.n_hash_cols, table.dim))
        staged = self._ready_rows(slot * group.world_size)
        rows = group.all_gather(staged.contiguous(), dim=1)
        expected = (slot * group.world_size, table.part_n_hash_cols * group.world_size, table.dim)
        if tuple(rows.shape) != expected:
            raise RuntimeError('HCU PCP Engram head collective returned an invalid layout')
        start = group.rank_in_group * slot
        rows = rows[start:start + count]
        if table.tp_size > 1:
            rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, :table.n_hash_cols]
