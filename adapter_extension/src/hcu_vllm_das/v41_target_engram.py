# SPDX-License-Identifier: Apache-2.0
"""Private registered host storage for native V41 PCP1 target Engram.

Keep native TP/Engram-DP sharding, hash gather, local token/head restoration,
checkpoint loading and gate forward. Only allocation and eager stream policy
are owned; no PCP coordinate is introduced into this DP path.
"""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType
import torch
from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v41.common import engram as native
from vllm.logger import init_logger
from .engram_host import allocate_table, check_table_storage

logger = init_logger(__name__)


def validate_host_config(cfg):
    p, settings = cfg.parallel_config, cfg.engram_config
    if settings is None or not settings.cpu_offload or settings.dp_shared_memory or settings.use_thp:
        raise ValueError('HCU target Engram host storage requires private CPU offload without shared memory or THP')
    if (p.prefill_context_parallel_size != 1 or p.decode_context_parallel_size != 1
            or p.pipeline_parallel_size != 1):
        raise ValueError('HCU target Engram host route requires PCP1/DCP1/PP1')
    if not cfg.use_v2_model_runner or not cfg.model_config.enforce_eager:
        raise ValueError('HCU target Engram host route requires eager MRV2; Graph not validated')
    if cfg.lora_config is not None or p.enable_eplb or p.enable_elastic_ep or p.enable_dbo:
        raise ValueError('HCU target Engram host route does not support LoRA/EPLB/ElasticEP/DBO')


def use_host_target(cfg):
    from .v41_pcp_activation import host_enabled
    if not host_enabled() or not getattr(cfg.model_config.hf_config, 'engram_layer_ids', None):
        return False
    if cfg.parallel_config.prefill_context_parallel_size != 1:
        return False
    validate_host_config(cfg)
    return True


class HcuV41HostEmbedding(native.ParallelEngramEmbedding):
    def __init__(self, *args, **kwargs):
        cfg = get_current_vllm_config()
        validate_host_config(cfg)
        bound = inspect.signature(native.ParallelEngramEmbedding.__init__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        if (not bound.arguments['cpu_offload'] or bound.arguments['dp_shared_memory']
                or bound.arguments['use_thp']):
            raise ValueError('HCU target Engram constructor does not select private registered host storage')
        super().__init__(*args, **kwargs)
        logger.info('HCU_V41_TARGET_ENGRAM_HOST: %.2f GiB/rank, TP=%d, EDP=%d, heads=%d',
                    self.part_num_embeddings * (self.dim + self.dim // self.block_size) / 1024**3,
                    self.tp_size, self.dp_size, self.part_n_hash_cols)

    def _allocate_weights(self):
        if torch.get_default_device().type == 'meta':
            return super()._allocate_weights()
        return allocate_table(self)

    def _storage(self):
        check_table_storage(self)
        return super()._storage()


def _factory():
    function = native.Engram._create_embedding
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if (actual != 'ef2909d6afcc990d4a36677768da25e46f8f88bf779830469a143c436f42ae5c'
            or function.__globals__.get('ParallelEngramEmbedding') is not native.ParallelEngramEmbedding):
        raise RuntimeError(f'HCU target Engram constructor source changed: {actual}')
    namespace = dict(function.__globals__, ParallelEngramEmbedding=HcuV41HostEmbedding)
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__, clone.__annotations__ = function.__kwdefaults__, function.__annotations__
    return clone


class HcuV41HostEngram(native.Engram):
    # Reuse native argument forwarding by a private class-name substitution;
    # do not copy the embedding parameter list or change Core globals.
    _create_embedding = _factory()

    def _init_staging(self, max_tokens, head_dim):
        # Eager target DP metadata drives native gather/restore. Keep lookup on
        # the calling stream; native asynchronous prefetch is not required.
        self.staged_rows = torch.empty(max_tokens * self.embed_tokens.dp_size,
            self.embed_tokens.part_n_hash_cols, head_dim, dtype=torch.bfloat16)
        self._prefetch_stream = self._prefetch_done = None
