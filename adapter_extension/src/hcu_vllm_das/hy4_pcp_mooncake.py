# SPDX-License-Identifier: Apache-2.0
"""PCP replica ownership for Core's direct Mooncake connector.

Select through kv_connector_module_path; the native connector stays unchanged.
Core PCP gathers complete MLA/indexer cache replicas. Only PCP rank zero may
publish a sender for the TP coordinate. All workers still report completion,
so Core's world-size aggregator retains KV until the real sender acknowledges.
"""
from vllm.distributed.parallel_state import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as native

from hcu_adapter.infra.native_namespace import bind_native


def _validate(config):
    if getattr(config.model_config.hf_config, 'model_type', None) != 'hy_v4':
        raise ValueError('HY4 PCP Mooncake requires the HY4 text model')
    parallel = config.parallel_config
    if parallel.prefill_context_parallel_size > 1:
        if any(getattr(parallel, name) != 1 for name in (
                'tensor_parallel_size', 'pipeline_parallel_size',
                'data_parallel_size', 'decode_context_parallel_size')):
            raise ValueError('HY4 PCP Mooncake producer requires TP1/PP1/DP1/DCP1')
        if config.kv_transfer_config.kv_role != 'kv_producer':
            raise ValueError('HY4 PCP Mooncake supports a PCP producer and non-PCP consumer')


class HY4PCPMooncakeWorker(native.MooncakeConnectorWorker):
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        self._native_entered = False
        _validate(vllm_config)
        parallel = vllm_config.parallel_config
        self._canonical_pcp_sender = True
        self._replica_pending = {}
        if parallel.prefill_context_parallel_size > 1:
            group = get_pcp_group()
            if group.world_size != parallel.prefill_context_parallel_size:
                raise ValueError('HY4 PCP Mooncake group size disagrees with configuration')
            self._canonical_pcp_sender = group.rank_in_group == 0
        self._native_entered = True
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def __del__(self):
        # Validation may reject before Core creates any engine resources.
        if getattr(self, '_native_entered', False):
            super().__del__()

    async def _mooncake_sender_listener(self, ready_event):
        if not self._canonical_pcp_sender:
            ready_event.set()
            return
        return await super()._mooncake_sender_listener(ready_event)

    async def record_send_reqs(self, metadata):
        if self._canonical_pcp_sender:
            return await super().record_send_reqs(metadata)
        for request_id, (transfer_id, blocks) in metadata.reqs_to_send.items():
            if blocks:
                self._replica_pending.pop(transfer_id, None)
                self.finished_sending_reqs.add(request_id)
            else:
                self._replica_pending[transfer_id] = request_id
        for transfer_id in metadata.reqs_not_processed:
            # Native request_finished already frees aborted requests. Matching
            # its cancellation path must not emit a replica-only completion.
            self._replica_pending.pop(transfer_id, None)


class HY4PCPMooncakeConnector(native.MooncakeConnector):
    # Reuse the complete native constructor and every scheduler/transport
    # method, substituting just this worker class in a private namespace.
    __init__ = bind_native(native.MooncakeConnector.__init__,
                          {'MooncakeConnectorWorker': HY4PCPMooncakeWorker})
