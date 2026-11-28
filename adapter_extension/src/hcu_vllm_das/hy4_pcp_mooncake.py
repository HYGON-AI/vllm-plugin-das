# SPDX-License-Identifier: Apache-2.0
"""PCP replica ownership for Core's direct Mooncake connector.

Select through kv_connector_module_path; the native connector stays unchanged.
Core PCP gathers complete MLA/indexer cache replicas. Only PCP rank zero may
publish a sender for the TP coordinate. All workers still report completion,
so Core's world-size aggregator retains KV until the real sender acknowledges.
"""
from hcu_adapter.infra.kill_switch import op_disabled
from hcu_adapter.infra.native_namespace import bind_native
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector as native,
)
from vllm.distributed.parallel_state import get_pcp_group

from .models.hy4_standard.pd_policy import (
    classify_standard_hy4_role,
    committed_remote_prompt_tokens,
    is_standard_vl_target,
    requires_last_prompt_replay,
)


def _validate(config):
    if is_standard_vl_target(config.model_config.hf_config):
        classify_standard_hy4_role(config)
        return
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


class HY4PCPMooncakeScheduler(native.MooncakeConnectorScheduler):
    """Keep native leases/ACKs, replay the last real prompt token for DFly.

    The proxy discards P's sampled token and D receives the original prompt.
    KV alone does not reconstruct the target hidden state used to start DFly:
    commit only N-1 prompt tokens and let D execute token N itself. P still
    computes the complete original prompt; no Mamba-style request truncation
    or synthetic hidden state is involved.
    """

    def __init__(self, vllm_config, engine_id, kv_cache_config):
        _validate(vllm_config)
        self._standard_dfly_replay = requires_last_prompt_replay(vllm_config)
        super().__init__(vllm_config, engine_id, kv_cache_config)
        if self._standard_dfly_replay and (self._has_mamba or self._bounded_replay):
            raise ValueError('Standard VL DFly PD requires full attention KV without bounded replay')

    def _get_remote_prefill_token_count(self, num_prompt_tokens):
        if self._standard_dfly_replay:
            return committed_remote_prompt_tokens(num_prompt_tokens)
        return super()._get_remote_prefill_token_count(num_prompt_tokens)


def _should_launch_bootstrap_server(config):
    if not native.should_launch_bootstrap_server(config):
        return False
    return (config.parallel_config.prefill_context_parallel_size == 1
            or get_pcp_group().rank_in_group == 0)


def _init_native(worker, config, engine_id, cache_config):
    # Bootstrap starts in Core's constructor, before the sender listener.
    # Keep its full body and all TP/PP/DP/external-bootstrap ownership rules.
    initialize = bind_native(native.MooncakeConnectorWorker.__init__,
        {'should_launch_bootstrap_server': _should_launch_bootstrap_server})
    initialize(worker, config, engine_id, cache_config)


class HY4PCPMooncakeWorker(native.MooncakeConnectorWorker):
    async def _build_transfer_params(self, ready_reqs, agent_meta,
                                     local_regions, remote_regions):
        build = native.MooncakeConnectorWorker._build_transfer_params
        if op_disabled('hy4_mooncake_grouping_reuse'):
            return await build(self, ready_reqs, agent_meta, local_regions, remote_regions)

        # Core copies per-region block lists. Cache by their complete values,
        # not object identity, so packed/shared groups retain their semantics.
        # This cache is invocation-local: later prefix hits or KV allocations
        # must never reuse an earlier request's transfer plan.
        grouped = {}
        group = native.group_concurrent_contiguous

        def reuse(source, destination):
            key = (tuple(source), tuple(destination))
            result = grouped.get(key)
            if result is None:
                result = group(source, destination)
                grouped[key] = result
            return result

        build = bind_native(build, {'group_concurrent_contiguous': reuse})
        return await build(self, ready_reqs, agent_meta, local_regions, remote_regions)

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
        try:
            _init_native(self, vllm_config, engine_id, kv_cache_config)
        except BaseException:
            self._cleanup_partial_init()
            raise
        self._native_entered = True

    def _cleanup_partial_init(self):
        # Core shutdown assumes its constructor reached async_zmq_ctx. A bind
        # or transport error may occur after threads started but before that.
        actions = []
        context = getattr(self, 'async_zmq_ctx', None)
        if context is not None:
            actions.append(context.term)
        executor = getattr(self, '_sender_executor', None)
        if executor is not None:
            actions.append(lambda: executor.shutdown(wait=False))
        for loop_name, thread_name in (('sender_loop', '_sender_listener_t'),
                                       ('receiver_loop', '_mooncake_receiver_t')):
            loop = getattr(self, loop_name, None)
            thread = getattr(self, thread_name, None)
            if loop is not None:
                # Queue the stop even when the new thread has not entered
                # run_forever yet; checking is_running alone races its start.
                actions.append(lambda loop=loop: loop.call_soon_threadsafe(loop.stop))
            if thread is not None:
                actions.append(lambda thread=thread: thread.join(timeout=5))
        bootstrap = getattr(self, 'bootstrap_server', None)
        if bootstrap is not None:
            actions.append(bootstrap.shutdown)
        for action in actions:
            try:
                action()
            except Exception:
                native.logger.exception('HY4 PCP Mooncake partial initialization cleanup failed')

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


class MooncakeConnector(native.MooncakeConnector):
    # Retain Core's connector identifier so its configuration compatibility
    # guards (including auxiliary output exclusion) still recognize Mooncake.
    # The explicit external module path selects this class, not a registry edit.
    # Reuse the complete native constructor and transport, substituting this
    # worker class in a private namespace.
    __init__ = bind_native(native.MooncakeConnector.__init__,
                          {'MooncakeConnectorWorker': HY4PCPMooncakeWorker,
                           'MooncakeConnectorScheduler': HY4PCPMooncakeScheduler})

    def bind_connector_metadata(self, connector_metadata):
        self._early_send_metadata = None
        super().bind_connector_metadata(connector_metadata)
        worker = self.connector_worker
        if worker is not None and worker.is_kv_producer and not worker.is_kv_consumer:
            # MRV2 defers async start_load_kv until post_forward. On P this
            # would hold the previous completed requests' ready notifications
            # behind another long prefill. Submit those notifications at bind;
            # the native worker still owns transfer, ACK and block lifetime.
            worker.start_load_kv(connector_metadata)
            self._early_send_metadata = connector_metadata

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._connector_metadata
        if metadata is not None and metadata is getattr(self, '_early_send_metadata', None):
            return
        return super().start_load_kv(forward_context, **kwargs)

    def clear_connector_metadata(self):
        super().clear_connector_metadata()
        self._early_send_metadata = None
