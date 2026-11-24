# SPDX-License-Identifier: Apache-2.0
"""Scoped Flash runner factory using the public Adapter runner seam."""
import os

_OWNER = "das-hcu-hyv4-flash-short-full-bank"
_registration = None
_pid = None
_failure = None
_runner_cls = None
_runner_methods = None


def is_qualified_config(cfg, *, breakable):
    """Configuration predicate only; platform/version ownership is separate."""
    if breakable:
        return False
    try:
        model, parallel = cfg.model_config, cfg.parallel_config
        compilation, spec, cache = (
            cfg.compilation_config, cfg.speculative_config, cfg.cache_config
        )
        if (not cfg.use_v2_model_runner or model.enforce_eager is not False
                or model.enable_prompt_embeds
                or model.hf_text_config.model_type != "hy_v4_flash"
                or compilation.mode.name != "VLLM_COMPILE"
                or compilation.cudagraph_mode.name != "FULL_AND_PIECEWISE"
                or parallel.tensor_parallel_size != 4
                or any(getattr(parallel, field) != 1 for field in (
                    "data_parallel_size", "pipeline_parallel_size",
                    "prefill_context_parallel_size", "decode_context_parallel_size"))
                or parallel.enable_dbo or parallel.enable_eplb
                or parallel.disable_custom_all_reduce is not True
                or cfg.lora_config is not None or cfg.kv_transfer_config is not None
                or spec is None or spec.method != "dfly"
                or spec.num_speculative_tokens != 5
                or cache.cache_dtype != "fp8_ds_mla"
                or cache.enable_prefix_caching is not False):
            return False
        mm = model.multimodal_config
        return mm is None or all(
            mm.get_limit_per_prompt(modality) == 0 for modality in ("image", "video")
        )
    except AttributeError:
        return False


def _fatal(message):
    global _failure
    from hcu_adapter.extensions import ExtensionCommitError
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def _check_registration():
    from hcu_adapter.bridges.vllm_model_runner import require_model_runner_factory
    if _failure is not None:
        _fatal(_failure)
    if _registration is None:
        raise RuntimeError("Flash graph-bank runner is not installed")
    try:
        if (_pid != os.getpid() or _registration.matches is not matches
                or _registration.create is not create):
            raise RuntimeError("Flash graph-bank runner changed owner/process")
        require_model_runner_factory(_registration)
        if _runner_cls is not None and (
            _runner_cls.initialize_kv_cache,
            _runner_cls.gather_batch_req_state,
        ) != _runner_methods:
            raise RuntimeError("Flash graph-bank runner methods changed owner")
    except BaseException:
        _fatal("Flash graph-bank runner ownership changed; restart the worker")


def check_health():
    """Public consumption requires the complete model publication to be healthy."""
    _check_registration()
    try:
        from hcu_vllm_das import hyv4_flash_model_activation
        if hyv4_flash_model_activation._publication is None:
            raise RuntimeError("Flash model publication has not committed")
        hyv4_flash_model_activation.check_health()
    except BaseException:
        _fatal("Flash model publication is unhealthy; restart the worker")


def matches(cfg):
    check_health()
    from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    from .runtime import is_hcu_platform
    return is_hcu_platform() and is_qualified_config(
        cfg, breakable=is_breakable_cudagraph_enabled()
    )


def runner_class():
    global _runner_cls, _runner_methods
    if _runner_cls is None:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.platforms import current_platform
        from .compressed_indexer import FlashCompressedIndexer
        from .fullgraph_banks import FlashFullGraphBanks, derive_short_bound

        class FlashBankModelRunner(GPUModelRunner):
            def initialize_kv_cache(self, *args, **kwargs):
                super().initialize_kv_cache(*args, **kwargs)
                wide = self.cudagraph_manager
                if type(wide) is not ModelCudaGraphManager:
                    raise RuntimeError("Flash bank requires the admitted Core manager")
                if (wide.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
                        or not wide._capture_descs.get(CUDAGraphMode.FULL)):
                    return
                owners = tuple(module for module in self.model.modules()
                               if isinstance(module, FlashCompressedIndexer))
                bound = derive_short_bound(owners)
                if (bound is None or bound >= self.max_model_len
                        or any(owner.max_model_len != self.max_model_len
                               for owner in owners)):
                    return
                short = ModelCudaGraphManager(
                    self.vllm_config, self.device, wide.cudagraph_mode,
                    decode_query_len=self.decode_query_len,
                    lora_capture_cases=wide.lora_capture_cases,
                    varlen_decode=wide.varlen_decode,
                    ubatch_runner=None,
                )
                self.cudagraph_manager = FlashFullGraphBanks(
                    wide, short, owners, bound, CUDAGraphMode.FULL,
                    short_pool_factory=current_platform.graph_pool_handle,
                )

            def gather_batch_req_state(self, scheduler_output, dummy_run):
                state, uniform = super().gather_batch_req_state(
                    scheduler_output, dummy_run
                )
                manager = self.cudagraph_manager
                if isinstance(manager, FlashFullGraphBanks):
                    manager.use_wide_bank()
                    if not dummy_run and state is not None:
                        manager.set_request_limits(
                            self.req_states.max_seq_len[state.idx_mapping_np],
                            self.decode_query_len,
                        )
                return state, uniform

        _runner_cls = FlashBankModelRunner
        _runner_methods = (
            _runner_cls.initialize_kv_cache, _runner_cls.gather_batch_req_state,
        )
    return _runner_cls


def create(cfg, device):
    if not matches(cfg):
        raise ValueError("Flash short FULL bank configuration was not admitted")
    return runner_class()(cfg, device)


def install():
    global _registration, _pid
    if _failure is not None:
        _fatal(_failure)
    if _registration is not None:
        _check_registration()
        return
    from hcu_adapter.bridges.vllm_model_runner import register_model_runner_factory
    try:
        _registration = register_model_runner_factory(
            _OWNER, matches=matches, create=create
        )
        _pid = os.getpid()
        _check_registration()
    except BaseException:
        _fatal("Flash graph-bank runner publication failed; restart the worker")
