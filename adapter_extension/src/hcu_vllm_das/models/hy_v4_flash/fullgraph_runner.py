# SPDX-License-Identifier: Apache-2.0
"""Scoped Flash runner factory using the public Adapter runner seam."""
import os

_OWNER = "das-hcu-hyv4-flash-short-full-bank"
_registration = None
_pid = None
_failure = None
_runner_cls = None
_runner_methods = None
_MIDDLE_SCORING_BOUND = 16384


def _qualified_worker_tuple(tp, arch, compute_units):
    """Keep worker-bank admission scoped to qualified platform/TP pairs."""
    return ((tp, arch, compute_units) in (
        (4, "gfx938", 64), (4, "gfx936", 80), (8, "gfx936", 80),
    ))


def _require_qualified_graph_worker(tp, device):
    """Check the actual worker after weights reside on its HCU device."""
    import torch

    properties = torch.cuda.get_device_properties(device)
    if not _qualified_worker_tuple(
            tp, str(getattr(properties, "gcnArchName", "")).split(":")[0],
            getattr(properties, "multi_processor_count", None)):
        raise NotImplementedError(
            "Flash W4A8 Graph worker requires gfx938/64CU TP4 or gfx936/80CU TP4/TP8")


def is_qualified_config(cfg, *, breakable):
    """Configuration predicate only; platform/version ownership is separate."""
    if breakable:
        return False
    # Core owns communicator selection and fallback. Either boolean policy
    # uses the same ordered capture/replay and independent scoring-bank pools.
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
                or parallel.tensor_parallel_size not in (4, 8)
                or any(getattr(parallel, field) != 1 for field in (
                    "data_parallel_size", "pipeline_parallel_size",
                    "prefill_context_parallel_size", "decode_context_parallel_size"))
                or parallel.enable_dbo or parallel.enable_eplb
                or type(parallel.disable_custom_all_reduce) is not bool
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


def middle_scoring_enabled(cfg, device):
    """Startup-only admission; the shared W4 Graph predicate stays unchanged."""
    if not is_qualified_config(cfg, breakable=False):
        return False
    from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    from hcu_vllm_das.environment import fp8_fusions_enabled
    if is_breakable_cudagraph_enabled() or not fp8_fusions_enabled():
        return False
    from .quantization import validate_flash_quant_config, validate_flash_quant_parallelism
    try:
        if validate_flash_quant_config(cfg.quant_config.config) != "channel_w4a8":
            return False
        validate_flash_quant_parallelism(cfg)
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return False
    from .runtime import is_hcu_platform
    if not is_hcu_platform():
        return False
    import torch
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device)
    return _qualified_worker_tuple(
        cfg.parallel_config.tensor_parallel_size,
        str(getattr(properties, "gcnArchName", "")).split(":")[0],
        getattr(properties, "multi_processor_count", None))


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
    if not is_hcu_platform() or not is_qualified_config(
        cfg, breakable=is_breakable_cudagraph_enabled()
    ):
        return False
    import torch
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return _qualified_worker_tuple(
        cfg.parallel_config.tensor_parallel_size,
        str(getattr(properties, "gcnArchName", "")).split(":")[0],
        getattr(properties, "multi_processor_count", None))


def runner_class():
    global _runner_cls, _runner_methods
    if _runner_cls is None:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.platforms import current_platform
        from .compressed_indexer import FlashCompressedIndexer
        from .fullgraph_banks import (
            FlashFullGraphBanks, derive_short_bound, indexer_capacity_covers,
        )

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
                if (bound is None
                        or not indexer_capacity_covers(owners, self.max_model_len)
                        or bound >= self.max_model_len):
                    return
                short = ModelCudaGraphManager(
                    self.vllm_config, self.device, wide.cudagraph_mode,
                    decode_query_len=self.decode_query_len,
                    lora_capture_cases=wide.lora_capture_cases,
                    varlen_decode=wide.varlen_decode,
                    ubatch_runner=None,
                )
                if not hasattr(self, "_hcu_middle_scoring_enabled"):
                    # Freeze the policy across throwaway profiling and real
                    # initialization; capture/replay must not resolve config.
                    self._hcu_middle_scoring_enabled = (
                        bound < _MIDDLE_SCORING_BOUND < self.max_model_len
                        and middle_scoring_enabled(self.vllm_config, self.device)
                    )
                scoring_banks = ()
                if (self._hcu_middle_scoring_enabled
                        and bound < _MIDDLE_SCORING_BOUND < self.max_model_len):
                    middle = ModelCudaGraphManager(
                        self.vllm_config, self.device, wide.cudagraph_mode,
                        decode_query_len=self.decode_query_len,
                        lora_capture_cases=wide.lora_capture_cases,
                        varlen_decode=wide.varlen_decode,
                        ubatch_runner=None,
                    )
                    scoring_banks = ((_MIDDLE_SCORING_BOUND, middle),)
                self.cudagraph_manager = FlashFullGraphBanks(
                    wide, short, owners, bound, CUDAGraphMode.FULL,
                    short_pool_factory=current_platform.graph_pool_handle,
                    scoring_banks=scoring_banks,
                )

            def gather_batch_req_state(self, scheduler_output, dummy_run):
                state, uniform = super().gather_batch_req_state(
                    scheduler_output, dummy_run
                )
                manager = self.cudagraph_manager
                if isinstance(manager, FlashFullGraphBanks):
                    manager.use_wide_bank()
                    if not dummy_run and state is not None:
                        try:
                            mapping = state.idx_mapping_np
                            if mapping is None:
                                return state, uniform
                            limits = self.req_states.max_seq_len[mapping]
                        except (AttributeError, TypeError, IndexError, KeyError):
                            return state, uniform
                        manager.set_request_limits(limits, self.decode_query_len)
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
