# SPDX-License-Identifier: Apache-2.0
"""Scoped legacy MiniMax PCP/EP dispatch over the public MoE runner factory."""
from vllm.distributed.parallel_state import get_ep_group, get_pcp_group
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


class MiniMax31PcpEpMoERunner(MoERunner):
    def _pcp_ep_group(self):
        parallel = self.moe_config.moe_parallel_config
        if (parallel.tp_size != 1 or parallel.dp_size != 1
                or parallel.pcp_size != 8 or parallel.ep_size != 8
                or not parallel.use_ep or parallel.is_sequence_parallel
                or parallel.all2all_backend != "allgather_reducescatter"
                or not parallel.use_all2all_kernels
                or self.do_naive_dispatch_combine):
            raise RuntimeError("unsupported MiniMax legacy PCP/EP dispatch")
        pcp, ep = get_pcp_group(), get_ep_group()
        if (pcp.world_size != 8 or ep.world_size != 8
                or pcp.rank_in_group != ep.rank_in_group
                or list(pcp.ranks) != list(ep.ranks)):
            raise RuntimeError("MiniMax PCP and EP group mismatch")
        return pcp

    def _maybe_dispatch(self, hidden_states, router_logits):
        group = self._pcp_ep_group()
        if (hidden_states.ndim != 2 or router_logits.ndim != 2
                or hidden_states.shape[0] != router_logits.shape[0]
                or router_logits.shape[1] != 128):
            raise RuntimeError("unexpected MiniMax legacy PCP routing inputs")
        return (group.all_gather(hidden_states.contiguous(), dim=0),
                group.all_gather(router_logits.contiguous(), dim=0))

    def _maybe_combine(self, shared_output, hidden_states):
        import torch
        group = self._pcp_ep_group()
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
            raise RuntimeError("unexpected MiniMax legacy PCP routed output")
        if hidden_states.shape[0] % 8:
            raise RuntimeError("PCP routed rows do not divide across ranks")
        local = group.reduce_scatter(hidden_states.contiguous(), dim=0)
        return super()._maybe_combine(shared_output, local)


_original_factory = None
_installed_factory = None


def install_minimax31_ep_runner(stock_module):
    """Use the public runner_cls seam only for this explicitly selected model."""
    global _original_factory, _installed_factory
    if _installed_factory is not None:
        check_minimax31_ep_runner(stock_module)
        return False
    original = stock_module.FusedMoEFactory
    if not callable(original) or getattr(original, "__module__", "") != "vllm.model_executor.layers.fused_moe.layer":
        raise RuntimeError("MiniMax MoE factory already owned or unsupported")
    def minimax31_ep_factory(*args, **kwargs):
        if kwargs.get("runner_cls") is not None:
            raise RuntimeError("foreign MiniMax MoE runner override")
        import os
        if os.environ.get("HCU_MINIMAX_DEEPEP_FIXEDCOUNTS") == "1":
            from .minimax31_deepep_runner import MiniMax31DeepEpRunner
            kwargs["runner_cls"] = MiniMax31DeepEpRunner
        else:
            kwargs["runner_cls"] = MiniMax31PcpEpMoERunner
        from .minimax31_static_contract import from_current
        placement = from_current()
        if placement is not None:
            from vllm.config import get_current_vllm_config
            current=get_current_vllm_config()
            if (current.parallel_config.enable_dbo or current.speculative_config is not None
                    or not current.model_config.enforce_eager):
                raise RuntimeError("static136 supports eager single-inflight execution without DBO/speculation")
            if (os.environ.get("HCU_MINIMAX_STATIC_EPLB") != "1"
                    or os.environ.get("HCU_MINIMAX_DEEPEP_FIXEDCOUNTS") != "1"
                    or os.environ.get("HCU_MINIMAX_HIPC_PACK_TILED") != "1"):
                raise RuntimeError("static136 requires explicit fixed-count tiled HIPC path")
            # Shared-stream launch and join are guarded by the DeepEP runner's
            # single-inflight lock. Static selection/completion stays on its
            # caller thread; shared TP1 MLP never reads the routing frame.
            if (args or kwargs.get("num_experts") != 128
                    or kwargs.get("enable_eplb", False) is not False
                    or kwargs.get("num_redundant_experts", 0) != 0
                    or any(kwargs.get(k) is not None for k in ("moe_config_cls", "routed_experts_cls", "routed_experts_args"))):
                raise RuntimeError("unexpected static MiniMax factory overrides")
            from .minimax31_static_experts import StaticMoEConfig, StaticRoutedExperts
            from .minimax31_static_runtime import install_identity_hooks
            install_identity_hooks()
            kwargs["moe_config_cls"] = StaticMoEConfig
            kwargs["routed_experts_cls"] = StaticRoutedExperts
        elif os.environ.get("HCU_MINIMAX_STATIC_EPLB") == "1":
            raise RuntimeError("static opt-in requires an immutable physical placement contract")
        return original(*args, **kwargs)
    _original_factory, _installed_factory = original, minimax31_ep_factory
    stock_module.FusedMoEFactory = minimax31_ep_factory
    return True


def check_minimax31_ep_runner(stock_module):
    if (_original_factory is None or _installed_factory is None
            or stock_module.FusedMoEFactory is not _installed_factory):
        raise RuntimeError("MiniMax MoE runner factory ownership changed")
