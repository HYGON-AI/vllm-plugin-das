"""Fuse the MiniMax C4 expert-output sum with the first consumer of that sum."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner, ZeroExpertRouter
from vllm.models.minimax_m3.amd.model import MiniMAXGemmaRMSNorm, MiniMaxM3Model, MiniMaxM3MoE
from vllm.triton_utils import tl, triton

from hcu_vllm_das.parallel import minimax31_msa as msa
from hcu_vllm_das.parallel.minimax31_deepep_runner import MiniMax31DeepEpRunner


@dataclass(frozen=True)
class ExpertOutputs:
    shared: torch.Tensor
    routed: torch.Tensor
    routed_scale: float

    @property
    def shape(self):
        return self.routed.shape

    def sum(self):
        if self.routed_scale == 1.0:
            return self.shared + self.routed
        return self.shared + self.routed * self.routed_scale


_model = ContextVar("c4_moe_rms_model", default=False)
_producer = ContextVar("c4_moe_rms_producer", default=False)
_enabled = False
_verify = False
_scale = False
_installed = False
_produced = Counter()
_fused = Counter()
_verified = Counter()
_mismatches = Counter()
_fallbacks = Counter()
_scaled = Counter()


from lightop.norm.minimax31_prefill_moe_rms import _sum_norm


def _local_band():
    host = msa._HOST
    if host is None:
        return False
    lengths, starts, rows = host
    return (rows in (2048, 6144) and len(lengths) == len(starts) == rows // 1024
            and all(n == 1024 for n in lengths)
            and all(131072 <= start <= 146432 for start in starts))


def _tensor(x, shape, device):
    return (isinstance(x, torch.Tensor) and x.shape == shape and x.dtype == torch.bfloat16
            and x.device == device and x.is_contiguous())


def _defer(runner, shared, routed, trunc_size, reduced, scale=1.0):
    if not _producer.get() or shared is None:
        return None
    cfg = runner.moe_config
    if (not isinstance(runner, MiniMax31DeepEpRunner) or cfg.tp_size != 1
            or cfg.is_sequence_parallel or trunc_size not in (None, 6144)
            or isinstance(runner.router, ZeroExpertRouter)):
        _fallbacks["runner"] += 1
        return None
    if cfg.skip_final_all_reduce:
        if runner._fused_output_is_reduced:
            _fallbacks["reduce"] += 1
            return None
    elif cfg.ep_size > 1 and not reduced:
        # The stock tail calls tensor_model_parallel_all_reduce here.
        # GroupCoordinator returns the input itself on a single-rank TP group.
        if get_tp_group().world_size != 1:
            _fallbacks["reduce"] += 1
            return None
    if (routed.shape not in ((2048, 6144), (6144, 6144)) or not routed.is_cuda
            or not _tensor(routed, routed.shape, routed.device)
            or not _tensor(shared, routed.shape, routed.device)):
        _fallbacks["layout"] += 1
        return None
    _produced[routed.shape[0]] += 1
    return ExpertOutputs(shared, routed, scale)


def _defer_scaled(runner, shared, routed, trunc_size, reduced):
    if (not _scale or not _producer.get() or runner.routed_output_transform is not None
            or runner.routed_scaling_factor != 2.0):
        return None
    return _defer(runner, shared, routed, trunc_size, reduced, 2.0)


def _install():
    global _installed
    if _installed:
        return
    from .prefill_runtime_candidate import _patch

    old_model = MiniMaxM3Model.forward
    old_norm = MiniMAXGemmaRMSNorm.forward
    runner = _patch(
        MoERunner.forward,
        "    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(\n"
        "        shared_output, fused_output\n"
        "    )\n"
        "\n"
        "    # Apply output transform (e.g. latent -> full dim)\n"
        "    fused_output = self.apply_routed_output_transform(fused_output)\n"
        "\n"
        "    if shared_output is not None:\n        result = shared_output + fused_output\n",
        "    deferred = defer_scaled(self, shared_output, fused_output, og_hidden_dim_post_xform,\n"
        "                            fused_output_is_reduced)\n"
        "    if deferred is not None:\n        return deferred\n"
        "    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(\n"
        "        shared_output, fused_output\n"
        "    )\n"
        "    fused_output = self.apply_routed_output_transform(fused_output)\n"
        "    deferred = defer(self, shared_output, fused_output, og_hidden_dim_post_xform,\n"
        "                     fused_output_is_reduced)\n"
        "    if deferred is not None:\n        return deferred\n"
        "    if shared_output is not None:\n        result = shared_output + fused_output\n",
        defer=_defer,
        defer_scaled=_defer_scaled,
    )
    moe = _patch(
        MiniMaxM3MoE.forward,
        "    final_hidden_states = self.experts(\n"
        "        hidden_states=hidden_states, router_logits=router_logits\n"
        "    )\n\n"
        "    return final_hidden_states.view(num_tokens, hidden_dim)\n",
        "    with producer(hidden_states):\n"
        "        final_hidden_states = self.experts(\n"
        "            hidden_states=hidden_states, router_logits=router_logits\n"
        "        )\n"
        "    if isinstance(final_hidden_states, ExpertOutputs):\n"
        "        return final_hidden_states\n"
        "    return final_hidden_states.view(num_tokens, hidden_dim)\n",
        producer=_producer_context,
        ExpertOutputs=ExpertOutputs,
    )

    @wraps(old_model)
    def model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        if not _enabled or not _local_band():
            return old_model(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        eligible = (get_pp_group().world_size == 1 and get_tp_group().world_size == 1
                    and intermediate_tensors is None
                    and not getattr(self, "aux_hidden_state_layers", None)
                    and self.specialized_decode is None
                    and not torch.cuda.is_current_stream_capturing())
        if not eligible:
            _fallbacks["model"] += 1
            return old_model(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        token = _model.set(True)
        try:
            return old_model(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        finally:
            _model.reset(token)

    @wraps(old_norm)
    def norm_forward(self, x, residual=None):
        if not isinstance(x, ExpertOutputs):
            return old_norm(self, x, residual)
        shared, routed = x.shared, x.routed
        rows = routed.shape[0]
        if (not _tensor(residual, routed.shape, routed.device)
                or not _tensor(self.weight, (6144,), routed.device)):
            _fallbacks["norm"] += 1
            return old_norm(self, x.sum(), residual)
        y, carry = torch.empty_like(routed), torch.empty_like(residual)
        _sum_norm[(rows,)](shared, routed, residual, self.weight, carry, y,
                           self.variance_epsilon, x.routed_scale, num_warps=16)
        _fused[rows] += 1
        if x.routed_scale != 1.0:
            _scaled[rows] += 1
        if _verify:
            reference_y, reference_carry = old_norm(self, x.sum(), residual)
            if (not torch.equal(y.view(torch.uint8), reference_y.view(torch.uint8))
                    or not torch.equal(carry.view(torch.uint8), reference_carry.view(torch.uint8))):
                _mismatches[rows] += 1
                return reference_y, reference_carry
            _verified[rows] += 1
        return y, carry

    MoERunner.forward = runner
    MiniMaxM3MoE.forward = moe
    MiniMaxM3Model.forward = model_forward
    MiniMAXGemmaRMSNorm.forward = norm_forward
    _installed = True


@contextmanager
def _producer_context(hidden):
    if not _model.get() or hidden.shape not in ((2048, 6144), (6144, 6144)):
        yield
        return
    token = _producer.set(True)
    try:
        yield
    finally:
        _producer.reset(token)


def configure(value):
    global _enabled, _verify, _scale
    if value == "status":
        return {"enabled": _enabled, "verify": _verify, "scale": _scale,
                "scaled": dict(_scaled), "produced": dict(_produced),
                "fused": dict(_fused), "verified": dict(_verified),
                "mismatches": dict(_mismatches), "fallbacks": dict(_fallbacks)}
    scaled = isinstance(value, dict)
    if scaled:
        if set(value) != {"include_scale", "verify"} or value["include_scale"] is not True:
            raise ValueError("Expected an explicit routed scale fusion plan")
        value = value["verify"]
        if type(value) is not bool:
            raise ValueError("Verify option must be boolean")
    if value is not None and type(value) is not bool:
        raise ValueError("Use null for stock, true for exact comparison, or false for timing")
    _install()
    _enabled = value is not None
    _verify = value is True
    _scale = scaled
    for counts in (_produced, _fused, _verified, _mismatches, _fallbacks, _scaled):
        counts.clear()
    return configure("status")
