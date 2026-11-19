# SPDX-License-Identifier: Apache-2.0
"""Instance-owned admission for exact W4A8 greedy draft sampling."""
from functools import lru_cache
import logging
import os
from types import MethodType, SimpleNamespace

import torch

from hcu_adapter.operator_contracts import supports_operator_contract


_LOG = logging.getLogger('vllm.' + __name__)


@lru_cache(maxsize=1)
def _native_contract():
    from vllm.models.deepseek_v41.amd import dspark as native_model
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator as native_spec

    valid = (supports_operator_contract(native_model, 'dsv41_greedy_markov')
             and supports_operator_contract(native_spec, 'dsv41_greedy_markov'))
    return (valid, getattr(native_model, "DSparkDeepseekV4ForCausalLM", None),
            getattr(native_spec, "DSparkSpeculator", None))


def supported(model, spec, num_reqs, hidden, *, require_qualification=True):
    steps = getattr(spec, 'num_speculative_steps', None)
    tokens = getattr(spec, 'draft_tokens', None)
    if (not getattr(model, '_hcu_markov_pointwise', False)
            or getattr(spec, 'model', None) is not model
            or type(num_reqs) is not int or not 1 <= num_reqs <= 64
            or type(steps) is not int or not 1 <= steps <= 7
            or getattr(spec, 'draft_logits', False) is not None
            or getattr(spec, 'draft_watermarker', False) is not None
            or getattr(spec, '_d2t_scatter_index', False) is not None
            or getattr(spec, '_draft_topk', False) is not None
            or getattr(spec, 'use_confidence_head', True) is not False
            or getattr(spec, 'acceptance_estimator', False) is not None
            or hasattr(model, 'apply_output_logit_transform')
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')):
        return False
    head = getattr(getattr(model, 'model', None), 'markov_head', None)
    if head is None or type(head.markov_w1) is not torch.nn.Embedding:
        return False
    embedding = head.markov_w1
    w1, w2 = embedding.weight, head.markov_w2.weight
    if (embedding.max_norm is not None or embedding._forward_hooks
            or embedding._forward_pre_hooks
            or w1.shape != (129280, 256) or w2.shape != w1.shape
            or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
            or not (w1.is_cuda and w1.is_contiguous() and w2.is_contiguous())
            or w2.device != w1.device or hidden.device != w1.device
            or hidden.dtype != torch.bfloat16 or hidden.ndim != 2
            or hidden.shape[1] != 5120):
        return False
    processor = model.logits_processor
    if (head.markov_w2.quant_method.__class__.__name__ != 'UnquantizedEmbeddingMethod'
            or head.markov_w2.tp_size != 1
            or processor.head_dtype not in (None, torch.bfloat16)
            or processor.scale != 1.0 or processor.soft_cap is not None
            or processor.logits_as_input or processor.org_vocab_size != 129280):
        return False
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2
            or tokens.shape[0] < num_reqs or tokens.shape[1] != steps
            or tokens.dtype != torch.int64 or tokens.stride(1) != 1
            or tokens.device != w1.device):
        return False
    valid, native_model, native_spec = _native_contract()
    if native_model is None or native_spec is None:
        return False
    if (require_qualification and not valid
            and getattr(model, '_hcu_markov_preflight', None)
            != (os.getpid(), native_spec._sample_sequential, steps)):
        return False
    for name in ('markov_embed', 'markov_bias', 'map_draft_to_target'):
        if getattr(getattr(model, name), '__func__', None) is not getattr(native_model, name):
            return False
    if getattr(spec._sample_logits, '__func__', None) is not native_spec._sample_logits:
        return False
    props = torch.cuda.get_device_properties(w1.device)
    return props.gcnArchName.split(':')[0] == 'gfx938' and props.multi_processor_count == 64


@lru_cache(maxsize=1)
def _log_admission():
    logging.getLogger('vllm.' + __name__).info(
        'HCU W4A8 exact greedy Markov pointwise admitted: V129280 R256; native bias projection retained')


@torch.inference_mode()
def qualify_markov(model, spec):
    """Compare against the native sampling loop with the loaded draft weights."""
    declared, _, native_spec = _native_contract()
    if declared:
        return True
    if native_spec is None or type(spec) is not native_spec:
        return False
    head = getattr(getattr(model, 'model', None), 'markov_head', None)
    if head is None or not getattr(model, '_hcu_markov_pointwise', False):
        return False
    device = head.markov_w1.weight.device
    if device.type != 'cuda':
        return False
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            steps = spec.num_speculative_steps
            if type(steps) is not int or not 1 <= steps <= 7:
                return False
            generator = torch.Generator(device=device).manual_seed(7851)
            for requests, dtype, magnitude in (
                    (1, torch.int32, 0.), (8, torch.int64, 1.),
                    (32, torch.int32, 8.), (64, torch.int64, 1.)):
                n = requests * steps
                indices = torch.arange(n, device=device)
                hidden = torch.empty((n, 5120), device=device, dtype=torch.bfloat16)
                probe = SimpleNamespace(
                    model=model, num_speculative_steps=spec.num_speculative_steps,
                    draft_logits=spec.draft_logits, draft_watermarker=spec.draft_watermarker,
                    _d2t_scatter_index=spec._d2t_scatter_index, _draft_topk=spec._draft_topk,
                    use_confidence_head=spec.use_confidence_head,
                    acceptance_estimator=spec.acceptance_estimator,
                    sample_indices=indices, sample_idx_mapping=indices // steps,
                    sample_pos=torch.ones_like(indices), _step_cols=spec._step_cols,
                    input_buffers=SimpleNamespace(input_ids=torch.randint(
                        129280, (requests,), device=device, dtype=dtype, generator=generator)),
                    _anchor_idx=torch.arange(requests, device=device),
                    draft_tokens=torch.full((requests + 1, steps), -1, device=device, dtype=torch.int64))
                probe._sample_logits = MethodType(native_spec._sample_logits, probe)
                probe._maybe_predict_acceptance = MethodType(native_spec._maybe_predict_acceptance, probe)
                if not supported(model, probe, requests, hidden, require_qualification=False):
                    return False
                base = torch.randn((n, 129280), device=device, dtype=torch.bfloat16,
                                   generator=generator).mul_(magnitude)
                # The proxy omits the optimization hook. The oracle executes the
                # installed sequential loop with the same logits and live head.
                probe.model = SimpleNamespace(
                    compute_draft_logits=lambda _: base, markov_embed=model.markov_embed,
                    markov_bias=model.markov_bias, map_draft_to_target=model.map_draft_to_target)
                native_spec._sample_sequential(probe, requests, hidden)
                expected = probe.draft_tokens.clone()
                probe.draft_tokens.fill_(-1)
                _write_draft_tokens(model, probe, requests, base)
                if not torch.equal(expected, probe.draft_tokens):
                    _LOG.warning('HCU greedy Markov preflight differs from the native sampling loop')
                    return False
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as error:
            _LOG.warning('HCU greedy Markov preflight unavailable: %s', error)
            return False
    model._hcu_markov_preflight = (os.getpid(), native_spec._sample_sequential, steps)
    _LOG.info('HCU greedy Markov admitted after native sampling preflight')
    return True


def _write_draft_tokens(model, spec, num_reqs, base):
    from .dspark_greedy_pointwise import greedy

    anchors = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    head = model.model.markov_head
    greedy(head.markov_w1.weight, head.markov_w2.weight,
           base.view(num_reqs, spec.num_speculative_steps, 129280), anchors,
           bias_fn=model.markov_bias, embed_fn=model.markov_embed,
           out=spec.draft_tokens[:num_reqs])


def sample_sequential(model, spec, num_reqs, head_hidden):
    if not supported(model, spec, num_reqs, head_hidden):
        return False
    n = num_reqs * spec.num_speculative_steps
    sample_hidden = head_hidden[spec.sample_indices[:n]]
    base = model.compute_draft_logits(sample_hidden)
    # The layout check runs before any custom writes to the draft buffer.
    if base.dtype != torch.bfloat16 or base.shape != (n, 129280) or not base.is_contiguous():
        return False
    _write_draft_tokens(model, spec, num_reqs, base)
    _log_admission()
    return True
