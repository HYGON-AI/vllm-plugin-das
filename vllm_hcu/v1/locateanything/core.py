# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""LocateAnything Fast: six temporary probes, independent real KV accounting.

Used only by the isolated Fast vLLM overlay. No speculative acceptance/rejection.
"""
import torch
import os
import json
from pathlib import Path
import triton
import triton.language as tl
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.output import SamplerOutput

PROBES = 6
BOX, BOX_END, REF, REF_END = 151668, 151669, 151672, 151673
COORD_LO, COORD_HI, NULL, EOS, MASK, NONE = 151677, 152677, 152678, 151645, 151676, 4064


@triton.jit
def _inputs(ids, rope, starts, mapping, computed, history, stride: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    req = tl.load(mapping + row)
    start = tl.load(starts + row)
    end = tl.load(starts + row + 1)
    c = tl.load(computed + req)
    q = end - start
    local = chunk * BLOCK + tl.arange(0, BLOCK)
    physical = c + local
    real_len = c + q - 6
    # Dummy capture rows can be shorter than one Fast block.
    fast = q >= 7
    is_real = local < q - 6
    source = tl.minimum(physical, real_len - 1)
    value = tl.load(history + req * stride + source,
                    mask=(local < q) & fast & (is_real | (local == q - 6)), other=0)
    value = tl.where(fast & (local > q - 6), 151676, value)
    tl.store(ids + start + local, value, local < q)
    position = physical - tl.cast(fast & ~is_real, tl.int32)
    tl.store(rope + start + local, position, local < q)


def prepare_fast_inputs(runner, mapping, starts, num_logits, max_query_len):
    state = runner.model_state
    if os.environ.get('LOCATE_FAST_TEACHER'):
        state.fast_runner = runner
    _inputs[(mapping.numel(), triton.cdiv(max_query_len, 256))](
        runner.input_buffers.input_ids, state.fast_positions, starts, mapping,
        runner.req_states.num_computed_tokens.gpu,
        runner.req_states.all_token_ids.gpu,
        runner.req_states.all_token_ids.gpu.stride(0), 256)
    return (starts[1:mapping.numel()+1, None] - 6
            + torch.arange(6, device=mapping.device)[None, :]).clamp_min_(0).flatten()


def select_blocks(probs, sampled):
    """Vectorized original decode_bbox_avg / decode_ref / handle_pattern."""
    b = probs.shape[0]
    out = sampled.clone()
    top = probs[:, 1:5].topk(4, dim=-1).indices
    coord_valid = (top >= COORD_LO) & (top <= COORD_HI)
    coords = top.gather(-1, coord_valid.long().argmax(-1, keepdim=True)).squeeze(-1)
    empty = ((probs[:, 0, BOX] >= .7) & (probs[:, 1, NONE] > .2)
             & (probs[:, 2, BOX_END] > .2) & (probs[:, 3, NULL] > .1)
             & (probs[:, 4, NULL] > .1))
    legal = ((probs[:, 5, [BOX_END, NULL, EOS]].sum(-1) >= .2)
             & coord_valid.any(-1).all(-1))
    box = torch.empty_like(out)
    box[:, 0], box[:, 1:5], box[:, 5] = BOX, coords, BOX_END
    empty_box = torch.tensor([BOX, NONE, BOX_END, NULL, NULL, NULL], device=out.device)
    box = torch.where(empty[:, None], empty_box, box)
    has_box = empty | legal

    top_ref = probs[:, 1:].topk(5, dim=-1).indices
    valid_ref = (top_ref < COORD_LO) | (top_ref > COORD_HI)
    ref = torch.empty_like(out)
    ref[:, 0] = REF
    ref[:, 1:] = top_ref.gather(-1, valid_ref.long().argmax(-1, keepdim=True)).squeeze(-1)
    has_ref = (probs[:, 0, REF] >= .6) & valid_ref.any(-1).all(-1)
    out = torch.where(has_box[:, None], box, torch.where(has_ref[:, None], ref, out))

    terminal = (out[:, 0] == NULL) | (out[:, 0] == EOS)
    box_first = out[:, 0] == BOX
    empty = box_first & (out[:, 1] == NONE)
    point = (box_first & (out[:, 1] >= COORD_LO) & (out[:, 1] <= COORD_HI)
             & (out[:, 2] >= COORD_LO) & (out[:, 2] <= COORD_HI)
             & (out[:, 3] == BOX_END))
    # Non-box branch truncates null and one duplicate trailing ref_end.
    positions = torch.arange(6, device=out.device)
    count = torch.where(out == NULL, positions, 6).min(-1).values
    last = out.gather(1, (count - 1).clamp_min(0)[:, None]).squeeze(1)
    previous = out.gather(1, (count - 2).clamp_min(0)[:, None]).squeeze(1)
    count -= ((count >= 2) & (last == REF_END) & (previous == REF_END)).int()
    count = torch.where(box_first, torch.where(empty, 3, torch.where(point, 4, 6)), count)
    count = torch.where(terminal, 1, count).to(torch.int32)
    out[:, 0] = torch.where(terminal, EOS, out[:, 0])
    out[:, 2] = torch.where(empty, BOX_END, out[:, 2])
    return out.to(torch.int32), count


class FastSampler(Sampler):
    @classmethod
    def from_base(cls, base):
        base.__class__ = cls
        base.fast_generators = {}
        base.fast_request_generators = {}
        base.fast_teachers = {}
        return base

    def add_request(self, req_idx, prompt_len, sampling_params):
        p = sampling_params
        if (p.n != 1 or p.logprobs is not None or p.prompt_logprobs is not None
                or p.presence_penalty or p.frequency_penalty or p.min_p
                or p.min_tokens or p.logit_bias or p.bad_words
                or p.allowed_token_ids or p.structured_outputs
                or p.stop or p.stop_token_ids or p.ignore_eos):
            raise ValueError('Fast supports the original sampling contract only; unsupported sampling option')
        super().add_request(req_idx, prompt_len, p)
        req_id = self.req_states.index_to_req_id[req_idx]
        g = self.fast_request_generators.get(req_id)
        if g is None:
            g = torch.Generator(device=self.req_states.last_sampled_tokens.device)
            g.manual_seed(p.seed if p.seed is not None else torch.seed())
            self.fast_request_generators[req_id] = g
        self.fast_generators[req_idx] = g
        self.fast_teachers.pop(req_idx, None)
        teacher_root = os.environ.get('LOCATE_FAST_TEACHER')
        if teacher_root and not req_id.startswith('_'):
            folder = Path(teacher_root) / str(p.seed)
            manifest = json.loads((folder / 'manifest.json').read_text())
            self.fast_teachers[req_idx] = {'folder':folder,'manifest':manifest,'step':0}

    def __call__(self, logits, input_batch):
        batch = input_batch
        raw_diagnostic = logits.detach().cpu() if any(int(i) in self.fast_teachers for i in batch.idx_mapping_np) else None
        logits = logits.float().clone()
        self.penalties_state.apply_penalties(
            logits, batch.expanded_idx_mapping, batch.idx_mapping_np,
            batch.input_ids[batch.logits_indices], batch.expanded_local_pos)
        logits = logits.view(batch.num_reqs, 6, -1)
        states = self.sampling_states
        samples, probabilities = [], []
        # Per-request RNG is independent of continuous-batch reorder/arrival.
        # All six draws are consumed before deterministic box/ref overrides.
        for row, req in enumerate(batch.idx_mapping_np):
            temp, p, k = (float(states.temperature.np[req]),
                          float(states.top_p.np[req]), int(states.top_k.np[req]))
            x = logits[row]
            if temp > 0:
                x = x / temp
            if p < 1:
                sorted_x, indices = torch.sort(x, descending=True, dim=-1)
                remove = sorted_x.softmax(-1).cumsum(-1) > p
                shifted = torch.zeros_like(remove)
                shifted[:, 1:] = remove[:, :-1]
                mask = torch.zeros_like(remove).scatter(-1, indices, shifted)
                x = x.masked_fill(mask, torch.finfo(x.dtype).min)
            if 0 < k < x.shape[-1]:
                threshold = x.topk(k, dim=-1).values[:, -1, None]
                x = x.masked_fill(x < threshold, torch.finfo(x.dtype).min)
            probs = x.softmax(-1)
            if temp > 0:
                normalized = probs / probs.sum(-1, keepdim=True)
                sampled = torch.multinomial(normalized, 1, replacement=True,
                                            generator=self.fast_generators[int(req)]).squeeze(-1)
            else:
                sampled = probs.argmax(-1)
            samples.append(sampled)
            probabilities.append(probs)
        tokens, counts = select_blocks(torch.stack(probabilities), torch.stack(samples))
        if raw_diagnostic is not None:
            from vllm_hcu.v1.locateanything.teacher_compare import compare_and_force
            tokens, counts = compare_and_force(self, batch, raw_diagnostic, tokens, counts)
        return SamplerOutput(sampled_token_ids=tokens, logprobs_tensors=None,
                             num_nans=None, num_sampled=counts,
                             num_rejected=torch.zeros_like(counts))


class FastModelState(DefaultModelState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.model_config.enforce_eager
        assert not self.scheduler_config.enable_chunked_prefill
        assert not self.scheduler_config.async_scheduling
        self.fast_positions = torch.zeros(self.max_num_tokens, dtype=torch.long, device=self.device)

    def prepare_inputs(self, input_batch, req_states):
        return {'positions': self.fast_positions[:input_batch.num_tokens_after_padding]}

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        result = super().prepare_dummy_inputs(num_reqs, num_tokens)
        result['positions'] = self.fast_positions[:num_tokens]
        return result

    def custom_sampler(self, sampler):
        sampler = FastSampler.from_base(sampler)
        sampler.fast_model_state = self
        return sampler, None


def profile_sampler(logits):
    """Exercise six-row sorting workspace during vLLM's memory-only dummy run."""
    x = logits.float().view(-1, 6, logits.shape[-1]) / .7
    sx, indices = x.sort(descending=True, dim=-1)
    remove = sx.softmax(-1).cumsum(-1) > .9
    shifted = torch.zeros_like(remove)
    shifted[..., 1:] = remove[..., :-1]
    mask = torch.zeros_like(remove).scatter(-1, indices, shifted)
    probs = x.masked_fill(mask, torch.finfo(x.dtype).min).softmax(-1)
    return select_blocks(probs, probs.argmax(-1))
