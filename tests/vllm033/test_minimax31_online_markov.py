# SPDX-License-Identifier: Apache-2.0
"""Exercise the MiniMax hook through vLLM's actual D-Spark sampler."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hcu_vllm_das.models.minimax31_dspark import DSparkMiniMaxDraftModel
from hcu_vllm_das.parallel import minimax31_markov as markov
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _draft_model(embedding, projection):
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w1 = nn.Embedding.from_pretrained(embedding)
    head.markov_w2 = nn.Module()
    head.markov_w2.weight = nn.Parameter(projection, requires_grad=False)
    head.markov_w2.tp_size = 1
    head.markov_w2.quant_method = UnquantizedEmbeddingMethod()

    vocab = embedding.shape[0]
    processor = LogitsProcessor.__new__(LogitsProcessor)
    nn.Module.__init__(processor)
    processor.vocab_size = processor.org_vocab_size = vocab
    processor.head_dtype = processor.soft_cap = None
    processor.scale = 1.0
    processor.logits_as_input = False

    draft = DSparkMiniMaxDraftModel.__new__(DSparkMiniMaxDraftModel)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.markov_head = head
    draft.logits_processor = processor
    draft.draft_id_to_target_id = None
    # Supply precomputed logits so the probe requires only the two Markov weights.
    draft.compute_draft_logits = lambda hidden: hidden
    return draft.eval()


def _speculator(model, batch, device):
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.model = model
    spec.num_speculative_steps = 7
    spec.draft_logits = spec.draft_watermarker = spec._draft_topk = None
    spec.acceptance_estimator = None
    spec.enable_adaptive_verification = spec.use_confidence_head = False
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(2 * batch, dtype=torch.int32, device=device)
    )
    spec._anchor_idx = torch.arange(1, 2 * batch, 2, device=device)
    count = batch * spec.num_speculative_steps
    spec.sample_indices = torch.arange(count - 1, -1, -1, device=device)
    spec.sample_idx_mapping = torch.zeros(count, dtype=torch.int64, device=device)
    spec.sample_pos = torch.zeros(count, dtype=torch.int32, device=device)
    spec._step_cols = torch.arange(spec.num_speculative_steps, device=device)
    spec.draft_tokens = torch.full((batch + 1, 7), -1, dtype=torch.int64, device=device)
    return spec


@pytest.fixture
def device():
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires gfx938")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx938":
        pytest.skip("requires gfx938")
    return torch.device("cuda:0")


def _load_greedy_policy(spec):
    context = SimpleNamespace(worker=SimpleNamespace(
        model_runner=SimpleNamespace(speculator=spec)
    ))
    markov.after_minimax31_markov_load(context)


def test_topk_fallback_delegates_to_native_sampler(monkeypatch):
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.model = DSparkMiniMaxDraftModel.__new__(DSparkMiniMaxDraftModel)
    spec.draft_logits = None
    spec._draft_topk = 8
    logits = torch.tensor([[float("nan"), 10.0, 9.0]])
    idx_map = torch.tensor([0])
    sample_pos = torch.tensor([42])
    selected = torch.tensor([2])

    def native(self, actual_logits, actual_indices, actual_positions, step):
        assert self is spec
        assert actual_logits is logits
        assert actual_indices is idx_map
        assert actual_positions is sample_pos
        assert step == 1
        return selected

    monkeypatch.setattr(DSparkSpeculator, "_sample_logits", native)
    _load_greedy_policy(spec)
    assert spec._sample_logits(logits, idx_map, sample_pos, 1) is selected


def _capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


@torch.inference_mode()
def _check_sampling(embedding, projection, monkeypatch, *, batch=16):
    device = embedding.device
    vocab = embedding.shape[0]
    rng = torch.Generator(device=device).manual_seed(2917)
    model = _draft_model(embedding, projection)
    ref_model = _draft_model(embedding, projection)
    ref_model.sample_sequential = None
    spec = _speculator(model, batch, device)
    ref = _speculator(ref_model, batch, device)

    # Independently model upstream's logits policy before the native reference
    # loop, and install the production policy only on the candidate instance.
    ref_model.apply_output_logit_transform = lambda logits: logits.masked_fill(
        logits.isnan(), float("-inf")
    )
    _load_greedy_policy(spec)
    hidden = torch.randn((batch * 7, vocab), device=device, dtype=torch.bfloat16,
                         generator=rng)
    assert spec.input_buffers.input_ids.dtype == torch.int32
    ref.input_buffers = spec.input_buffers

    def fused():
        spec._sample_sequential(batch, hidden)

    def stock():
        ref._sample_sequential(batch, hidden)

    original = markov.fused_markov_greedy
    hits = []

    def checked(base, seed, *weights, out):
        assert seed.dtype == torch.int32
        hits.append(out.data_ptr())
        return original(base, seed, *weights, out=out)

    with monkeypatch.context() as patch:
        patch.setattr(markov, "fused_markov_greedy", checked)
        fused()
    assert hits == [spec.draft_tokens.data_ptr()]
    stock()
    torch.testing.assert_close(spec.draft_tokens, ref.draft_tokens, rtol=0, atol=0)
    assert spec.draft_tokens[-1].eq(-1).all()

    stock_graph = _capture(stock)
    fused_graph = _capture(fused)
    for i in range(5):
        spec.input_buffers.input_ids.random_(0, vocab, generator=rng)
        hidden.normal_(generator=rng)
        if i == 0:
            hidden[:, 1] = hidden[:, -1] = float("inf")
        elif i == 1:
            hidden[:, 5] = hidden[:, -1] = float("nan")
        elif i == 2:
            hidden.fill_(float("-inf"))
        elif i == 3:
            hidden.fill_(float("nan"))
        stock_graph.replay()
        fused_graph.replay()
        torch.testing.assert_close(spec.draft_tokens, ref.draft_tokens, rtol=0, atol=0)


@pytest.mark.hcu
def test_online_markov_route_and_changed_input_graph(device, monkeypatch):
    rng = torch.Generator(device=device).manual_seed(42)
    weights = [
        torch.randn((576, 256), device=device, dtype=torch.bfloat16, generator=rng) / 16
        for _ in range(2)
    ]
    _check_sampling(*weights, monkeypatch)


@pytest.mark.hcu
@torch.inference_mode()
def test_online_markov_falls_back_to_native_loop(device, monkeypatch):
    weight = torch.zeros((128, 256), device=device, dtype=torch.bfloat16)
    model = _draft_model(weight, weight)
    spec = _speculator(model, 3, device)
    _load_greedy_policy(spec)
    installed = spec._sample_logits
    _load_greedy_policy(spec)
    assert spec._sample_logits is installed
    # An output transform changes the online argmax. The native loop must run it.
    model.apply_output_logit_transform = torch.neg
    hidden = torch.arange(128, device=device, dtype=torch.bfloat16).repeat(21, 1)
    hidden[:, 0] = float("nan")

    def unexpected(*args, **kwargs):
        pytest.fail("an unsupported online mode dispatched the fused kernel")

    monkeypatch.setattr(markov, "fused_markov_greedy", unexpected)
    spec._sample_sequential(3, hidden)
    assert spec.draft_tokens[:3].eq(1).all()

    # Even when every value is NaN, the input remains unmodified and ties
    # select the smallest in-vocabulary index, matching upstream.
    logits = torch.full((3, 128), float("nan"), device=device)
    assert spec._sample_logits(logits, spec.sample_idx_mapping[:3],
                               spec.sample_pos[:3], 0).eq(0).all()
    assert logits.isnan().all()
    del model.apply_output_logit_transform
    for name, value in (
        ("draft_logits", torch.empty(0)),
        ("draft_watermarker", object()),
        ("enable_adaptive_verification", True),
        ("use_confidence_head", True),
        ("acceptance_estimator", object()),
        ("_draft_topk", 8),
        ("num_speculative_steps", 6),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(spec, name, value)
            assert model.sample_sequential(spec, 3, hidden) is False
