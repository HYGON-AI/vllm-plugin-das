# SPDX-License-Identifier: Apache-2.0
"""Check the Markov fusion on a single GPU with the two real draft weights."""

import argparse
import json
import statistics
from types import SimpleNamespace
from unittest.mock import patch

from safetensors import safe_open
import torch
import torch.nn.functional as F

from hcu_vllm_das.parallel.minimax31_markov import fused_markov_greedy, greedy_markov_tokens
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def reference(base, seed, embedding, project, out):
    token = seed
    for index in range(base.shape[1]):
        bias = project(F.embedding(token, embedding))
        logits = base[:, index] + bias
        token = logits.masked_fill(logits.isnan(), float("-inf")).argmax(-1)
        out[:, index].copy_(token)
    return out


def assert_same(expected, actual, label):
    mismatch = (expected != actual).nonzero()
    if mismatch.numel():
        row, step = mismatch[0].tolist()
        raise AssertionError(
            f"{label} at request {row}, step {step}: "
            f"expected {expected[row, step].item()}, got {actual[row, step].item()}"
        )


def corner_cases():
    vocab = 513
    weights = torch.zeros((vocab, 272), device="cuda", dtype=torch.bfloat16)[:, :256]
    base = torch.full((4, 7, vocab), float("-inf"), device="cuda", dtype=torch.bfloat16)
    seed = torch.tensor([0, 32, 512, 51], device="cuda")
    base[0, :, 128] = 3
    base[0, :, 256] = 3
    base[1, :, 255] = float("nan")
    base[1, :, 512] = float("nan")
    base[1, :, 300] = 2
    base[3].fill_(float("nan"))
    expected = reference(base, seed, weights, lambda inputs: F.linear(inputs, weights),
                         torch.empty_like(base[:, :, 0], dtype=torch.long))
    assert_same(expected, fused_markov_greedy(base, seed, weights, weights), "ties, NaN and all -inf")
    assert expected[:, 0].tolist() == [128, 300, 0, 0]


def capture(function):
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def verify_prefix(expected, actual, target):
    # Greedy accepts the longest consecutive prefix shared with the target.
    expected_lengths = (expected == target).cumprod(-1).sum(-1)
    actual_lengths = (actual == target).cumprod(-1).sum(-1)
    torch.testing.assert_close(expected_lengths, actual_lengths, atol=0, rtol=0)


def run_case(batch, embedding, projection, *, measure):
    vocab = embedding.shape[0]
    base = torch.randn((batch, 7, vocab), device="cuda", dtype=torch.bfloat16)
    seed = torch.randint(vocab, (batch,), device="cuda", dtype=torch.long)
    reference_ids = torch.empty((batch, 8), dtype=torch.long, device="cuda")[:, 1:]
    # The offline Verify8 path writes directly into a strided proposal slice.
    fused_ids = torch.empty((batch, 8), dtype=torch.long, device="cuda")[:, 1:]
    head = SimpleNamespace(
        markov_w1=SimpleNamespace(weight=embedding),
        markov_w2=SimpleNamespace(weight=projection, tp_size=1, quant_method=UnquantizedLinearMethod()),
    )
    # This double supplies the real offline dispatch metadata. An unintended
    # fallback fails because the reference model methods are absent here.
    model = SimpleNamespace(
        model=SimpleNamespace(markov_head=head),
        logits_processor=SimpleNamespace(vocab_size=vocab, scale=1.0, soft_cap=None,
                                        logits_as_input=False, head_dtype=None),
        draft_id_to_target_id=None,
    )
    def project(inputs):
        return head.markov_w2.quant_method.apply(head.markov_w2, inputs)

    baseline = lambda: reference(base, seed, embedding, project, reference_ids)
    candidate = lambda: greedy_markov_tokens(model, base, seed, out=fused_ids, fused=True)
    baseline()
    candidate()
    assert_same(reference_ids, fused_ids, f"batch {batch}, eager")
    graphs = [capture(operation) for operation in (baseline, candidate)]
    for repeat in range(8):
        base.normal_()
        seed.random_(vocab)
        for graph in graphs:
            graph.replay()
        assert_same(reference_ids, fused_ids, f"batch {batch}, replay {repeat}")
        target = reference_ids.clone()
        if repeat < 7:
            target[:, repeat].add_(1).remainder_(vocab)
        verify_prefix(reference_ids, fused_ids, target)

    result = {"batch": batch, "matched_tokens": batch * 7 * 9,
              "exact_greedy_prefix": True}
    if measure:
        samples = [[], []]
        for repeat in range(6):
            for variant in (range(2) if repeat % 2 == 0 else (1, 0)):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graphs[variant].replay()
                stop.record()
                stop.synchronize()
                samples[variant].append(start.elapsed_time(stop) / 20)
        result["reference_ms"], result["fused_ms"] = map(statistics.median, samples)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--measure", action="store_true")
    options = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Markov unit test requires exactly one visible GPU")
    torch.manual_seed(1219)
    free, capacity = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        raise RuntimeError("Markov unit test requires 2 GiB free")
    torch.cuda.set_per_process_memory_fraction(1024**3 / capacity)
    prefix = "language_model.model.dspark.markov_head."
    with safe_open(options.checkpoint, framework="pt", device="cpu") as handle:
        embedding, projection = (
            handle.get_tensor(prefix + name + ".weight").to("cuda")
            for name in ("markov_w1", "markov_w2")
        )
    assert embedding.shape == projection.shape == (200064, 256)
    assert embedding.dtype == projection.dtype == torch.bfloat16
    with torch.inference_mode():
        corner_cases()
        for batch in (8, 16, 18):
            print(json.dumps(run_case(batch, embedding, projection, measure=options.measure)),
                  flush=True)


if __name__ == "__main__":
    main()
