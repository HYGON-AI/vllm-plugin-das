# Kimi-K2.5 W4A16 language-only validation

## Validated topology

`/model/kimi-K2.5` was validated on one eight-device gfx936 node with
TP8+EP8, `allgather_reducescatter`, 16 GiB/rank CPU offload, explicit
FLASHMLA, AITER-preferred W4A16 MoE, E5M2 KV cache, prefix caching, Model
Runner V2, synchronous scheduling, and the default FULL_AND_PIECEWISE graph
mode. The weight index has no MTP or next-N layers, so speculative decoding is
not enabled.

The final chat server also uses:

```text
--tool-call-parser kimi_k2
--enable-auto-tool-choice
--reasoning-parser kimi_k2
```

A live OpenAI tools request returned `finish_reason=tool_calls`, function
`get_weather`, and JSON arguments `{"city":"Beijing"}`. The runtime resolved
LBNHC and exposed 42,560 E5M2 KV tokens. Three identical long-prefix requests
added 22,899 queried tokens and 15,232 cache hits.

## EvalScope HumanEval filter

Kimi-K2.5 prepended one whitespace character to all 16 deterministic
HumanEval completions. EvalScope 1.12.0 preserves that character when an
answer is not fenced, so Python rejects otherwise-correct complete programs as
unexpected top-level indentation. The unfiltered report was 1/16. Reusing all
16 predictions with EvalScope's official `remove_whitespace` filter produced
Accuracy 16/16 and Pass@1 16/16, with no model requests or sampling changes.
This confirms evaluation extraction rather than E5M2 accuracy loss.

Do not change vLLM response text or patch the installed EvalScope package.
EvalScope provides an official, dataset-scoped filter for this case. The
checked-in test configuration passes it through `--dataset-args`:

```bash
--dataset-args '{"humaneval":{"filters":{"remove_whitespace":{}}}}'
```

The filter must remain scoped to Kimi HumanEval. It is not a global server
setting and must not be applied to benchmarks whose expected completion is an
indented function-body continuation.

To rescore preserved predictions without starting the model again, retain the
same model, dataset, limit, seed, and generation arguments, then add:

```bash
--dataset-args '{"humaneval":{"filters":{"remove_whitespace":{}}}}' \
--use-cache /path/to/original/evalscope-work-dir \
--rerun-review \
--work-dir /path/to/filtered-evalscope-work-dir
```

The executable server and evaluation contract is
`tests/models/kimi_k25_humaneval_evalscope.yaml`; its portable command test is
`tests/integration/server/test_evalscope_kimi_k25_humaneval.py`.
