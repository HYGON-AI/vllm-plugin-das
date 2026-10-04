# Kimi-K2.6 gfx938 TP8 HumanEval-64 validation

This record covers the local `/models/Kimi-K2.6` checkpoint on the eight-card
gfx938 host with the pinned vLLM 0.28.1 DTK runtime recorded in
`v0281-gfx938-provenance.md`. HumanEval generated code was executed only after
the operator asserted the isolated evaluation boundary with
`VLLM_HCU_HUMANEVAL_ISOLATED=1`.

## Checkpoint and route

- 64 safetensors shards, about 555 GiB, with 208,550 indexed tensors and no
  missing or extra shard entries.
- `KimiK25ForConditionalGeneration` multimodal wrapper with a
  `DeepseekV3ForCausalLM` language model, compressed-tensors W4A16 group
  quantization, 61 layers, 384 routed experts, top-8 routing, and one shared
  expert.
- `num_nextn_predict_layers=0`, so the checkpoint does not support MTP.
- Code evaluation uses `--language-model-only`, TP8, HcuGPUModelRunnerV2,
  regular `FLASHMLA`, Triton WNA16 MoE, the `kimi_k2` reasoning parser,
  prefix caching, BF16/auto KV, and the default FULL_AND_PIECEWISE Graph
  policy. FLASHMLA retains its native LBNHC KV layout and 64-token cache page.

The validated service command is:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /models/Kimi-K2.6 \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA \
  --moe-backend triton \
  --reasoning-parser kimi_k2 \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --generation-config vllm
```

The runtime loaded 74.92 GiB per rank, allocated an 858,368-token KV cache,
captured both PIECEWISE and FULL graphs, and produced prefix-cache hits. It
reported `CompressedTensorsWNA16MoEMethod`, `Using 'TRITON' WNA16 MoE
backend`, and `Using TritonWNA16Experts`. AITER rejected this WNA16 MoE route
by contract, while Humming has no HIP implementation; neither is presented as
a viable backend for this checkpoint.

## Template, sampling, and scoring controls

The Kimi parser separates each response into reasoning and final text. The
checkpoint README recommends `temperature=1.0, top_p=0.95` for Thinking mode
and `temperature=0.6, top_p=0.95` for Instant mode. Greedy decoding is not a
valid default for either route: it repeatedly degenerated on otherwise simple
HumanEval prompts.

The final complete run used Instant mode with
`chat_template_kwargs.thinking=false`, temperature 0.6, top-p 0.95, and
BF16/auto KV. It produced 64 predictions and 64 reviews. EvalScope's raw
checker scored 22/64; the independent normalizer executed the returned code
and passed 54/64 (84.375%). Failed tasks were HumanEval/19, /21, /25, /31,
/36, /39, /46, /56, /60, and /61. The stable length-capped failures included
/21, /25, /31, /39, /56, /60, and /61.

Two earlier complete 64-sample Instant controls with greedy decoding
established that E4M3 KV was not the cause of the degeneration:

| KV route | Normalized HumanEval-64 | Observation |
| --- | ---: | --- |
| E4M3 | 52/64 | Several responses repeated until the 2,048-token control limit |
| BF16/auto | 51/64 | A different failure set; the common misses were not KV-format-specific |

Thinking-mode diagnostics were also unsuitable for this low-latency code
gate: greedy decoding reached 4096 tokens with empty final code on
HumanEval/10, while official temperature/top-p sampling remained materially
slower and was stopped after 19 completed samples once the same prompt again
became an extreme long-tail request. Instant mode completed HumanEval/10 in
209 tokens with valid code.

Targeted repetition penalties 1.05 and 1.1 on HumanEval/21 and /25 either
still hit their 1,024-token diagnostic limit or stopped with irrelevant,
prompt-misinterpreting prose. Repetition penalty is therefore not accepted as
a fix. The cross-KV and cross-template evidence does not justify an HCU kernel
change; the profile is retained as a diagnostic artifact gate and does not
claim 64/64 accuracy.

The checked-in profile sets `record_normalized_score: true` together with
`normalize_code_fences: true`. A rerun therefore records the normalized score
and report after confirming all 64 predictions/reviews, but deliberately does
not enforce a fixed score for this stochastic sampling route. This opt-in is
rejected for EvalScope-sandboxed output so generated code is never
re-executed across the sandbox boundary.

EvalScope's raw checker does not consistently handle the model's complete,
fenced function response. The independent evidence normalizer removes one
complete or truncated Markdown fence. If the returned full module begins with
horizontal whitespace, it removes that whitespace only when Python parsing
succeeds and the module contains the expected top-level HumanEval entry point.
This preserves valid indented body-only completions and fails closed on syntax
or parser resource errors.

Evidence directories:

- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-thinking`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-official-sampling`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-instant-official`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-kvbf16`
