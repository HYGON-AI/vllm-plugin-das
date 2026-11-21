# Gemma-4-26B-A4B BW1000 optimization bundle: scope and validation

Everything in this bundle is reached through one environment variable,
`VLLM_HCU_GEMMA_BW1000=1`.

## Baseline and layer

Baseline is `v0.28.1-dev` at `05e1ada`, merged into the branch as `2b8e16d`
(24 upstream commits). The only conflict was the `package_data` list in
`setup.py`, where both sides appended non-overlapping entries; both were kept.

Every change is in the plugin layer (`vllm_hcu/`). No community vLLM source
is modified. Relative to the baseline: 13 files, +1481 lines, all source and
packaging configuration.

## Applicability

| Requirement | Value |
|---|---|
| Hardware | BW1000, gfx936 |
| Devices | TP1 |
| Weights | BF16, **unquantized** |
| KV cache | FP8 (`--kv-cache-dtype fp8`), paged, gathered as **NHD** from the pool's own strides |
| Model | `model_type == gemma4`, `--language-model-only` |
| Speculative decoding | MTP4, draft returning indices rather than logits |

The switch is opt-in and does not default to on. The install paths assert
Gemma4-specific invariants -- vocabulary 262144, `soft_cap` 30.0, a bf16
lm_head of a known shape, TP1 -- and load a rocBLAS table validated for one
torch/HIP/rocBLAS/arch combination. This plugin also serves DeepSeek, GLM and
Kimi, where those assertions fail at `load_model`, and the FP8 checkpoint of
this same model has a quantized lm_head and fails too. Defaulting to on would
mean making every one of those assertions degrade silently, which is the
opposite of what they are for: they are what catches an "installed but doing
nothing" state, which is how the 320-only row bucket came to look installed
while doing nothing in production.

### Retired switches

`VLLM_HCU_GEMMA4_USE_FA` and `VLLM_HCU_GEMMA4_BW1000_OPT` are now one switch.
Nothing was ever measured with only one of them set, and keeping them apart
made it possible to serve half the stack by accident.

A retired name is not silently ignored, but refusal is narrow: startup fails
only when a retired name is set, the new switch is not, and the model being
loaded really is Gemma4 -- the case where the operator believes the
optimization is on, it is off, and nothing in the log says so. Every other
case warns instead, because the switch is also read from `register_model()`,
the `hcu_model` entry point in `vllm.general_plugins`, whose invocation vLLM
does not wrap in `try`/`except`: raising there would abort startup for a
DeepSeek deployment over a variable that has nothing to do with it.

`VLLM_HCU_GEMMA_BW1000_MAX_SEQS` (default 256) bounds only the global head512
tiling's sequence count. The retired spelling
`VLLM_HCU_GEMMA4_BW1000_MAX_SEQS` is refused in `install()` rather than read
as the default.

## What the switch enables

| # | Change | Mechanism |
|---|---|---|
| 1 | FA prefill gathers paged FP8 KV | No whole-pool FP8->BF16 conversion; only gathered rows are cast |
| 2 | `seqused_k.sum().item()` D2H sync removed | Sums the runner's `seq_lens_cpu_upper_bound` on the CPU |
| 3 | `max(262144, used_tokens)` reservation removed | `_can_use_fa` declines capturing batches (see limits) |
| 4 | Softcap by lookup table | 65536-entry bf16 table reproducing `tanh(x/30)*30`, indexed by bit pattern |
| 5 | SWA reads FP8 as BF16 | `KV_QUANT_MODE=0`, unbound from the `32<=N<=64` gate |
| 6 | Global head512 tiling bound 64 -> 256 | `BLOCK_M=64, BLOCK_Q=8, TILE_SIZE=32`, 8 warps, grid recomputed for `BLOCK_Q` |
| 7 | Fused draft argmax 33..64 -> 33..256 | No `[rows, 262144]` logits tensor; BM follows the row count (<=64 -> 32, 65-96 -> 64, >96 -> 128) |
| 8 | 400-row head bucket plus its tuned entry | Buckets 320 -> {320, 400}; `install_head_bucket` refuses to install a bucket with no tuned entry |

## Precision gate

Target-side changes are covered by element-wise equivalence, not by an
end-to-end evaluation:

| Change | Evidence |
|---|---|
| Softcap table | All 65536 bf16 bit patterns **bit-identical**, including negatives, 254 NaNs and 2 Infs |
| Row bucket and table | GEMM rows are independent; `max_abs_diff` **0**. Rows 320 and 400 are tuned shapes and are not padded |
| Attention launch tuning | Pure tiling and launch change; `max_abs_err` **0.0** |
| SWA FP8 read as BF16 | Every e4m3 value is representable in bf16 and both scales are verified to be 1, so dequantization is the identity; 8 prompts agreed character for character |
| FA prefill gather | In-process element-wise comparison of both paths, 40/40 calls, `max_abs_diff` **0** |

Item 7 is not a bit-identical substitution -- the GEMM's accumulation order
differs from the library's, so a logit on a rounding boundary can change
which of several tied columns wins. Under `temperature=0` this does not change
the output. vLLM's greedy rejection kernel stores `target_argmax_id` for every
verified position and uses the draft token only to decide where acceptance
stops:

```python
token_id = target_argmax_id
rejected = draft_token_id != target_argmax_id
```

A draft-side tie resolving differently therefore changes the acceptance length
-- the speed -- and not the emitted text. The tie rule itself matches
`torch.argmax`: lowest index wins, NaN outranks everything. Over 209 rows of
random and small-magnitude input the indices agreed with the stock path
exactly, with explicit ties, a suppressed tied winner and NaN rows all
resolving the same way.

The exception is `temperature>0`. This path installs only when the draft
returns indices rather than logits (`spec.draft_logits is None`), which is the
sampler's `NO_DRAFT_PROBS` branch: `draft_prob = 1`, acceptance is
`uniform < target_prob`, and an accepted draft token is emitted directly. A
particular seeded sample can therefore differ, while speculative sampling
still targets the same distribution. **Sampling is not validated here.**

### MTP acceptance across the two attention paths

Acceptance is a sensitive probe for target-side numerical change. Stock Triton
**0.5725** against the FA path **0.5762**, **+0.64%**, per round
57.45/57.45/56.85 against 57.45/57.46/57.94 -- below the ~1% line set for this
check.

The probe's sensitivity has independent support: under one unchanged
configuration three separate processes reported acceptance identical to 16
digits, and it splits into different values as soon as target-side numerics
change; a KV layout bug once showed as -15.85%.

Acceptance is not a task accuracy number, and equal acceptance does not prove
two paths produce identical logits -- different logit vectors can share an
argmax.

## Tests

```bash
# Switch contract (new; registered under contract-hcu-gfx936)
python -m pytest tests/gemma4_test/ -q                                  # 32 passed

# Plugin lifecycle, process bootstrap, worker -- what this branch touches
python -m pytest tests/patch/test_plugin_lifecycle.py tests/gemma4_test/ -q
                                                                        # 80 passed
python -m pytest tests/runtime_patch/test_worker_framework_opt.py \
                 tests/patch/test_clean_process_bootstrap.py -q         # 78 passed

# Upstream work the merge brought in
python -m pytest tests/runtime_patch/test_grouped_w4a16_scale_reload.py -q
                                                                        # 12 passed

# Repository static gate
python3 .github/scripts/hcu_ci/compile_changed_python.py --base 05e1ada  # 28 files
```

Live service smoke on a14r4n17 card0, with `VLLM_HCU_GEMMA_BW1000=1` as the
only switch in the environment: all four install lines present, `csv_sha`
`c6a2cac8...`, 28/28 SWA unit-KV audits enabled, zero `Failed to load plugin`
lines, two real requests correct at natural EOS, zero errors in the server log.
The same parameters with the switch unset were also checked: startup
completes, health 200, **zero** `Gemma4 BW1000` install lines, no errors,
requests correct.

One unrelated failure: `tests/kernels/test_kimi_k3_kda_cache_reuse.py`, 13
failed / 12 passed, with `hipErrorNoBinaryForGpu`. That test arrived with this
merge and the module it exercises was changed by the merge, while the
container's `hcu_ops*.so` is a Nov 9 build under an editable install pointing
at the source tree. Every failure is in the `hcu_conv` variants that enter HCU
custom ops; all 12 `triton_conv` variants pass. None of this branch's 13
changed files touch kimi. Needs a matching image to validate.

## Performance protocol

The three end-to-end measurements use different baselines and protocols and
**cannot be multiplied together**.

| Measurement | Covers | Result | Protocol |
|---|---|---|---|
| Fixed work, no MTP | Items 1-3 | **+3.10% TPS, -28.64% mean TTFT** | 447 real prompts, fixed token count |
| Three arms, MTP4 | Items 1-4 | **+4.89% TPS, -12.22% TTFT** | `adbd6c..aabb396` |
| Single-arm control, MTP4 C80 | Items 6-8 | **median TPOT -10.19%, TPS +9.88%** | 447x128, control run at `d4c7a51`, **2 rounds only** |

Item 5 was never measured end to end on its own: it was already active in the
third measurement's control run. **There is no "versus the switch off" or
"versus the factory image" total**; that needs a two-arm B->C run on this
branch's HEAD, which has not been done.

Operator-level figures, for locating work rather than as end-to-end gains:
item 1 30.14 ms -> 0.245-0.345 ms; item 5 1.24-1.73x across the range; item 6
3.67x at N=80 (2.75x at 16, 4.51x at 256); item 7 2.20x at 76-80 rows (1.52x
at 33, 4.93x at 256); item 8 4.76-5.61x at 348-400 rows.

## Known limits and fallbacks

1. **The rocBLAS table is specific to this image.** The Validator rows in
   `gemma4_head_results0.csv` are compared character by character against the
   PyTorch, HIP, hipBLASLt and rocBLAS versions and `GCN_ARCH_NAME`, and two
   of those version strings carry `-dirty`. When the comparison fails,
   `load_head_table()` raises and a server with the switch on does not start.
   A different container image or ROCm component version needs the table
   re-tuned.
2. **Item 3 narrows behavior**: `_can_use_fa` declines capturing batches and
   leaves them to Triton. The old 262144 reservation was not a provable bound
   -- 80 requests of 4000 tokens is 320k rows, and under CUDA graphs the
   overrun is a silent out-of-bounds write. D512 paged decode is unsupported
   anyway, so FA has only ever served eager prefill and mixed batches; no
   existing capability is lost.
3. **`temperature>0` is not validated.** See the precision gate.
4. Item 6 keeps its floor of 9. The sweep found 1.57x at N=1 and no loss
   anywhere, but the floor was left alone to keep one change per commit.
5. **No bucket above 400.** At C80 the row count tops out at 80x5=400, and
   tuning 512 gets the process OOM-killed.
6. Items 6-8 have **2 end-to-end rounds**, where the protocol asks for 3, and
   the three cannot be separated from each other.
7. Fixed-work A/B runs use `ignore_eos` synthetic output and **no task-level
   accuracy evaluation**. Under `temperature=0` the output is determined by the
   target's argmax and every target-side change has element-wise equivalence,
   so such a run would measure the same model.
8. Prefill's `fused_moe_kernel` (20.2%, 725.7 ms) is untouched.
   `VLLM_TUNED_CONFIG_FOLDER` is field-for-field identical to the default
   formula at M>=1024, so it cannot reach prefill; on the decode side a
   six-session alternating A/B showed its operator-level 1.18-1.26x **not
   reaching end-to-end throughput**, and that arm changes output -- MTP
   acceptance was not reproducible across processes. Not adopted.
9. **There is no decode-side kernel profile**, only a prefill one, while all
   three widened gates are on the decode side.
