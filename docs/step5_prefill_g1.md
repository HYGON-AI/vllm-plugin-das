# Step5 prefill G1 specialization

Historical measurement record. Current cleanup removes the retired Triton G1
experiment and keeps tuning data external; see [current source contract](step5_pd_verification_20261024.md#source-cleanup-after-the-recorded-measurements).

`HCU_STEP5_PREFILL_G1=1` enables a shape-specialized Triton projection in the
existing channel-FP8 EP path. It consumes the original contiguous NK weights
and expert-sorted buffers. FP32 scaling, the BF16 projection boundary,
SwigluStep, activation quantization, GEMM2 and Directsum remain unchanged.
There is no additional native library or worker extension.

The implementation is limited to gfx938, E22, N3072/K4096, top-k 8, a 256-row
sorting block and a global MoE microbatch of 65536 tokens. The latter is the
DP16 DBO case; the per-DP `--max-num-batched-tokens` remains 8192. Unsupported
shapes, layouts, dtypes and devices use the existing projection. The feature
is disabled by default and allocates no temporary tensors or weight copy.

The source benchmark runs eight independent FP8 weight/input pairs through
the complete native MoE pipeline. It checks changed inputs, zero inputs,
activation saturation, an empty local route set and Directsum overflow before
paired eager and CUDA graph timing:

```bash
python benchmarks/step5/check_prefill_g1_specialized.py \
  --run-gpu --fixture fixture.npz --fixture-meta fixture.json \
  --output /tmp/step5-prefill-g1-result.json
```

The fixture contains `[65536, 8]` physical expert IDs; its metadata identifies
the local E22 range and verifies the NPZ SHA256. Run this isolated benchmark
on a spare gfx938 GPU. It reports operator latency, not model tokens/s.

On the 1021 image, three frozen routing distributions passed 102 bit-exact
complete-MoE comparisons with approximately 2.5–2.7% lower MoE latency. The
normal-package entry passed another 34 bit-exact comparisons and measured
6.953 ms versus 6.775 ms on
the middle routing distribution. Separate projection tests cover all-local
routing and sorted addresses beyond 2 GiB. These isolated operator results do not establish the 45,000 tokens/s target.
Later full-model measurements are recorded below.


## Later full-model evidence (2026-10-23)

Three clean synthetic rounds measured 42045.84 tokens/s (SD28.97), versus
41542.66 tokens/s for the same-source flag-off control (+1.21%). G1 remains
opt-in. Separate standard-MTP GSM100 runs scored96/100 twice; fresh off-control
runs scored98/100 twice. These short prompts do not hit M65536, so the score
difference alone does not establish a specialization defect or equivalence.
Long24K16 and long64K16 smoke scores were15/16 with zero request errors.
See `step5_pd_optimizations_20261023.md` for snapshot scope and limitations.
