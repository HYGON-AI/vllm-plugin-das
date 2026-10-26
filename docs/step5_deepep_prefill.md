# Step5 Prefill with DeepEP and DeepGEMM

The optional Step5 channel-FP8 path uses DeepEP NORMAL MNVL dispatch-scatter,
fixed E23 replicas and the existing DeepGEMM WASP kernels. On four gfx938 hosts
with four cards each, the measured configuration reached 45,621.7 input tokens/s.
This is a single warmed formal run, rather than a repeated-run average.

## Execution and memory ownership

The kernel factory snapshots Step5 ownership and DBO configuration during model
construction. It reserves two manager-owned DeepEP buffers before KV planning.
Each microbatch owns its communication storage and retains its input/output
references until the asynchronous return receiver completes. The return copy
waits on the communication stream which performs that copy.

Dispatch-scatter writes expert-major FP8 inputs, scales and inverse indices.
The expert kernel uses the actual post-dispatch padded counts to bound runtime
work. Startup workspace planning retains the conservative bound. Existing
SwigluStep/FP8 quantization, DeepGEMM G1/G2 and the LightOP Gather are preserved.
The rejected combine-gather prototype is not part of this implementation.

All new controls are disabled by default. `HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT=1`
enables construction-time output accounting; `HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR=1`
requires Step5 EP16 NORMAL MNVL and configured DBO. `HCU_STEP5_DEEPEP_FUSED_SCATTER=1`
also requires TP1 channel-FP8 DeepGemmExperts. `HCU_STEP5_DEEPEP_COMPACT_ROWS=1`
selects the bounded runtime view only for the owning Step5 model.

`HCU_STEP5_DEEPEP_RUNTIME_CONFIG` selects an external communication profile.
Its versioned fields are `num_sms`, `dispatch_send`, `combine_send`, `recv_tokens`,
`async_layout` and `overlap`. Change that file only between drained workloads and
keep all16 ranks consistent. The selected buffers must fit the profile's NVL
capacity requirement. Machine tuning tables and static expert maps remain
external assets; they are not embedded in the implementation.

## Dependency provenance

The tested DeepEP fork is based on0d4a8db11523475c831cdd0ab10f6ad4fec27d98.
It adds subgroup-correct MNVL receive scatter, exact-count index resizing and a
stream dependency for resize preservation copies. Its source, two-file patch,
build helper and source manifest are supplied in the separate local dependency
bundle. The tested extension SHA256 is
`b8f7274c732daa73a2b91ee19d7a9165518aa6faf086db818344d70835d07e3c`.

That experimental build uses `DISABLE_ROCSHMEM` and supports NORMAL MNVL;
its RDMA and low-latency modes are not qualified. The image's general DeepEP
installation remains intact. Select the tested fork only for this Step5 profile.
DeepGEMM, LightOP, AITER source, FlashAttention, vLLM and PyTorch were unchanged.
There is no `LD_PRELOAD` or `LD_AUDIT` injection. The existing plugin native
extension is unchanged; this change modifies Python integration only.

## Validation

The frozen full GSM8K test completed1319 requests with1277 strictly correct
numeric answers (96.8158%). All1319 finished with `stop`; there were no request
errors or output-limit finishes. Accuracy uses standard MTP3, temperature1,
top-p0.95 and seed42, matching the prior frozen test. Synthetic forced acceptance
is used only for performance measurements.

Performance keeps DP16/TP1/EP16, C64,65,536 input tokens and one output token,
`max-num-batched-tokens=8192`, `max-model-len=70000`, FP8 KV and prefix caching off.
MTP3 uses the synthetic rates `[0.838992,0.598322,0.362686]`.
The selected communication budget is SM18, dispatch8/combine32, receive512,
with asynchronous layout and independent-buffer DBO overlap.

Full GSM8K exercised the frozen measured source before cleanup. Cleanup removes
inactive return-fusion code, moves construction policy into one helper and adds
an explicit EP16/MNVL gate; GEMM math and the enabled execution path are unchanged.
CPU compatibility coverage includes151 tests and8 subtests. The cleaned version
passed a real16-rank GPU smoke with208 checks, including DBO output canaries,
empty-rank routing and conservative startup planning. Post-cleanup formal repeats were45,578.4 /45,598.6 /45,549.0tps
(mean45,575.4), all64 complete and zero preemption. A calibrated16-rank
trace captured6 steps each and aligned5 common global steps.

E23 removed the persistent rank0 tail: rank0 was latest in58/880 observed G2
waves; ranks2,4,3 were most often late. This ordered-wave comparison includes
upstream attention delay and is not an isolated GEMM latency attribution.
Heavy ranks spent roughly730-760ms in G1/G2 and14-48ms in exposed communication.

SM12 did not improve throughput. SM18 dispatch8/combine8 produced one47,967.0tps
round but two repeats were45,796.6 /45,795.2tps;47,967 is not a sustained score.
The48,000tps follow-up target has not yet been achieved. Complete trial data
are retained in the accompanying aggregate result JSON.
