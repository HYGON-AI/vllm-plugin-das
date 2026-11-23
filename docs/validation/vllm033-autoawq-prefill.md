# AutoAWQ packed-layout prefill candidate

The observed public LightOp GEMM already uses dequantization followed by
`F.linear` for M>64. Its dequantization dominates large prefill calls. The
Plugin retains that boundary and its small-M public vendor path, but can
convert validated packed storage directly to a temporary FP16 KN matrix for
ordinary `torch.matmul`. Core retains its configuration, checkpoint loader,
quantization registry and complete fallback. This change requires no Core
patch or new provider switch.

The reader supports the existing group-128 FP16 dense candidate shapes. It
decodes the K-axis nibble order and packed zero/scale halfwords, using the
same arithmetic as official AWQ dequantization. Post-load admission compares
every repacked int32 value with the expected tile layout before enabling the
reader. A different compatible vendor layout still uses the public GEMM;
matching function signatures or tensor shapes alone cannot enable the reader.
Reload rejects a change in reader mode before publishing new parameters,
since compiled calls and captured graphs retain the previous boolean.

Conversion allocates one temporary K-by-N FP16 matrix per large-M call, as
does the previous vendor branch. It retains no second checkpoint/weight copy.
The output remains non-aliasing. Both reader modes use the same dispatcher
custom op and FakeTensor rule; three-argument callers retain the vendor path.

The conversion grid traverses N blocks before K blocks, following adjacent
regions of the packed backing. The physical packing tile, FP16 KN output
strides and ordinary Torch matmul stay unchanged. For admitted shapes with
K>=4096, N>=7424 and a 256-row packing tile, the reader uses 128 columns and
eight warps per program. Other shapes
retain 64 columns and four warps. A full admitted-shape conversion-plus-GEMM
prototype motivated this policy: applying the wider launch to every shape
regressed smaller shards. Installed Graph, model and serving performance
results must still be checked on this implementation; prototype timings do
not close the previously observed TTFT regression.

Time conversion and downstream matmul together when comparing temporary
layouts. A transposed KN view can make conversion faster while selecting a
slower BLAS path. In serving traces, compare the same prefill geometry:
whole-trace dequantization totals can also contain different decode paths.

The CPU regressions cover actual packed-value admission and contamination,
the real Core reload boundary, FakeTensor shapes, and M=64/65 dispatch. The
native suite covers all 16 admitted shapes, both packing tiles, exact
dequantization against Core, fullgraph compilation, and fixed-address Graph
replay with changing inputs. Its six M values include the vendor boundary's
large-M branch. Native output checks retain the original numerical limits.

Component and prototype results do not qualify this production candidate.
Installed-wheel checks, model output/HumanEval, normal exit within the
original deadline, throughput and TTFT each require their own results bound
to the final source. In particular, faster dequantization does not establish
a serving latency improvement or repair a previously observed shutdown error.

## Serving comparison

Keep `VLLM_HCU_USE_CUSTOM_OPS` and the selected RMSNorm policy identical in
both processes; switch only `VLLM_HCU_USE_LIGHTOP_AWQ`. Toggling the master
along with AWQ can select different unrelated providers. Retain that older
experiment under its original protocol, and bind the new comparison to its
own source, installed wheel, checkpoint and request hashes. Observe the actual
AWQ methods, RMSNorm classes, attention implementation, Graph and prefix
configuration in each arm. A fixed switch alone does not establish identical
execution outside AWQ.

Keep read-only worker inspection modules in a private evidence directory,
outside the installed packages. When using `PYTHONPATH` for the worker
extension, launch the real CLI with `-P`, an explicit helper path and a neutral
working directory; `-I` ignores that path. Hash the helper before and after
the experiment and independently verify installed payloads and RECORDs.
Observe cleanup registration from already loaded modules in both canonical
and legacy Adapter namespaces. Its expected value follows the actual tested
Adapter source; the normal-exit deadline and diagnostics remain separate gates.

Before each arm, recheck the physical device identity, free memory, other
clients' activity and listener ports. Resource occupancy delays the experiment;
it does not qualify the new model pair or change historical failure results.
