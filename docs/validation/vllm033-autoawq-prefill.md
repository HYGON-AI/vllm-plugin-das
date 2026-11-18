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
