# Instance-owned V41 Channel-FP8 router projection

The existing owned FP4 decoder construction seam binds one exact GateLinear
instance after native construction. Native classes, checkpoint parameter names,
weight storage and TopK are unchanged. Unknown instance forwards, quantized
weights, other router tiers, DeepSeekV4 BF16-output policy, non-FP32 outputs,
TP/PCP/DCP !=1 and the existing custom-ops disable switch retain native behavior.
The optional Adapter owns tensor/device/shape qualification and the Triton
kernel. Unsupported runtime metadata calls the saved original bound method;
in-place weight reloads read the same live Parameter. return_bias follows the
native `_return` contract. There are no process-global Core replacements.

The native old GSU1 projection's reduction order can be reproduced on the
qualified generic row domain; GSU8 atomic cases remain native. The kernel does
not embed batch7, input65536, output1024 or a fixed graph batch42/48. See Adapter
`docs/v41-router-stagger.md` for the exhaustive survey and rejected proposal.

Validation artifacts live under `run/dsv4.1/033/fp8_optimization_20261123`:
`ROUTER_STAGGER_GATE.json`, the original failed all-row/native-control evidence,
maintained kernel/contract/L0 checks, real checkpoint consumer and qualified
126-scenario gate. Whole model HumanEval164 / trace / serving results must be
read separately; isolated kernel timing does not establish end-to-end speedup.
