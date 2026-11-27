# V4.1 Channel-FP8 DSpark shared activation

The native draft constructor creates its decoder layers independently of the
target's `HcuV41Fp4DecoderLayer`. Consequently the target's shared activation
binding did not reach any draft layer: the actual candidate profile still
contained three native clamped SiLU kernels, while all 40 target layers used
the fused activation/token-FP8 kernel.

The owned draft constructor now binds the same FP8 consumer contract after
native construction, using its actual private construction config. This does
not replace the native constructor, weight loading, routing, or AG-RS policy.
It visits the actual draft layer list, with no assumption about layer count,
batch size or input length. CT INT8 stays on its original route. The existing
architecture, topology, MLP, activation and quantizer guards still apply;
unsupported consumers keep their original activation.

Local validation: 12 checks passed on gfx938 with the installed vLLM 0.33.0
wheel. These include actual native MLP/Channel-FP8 scheme construction with
1/3/5 draft layers, DP1/DP8 construction config ownership, idempotence and
mega-MoE fallback. Real FP8 down-projection output is byte identical for
M=1/7/35/42/48/257 in eager execution and three changed-input graph replays.
The previously validated Adapter provider is reused without changes.

Full HumanEval164, unprofiled serving performance and actual draft trace
validation must be recorded separately before claiming a model-level gain.
