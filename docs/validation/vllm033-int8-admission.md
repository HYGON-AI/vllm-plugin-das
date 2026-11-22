# INT8 expert admission review

The subsequent [full-model comparison](vllm033-int8-model.md) completed:
normalized32/32 in both arms, with its frozen ASM wheel and compiler-policy
limits preserved. The pending status below records the admission-review snapshot.

Production fix: `ec01e209`, on existing [PR330](https://github.com/HYGON-AI/vllm-plugin-das/pull/330),
target `v0.33.0`. [Candidate and evidence hashes](../../validation/v0.33.0/int8-admission-gfx936-candidate.json).
AI assistance was used. Core runtime and Adapter production are unchanged.

The INT8 subclass inherited the official AITER parent's SITU and SwiGLU-OAI
support, but its vendor wrapper does not forward their beta/clamp parameters.
It also admitted expert biases allocated by v0.33 compressed-tensors loading,
while forwarding neither bias to native execution nor to its fallback. These
contracts could therefore be accepted with incomplete mathematics.

Reject expert bias during official selection, before weight loading/conversion.
Declare only the SiLU/GELU activation contracts that this wrapper forwards.
Direct-provider checks reject bias and unforwarded activations before consulting
the vendor or launching kernels. Explicit unsupported AITER requests receive an
error; official Triton and automatic selection remain available. This does not
claim new vendor support for biased or parameterized experts.

Validation:

- Nine new portable regressions fail before the fix; 55 focused cases pass after.
- Full portable suite: **262 passed, 66 skipped**. Ruff F and diff checks pass.
- The actual installed v0.33 parent/subclass admission is exercised without a
  stub or device execution. The prior wheel accepts all ten combinations of
  five activations and bias on/off; the new wheel accepts only SiLU/GELU without
  bias. The source probe and both raw results have retained hashes.
- Rebuilt installed native INT8 suite: **11 passed**, 14 deprecation warnings.
  This includes eight EP cases and 24 changed-input Graph replays at the original
  numerical limits. All 14 Python files match source, wheel and the separate
  installed review environment. The ASM mathematics is unchanged from `529855a5`.

The separate full Qwen3.5-35B-A3B-W8A8 experiment uses the frozen `529855a5` wheel,
identical checkpoint, Triton attention and dense INT8, comparing official Triton
versus explicit AITER MoE. Graph and prefix remain enabled, MTP is off, and both
arms explicitly disable Inductor combo kernels. Its native stage1/stage2 ASM
code objects have loaded, but full-model quality is still pending at this
snapshot. No result is transferred to an untested build or default compiler
policy.

**Merge decision remains HOLD.** The earlier E5M2 and hybrid-model HumanEval
differences and unchanged vendor-Triton strict numerical row remain open.
See the [original review](vllm033-review.md) and [ASM math follow-up](vllm033-int8-math.md).
The current fixes and passing operator checks do not waive those gates or mark
the full upgrade as last-good.
