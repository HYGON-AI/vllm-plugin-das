# Public INT8 EP through canonical local expert routing

User asks full v0.33 upgrade, official-first, reusable vendor-only capabilities,
MoE AITER priority, original HumanEval32 and normal autonomous reviewed merges.
This extends existing explicit public INT8 backend; it does not replace the
full148-feature goal. Current source PR373 d135a81f, targeta79c44b6; fresh receipt
is evidence/v033-int8-ep-public-candidate-20261110/remote-start.json relative
to the upgrade root. New isolated branch starts at actual fetched PR373 commit.
Frozen runtime/evidence from earlier cohorts is never changed.

## Approaches and agent implementation decision

1. Preserve binary ownership mask and cache canonical-map verification in loader
   hooks. This retains current sorting but rejects useful local permutations and
   needs tensor-lifetime/mutation guards through loading/reload/Graph.
2. Remap global route IDs through the actual Core canonical map before existing
   public native sorting. Native local domain is actual local weight expert count;
   remote/padding routes are -1 and existing sorting ignores them. This preserves
   local permutations, remains device-side/Graph-compatible, and needs no private
   state/cache or registry seam. Chosen as the narrow reusable implementation.
3. Add a vendor native canonical-map ABI. Broader binary rebuild/distribution work
   is unnecessary if existing public sorting plus a small torch gather is correct.

Choice2 is an agent implementation decision within the user's already-authorized
MoE adaptation. Core continues to own CT validation/load/postprocess, actual
factory/selector/modular kernel/PF/runner/router and collectives; Adapter owns
platform/process health. Plugin owns INT8 stage composition and qualified vendor
plan compatibility. No new Core seam, copied oracle or platform spoof.

## Routing contract

ASM boundary receives canonical -1/local-index map unchanged, including arbitrary
valid local-index permutations. CK still receives its existing binary mask ABI;
non-ASM vendor paths and complete official fallback keep their existing semantics.
For ASM, convert global IDs to local IDs via safe gather: IDs outside global
range, including -1, stay -1 and cannot wrap to another expert. Sorting domain
becomes w1.shape[0], with no native mask; sorted token/route encoding and original
routing weights remain untouched. Native stages write locally owned routes only;
remote workspaces stay zero. Native sorting requires topk<=local domain;
localE<topk (including zero local experts) must take official fallback before
native launch. Range checks use original int64 IDs before narrowing and use
global_num_experts even when the map includes additional entries.
Identity/linear/RR/reversed/random permutations,
all-remote and all-owned routing and changed-input Graph require direct/reference
checks. No Python readback of route values, private caches or vendor CSV edits.
The native sorter requires unique experts per token under standard Core top-k;
R1/R2 duplicate-overwrite failures remain preserved and are not waived.

## Public EP and plan selection

Keep production EP refusal until the corrected routing has device qualification.
Then separately implement supported EP admission and honest plan selection.
Initial required model geometry is gfx936/BF16/global256/local128/H2048/N512/top8,
TP-only EP2, DP/PCP/SP1, no EPLB/bias/padding/LoRA/parameterized activation/shared
fusion. Ordinary non-EP support remains unchanged. The implementation must not
pretend E128 vendor config lookup succeeded: record actual local misses, and
label any qualified E256 global routing profile reuse as compatibility selection,
not a locally tuned profile. Unsupported plans/shapes take whole-call official
Triton before packing/launch; post-native errors propagate. Plan compatibility
is bounded by actual qualification; expand coverage through tests, not assumed
safe profiles. Keep all remaining architectures/topologies/features explicitly
open in the full migration matrix.

Required integration: actual public CT factory/loader and official runner on
real two ranks; canonical map/weight/scales, source/installed identity, genuine
NoDPEP/deferred input and final collective, changed Graph, checkpoint load/reload
and original model HumanEval32. No admission from a hand-driven diagnostic alone.
Full model defaultGraph/HE32 and lifecycle/performance remain distinct gates.
Current Core75/76 merged changes must be included or explicitly excluded in each
new artifact identity. CI image/independent approval rules remain mandatory.

## Evidence and non-goals

R3 one-device local stages and R4 two-rank manually composed native/PF/reduce
passed scoped strict limits relativeL2<=.005/maxabs<=.02/cos>=.9999. Their reversed
local-index negative exposed binary-mask information loss. They are evidence for
this missing routing behavior, not public EP/model admission. No new MR until
fresh remote actual code/open-diff audit, review and required tests; no CI bypass.
