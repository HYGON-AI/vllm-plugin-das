# HY V4 ModelOpt MXFP8 Blockwise Bring-up Plan

**Goal:** Load and serve `/models/Hy4-preview-FP8-Testing` with TP=8 on HCU, verify generation, and run a reproducible accuracy sample.

## Task 1: Regress the excluded LM-head layout failure

- Add a focused test that constructs an excluded ModelOpt MXFP8 `ParallelLMHead` under HCU's default NN layout.
- Assert that its storage remains `[vocab_shard, hidden]` and that the standard vocab weight loader accepts a checkpoint tensor.
- Run the test before the fix and capture the expected failure.

## Task 2: Fix HY V4 LM-head quantization routing

- In `HYV4ForCausalLM`, do not pass a quantization config to `ParallelLMHead` when that exact prefix is excluded by the config.
- Preserve quantized-head behavior and non-HY-V4 linear NN layout behavior.
- Run the focused test and the HY V4/unit quantization suites.

## Task 3: Bring up the real blockwise checkpoint

- Serve with TP=8 and the portable Triton MXFP8 backend while retaining serialized MXFP8 weights at load time.
- Verify `/health`, model listing, and at least one deterministic chat completion.
- If another real-checkpoint failure appears, repeat diagnosis and add a regression test before fixing it.

## Task 4: Measure accuracy

- Install EvalScope if needed without modifying project dependency files.
- Run a documented, bounded benchmark against the live OpenAI-compatible endpoint.
- Record dataset, sample count, generation settings, score, and any backend/performance limitations.

## Task 5: Final verification and handoff

- Re-run all targeted tests and inspect the final diff for overlap with pre-existing user changes.
- Leave the service state explicit and provide exact server/client/evaluation commands.
