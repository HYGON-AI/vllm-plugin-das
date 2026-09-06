# Model and dataset configuration

This directory will contain YAML configuration only, not checkpoints or large
datasets.

Resource resolution:

1. explicit pytest option (`--model-root` or `--dataset-root`);
2. `VLLM_HCU_TEST_MODEL_ROOT` or `VLLM_HCU_TEST_DATASET_ROOT`;
3. an absolute path declared by an explicitly selected local configuration;
4. remote resolution only when `--allow-model-download` is set.

CI model/nightly runs should use `--strict-test-resources` so a missing
checkpoint or dataset fails instead of silently skipping.

Available local configurations:

- `deepseek_r1_gsm8k_evalscope.yaml`: DeepSeek-R1 Channel-FP8 W8A8 server
  plus EvalScope GSM8K.
- `deepseek_v4_flash_0731_dspark_humaneval.yaml`: DeepSeek-V4-Flash-0731
  Channel-FP8 TP8 and unified DP8+EP8 DSpark server profiles plus strict
  ModelScope HumanEval-32 acceptance.
- `deepseek_v4_flash_0731_int8_dspark_humaneval.yaml`:
  DeepSeek-V4-Flash-0731 Channel-INT8 TP8 and unified DP8+EP8 DSpark server
  profiles plus strict ModelScope HumanEval-32 acceptance.
- `deepseek_v4_int8_humaneval_evalscope.yaml`: TP4 feature-off/feature-on
  profiles for the local DeepSeek-V4 Flash Channel-INT8 checkpoint. Both
  profiles require exactly 32 HumanEval predictions and reviews; the enabled
  profile must execute the LightOp sqrt-softplus route.
- `glm52_pcp_humaneval_evalscope.yaml`: GLM-5.2 Channel-FP8 W8A8
  model-runner-v2 server with TP=4, PCP=2, EP, and EvalScope HumanEval (32
  deterministic samples).
- `qwen3_8b_gsm8k_evalscope.yaml`: Qwen3-8B server plus EvalScope GSM8K.
- `qwen35_9b_gsm8k_evalscope.yaml`: Qwen3.5-9B server plus EvalScope GSM8K.
- `qwen3_vl_8b_mmmu_evalscope.yaml`: Qwen3-VL-8B-Instruct server plus
  EvalScope MMMU multimodal accuracy.
- `qwen36_35b_a3b_humaneval_evalscope.yaml`: TP1 feature-off/feature-on
  profiles for the local Qwen3.6-35B-A3B checkpoint, with a 16-token scheduler
  bound matching the accepted LightOp W16A16 range and exact HumanEval-32
  artifact checks.
- `qwen36_27b_humaneval_evalscope.yaml`: TP1 feature-off/feature-on profiles
  for the local Qwen3.6-27B checkpoint and strict gated-RMSNorm HumanEval-32
  acceptance.
- `qwen35_35b_a3b_w8a8_humaneval_evalscope.yaml`: TP1 feature-off/feature-on
  profiles for the quantized Qwen3.5 checkpoint. Its FP32 gated-norm weight
  makes this an explicit strict-BF16 fallback and accuracy control.
