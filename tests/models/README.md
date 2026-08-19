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
- `glm52_pcp_humaneval_evalscope.yaml`: GLM-5.2 Channel-FP8 W8A8
  model-runner-v2 server with TP=4, PCP=2, EP, and EvalScope HumanEval (32
  deterministic samples).
- `qwen3_8b_gsm8k_evalscope.yaml`: Qwen3-8B server plus EvalScope GSM8K.
- `qwen35_9b_gsm8k_evalscope.yaml`: Qwen3.5-9B server plus EvalScope GSM8K.
- `qwen3_vl_8b_mmmu_evalscope.yaml`: Qwen3-VL-8B-Instruct server plus
  EvalScope MMMU multimodal accuracy.
