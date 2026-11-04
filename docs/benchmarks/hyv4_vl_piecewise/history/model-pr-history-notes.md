# Stage 3 decisions

Use BBuf SKILLS at 371dee04ee30ed918b240823b2a12c139d1a67fa:
- skills/llm-torch-profiler-analysis/SKILL.md
- model-pr-optimization-history/SKILL.md
- model-pr-optimization-history/vllm/hunyuan4/README.zh.md

Previous all-eight-rank traces are reused as diagnostic leads, not fresh stage-3 timing. No live profiler is configured after the observed stage-2 graph/profiler VMFault. Current installed source is authoritative. Historical #57526 and #57811 remain discovery leads, not HCU compatibility proofs; no code is copied.

Installed config/vllm.py explicitly permits PIECEWISE with breakable graphs and disables torch.compile. Capture sizes must include image-prefill token counts; [1,2] alone cannot test that lane. The candidate uses [1,2,128,256,512,768,1024]. Actual modes/shapes must be recorded on every rank; padding can change GEMM shapes and task predictions, so no precision claim before a full same-weight pair.

The HYV4VL adapter does not implement SupportsEncoderCudaGraph; merely toggling cudagraph_mm_encoder is not a valid ViT optimization. An implementation would need grid-aware capture/replay inputs, exact per-image TP shape preservation, full-embedding checks and end-to-end task accuracy.

The installed channel-FP8 Triton MoE has no BW1101 E256/N256 tuning file. An offline TP0 gate/up GEMM tile probe may establish local opportunity using actual weights and synthetic activations/routing; it cannot establish whole-MoE or serving speed/accuracy. Keep K=128 and reject candidates with any sampled output difference. No tuning is applied to serving.

Installed experts/triton_moe.py confirms that gate/up and down GEMMs reuse the same tile config. Its stock fused activation/quant branch is gated on block_shape=[128,128], so the actual channel-FP8 route retains separate clamped activation and per-token quantization. This is a source-backed fusion lead only; positive clamp, BF16 intermediate rounding and native FP8 scaling are required and no replacement is tested.
