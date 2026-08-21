# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU adapter for the upstream DeepSeek-V4 DSpark draft model."""

import functools
import torch

from vllm.model_executor.layers.mhc import HCHeadOp
from vllm.models.deepseek_v4.nvidia import dspark as _dspark


def _install_dynamo_metrics_compat() -> None:
    """Work around Torch/AITER logging-config metrics incompatibility.

    AITER stores logger callables in Dynamo's ``ignore_logging_functions``.
    This Torch build still excludes the old ``ignore_logger_methods`` name
    when JSON-encoding metrics, producing a noisy TypeError after every
    compiled kernel.  Preserve compilation behavior and suppress only that
    metrics serialization failure.
    """
    import torch._dynamo.utils as dynamo_utils

    original = dynamo_utils._get_dynamo_config_for_logging
    if getattr(original, "_vllm_hcu_aiter_metrics_compat", False):
        return

    @functools.wraps(original)
    def safe_get_dynamo_config_for_logging():
        try:
            return original()
        except TypeError as error:
            if "function is not JSON serializable" not in str(error):
                raise
            return None

    safe_get_dynamo_config_for_logging._vllm_hcu_aiter_metrics_compat = True
    dynamo_utils._get_dynamo_config_for_logging = safe_get_dynamo_config_for_logging


class DSparkDeepseekV4Model(_dspark.DSparkDeepseekV4Model):
    """Use the HCU decoder, mHC operators, and cache insertion kernel."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        _install_dynamo_metrics_compat()
        # ModelRegistry inspects model classes in a clean subprocess where HCU's
        # canonical module exchanges are intentionally not armed.  Keep HCU-only
        # imports out of module scope and resolve them when the worker constructs
        # the model after platform bootstrap.
        from vllm.models.deepseek_v4.amd.model import (
            DeepseekV4DecoderLayer as AMDDeepseekV4DecoderLayer,
        )
        from vllm.platforms import current_platform

        class HCUDeepseekV4DecoderLayer(AMDDeepseekV4DecoderLayer):
            """Adapt the AMD decoder constructor to the DSpark layer API."""

            def __init__(self, vllm_config, prefix):
                config = vllm_config.model_config.hf_config
                topk_indices_buffer = torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.index_topk,
                    dtype=torch.int32,
                    device=current_platform.device_type,
                )
                super().__init__(
                    vllm_config,
                    prefix,
                    topk_indices_buffer=topk_indices_buffer,
                    aux_stream_list=None,
                )
                # NVIDIA's DSpark checkpoint loader branches on this flag.
                # The AMD decoder always uses the regular FusedMoE layout,
                # whose equivalent value is False.
                self.ffn.use_mega_moe = False

        def make_deepseek_v4_expert_params_mapping(num_experts):
            return [
                (
                    "experts.w13_"
                    if shard_id in ("w1", "w3")
                    else "experts.w2_",
                    f"experts.{expert_id}.{weight_name}.",
                    expert_id,
                    shard_id,
                )
                for expert_id in range(num_experts)
                for shard_id, weight_name in (
                    ("w1", "w1"),
                    ("w2", "w2"),
                    ("w3", "w3"),
                )
            ]

        _dspark.DeepseekV4DecoderLayer = HCUDeepseekV4DecoderLayer
        _dspark.make_deepseek_v4_expert_params_mapping = (
            make_deepseek_v4_expert_params_mapping
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.hc_head_op = HCHeadOp()

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = None
        layer = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        assert layer is not None
        if layer.use_fused_mhc:
            hidden_states = layer.hc_post(
                hidden_states, residual, post_mix, res_mix
            )
        return self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


def _insert_context_kv(attn, kv, positions, slot_mapping) -> None:
    """Insert DSpark context KV using the HCU fused RoPE/FP8 cache op."""
    import lightop

    num_tokens = kv.shape[0]
    dummy_q = torch.zeros(
        (num_tokens, attn.n_local_heads, attn.head_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    swa_cache = attn.swa_cache_layer.kv_cache
    lightop.op.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        dummy_q,
        kv,
        swa_cache.view(swa_cache.shape[0], -1),
        slot_mapping,
        positions.to(torch.int64),
        attn.rotary_emb.cos_sin_cache,
        attn.eps,
        attn.swa_cache_layer.block_size,
    )


_dspark.DSparkDeepseekV4Model = DSparkDeepseekV4Model
_dspark._insert_context_kv = _insert_context_kv


class DSparkDeepseekV4ForCausalLM(_dspark.DSparkDeepseekV4ForCausalLM):
    """Upstream DSpark semantics backed by HCU model components."""

    def _finalize_moe(self) -> None:
        """Finalize only the NVIDIA MegaMoE layout when one is present."""
        for layer in self.model.layers:
            ffn = layer.ffn
            if getattr(ffn, "use_mega_moe", False):
                ffn.finalize_mega_moe_weights()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        """Expose AMD channel-FP8 scales under DSpark's expected names.

        Compressed-tensors registers HCU linear scales as ``weight_scale``,
        while the NVIDIA DSpark checkpoint loader maps ``.scale`` to
        ``weight_scale_inv``.  Publish aliases only while loading so normal
        module traversal retains PyTorch's standard parameter names.
        """
        parameters = super().named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        for name, parameter in parameters:
            yield name, parameter
            if not getattr(self, "_loading_dspark_weights", False):
                continue
            if name.endswith(".weight_scale"):
                yield name + "_inv", parameter
            elif name.endswith("_weight_scale"):
                yield name + "_inv", parameter

    def load_weights(self, weights):
        self._loading_dspark_weights = True
        try:
            return super().load_weights(weights)
        finally:
            self._loading_dspark_weights = False
