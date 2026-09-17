# SPDX-License-Identifier: Apache-2.0
"""Compatibility fixes for Kimi-K3's wider Kimi-K2.5 vision QKV layout."""

from __future__ import annotations

from types import ModuleType
from copy import deepcopy

import torch
from torch import nn


def _make_norm(module: ModuleType, norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "rmsnorm":
        return nn.RMSNorm(hidden_dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    raise NotImplementedError(f"Unsupported Kimi vision norm type: {norm_type}")


def install_kimi_k25_qkv_layout_compat(module: ModuleType) -> None:
    """Restore K3 vision layouts missing from the legacy K25 module."""

    if getattr(module, "_hcu_kimi_k25_qkv_layout_patch_applied", False):
        return

    base_layer = module.MoonViTEncoderLayer
    base_encoder = module.MoonViT3dEncoder
    base_tower = module.MoonViT3dPretrainedModel
    base_projector = module.KimiK25MultiModalProjector
    base_projector_forward = module.mm_projector_forward

    class HcuMoonViTEncoderLayer(base_layer):
        def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            quant_config=None,
            prefix: str = "",
            *,
            activation=module.F.gelu,
            attn_bias: bool = False,
            qkv_hidden_size: int | None = None,
            norm_type: str = "layernorm",
            mlp_type: str = "mlp2",
            linear_bias: bool = True,
        ) -> None:
            nn.Module.__init__(self)
            self.use_data_parallel = module.is_vit_use_data_parallel()
            self.num_heads = num_heads
            self.hidden_dim = hidden_dim
            self.qkv_hidden_size = qkv_hidden_size or hidden_dim
            self.hidden_size_per_attention_head = self.qkv_hidden_size // num_heads
            self.tp_size = (
                1
                if self.use_data_parallel
                else module.get_tensor_model_parallel_world_size()
            )
            self.num_attention_heads_per_partition = module.divide(
                num_heads, self.tp_size
            )
            self.norm0 = _make_norm(module, norm_type, hidden_dim)
            self.norm1 = _make_norm(module, norm_type, hidden_dim)
            if mlp_type != "mlp2":
                raise NotImplementedError(f"Unsupported Kimi vision MLP type: {mlp_type}")
            self.mlp = module.MLP2(
                [hidden_dim, mlp_dim, hidden_dim],
                activation,
                bias=linear_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                use_data_parallel=self.use_data_parallel,
            )
            self.wqkv = module.QKVParallelLinear(
                hidden_size=hidden_dim,
                head_size=self.hidden_size_per_attention_head,
                total_num_heads=num_heads,
                total_num_kv_heads=num_heads,
                bias=attn_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.wqkv",
                disable_tp=self.use_data_parallel,
            )
            self.wo = module.RowParallelLinear(
                self.qkv_hidden_size,
                hidden_dim,
                bias=attn_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.wo",
                disable_tp=self.use_data_parallel,
            )
            self.attn = module.MMEncoderAttention(
                num_heads=self.num_attention_heads_per_partition,
                head_size=self.hidden_size_per_attention_head,
                scale=self.hidden_size_per_attention_head**-0.5,
                prefix=f"{prefix}.attn",
            )

    class HcuMoonViT3dEncoder(base_encoder):
        def __init__(
            self,
            hidden_dim: int,
            num_layers: int,
            block_cfg: dict,
            video_attn_type: str = "spatial_temporal",
            quant_config=None,
            prefix: str = "",
        ) -> None:
            nn.Module.__init__(self)
            if video_attn_type != "spatial_temporal":
                raise AssertionError(
                    f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
                )
            self.video_attn_type = video_attn_type
            qkv_hidden_size = block_cfg.get("qkv_hidden_size") or block_cfg["hidden_dim"]
            block_cfg = dict(block_cfg)
            block_cfg["qkv_hidden_size"] = qkv_hidden_size
            self.rope_2d = module.Rope2DPosEmbRepeated(
                qkv_hidden_size // block_cfg["num_heads"], 512, 512
            )
            self.blocks = nn.ModuleList(
                [
                    HcuMoonViTEncoderLayer(
                        **block_cfg,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{layer_idx}",
                    )
                    for layer_idx in range(num_layers)
                ]
            )
            self.final_layernorm = _make_norm(
                module, block_cfg.get("norm_type", "layernorm"), hidden_dim
            )

    class HcuMoonViT3dPretrainedModel(base_tower):
        def __init__(self, config, quant_config=None, prefix: str = ""):
            if getattr(config, "model_type", None) != "kimi_k3_vision":
                super().__init__(config, quant_config=quant_config, prefix=prefix)
                return
            nn.Module.__init__(self)
            config = deepcopy(config)
            self.config = config
            self.merge_kernel_size = config.merge_kernel_size
            self.patch_size = config.patch_size
            self.merge_type = config.merge_type
            self.patch_embed = module.MoonVision3dPatchEmbed(
                out_dim=config.hidden_size,
                patch_size=config.patch_size,
                pos_emb_height=config.init_pos_emb_height,
                pos_emb_width=config.init_pos_emb_width,
                pos_emb_time=config.init_pos_emb_time,
                pos_emb_type=config.pos_emb_type,
            )
            if not config.patch_embed_proj_bias:
                self.patch_embed.proj.register_parameter("bias", None)
            self.patch_embed.pos_emb.interpolation_mode = config.pos_emb_interpolation_mode
            self.encoder = HcuMoonViT3dEncoder(
                hidden_dim=config.hidden_size,
                num_layers=config.num_hidden_layers,
                block_cfg={
                    "num_heads": config.num_attention_heads,
                    "hidden_dim": config.hidden_size,
                    "mlp_dim": config.intermediate_size,
                    "qkv_hidden_size": config.qkv_hidden_size,
                    "norm_type": config.norm_type,
                    "attn_bias": config.attn_bias,
                    "linear_bias": config.linear_bias,
                    "mlp_type": config.mlp_type,
                    "activation": module.get_act_fn(config.activation_func),
                },
                video_attn_type=config.video_attn_type,
                quant_config=quant_config,
                prefix=module.maybe_prefix(prefix, "encoder"),
            )

    class HcuKimiK25MultiModalProjector(base_projector):
        """K25 projector with K3's ``patchmergerv2`` branch."""

        def __init__(
            self,
            config,
            use_data_parallel: bool = False,
            quant_config=None,
            prefix: str = "",
        ) -> None:
            if getattr(config, "mm_projector_type", "patchmerger") != "patchmergerv2":
                super().__init__(
                    config,
                    use_data_parallel=use_data_parallel,
                    quant_config=quant_config,
                    prefix=prefix,
                )
                return

            nn.Module.__init__(self)
            self.use_data_parallel = use_data_parallel
            self.mm_projector_type = "patchmergerv2"
            merge_h, merge_w = config.merge_kernel_size
            self.hidden_size = config.hidden_size * merge_h * merge_w
            output_size = getattr(config, "text_hidden_size", None)
            if output_size is None:
                output_size = config.mm_hidden_size
            self.linear_1 = module.ReplicatedLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_1",
            )
            self.linear_2 = module.ReplicatedLinear(
                self.hidden_size,
                output_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_2",
            )
            self.post_norm = nn.RMSNorm(
                output_size,
                eps=getattr(config, "projector_ln_eps", 1e-5),
            )
            self.act = module.GELUActivation()

        def forward(self, image_features: torch.Tensor) -> torch.Tensor:
            if self.mm_projector_type != "patchmergerv2":
                return super().forward(image_features)
            hidden_states = image_features.view(image_features.shape[0], -1)
            hidden_states, _ = self.linear_1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states, _ = self.linear_2(hidden_states)
            return self.post_norm(hidden_states)

    @torch.inference_mode()
    def hcu_mm_projector_forward(
        mm_projector: nn.Module, vt_output: list[torch.Tensor]
    ):
        if getattr(mm_projector, "mm_projector_type", "patchmerger") != "patchmergerv2":
            return base_projector_forward(mm_projector, vt_output)
        num_embedding_list = [x.shape[0] for x in vt_output]
        batched = torch.cat(vt_output, dim=0)
        projector_dtype = mm_projector.linear_1.weight.dtype
        if batched.dtype != projector_dtype:
            batched = batched.to(projector_dtype)
        proj_out = mm_projector(batched).reshape(-1, mm_projector.linear_2.output_size)
        return torch.split(proj_out, num_embedding_list)

    module.MoonViTEncoderLayer = HcuMoonViTEncoderLayer
    module.MoonViT3dEncoder = HcuMoonViT3dEncoder
    module.MoonViT3dPretrainedModel = HcuMoonViT3dPretrainedModel
    module.KimiK25MultiModalProjector = HcuKimiK25MultiModalProjector
    module.mm_projector_forward = hcu_mm_projector_forward
    module._hcu_kimi_k25_qkv_layout_patch_applied = True


__all__ = ["install_kimi_k25_qkv_layout_compat"]
