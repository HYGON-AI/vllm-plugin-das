# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Independent Hyper-Connection layers for HY V4 on HCU."""

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.layers.linear import ReplicatedLinear


class HYV4HCPreLayer(nn.Module):
    """Reduce parallel residual channels and produce post gates."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_dim: int,
        hc_mult: int = 4,
        magnitude: float = 2.0,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        hc_eps: float = 1e-6,
        layernorm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.hc_mult = hc_mult
        self.magnitude = magnitude
        self.hc_eps = hc_eps
        self.layernorm_epsilon = layernorm_epsilon
        self.hc_fn = ReplicatedLinear(
            input_size=hc_mult * hidden_dim,
            output_size=2 * hc_mult,
            params_dtype=torch.float32,
            bias=False,
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(2 * hc_mult, dtype=torch.float32))
        self.reset_parameters(init_std, base_noise_std)

    def reset_parameters(
        self,
        init_std: float,
        base_noise_std: float = 0.0,
    ) -> None:
        del init_std
        nn.init.constant_(self.hc_scale, 0.01)
        with torch.no_grad():
            self.hc_base[: self.hc_mult].fill_(
                -torch.log(
                    torch.tensor(
                        self.hc_mult - 1.0,
                        dtype=self.hc_base.dtype,
                    )
                )
            )
            self.hc_base[self.hc_mult :].fill_(0.0)
            if base_noise_std > 0.0:
                self.hc_base.add_(
                    torch.randn_like(self.hc_base) * base_noise_std
                )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.size()
        x_flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.layernorm_epsilon
        )
        mixes = self.hc_fn(x_flat)[0] * rsqrt
        pre_raw = mixes[..., : self.hc_mult]
        post_raw = mixes[..., self.hc_mult :]
        pre = (
            torch.sigmoid(
                pre_raw * self.hc_scale[0].float()
                + self.hc_base[: self.hc_mult].float()
            )
            + self.hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(
                post_raw * self.hc_scale[1].float()
                + self.hc_base[self.hc_mult :].float()
            )
            + self.hc_eps
        )
        reduced = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)
        return reduced.to(x.dtype), post


class HYV4HCPostLayer(nn.Module):
    """Scatter a sub-block output back onto the residual channels."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        dtype = x.dtype
        result = post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
        return (result + residual.float()).to(dtype)


class HYV4HCHeadLayer(nn.Module):
    """Merge parallel residual channels before the output normalization."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_head_fn = ReplicatedLinear(
            input_size=hc_mult * hidden_size,
            output_size=hc_mult,
            params_dtype=torch.float32,
            bias=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.reset_parameters(init_std, base_noise_std)

    def reset_parameters(
        self,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
    ) -> None:
        del init_std
        nn.init.constant_(self.hc_head_scale, 0.01)
        with torch.no_grad():
            self.hc_head_base.fill_(
                -torch.log(
                    torch.tensor(
                        self.hc_mult - 1.0,
                        dtype=self.hc_head_base.dtype,
                    )
                )
            )
            if base_noise_std > 0.0:
                self.hc_head_base.add_(
                    torch.randn_like(self.hc_head_base) * base_noise_std
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        dtype = x.dtype
        x_flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        mixes = self.hc_head_fn(x_flat)[0] * rsqrt
        pre = (
            torch.sigmoid(
                mixes * self.hc_head_scale.float()
                + self.hc_head_base.float()
            )
            + self.hc_eps
        )
        result = torch.sum(pre.unsqueeze(-1) * x_flat.reshape(shape), dim=1)
        return result.to(dtype)


class HYV4HCLayer(nn.Module):
    """Own the pre/post iHC boundary around one decoder sub-block."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_ihc = getattr(config, "enable_ihc", False)
        if self.enable_ihc:
            self.hc_pre = HYV4HCPreLayer(
                config,
                config.hidden_size,
                config.hc_mult,
                config.hc_magnitude,
                init_std,
                base_noise_std,
                config.hc_eps,
                config.rms_norm_eps,
            )
            self.hc_post = HYV4HCPostLayer(config)

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enable_ihc:
            return hidden_states
        return self._prepare_input_to_3d(hidden_states)

    def _prepare_input_to_3d(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            return hidden_states
        if hidden_states.dim() != 2:
            raise RuntimeError(
                "HC expects a 2D/3D tensor, "
                f"got shape={tuple(hidden_states.shape)}"
            )

        num_tokens, width = hidden_states.shape
        hidden_size = self.config.hidden_size
        hc_mult = self.config.hc_mult
        if width == hidden_size:
            return hidden_states.unsqueeze(1).repeat(1, hc_mult, 1)

        expected = hc_mult * hidden_size
        if width == expected:
            return hidden_states.reshape(num_tokens, hc_mult, hidden_size)

        raise RuntimeError(
            f"HC expects last dim to be hidden_size ({hidden_size})"
            f"or hc_mult*hidden_size ({expected}), got {width}. "
            f"hc_mult={hc_mult}, hidden_size={hidden_size}."
        )

    def pre(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.enable_ihc:
            return hidden_states, None, hidden_states
        reduced, post_gates = self.hc_pre(hidden_states)
        return reduced, post_gates, hidden_states

    def post(
        self,
        output_with_bias: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enable_ihc:
            return output_with_bias + residual
        assert post_gates is not None
        return self.hc_post(output_with_bias, residual, post_gates)


__all__ = [
    "HYV4HCHeadLayer",
    "HYV4HCLayer",
    "HYV4HCPostLayer",
    "HYV4HCPreLayer",
]
