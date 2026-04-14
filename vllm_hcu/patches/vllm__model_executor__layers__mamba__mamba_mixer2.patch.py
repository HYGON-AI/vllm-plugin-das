# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.mamba.mamba_mixer2 mamba_v2_sharded_weight_loader and MambaMixer2
"""

PATCHES = [
(
"""
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
""",
"""
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
        boundary, loaded_boundary = 0, 0
""",
"""        
        boundary, loaded_boundary = 0, 0
        if henvs.VLLM_USE_NN:
            loaded_total_dim = sum(full_dim - extra
                               for full_dim, extra, _ in shard_spec)
            param_out_axis = 0 if param.dim() == 1 else (param.dim() - 1)
            loaded_out_axis = 0
            if (loaded_weight.dim() > 1 and loaded_weight.shape[-1] == loaded_total_dim
                    and loaded_weight.shape[0] != loaded_total_dim):
                loaded_out_axis = loaded_weight.dim() - 1
""",
    ),

(
"""
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (
                    loaded_start_idx + take
                )  # type: ignore[misc]
            ]  # type: ignore[misc]
""",
"""        
            if henvs.VLLM_USE_NN:
                if take > 0:
                    param_slice = param.data.narrow(param_out_axis, boundary, take)
                    loaded_slice = loaded_weight.narrow(loaded_out_axis,
                                                        loaded_start_idx, take)

                    if (param_slice.dim() == loaded_slice.dim() + 1
                            and param_slice.shape[1] == 1):
                        loaded_slice = loaded_slice.unsqueeze(1)
                    elif (loaded_slice.dim() == param_slice.dim() + 1
                        and loaded_slice.shape[1] == 1):
                        loaded_slice = loaded_slice.squeeze(1)

                    if param_slice.shape != loaded_slice.shape:
                        loaded_slice = loaded_slice.permute(*reversed(range(loaded_slice.dim())))

                    if param_slice.shape != loaded_slice.shape:
                        raise RuntimeError(
                            "mamba_v2_sharded_weight_loader shape mismatch: "
                            f"param_slice={tuple(param_slice.shape)} "
                            f"loaded_slice={tuple(loaded_slice.shape)} "
                            f"(param_out_axis={param_out_axis}, "
                            f"loaded_out_axis={loaded_out_axis})")

                    param_slice.copy_(loaded_slice)
            else:
                param.data[
                    boundary : (boundary + take), ...  # type: ignore[misc]
                ] = loaded_weight[
                    loaded_start_idx : (
                        loaded_start_idx + take
                    )  # type: ignore[misc]
                ]  # type: ignore[misc]
""",
    ),

(
"""
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
""",
"""        
        if henvs.VLLM_USE_NN:
            conv_weights = self.conv1d.weight.squeeze(1).transpose(
                0, 1).contiguous()
        else:
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )
""",
    ),
]