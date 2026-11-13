# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from . import rotary_embedding
from . import silu_and_mul
from . import gemma_rms_norm
from . import rms_norm
from . import rms_norm_gated

# Fused Qwen GDN-path ops (port of the 0.25 Qwen3.5 stack): registers
# torch.ops.vllm.sigmoid_mul / hcu_rms_norm_gated_strided_z /
# hcu_rearrange_mixed_qkv.
from . import sigmoid_mul  # noqa: F401,E402
from . import hcu_rms_norm_gated_strided_z_op  # noqa: F401,E402
from . import hcu_rearrange_mixed_qkv  # noqa: F401,E402
