# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Admit symmetric channel INT8 WK into Core's BF16 indexer fusion."""
import torch
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk as _native


def load_indexer_wk(name, tensor, buf, params_dict, loaded_params,
                    pp_missing_layer_names):
    if '.indexer.wk.' in name and name.endswith('.weight_zero_point'):
        raise ValueError('HY4 fused WK loader requires symmetric checkpoint weights')
    if ('.indexer.wk.' in name and name.endswith('.weight')
            and tensor.dtype == torch.int8):
        if any(name.startswith(prefix) for prefix in pp_missing_layer_names):
            return True
        prefix = name.rsplit('.wk.', 1)[0]
        entry = buf.setdefault(prefix, {})
        entry['weight'] = tensor
        if 'scale' not in entry:
            return True
        # Core's scaled_dequantize already supports integer input. Replay the
        # scale arrival so its existing pairing/dequant/fused shard loader runs.
        return _native(prefix + '.wk.weight_scale', entry['scale'], buf,
                       params_dict, loaded_params, pp_missing_layer_names)
    return _native(name, tensor, buf, params_dict, loaded_params,
                   pp_missing_layer_names)
