# SPDX-License-Identifier: Apache-2.0
import pytest
from hcu_vllm_das.quantization.slimquant_policy import weight_precision, fused_precision


def test_checkpoint_weight_precision_and_exclusion():
    w = {"m.self_attn.q_proj.weight":"x", "m.self_attn.q_proj.weight_scale":"x",
         "m.self_attn.g_proj.weight":"x", "m.block_sparse_moe.experts.2.w1.weight":"x",
         "m.block_sparse_moe.experts.2.w1.weight_scale":"x"}
    ignore = ["self_attn.g_proj"]
    assert weight_precision("m.self_attn.q_proj",w,ignore) == "w8a8"
    assert weight_precision("m.self_attn.g_proj",w,ignore) == "unquantized"
    assert weight_precision("m.block_sparse_moe.experts.2.w1",w,ignore) == "w4a8"
    with pytest.raises(ValueError, match="mixed precision"):
        fused_precision(["m.self_attn.q_proj", "m.self_attn.g_proj"],w,ignore)
    with pytest.raises(ValueError, match="explicitly excluded"):
        weight_precision("m.self_attn.q_proj",w,["self_attn.q_proj"])
    with pytest.raises(KeyError):
        weight_precision("m.self_attn.unknown",w,ignore)
