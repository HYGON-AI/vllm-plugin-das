# SPDX-License-Identifier: Apache-2.0
"""Real standard draft configuration on unmodified Core c440."""

import pickle
from copy import deepcopy

import pytest
from hcu_vllm_das.models.hy4_standard.dfly_config import (
    derive_standard_dfly_config,
    resolve_dfly_width,
)
from vllm.transformers_utils.configs.hy_v4 import HYV4Config


def checkpoint():
    return HYV4Config(
        num_hidden_layers=78,
        dfly_num_hidden_layers=5,
        dfly_target_index_layer=[61, 65, 69, 73, 77],
        dfly_target_kv_layers=[61, 65, 69, 73, 77],
        dfly_mask_token_id=120009,
        dfly_block_size=7,
        num_nextn_predict_layers=0,
        layer_types=["deepseek_sparse_attention"] * 78,
        indexer_types=["full"] * 78,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [12, 10, 10],
        },
    )


def test_stock_core_checkpoint_derive_serialization():
    target = checkpoint()
    before = deepcopy(target.to_dict())
    draft = derive_standard_dfly_config(target)
    assert draft.model_type == "hy_v4_dfly"
    assert draft.to_dict()["model_type"] == "hy_v4_dfly"
    restored = type(draft).from_dict(pickle.loads(pickle.dumps(draft)).to_dict())
    assert restored.model_type == "hy_v4_dfly"
    assert restored.target_num_hidden_layers == 78 and restored.num_hidden_layers == 5
    assert restored.target_kv_layers == [61, 65, 69, 73, 77]
    assert restored.num_nextn_predict_layers == 0 and not restored.enable_ihc
    assert restored.rope_parameters["mrope_section"] == [12, 10, 10]
    assert target.to_dict() == before


@pytest.mark.parametrize("requested,expected", [(None, 6), (5, 5), (6, 6)])
def test_native_block_width_is_independent_of_draft_depth(requested, expected):
    assert resolve_dfly_width(checkpoint(), requested) == expected


@pytest.mark.parametrize("requested", [True, 0, 17, -1])
def test_invalid_native_width(requested):
    with pytest.raises(ValueError):
        resolve_dfly_width(checkpoint(), requested)
