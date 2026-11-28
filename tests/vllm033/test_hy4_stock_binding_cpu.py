# SPDX-License-Identifier: Apache-2.0
"""Installed stock-source ownership contracts; native classes are not patched."""

from hcu_vllm_das.models.hy4_standard import attention, model
from vllm.models.hy_v4.amd import model as native_model
from vllm.models.hy_v4.nvidia import attention as native_attention


def test_native_constructor_code_and_globals_are_retained():
    attention.check_stock_attention_anchors()
    model.check_stock_model_anchors()
    assert (
        attention._BOUND_INIT.__code__
        is native_attention.HYV4MLAAttention.__init__.__code__
    )
    assert (
        attention._BOUND_INIT.__globals__
        is not native_attention.HYV4MLAAttention.__init__.__globals__
    )
    assert (
        native_attention.HYV4MLAAttention.__init__.__globals__["Indexer"]
        is native_attention.Indexer
    )
    assert attention._BOUND_INIT.__globals__["Indexer"] is attention.HYV4SnapshotIndexer
    assert model.HYV4StandardModel.forward is native_model.HYV4Model.forward
    assert (
        model._BOUND_DECODER_INIT.__code__
        is native_model.HYV4DecoderLayer.__init__.__code__
    )
    assert (
        native_model.HYV4DecoderLayer.__init__.__globals__["HYV4MLAAttention"]
        is native_attention.HYV4MLAAttention
    )


def test_external_draft_skip_policy_preserves_real_target_indexer_pattern():
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config

    target = HYV4Config(
        num_hidden_layers=5, indexer_types=["full", "shared", "full", "shared", "full"]
    )
    assert attention.standard_skip_topk(target) == {1, 3}
    draft = HYV4Config(
        num_hidden_layers=5, dfly_external_topk=True, indexer_types=["full"] * 5
    )
    assert attention.standard_skip_topk(draft) == set(range(5))
    assert target.indexer_types == ["full", "shared", "full", "shared", "full"]
