# SPDX-License-Identifier: Apache-2.0
"""Complete serialized INT4 extents must precede padding-tolerant owner loads."""
import pytest
import torch

from tests.accuracy.test_hyv4_w4a8_kernels import _scale_experts
from tests.models.hy_v4.test_weight_loading import checkpoint_model


def _parameter(owner, shard):
    name = "w2_weight" if shard == "w2" else "w13_weight"
    return name, getattr(owner, name)


def _reject(owner, shard, value, match="HYV4.*packed", parameter=None):
    name, param = _parameter(owner, shard)
    if parameter is not None:
        name, param = parameter, getattr(owner, parameter)
    param.data.fill_(85)
    before, version = param.clone(), param._version
    try:
        with pytest.raises(ValueError, match=match):
            param.weight_loader(param, value, name, shard, 0, return_success=True)
    finally:
        # RoutedExperts uses param.data, so the sentinel is authoritative even
        # when a copy does not advance the Parameter's own version counter.
        torch.testing.assert_close(param, before)
        assert param._version == version


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("shard", ["w1", "w3", "w2"])
@pytest.mark.parametrize("bad", ["row_short", "row_long", "col_short", "col_long",
                                 "empty_rows", "empty_cols", "rank1", "rank3",
                                 "int8", "float", "swapped"])
def test_packed_extent_rejects_before_owner_mutation(tmp_path, native, tp, rank, shard, bad):
    _, owner = _scale_experts(tmp_path, native, tp, rank)
    rows, cols = (4, tp) if shard == "w2" else (2 * tp, 2)
    # Avoid an indistinguishable transpose for the square TP1 gate/up case.
    if bad == "swapped" and rows == cols:
        owner.moe_config.hidden_dim_unpadded = 2
        cols = 1
    shape = {"row_short": (rows - 1, cols), "row_long": (rows + 1, cols),
             "col_short": (rows, cols - 1), "col_long": (rows, cols + 1),
             "empty_rows": (0, cols), "empty_cols": (rows, 0),
             "rank1": (rows * cols,), "rank3": (1, rows, cols),
             "swapped": (cols, rows)}.get(bad, (rows, cols))
    dtype = {"int8": torch.int8, "float": torch.float32}.get(bad, torch.uint8)
    _reject(owner, shard, torch.full(shape, 0x12, dtype=dtype))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("shard", ["w1", "w3", "w2"])
def test_packed_extent_preserves_tp_bytes_and_declared_padding(tmp_path, native, tp, rank, padded, shard):
    _, owner = _scale_experts(tmp_path, native, tp, rank, padded)
    name, param = _parameter(owner, shard)
    param.data.fill_(85)
    expected = torch.full_like(param, 85)
    if shard == "w2":
        source = torch.tensor([[0x12], [0x34], [0x56], [0x78]], dtype=torch.uint8)
        if tp == 2:
            source = torch.cat((source, torch.tensor([[0x21], [0x43], [0x65], [0x07]], dtype=torch.uint8)), 1)
        want = [0x21, 0x43, 0x65, -121] if rank == 0 else [0x12, 0x34, 0x56, 0x70]
        expected[0, :4, 0] = torch.tensor(want, dtype=torch.int8)
    else:
        source = torch.tensor([[0x12, 0x34], [0x56, 0x78]], dtype=torch.uint8)
        if tp == 2:
            source = torch.cat((source, torch.tensor([[0x21, 0x43], [0x65, 0x07]], dtype=torch.uint8)), 0)
        want = [[0x21, 0x43], [0x65, -121]] if rank == 0 else [[0x12, 0x34], [0x56, 0x70]]
        start = (4 if padded else 2) if shard == "w3" else 0
        expected[0, start:start + 2, :2] = torch.tensor(want, dtype=torch.int8)
    assert param.weight_loader(param, source, name, shard_id=shard, expert_id=0, return_success=True)
    torch.testing.assert_close(param, expected)
    assert param.weight_loader.__self__ is owner


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("field", ["hidden_dim_unpadded", "intermediate_size_per_partition_unpadded"])
@pytest.mark.parametrize("invalid", [None, 0, -1, 9, True, 2.0, "2"])
@pytest.mark.parametrize("shard", ["w1", "w3", "w2"])
def test_packed_extent_invalid_metadata_never_falls_back(tmp_path, native, field, invalid, shard):
    _, owner = _scale_experts(tmp_path, native)
    setattr(owner.moe_config, field, invalid)
    _reject(owner, shard, torch.full((4, 1) if shard == "w2" else (2, 2), 0x12, dtype=torch.uint8))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("field,shard,shape", [
    ("hidden_dim_unpadded", "w1", (2, 1)),
    ("hidden_dim_unpadded", "w3", (2, 1)),
    ("intermediate_size_per_partition_unpadded", "w2", (4, 0)),
])
def test_packed_extent_odd_logical_input_rejected(tmp_path, native, tp, rank, field, shard, shape):
    _, owner = _scale_experts(tmp_path, native, tp, rank)
    setattr(owner.moe_config, field, 3 if field == "hidden_dim_unpadded" else 1)
    if shard == "w2":
        shape = (4, tp // 2)
    else:
        shape = (2 * tp, 1)
    _reject(owner, shard, torch.full(shape, 0x12, dtype=torch.uint8))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("parameter,shard", [("w13_weight", "w2"), ("w2_weight", "w1"),
    ("w2_weight", "w3"), ("w13_weight", "bad"), ("w13_weight", None)])
def test_packed_extent_invalid_shard_rejected(tmp_path, native, parameter, shard):
    _, owner = _scale_experts(tmp_path, native)
    _reject(owner, shard, torch.full((2, 2), 0x12, dtype=torch.uint8), parameter=parameter)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("shard", ["w1", "w3", "w2"])
def test_packed_extent_absent_metadata_uses_create_allocation(tmp_path, native, shard):
    _, owner = _scale_experts(tmp_path, native)
    del owner.moe_config.hidden_dim_unpadded
    del owner.moe_config.intermediate_size_per_partition_unpadded
    name, param = _parameter(owner, shard)
    shape = (4, 1) if shard == "w2" else (2, 2)
    assert param.weight_loader(param, torch.full(shape, 0x12, dtype=torch.uint8), name, shard, 0, True)


def _target_checkpoint(tmp_path, model, native, fused=False):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    config, owner = _scale_experts(tmp_path, native)
    retained = [(name + (".weight" if native else ""), torch.ones_like(value))
                for name, value in model.named_parameters()]
    model.quant_config = config
    model.model.mlp = torch.nn.Module()
    model.model.mlp.experts = torch.nn.Module()
    model.model.mlp.experts.routed_experts = owner
    model.model.config.num_experts = 3
    model.model.num_redundant_experts = 1
    model.model.get_expert_mapping = lambda: RoutedExperts.build_expert_params_mapping("gate_proj", "down_proj", "up_proj", 3)
    pieces = []
    if fused:
        for projection, shape in [("gate_up_proj", (3, 4, 2)), ("down_proj", (3, 4, 1))]:
            prefix = "model.mlp.experts." + projection
            # Native's canonical alias stream supports rank-3 fused tensors;
            # its serialized .weight.packed spelling only supports rank 2.
            pieces += [(prefix if native else prefix + ".int4_packed", torch.full(shape, 0x12, dtype=torch.uint8)),
                       (prefix + "_scale", torch.ones(3, 4, 1))]
    else:
        for expert in range(3):
            for projection, shape in [("gate_proj", (2, 2)), ("up_proj", (2, 2)), ("down_proj", (4, 1))]:
                prefix = f"model.mlp.experts.{expert}.{projection}.weight"
                pieces += [(prefix + (".packed" if native else ".int4_packed"), torch.full(shape, 0x12, dtype=torch.uint8)),
                           (prefix + "_scale", torch.ones(shape[0], 1))]
    return owner, retained, pieces


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("projection,shape", [("gate_proj", (1, 2)), ("up_proj", (1, 2)), ("down_proj", (3, 1))])
def test_real_target_strict_ledger_rejects_short_packed_shard(tmp_path, checkpoint_model, native, projection, shape):
    owner, retained, pieces = _target_checkpoint(tmp_path, checkpoint_model, native)
    bad_name = f"model.mlp.experts.0.{projection}.weight" + (".packed" if native else ".int4_packed")
    bad = (bad_name, torch.full(shape, 0x12, dtype=torch.uint8))
    rest = [(name, value) for name, value in pieces if name != bad_name]
    name, param = _parameter(owner, "w2" if projection == "down_proj" else "w1")
    param.data.fill_(85)
    before = param.clone()
    try:
        with pytest.raises(ValueError, match="HYV4.*packed"):
            loaded = checkpoint_model.load_weights(iter([bad] + retained + rest))
            print(f"ACCEPTED {projection}: shape={shape}, strict_complete="
                  f"{loaded == set(dict(checkpoint_model.named_parameters()))}, "
                  f"destination={param.tolist()}")
    finally:
        torch.testing.assert_close(param, before)
    assert not hasattr(checkpoint_model.model, "_checkpoint_accounting")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_real_target_complete_packed_split_fused_alias_ledger(tmp_path, checkpoint_model, native, fused):
    owner, retained, pieces = _target_checkpoint(tmp_path, checkpoint_model, native, fused)
    weights = retained + pieces
    assert checkpoint_model.load_weights(iter(weights)) == set(dict(checkpoint_model.named_parameters()))
    torch.testing.assert_close(owner.w13_weight, torch.full_like(owner.w13_weight, 0x21))
    torch.testing.assert_close(owner.w2_weight, torch.full_like(owner.w2_weight, 0x21))
    with pytest.raises(RuntimeError, match="Duplicate"):
        checkpoint_model.load_weights(iter(weights + [pieces[0]]))
    with pytest.raises(RuntimeError, match="Missing"):
        checkpoint_model.load_weights(iter(retained + pieces[1:]))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("projection,shape", [("gate_up_proj", (3, 2, 2)),
                                              ("down_proj", (3, 3, 1))])
def test_real_target_fused_short_piece_rejected_before_owner(tmp_path, checkpoint_model, native, projection, shape):
    owner, retained, pieces = _target_checkpoint(tmp_path, checkpoint_model, native, fused=True)
    bad_name = "model.mlp.experts." + projection + ("" if native else ".int4_packed")
    weights = [(bad_name, torch.full(shape, 0x12, dtype=torch.uint8))]
    weights += retained + [(name, value) for name, value in pieces if name != bad_name]
    _, param = _parameter(owner, "w2" if projection == "down_proj" else "w1")
    param.data.fill_(85)
    before = param.clone()
    try:
        with pytest.raises(ValueError, match=r"HYV4 packed expert w[12] requires shape .*got"):
            checkpoint_model.load_weights(iter(weights))
    finally:
        torch.testing.assert_close(param, before)
