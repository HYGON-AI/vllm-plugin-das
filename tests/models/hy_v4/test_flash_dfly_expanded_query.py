# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact production HIP kernels for a 512-target / 768-draft query block.

These are small-tensor kernel regressions, not model/attention/profile-memory
qualification. References use CPU Torch independently of the kernel outputs.
Target positions/hidden states and target snapshots have exactly 512 rows;
all writable draft buffers have prefix/tail canaries around their logical
capacity. Canaries detect writes, not every possible out-of-bounds read: use
an accelerator memory checker as well when qualifying memory safety.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        not torch.version.hip or not torch.cuda.is_available(),
        reason="expanded-query regressions require a live HIP device",
    ),
]

TARGET_TOKENS = 512
REQUESTS = 128
SPECULATIVE_TOKENS = 5
QUERY_WIDTH = SPECULATIVE_TOKENS + 1
DRAFT_TOKENS = REQUESTS * QUERY_WIDTH
HIDDEN_SIZE = 3072
MAX_MODEL_LEN = 8192
BLOCK_SIZE = 64
TARGET_TOPK = 2048
MASK_TOKEN_ID = 99
GUARD = 8
SENTINEL = -777


def _inputs(case, mrope):
    """CPU inputs: four target tokens per request, never 768 target rows."""
    requests = torch.arange(REQUESTS)
    positions = torch.arange(4).repeat(REQUESTS)
    if case == "mixed_boundary":
        # req % 4 == 0 ends exactly at max_model_len; == 1 exceeds it
        # by one; == 2 is mid-prefill; == 3 remains a short active request.
        anchors = torch.where(
            requests % 4 == 0,
            MAX_MODEL_LEN - QUERY_WIDTH,
            torch.where(requests % 4 == 1, MAX_MODEL_LEN - QUERY_WIDTH + 1, 4),
        )
        positions = (anchors[:, None] - 4 + torch.arange(4)).flatten()
    profile = case == "padding_profile"
    count = 0 if profile else REQUESTS
    hidden = (
        torch.arange(TARGET_TOKENS * HIDDEN_SIZE).reshape(TARGET_TOKENS, HIDDEN_SIZE)
        % 127
    ).to(torch.bfloat16)
    if profile:
        # A masked target read must not leak these poison values into outputs.
        positions.fill_(MAX_MODEL_LEN + 123)
        hidden.fill_(float("nan"))
    return dict(
        target_query_start_loc=torch.arange(count + 1, dtype=torch.int32) * 4,
        target_positions=positions.to(torch.int64),
        idx_mapping=((requests[:count] * 37 + 11) % REQUESTS).to(torch.int32),
        num_sampled=(
            (requests[:count] % 4 != 2).to(torch.int32)
            if case == "mixed_boundary"
            else torch.ones(count, dtype=torch.int32)
        ),
        num_rejected=(
            (requests[:count] % 3).to(torch.int32)
            if case == "short_prefill"
            else torch.zeros(count, dtype=torch.int32)
        ),
        last_sampled=torch.arange(REQUESTS * 2).reshape(REQUESTS, 2) + 1000,
        target_hidden=hidden,
        mrope_delta=(requests.to(torch.int32) * 7 - 128) if mrope else None,
        temperature=requests.to(torch.float32) / 128 + 0.5,
        seeds=requests.to(torch.int64) * 101 + 17,
        max_model_len=torch.tensor(MAX_MODEL_LEN, dtype=torch.int32),
        mask_token_id=MASK_TOKEN_ID,
        query_width=QUERY_WIDTH,
        num_reqs=count,
        padded_batch=REQUESTS,
    )


def _reference_prepare(inputs):
    """Independent CPU gather/scatter reference, including inactive real rows."""
    count = inputs["num_reqs"]
    rows = torch.zeros(REQUESTS, dtype=torch.int64)
    anchors = torch.zeros(REQUESTS, dtype=torch.int64)
    active = torch.zeros(REQUESTS, dtype=torch.bool)
    hidden = torch.zeros(REQUESTS, HIDDEN_SIZE, dtype=torch.bfloat16)
    mapping = torch.full((REQUESTS,), -1, dtype=torch.int64)
    temperatures = torch.full((REQUESTS,), SENTINEL, dtype=torch.float32)
    seeds = torch.full((REQUESTS,), SENTINEL, dtype=torch.int64)
    bonus = torch.zeros(REQUESTS, dtype=torch.int32)
    delta = torch.zeros(REQUESTS, dtype=torch.int64)
    if count:
        rows[:count] = (
            inputs["target_query_start_loc"][1:].long()
            - inputs["num_rejected"].long()
            - 1
        )
        anchors[:count] = inputs["target_positions"][rows[:count]] + 1
        active[:count] = (inputs["num_sampled"] > 0) & (
            anchors[:count] + QUERY_WIDTH <= MAX_MODEL_LEN
        )
        states = inputs["idx_mapping"].long()
        hidden[:count] = inputs["target_hidden"][rows[:count]]
        mapping[:count] = torch.where(active[:count], states, -1)
        temperatures[states] = inputs["temperature"][states]
        seeds[states] = inputs["seeds"][states]
        bonus[:count] = torch.where(
            active[:count], inputs["last_sampled"][states, 0], 0
        ).to(torch.int32)
        if inputs["mrope_delta"] is not None:
            delta[:count] = inputs["mrope_delta"][states].long()
    offsets = torch.arange(QUERY_WIDTH)
    positions = torch.where(active[:, None], anchors[:, None] + offsets, offsets)
    ids = torch.full((REQUESTS, QUERY_WIDTH), MASK_TOKEN_ID, dtype=torch.int32)
    ids[:, 0] = bonus
    rope = torch.where(active[:, None], positions + delta[:, None], 0)
    return dict(
        input_ids=ids.flatten(),
        positions=positions.flatten(),
        is_padding=(~active[:, None]).expand(-1, QUERY_WIDTH).flatten(),
        query_start_loc=torch.arange(REQUESTS + 1, dtype=torch.int32) * QUERY_WIDTH,
        seq_lens=(torch.where(active, anchors, 0) + QUERY_WIDTH).to(torch.int32),
        active=active,
        source_rows=rows,
        anchors=anchors,
        target_hidden_out=hidden,
        mrope_positions=(
            rope.flatten().repeat(3, 1) if inputs["mrope_delta"] is not None else None
        ),
        draft_idx_mapping=mapping,
        out_temperature=temperatures,
        out_seeds=seeds,
    )


def _guarded(shape, dtype, device, guards, *, axis=0, value=SENTINEL):
    """Return a logical-capacity view with independently checked red zones."""
    padded = list(shape)
    padded[axis] += 2 * GUARD
    fill = value if dtype != torch.bool else bool(value)
    storage = torch.full(padded, fill, dtype=dtype, device=device)
    body = storage.narrow(axis, GUARD, shape[axis])
    guards.append((storage, axis, shape[axis], fill))
    return body


def _check_guards(guards):
    for storage, axis, size, value in guards:
        for start in (0, GUARD + size):
            region = storage.narrow(axis, start, GUARD)
            torch.testing.assert_close(
                region, torch.full_like(region, value), rtol=0, atol=0
            )


def _reference_slots(reference, tables):
    positions = reference["positions"].reshape(REQUESTS, QUERY_WIDTH)
    blocks = tables.gather(1, positions // BLOCK_SIZE).long()
    valid = reference["active"][:, None] & (blocks != 0)
    return torch.where(
        valid, blocks * BLOCK_SIZE + positions % BLOCK_SIZE, -1
    ).flatten()


def _reference_reuse(reference, snapshots, layer_slots, window, width):
    """CPU reference: history, local window, causal query block, then -1 pad."""
    anchors = reference["anchors"]
    active = reference["active"]
    start = anchors - window
    offsets = torch.arange(QUERY_WIDTH)
    result = torch.full(
        (len(layer_slots), REQUESTS, QUERY_WIDTH, width), -1, dtype=torch.int32
    )
    for layer, slot in enumerate(layer_slots.tolist()):
        history = snapshots[slot, reference["source_rows"]]
        history = torch.where((history >= 0) & (history < start[:, None]), history, -1)
        result[layer, :, :, :TARGET_TOPK] = history[:, None, :]
        result[layer, :, :, TARGET_TOPK : TARGET_TOPK + window] = (
            start[:, None] + torch.arange(window)
        ).clamp(min=-1)[:, None, :]
        causal = offsets[None, :] <= offsets[:, None]
        result[
            layer, :, :, TARGET_TOPK + window : TARGET_TOPK + window + QUERY_WIDTH
        ] = torch.where(causal[None], anchors[:, None, None] + offsets, -1)
    result.masked_fill_(~active[None, :, None, None], -1)
    return result.reshape(len(layer_slots), DRAFT_TOKENS, width)


@pytest.mark.parametrize("mrope", [False, True], ids=["rope", "mrope"])
@pytest.mark.parametrize("window", [0, 128], ids=["no_window", "window128"])
@pytest.mark.parametrize("case", ["short_prefill", "padding_profile", "mixed_boundary"])
def test_expanded_512_target_768_queries(case, window, mrope):
    # Import the actual production module; no copied kernels, AST, or mocks.
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_inputs import (
        build_dfly_reuse_indices,
        prepare_dfly_inputs,
        write_dfly_query_slots,
    )

    assert PAD_SLOT_ID == -1
    assert DRAFT_TOKENS == 768 > TARGET_TOKENS
    device = torch.device("cuda", torch.cuda.current_device())
    inputs_cpu = _inputs(case, mrope)
    reference = _reference_prepare(inputs_cpu)
    inputs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs_cpu.items()
    }
    assert inputs["target_positions"].shape == (TARGET_TOKENS,)
    assert inputs["target_hidden"].shape == (TARGET_TOKENS, HIDDEN_SIZE)
    assert inputs["target_positions"].is_contiguous()
    assert inputs["target_hidden"].is_contiguous()
    if case == "padding_profile":
        assert inputs["num_reqs"] == 0
        assert inputs["target_query_start_loc"].numel() == 1
        assert inputs["idx_mapping"].numel() == 0
    guards = []
    outputs = {
        name: (
            None
            if expected is None
            else _guarded(
                expected.shape,
                expected.dtype,
                device,
                guards,
                axis=1 if name == "mrope_positions" else 0,
                value=(
                    case != "padding_profile"
                    if name == "is_padding"
                    else case == "padding_profile" if name == "active" else SENTINEL
                ),
            )
        )
        for name, expected in reference.items()
    }
    prepare_dfly_inputs(**outputs, **inputs)
    torch.cuda.synchronize(device)
    for name, expected in reference.items():
        if expected is not None:
            torch.testing.assert_close(outputs[name].cpu(), expected, rtol=0, atol=0)
    _check_guards(guards)
    if case == "short_prefill":
        assert outputs["active"].sum().item() == REQUESTS
        # Explicitly assert the expansion beyond the 512-row target boundary.
        assert not outputs["is_padding"][TARGET_TOKENS:].any().item()
        assert outputs["source_rows"][-1].item() == 510
    elif case == "mixed_boundary":
        assert outputs["active"][:4].tolist() == [True, False, False, True]
        assert outputs["positions"][:QUERY_WIDTH].tolist() == list(
            range(MAX_MODEL_LEN - QUERY_WIDTH, MAX_MODEL_LEN)
        )
    else:
        assert outputs["is_padding"].all().item()
        assert not outputs["active"].any().item()

    tables = torch.arange(
        1, REQUESTS * (MAX_MODEL_LEN // BLOCK_SIZE) + 1, dtype=torch.int32
    ).reshape(REQUESTS, -1)
    # A live null block in short-prefill and at the near-length boundary.
    tables[::8, 0] = 0
    tables[::8, -1] = 0
    slots = _guarded((DRAFT_TOKENS,), torch.int64, device, guards)
    write_dfly_query_slots(
        slots,
        outputs["positions"],
        outputs["active"],
        tables.to(device),
        BLOCK_SIZE,
        inputs["max_model_len"],
        QUERY_WIDTH,
        REQUESTS,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        slots.cpu(), _reference_slots(reference, tables), rtol=0, atol=0
    )
    _check_guards(guards)

    # Two target snapshot slots, three draft layers, including a repeated alias.
    snapshots_cpu = (
        torch.arange(2)[:, None, None] * 13
        + torch.arange(TARGET_TOKENS)[None, :, None] * 17
        + torch.arange(TARGET_TOPK)[None, None, :] * 5
    ) % (MAX_MODEL_LEN + 17) - 4
    snapshots_cpu = snapshots_cpu.to(torch.int32)
    snapshots_cpu[:, :, :8] = torch.tensor([-1, 0, 1, 2, 3, 4, 8185, 8192])
    snapshots = snapshots_cpu.to(device)
    assert snapshots.shape == (2, TARGET_TOKENS, TARGET_TOPK)
    assert snapshots.is_contiguous()
    layer_slots = torch.tensor([1, 0, 1], dtype=torch.int32)
    width = ((TARGET_TOPK + window + QUERY_WIDTH + 127) // 128) * 128
    reuse = _guarded((3, DRAFT_TOKENS, width), torch.int32, device, guards, axis=1)
    # Inactive rows must not dereference snapshots even with invalid row IDs.
    # Expected reuse indices are derived from CPU inputs, not these mutations.
    outputs["source_rows"].masked_fill_(~outputs["active"], TARGET_TOKENS + 4096)
    build_dfly_reuse_indices(
        reuse,
        snapshots,
        layer_slots.to(device),
        outputs["active"],
        outputs["source_rows"],
        outputs["anchors"],
        TARGET_TOPK,
        window,
        QUERY_WIDTH,
        REQUESTS,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        reuse.cpu(),
        _reference_reuse(reference, snapshots_cpu, layer_slots, window, width),
        rtol=0,
        atol=0,
    )
    _check_guards(guards)
