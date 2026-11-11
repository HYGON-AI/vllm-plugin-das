# SPDX-License-Identifier: Apache-2.0
"""Native MRV2 image merge contracts. Run with the pinned vLLM CPU stack.

The entire RoPE object is a CPU stand-in emitting canonical-minus-three values.
Image ranges, cache lookup, multimodal merge, native state methods and PCP
partition/gather execute production code. Existing VL tests cover rectangular
geometry separately; native GPU RoPE/delta/storage validation remains Task5.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.worker.gpu.model_states import default

from tests.runtime_patch import test_glm52_pcp_runner as runner_tests
from tests.runtime_patch.test_glm52_pcp_manager import _make_batch, _make_managers
from vllm_hcu.models.hy_v4.vl_mrv2 import HYV4VLModelState

pcp_runner_module = runner_tests.pcp_runner_module


@pytest.mark.parametrize("pcp,valid", [(16, False), (16, True), (1, False)])
def test_resolved_pcp_quantization_is_checked_before_vision_allocation(
    monkeypatch, pcp, valid
):
    from vllm_hcu.models.hy_v4 import vl
    from vllm_hcu.models.hy_v4.model import HYV4ForCausalLM

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["HYV4VLForConditionalGeneration"],
            multimodal_config=None,
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(projector_type="rmsnorm_mlp_adapter")
            ),
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        lora_config=None,
        speculative_config=None,
    )
    method = SimpleNamespace(
        _vllm_hcu_compressed_tensors_w4a8_int8_moe=valid,
        _hcu_static_eplb=True,
        supports_eplb=True,
    )

    def backbone(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.model = SimpleNamespace(layers=[])
        self.moe_layers = [
            SimpleNamespace(routed_experts=SimpleNamespace(quant_method=method))
        ]

    towers = []

    def tower(*args, **kwargs):
        assert pcp == 1 or valid, (
            "invalid resolved quantization reached vision allocation"
        )
        towers.append(True)
        return torch.nn.Identity()

    monkeypatch.setattr(HYV4ForCausalLM, "__init__", backbone)
    monkeypatch.setattr(
        vl.HYV4VLForConditionalGeneration,
        "_mark_tower_model",
        lambda *args: nullcontext(),
    )
    monkeypatch.setattr(vl, "HYV4VisionTransformer", tower)
    if pcp > 1 and not valid:
        with pytest.raises(ValueError, match="W4A8"):
            vl.HYV4VLForConditionalGeneration(vllm_config=config)
        assert not towers
    else:
        vl.HYV4VLForConditionalGeneration(vllm_config=config)
        assert towers == [True]


class _Model:
    def embed_multimodal(self, pixel_values):
        self.encoder_calls = getattr(self, "encoder_calls", 0) + 1
        return tuple(pixel_values.unbind(0))

    def embed_input_ids(self, ids, multimodal_embeddings=None, *, is_multimodal=None):
        text = ids.to(torch.float32)[:, None].repeat(1, 2)
        return _merge_multimodal_embeddings(
            text, multimodal_embeddings, is_multimodal=is_multimodal
        )


class _CPURope:
    """RoPE object stand-in isolating native prepare_inputs batch selection."""

    def __init__(self):
        self.positions = torch.full((3, 129), -71, dtype=torch.int64)
        self.seen = []

    def get_positions(self, count):
        return self.positions[:, :count]

    def prepare_positions(self, indices, starts, prefill, computed):
        self.seen.append(indices.clone())
        for row, index in enumerate(indices.tolist()):
            lo, hi = starts[row : row + 2].tolist()
            self.positions[:, lo:hi] = (
                torch.arange(int(computed[index]), int(computed[index]) + hi - lo) - 3
            ).repeat(3, 1)


@pytest.fixture
def make_state(monkeypatch):
    monkeypatch.setattr(default, "get_rope_state", lambda *a, **k: _CPURope())
    monkeypatch.setattr(default, "maybe_create_mm_pruner", lambda *a: None)

    def make(pcp=16, *, with_encoder_cache=True):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=128,
                get_inputs_embeds_size=lambda: 2,
                dtype=torch.float32,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=16, max_num_batched_tokens=128
            ),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        )
        cache = (
            SimpleNamespace(mm_features={}, encoder_outputs={})
            if with_encoder_cache
            else None
        )
        state = HYV4VLModelState(config, _Model(), cache, torch.device("cpu"))
        return state, cache

    return make


def _feature(name, start, length, mask=None):
    return SimpleNamespace(
        identifier=name,
        data=None,
        modality="image",
        mm_position=PlaceholderRange(offset=start, length=length, is_embed=mask),
    )


def _req_states(batch):
    computed = torch.zeros(32, dtype=torch.int32)
    prefill = torch.zeros(32, dtype=torch.int32)
    computed[batch.idx_mapping.long()] = torch.from_numpy(batch.num_computed_tokens_np)
    prefill[batch.idx_mapping.long()] = torch.from_numpy(batch.prefill_len_np)
    return SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=computed),
        prefill_len=SimpleNamespace(gpu=prefill),
    )


@pytest.mark.parametrize("computed,count", [(0, 70), (3, 45), (33, 21), (56, 9)])
def test_mtp_shifted_images_reach_predictor_across_pcp16_chunks(
    make_state, monkeypatch, computed, count
):
    from vllm_hcu.models.hy_v4.mtp import HYV4MTP, HYV4MultiTokenPredictor
    from vllm_hcu.v1.spec_decode import pcp_mtp

    monkeypatch.setattr(
        pcp_mtp.ar, "set_forward_context", lambda *a, **k: nullcontext()
    )
    batch = _make_batch(
        [
            ("text", [91], 20, False),
            (
                "images",
                list(range(computed + 1, computed + count + 1)),
                computed + count,
                True,
            ),
            ("text-prefill", [81, 82, 83], 13, True),
        ]
    )
    shifted = batch.input_ids.clone()
    shifted[0] = 92
    shifted[1 : count + 1] += 1
    shifted[count + 1 :] += 1
    managers, _ = _make_managers(16)
    for manager in managers:
        state, cache = make_state()
        cache.mm_features.update(
            text=[],
            images=[
                _feature("a", 2, 36),
                _feature("b", 42, 19),
            ],
            **{"text-prefill": []},
        )
        predictor = HYV4MultiTokenPredictor.__new__(HYV4MultiTokenPredictor)
        torch.nn.Module.__init__(predictor)
        predictor.embed_tokens = lambda ids: ids.float()[:, None].repeat(1, 2)
        predictor.requires_topk_indices_buffer = False
        predictor.mtp_start_layer_idx = 78
        predictor.layers = {"78": lambda ids, pos, hidden, embeds: embeds}
        model = HYV4MTP.__new__(HYV4MTP)
        torch.nn.Module.__init__(model)
        model.model = predictor
        local = manager.partition_batch(batch)
        manager.prepare_draft_prefill(batch, shifted)
        draft = object.__new__(pcp_mtp.HcuPCPMTPSpeculator)
        draft.pcp_manager = manager
        draft.target_model_state = None
        draft.input_buffers = SimpleNamespace(input_ids=shifted)
        draft.vllm_config = None
        draft.model = model
        draft.supports_mm_inputs = True
        draft.inputs_embeds = torch.full((128, 2), -77.0)
        draft._pcp_inputs_embeds = torch.full((128, 2), -88.0)
        draft.hidden_states = torch.ones(128, 2)
        for change in (0, 500):
            cache.encoder_outputs.update(
                a=(torch.arange(2, 38) + 1000 + change)[:, None].repeat(1, 2),
                b=(torch.arange(42, 61) + 1000 + change)[:, None].repeat(1, 2),
            )
            # Execute native cache slicing, including +1 at chunk boundaries.
            mm_inputs = state.gather_mm_embeddings(batch, draft_lookahead=1)
            actual, _ = draft._run_model(
                local.num_tokens_after_padding,
                {},
                {},
                None,
                mm_inputs=mm_inputs,
            )
            for row, req in enumerate(local.req_ids):
                lo, hi = local.query_start_loc_np[row : row + 2]
                for j in range(int(lo), int(hi)):
                    pos = int(local.positions[j])
                    if local.is_padding[j] or pos == 0:
                        expected = 0
                    elif req == "images":
                        next_pos = pos + 1
                        expected = (
                            1000 + change + next_pos
                            if 2 <= next_pos < 38 or 42 <= next_pos < 61
                            else next_pos + 1
                        )
                    elif req == "text":
                        expected = 92
                    else:
                        expected = 82 + pos - 10
                    assert actual[j].tolist() == [expected, expected]

        # Later draft steps are replicated text-only rows. They must overwrite
        # the image-bearing global scratch rather than reuse stale embeddings.
        manager.draft_prefill_batch = None
        draft.input_buffers = SimpleNamespace(
            input_ids=torch.tensor([9]),
            positions=torch.tensor([72]),
        )
        for mm_inputs in (None, ([], torch.zeros(1, dtype=torch.bool))):
            actual, _ = draft._run_model(1, {}, {}, None, mm_inputs=mm_inputs)
            assert actual.tolist() == [[9, 9]]
        draft.supports_mm_inputs = False
        actual, _ = draft._run_model(1, {}, {}, None)
        assert actual.tolist() == [[9, 9]]


@pytest.mark.parametrize(
    "capable,target_mm", [(True, True), (True, False), (False, True)]
)
def test_mtp_target_embedding_capability_respects_target_modalities(
    monkeypatch, capable, target_mm
):
    from vllm.multimodal import MULTIMODAL_REGISTRY

    from vllm_hcu.models.hy_v4.mtp import HYV4MTP
    from vllm_hcu.v1.spec_decode import pcp_mtp

    monkeypatch.setattr(pcp_mtp.MTPSpeculator, "load_model", lambda *a: None)
    monkeypatch.setattr(
        MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda config: target_mm,
    )
    draft = object.__new__(pcp_mtp.HcuPCPMTPSpeculator)
    draft.model = HYV4MTP if capable else SimpleNamespace()
    draft.vllm_config = SimpleNamespace(model_config=object())
    draft.supports_mm_inputs = False
    draft.hidden_states = torch.zeros(128, 2)
    draft.load_model(object())
    assert draft.supports_mm_inputs == (capable and target_mm)
    if draft.supports_mm_inputs:
        assert draft.inputs_embeds.shape == draft.hidden_states.shape
        assert draft.inputs_embeds.data_ptr() != draft._pcp_inputs_embeds.data_ptr()


@pytest.mark.parametrize("computed,count", [(0, 16), (3, 10), (6, 8), (11, 5)])
def test_native_multi_image_partial_windows_and_both_pcp_segments(
    make_state, computed, count
):
    """Local virtual rows must not restart native image windows at their first chunk."""
    batch = _make_batch(
        [
            (
                "images",
                list(range(computed + 1, computed + count + 1)),
                computed + count,
                True,
            )
        ]
    )
    managers, _ = _make_managers(16)
    canonical = batch.positions.clone()
    # Hand-derived full prompt: text2, image4, text3, image6, text1.
    full = [1, 2, 101, 102, 103, 104, 7, 8, 9, 201, 202, 203, 204, 205, 206, 16]
    for manager in managers:
        state, cache = make_state()
        cache.mm_features["images"] = [_feature("a", 2, 4), _feature("b", 9, 6)]
        cache.encoder_outputs.update(
            a=torch.arange(101, 105)[:, None].repeat(1, 2),
            b=torch.arange(201, 207)[:, None].repeat(1, 2),
        )
        state.pcp_manager = manager
        local = manager.partition_batch(batch)
        result = state.get_mm_embeddings({}, local, _req_states(batch))
        actual_positions = state.prepare_inputs(local, _req_states(batch))["positions"]
        assert result.data_ptr() != state.encoder_runner.inputs_embeds.data_ptr()
        assert actual_positions.data_ptr() != state.rope_state.positions.data_ptr()
        expected = torch.tensor(full[computed : computed + count], dtype=torch.float32)
        # Independently use canonical token ordinals to identify each local token.
        for j in range(local.num_tokens_after_padding):
            if local.is_padding[j]:
                assert result[j].tolist() == [0, 0]
                assert actual_positions[:, j].tolist() == [0, 0, 0]
            else:
                index = int(local.positions[j]) - computed
                assert result[j].tolist() == [expected[index], expected[index]]
                assert (
                    actual_positions[:, j].tolist() == [int(local.positions[j]) - 3] * 3
                )
        assert torch.equal(batch.positions, canonical)
        assert torch.equal(state.rope_state.seen[-1], batch.idx_mapping)


def test_native_custom_embed_holes_and_mixed_request_reorder(make_state):
    batch = _make_batch(
        [("text", [50], 40, False), ("image", list(range(1, 18)), 17, True)]
    )
    managers, _ = _make_managers(2)
    for manager in managers:
        state, cache = make_state(2)
        cache.mm_features.update(
            text=[],
            image=[
                _feature(
                    "holes", 2, 6, torch.tensor([True, False, True, False, True, True])
                )
            ],
        )
        cache.encoder_outputs["holes"] = torch.tensor(
            [[101, 101], [102, 102], [103, 103], [104, 104]]
        )
        state.pcp_manager = manager
        local = manager.partition_batch(batch)
        result = state.get_mm_embeddings({}, local, _req_states(batch))
        expected_image = [
            1,
            2,
            101,
            4,
            102,
            6,
            103,
            104,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
        ]
        for row, req in enumerate(local.req_ids):
            lo, hi = local.query_start_loc_np[row : row + 2]
            for j in range(int(lo), int(hi)):
                if local.is_padding[j]:
                    assert not result[j].any()
                else:
                    want = (
                        50 if req == "text" else expected_image[int(local.positions[j])]
                    )
                    assert result[j].tolist() == [want, want]


def test_decode_replicas_keep_embeddings_and_canonical_kv_position(make_state):
    batch = _make_batch([("decode", [91], 17, False)])
    managers, _ = _make_managers(16)
    for manager in managers:
        state, cache = make_state()
        cache.mm_features["decode"] = [_feature("already-transferred", 2, 4)]
        local = manager.partition_batch(batch)
        state.pcp_manager = manager
        assert state.get_mm_embeddings({}, local, _req_states(batch))[0].tolist() == [
            91,
            91,
        ]
        assert state.prepare_inputs(local, _req_states(batch))["positions"][
            :, 0
        ].tolist() == [13, 13, 13]
        assert local.positions[0].item() == 16
        assert local.seq_lens[0].item() == 17


def test_d_cached_image_skip_and_uncached_suffix_requires_native_cache(make_state):
    state, cache = make_state(1)
    cache.mm_features["image"] = [_feature("image-hash", 2, 6)]
    complete = _make_batch([("image", [90], 17, False)])
    assert state.get_mm_embeddings({}, complete, _req_states(complete))[0].tolist() == [
        90,
        90,
    ]
    suffix = _make_batch([("image", [6, 7, 8, 9], 9, True)])
    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        state.get_mm_embeddings({}, suffix, _req_states(suffix))
    cache.encoder_outputs["image-hash"] = torch.arange(101, 107)[:, None].repeat(1, 2)
    assert state.get_mm_embeddings({}, suffix, _req_states(suffix)).tolist() == [
        [104, 104],
        [105, 105],
        [106, 106],
        [9, 9],
    ]


def test_d_image_suffix_runs_native_encoder_and_populates_hash_cache(make_state):
    state, cache = make_state(1)
    feature = _feature("image-hash", 2, 6)
    output = torch.arange(101, 107, dtype=torch.float32)[:, None].repeat(1, 2)
    field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    feature.data = MultiModalKwargsItem(
        pixel_values=field.build_elems("pixel_values", output[None])[0]
    )
    cache.mm_features["image"] = [feature]
    suffix = _make_batch([("image", [6, 7, 8, 9], 9, True)])
    assert state.get_mm_embeddings(
        {"image": [0]}, suffix, _req_states(suffix)
    ).tolist() == [[104, 104], [105, 105], [106, 106], [9, 9]]
    assert torch.equal(cache.encoder_outputs["image-hash"], output)
    assert state.model.encoder_calls == 1
    complete = _make_batch([("image", [90], 17, False)])
    assert state.get_mm_embeddings({}, complete, _req_states(complete))[0].tolist() == [
        90,
        90,
    ]
    assert state.model.encoder_calls == 1


@pytest.mark.parametrize("pcp", [1, 16])
def test_explicit_dummy_before_manager_after_real_and_nested_exception(make_state, pcp):
    state, cache = make_state(pcp)
    batch = SimpleNamespace(num_tokens_after_padding=7)
    # Dummy must not read request fields or a saved real global layout.
    state.pcp_manager = SimpleNamespace(
        get_global_batch=lambda *_: pytest.fail("stale global read")
    )
    with state.dummy_run_scope(True):
        assert not state.prepare_inputs(batch, object())["positions"].any()
        with pytest.raises(ValueError), state.dummy_run_scope(True):
            raise ValueError("nested")
        assert not state.prepare_inputs(batch, object())["positions"].any()
    real = _make_batch([("text", list(range(1, 18)), 17, True)])
    cache.mm_features["text"] = []
    local = real
    if pcp > 1:
        managers, _ = _make_managers(pcp)
        state.bind_pcp_manager(managers[0])
        local = managers[0].partition_batch(real)
    state.get_mm_embeddings({}, local, _req_states(real))
    state.prepare_inputs(local, _req_states(real))
    with state.dummy_run_scope(True):
        assert state.prepare_inputs(batch, object())["positions"].shape == (3, 7)
        assert not state.prepare_inputs(batch, object())["positions"].any()
    if pcp > 1:
        state.pcp_manager = None
        with pytest.raises(RuntimeError, match="manager"):
            state.prepare_inputs(batch, object())
    with state.dummy_run_scope(True):
        state.dummy_inputs_embeds(7).fill_(17)
        assert not state.dummy_inputs_embeds(7).any()


def test_pcp1_capture_and_dummy_use_native_storage(make_state):
    state, cache = make_state(1)
    batch = _make_batch([("text", [1, 2, 3], 3, True)])
    cache.mm_features["text"] = []
    real_embed = state.get_mm_embeddings({}, batch, _req_states(batch))
    real_rope = state.prepare_inputs(batch, _req_states(batch))["positions"]
    capture = state.prepare_dummy_inputs(1, 3)
    with state.dummy_run_scope(True):
        dummy = state.prepare_inputs(batch, object())["positions"]
        assert (
            dummy.data_ptr() == real_rope.data_ptr() == capture["positions"].data_ptr()
        )
        assert (
            state.dummy_inputs_embeds(3).data_ptr()
            == real_embed.data_ptr()
            == capture["inputs_embeds"].data_ptr()
        )
        assert not dummy.any()
    assert state._hcu_vl_persistent_memory_bytes == 0
    assert not any(
        "local" in name and isinstance(value, torch.Tensor)
        for name, value in vars(state).items()
    )


def test_pcp1_disabled_images_prepare_positions_without_encoder(make_state):
    state, cache = make_state(1, with_encoder_cache=False)
    assert cache is None
    assert not state.supports_mm_inputs
    assert not hasattr(state, "encoder_runner")
    positions = state.rope_state.get_positions(5)
    positions.fill_(31)

    capture = state.prepare_dummy_inputs(2, 5)

    assert set(capture) == {"positions"}
    assert capture["positions"].shape == (3, 5)
    assert capture["positions"].data_ptr() == positions.data_ptr()
    assert not capture["positions"].any()


def test_native_request_and_attention_lifecycle_remains_inherited():
    for name in (
        "add_request",
        "remove_request",
        "apply_staged_writes",
        "preprocess_state",
        "postprocess_state",
        "prepare_attn",
    ):
        assert getattr(HYV4VLModelState, name) is getattr(
            default.DefaultModelState, name
        )


@pytest.mark.parametrize("negative", [False, True])
def test_state_memory_ledger_measures_native_and_local_allocation_delta(
    make_state, monkeypatch, negative
):
    # Exercise constructor accounting without touching CUDA: allocation requests
    # execute with CPU tensors; only the allocator counter is a test boundary.
    source, cache = make_state()
    allocated = 1000
    probes = 0

    def memory_allocated(device):
        nonlocal probes
        probes += 1
        return 0 if negative and probes == 2 else allocated

    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_allocated)
    zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        nonlocal allocated
        kwargs["device"] = "cpu"
        value = zeros(*args, **kwargs)
        allocated += value.numel() * value.element_size()
        return value

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    if negative:
        with pytest.raises(RuntimeError, match="negative"):
            HYV4VLModelState(
                source.vllm_config, source.model, cache, torch.device("cuda")
            )
    else:
        state = HYV4VLModelState(
            source.vllm_config, source.model, cache, torch.device("cuda")
        )
        # Native embed1024 + local embed1024 + local rope3096 bytes.
        assert state._hcu_vl_persistent_memory_bytes == 5144


@pytest.mark.parametrize("positional", [False, True])
def test_runner_carries_explicit_dummy_scope_and_restores_on_exception(
    pcp_runner_module, make_state, monkeypatch, positional
):
    module, _ = pcp_runner_module
    runner = module.HcuGPUModelRunnerV2(SimpleNamespace(), torch.device("cpu"))
    runner.model_state, _ = make_state()

    def native_execute(self, *args, **kwargs):
        result = self.model_state.prepare_inputs(
            SimpleNamespace(num_tokens_after_padding=5), object()
        )
        assert not result["positions"].any()
        raise ValueError("native execution failed")

    monkeypatch.setattr(module.GPUModelRunner, "execute_model", native_execute)
    with pytest.raises(ValueError, match="native execution failed"):
        if positional:
            runner.execute_model(object(), None, True)
        else:
            runner.execute_model(object(), dummy_run=True)
    with pytest.raises(RuntimeError, match="manager"):
        runner.model_state.prepare_inputs(
            SimpleNamespace(num_tokens_after_padding=5), object()
        )


@pytest.mark.parametrize("fail", [False, True])
def test_profile_context_sees_global_budget_and_load_accounts_state(
    pcp_runner_module, monkeypatch, fail
):
    from vllm_hcu.v1 import hyv4_vl_profile

    module, _ = pcp_runner_module
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=16),
        model_config=SimpleNamespace(architectures=["HYV4VLForConditionalGeneration"]),
    )
    runner = module.HcuGPUModelRunnerV2(config, torch.device("cpu"))
    runner.max_num_tokens = 128
    runner.model_config = SimpleNamespace(
        architectures=["HYV4VLForConditionalGeneration"]
    )
    seen = []

    @contextmanager
    def profile(self):
        seen.append(("enter", self.max_num_tokens))
        try:
            yield
        finally:
            seen.append(("exit", self.max_num_tokens))

    monkeypatch.setattr(hyv4_vl_profile, "profile_hyv4_vl_vision", profile)
    monkeypatch.setattr(
        hyv4_vl_profile,
        "account_hyv4_vl_state_memory",
        lambda self: seen.append(("account", self.model_state)),
    )

    def decoder(self):
        seen.append(("decoder", self.max_num_tokens))
        if fail:
            raise ValueError("decoder failed")

    monkeypatch.setattr(module.GPUModelRunner, "profile_run", decoder, raising=False)
    sentinel = object()
    monkeypatch.setattr(
        module.GPUModelRunner,
        "load_model",
        lambda self, *a, **kw: setattr(self, "model_state", sentinel),
        raising=False,
    )
    runner.load_model()
    if fail:
        with pytest.raises(ValueError, match="decoder failed"):
            runner.profile_run()
    else:
        runner.profile_run()
    assert seen == [
        ("account", sentinel),
        ("enter", 128),
        ("decoder", 9),
        ("exit", 128),
    ]


def test_runner_binds_manager_to_state_after_native_cache_initialization(
    pcp_runner_module, make_state, monkeypatch
):
    module, _ = pcp_runner_module
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=16, decode_context_parallel_size=1
        )
    )
    runner = module.HcuGPUModelRunnerV2(config, torch.device("cpu"))
    runner.model_state, cache = make_state()
    managers, _ = _make_managers(16)
    manager = managers[0]
    monkeypatch.setattr(module, "maybe_build_pcp_manager", lambda *args: manager)
    runner.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[object()]))
    batch = _make_batch([("text", [1, 2, 3], 3, True)])
    cache.mm_features["text"] = []
    local = manager.partition_batch(batch)
    embeds = runner.model_state.get_mm_embeddings({}, local, _req_states(batch))
    assert embeds[0].tolist() == [1, 1]
