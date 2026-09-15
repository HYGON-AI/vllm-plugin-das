# SPDX-License-Identifier: Apache-2.0
"""Check Omni metadata and lifecycle behavior with the HCU async policy."""

from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tests.runtime_patch.test_steady_decode_scheduler import (
    feature_env,
    make_scheduler,
    model_output,
    primed,
    state,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import PauseState
from vllm.v1.request import RequestStatus
from vllm_hcu.patch.platform.framework_opt import patch_scheduler
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.v1.core.sched.scheduler import HcuScheduler
from vllm_hcu.v1.core.sched.steady_decode_scheduler import try_steady_decode_schedule
from vllm_omni.core.sched.omni_hcu_scheduler import (
    OmniHcuAsyncScheduler,
    OmniHcuScheduler,
)
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.request import OmniRequest


def omni_request(rid, output=50, eos=False):
    sampling = SamplingParams(max_tokens=output, ignore_eos=not eos)
    sampling.update_from_generation_config({}, eos_token_id=42)
    return OmniRequest(
        request_id=rid,
        external_req_id="external-" + rid,
        prompt_token_ids=[1] * 4,
        sampling_params=sampling,
        pooling_params=None,
        arrival_time=1.0,
        additional_information=serialize_additional_information(
            {"omni_final_stage_id": 0}
        ),
    )


@pytest.mark.parametrize("cls", [OmniHcuScheduler, OmniHcuAsyncScheduler])
def test_bridge_preserves_final_stage_metadata(cls):
    scheduler = make_scheduler(cls)
    scheduler.add_request(omni_request("a"))
    output = scheduler.schedule().scheduled_new_reqs[0]
    assert output.external_req_id == "external-a"
    assert deserialize_additional_information(output.additional_information) == {
        "omni_final_stage_id": 0
    }
    original = make_scheduler(HcuScheduler)
    original.add_request(omni_request("a"))
    assert not hasattr(
        original.schedule().scheduled_new_reqs[0], "additional_information"
    )


@pytest.mark.parametrize("case", ["normal", "abort", "eos", "low_kv"])
def test_omni_fast_and_full_lifecycle_match(monkeypatch, case):
    full = make_scheduler(OmniHcuAsyncScheduler, blocks=8 if case == "low_kv" else 128)
    fast = make_scheduler(OmniHcuAsyncScheduler, blocks=8 if case == "low_kv" else 128)
    for scheduler in (full, fast):
        for i in range(8):
            scheduler.add_request(omni_request(str(i), eos=case == "eos"))
    queues = [deque(), deque()]
    tokens = [{}, {}]
    for step in range(600):
        outputs = []
        for scheduler, queue, enable in zip((full, fast), queues, (False, True)):
            monkeypatch.setattr(henvs, "VLLM_HCU_STEADY_DECODE_SCHED_FASTPATH", enable)
            output = scheduler.schedule()
            outputs.append(asdict(output))
            result = model_output(output)
            # The runner discards sampled tokens for partial recompute/prefill.
            # Snapshot this at submission, before the next in-flight schedule.
            for rid, index in result.req_id_to_index.items():
                if scheduler.requests[rid].is_prefill_chunk:
                    result.sampled_token_ids[index] = []
            queue.append((output, result))
        assert outputs[0] == outputs[1], step
        assert state(full) == state(fast), step
        if case == "abort" and step == 4:
            for scheduler in (full, fast):
                scheduler.finish_requests(["0", "1"], RequestStatus.FINISHED_ABORTED)
        if len(queues[0]) >= 2:
            results = []
            for scheduler, queue, collected in zip((full, fast), queues, tokens):
                output, runner_output = queue.popleft()
                result = scheduler.update_from_output(output, runner_output)
                results.append(result)
                for batch in result.values():
                    for item in batch.outputs:
                        collected.setdefault(item.request_id, []).extend(
                            item.new_token_ids
                        )
            assert tokens[0] == tokens[1], step
            assert state(full) == state(fast), step
        if not full.requests and not fast.requests:
            break
    else:
        pytest.fail("requests did not finish")
    assert full.kv_cache_manager.block_pool.get_num_free_blocks() == (
        7 if case == "low_kv" else 127
    )
    if case in ("normal", "low_kv"):
        assert all(len(t) == 50 for t in tokens[0].values())
        assert len(tokens[0]) == 8
        assert fast._vllm_hcu_steady_decode_hits > 0
    if case == "eos":
        assert all(t == [42] for t in tokens[0].values())


@pytest.mark.parametrize("pause", [PauseState.PAUSED_ALL, PauseState.PAUSED_NEW])
def test_paused_fastpath_falls_back_without_mutation(pause):
    scheduler = primed()
    scheduler.set_pause_state(pause)
    before = state(scheduler)
    assert try_steady_decode_schedule(scheduler) is None
    assert scheduler._vllm_hcu_steady_decode_last_reason == "paused"
    assert state(scheduler) == before


@pytest.mark.parametrize("capacity_bound", [False, True])
@pytest.mark.parametrize("throttle", [False, True])
def test_throttle_preserves_full_scheduler_capacity_state(capacity_bound, throttle):
    full = primed()
    fast = primed()
    full.prefill_capacity_bound = fast.prefill_capacity_bound = capacity_bound
    expected = full.schedule(throttle_prefills=throttle)
    actual = try_steady_decode_schedule(fast, throttle_prefills=throttle)
    assert asdict(actual) == asdict(expected)
    assert state(full) == state(fast)


@pytest.mark.parametrize(
    "path,async_mode",
    [
        (patch_scheduler.OMNI_HCU_SCHEDULER_PATH, False),
        (patch_scheduler.OMNI_HCU_ASYNC_SCHEDULER_PATH, True),
        (patch_scheduler.HCU_ASYNC_SCHEDULER_PATH, True),
    ],
)
def test_selector_retains_explicit_bridge(path, async_mode):
    config = SimpleNamespace(
        additional_config={"hcu": {}},
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            scheduler_cls=path, async_scheduling=async_mode
        ),
    )
    assert patch_scheduler.select_hcu_scheduler(config) is False
    assert config.scheduler_config.scheduler_cls == path
    config.scheduler_config.async_scheduling = not async_mode
    with pytest.raises(RuntimeError):
        patch_scheduler.select_hcu_scheduler(config)


def test_async_bridge_rejects_audio_output():
    class AudioScheduler(OmniHcuAsyncScheduler):
        def __init__(self, **kwargs):
            kwargs["vllm_config"].model_config.engine_output_type = "audio"
            super().__init__(**kwargs)

    with pytest.raises(ValueError, match="single text-final thinker"):
        make_scheduler(AudioScheduler)


def test_sync_bridge_keeps_downstream_payload_marker():
    scheduler = make_scheduler(OmniHcuScheduler)
    request = omni_request("downstream")
    request.additional_information = serialize_additional_information(
        {"omni_final_stage_id": 1}
    )
    scheduler.add_request(request)
    output = scheduler.schedule().scheduled_new_reqs[0]
    assert (
        deserialize_additional_information(output.additional_information)[
            "omni_final_stage_id"
        ]
        == 1
    )


def test_full_running_batch_preserves_skipped_waiting_queue():
    full = primed()
    fast = primed()
    for scheduler in (full, fast):
        request = omni_request("waiting")
        scheduler.add_request(request)
        scheduler.skipped_waiting.add_request(scheduler.waiting.pop_request())
    expected = full.schedule()
    actual = try_steady_decode_schedule(fast)
    assert asdict(actual) == asdict(expected)
    assert state(full) == state(fast)
