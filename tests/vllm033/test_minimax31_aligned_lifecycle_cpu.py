"""Lifecycle contract tests; no mock results are treated as device numerics."""
from types import SimpleNamespace
import threading
import pytest
from triton_tmp_dir import minimax31_moe_sglang_v1 as module


def bare():
    obj = module.AlignedMoE.__new__(module.AlignedMoE)
    obj.device = "test-device"
    obj.closed = False
    obj._lock = threading.Lock()
    obj.stream = 11
    obj.experts = SimpleNamespace(stream=11)
    return obj


def test_wrong_stream_is_rejected_before_any_work(monkeypatch):
    obj = bare()
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=12))
    obj._run = lambda *args: pytest.fail("dispatched on foreign stream")
    with pytest.raises(RuntimeError, match="stream"):
        obj(None, None, None)
    assert not obj._lock.locked()


def test_close_rejects_inflight_call_before_synchronize_or_destroy(monkeypatch):
    obj = bare()
    obj.buffer = SimpleNamespace(destroy=lambda: pytest.fail("destroyed inflight buffer"))
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: pytest.fail("synchronized inflight call"))
    obj._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="inflight"):
            obj.close()
    finally:
        obj._lock.release()
    assert not obj.closed


@pytest.mark.parametrize("failure", ["shared", "routed"])
def test_exception_joins_shared_work_and_records_input_first(monkeypatch, failure):
    from contextlib import nullcontext
    events = []
    main = SimpleNamespace(cuda_stream=1, wait_stream=lambda s: events.append("join"))
    side = SimpleNamespace(wait_stream=lambda s: events.append("side-wait"))
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda d: main)
    monkeypatch.setattr(module.torch.cuda, "stream", lambda s: nullcontext())
    obj = bare()
    obj.stream = 1
    obj.device = "device"
    obj.experts = SimpleNamespace(h=512, stream=1)
    obj.shared_stream = side
    def shared(hidden):
        events.append("shared")
        if failure == "shared":
            raise RuntimeError("test failure")
        return object()
    obj.shared_fn = shared
    def quant(hidden):
        events.append("routed")
        raise RuntimeError("test failure")
    obj.quant = quant
    hidden = SimpleNamespace(shape=(0,512), ndim=2, dtype=module.torch.bfloat16,
        device="device", is_contiguous=lambda: True,
        record_stream=lambda s: events.append("record-input"))
    ids = SimpleNamespace(shape=(0,4), dtype=module.torch.int64, device="device",
                          is_contiguous=lambda: True, numel=lambda: 0)
    weights = SimpleNamespace(shape=(0,4), dtype=module.torch.float32,
                              device="device", is_contiguous=lambda: True)
    with pytest.raises(RuntimeError, match="test failure"):
        obj(hidden, ids, weights)
    assert events.index("record-input") < events.index("shared")
    assert events[-1] == "join"
    assert not obj._lock.locked()
