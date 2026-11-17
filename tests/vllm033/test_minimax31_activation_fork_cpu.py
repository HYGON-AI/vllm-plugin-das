"""Extension admission must not wait on locks inherited from parent threads."""
import importlib
import os
import select
import signal
import threading
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize("module", [
    "hcu_vllm_das.quantization.minimax31_activation",
    "hcu_vllm_das.parallel.minimax31_pcp_activation",
    "hcu_vllm_das.quantization.preload_activation",
])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("operation", ["activate", "check_health"])
def test_child_admission_precedes_inherited_lock(monkeypatch, module, published, operation):
    owner = importlib.import_module(module)
    monkeypatch.setattr(owner, "_publication", object() if published else None)
    monkeypatch.setattr(owner, "_failure", None)
    held, release = threading.Event(), threading.Event()
    def hold():
        with owner._lock:
            held.set()
            release.wait(5)
    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(2)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            if operation == "activate":
                owner.activate(SimpleNamespace(requested_extensions=(), requested_capabilities=()))
            else:
                owner.check_health()
        except RuntimeError as error:
            os.write(write, str(error).encode())
        else:
            os.write(write, b"unexpected success")
        os._exit(0)
    os.close(write)
    try:
        assert select.select([read], [], [], 3)[0], "inherited extension lock deadlocked"
        message = os.read(read, 4096).decode()
        assert "process" in message if published else "unexpected success" not in message
    finally:
        release.set()
        thread.join(3)
        os.close(read)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
