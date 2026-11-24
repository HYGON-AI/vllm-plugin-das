"""Reusing target owners must refuse PID reuse and missing processes."""
import os
from pathlib import Path
import pytest
from hcu_vllm_das.parallel.minimax31_decode_preload import check_peer_owners


def test_reuse_requires_same_live_container_process():
    birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
    row={'pid':os.getpid(),'pid_namespace':'container','process_birth':birth}
    check_peer_owners([row])
    for bad in (dict(row,pid=True),dict(row,pid_namespace='host')):
        with pytest.raises(ValueError):check_peer_owners([bad])
    for bad in (dict(row,process_birth='stale'),dict(row,pid=2147483647)):
        with pytest.raises(RuntimeError):check_peer_owners([bad])
