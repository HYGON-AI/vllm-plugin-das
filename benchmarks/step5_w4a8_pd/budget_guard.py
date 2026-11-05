"""Explicit launch-time96GiB watchdog. Default invocation starts nothing."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--reservation-identity", type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.execute:
        print("Prepared only; no processes started. Explicit --execute is required.")
        return
    import psutil

    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        raise ValueError("Specify the manually reviewed serve command")
    output = Path(os.environ["STEP5_EXP_RUN"])
    output.mkdir(parents=True, exist_ok=False)
    reservation = 0
    holder = None
    if a.reservation_identity:
        holder = json.loads(a.reservation_identity.read_text())
        if holder["bytes_per_device"] != 48 * 2**30:
            raise RuntimeError("Unexpected reservation size")
        process = psutil.Process(holder["pid"])
        if abs(process.create_time() - holder["created"]) >= 0.05 or not any(
            arg.endswith("/hold_48gib.py") for arg in process.cmdline()
        ):
            raise RuntimeError("Reservation identity changed")
        reservation = holder["bytes_per_device"]

    def memory():
        values = {
            f.parts[-3]: int(f.read_text())
            for f in Path("/sys/class/drm").glob("card[0-9]*/device/mem_info_vram_used")
        }
        if len(values) != 8:
            raise RuntimeError("Expected eight GPU memory counters")
        if holder:
            process = psutil.Process(holder["pid"])
            if (
                abs(process.create_time() - holder["created"]) >= 0.05
                or process.status() == psutil.STATUS_ZOMBIE
            ):
                raise RuntimeError("Reservation exited; stop owned test")
            if min(values.values()) < reservation:
                raise RuntimeError("Reservation memory is missing")
        return {name: max(0, value - reservation) for name, value in values.items()}

    # Subtract only the exact live, user-authorized holder. Preserve other workloads.
    idle_deadline = time.monotonic() + 60
    while max(memory().values()) >= 2 * 2**30:
        if time.monotonic() >= idle_deadline:
            raise RuntimeError("GPUs are occupied; preserve other workloads")
        time.sleep(0.25)
    owned = {}
    child = None
    peak = {}
    reason = "completed"
    code = None

    def live(pid, created):
        try:
            q = psutil.Process(pid)
            return (
                abs(q.create_time() - created) < 0.05
                and q.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            return False

    try:
        guard = {"pid": os.getpid(), "created": psutil.Process().create_time()}
        (output / "guard-identity.json").write_text(json.dumps(guard))
        environment = dict(os.environ, STEP5_W4_BUDGET_GUARD=str(os.getpid()))
        with (output / "server.log").open("x") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
        identity = (child.pid, psutil.Process(child.pid).create_time())
        owned[identity[0]] = identity[1]
        while child.poll() is None:
            if live(*identity):
                parent = psutil.Process(child.pid)
                for q in [parent, *parent.children(recursive=True)]:
                    owned[q.pid] = q.create_time()
            values = memory()
            peak = {k: max(v, peak.get(k, 0)) for k, v in values.items()}
            (output / "memory-peak.json").write_text(
                json.dumps(
                    {
                        "global_vram_peak_bytes": peak,
                        "soft_stop_bytes": 94 * 2**30,
                        "validation_limit_bytes": 96 * 2**30,
                        "reservation_subtracted": reservation,
                    }
                )
            )
            if max(values.values()) >= 94 * 2**30:
                reason = "memory_soft_stop"
                raise RuntimeError("Stop owned service before96GiB validation limit")
            time.sleep(0.05)
        code = child.returncode
    except BaseException as error:
        reason = reason if reason != "completed" else str(error)
        raise
    finally:
        for action, seconds in [("terminate", 20), ("kill", 10)]:
            for pid, created in owned.items():
                if live(pid, created):
                    getattr(psutil.Process(pid), action)()
            end = time.monotonic() + seconds
            while (
                any(live(pid, t) for pid, t in owned.items()) and time.monotonic() < end
            ):
                time.sleep(0.1)
        (output / "guard-result.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "returncode": code,
                    "peak_bytes": peak,
                    "owns_only_recorded_pid_creation_pairs": True,
                    "holder_restarted": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
