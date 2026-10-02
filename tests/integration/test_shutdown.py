"""Shutdown of a real process by a real SIGTERM (DESIGN.md §5.7)."""

import select
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAIN_SECONDS = 2.0
READY_TIMEOUT_SECONDS = 30.0
DEADLINE = "Shutdown deadline reached"


def run_until_ready(mode: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(  # noqa: S603 - fixed arguments
        [sys.executable, "-m", "tests.integration.shutdown_harness", mode],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], READY_TIMEOUT_SECONDS)
    assert readable, "the harness did not become ready"
    assert process.stdout.readline() == "READY\n"
    return process


def terminate(process: subprocess.Popen[str]) -> tuple[int, float, str]:
    started = time.monotonic()
    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=DRAIN_SECONDS + 10)
    return process.wait(), time.monotonic() - started, stderr


def test_a_clean_drain_exits_zero() -> None:
    code, _, stderr = terminate(run_until_ready("quick"))
    assert code == 0, stderr
    assert "Drained with every authoritative write confirmed" in stderr
    assert DEADLINE not in stderr


def test_a_write_that_never_returns_ends_at_the_deadline_with_exit_one() -> None:
    code, elapsed, stderr = terminate(run_until_ready("hang"))
    assert code == 1, stderr
    assert elapsed >= DRAIN_SECONDS
    assert stderr.endswith(
        "Shutdown deadline reached. Uncertain (in flight, may have been written): 1 [Events=1]. "
        "Lost (not written): 0 [].\n"
    )


def test_a_blocked_event_loop_cannot_delay_the_deadline() -> None:
    code, elapsed, stderr = terminate(run_until_ready("stuck"))
    assert code == 0, stderr
    assert elapsed >= DRAIN_SECONDS
    assert "Shutdown deadline reached with no authoritative records outstanding." in stderr


def test_a_failing_component_exits_one_without_a_signal() -> None:
    process = run_until_ready("fail")
    _, stderr = process.communicate(timeout=10)
    assert process.wait() == 1
    assert "harness failed" in stderr
    assert "RuntimeError: handling failed" in stderr


def test_a_signal_during_startup_stops_cleanly() -> None:
    code, _, stderr = terminate(run_until_ready("startup"))
    assert code == 0, stderr
    assert "Shutdown began during startup" in stderr
    assert DEADLINE not in stderr
