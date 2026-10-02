import asyncio
import os
import signal
import threading

import pytest

from crucible_entity_manager.crucible.writer import WriteLedger
from crucible_entity_manager.runtime import shutdown
from crucible_entity_manager.runtime.shutdown import (
    EXIT_CLEAN,
    EXIT_FAILURE,
    ShutdownWatchdog,
    deadline_summary,
    exit_code,
)


def test_exit_code_and_summary_follow_the_snapshot() -> None:
    ledger = WriteLedger()
    assert exit_code(ledger.snapshot) == EXIT_CLEAN
    assert deadline_summary(ledger.snapshot) == (
        "Shutdown deadline reached with no authoritative records outstanding.\n"
    )
    ledger.add("Events", 3)
    ledger.add("Heads", 2)
    ledger.settle("Heads", 2, lost=1)
    assert exit_code(ledger.snapshot) == EXIT_FAILURE
    assert deadline_summary(ledger.snapshot) == (
        "Shutdown deadline reached. Uncertain (in flight, may have been written): 3 [Events=3]. "
        "Lost (not written): 1 [Heads=1].\n"
    )


class Harness:
    """Drives a watchdog without real signals, sleeping or exiting."""

    def __init__(self, ledger: WriteLedger) -> None:
        self.signal = threading.Event()
        self.written: list[bytes] = []
        self.exits: list[int] = []
        self.slept: list[float] = []
        self.watchdog = ShutdownWatchdog(
            ledger,
            20.0,
            wait_for_signal=self._wait,
            sleep=self.slept.append,
            clock=lambda: 1000.0,
            write_stderr=self.written.append,
            exit_process=self.exits.append,
        )

    def _wait(self) -> int:
        self.signal.wait()
        return int(signal.SIGTERM)


async def test_a_signal_starts_the_drain_and_the_deadline_exits() -> None:
    ledger = WriteLedger()
    ledger.add("Events", 1)
    harness = Harness(ledger)
    deadlines: asyncio.Queue[float] = asyncio.Queue()
    harness.watchdog.attach(asyncio.get_running_loop(), deadlines.put_nowait)
    thread = harness.watchdog.start()
    harness.signal.set()
    assert await asyncio.wait_for(deadlines.get(), 1) == 1020.0
    await asyncio.to_thread(thread.join, 1)
    assert harness.slept == [20.0]
    assert harness.exits == [EXIT_FAILURE]
    assert b"Uncertain (in flight, may have been written): 1 [Events=1]" in harness.written[0]


async def test_a_signal_before_attaching_starts_the_drain_on_attach() -> None:
    harness = Harness(WriteLedger())
    thread = harness.watchdog.start()
    harness.signal.set()
    await asyncio.to_thread(thread.join, 1)
    assert harness.exits == [EXIT_CLEAN]
    deadlines: asyncio.Queue[float] = asyncio.Queue()
    harness.watchdog.attach(asyncio.get_running_loop(), deadlines.put_nowait)
    assert await asyncio.wait_for(deadlines.get(), 1) == 1020.0


def test_stderr_is_written_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(os, "set_blocking", lambda fd, flag: calls.append(("blocking", (fd, flag))))
    monkeypatch.setattr(
        os, "write", lambda fd, data: calls.append(("write", (fd, data))) or len(data)
    )
    shutdown._write_stderr_without_blocking(b"summary")
    assert calls == [("blocking", (2, False)), ("write", (2, b"summary"))]


def test_a_full_stderr_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    def full(fd: int, data: bytes) -> int:
        raise BlockingIOError

    monkeypatch.setattr(os, "set_blocking", lambda fd, flag: None)
    monkeypatch.setattr(os, "write", full)
    shutdown._write_stderr_without_blocking(b"summary")
