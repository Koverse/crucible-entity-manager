"""Signal handling and the hard shutdown deadline (DESIGN.md §5.7).

`block_shutdown_signals` must run in the main thread before any other thread
starts. Threads inherit the mask, so SIGTERM and SIGINT are then delivered only
to `ShutdownWatchdog`, which waits for them with ``sigwait``. Neither the
event loop nor the main thread has to be responsive for shutdown to begin, or
for the deadline to be enforced.
"""

import asyncio
import os
import signal
import threading
import time
from collections.abc import Callable
from typing import Final

from crucible_entity_manager.crucible.writer import LedgerSnapshot, WriteLedger

SHUTDOWN_SIGNALS: Final = frozenset({signal.SIGTERM, signal.SIGINT})
EXIT_CLEAN: Final = 0
EXIT_FAILURE: Final = 1
"""Authoritative records were outstanding or lost at exit, or the process failed."""


def block_shutdown_signals() -> None:
    """Block SIGTERM and SIGINT in the calling thread and every thread it starts."""
    signal.pthread_sigmask(signal.SIG_BLOCK, SHUTDOWN_SIGNALS)


def exit_code(snapshot: LedgerSnapshot) -> int:
    """The exit code a drain ending with `snapshot` reports."""
    return EXIT_CLEAN if snapshot.clean else EXIT_FAILURE


def deadline_summary(snapshot: LedgerSnapshot) -> str:
    """Describe what was left unwritten when the deadline expired."""
    if snapshot.clean:
        return "Shutdown deadline reached with no authoritative records outstanding.\n"
    outstanding = ", ".join(
        f"{name}={count}" for name, count in sorted(snapshot.outstanding.items())
    )
    lost = ", ".join(f"{name}={count}" for name, count in sorted(snapshot.lost.items()))
    return (
        "Shutdown deadline reached. "
        "Uncertain (in flight, may have been written): "
        f"{snapshot.total_outstanding} [{outstanding}]. "
        f"Lost (not written): {snapshot.total_lost} [{lost}].\n"
    )


class ShutdownWatchdog:
    """Turns a shutdown signal into a graceful drain, and ends the process at the deadline."""

    def __init__(  # noqa: PLR0913 - process hooks are injected for testing
        self,
        ledger: WriteLedger,
        drain_seconds: float,
        *,
        wait_for_signal: Callable[[], int] = lambda: signal.sigwait(SHUTDOWN_SIGNALS),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        write_stderr: Callable[[bytes], None] | None = None,
        exit_process: Callable[[int], None] = os._exit,
    ) -> None:
        self._ledger = ledger
        self._drain_seconds = drain_seconds
        self._wait_for_signal = wait_for_signal
        self._sleep = sleep
        self._clock = clock
        self._write_stderr = write_stderr or _write_stderr_without_blocking
        self._exit_process = exit_process
        self._lock = threading.Lock()
        self._deadline: float | None = None
        self._on_signal: Callable[[float], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> threading.Thread:
        """Start waiting for a signal in a daemon thread."""
        thread = threading.Thread(target=self._run, name="shutdown-watchdog", daemon=True)
        thread.start()
        return thread

    def attach(self, loop: asyncio.AbstractEventLoop, on_signal: Callable[[float], None]) -> None:
        """Call `on_signal(deadline)` on `loop` when a signal arrives, or now if one has.

        The deadline is on the monotonic clock, `drain_seconds` after the signal.
        """
        with self._lock:
            self._loop, self._on_signal = loop, on_signal
            deadline = self._deadline
        if deadline is not None:
            loop.call_soon_threadsafe(on_signal, deadline)

    def _run(self) -> None:
        self._wait_for_signal()
        deadline = self._clock() + self._drain_seconds
        with self._lock:
            self._deadline = deadline
            loop, on_signal = self._loop, self._on_signal
        if loop is not None and on_signal is not None:
            loop.call_soon_threadsafe(on_signal, deadline)
        self._sleep(max(0.0, deadline - self._clock()))
        snapshot = self._ledger.snapshot
        self._write_stderr(deadline_summary(snapshot).encode())
        self._exit_process(exit_code(snapshot))


def _write_stderr_without_blocking(message: bytes) -> None:
    """Write once to file descriptor 2, giving up rather than waiting on a full pipe."""
    try:
        os.set_blocking(2, False)
        os.write(2, message)
    except OSError:
        return
