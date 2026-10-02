"""The health check behind ``GET /healthz`` (DESIGN.md §5.6)."""

import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

from aiohttp import web

from crucible_entity_manager.components.base import RecordSource
from crucible_entity_manager.runtime.queue import RecordQueue

HEALTH_PATH: Final = "/healthz"


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Whether the process is healthy, and why not."""

    healthy: bool
    problems: tuple[str, ...]


class HealthMonitor:
    """Judges health from the owner loop's progress and each source's activity."""

    def __init__(
        self,
        queue: RecordQueue,
        sources: Sequence[RecordSource],
        *,
        stall_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Watch `queue` and `sources`.

        Args:
            queue: The owner loop's input queue.
            sources: The record sources.
            stall_seconds: Longest time the owner loop may go without progress
                while input is waiting.
            clock: Monotonic time.

        Before `mark_started` the process is preloading and reports healthy.
        After it, the owner loop must make progress within `stall_seconds`
        while it is busy or input is waiting. A source must show activity, a
        connection, or the start itself within its `watchdog_seconds`; a
        source whose watchdog is disabled is not checked.
        """
        self._queue = queue
        self._sources = sources
        self._stall_seconds = stall_seconds
        self._clock = clock
        self._started_at: float | None = None
        self._last_progress = clock()
        self._busy = False

    def mark_started(self) -> None:
        """Record that preload is over and the sources and owner loop are starting."""
        self._started_at = self._last_progress = self._clock()

    def mark_busy(self) -> None:
        """Record that the owner loop has work in hand."""
        self._busy = True

    def mark_progress(self) -> None:
        """Record that the owner loop completed an iteration."""
        self._last_progress = self._clock()
        self._busy = False

    def report(self) -> HealthReport:
        """Check the owner loop and every source."""
        if self._started_at is None:
            return HealthReport(healthy=True, problems=())
        now = self._clock()
        problems: list[str] = []
        idle = now - self._last_progress
        if (self._busy or self._queue.records_waiting) and idle > self._stall_seconds:
            problems.append(
                f"owner loop has made no progress for {idle:.0f}s; "
                f"{self._queue.records_waiting} record(s) waiting"
            )
        problems.extend(
            f"source {source.name} has shown no activity within {source.watchdog_seconds:.0f}s"
            for source in self._sources
            if source.watchdog_seconds > 0
            and not self._source_active(source, self._started_at, now)
        )
        return HealthReport(healthy=not problems, problems=tuple(problems))

    def _source_active(self, source: RecordSource, started_at: float, now: float) -> bool:
        latest = max(
            moment
            for moment in (
                source.last_activity_monotonic,
                source.connected_since_monotonic,
                started_at,
            )
            if moment is not None
        )
        return now - latest <= source.watchdog_seconds


def health_app(monitor: HealthMonitor) -> web.Application:
    """Build the web app serving `HEALTH_PATH`: 200 when healthy, 503 otherwise."""

    async def healthz(request: web.Request) -> web.Response:
        del request
        report = monitor.report()
        return web.json_response(
            {"healthy": report.healthy, "problems": list(report.problems)},
            status=200 if report.healthy else 503,
        )

    app = web.Application()
    app.router.add_get(HEALTH_PATH, healthz)
    return app


@asynccontextmanager
async def serving_health(monitor: HealthMonitor, port: int | None) -> AsyncIterator[None]:
    """Serve `HEALTH_PATH` on `port` for the duration of the block; ``None`` serves nothing."""
    if port is None:
        yield
        return
    runner = web.AppRunner(health_app(monitor), access_log=None)
    await runner.setup()
    try:
        # Every interface: the kubelet probes the pod's IP, not loopback.
        await web.TCPSite(runner, host="0.0.0.0", port=port).start()  # noqa: S104
        yield
    finally:
        await runner.cleanup()
