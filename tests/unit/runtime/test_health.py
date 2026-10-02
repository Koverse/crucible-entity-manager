import socket

import aiohttp

from crucible_entity_manager.runtime.health import (
    HEALTH_PATH,
    HealthMonitor,
    HealthReport,
    serving_health,
)
from crucible_entity_manager.runtime.queue import Batch, RecordQueue
from tests.unit.runtime.fakes import FakeSource


def monitor(
    queue: RecordQueue, *sources: FakeSource, now: list[float], stall: float = 10.0
) -> HealthMonitor:
    return HealthMonitor(queue, sources, stall_seconds=stall, clock=lambda: now[0])


class TestHealthMonitor:
    def test_healthy_while_preloading(self) -> None:
        now = [0.0]
        health = monitor(RecordQueue(10), FakeSource("tracks", watchdog_seconds=1.0), now=now)
        now[0] = 1e6
        assert health.report() == HealthReport(healthy=True, problems=())

    async def test_a_stalled_owner_loop_is_unhealthy_only_while_input_waits(self) -> None:
        now = [0.0]
        queue = RecordQueue(10)
        health = monitor(queue, now=now)
        health.mark_started()
        now[0] = 60.0
        assert health.report().healthy
        await queue.put(Batch("a", [{"id": 1}]))
        assert health.report().problems == (
            "owner loop has made no progress for 60s; 1 record(s) waiting",
        )
        health.mark_progress()
        assert health.report().healthy

    def test_a_stalled_owner_loop_is_unhealthy_while_busy(self) -> None:
        now = [0.0]
        health = monitor(RecordQueue(10), now=now)
        health.mark_started()
        health.mark_busy()
        now[0] = 11.0
        assert health.report().problems == (
            "owner loop has made no progress for 11s; 0 record(s) waiting",
        )
        health.mark_progress()
        assert health.report().healthy

    def test_a_source_needs_activity_a_connection_or_the_start_within_its_watchdog(self) -> None:
        now = [100.0]
        source = FakeSource("tracks", watchdog_seconds=30.0)
        health = monitor(RecordQueue(10), source, now=now)
        health.mark_started()
        now[0] = 130.0
        assert health.report().healthy
        now[0] = 131.0
        assert health.report().problems == ("source tracks has shown no activity within 30s",)
        source.connected_since_monotonic = 120.0
        assert health.report().healthy
        source.last_activity_monotonic = 125.0
        now[0] = 155.0
        assert health.report().healthy
        now[0] = 156.0
        assert not health.report().healthy

    def test_a_source_without_a_watchdog_is_not_checked(self) -> None:
        now = [0.0]
        health = monitor(RecordQueue(10), FakeSource(), now=now)
        health.mark_started()
        now[0] = 1e6
        assert health.report().healthy


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def test_serves_the_report_over_http() -> None:
    now = [100.0]
    source = FakeSource("tracks", watchdog_seconds=30.0)
    health = monitor(RecordQueue(10), source, now=now)
    health.mark_started()
    now[0] = 200.0
    port = free_port()
    async with serving_health(health, port), aiohttp.ClientSession() as session:
        url = f"http://127.0.0.1:{port}{HEALTH_PATH}"
        async with session.get(url) as response:
            assert response.status == 503
            assert await response.json() == {
                "healthy": False,
                "problems": ["source tracks has shown no activity within 30s"],
            }
        source.last_activity_monotonic = 200.0
        async with session.get(url) as response:
            assert response.status == 200
            assert await response.json() == {"healthy": True, "problems": []}


async def test_serves_nothing_without_a_port() -> None:
    async with serving_health(monitor(RecordQueue(1), now=[0.0]), None):
        pass
