import asyncio
from collections.abc import Callable

import pytest

from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.crucible.writer import WriteLedger
from crucible_entity_manager.runtime.service import (
    STOPPED,
    Service,
    SourceEndedError,
    unless_stopped,
)
from crucible_entity_manager.runtime.shutdown import EXIT_CLEAN, EXIT_FAILURE
from tests.unit.runtime.fakes import FakeComponent, FakeSource

SETTINGS = RuntimeSettings(component="fake", perspective="LIVE", tick_seconds=0.01)


async def eventually(condition: Callable[[], bool], limit_seconds: float = 2.0) -> None:
    """Wait until `condition` holds, yielding to other tasks in between."""
    async with asyncio.timeout(limit_seconds):
        while not condition():  # noqa: ASYNC110 - the fakes' state has no event to await
            await asyncio.sleep(0.001)


def start(
    component: FakeComponent, ledger: WriteLedger | None = None
) -> tuple[asyncio.Task[int], asyncio.Event]:
    stopping = asyncio.Event()
    task = asyncio.create_task(Service(SETTINGS, ledger or WriteLedger(), stopping).run(component))
    return task, stopping


class TestUnlessStopped:
    async def test_returns_the_result_when_work_finishes_first(self) -> None:
        async def work() -> int:
            return 7

        assert await unless_stopped(asyncio.Event(), work()) == 7

    async def test_cancels_the_work_when_stopping(self) -> None:
        stopping = asyncio.Event()
        running, cancelled = asyncio.Event(), asyncio.Event()

        async def forever() -> None:
            running.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        task = asyncio.create_task(unless_stopped(stopping, forever()))
        await running.wait()
        stopping.set()
        assert await task is STOPPED
        assert cancelled.is_set()

    async def test_work_that_finishes_despite_cancellation_returns_its_result(self) -> None:
        stopping = asyncio.Event()
        running = asyncio.Event()

        async def stubborn() -> int:
            running.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return 3
            return 0

        task = asyncio.create_task(unless_stopped(stopping, stubborn()))
        await running.wait()
        stopping.set()
        assert await task == 3

    async def test_propagates_failures(self) -> None:
        async def broken() -> None:
            raise ValueError("broken")

        with pytest.raises(ValueError, match="broken"):
            await unless_stopped(asyncio.Event(), broken())

    async def test_work_cancelling_itself_propagates(self) -> None:
        async def cancelled_inside() -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await unless_stopped(asyncio.Event(), cancelled_inside())

    @pytest.mark.parametrize("stop_first", [False, True])
    async def test_cancelling_the_caller_propagates_and_cancels_the_work(
        self, *, stop_first: bool
    ) -> None:
        stopping = asyncio.Event()
        running, release = asyncio.Event(), asyncio.Event()
        cancelled = asyncio.Event()

        async def slow_to_cancel() -> None:
            running.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
                await release.wait()

        task = asyncio.create_task(unless_stopped(stopping, slow_to_cancel()))
        await running.wait()
        if stop_first:
            stopping.set()
            await cancelled.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        release.set()


class TestService:
    async def test_handles_records_ticks_and_drains(self) -> None:
        tracks, events = FakeSource("tracks"), FakeSource("events")
        component = FakeComponent([tracks, events])
        task, stopping = start(component)
        tracks.push({"id": 1}, {"id": 2})
        events.push({"id": 3})
        await eventually(lambda: len(component.batches) == 2 and "tick" in component.calls)
        stopping.set()
        assert await task == EXIT_CLEAN
        assert component.calls[0] == "prepare"
        assert sorted(component.batches) == [("events", [3]), ("tracks", [1, 2])]
        assert component.calls[-1] == "close"

    async def test_unconfirmed_writes_make_the_exit_code_a_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ledger = WriteLedger()
        ledger.add("Events", 2)
        component = FakeComponent([FakeSource()])
        task, stopping = start(component, ledger)
        await eventually(lambda: "prepare" in component.calls)
        stopping.set()
        assert await task == EXIT_FAILURE
        assert "Uncertain: {'Events': 2}" in caplog.text

    async def test_finishes_the_current_batch_and_reports_what_is_left(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        source = FakeSource()
        gate = asyncio.Event()
        component = FakeComponent([source], gate=gate)
        task, stopping = start(component)
        source.push({"id": 1})
        await eventually(lambda: component.waiting)
        source.push({"id": 2})
        await eventually(source.drained)
        stopping.set()
        gate.set()
        assert await task == EXIT_CLEAN
        assert component.batches == [("source", [1])]
        assert component.calls[-1] == "close"
        assert "Shutdown left 1 received record(s) unprocessed" in caplog.text

    async def test_shutdown_during_preload_skips_processing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        component = FakeComponent([FakeSource()], prepare_forever=True)
        task, stopping = start(component)
        await eventually(lambda: "prepare" in component.calls)
        stopping.set()
        assert await task == EXIT_CLEAN
        assert component.calls == ["prepare"]
        assert "during preload" in caplog.text

    async def test_a_source_that_ends_fails_the_service(self) -> None:
        source = FakeSource("tracks")
        task, _ = start(FakeComponent([source]))
        source.end()
        with pytest.raises(ExceptionGroup) as raised:
            await task
        assert raised.group_contains(SourceEndedError, match="source tracks ended")

    async def test_a_failing_handler_fails_the_service(self) -> None:
        source = FakeSource()
        task, _ = start(FakeComponent([source], fail_on="bad"))
        source.push({"id": "bad"})
        with pytest.raises(ExceptionGroup) as raised:
            await task
        assert raised.group_contains(RuntimeError, match="handler failed")
