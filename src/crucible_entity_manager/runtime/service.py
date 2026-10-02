"""Running one component: its sources, the owner loop and the drain (DESIGN.md §5.1, §5.7)."""

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from enum import Enum
from typing import Final, Literal

from crucible_entity_manager.components.base import Component, Subscription
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.crucible.writer import WriteLedger
from crucible_entity_manager.runtime.health import HealthMonitor, serving_health
from crucible_entity_manager.runtime.queue import RecordQueue, feed
from crucible_entity_manager.runtime.shutdown import exit_code

logger = logging.getLogger(__name__)


class Stopped(Enum):
    """The result of work abandoned because shutdown began."""

    STOPPED = "stopped"


STOPPED: Final = Stopped.STOPPED


class SourceEndedError(RuntimeError):
    """A record source stopped yielding, which a live subscription never does."""


async def unless_stopped[T](
    stopping: asyncio.Event, work: Awaitable[T]
) -> T | Literal[Stopped.STOPPED]:
    """Run `work`, cancelling it if `stopping` is set first.

    Returns `STOPPED` only when this function cancelled `work`. Cancelling the
    caller cancels `work` too, and propagates.

    Raises:
        Exception: Whatever `work` raises.
    """
    task = asyncio.ensure_future(work)
    stop = asyncio.ensure_future(stopping.wait())
    abandoned = False
    try:
        await asyncio.wait({task, stop}, return_when=asyncio.FIRST_COMPLETED)
        if not task.done():
            abandoned = True
            task.cancel()
            await asyncio.wait({task})
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        stop.cancel()
    if abandoned and task.cancelled():
        return STOPPED
    return task.result()


class Service:
    """Runs a component from preload until it has drained."""

    def __init__(
        self, settings: RuntimeSettings, ledger: WriteLedger, stopping: asyncio.Event
    ) -> None:
        """Run under `settings`; `stopping` is set when shutdown begins."""
        self._settings = settings
        self._ledger = ledger
        self._stopping = stopping

    async def run(self, component: Component) -> int:
        """Run `component` until shutdown, and return the exit code.

        Raises:
            ExceptionGroup: If a source, the owner loop or the component fails.
        """
        subscriptions = component.subscriptions()
        queue = RecordQueue(component.queue_max_records)
        monitor = HealthMonitor(
            queue,
            [subscription.source for subscription in subscriptions],
            stall_seconds=self._settings.stall_seconds,
        )
        async with serving_health(monitor, self._settings.health_port):
            if await unless_stopped(self._stopping, component.prepare()) is STOPPED:
                logger.warning("Shutdown began during preload; nothing was processed")
                return exit_code(self._ledger.snapshot)
            logger.info("Preload complete; consuming %d source(s)", len(subscriptions))
            monitor.mark_started()
            await self._consume(component, subscriptions, queue, monitor)
        snapshot = self._ledger.snapshot
        if snapshot.clean:
            logger.info("Drained with every authoritative write confirmed")
        else:
            logger.error(
                "Drained with authoritative writes unconfirmed. Uncertain: %s. Lost: %s.",
                dict(snapshot.outstanding),
                dict(snapshot.lost),
            )
        return exit_code(snapshot)

    async def _consume(
        self,
        component: Component,
        subscriptions: Sequence[Subscription],
        queue: RecordQueue,
        monitor: HealthMonitor,
    ) -> None:
        async with asyncio.TaskGroup() as tasks:
            feeders = [
                tasks.create_task(
                    self._feed(subscription, queue), name=f"source-{subscription.name}"
                )
                for subscription in subscriptions
            ]
            tasks.create_task(self._own(component, queue, monitor), name="owner")
            await self._stopping.wait()
            for feeder in feeders:
                feeder.cancel()
            await queue.close()

    async def _feed(self, subscription: Subscription, queue: RecordQueue) -> None:
        await feed(subscription, queue, self._settings.max_batch_records)
        msg = f"source {subscription.name} ended"
        raise SourceEndedError(msg)

    async def _own(self, component: Component, queue: RecordQueue, monitor: HealthMonitor) -> None:
        """The owner loop: the only task that calls the component after preload.

        Runs until the queue is closed, then reports what was left and closes
        the component.
        """
        loop = asyncio.get_running_loop()
        tick_seconds = self._settings.tick_seconds
        next_tick = loop.time() + tick_seconds
        while (
            batches := await queue.take(
                self._settings.max_batch_records, max(0.0, next_tick - loop.time())
            )
        ) is not None:
            monitor.mark_busy()
            for batch in batches:
                await component.handle(batch.subscription, batch.records)
            if loop.time() >= next_tick:
                await component.tick()
                next_tick = loop.time() + tick_seconds
            monitor.mark_progress()
        if queue.records_waiting:
            logger.warning("Shutdown left %d received record(s) unprocessed", queue.records_waiting)
        monitor.mark_busy()
        await component.close()
