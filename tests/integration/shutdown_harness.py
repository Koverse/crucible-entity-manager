"""A process that runs a stand-in component, for the shutdown tests.

Usage: ``python -m tests.integration.shutdown_harness MODE`` where MODE is
``quick`` (the final write returns), ``hang`` (the final write never returns),
``stuck`` (the owner loop blocks the event loop), ``fail`` (handling fails) or
``startup`` (building the component never finishes). Prints READY once records
are being handled, or once building has started.
"""

import asyncio
import logging
import sys
import threading
import time
from collections.abc import AsyncIterator, Sequence

from crucible_entity_manager.components.base import Component, Subscription, owns_everything
from crucible_entity_manager.config.perspective import WriteSettings
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.writer import (
    BatchWriter,
    DrainBudget,
    WriteClass,
    WriteLedger,
)
from crucible_entity_manager.runtime.process import exit_process, run_process

SETTINGS = RuntimeSettings(
    component="harness",
    perspective="TEST",
    drain_seconds=2.0,
    request_timeout_seconds=0.5,
    tick_seconds=0.1,
)


class Ticking:
    """A source yielding one record every 50 ms."""

    name = "ticking"
    last_activity_monotonic: float | None = None
    connected_since_monotonic: float | None = None
    watchdog_seconds = 0.0

    async def _iterate(self) -> AsyncIterator[list[JSONObject]]:
        while True:
            await asyncio.sleep(0.05)
            yield [{"id": 1}]

    def __aiter__(self) -> AsyncIterator[list[JSONObject]]:
        return self._iterate()


class Writes:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def write_batch(self, dataset: str, records: list[JSONObject]) -> None:
        del dataset, records
        if self.mode == "hang":
            threading.Event().wait()

    def update_batch(self, dataset: str, records: list[JSONObject]) -> list[JSONObject]:
        del dataset, records
        return []


class Search:
    async def search(self, sql: str, *, auto_backtick: bool = True) -> list[JSONObject]:
        del sql, auto_backtick
        return []


class Harness:
    name = "harness"
    queue_max_records = 10

    def __init__(self, mode: str, writer: BatchWriter) -> None:
        self.mode = mode
        self.writer = writer
        self.ready = False

    def subscriptions(self) -> Sequence[Subscription]:
        return [Subscription("ticking", Ticking(), owns_everything)]

    async def prepare(self) -> None:
        return

    async def handle(self, subscription: str, records: list[JSONObject]) -> None:
        del subscription, records
        if not self.ready:
            self.ready = True
            print("READY", flush=True)  # noqa: T201 - the test waits for this line
        if self.mode == "fail":
            msg = "handling failed"
            raise RuntimeError(msg)
        if self.mode == "stuck":
            time.sleep(60)  # noqa: ASYNC251 - deliberately blocks the event loop

    async def tick(self) -> None:
        return

    async def close(self) -> None:
        await self.writer.post("Events", [{"id": 1}], write_class=WriteClass.AUTHORITATIVE)


def main() -> None:
    mode = sys.argv[1]
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    async def build(ledger: WriteLedger, budget: DrainBudget) -> Component:
        if mode == "startup":
            print("READY", flush=True)  # noqa: T201 - the test waits for this line
            await asyncio.Event().wait()
        settings = WriteSettings(chunk_size=10, update_chunk_size=10, max_concurrent=1)
        writer = BatchWriter(Writes(mode), Search(), settings, ledger=ledger, budget=budget)
        return Harness(mode, writer)

    exit_process(run_process(SETTINGS, build))


if __name__ == "__main__":
    main()
