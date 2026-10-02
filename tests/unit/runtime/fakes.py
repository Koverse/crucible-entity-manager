import asyncio
from collections.abc import AsyncIterator, Sequence

from crucible_entity_manager.components.base import Subscription, owns_everything
from crucible_entity_manager.core.aliases import JSONObject, JSONValue


class FakeSource:
    """Yields the batches put into it; `end` makes iteration stop."""

    def __init__(self, name: str = "source", watchdog_seconds: float = 0.0) -> None:
        self.name = name
        self.last_activity_monotonic: float | None = None
        self.connected_since_monotonic: float | None = None
        self._watchdog_seconds = watchdog_seconds
        self._batches: asyncio.Queue[list[JSONObject] | None] = asyncio.Queue()

    @property
    def watchdog_seconds(self) -> float:
        return self._watchdog_seconds

    def push(self, *records: JSONObject) -> None:
        self._batches.put_nowait(list(records))

    def end(self) -> None:
        self._batches.put_nowait(None)

    def drained(self) -> bool:
        """Whether every pushed batch has been read."""
        return self._batches.empty()

    async def _iterate(self) -> AsyncIterator[list[JSONObject]]:
        while (batch := await self._batches.get()) is not None:
            yield batch

    def __aiter__(self) -> AsyncIterator[list[JSONObject]]:
        return self._iterate()


class FakeComponent:
    """Records every call; can block in `prepare`, wait on a gate or fail in `handle`."""

    name = "fake"

    def __init__(
        self,
        sources: Sequence[FakeSource],
        *,
        queue_max_records: int = 100,
        prepare_forever: bool = False,
        fail_on: str | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.queue_max_records = queue_max_records
        self.sources = sources
        self.prepare_forever = prepare_forever
        self.fail_on = fail_on
        self.gate = gate
        self.waiting = False
        self.calls: list[str] = []
        self.batches: list[tuple[str, list[JSONValue]]] = []

    def subscriptions(self) -> Sequence[Subscription]:
        return [Subscription(source.name, source, owns_everything) for source in self.sources]

    async def prepare(self) -> None:
        self.calls.append("prepare")
        if self.prepare_forever:
            await asyncio.Event().wait()

    async def handle(self, subscription: str, records: list[JSONObject]) -> None:
        if self.gate is not None:
            self.waiting = True
            await self.gate.wait()
        if any(record.get("id") == self.fail_on for record in records):
            raise RuntimeError("handler failed")
        self.calls.append("handle")
        self.batches.append((subscription, [record["id"] for record in records]))

    async def tick(self) -> None:
        self.calls.append("tick")

    async def close(self) -> None:
        self.calls.append("close")
