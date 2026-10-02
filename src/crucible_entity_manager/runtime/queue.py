"""The bounded queue between record sources and the owner loop."""

import asyncio
from collections import deque
from dataclasses import dataclass

from crucible_entity_manager.components.base import Subscription
from crucible_entity_manager.core.aliases import JSONObject


class QueueClosedError(Exception):
    """A batch was offered to a closed queue."""


@dataclass(frozen=True, slots=True)
class Batch:
    """Records from one subscription, in arrival order."""

    subscription: str
    records: list[JSONObject]


class RecordQueue:
    """A FIFO of record batches, bounded by the total number of records.

    `put` waits while the queue is full, which stops the source reading and
    pushes backpressure down to TCP (DESIGN.md, decision D1). A batch larger
    than the bound is admitted only into an empty queue, so it can't wait forever.
    Once closed, `take` returns ``None`` and the remaining records stay unread.
    """

    def __init__(self, max_records: int) -> None:
        if max_records < 1:
            msg = f"max_records must be at least 1, got {max_records}"
            raise ValueError(msg)
        self._max_records = max_records
        self._batches: deque[Batch] = deque()
        self._records = 0
        self._closed = False
        self._changed = asyncio.Condition()

    @property
    def records_waiting(self) -> int:
        """Records queued and not yet taken."""
        return self._records

    async def put(self, batch: Batch) -> None:
        """Add `batch`, waiting for room.

        Raises:
            QueueClosedError: If the queue is closed before `batch` fits.
        """
        async with self._changed:
            await self._changed.wait_for(
                lambda: (
                    self._closed
                    or not self._batches
                    or self._records + len(batch.records) <= self._max_records
                )
            )
            if self._closed:
                raise QueueClosedError
            self._batches.append(batch)
            self._records += len(batch.records)
            self._changed.notify_all()

    def _ready(self) -> bool:
        return self._closed or bool(self._batches)

    async def close(self) -> None:
        """Stop accepting and handing out batches, waking anything waiting."""
        async with self._changed:
            self._closed = True
            self._changed.notify_all()

    async def take(self, max_records: int, wait_seconds: float) -> list[Batch] | None:
        """Take the oldest batches, up to `max_records` records in total.

        Waits up to `wait_seconds` for the first batch, and returns an empty
        list if none arrives, or ``None`` once the queue is closed. The first
        batch is always taken whole. Consecutive batches from the same
        subscription are merged.
        """
        async with self._changed:
            if not self._ready():
                # Checked first: wait_for with a zero timeout gives up without looking.
                try:
                    await asyncio.wait_for(self._changed.wait_for(self._ready), wait_seconds)
                except TimeoutError:
                    return []
            if self._closed:
                return None
            taken: list[Batch] = []
            count = 0
            while self._batches and (
                not taken or count + len(self._batches[0].records) <= max_records
            ):
                batch = self._batches.popleft()
                count += len(batch.records)
                if taken and taken[-1].subscription == batch.subscription:
                    taken[-1] = Batch(batch.subscription, taken[-1].records + batch.records)
                else:
                    taken.append(batch)
            self._records -= count
            self._changed.notify_all()
            return taken


async def feed(subscription: Subscription, queue: RecordQueue, max_batch_records: int) -> None:
    """Move `subscription`'s owned records into `queue`, in chunks, until cancelled."""
    async for records in subscription.source:
        owned = [record for record in records if subscription.owns(record)]
        for start in range(0, len(owned), max_batch_records):
            await queue.put(Batch(subscription.name, owned[start : start + max_batch_records]))
