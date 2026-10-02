import asyncio

import pytest

from crucible_entity_manager.components.base import Subscription
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.runtime.queue import Batch, QueueClosedError, RecordQueue, feed
from tests.unit.runtime.fakes import FakeSource


def batch(subscription: str, *ids: int) -> Batch:
    return Batch(subscription, [{"id": index} for index in ids])


def ids(batches: list[Batch] | None) -> list[tuple[str, list[object]]]:
    assert batches is not None
    return [(item.subscription, [record["id"] for record in item.records]) for item in batches]


class TestRecordQueue:
    def test_rejects_a_bound_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RecordQueue(0)

    async def test_takes_in_order_and_merges_runs_from_one_subscription(self) -> None:
        queue = RecordQueue(100)
        for item in (batch("a", 1), batch("a", 2), batch("b", 3), batch("a", 4)):
            await queue.put(item)
        assert ids(await queue.take(100, 0)) == [("a", [1, 2]), ("b", [3]), ("a", [4])]
        assert queue.records_waiting == 0

    async def test_take_stops_at_the_record_limit_but_always_takes_one_batch(self) -> None:
        queue = RecordQueue(100)
        await queue.put(batch("a", 1, 2, 3))
        await queue.put(batch("a", 4))
        assert ids(await queue.take(2, 0)) == [("a", [1, 2, 3])]
        assert queue.records_waiting == 1

    async def test_take_returns_nothing_when_no_batch_arrives_in_time(self) -> None:
        assert await RecordQueue(10).take(10, 0.01) == []

    async def test_put_waits_for_room(self) -> None:
        queue = RecordQueue(3)
        await queue.put(batch("a", 1, 2))
        waiting = asyncio.create_task(queue.put(batch("a", 3, 4)))
        await asyncio.sleep(0)
        assert not waiting.done()
        await queue.take(10, 0)
        await waiting
        assert queue.records_waiting == 2

    async def test_an_oversized_batch_enters_an_empty_queue(self) -> None:
        queue = RecordQueue(2)
        await asyncio.wait_for(queue.put(batch("a", 1, 2, 3)), 1)
        assert queue.records_waiting == 3

    async def test_closing_wakes_a_waiting_take_and_leaves_records_unread(self) -> None:
        queue = RecordQueue(10)
        await queue.put(batch("a", 1))
        await queue.take(10, 0)
        waiting = asyncio.create_task(queue.take(10, 60))
        await asyncio.sleep(0)
        await queue.put(batch("a", 2))
        assert ids(await waiting) == [("a", [2])]
        await queue.put(batch("a", 3))
        await queue.close()
        assert await queue.take(10, 60) is None
        assert queue.records_waiting == 1

    async def test_closing_wakes_a_waiting_take(self) -> None:
        queue = RecordQueue(10)
        waiting = asyncio.create_task(queue.take(10, 60))
        await asyncio.sleep(0)
        await queue.close()
        assert await asyncio.wait_for(waiting, 1) is None

    async def test_a_closed_queue_rejects_batches_including_waiting_ones(self) -> None:
        queue = RecordQueue(1)
        await queue.put(batch("a", 1))
        waiting = asyncio.create_task(queue.put(batch("a", 2)))
        await asyncio.sleep(0)
        await queue.close()
        with pytest.raises(QueueClosedError):
            await asyncio.wait_for(waiting, 1)
        with pytest.raises(QueueClosedError):
            await queue.put(batch("a", 3))


async def test_feed_keeps_owned_records_in_chunks() -> None:
    source = FakeSource("tracks")

    def even(record: JSONObject) -> bool:
        return int(str(record["id"])) % 2 == 0

    records: list[JSONObject] = [{"id": index} for index in range(10)]
    source.push(*records)
    source.end()
    queue = RecordQueue(100)
    await feed(Subscription("tracks", source, even), queue, max_batch_records=2)
    assert ids(await queue.take(2, 0)) == [("tracks", [0, 2])]
    assert ids(await queue.take(100, 0)) == [("tracks", [4, 6, 8])]
