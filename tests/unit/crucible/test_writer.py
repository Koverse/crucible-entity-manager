import asyncio
import threading
import time
from collections.abc import Awaitable, Callable

import pytest

from crucible_entity_manager.config.perspective import WriteSettings
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.protocols import (
    AuthorizationError,
    CrucibleError,
    RequestError,
    TransientError,
)
from crucible_entity_manager.crucible.writer import (
    BatchWriter,
    DrainBudget,
    HeadOutcome,
    WriteClass,
    WriteLedger,
    record_key,
    waves,
)

AUTHORITATIVE = WriteClass.AUTHORITATIVE
BEST_EFFORT = WriteClass.BEST_EFFORT

type Behavior = Callable[[str, str, list[JSONObject]], list[JSONObject] | None]
"""Decides one call's outcome from (method, dataset, records): raise, or return PUT failures."""


def ids(records: list[JSONObject]) -> list[str]:
    return [str(record["trackId"]) for record in records]


def track(index: int) -> str:
    return f"{index:032x}"


class FakeWrites:
    """A WriteClient whose outcomes come from `behavior`, recording every call."""

    def __init__(self, behavior: Behavior | None = None, delay: float = 0.0) -> None:
        self.behavior = behavior
        self.delay = delay
        self.calls: list[tuple[str, str, list[str]]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.ledger_seen: list[int] = []
        self.ledger: WriteLedger | None = None
        self._lock = threading.Lock()

    def _call(self, method: str, dataset: str, records: list[JSONObject]) -> list[JSONObject]:
        with self._lock:
            self.calls.append((method, dataset, ids(records)))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if self.ledger is not None:
                self.ledger_seen.append(self.ledger.snapshot.total_outstanding)
        try:
            if self.delay:
                time.sleep(self.delay)
            result = self.behavior(method, dataset, records) if self.behavior else None
            return result or []
        finally:
            with self._lock:
                self.in_flight -= 1

    def write_batch(self, dataset: str, records: list[JSONObject]) -> None:
        self._call("post", dataset, records)

    def update_batch(self, dataset: str, records: list[JSONObject]) -> list[JSONObject]:
        return self._call("put", dataset, records)


class FakeSearch:
    def __init__(
        self, existing: set[str] | None = None, error: CrucibleError | None = None
    ) -> None:
        self.existing = existing or set()
        self.error = error
        self.queries: list[str] = []

    async def search(self, sql: str, *, auto_backtick: bool = True) -> list[JSONObject]:
        del auto_backtick
        self.queries.append(sql)
        if self.error:
            raise self.error
        return [{"trackId": head_id} for head_id in sorted(self.existing) if f"'{head_id}'" in sql]


def records(count: int, start: int = 0) -> list[JSONObject]:
    return [{"trackId": track(index), "value": index} for index in range(start, start + count)]


def writer(
    client: FakeWrites,
    search: FakeSearch | None = None,
    settings: WriteSettings = WriteSettings(chunk_size=4, update_chunk_size=4, max_concurrent=2),  # noqa: B008
    budget: DrainBudget | None = None,
    ledger: WriteLedger | None = None,
) -> BatchWriter:
    ledger = ledger or WriteLedger()
    client.ledger = ledger
    return BatchWriter(
        client,
        search or FakeSearch(),
        settings,
        ledger=ledger,
        budget=budget or DrainBudget(request_seconds=1.0),
    )


class TestPost:
    async def test_writes_in_chunks(self) -> None:
        client = FakeWrites()
        outcome = await writer(client).post("Events", records(10), write_class=AUTHORITATIVE)
        assert outcome.failed == ()
        assert sorted(len(call[2]) for call in client.calls) == [2, 4, 4]

    async def test_bounds_concurrent_requests(self) -> None:
        client = FakeWrites(delay=0.02)
        await writer(client).post("Events", records(20), write_class=AUTHORITATIVE)
        assert client.max_in_flight == 2

    async def test_empty_batches_send_nothing(self) -> None:
        client = FakeWrites()
        assert (await writer(client).post("Events", [], write_class=AUTHORITATIVE)).failed == ()
        assert client.calls == []

    async def test_transient_failure_is_retried_once(self) -> None:
        attempts: list[int] = []

        def flaky(method: str, dataset: str, batch: list[JSONObject]) -> None:
            del method, dataset, batch
            attempts.append(1)
            if len(attempts) == 1:
                raise TransientError("timeout")

        outcome = await writer(FakeWrites(flaky)).post(
            "Events", records(3), write_class=AUTHORITATIVE
        )
        assert outcome.failed == ()
        assert len(attempts) == 2

    async def test_a_second_transient_failure_fails_the_chunk(self) -> None:
        def down(method: str, dataset: str, batch: list[JSONObject]) -> None:
            raise TransientError("timeout")

        client = FakeWrites(down)
        outcome = await writer(client).post("Events", records(3), write_class=AUTHORITATIVE)
        assert ids(list(outcome.failed)) == [track(0), track(1), track(2)]
        assert len(client.calls) == 2

    async def test_authorization_failure_fails_the_chunk_without_fallback(self) -> None:
        def denied(method: str, dataset: str, batch: list[JSONObject]) -> None:
            raise AuthorizationError("401")

        client = FakeWrites(denied)
        outcome = await writer(client).post("Events", records(4), write_class=AUTHORITATIVE)
        assert len(outcome.failed) == 4
        assert len(client.calls) == 1

    async def test_rejection_isolates_bad_records_in_sub_chunks(self) -> None:
        bad = track(5)

        def strict(method: str, dataset: str, batch: list[JSONObject]) -> None:
            if bad in ids(batch):
                raise RequestError("400", status=400, body="bad field")

        settings = WriteSettings(chunk_size=8, update_chunk_size=8, max_concurrent=1)
        client = FakeWrites(strict)
        outcome = await writer(client, settings=settings).post(
            "Events", records(8), write_class=AUTHORITATIVE
        )
        assert ids(list(outcome.failed)) == [track(4), track(5)]
        assert [len(call[2]) for call in client.calls] == [8, 2, 2, 2, 2]

    async def test_transient_failure_in_sub_chunks_gives_up_on_the_rest(self) -> None:
        def failing(method: str, dataset: str, batch: list[JSONObject]) -> None:
            if len(batch) == 8:
                raise RequestError("400", status=400, body="")
            if track(2) in ids(batch):
                raise TransientError("timeout")

        settings = WriteSettings(chunk_size=8, update_chunk_size=8, max_concurrent=1)
        outcome = await writer(FakeWrites(failing), settings=settings).post(
            "Events", records(8), write_class=AUTHORITATIVE
        )
        assert ids(list(outcome.failed)) == [track(index) for index in range(2, 8)]

    async def test_sub_chunk_rejections_include_authorization_failures(self) -> None:
        def failing(method: str, dataset: str, batch: list[JSONObject]) -> None:
            if len(batch) == 8:
                raise RequestError("400", status=400, body="")
            if track(0) in ids(batch):
                raise AuthorizationError("403")

        settings = WriteSettings(chunk_size=8, update_chunk_size=8, max_concurrent=1)
        outcome = await writer(FakeWrites(failing), settings=settings).post(
            "Events", records(8), write_class=AUTHORITATIVE
        )
        assert ids(list(outcome.failed)) == [track(0), track(1)]


class TestWaves:
    def test_each_key_appears_once_per_wave_in_order(self) -> None:
        batch: list[JSONObject] = [
            {"trackId": "a", "n": 1},
            {"trackId": "b", "n": 1},
            {"trackId": "a", "n": 2},
            {"trackId": {"uuid": "a"}, "n": 3},
            {"n": 0},
        ]
        result = waves(batch, "trackId")
        assert [[record["n"] for record in wave] for wave in result] == [[1, 1, 0], [2], [3]]

    async def test_waves_are_written_one_after_another(self) -> None:
        client = FakeWrites()
        batch = records(2) + records(2)
        await writer(client).post_in_waves(
            "Events", batch, key="trackId", write_class=AUTHORITATIVE
        )
        assert [call[2] for call in client.calls] == [[track(0), track(1)], [track(0), track(1)]]


class TestUpdate:
    async def test_reports_the_records_the_server_rejects(self) -> None:
        def partial(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return [{"trackId": {"uuid": track(1)}}]

        outcome = await writer(FakeWrites(partial)).update(
            "Heads", records(3), key="trackId", write_class=BEST_EFFORT
        )
        assert ids(list(outcome.failed)) == [track(1)]

    async def test_uses_the_update_chunk_size(self) -> None:
        client = FakeWrites()
        settings = WriteSettings(chunk_size=100, update_chunk_size=2, max_concurrent=1)
        await writer(client, settings=settings).update(
            "Heads", records(5), key="trackId", write_class=BEST_EFFORT
        )
        assert [len(call[2]) for call in client.calls] == [2, 2, 1]
        assert {call[0] for call in client.calls} == {"put"}


class TestRetryUnderDrain:
    async def test_budget_expiring_before_the_retry_fails_the_chunk(self) -> None:
        now = [0.0]
        budget = DrainBudget(request_seconds=1.0, clock=lambda: now[0])
        budget.start(10.0)

        def timeout_then_expire(method: str, dataset: str, batch: list[JSONObject]) -> None:
            now[0] = 9.5
            raise TransientError("timeout")

        client = FakeWrites(timeout_then_expire)
        outcome = await writer(client, budget=budget).post(
            "Events", records(2), write_class=AUTHORITATIVE
        )
        assert len(outcome.failed) == 2
        assert len(client.calls) == 1


class TestDrainBudget:
    def test_unlimited_until_draining(self) -> None:
        budget = DrainBudget(request_seconds=15.0, clock=lambda: 0.0)
        assert budget.allows_request()
        assert not budget.draining

    def test_requests_need_their_worst_case_before_the_deadline(self) -> None:
        now = [100.0]
        budget = DrainBudget(request_seconds=5.0, clock=lambda: now[0])
        budget.start(120.0)
        assert budget.draining
        assert budget.allows_request()
        now[0] = 115.5
        assert not budget.allows_request()

    async def test_writes_that_cannot_start_fail_without_a_request(self) -> None:
        budget = DrainBudget(request_seconds=15.0)
        budget.start(time.monotonic() + 5.0)
        client = FakeWrites()
        outcome = await writer(client, budget=budget).post(
            "Events", records(6), write_class=AUTHORITATIVE
        )
        assert len(outcome.failed) == 6
        assert client.calls == []

    async def test_sub_chunks_stop_when_the_budget_runs_out(self) -> None:
        now = [0.0]
        budget = DrainBudget(request_seconds=1.0, clock=lambda: now[0])
        budget.start(10.0)

        def reject_then_expire(method: str, dataset: str, batch: list[JSONObject]) -> None:
            now[0] = 9.5
            if len(batch) == 8:
                raise RequestError("400", status=400, body="")

        settings = WriteSettings(chunk_size=8, update_chunk_size=8, max_concurrent=1)
        client = FakeWrites(reject_then_expire)
        outcome = await writer(client, settings=settings, budget=budget).post(
            "Events", records(8), write_class=AUTHORITATIVE
        )
        assert len(outcome.failed) == 8
        assert len(client.calls) == 1


class TestLedger:
    async def test_authoritative_records_are_outstanding_until_settled(self) -> None:
        ledger = WriteLedger()
        client = FakeWrites()
        await writer(client, ledger=ledger).post("Events", records(6), write_class=AUTHORITATIVE)
        assert client.ledger_seen == [6, 6]
        assert ledger.snapshot.clean

    async def test_best_effort_records_are_not_counted(self) -> None:
        client = FakeWrites()
        await writer(client).update("Heads", records(2), key="trackId", write_class=BEST_EFFORT)
        assert client.ledger_seen == [0]

    async def test_failures_before_draining_are_not_lost(self) -> None:
        def down(method: str, dataset: str, batch: list[JSONObject]) -> None:
            raise AuthorizationError("401")

        ledger = WriteLedger()
        await writer(FakeWrites(down), ledger=ledger).post(
            "Events", records(3), write_class=AUTHORITATIVE
        )
        assert ledger.snapshot.clean

    async def test_failures_while_draining_are_lost(self) -> None:
        budget = DrainBudget(request_seconds=15.0)
        budget.start(time.monotonic() + 5.0)
        ledger = WriteLedger()
        await writer(FakeWrites(), ledger=ledger, budget=budget).post(
            "Events", records(3), write_class=AUTHORITATIVE
        )
        assert ledger.snapshot.total_outstanding == 0
        assert dict(ledger.snapshot.lost) == {"Events": 3}
        assert not ledger.snapshot.clean

    async def test_best_effort_failures_while_draining_are_not_lost(self) -> None:
        budget = DrainBudget(request_seconds=15.0)
        budget.start(time.monotonic() + 5.0)
        ledger = WriteLedger()
        await writer(FakeWrites(), ledger=ledger, budget=budget).update(
            "Heads", records(3), key="trackId", write_class=BEST_EFFORT
        )
        assert ledger.snapshot.clean

    async def test_a_write_that_raises_is_lost_while_draining(self) -> None:
        budget = DrainBudget(request_seconds=0.0)
        budget.start(time.monotonic() + 5.0)
        ledger = WriteLedger()
        with pytest.raises(ValueError, match="valid dataset name"):
            await writer(
                FakeWrites(lambda *_: list(_[2])), ledger=ledger, budget=budget
            ).write_heads("Bad-Name", records(2), key="trackId")
        assert dict(ledger.snapshot.lost) == {"Bad-Name": 2}

    async def test_a_cancelled_write_stays_outstanding(self) -> None:
        ledger = WriteLedger()
        client = FakeWrites(delay=0.2)
        task = asyncio.create_task(
            writer(client, ledger=ledger).post("Events", records(2), write_class=AUTHORITATIVE)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ledger.snapshot.total_outstanding == 2

    def test_snapshots_are_replaced_not_mutated(self) -> None:
        ledger = WriteLedger()
        ledger.add("A", 3)
        first = ledger.snapshot
        ledger.add("B", 2)
        ledger.settle("A", 3, lost=1)
        assert dict(first.outstanding) == {"A": 3}
        assert dict(ledger.snapshot.outstanding) == {"B": 2}
        assert dict(ledger.snapshot.lost) == {"A": 1}
        assert (ledger.snapshot.total_outstanding, ledger.snapshot.total_lost) == (2, 1)


class TestConcurrencySlots:
    async def test_cancelled_requests_hold_their_slot_until_the_thread_finishes(self) -> None:
        client = FakeWrites(delay=0.2)
        settings = WriteSettings(chunk_size=1, update_chunk_size=1, max_concurrent=1)
        batch_writer = writer(client, settings=settings)
        first = asyncio.create_task(
            batch_writer.post("Events", records(1), write_class=AUTHORITATIVE)
        )
        await asyncio.sleep(0.05)
        first.cancel()
        second = asyncio.create_task(
            batch_writer.post("Events", records(1, start=1), write_class=AUTHORITATIVE)
        )
        await asyncio.sleep(0.05)
        assert client.in_flight == 1
        assert len(client.calls) == 1
        await second
        assert client.max_in_flight == 1

    async def test_a_send_that_fails_before_starting_releases_its_slot(self) -> None:
        settings = WriteSettings(chunk_size=1, update_chunk_size=1, max_concurrent=1)
        batch_writer = writer(FakeWrites(), settings=settings)

        def broken(chunk: list[JSONObject]) -> Awaitable[list[JSONObject]]:
            raise RuntimeError("cannot start")

        with pytest.raises(RuntimeError, match="cannot start"):
            await batch_writer._request(broken, records(1))
        outcome = await asyncio.wait_for(
            batch_writer.post("Events", records(1), write_class=AUTHORITATIVE), timeout=1.0
        )
        assert outcome.failed == ()

    async def test_an_abandoned_failure_is_not_reported_as_unhandled(self) -> None:
        def slow_failure(method: str, dataset: str, batch: list[JSONObject]) -> None:
            time.sleep(0.1)
            raise TransientError("late")

        task = asyncio.create_task(
            writer(FakeWrites(slow_failure)).post("Events", records(1), write_class=AUTHORITATIVE)
        )
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.15)


class TestWriteHeads:
    async def write(
        self,
        client: FakeWrites,
        search: FakeSearch,
        heads: list[JSONObject],
    ) -> HeadOutcome:
        return await writer(client, search).write_heads("Heads", heads, key="trackId")

    async def test_heads_that_update_need_nothing_else(self) -> None:
        client, search = FakeWrites(), FakeSearch()
        outcome = await self.write(client, search, records(3))
        assert outcome == HeadOutcome(written=frozenset(map(track, range(3))), failed=frozenset())
        assert {call[0] for call in client.calls} == {"put"}
        assert search.queries == []

    async def test_missing_heads_are_created_and_existing_ones_are_not_overwritten(self) -> None:
        def put(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return [record for record in batch if method == "put" and record["value"] in (1, 2)]

        client, search = FakeWrites(put), FakeSearch(existing={track(1)})
        outcome = await self.write(client, search, records(3))
        assert outcome.failed == frozenset({track(1)})
        assert outcome.written == frozenset({track(0), track(2)})
        assert ("post", "Heads", [track(2)]) in client.calls
        (query,) = search.queries
        assert query == f"SELECT trackId FROM Heads WHERE trackId IN ('{track(1)}', '{track(2)}')"

    async def test_existence_query_failure_creates_nothing(self) -> None:
        def put(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return batch if method == "put" else []

        client, search = FakeWrites(put), FakeSearch(error=TransientError("down"))
        outcome = await self.write(client, search, records(2))
        assert outcome.failed == frozenset({track(0), track(1)})
        assert [call[0] for call in client.calls] == ["put"]

    async def test_keys_that_are_not_track_ids_are_never_created(self) -> None:
        def put(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return batch if method == "put" else []

        client, search = FakeWrites(put), FakeSearch()
        outcome = await self.write(client, search, [{"trackId": "not-a-uuid"}])
        assert outcome.failed == frozenset({"not-a-uuid"})
        assert search.queries == []
        assert [call[0] for call in client.calls] == ["put"]

    async def test_failed_creates_are_reported(self) -> None:
        def failing(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            if method == "post":
                raise AuthorizationError("401")
            return batch

        outcome = await self.write(FakeWrites(failing), FakeSearch(), records(1))
        assert outcome.failed == frozenset({track(0)})

    async def test_existence_is_checked_in_bounded_queries(self) -> None:
        def put(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return batch if method == "put" else []

        settings = WriteSettings(chunk_size=500, update_chunk_size=500, max_concurrent=1)
        search = FakeSearch()
        await writer(FakeWrites(put), search, settings=settings).write_heads(
            "Heads", records(450), key="trackId"
        )
        assert len(search.queries) == 3

    async def test_heads_are_authoritative(self) -> None:
        ledger = WriteLedger()
        client = FakeWrites()
        await writer(client, ledger=ledger).write_heads("Heads", records(2), key="trackId")
        assert client.ledger_seen == [2]
        assert ledger.snapshot.clean

    async def test_heads_without_a_key_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no trackId"):
            await writer(FakeWrites()).write_heads("Heads", [{"value": 1}], key="trackId")

    async def test_put_failures_for_unsent_keys_fail_the_whole_chunk(self) -> None:
        def foreign(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return [{"trackId": track(99)}] if method == "put" else []

        client, search = FakeWrites(foreign), FakeSearch(existing={track(0)})
        outcome = await self.write(client, search, records(1))
        assert outcome.failed == frozenset({track(0)})

    async def test_unmappable_put_failures_fail_the_whole_chunk(self) -> None:
        def anonymous(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return [{"error": "bad"}] if method == "put" else []

        client, search = FakeWrites(anonymous), FakeSearch(existing={track(0), track(1)})
        outcome = await self.write(client, search, records(2))
        assert outcome.failed == frozenset({track(0), track(1)})

    @pytest.mark.parametrize("dataset", ["Heads; DROP", "Heads-1"])
    async def test_rejects_unsafe_dataset_names(self, dataset: str) -> None:
        def put(method: str, dataset: str, batch: list[JSONObject]) -> list[JSONObject]:
            return batch

        with pytest.raises(ValueError, match="valid dataset name"):
            await writer(FakeWrites(put)).write_heads(dataset, records(1), key="trackId")


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"trackId": "a"}, "a"),
        ({"trackId": {"uuid": "b"}}, "b"),
        ({"trackId": None}, None),
        ({}, None),
    ],
)
def test_record_key(record: JSONObject, expected: str | None) -> None:
    assert record_key(record, "trackId") == expected
