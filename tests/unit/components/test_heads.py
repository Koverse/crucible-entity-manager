import asyncio

import pytest

from crucible_entity_manager.components import heads as heads_module
from crucible_entity_manager.components.heads import HeadSync, HeadTargets
from crucible_entity_manager.config.perspective import WriteSettings
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.protocols import TransientError
from crucible_entity_manager.crucible.writer import BatchWriter, DrainBudget, WriteLedger
from tests.unit.components.fakes import FakeCrucible

HEADS, EVENTS = "Heads", "Events"


def key(index: int) -> str:
    return f"{index:032x}"


def head(index: int, version: int = 0) -> JSONObject:
    return {"trackId": key(index), "version": version}


def event(index: int, sequence: int) -> JSONObject:
    return {"trackId": key(index), "sequence": sequence}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def sync(
    crucible: FakeCrucible, clock: Clock | None = None, interval: float = 10.0
) -> tuple[HeadSync, WriteLedger]:
    ledger = WriteLedger()
    writer = BatchWriter(
        crucible,
        crucible,
        WriteSettings(chunk_size=100, update_chunk_size=100, max_concurrent=1),
        ledger=ledger,
        budget=DrainBudget(request_seconds=1.0),
    )
    return (
        HeadSync(
            writer,
            HeadTargets(HEADS, EVENTS),
            update_interval_seconds=interval,
            label="[test] ",
            clock=clock or Clock(),
        ),
        ledger,
    )


def sequences(crucible: FakeCrucible) -> list[tuple[str, object]]:
    return [
        (str(record["trackId"]), record["sequence"]) for record in crucible.posted.get(EVENTS, [])
    ]


def by_key(crucible: FakeCrucible) -> dict[str, list[object]]:
    """Each key's event sequence numbers, in write order (only per-key order is defined)."""
    ordered: dict[str, list[object]] = {}
    for track, sequence in sequences(crucible):
        ordered.setdefault(track, []).append(sequence)
    return ordered


async def settle(heads: HeadSync) -> None:
    """Wait for the background flush, if one is in flight."""
    if heads._flush is not None:
        await asyncio.wait({heads._flush})


class TestUnknownKeys:
    async def test_a_new_head_is_written_before_its_events(self) -> None:
        crucible = FakeCrucible()
        crucible.put_fails = {key(1)}
        heads, ledger = sync(crucible)
        await heads.publish([head(1)], [event(1, 1), event(1, 2)])
        assert crucible.log == [
            ("put", HEADS, [key(1)]),
            ("post", HEADS, [key(1)]),
            ("post", EVENTS, [key(1)]),
            ("post", EVENTS, [key(1)]),
        ]
        assert sequences(crucible) == [(key(1), 1), (key(1), 2)]
        assert heads.is_known(key(1))
        assert ledger.snapshot.clean

    async def test_an_existing_head_that_accepts_the_put_is_not_created(self) -> None:
        crucible = FakeCrucible()
        heads, _ = sync(crucible)
        await heads.publish([head(1)], [event(1, 1)])
        assert [entry[0] for entry in crucible.log] == ["put", "post"]
        assert crucible.queries == []

    async def test_events_wait_for_a_failed_create_and_are_released_in_order(self) -> None:
        crucible = FakeCrucible()
        crucible.put_fails = {key(1)}
        crucible.post_errors = {HEADS: TransientError("down")}
        heads, _ = sync(crucible)
        await heads.publish([head(1)], [event(1, 1)])
        await heads.publish([head(1, version=2)], [event(1, 2)])
        assert EVENTS not in crucible.posted
        assert heads.withheld == 2
        del crucible.post_errors[HEADS]
        await heads.tick()
        assert sequences(crucible) == [(key(1), 1), (key(1), 2)]
        assert crucible.posted[HEADS] == [head(1, version=2)]
        assert heads.withheld == 0

    async def test_a_head_that_exists_but_rejects_the_put_is_never_overwritten(self) -> None:
        crucible = FakeCrucible()
        crucible.put_fails = {key(1)}
        crucible.exists = {key(1)}
        heads, _ = sync(crucible)
        await heads.publish([head(1)], [event(1, 1)])
        assert HEADS not in crucible.posted
        assert heads.withheld == 1
        assert not heads.is_known(key(1))

    async def test_withheld_events_are_bounded_per_key_and_in_total(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(heads_module, "WITHHELD_PER_KEY", 2)
        monkeypatch.setattr(heads_module, "WITHHELD_TOTAL", 3)
        crucible = FakeCrucible()
        crucible.put_fails = {key(1), key(2)}
        crucible.exists = {key(1), key(2)}
        heads, _ = sync(crucible)
        await heads.publish([head(1)], [event(1, 1), event(1, 2), event(1, 3)])
        await heads.publish([head(2)], [event(2, 1), event(2, 2)])
        assert heads.withheld == 3
        crucible.put_fails = set()
        await heads.tick()
        assert by_key(crucible) == {key(1): [3], key(2): [1, 2]}
        assert "Dropped 2 withheld event(s)" in caplog.text

    async def test_a_new_key_at_the_total_limit_displaces_the_oldest_event(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(heads_module, "WITHHELD_TOTAL", 1)
        crucible = FakeCrucible()
        crucible.put_fails = {key(1), key(2)}
        crucible.exists = {key(1), key(2)}
        heads, _ = sync(crucible)
        await heads.publish([head(1), head(2)], [event(1, 1), event(2, 1)])
        assert heads.withheld == 1
        crucible.put_fails = set()
        await heads.tick()
        assert by_key(crucible) == {key(2): [1]}
        assert "Dropped 1 withheld event(s)" in caplog.text

    async def test_the_arrival_index_is_compacted(self) -> None:
        crucible = FakeCrucible()
        crucible.put_fails = {key(1)}
        crucible.exists = {key(1)}
        heads, _ = sync(crucible)
        for sequence in range(400):
            await heads.publish([head(1)], [event(1, sequence)])
        assert heads.withheld == heads_module.WITHHELD_PER_KEY
        assert len(heads._withheld._arrivals) <= 2 * heads.withheld + 64

    async def test_idle_unknown_keys_expire_with_their_events(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = Clock()
        crucible = FakeCrucible()
        crucible.put_fails = {key(1)}
        crucible.exists = {key(1)}
        heads, _ = sync(crucible, clock)
        await heads.publish([head(1)], [event(1, 1)])
        clock.now = 31 * 60.0
        await heads.tick()
        assert heads.withheld == 0
        assert "Dropped 1 withheld event(s)" in caplog.text

    async def test_records_without_a_key_are_ignored(self) -> None:
        crucible = FakeCrucible()
        heads, _ = sync(crucible)
        await heads.publish([{"other": 1}], [{"other": 2}])
        assert crucible.log == []


class TestKnownKeys:
    async def test_updates_are_deferred_to_the_interval_and_coalesced(self) -> None:
        clock = Clock()
        crucible = FakeCrucible()
        heads, _ = sync(crucible, clock, interval=10.0)
        heads.mark_known([key(1)])
        await heads.publish([head(1, version=1)], [event(1, 1)])
        await settle(heads)
        assert crucible.updated[HEADS] == [head(1, version=1)]
        clock.now = 5.0
        await heads.publish([head(1, version=2)], [])
        await heads.publish([head(1, version=3)], [])
        await settle(heads)
        assert crucible.updated[HEADS] == [head(1, version=1)]
        assert heads.pending_updates == 1
        clock.now = 10.0
        await heads.tick()
        await settle(heads)
        assert crucible.updated[HEADS] == [head(1, version=1), head(1, version=3)]

    async def test_one_flush_is_in_flight_at_a_time(self) -> None:
        crucible = FakeCrucible()
        heads, _ = sync(crucible, interval=0.0)
        heads.mark_known([key(1), key(2)])
        await heads.publish([head(1)], [])
        await heads.publish([head(2)], [])
        assert heads.pending_updates == 1
        await settle(heads)
        await heads.tick()
        await settle(heads)
        assert [entry[2] for entry in crucible.log] == [[key(1)], [key(2)]]

    async def test_a_failed_update_makes_the_key_unknown(self) -> None:
        crucible = FakeCrucible()
        crucible.put_fails = {key(1)}
        heads, _ = sync(crucible, interval=0.0)
        heads.mark_known([key(1)])
        await heads.publish([head(1)], [])
        await settle(heads)
        await heads.publish([head(1, version=2)], [event(1, 1)])
        assert ("post", HEADS, [key(1)]) in crucible.log
        assert sequences(crucible) == [(key(1), 1)]

    async def test_pending_updates_are_capped(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(heads_module, "PENDING_UPDATES_TOTAL", 1)
        clock = Clock()
        crucible = FakeCrucible()
        heads, _ = sync(crucible, clock, interval=10.0)
        heads.mark_known([key(1), key(2)])
        await heads.publish([head(1)], [])
        await settle(heads)
        clock.now = 1.0
        await heads.publish([head(1, version=2)], [])
        await heads.publish([head(2, version=2)], [])
        await settle(heads)
        assert crucible.updated[HEADS] == [head(1), head(2, version=2)]
        assert heads.pending_updates == 0
        await heads.tick()
        assert "Dropped 1 pending best-effort head update(s)" in caplog.text

    async def test_close_sends_updates_that_are_not_yet_due(self) -> None:
        clock = Clock()
        crucible = FakeCrucible()
        heads, ledger = sync(crucible, clock, interval=60.0)
        heads.mark_known([key(1)])
        await heads.publish([head(1, version=1)], [])
        await settle(heads)
        await heads.publish([head(1, version=2)], [])
        await heads.close()
        assert crucible.updated[HEADS] == [head(1, version=1), head(1, version=2)]
        assert heads.pending_updates == 0
        assert ledger.snapshot.clean

    async def test_close_waits_for_a_flush_in_flight(self) -> None:
        crucible = FakeCrucible()
        heads, _ = sync(crucible, interval=0.0)
        heads.mark_known([key(1)])
        await heads.publish([head(1)], [])
        await heads.close()
        assert crucible.updated[HEADS] == [head(1)]

    async def test_an_update_for_a_key_that_expired_is_still_sent(self) -> None:
        clock = Clock()
        crucible = FakeCrucible()
        heads, _ = sync(crucible, clock, interval=60.0)
        heads.mark_known([key(1)])
        await heads.publish([head(1)], [])
        await settle(heads)
        await heads.publish([head(1, version=2)], [])
        clock.now = 31 * 60.0
        await heads.tick()
        await settle(heads)
        assert crucible.updated[HEADS] == [head(1), head(1, version=2)]

    async def test_known_keys_expire_when_idle(self) -> None:
        clock = Clock()
        heads, _ = sync(FakeCrucible(), clock)
        heads.mark_known([key(1)])
        clock.now = 31 * 60.0
        await heads.tick()
        assert not heads.is_known(key(1))
