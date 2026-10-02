import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crucible_entity_manager.components import duplicates as duplicates_module
from crucible_entity_manager.components.detection import Duplicate
from crucible_entity_manager.components.duplicates import (
    DuplicateIdentifier,
    build,
    events_query,
    supersede_event,
)
from crucible_entity_manager.config.perspective import parse_perspective
from crucible_entity_manager.config.runtime import DuplicateOptions, RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.protocols import TransientError
from tests.unit.components.fakes import FakeServices
from tests.unit.config.test_perspective import feed_row, perspective_row

T0 = datetime(2026, 9, 30, 12, tzinfo=UTC)
A, B, C, D = (f"{index:032x}" for index in range(1, 5))
ORIGIN = np.array([4_000_000.0, 1_000_000.0, 4_800_000.0])


def stamp(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def track_events(track: str, offset_m: float, start: float = 0.0) -> list[JSONObject]:
    events: list[JSONObject] = []
    for step in range(12):
        seconds = start + step * 20.0
        x, y, z = ORIGIN + np.array([offset_m, 0.0, 0.0]) + np.array([5.0, 0.0, 0.0]) * seconds
        events.append(
            {
                "trackId": track,
                "interceptTimestamp": stamp(seconds),
                "environment": "SEA_SURFACE",
                "ecefPosition": {"x": float(x), "y": float(y), "z": float(z)},
                "ecefVelocity": {"x": 5.0, "y": 0.0, "z": 0.0},
            }
        )
    return events


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def services(
    *,
    events: list[JSONObject],
    heads: list[JSONObject] | None = None,
    management: list[JSONObject] | None = None,
) -> FakeServices:
    row = perspective_row(
        component_track_event_dataset="ComponentTrackEvents",
        component_track_head_dataset="ComponentTrackHeads",
    )
    settings = RuntimeSettings(component="duplicates", perspective="Blue")
    service = FakeServices(
        settings,
        parse_perspective("Blue", [row, feed_row("Radar")], {}),
        clock=lambda: T0 + timedelta(minutes=10),
    )

    def respond(sql: str) -> list[JSONObject]:
        if "FROM ComponentTrackEvents" in sql:
            return events
        if "FROM ComponentTrackHeads" in sql:
            return heads or []
        if "FROM EntityManagementEvents" in sql:
            return management or []
        return []

    service.crucible.responder = respond
    return service


async def identifier(service: FakeServices, clock: Clock | None = None) -> DuplicateIdentifier:
    component = await build(service)
    assert isinstance(component, DuplicateIdentifier)
    if clock is not None:
        component._monotonic = clock
    return component


def written(service: FakeServices) -> list[tuple[str, str]]:
    return [
        (str(event["trackId"]), str(event["supersededBy"]))
        for event in service.crucible.posted.get("EntityManagementEvents", [])
    ]


def test_events_query_matches_the_baseline() -> None:
    assert events_query("Events", T0) == (
        "SELECT * FROM Events WHERE Events.`ecefPosition`.x IS NOT NULL "
        "AND Events.crucibleHeader.createdDate > '2026-09-30T12:00:00.000Z' "
        "ORDER BY Events.interceptTimestamp DESC LIMIT 50000"
    )


def test_supersede_event() -> None:
    duplicate = Duplicate(A, B, 1.234, 0.5, 12, 200.0, 0.876)
    event = supersede_event(B, A, duplicate)
    assert (event["action"], event["trackId"], event["supersededBy"]) == ("SUPERSEDE", B, A)
    assert event["source"] == {"datasetName": "entity_duplicate_identifier", "uuid": "0" * 32}
    assert event["source"] == event["sourceRaw"]
    assert event["source"] is not event["sourceRaw"]
    assert (event["mechanism"], event["status"], event["edhControlSet"]) == (
        "TRACKER",
        "COMPLETED",
        ["CLS:U"],
    )
    reason = str(event["reason"])
    assert reason.startswith(
        "Duplicate entity detected: avg_mahalanobis=1.2sigma, avg_vel_diff=0.5m/s"
    )
    assert len(reason) == 100


class TestPoll:
    async def test_supersedes_the_later_created_track(self) -> None:
        heads: list[JSONObject] = [
            {"trackId": A, "trackOriginatedTimestamp": stamp(60.0)},
            {"trackId": B, "trackOriginatedTimestamp": stamp(0.0)},
        ]
        service = services(events=track_events(A, 0.0) + track_events(B, 30.0, 3.0), heads=heads)
        component = await identifier(service)
        assert await component.poll() == 1
        assert written(service) == [(A, B)]
        head_query = next(
            query for query in service.crucible.queries if "ComponentTrackHeads" in query
        )
        assert head_query == f"SELECT * FROM ComponentTrackHeads WHERE trackId IN ('{A}', '{B}')"
        assert service.ledger.snapshot.clean

    async def test_a_track_superseded_earlier_in_the_poll_is_not_superseded_again(self) -> None:
        heads: list[JSONObject] = [
            {"trackId": A, "trackOriginatedTimestamp": stamp(0.0)},
            {"trackId": B, "trackOriginatedTimestamp": stamp(10.0)},
            {"trackId": C},
            {"trackOriginatedTimestamp": stamp(0.0)},
        ]
        events = track_events(A, 0.0) + track_events(B, 20.0, 1.0) + track_events(C, 40.0, 2.0)
        service = services(events=events, heads=heads)
        assert await (await identifier(service)).poll() >= 1
        pairs = written(service)
        superseded = [old for old, _ in pairs]
        survivors = {new for _, new in pairs}
        assert len(superseded) == len(set(superseded))
        assert not survivors & set(superseded)

    async def test_writes_stay_protected_while_the_search_lags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pairs = [
            Duplicate(A, B, 0.1, 0.1, 10, 100.0, 0.9),
            Duplicate(A, C, 0.1, 0.1, 10, 100.0, 0.8),
        ]
        pairs.append(Duplicate(B, D, 0.1, 0.1, 10, 100.0, 0.7))
        monkeypatch.setattr(duplicates_module, "find_duplicates", lambda histories, fallback: pairs)
        heads: list[JSONObject] = [
            {"trackId": A, "trackOriginatedTimestamp": stamp(0.0)},
            {"trackId": B, "trackOriginatedTimestamp": stamp(10.0)},
            {"trackId": C, "trackOriginatedTimestamp": stamp(20.0)},
            {"trackId": D, "trackOriginatedTimestamp": stamp(-10.0)},
        ]
        service = services(events=[], heads=heads)
        assert await (await identifier(service)).poll() == 2
        assert written(service) == [(B, A), (C, A)]

    async def test_a_survivor_superseded_since_detection_is_not_written_to(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        pairs = [Duplicate(A, B, 0.1, 0.1, 10, 100.0, 0.9)]
        monkeypatch.setattr(duplicates_module, "find_duplicates", lambda histories, fallback: pairs)
        heads: list[JSONObject] = [
            {"trackId": A, "trackOriginatedTimestamp": stamp(0.0)},
            {"trackId": B, "trackOriginatedTimestamp": stamp(10.0)},
        ]
        service = services(events=[], heads=heads)
        reads = 0

        def respond(sql: str) -> list[JSONObject]:
            nonlocal reads
            if "FROM ComponentTrackHeads" in sql:
                return heads
            if "action = 'SUPERSEDE'" in sql:
                reads += 1
                if reads > 1:
                    return [
                        {
                            "trackId": A,
                            "action": "SUPERSEDE",
                            "supersededBy": C,
                            "crucibleHeader": {"updatedDate": stamp(0.0)},
                        }
                    ]
            return []

        service.crucible.responder = respond
        assert await (await identifier(service)).poll() == 0
        assert written(service) == []
        assert f"Skipped {B}: its survivor {A} was superseded, deleted or restored" in caplog.text

    async def test_protected_tracks_are_left_alone(self) -> None:
        management: list[JSONObject] = [
            {
                "trackId": A,
                "action": "SUPERSEDE",
                "supersededBy": C,
                "crucibleHeader": {"updatedDate": stamp(0.0)},
            }
        ]
        service = services(
            events=track_events(A, 0.0) + track_events(B, 30.0), management=management
        )
        component = await identifier(service)
        assert await component.poll() == 0
        assert written(service) == []

    async def test_a_track_protected_since_detection_is_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        service = services(events=track_events(A, 0.0) + track_events(B, 30.0))
        reads = 0

        def respond(sql: str) -> list[JSONObject]:
            nonlocal reads
            if "FROM ComponentTrackEvents" in sql:
                return track_events(A, 0.0) + track_events(B, 30.0)
            if "action = 'SUPERSEDE'" in sql:
                reads += 1
                if reads > 1:
                    return [
                        {
                            "trackId": B,
                            "action": "SUPERSEDE",
                            "supersededBy": C,
                            "crucibleHeader": {"updatedDate": stamp(0.0)},
                        }
                    ]
            return []

        service.crucible.responder = respond
        component = await identifier(service)
        assert await component.poll() == 0
        assert f"Skipped {B}: it became protected" in caplog.text

    async def test_nothing_to_do(self) -> None:
        service = services(events=track_events(A, 0.0))
        assert await (await identifier(service)).poll() == 0
        assert written(service) == []

    async def test_a_failed_write_is_not_counted(self) -> None:
        service = services(events=track_events(A, 0.0) + track_events(B, 30.0))
        service.crucible.post_errors = {"EntityManagementEvents": TransientError("down")}
        assert await (await identifier(service)).poll() == 0


class TestTick:
    async def test_polls_on_the_interval(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        clock = Clock()
        service = services(events=[])
        component = await identifier(service, clock)
        await component.prepare()
        await component.tick()
        clock.now = 10.0
        await component.tick()
        clock.now = 30.0
        await component.tick()
        await component.close()
        event_queries = [
            query for query in service.crucible.queries if "ComponentTrackEvents" in query
        ]
        assert len(event_queries) == 2
        assert "Poll complete: 0 SUPERSEDE event(s) written" in caplog.text

    async def test_the_next_poll_waits_a_full_interval_after_a_slow_one(self) -> None:
        clock = Clock()
        service = services(events=[])
        component = await identifier(service, clock)

        def slow(sql: str) -> list[JSONObject]:
            clock.now += 50.0
            return []

        service.crucible.responder = slow
        await component.tick()
        finished = clock.now
        clock.now = finished + 29.0
        polls = len(service.crucible.queries)
        await component.tick()
        assert len(service.crucible.queries) == polls
        clock.now = finished + 30.0
        await component.tick()
        assert len(service.crucible.queries) > polls

    async def test_a_failed_poll_is_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        service = services(events=[])

        def unavailable(sql: str) -> list[JSONObject]:
            raise TransientError("down")

        service.crucible.responder = unavailable
        await (await identifier(service)).tick()
        assert "Poll skipped: down" in caplog.text

    async def test_it_has_no_subscriptions(self) -> None:
        component = await identifier(services(events=[]))
        assert component.subscriptions() == ()
        with pytest.raises(RuntimeError, match="no subscriptions"):
            await component.handle("x", [])


async def test_build_uses_options_and_falls_back_to_principal_datasets() -> None:
    settings = RuntimeSettings(
        component="duplicates",
        perspective="Blue",
        duplicates=DuplicateOptions(poll_interval_seconds=60.0, time_window_hours=2.0),
    )
    service = FakeServices(
        settings, parse_perspective("Blue", [perspective_row(), feed_row("Radar")], {})
    )
    component = await build(service)
    assert isinstance(component, DuplicateIdentifier)
    assert component._setup.targets.events == "ComponentTrackEvents"
    assert component._setup.options.poll_interval_seconds == 60.0
    row = perspective_row()
    del row["component_track_event_dataset"]
    del row["component_track_head_dataset"]
    fallback = FakeServices(settings, parse_perspective("Blue", [row, feed_row("Radar")], {}))
    built = await build(fallback)
    assert isinstance(built, DuplicateIdentifier)
    targets = built._setup.targets
    assert (targets.events, targets.heads) == ("PrincipalTrackEvents", "PrincipalTrackHeads")
