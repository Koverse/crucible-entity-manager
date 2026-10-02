import asyncio
import logging

import pytest

from crucible_entity_manager.components.tracker import (
    FeedTracker,
    Tracker,
    TrackOwner,
    build,
    report_query,
    select_tracked_feeds,
    track_id_for,
)
from crucible_entity_manager.config.perspective import ConfigError, FeedConfig, parse_perspective
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.partition import PartitionSpec
from crucible_entity_manager.core.records import get_path
from crucible_entity_manager.crucible.protocols import TransientError
from tests.unit.components.fakes import FakeServices
from tests.unit.config.test_perspective import feed_row, perspective_row

HEADS, EVENTS = "ComponentTrackHeads", "ComponentTrackEvents"


def tracked_row(origin: str = "Radar", mode: str = "kalman", **extra: JSONValue) -> JSONObject:
    return feed_row(
        origin,
        crucible_tracker=mode,
        component_track_head_dataset=HEADS,
        component_track_event_dataset=EVENTS,
        **extra,
    )


def services(
    *rows: JSONObject, feed: str | None = None, partition: PartitionSpec | None = None
) -> FakeServices:
    settings = RuntimeSettings(
        component="tracker",
        perspective="Blue",
        feed=feed,
        partition=partition or PartitionSpec.single(),
    )
    return FakeServices(settings, parse_perspective("Blue", [perspective_row(), *rows], {}))


def radar_report(callsign: str, seconds: int, serial: int, **extra: JSONValue) -> JSONObject:
    record: JSONObject = {
        "identity": {"callsign": callsign},
        "estimatedKinematics": {"kinematicsTimestamp": f"2026-09-30T00:00:{seconds:02d}.000Z"},
        "ecefPosition": {"x": 1_000.0 + seconds, "y": 2_000.0, "z": 3_000.0},
        "source": {"uuid": f"report-{serial}", "datasetName": "Radar"},
    }
    record.update(extra)
    return record


async def started(service: FakeServices) -> Tracker:
    component = await build(service)
    assert isinstance(component, Tracker)
    await component.prepare()
    return component


async def drain(component: Tracker) -> None:
    await component.close()


class TestSelectTrackedFeeds:
    def feeds(self, *rows: JSONObject) -> tuple[FeedConfig, ...]:
        return parse_perspective("Blue", [perspective_row(), *rows], {}).feeds

    def test_keeps_tracked_feeds_with_component_datasets(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        feeds = self.feeds(
            tracked_row("A"),
            tracked_row("B", mode="passthrough"),
            tracked_row("C", mode="skip"),
        )
        selected = select_tracked_feeds(feeds, None)
        assert [tracked.feed.origin_dataset for tracked in selected] == ["A", "B"]
        assert "[C] Not tracked: crucible_tracker is 'skip'" in caplog.text

    def test_skips_feeds_without_component_datasets(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        perspective = perspective_row()
        del perspective["component_track_head_dataset"]
        rows = [perspective, feed_row("D", crucible_tracker="kalman"), tracked_row("E")]
        feeds = parse_perspective("Blue", rows, {}).feeds
        selected = select_tracked_feeds(feeds, None)
        assert [tracked.feed.origin_dataset for tracked in selected] == ["E"]
        assert "[D] Not tracked: component track datasets are not set" in caplog.text

    def test_an_unrecognized_mode_fails(self) -> None:
        with pytest.raises(ConfigError, match="'magic' names no tracker mode"):
            select_tracked_feeds(self.feeds(tracked_row("A", mode="magic")), None)

    def test_nothing_tracked_fails(self) -> None:
        with pytest.raises(ConfigError, match="no tracked feeds"):
            select_tracked_feeds(self.feeds(tracked_row("A", mode="skip")), None)
        with pytest.raises(ConfigError, match="no tracked feed is named 'B'"):
            select_tracked_feeds(self.feeds(tracked_row("A")), "B")


def test_report_query() -> None:
    assert report_query("ReportEvents", "Ra'dar") == (
        "SELECT * FROM ReportEvents WHERE ReportEvents.source.datasetName = 'Ra''dar'"
    )


def test_track_owner_partitions_by_track_id() -> None:
    (feed,) = parse_perspective("Blue", [perspective_row(), tracked_row()], {}).feeds
    reports = [radar_report(f"C{index}", 0, index) for index in range(40)]
    owners = [TrackOwner(feed, PartitionSpec(index, 3)) for index in range(3)]
    assert all(sum(owner(report) for owner in owners) == 1 for report in reports)
    report = reports[0]
    owner = next(owner for owner in owners if owner(report))
    assert owner.partition.owns(track_id_for(report, feed))


class TestTracker:
    async def test_tracks_reports_into_heads_then_events(self) -> None:
        service = services(tracked_row())
        component = await started(service)
        assert service.sources["Radar"][0] == report_query("ReportEvents", "Radar")
        assert component.queue_max_records == 50_000
        reports = [radar_report("A", 1, 1), radar_report("A", 2, 2), radar_report("B", 1, 3)]
        await component.handle("Radar", reports)
        await drain(component)
        crucible = service.crucible
        assert [entry[:2] for entry in crucible.log[:1]] == [("put", HEADS)]
        events = crucible.posted[EVENTS]
        assert [event["reportIds"] for event in events] == [
            ["report-1"],
            ["report-3"],
            ["report-2"],
        ]
        heads = crucible.updated[HEADS]
        assert sorted(str(head["trackUpdatedTimestamp"]) for head in heads) == [
            "2026-09-30T00:00:01.000Z",
            "2026-09-30T00:00:02.000Z",
        ]
        assert all("geodetic" in record for record in [*events, *heads])
        assert all("stale" in head and "reportIds" not in head for head in heads)
        assert service.ledger.snapshot.clean

    async def test_report_ids_follow_their_report_past_a_skipped_one(self) -> None:
        service = services(tracked_row())
        component = await started(service)
        reports = [
            radar_report("A", 1, 1),
            radar_report("B", 2, 2, ecefPosition={"x": 1.0}),
            radar_report("C", 3, 3),
        ]
        await component.handle("Radar", reports)
        await drain(component)
        assert [event["reportIds"] for event in service.crucible.posted[EVENTS]] == [
            ["report-1"],
            ["report-3"],
        ]

    async def test_repeated_reports_are_applied_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        service = services(tracked_row())
        component = await started(service)
        report = radar_report("A", 1, 1)
        copy: JSONObject = {**report, "crucibleHeader": {"uuid": "copy"}}
        await component.handle("Radar", [report, copy])
        await component.handle("Radar", [report])
        untraced = radar_report("A", 2, 2)
        del untraced["source"]
        await component.handle("Radar", [untraced, untraced])
        await drain(component)
        assert len(service.crucible.posted[EVENTS]) == 3
        assert "Dropped 1 report(s) already applied" in caplog.text
        assert "2 report(s) had no source.uuid and bypassed the repeat check" in caplog.text

    async def test_stale_and_timeless_reports_produce_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        service = services(tracked_row())
        component = await started(service)
        await component.handle("Radar", [radar_report("A", 5, 1)])
        timeless = radar_report("A", 6, 3)
        timeless["estimatedKinematics"] = {"kinematicsTimestamp": "soon"}
        await component.handle("Radar", [radar_report("A", 3, 2), timeless])
        await drain(component)
        assert [event["reportIds"] for event in service.crucible.posted[EVENTS]] == [["report-1"]]
        assert "Skipped 1 report(s) older than their track's state" in caplog.text
        assert (
            "Dropped 1 report(s) without a parseable estimatedKinematics.kinematicsTimestamp"
            in caplog.text
        )

    async def test_passthrough_copies_reports(self) -> None:
        service = services(tracked_row(mode="passthrough"))
        component = await started(service)
        await component.handle("Radar", [radar_report("A", 1, 1, geodetic={"latitude": 0.1})])
        await drain(component)
        (event,) = service.crucible.posted[EVENTS]
        assert event["geodetic"] == {"latitude": 0.1}
        (head,) = service.crucible.updated[HEADS]
        assert (head["speed"], head["heading"]) == (0.0, 0.0)

    async def test_preload_restores_owned_heads_for_the_feed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        (feed,) = parse_perspective("Blue", [perspective_row(), tracked_row()], {}).feeds
        known = radar_report("A", 0, 0)
        head: JSONObject = {
            "trackId": track_id_for(known, feed),
            "identity": {"callsign": "A"},
            "trackUpdatedTimestamp": "2026-09-30T00:00:00.000Z",
            "ecefPosition": {"x": 1_000.0, "y": 2_000.0, "z": 3_000.0},
            "positionCovariance": {"xx": 10.0, "yy": 10.0, "zz": 10.0},
            "velocityCovariance": {"dxdx": 1.0, "dydy": 1.0, "dzdz": 1.0},
        }
        foreign: JSONObject = {"trackId": "f" * 32, "identity": {"callsign": "Z"}}
        service = services(tracked_row(), tracked_row("Other", query=None))
        service.crucible.responder = lambda sql: [head, foreign, {"no": "id"}]
        component = await started(service)
        assert service.crucible.queries == [
            f"SELECT * FROM {HEADS} ORDER BY {HEADS}.crucibleHeader.updatedDate DESC LIMIT 100000"
        ]
        assert "[Radar] Restored 1 track(s) from heads" in caplog.text
        await component.handle("Radar", [radar_report("A", 1, 1)])
        await drain(component)
        assert ("post", HEADS, [str(head["trackId"])]) not in service.crucible.log
        (event,) = service.crucible.posted[EVENTS]
        assert get_path(event, "positionCovariance.xx") == pytest.approx(12.0, rel=1e-5)

    async def test_feeds_sharing_heads_load_once_to_the_largest_limit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        rows = [
            tracked_row("A", tracker_head_preload_limit=1),
            tracked_row("B", tracker_head_preload_limit=3),
        ]
        feeds = {
            feed.origin_dataset: feed
            for feed in parse_perspective("Blue", [perspective_row(), *rows], {}).feeds
        }

        def head(origin: str, callsign: str) -> JSONObject:
            report = radar_report(callsign, 0, 0)
            return {
                "trackId": track_id_for(report, feeds[origin]),
                "identity": {"callsign": callsign},
            }

        service = services(*rows)
        service.crucible.responder = lambda sql: [head("A", "A1"), head("B", "B1"), head("B", "B2")]
        await started(service)
        assert service.crucible.queries == [
            f"SELECT * FROM {HEADS} ORDER BY {HEADS}.crucibleHeader.updatedDate DESC LIMIT 3"
        ]
        assert "[A] Restored 1 track(s) from heads" in caplog.text
        assert "[B] Restored 2 track(s) from heads" in caplog.text

    async def test_preload_honors_skip_partition_and_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        skipped = services(tracked_row(skip_head_preload=True))
        await started(skipped)
        assert skipped.crucible.queries == []
        assert "[Radar] Head preload skipped" in caplog.text

        failing = services(tracked_row())

        def unavailable(sql: str) -> list[JSONObject]:
            raise TransientError("down")

        failing.crucible.responder = unavailable
        await started(failing)
        assert "Head preload from ComponentTrackHeads failed (down)" in caplog.text

    async def test_preload_warns_at_its_limit(self, caplog: pytest.LogCaptureFixture) -> None:
        service = services(tracked_row(tracker_head_preload_limit=1))
        service.crucible.responder = lambda sql: [{"trackId": "a" * 32}]
        await started(service)
        assert "hit its 1-row limit" in caplog.text

    async def test_tick_expires_idle_tracks_and_writes_due_heads(self) -> None:
        service = services(tracked_row())
        component = await started(service)
        await component.handle("Radar", [radar_report("A", 1, 1)])
        await component.tick()
        await drain(component)
        assert len(service.crucible.posted[EVENTS]) == 1

    async def test_selects_one_feed(self) -> None:
        service = services(tracked_row("A"), tracked_row("B"), feed="B")
        component = await started(service)
        assert [subscription.name for subscription in component.subscriptions()] == ["B"]
        assert isinstance(component.subscriptions()[0].owns, TrackOwner)


async def test_an_empty_batch_writes_nothing() -> None:
    service = services(tracked_row())
    component = await started(service)
    await component.handle("Radar", [])
    await drain(component)
    assert service.crucible.log == []


async def test_feed_tracker_outputs_are_compacted() -> None:
    service = services(tracked_row())
    component = await started(service)
    tracker = component._feeds["Radar"]
    assert isinstance(tracker, FeedTracker)
    _, events = tracker.outputs([radar_report("A", 1, 1, trackQuality=None)])
    assert "trackQuality" not in events[0]
    assert all(value is not None for value in events[0].values())
    await asyncio.sleep(0)
