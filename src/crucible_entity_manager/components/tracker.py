"""The tracker: report events in, component tracks out (DESIGN.md §5.2 to §5.5).

Each tracked feed subscribes to its reports in the report event dataset. A
batch is processed in timestamp order:

1. Reports without a parseable kinematics timestamp are dropped.
2. Each report gets its deterministic component ``trackId``.
3. Reports this track has already applied are dropped (D4).
4. Kalman feeds filter each report into a track state; passthrough feeds copy
   the report's kinematics.
5. Track events carry the ``reportIds`` of the report that produced them. The
   newest event per track becomes its head.

`HeadSync` then writes heads before events (D8, D11).
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, override

from crucible_entity_manager.components.base import Component, RecordSource, Services, Subscription
from crucible_entity_manager.components.heads import HeadSync, HeadTargets, preload_heads
from crucible_entity_manager.components.keyed import KeyedState, input_fingerprint
from crucible_entity_manager.components.tracking import (
    TIMESTAMP_PATH,
    Fusion,
    KalmanTracker,
    Outcome,
    TrackerParams,
    TrackState,
    head_from_event,
    kalman_track_event,
    measurement_from_report,
    passthrough_track_event,
)
from crucible_entity_manager.config.perspective import ConfigError, FeedConfig, TrackerMode
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.core.geodesy import set_track_geodetic
from crucible_entity_manager.core.identity import component_track_id, identity_custom_id
from crucible_entity_manager.core.partition import PartitionSpec
from crucible_entity_manager.core.records import MISSING, compact_record, get_path
from crucible_entity_manager.core.timeutil import format_timestamp, parse_timestamp
from crucible_entity_manager.crucible.protocols import SearchClient
from crucible_entity_manager.crucible.sql import identifier, string_literal
from crucible_entity_manager.crucible.writer import record_key

logger = logging.getLogger(__name__)

TRACKED_MODES: Final = frozenset(
    {TrackerMode.KALMAN, TrackerMode.KALMAN_CI, TrackerMode.PASSTHROUGH}
)
HEAD_STALE_AFTER: Final = timedelta(hours=10)
"""A head's ``stale`` time is this long after it was written, as at ``1b534df``."""


@dataclass(frozen=True, slots=True)
class TrackOwner:
    """Owns a report when its component ``trackId`` shards to this partition."""

    feed: FeedConfig
    partition: PartitionSpec

    def __call__(self, report: JSONObject) -> bool:
        """Whether this partition tracks `report`."""
        return self.partition.owns(track_id_for(report, self.feed))


def track_id_for(report: JSONObject, feed: FeedConfig) -> str:
    """The deterministic component ``trackId`` of a report in `feed`."""
    return component_track_id(feed.origin_dataset, identity_custom_id(report, feed.track_id_fields))


def report_query(reports_dataset: str, origin_dataset: str) -> str:
    """The SSE query for one feed's reports, as at ``1b534df``."""
    table = identifier(reports_dataset)
    return (
        f"SELECT * FROM {table} WHERE {table}.source.datasetName = {string_literal(origin_dataset)}"  # noqa: S608
    )


class FeedTracker:
    """Tracks one feed: its filter state and its head and event writes."""

    def __init__(
        self,
        feed: FeedConfig,
        heads_dataset: str,
        heads: HeadSync,
        tracks: KeyedState[TrackState],
        clock: Callable[[], datetime],
    ) -> None:
        self.feed = feed
        self.heads_dataset = heads_dataset
        self.heads = heads
        self._tracks = tracks
        self._clock = clock
        self._label = f"[{feed.origin_dataset}] "
        self._kalman = (
            None
            if feed.tracker.mode is TrackerMode.PASSTHROUGH
            else KalmanTracker(_params(feed), tracks, self._label)
        )

    def restore(self, heads: Sequence[JSONObject]) -> int:
        """Seed state from preloaded heads that belong to this feed; return how many."""
        restored: list[str] = []
        for head in heads:
            track_id = record_key(head, "trackId")
            if track_id is None or track_id != track_id_for(head, self.feed):
                continue
            restored.append(track_id)
            if self._kalman is not None:
                value = get_path(head, "trackUpdatedTimestamp")
                updated = None if value is MISSING else parse_timestamp(value)
                self._kalman.restore(track_id, head, updated or self._clock())
        self.heads.mark_known(restored)
        return len(restored)

    async def tick(self) -> None:
        """Expire idle tracks and do the head writes that are due."""
        self._tracks.expire()
        await self.heads.tick()

    async def handle(self, reports: list[JSONObject]) -> None:
        """Track a batch of reports and write the results."""
        heads, events = self.outputs(reports)
        if events:
            await self.heads.publish(heads, events)

    def outputs(self, reports: list[JSONObject]) -> tuple[list[JSONObject], list[JSONObject]]:
        """The compacted heads (newest per track) and events for a batch."""
        events, latest = self.track(reports)
        stale = format_timestamp(self._clock() + HEAD_STALE_AFTER)
        heads = [head_from_event(event, stale) for event in latest]
        if self._kalman is not None:
            set_track_geodetic(events)
            set_track_geodetic(heads)
        else:
            for head in heads:
                head.setdefault("speed", 0.0)
                head.setdefault("heading", 0.0)
        return [compact_record(head) for head in heads], [compact_record(event) for event in events]

    def track(self, reports: list[JSONObject]) -> tuple[list[JSONObject], list[JSONObject]]:
        """Return the batch's track events, and the newest event per track."""
        timed = self._timed(reports)
        events: list[JSONObject] = []
        newest: dict[str, JSONObject] = {}
        repeated = untraceable = stale = 0
        for timestamp, report in timed:
            track_id = track_id_for(report, self.feed)
            identity = _input_identity(report, track_id)
            state = self._tracks.get(track_id)
            if identity is None:
                untraceable += 1
            elif identity in state.applied:
                repeated += 1
                continue
            event = self._event(report, track_id, timestamp)
            if event is None:
                stale += 1
                continue
            if identity is not None:
                state.applied.add(identity)
            event_with_ids = dict(event)
            source_id = get_path(report, "source.uuid")
            if source_id not in (MISSING, None):
                event_with_ids["reportIds"] = [source_id]
            events.append(event_with_ids)
            newest[track_id] = event
        self._log_skips(repeated=repeated, untraceable=untraceable, stale=stale)
        return events, list(newest.values())

    def _timed(self, reports: list[JSONObject]) -> list[tuple[datetime, JSONObject]]:
        timed: list[tuple[datetime, JSONObject]] = []
        for report in reports:
            value = get_path(report, TIMESTAMP_PATH)
            timestamp = None if value is MISSING else parse_timestamp(value)
            if timestamp is not None:
                timed.append((timestamp, report))
        if len(timed) < len(reports):
            logger.warning(
                "%sDropped %d report(s) without a parseable %s",
                self._label,
                len(reports) - len(timed),
                TIMESTAMP_PATH,
            )
        timed.sort(key=lambda pair: pair[0])
        return timed

    def _event(self, report: JSONObject, track_id: str, timestamp: datetime) -> JSONObject | None:
        """The track event for one report, or ``None`` if it is stale or unusable."""
        if self._kalman is None:
            return passthrough_track_event(report, track_id)
        self._kalman.note_environment(track_id, report)
        measurement = measurement_from_report(
            report, timestamp, self._kalman.velocity_variance(track_id)
        )
        if measurement is None:
            logger.warning(
                "%sSkipping a report for track %s: its position contains NaN",
                self._label,
                track_id,
            )
            return None
        outcome, posterior = self._kalman.apply(track_id, measurement)
        if outcome is Outcome.STALE:
            return None
        return kalman_track_event(report, track_id, posterior)

    def _log_skips(self, *, repeated: int, untraceable: int, stale: int) -> None:
        if repeated:
            logger.info("%sDropped %d report(s) already applied", self._label, repeated)
        if untraceable:
            logger.debug(
                "%s%d report(s) had no source.uuid and bypassed the repeat check",
                self._label,
                untraceable,
            )
        if stale:
            logger.info(
                "%sSkipped %d report(s) older than their track's state or without a position",
                self._label,
                stale,
            )


def _input_identity(report: JSONObject, track_id: str) -> tuple[str, ...] | None:
    """The D4 identity of a report, or ``None`` without a ``source.uuid``."""
    source_id = get_path(report, "source.uuid")
    if source_id in (MISSING, None, ""):
        return None
    timestamp = get_path(report, TIMESTAMP_PATH)
    return (track_id, str(source_id), str(timestamp), input_fingerprint(report))


def _params(feed: FeedConfig) -> TrackerParams:
    fusion = (
        Fusion.COVARIANCE_INTERSECTION
        if feed.tracker.mode is TrackerMode.KALMAN_CI
        else Fusion.KALMAN
    )
    if feed.tracker.process_noise_q is None:
        return TrackerParams(fusion)
    return TrackerParams(fusion, default_process_noise=feed.tracker.process_noise_q)


class Tracker(Component):
    """Tracks every selected feed's reports into component tracks."""

    name = "tracker"

    def __init__(
        self,
        feeds: Sequence[FeedTracker],
        sources: Mapping[str, RecordSource],
        search: SearchClient,
        partition: PartitionSpec,
    ) -> None:
        self._feeds = {tracker.feed.origin_dataset: tracker for tracker in feeds}
        self._sources = sources
        self._search = search
        self._partition = partition
        self.queue_max_records = sum(tracker.feed.source_queue_max_records for tracker in feeds)

    @override
    def subscriptions(self) -> Sequence[Subscription]:
        return [
            Subscription(name, self._sources[name], TrackOwner(tracker.feed, self._partition))
            for name, tracker in self._feeds.items()
        ]

    @override
    async def prepare(self) -> None:
        preloading = {
            name: tracker for name, tracker in self._feeds.items() if not tracker.feed.preload.skip
        }
        for name in self._feeds.keys() - preloading.keys():
            logger.info("[%s] Head preload skipped; starting with empty state", name)
        limits: dict[str, int] = {}
        for tracker in preloading.values():
            dataset = tracker.heads_dataset
            limits[dataset] = max(limits.get(dataset, 0), tracker.feed.preload.tracker_limit)
        loaded = {
            dataset: await preload_heads(
                self._search, dataset, limit=limit, setting="tracker_head_preload_limit"
            )
            for dataset, limit in limits.items()
        }
        for name, tracker in preloading.items():
            newest = loaded[tracker.heads_dataset][: tracker.feed.preload.tracker_limit]
            owned = [
                head
                for head in newest
                if (track_id := record_key(head, "trackId")) is not None
                and self._partition.owns(track_id)
            ]
            logger.info("[%s] Restored %d track(s) from heads", name, tracker.restore(owned))

    @override
    async def handle(self, subscription: str, records: list[JSONObject]) -> None:
        await self._feeds[subscription].handle(records)

    @override
    async def tick(self) -> None:
        for tracker in self._feeds.values():
            await tracker.tick()

    @override
    async def close(self) -> None:
        for tracker in self._feeds.values():
            await tracker.heads.close()


@dataclass(frozen=True, slots=True)
class TrackedFeed:
    """A feed the tracker runs, with its component track datasets."""

    feed: FeedConfig
    heads: str
    events: str


def select_tracked_feeds(feeds: Sequence[FeedConfig], only: str | None) -> list[TrackedFeed]:
    """The feeds this tracker runs: tracked modes with component datasets set.

    Raises:
        ConfigError: If a selected feed's tracker mode is unrecognized, or no
            feed is tracked (or `only` names none).
    """
    selected: list[TrackedFeed] = []
    for feed in feeds:
        if only is not None and feed.origin_dataset != only:
            continue
        name, mode = feed.origin_dataset, feed.tracker.mode
        if mode is TrackerMode.UNRECOGNIZED:
            msg = f"{name}: crucible_tracker {feed.tracker.raw!r} names no tracker mode"
            raise ConfigError(msg)
        heads, events = feed.datasets.component_track_heads, feed.datasets.component_track_events
        if mode not in TRACKED_MODES:
            logger.info("[%s] Not tracked: crucible_tracker is %r", name, feed.tracker.raw)
        elif heads is None or events is None:
            logger.info("[%s] Not tracked: component track datasets are not set", name)
        else:
            selected.append(TrackedFeed(feed, heads, events))
    if not selected:
        msg = (
            f"no tracked feed is named {only!r}" if only else "the perspective has no tracked feeds"
        )
        raise ConfigError(msg)
    return selected


async def build(services: Services) -> Component:
    """Build the tracker for the feeds this process runs.

    Raises:
        ConfigError: If no feed is tracked, or `--feed` names none.
    """
    trackers: list[FeedTracker] = []
    sources: dict[str, RecordSource] = {}
    for tracked in select_tracked_feeds(services.perspective.feeds, services.settings.feed):
        feed = tracked.feed
        sync = HeadSync(
            services.writer(feed.writes),
            HeadTargets(heads=tracked.heads, events=tracked.events),
            update_interval_seconds=feed.head_update_interval_seconds,
            label=f"[{feed.origin_dataset}] ",
        )
        trackers.append(
            FeedTracker(feed, tracked.heads, sync, KeyedState(TrackState), services.clock)
        )
        sources[feed.origin_dataset] = services.source(
            feed.origin_dataset, report_query(feed.datasets.report_events, feed.origin_dataset)
        )
    return Tracker(trackers, sources, services.crucible, services.settings.partition)
