"""The duplicate identifier: SUPERSEDE events for duplicate component tracks (DESIGN.md §5.5).

One identifier runs per perspective, as a single partition, polling rather
than streaming. Every poll interval it:

1. Reads the protected track IDs: both sides of every supersede mapping, and
   tracks restored within the cooldown. They are left out of detection.
2. Reads the component track events of the last `time_window_hours` and finds
   duplicate pairs (`detection`), in a worker thread.
3. For each pair, most confident first, keeps the track created first, judged
   from the two tracks' own heads, and writes a SUPERSEDE event for the other.
   The protected IDs are read again immediately before each write, which
   narrows the race with an operator's RESTORE or another detector.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, override

from crucible_entity_manager.components.base import Component, Services, Subscription
from crucible_entity_manager.components.detection import (
    DetectionParams,
    Duplicate,
    build_histories,
    choose_survivor,
    creation_time,
    find_duplicates,
)
from crucible_entity_manager.config.runtime import DuplicateOptions
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.core.identity import is_track_id
from crucible_entity_manager.core.timeutil import format_timestamp
from crucible_entity_manager.crucible.management import (
    Action,
    Protection,
    duplicate_protected_ids,
)
from crucible_entity_manager.crucible.protocols import CrucibleError, SearchClient
from crucible_entity_manager.crucible.sql import identifier, string_list
from crucible_entity_manager.crucible.writer import (
    EXISTENCE_QUERY_CHUNK,
    BatchWriter,
    WriteClass,
    record_key,
)

logger = logging.getLogger(__name__)

EVENT_LIMIT: Final = 50_000
SOURCE: Final = "entity_duplicate_identifier"
REASON_LENGTH: Final = 100


@dataclass(frozen=True, slots=True)
class DuplicateTargets:
    """The datasets the identifier reads and writes."""

    events: str
    heads: str
    management: str


@dataclass(frozen=True, slots=True)
class IdentifierSetup:
    """The identifier's settings."""

    targets: DuplicateTargets
    options: DuplicateOptions
    restore_cooldown_seconds: int
    lookback_days: int
    clock: Callable[[], datetime]


def events_query(dataset: str, since: datetime) -> str:
    """Recent component track events with a position, newest first, as at ``1b534df``."""
    table = identifier(dataset)
    return (
        f"SELECT * FROM {table} WHERE {table}.`ecefPosition`.x IS NOT NULL "  # noqa: S608
        f"AND {table}.crucibleHeader.createdDate > '{format_timestamp(since)}' "
        f"ORDER BY {table}.interceptTimestamp DESC LIMIT {EVENT_LIMIT}"
    )


def supersede_event(superseded: str, survivor: str, duplicate: Duplicate) -> JSONObject:
    """The management event that supersedes `superseded` by `survivor`."""
    reason = (
        f"Duplicate entity detected: avg_mahalanobis={duplicate.mean_mahalanobis:.1f}sigma, "
        f"avg_vel_diff={duplicate.mean_velocity_difference:.1f}m/s, "
        f"matching_points={duplicate.matching_points}, "
        f"confidence={duplicate.confidence:.2f}"
    )
    source: JSONObject = {"datasetName": SOURCE, "uuid": "0" * 32}
    return {
        "action": str(Action.SUPERSEDE),
        "trackId": superseded,
        "supersededBy": survivor,
        "source": dict(source),
        "sourceRaw": dict(source),
        "mechanism": "TRACKER",
        "status": "COMPLETED",
        "edhControlSet": ["CLS:U"],
        "reason": reason[:REASON_LENGTH],
    }


class DuplicateIdentifier(Component):
    """Finds duplicate component tracks and supersedes the later of each pair."""

    name = "duplicates"
    queue_max_records = 1

    def __init__(
        self,
        setup: IdentifierSetup,
        writer: BatchWriter,
        search: SearchClient,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._setup = setup
        self._writer = writer
        self._search = search
        self._monotonic = monotonic
        options = setup.options
        self._fallback = DetectionParams(
            candidate_search_radius_m=options.candidate_search_radius_m,
            distance_threshold_sigma=options.mahalanobis_threshold_sigma,
            velocity_threshold_mps=options.velocity_threshold_mps,
            time_alignment_seconds=options.time_alignment_seconds,
            min_matching_points=options.min_matching_points,
            min_confidence=options.min_confidence,
        )
        self._next_poll: float | None = None
        self._label = "[duplicates] "

    @override
    def subscriptions(self) -> Sequence[Subscription]:
        return ()

    @override
    async def prepare(self) -> None:
        return

    @override
    async def handle(self, subscription: str, records: list[JSONObject]) -> None:
        msg = f"the duplicate identifier has no subscriptions, got {subscription}"
        raise RuntimeError(msg)

    @override
    async def tick(self) -> None:
        now = self._monotonic()
        if self._next_poll is not None and now < self._next_poll:
            return
        try:
            written = await self.poll()
        except CrucibleError as error:
            logger.warning("%sPoll skipped: %s", self._label, error)
            return
        finally:
            # Measured from the end of the poll, as the baseline slept after each search.
            self._next_poll = self._monotonic() + self._setup.options.poll_interval_seconds
        logger.info("%sPoll complete: %d SUPERSEDE event(s) written", self._label, written)

    @override
    async def close(self) -> None:
        return

    async def poll(self) -> int:
        """Detect duplicates once and write their SUPERSEDE events; return how many.

        Raises:
            CrucibleError: If the protected IDs or the track events can't be read.
        """
        protection = await self._protection()
        since = self._setup.clock() - timedelta(hours=self._setup.options.time_window_hours)
        events = await self._search.search(
            events_query(self._setup.targets.events, since), auto_backtick=False
        )
        histories = await asyncio.to_thread(build_histories, events, set(protection.protected))
        duplicates = await asyncio.to_thread(find_duplicates, histories, self._fallback)
        logger.info(
            "%sCompared %d track(s) from %d event(s); %d duplicate pair(s)",
            self._label,
            len(histories),
            len(events),
            len(duplicates),
        )
        if not duplicates:
            return 0
        created = await self._creation_times(duplicates)
        # This poll's writes, which a search may not show yet.
        superseded_here: set[str] = set()
        survivors_here: set[str] = set()
        for duplicate in duplicates:
            superseded, survivor = choose_survivor(
                duplicate.track_id_1, duplicate.track_id_2, created.get
            )
            if superseded in protection.protected | superseded_here | survivors_here:
                continue
            protection = await self._protection()
            if superseded in protection.protected | superseded_here | survivors_here:
                logger.info("%sSkipped %s: it became protected", self._label, superseded)
                continue
            if not protection.may_survive(survivor) or survivor in superseded_here:
                logger.info(
                    "%sSkipped %s: its survivor %s was superseded, deleted or restored",
                    self._label,
                    superseded,
                    survivor,
                )
                continue
            if await self._supersede(superseded, survivor, duplicate):
                superseded_here.add(superseded)
                survivors_here.add(survivor)
        return len(superseded_here)

    async def _protection(self) -> Protection:
        return await duplicate_protected_ids(
            self._search,
            self._setup.targets.management,
            now=self._setup.clock(),
            restore_cooldown_seconds=self._setup.restore_cooldown_seconds,
            lookback_days=self._setup.lookback_days,
        )

    async def _creation_times(self, duplicates: list[Duplicate]) -> dict[str, datetime]:
        """The creation time of each track in `duplicates`, from its own head."""
        ids = sorted(
            {
                track
                for duplicate in duplicates
                for track in (duplicate.track_id_1, duplicate.track_id_2)
            }
        )
        queryable = [track for track in ids if is_track_id(track)]
        table = identifier(self._setup.targets.heads)
        created: dict[str, datetime] = {}
        for start in range(0, len(queryable), EXISTENCE_QUERY_CHUNK):
            chunk = string_list(queryable[start : start + EXISTENCE_QUERY_CHUNK])
            heads = await self._search.search(f"SELECT * FROM {table} WHERE trackId IN {chunk}")  # noqa: S608
            for head in heads:
                track = record_key(head, "trackId")
                when = creation_time(head)
                if track is not None and when is not None:
                    created[track] = when
        return created

    async def _supersede(self, superseded: str, survivor: str, duplicate: Duplicate) -> bool:
        event = supersede_event(superseded, survivor, duplicate)
        outcome = await self._writer.post(
            self._setup.targets.management,
            [event],
            write_class=WriteClass.AUTHORITATIVE,
            label=self._label,
        )
        if outcome.failed:
            return False
        logger.info(
            "%sSuperseded %s by %s (confidence %.2f)",
            self._label,
            superseded,
            survivor,
            duplicate.confidence,
        )
        return True


async def build(services: Services) -> Component:
    """Build the duplicate identifier for the perspective."""
    datasets = services.perspective.datasets
    targets = DuplicateTargets(
        events=datasets.component_track_events or datasets.principal_track_events,
        heads=datasets.component_track_heads or datasets.principal_track_heads,
        management=datasets.management_events,
    )
    setup = IdentifierSetup(
        targets=targets,
        options=services.settings.duplicates,
        restore_cooldown_seconds=services.perspective.restore_cooldown_seconds,
        lookback_days=services.settings.management_lookback_days,
        clock=services.clock,
    )
    return DuplicateIdentifier(
        setup, services.writer(services.perspective.writes), services.crucible
    )
