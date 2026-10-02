"""Entity management events: SUPERSEDE, DELETE and RESTORE.

The supersede map sends each managed component trackId to its final survivor,
or to ``None`` if the chain ends in a deletion. For each trackId, the most
recent event wins: SUPERSEDE points the track at ``supersededBy``, DELETE maps it
to ``None``, and RESTORE removes it. Chains are followed to their end, and a
cycle resolves to the point where it closes.
"""

import enum
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.records import MISSING, get_path
from crucible_entity_manager.core.timeutil import parse_timestamp
from crucible_entity_manager.crucible.protocols import CrucibleError, RequestError, SearchClient
from crucible_entity_manager.crucible.sql import identifier, string_literal

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS: Final = 30
DEFAULT_PAGE_SIZE: Final = 10_000
_EARLIEST: Final = datetime.min.replace(tzinfo=UTC)
_BAD_REQUEST: Final = 400


class Action(enum.StrEnum):
    """A management event's ``action``."""

    SUPERSEDE = "SUPERSEDE"
    DELETE = "DELETE"
    RESTORE = "RESTORE"


type SupersedeMap = dict[str, str | None]


def build_supersede_map(
    events: Iterable[JSONObject], initial: Mapping[str, str | None] | None = None
) -> SupersedeMap:
    """Apply management events on top of `initial` and resolve every chain.

    Events are ordered by ``crucibleHeader.updatedDate``. Among events with equal
    or missing dates, the first in input order wins. A missing, empty or NaN
    ``supersededBy`` counts as a deletion.
    """
    latest: dict[str, tuple[datetime, JSONObject]] = {}
    for event in events:
        track_id = get_path(event, "trackId")
        if track_id is MISSING or track_id is None:
            continue
        key = str(track_id)
        when = parse_timestamp(_value(event, "crucibleHeader.updatedDate")) or _EARLIEST
        if key not in latest or when > latest[key][0]:
            latest[key] = (when, event)

    links: SupersedeMap = dict(initial or {})
    for track_id, (_, event) in latest.items():
        action = get_path(event, "action")
        if action == Action.SUPERSEDE:
            links[track_id] = _target(_value(event, "supersededBy"))
        elif action == Action.DELETE:
            links[track_id] = None
        elif action == Action.RESTORE:
            links.pop(track_id, None)
    return {track_id: _resolve(track_id, links) for track_id in links}


def resolve_root(track_id: str, supersede_map: Mapping[str, str | None]) -> str | None:
    """Return the survivor that `track_id`'s events belong to.

    An unmanaged track is its own root. A track whose chain ends in a deletion
    has no root and gives ``None``. The chain is followed, so `supersede_map`
    need not be resolved; a cycle ends where it closes.
    """
    current = track_id
    visited: set[str] = set()
    while current in supersede_map and current not in visited:
        visited.add(current)
        target = supersede_map[current]
        if target is None:
            return None
        current = target
    return current


async def fetch_management_events(  # noqa: PLR0913 - one query, several independent filters
    client: SearchClient,
    dataset: str,
    *,
    actions: Iterable[Action] = tuple(Action),
    since: datetime | None = None,
    now: datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[JSONObject]:
    """Read every matching event, page by page, oldest first.

    Args:
        client: The search client.
        dataset: The management event dataset.
        actions: Which actions to read.
        since: Read events updated after this time. Without it, read the last
            `lookback_days` days.
        now: The current time, which `since` is measured against.
        lookback_days: The window used when `since` is not given.
        page_size: Rows per query.

    Returns:
        The events. A 400 response to the first page gives an empty list:
        Crucible answers queries on a dataset with no rows yet with a 400, as
        the baseline assumed. The exact body isn't known, so the status is
        what counts, and the body is logged.

    Raises:
        CrucibleError: If a query fails for any other reason. A partial result
            is never returned, because a truncated map would misroute tracks.
    """
    table = identifier(dataset)
    action_filter = " OR ".join(f"{table}.action = {string_literal(action)}" for action in actions)
    if since is None:
        window = f"TIMESTAMP_OFFSET(-{int(lookback_days)},'days')"
    else:
        age_seconds = max(1, math.ceil((now - since).total_seconds()))
        window = f"TIMESTAMP_OFFSET(-{age_seconds},'seconds')"

    events: list[JSONObject] = []
    offset = 0
    while True:
        sql = (
            f"SELECT * FROM {table} "  # noqa: S608 - names are validated, values quoted
            f"WHERE ({action_filter}) AND {table}.crucibleHeader.updatedDate > {window} "
            f"ORDER BY {table}.crucibleHeader.updatedDate ASC, {table}.crucibleHeader.uuid ASC "
            f"OFFSET {offset} ROWS FETCH NEXT {page_size} ROWS ONLY"
        )
        try:
            page = await client.search(sql, auto_backtick=False)
        except RequestError as error:
            if offset == 0 and error.status == _BAD_REQUEST:
                logger.warning(
                    "Query on management dataset %s returned 400; treating the dataset as "
                    "empty. Response: %s",
                    dataset,
                    error.body,
                )
                return []
            raise
        events.extend(page)
        if len(page) < page_size:
            return events
        offset += page_size


async def load_supersede_map(
    client: SearchClient, dataset: str, *, now: datetime, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> SupersedeMap:
    """Build the supersede map from the last `lookback_days` days of events."""
    events = await fetch_management_events(client, dataset, now=now, lookback_days=lookback_days)
    supersede_map = build_supersede_map(events)
    logger.info("Loaded %d supersede mapping(s) from %d event(s)", len(supersede_map), len(events))
    return supersede_map


@dataclass(frozen=True, slots=True)
class Protection:
    """The tracks the duplicate identifier must treat with care, by role."""

    managed: frozenset[str]
    """Superseded or deleted: the keys of the supersede map."""
    survivors: frozenset[str]
    """Tracks that others are superseded by."""
    restored: frozenset[str]
    """Restored within the cooldown."""

    @property
    def protected(self) -> frozenset[str]:
        """Tracks that must not be superseded, nor compared."""
        return self.managed | self.survivors | self.restored

    def may_survive(self, track_id: str) -> bool:
        """Whether a SUPERSEDE may point at `track_id`.

        A survivor may already be another track's survivor, but not superseded,
        deleted or just restored.
        """
        return track_id not in self.managed and track_id not in self.restored


async def duplicate_protected_ids(
    client: SearchClient,
    dataset: str,
    *,
    now: datetime,
    restore_cooldown_seconds: int,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Protection:
    """Return the tracks the duplicate identifier must treat with care.

    These are both sides of every supersede mapping, plus tracks restored within
    the cooldown. If reading recent restores fails for any reason, only that
    cooldown protection is lost, and the failure is logged. Failing to read the
    map raises.
    """
    supersede_map = await load_supersede_map(client, dataset, now=now, lookback_days=lookback_days)
    managed = frozenset(supersede_map)
    survivors = frozenset(target for target in supersede_map.values() if target is not None)
    if restore_cooldown_seconds <= 0:
        return Protection(managed, survivors, frozenset())
    try:
        restores = await fetch_management_events(
            client,
            dataset,
            actions=(Action.RESTORE,),
            since=now - timedelta(seconds=restore_cooldown_seconds),
            now=now,
        )
    except CrucibleError as error:
        logger.warning(
            "Could not read recent RESTORE events; cooldown protection skipped: %s", error
        )
        return Protection(managed, survivors, frozenset())
    restored = frozenset(
        str(track_id) for event in restores if (track_id := _value(event, "trackId")) is not None
    )
    return Protection(managed, survivors, restored)


def _value(record: JSONObject, path: str) -> JSONValue:
    value = get_path(record, path)
    return None if value is MISSING else value


def _target(value: JSONValue) -> str | None:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _resolve(track_id: str, links: Mapping[str, str | None]) -> str | None:
    """Follow `track_id`'s chain to its end; a cycle ends where it closes."""
    visited = {track_id}
    target = links.get(track_id)
    while target is not None and target in links and target not in visited:
        visited.add(target)
        target = links[target]
    return target
