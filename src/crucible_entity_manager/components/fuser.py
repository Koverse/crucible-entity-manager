"""The fuser: component tracks in, principal tracks out (DESIGN.md §5.2 to §5.5).

One fuser runs per perspective, as a single partition. It consumes every
component track event and the management events that supersede, delete and
restore tracks.

A component track belongs to the principal track of its supersede root. An
association, once made, holds until a supersede change moves it to the
survivor's principal (as at ``1b534df``). A component track that is deleted,
or whose chain ends in a deletion, is not fused.

For each component track event, in time order:

1. A component track event already fused is dropped (D4).
2. The principal's filter fuses its state (unless passthrough), and the event is
   copied into a principal track event and head, carrying the fused state and
   the principal's fused identity.
3. A new association is stamped on the component's head
   (``associatedPrincipalTrack``), best-effort and retried.

`HeadSync` writes principal heads before principal events (D8, D11).
"""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, override

from crucible_entity_manager.components.base import (
    Component,
    RecordSource,
    Services,
    Subscription,
    owns_everything,
)
from crucible_entity_manager.components.fusion import (
    FusedIdentity,
    Fusion,
    FusionParams,
    PrincipalFilter,
    PrincipalState,
    component_measurement,
    principal_record,
    state_fields,
)
from crucible_entity_manager.components.heads import HeadSync, HeadTargets, preload_heads
from crucible_entity_manager.components.keyed import (
    DEFAULT_MAX_KEYS,
    AppliedInputs,
    InputIdentity,
    KeyedState,
    input_fingerprint,
)
from crucible_entity_manager.config.perspective import ConfigError, Datasets, PerspectiveConfig
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.geodesy import set_track_geodetic
from crucible_entity_manager.core.identity import is_track_id, principal_track_id
from crucible_entity_manager.core.records import MISSING, compact_record, get_path
from crucible_entity_manager.core.timeutil import format_timestamp, parse_timestamp
from crucible_entity_manager.crucible.management import (
    Action,
    SupersedeMap,
    build_supersede_map,
    load_supersede_map,
    resolve_root,
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

COMPONENTS: Final = "components"
MANAGEMENT: Final = "management"
HEAD_STALE_AFTER: Final = timedelta(hours=10)
PENDING_STAMPS: Final = 50_000
STAMP_MAX_AGE_SECONDS: Final = 3_600.0
_LIMIT_SETTING: Final = "fusion_head_preload_limit"


@dataclass(frozen=True, slots=True)
class FuserDatasets:
    """The datasets the fuser reads and writes."""

    component_events: str
    component_heads: str | None
    principal_events: str
    principal_heads: str
    management_events: str

    @classmethod
    def of(cls, datasets: Datasets) -> "FuserDatasets":
        """Select the fuser's datasets.

        Raises:
            ConfigError: If the component track event dataset is not set.
        """
        if datasets.component_track_events is None:
            msg = "the fuser needs component_track_event_dataset"
            raise ConfigError(msg)
        return cls(
            component_events=datasets.component_track_events,
            component_heads=datasets.component_track_heads,
            principal_events=datasets.principal_track_events,
            principal_heads=datasets.principal_track_heads,
            management_events=datasets.management_events,
        )


def management_query(dataset: str) -> str:
    """The SSE query for supersede, delete and restore events, as at ``1b534df``."""
    table = identifier(dataset)
    actions = string_list([Action.SUPERSEDE, Action.DELETE, Action.RESTORE])
    return f"SELECT * FROM {table} WHERE action IN {actions}"  # noqa: S608


class AssociationStamps:
    """``associatedPrincipalTrack`` stamps on component heads.

    Stamps are partial updates (``trackId`` and ``associatedPrincipalTrack``
    only) and never create a head: a sparse create would make an incomplete
    head. A stamp that fails stays pending and is retried until the tracker has
    created the head, up to `STAMP_MAX_AGE_SECONDS`. One write is in flight at
    a time; at most `PENDING_STAMPS` wait, and the oldest go first.
    """

    def __init__(
        self,
        writer: BatchWriter,
        dataset: str,
        *,
        label: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._writer = writer
        self._dataset = dataset
        self._label = label
        self._clock = clock
        self._pending: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._flight: asyncio.Task[dict[str, tuple[str, float]]] | None = None
        self._in_flight = 0
        self._lost: OrderedDict[str, None] = OrderedDict()
        self.dropped = 0

    @property
    def pending(self) -> int:
        """Stamps not yet written."""
        return len(self._pending)

    def stamp(self, component: str, principal: str) -> None:
        """Queue `component`'s association with `principal`, replacing an older one."""
        self._lost.pop(component, None)
        self._queue(component, principal, self._clock())

    def lost(self, component: str) -> bool:
        """Whether `component`'s last stamp was dropped unwritten, so it must be stamped again."""
        return component in self._lost

    def flush(self) -> None:
        """Collect a finished write and start the next, if none is in flight."""
        self._collect()
        if self._flight is not None:
            return
        self._expire()
        if self._pending:
            batch, self._pending = dict(self._pending), OrderedDict()
            self._in_flight = len(batch)
            self._flight = asyncio.create_task(self._write(batch))

    async def close(self) -> None:
        """Write what is pending, once."""
        if self._flight is not None:
            await asyncio.wait({self._flight})
        self.flush()
        if self._flight is not None:
            await asyncio.wait({self._flight})
            self._collect()

    def _queue(self, component: str, principal: str, since: float) -> None:
        if component in self._pending:
            self._pending.move_to_end(component)
        elif len(self._pending) + self._in_flight >= PENDING_STAMPS:
            if not self._pending:
                self._drop(component)
                return
            oldest, _ = self._pending.popitem(last=False)
            self._drop(oldest)
        self._pending[component] = (principal, since)

    def _drop(self, component: str) -> None:
        """Count a stamp as dropped, and remember it so the next event stamps again."""
        self.dropped += 1
        self._lost[component] = None
        while len(self._lost) > PENDING_STAMPS:
            self._lost.popitem(last=False)

    def _expire(self) -> None:
        cutoff = self._clock() - STAMP_MAX_AGE_SECONDS
        expired = [component for component, (_, since) in self._pending.items() if since < cutoff]
        for component in expired:
            del self._pending[component]
            self._drop(component)

    def _collect(self) -> None:
        if self._flight is None or not self._flight.done():
            return
        flight, self._flight = self._flight, None
        self._in_flight = 0
        for component, (principal, since) in flight.result().items():
            if component not in self._pending:
                self._queue(component, principal, since)

    async def _write(self, batch: dict[str, tuple[str, float]]) -> dict[str, tuple[str, float]]:
        """Write `batch`; return the stamps that failed."""
        stamps: list[JSONObject] = [
            {"trackId": component, "associatedPrincipalTrack": principal}
            for component, (principal, _) in batch.items()
        ]
        outcome = await self._writer.update(
            self._dataset,
            stamps,
            key="trackId",
            write_class=WriteClass.BEST_EFFORT,
            label=self._label,
        )
        failed = {key for record in outcome.failed if (key := record_key(record, "trackId"))}
        return {component: batch[component] for component in failed if component in batch}


@dataclass(slots=True)
class _Association:
    principal: str = ""


class Associations:
    """Which principal track each component track belongs to (bounded, D2)."""

    def __init__(self, max_keys: int = DEFAULT_MAX_KEYS) -> None:
        self._principal_of: KeyedState[_Association] = KeyedState(
            _Association, max_keys=max_keys, idle_seconds=None, on_evict=self._evicted
        )
        self._members: dict[str, set[str]] = {}
        self._principal_for_root: KeyedState[_Association] = KeyedState(
            _Association, max_keys=max_keys, idle_seconds=None
        )

    def principal_of(self, component: str) -> str | None:
        """The principal `component` is associated with, if any."""
        association = self._principal_of.peek(component)
        return association.principal if association else None

    def link(self, component: str, principal: str) -> None:
        """Associate `component` with `principal`."""
        association = self._principal_of.get(component)
        if association.principal:
            self._members.get(association.principal, set()).discard(component)
        association.principal = principal
        self._members.setdefault(principal, set()).add(component)

    def associate(self, component: str, root: str) -> tuple[str, bool]:
        """`component`'s principal, associating it with its root's if it has none.

        Returns:
            The principal, and whether the association is new.
        """
        existing = self._principal_of.peek(component)
        if existing is not None and existing.principal:
            self._principal_of.get(component)
            return existing.principal, False
        principal = self.root_principal(root)
        self.set_root_principal(root, principal)
        self.link(component, principal)
        return principal, True

    def root_principal(self, root: str) -> str:
        """The principal for `root`'s group, without changing anything.

        The principal already serving the root, else the root component's own
        association (a preloaded one keeps its group), else the root's
        deterministic principal.
        """
        return self.principal_for_root(root) or self.principal_of(root) or principal_track_id(root)

    def resolve(self, component: str, root: str) -> str:
        """The principal `component` belongs to, as `associate` would choose it, read-only."""
        return self.principal_of(component) or self.root_principal(root)

    def set_root_principal(self, root: str, principal: str) -> None:
        """Record that `principal` serves `root`."""
        self._principal_for_root.get(root).principal = principal

    def principal_for_root(self, root: str) -> str | None:
        """The principal already serving `root`, if one does."""
        association = self._principal_for_root.peek(root)
        return association.principal if association else None

    def members(self, principal: str) -> set[str]:
        """The components associated with `principal`."""
        return set(self._members.get(principal, set()))

    def components(self) -> list[str]:
        """Every associated component."""
        return list(self._principal_of)

    def discard_principal(self, principal: str) -> None:
        """Forget `principal`, its components' associations and its roots."""
        for component in self._members.pop(principal, set()):
            self._principal_of.pop(component)
        for root in list(self._principal_for_root):
            if self.principal_for_root(root) == principal:
                self._principal_for_root.pop(root)

    def _evicted(self, component: str, association: _Association) -> None:
        self._members.get(association.principal, set()).discard(component)


@dataclass(frozen=True, slots=True)
class FuserSetup:
    """The fuser's configuration."""

    perspective: PerspectiveConfig
    datasets: FuserDatasets
    params: FusionParams | None
    """How to filter, or ``None`` for passthrough."""
    lookback_days: int
    clock: Callable[[], datetime]


class Fuser(Component):
    """Fuses every component track into principal tracks."""

    name = "fuser"

    def __init__(
        self,
        setup: FuserSetup,
        heads: HeadSync,
        stamps: AssociationStamps | None,
        sources: Mapping[str, RecordSource],
        search: SearchClient,
    ) -> None:
        self._setup = setup
        self._datasets = setup.datasets
        self._heads = heads
        self._stamps = stamps
        self._sources = sources
        self._search = search
        self._clock = setup.clock
        self._label = "[fuser] "
        self._principals: KeyedState[PrincipalState] = KeyedState(PrincipalState, idle_seconds=None)
        self._filter = (
            None
            if setup.params is None
            else PrincipalFilter(setup.params, self._principals, self._label)
        )
        self._identities: KeyedState[FusedIdentity] = KeyedState(FusedIdentity, idle_seconds=None)
        self._applied: KeyedState[AppliedInputs] = KeyedState(AppliedInputs)
        self._associations = Associations()
        self._supersede: SupersedeMap = {}
        self._management_connection: float | None = None
        self.queue_max_records = setup.perspective.feeds[0].source_queue_max_records

    @override
    def subscriptions(self) -> Sequence[Subscription]:
        return [
            Subscription(name, source, owns_everything) for name, source in self._sources.items()
        ]

    @override
    async def prepare(self) -> None:
        self._supersede = await self._load_supersede_map()
        preload = self._setup.perspective.preload
        if preload.skip:
            logger.info("%sHead preload skipped; starting with empty state", self._label)
            return
        limit = preload.fusion_limit
        for head in await preload_heads(
            self._search, self._datasets.principal_heads, limit=limit, setting=_LIMIT_SETTING
        ):
            self._restore_principal(head)
        if self._datasets.component_heads is not None:
            for head in await preload_heads(
                self._search, self._datasets.component_heads, limit=limit, setting=_LIMIT_SETTING
            ):
                component = record_key(head, "trackId")
                principal = record_key(head, "associatedPrincipalTrack")
                if component is not None and principal is not None:
                    self._associations.link(component, principal)
        moved = self._reassociate()
        if moved:
            logger.info(
                "%sMoved %d preloaded association(s) to their supersede root", self._label, moved
            )

    @override
    async def handle(self, subscription: str, records: list[JSONObject]) -> None:
        if subscription == MANAGEMENT:
            await self._reload_after_reconnect()
            self._apply_supersede(build_supersede_map(records, self._supersede))
            return
        await self._rehydrate(records)
        heads, events = self.outputs(records)
        if events:
            await self._heads.publish(heads, events)
        if self._stamps is not None:
            self._stamps.flush()

    @override
    async def tick(self) -> None:
        await self._reload_after_reconnect()
        self._applied.expire()
        await self._heads.tick()
        if self._stamps is not None:
            self._stamps.flush()

    @override
    async def close(self) -> None:
        await self._heads.close()
        if self._stamps is not None:
            await self._stamps.close()

    def outputs(self, components: list[JSONObject]) -> tuple[list[JSONObject], list[JSONObject]]:
        """The compacted principal heads (newest per principal) and events for a batch."""
        stale = format_timestamp(self._clock() + HEAD_STALE_AFTER)
        events: list[JSONObject] = []
        heads: dict[str, JSONObject] = {}
        deleted = repeated = 0
        for timestamp, component in self._timed(components):
            component_id = record_key(component, "trackId")
            root = None if component_id is None else resolve_root(component_id, self._supersede)
            if component_id is None or root is None:
                deleted += 1
                continue
            identity = _input_identity(component, component_id)
            if identity is not None:
                applied = self._applied.get(component_id)
                if identity in applied:
                    repeated += 1
                    continue
                applied.add(identity)
            principal, new = self._associations.associate(component_id, root)
            if self._stamps is not None and (new or self._stamps.lost(component_id)):
                self._stamps.stamp(component_id, principal)
            event, head = self._fuse(component, component_id, principal, timestamp, stale)
            events.append(event)
            heads[principal] = head
        if deleted:
            logger.info(
                "%sSkipped %d component track event(s) without a trackId or of deleted tracks",
                self._label,
                deleted,
            )
        if repeated:
            logger.info(
                "%sDropped %d component track event(s) already fused", self._label, repeated
            )
        newest = list(heads.values())
        set_track_geodetic(events)
        set_track_geodetic(newest)
        return [compact_record(head) for head in newest], [
            compact_record(event) for event in events
        ]

    def _fuse(
        self,
        component: JSONObject,
        component_id: str,
        principal: str,
        timestamp: datetime,
        stale: str,
    ) -> tuple[JSONObject, JSONObject]:
        environment = get_path(component, "environment")
        event = principal_record(component, principal, head=False, stale=stale)
        head = principal_record(component, principal, head=True, stale=stale)
        if self._filter is not None:
            self._filter.note_environment(
                principal, environment if isinstance(environment, str) else None
            )
            measurement = component_measurement(
                component, timestamp, self._filter.velocity_variance(principal)
            )
            posterior = None if measurement is None else self._filter.apply(principal, measurement)
            if posterior is not None:
                fused = state_fields(posterior)
                event.update(fused)
                head.update(fused)
        identity = self._identities.get(principal)
        identity.merge(
            component.get("identity"), superseded=self._supersede.get(component_id) is not None
        )
        for record in (event, head):
            _overlay_identity(record, identity.values())
        return event, head

    def _timed(self, components: list[JSONObject]) -> list[tuple[datetime, JSONObject]]:
        timed: list[tuple[datetime, JSONObject]] = []
        for component in components:
            value = get_path(component, "interceptTimestamp")
            timestamp = None if value is MISSING else parse_timestamp(value)
            if timestamp is not None:
                timed.append((timestamp, component))
        if len(timed) < len(components):
            logger.warning(
                "%sDropped %d component track event(s) without a parseable interceptTimestamp",
                self._label,
                len(components) - len(timed),
            )
        timed.sort(key=lambda pair: pair[0])
        return timed

    def _restore_principal(self, head: JSONObject) -> None:
        principal = record_key(head, "trackId")
        if principal is None:
            return
        self._heads.mark_known([principal])
        self._identities.get(principal).merge(head.get("identity"), superseded=False)
        if self._filter is not None:
            timestamp = _head_time(head) or self._clock()
            self._filter.restore(principal, head, timestamp)

    def _apply_supersede(self, updated: SupersedeMap) -> None:
        """Adopt a new supersede map: reset restored groups, then move associations."""
        restored = self._supersede.keys() - updated.keys()
        restored_principals = {
            principal
            for component in restored
            if (principal := self._associations.principal_of(component)) is not None
        }
        for principal in restored_principals:
            self._associations.discard_principal(principal)
            self._identities.pop(principal)
            # An empty accumulator, not a missing one: rehydrating from the stored
            # head would bring back the restored track's identity.
            self._identities.get(principal)
            if self._filter is not None:
                self._filter.forget(principal)
            self._heads.forget_update(principal)
        self._supersede = dict(updated)
        moved = self._reassociate()
        if restored_principals or moved:
            logger.info(
                "%sSupersede update: reset %d restored principal(s), moved %d association(s)",
                self._label,
                len(restored_principals),
                moved,
            )

    def _reassociate(self) -> int:
        """Move each association to its supersede root's principal; return how many moved."""
        moved = 0
        for component in self._associations.components():
            root = resolve_root(component, self._supersede)
            if root is None:
                continue
            expected = self._associations.root_principal(root)
            self._associations.set_root_principal(root, expected)
            if expected != self._associations.principal_of(component):
                self._associations.link(component, expected)
                if self._stamps is not None:
                    self._stamps.stamp(component, expected)
                moved += 1
        return moved

    async def _reload_after_reconnect(self) -> None:
        """Reload the whole supersede map when the management stream has reconnected (§5.4)."""
        source = self._sources.get(MANAGEMENT)
        connected = None if source is None else source.connected_since_monotonic
        if connected is None or connected == self._management_connection:
            return
        if self._management_connection is None:
            self._management_connection = connected
            return
        try:
            reloaded = await self._load_supersede_map()
        except CrucibleError as error:
            logger.warning(
                "%sSupersede map reload failed (%s); keeping the current map and retrying",
                self._label,
                error,
            )
            return
        self._management_connection = connected
        logger.info("%sReloaded the supersede map after a reconnect", self._label)
        self._apply_supersede(reloaded)

    async def _load_supersede_map(self) -> SupersedeMap:
        return await load_supersede_map(
            self._search,
            self._datasets.management_events,
            now=self._clock(),
            lookback_days=self._setup.lookback_days,
        )

    async def _rehydrate(self, components: list[JSONObject]) -> None:
        """Seed fused identity for principals that have none, from their stored heads (D2).

        This covers principals beyond the preload and accumulators evicted past
        the cap. A principal without a stored head starts empty, and so does
        one whose read fails.
        """
        missing: set[str] = set()
        for component in components:
            component_id = record_key(component, "trackId")
            root = None if component_id is None else resolve_root(component_id, self._supersede)
            if component_id is None or root is None:
                continue
            principal = self._associations.resolve(component_id, root)
            if self._identities.peek(principal) is None:
                missing.add(principal)
        table = identifier(self._datasets.principal_heads)
        queryable = sorted(principal for principal in missing if is_track_id(principal))
        for start in range(0, len(queryable), EXISTENCE_QUERY_CHUNK):
            chunk = string_list(queryable[start : start + EXISTENCE_QUERY_CHUNK])
            query = f"SELECT trackId, identity FROM {table} WHERE trackId IN {chunk}"  # noqa: S608
            try:
                rows = await self._search.search(query)
            except CrucibleError as error:
                logger.warning("%sIdentity rehydration failed (%s)", self._label, error)
                continue
            for row in rows:
                principal = record_key(row, "trackId")
                if principal is not None and principal in missing:
                    self._identities.get(principal).merge(row.get("identity"), superseded=False)
        for principal in missing:
            self._identities.get(principal)


def _overlay_identity(record: JSONObject, fused: Mapping[str, JSONValue]) -> None:
    if not fused:
        return
    identity = record.get("identity")
    merged: JSONObject = {}
    if isinstance(identity, dict):
        merged.update(identity)
    merged.update(fused)
    record["identity"] = merged


def _input_identity(component: JSONObject, component_id: str) -> InputIdentity | None:
    """The D4 identity of a component track event, or ``None`` without ``reportIds``."""
    report_ids = get_path(component, "reportIds")
    if report_ids in (MISSING, None, []):
        return None
    return (
        component_id,
        str(get_path(component, "interceptTimestamp")),
        str(report_ids),
        input_fingerprint(component),
    )


def _head_time(head: JSONObject) -> datetime | None:
    for path in ("interceptTimestamp", "trackUpdatedTimestamp", "crucibleHeader.createdDate"):
        value = get_path(head, path)
        if value not in (MISSING, None, ""):
            return parse_timestamp(value)
    return None


async def build(services: Services) -> Component:
    """Build the fuser for the perspective.

    Raises:
        ConfigError: If the perspective has no enabled feed (the fuser takes
            its head interval and queue bound from the first, as at
            ``1b534df``) or no component track event dataset.
    """
    perspective = services.perspective
    if not perspective.feeds:
        msg = "the fuser needs an enabled feed"
        raise ConfigError(msg)
    datasets = FuserDatasets.of(perspective.datasets)
    options = services.settings.fuser
    params = (
        None
        if options.passthrough
        else FusionParams(
            Fusion.COVARIANCE_INTERSECTION if options.covariance_intersection else Fusion.KALMAN,
            options.ci_omega,
        )
    )
    writer = services.writer(perspective.writes)
    label = "[fuser] "
    heads = HeadSync(
        writer,
        HeadTargets(heads=datasets.principal_heads, events=datasets.principal_events),
        update_interval_seconds=perspective.feeds[0].head_update_interval_seconds,
        label=label,
    )
    stamps = (
        None
        if datasets.component_heads is None
        else AssociationStamps(writer, datasets.component_heads, label=label)
    )
    sources = {
        COMPONENTS: services.source(
            COMPONENTS,
            f"SELECT * FROM {identifier(datasets.component_events)}",  # noqa: S608
        ),
        MANAGEMENT: services.source(MANAGEMENT, management_query(datasets.management_events)),
    }
    setup = FuserSetup(
        perspective=perspective,
        datasets=datasets,
        params=params,
        lookback_days=services.settings.management_lookback_days,
        clock=services.clock,
    )
    return Fuser(setup, heads, stamps, sources, services.crucible)
