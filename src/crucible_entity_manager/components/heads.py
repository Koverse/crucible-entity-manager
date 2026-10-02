"""Keeping a head dataset in step with its event stream (DESIGN.md §5.3, §5.5).

`HeadSync` owns the order of writes for one component:

1. A head for an unknown key goes through the D11 path (PUT, existence check,
   create). The key becomes known once its head is written.
2. A key's events are written only after its head exists. Until then they are
   withheld, a bounded number per key (D8), and released in order once the
   create succeeds on a later cycle.
3. A head for a known key is best-effort. The newest per key is kept and sent
   at most once per update interval, by a single background task. A key whose
   update fails becomes unknown again, so its next head takes the D11 path.

The caller must pass a head for every key it passes events for, unless the key
is already known; an event whose key has neither waits until it expires.

Every method runs on the component's owner task. The background task only
sends the records the owner selected; the owner applies its result later.
"""

import asyncio
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from crucible_entity_manager.components.keyed import DEFAULT_MAX_KEYS, KeyedState
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.protocols import CrucibleError, SearchClient
from crucible_entity_manager.crucible.sql import identifier
from crucible_entity_manager.crucible.writer import BatchWriter, WriteClass, record_key

logger = logging.getLogger(__name__)

WITHHELD_PER_KEY: Final = 100
WITHHELD_TOTAL: Final = 10_000
PENDING_UPDATES_TOTAL: Final = 50_000


@dataclass(frozen=True, slots=True)
class HeadTargets:
    """Where one component writes its heads and events, and their key field."""

    heads: str
    events: str
    key: str = "trackId"


@dataclass(slots=True)
class _Unknown:
    """A key whose head is not known to exist, with its newest head."""

    head: JSONObject | None = None


class _Withheld:
    """Events waiting for their key's head (D8).

    Each key keeps its events in order, at most `per_key` of them, and all keys
    together at most `total`. Past either bound the oldest event goes: the
    key's own at `per_key`, the oldest of all at `total`. Arrival order is kept
    in one queue whose entries for events that have already left are skipped,
    and the queue is rebuilt when such entries outnumber the live ones.
    """

    def __init__(self, per_key: int, total: int) -> None:
        self._per_key = per_key
        self._total_limit = total
        self._by_key: dict[str, deque[tuple[int, JSONObject]]] = {}
        self._arrivals: deque[tuple[int, str]] = deque()
        self._sequence = 0
        self.total = 0
        self.dropped = 0

    def add(self, key: str, event: JSONObject) -> None:
        held = self._by_key.setdefault(key, deque())
        if len(held) >= self._per_key:
            held.popleft()
            self.total -= 1
            self.dropped += 1
        elif self.total >= self._total_limit:
            self._drop_oldest()
            held = self._by_key.setdefault(key, deque())
        held.append((self._sequence, event))
        self._arrivals.append((self._sequence, key))
        self._sequence += 1
        self.total += 1
        if len(self._arrivals) > 2 * self.total + 64:
            self._arrivals = deque(
                sorted(
                    (number, owner) for owner, queue in self._by_key.items() for number, _ in queue
                )
            )

    def release(self, key: str) -> list[JSONObject]:
        """Remove and return `key`'s events, in order."""
        held = self._by_key.pop(key, None)
        if not held:
            return []
        self.total -= len(held)
        return [event for _, event in held]

    def discard(self, key: str) -> None:
        """Drop `key`'s events, counting them as lost."""
        lost = len(self.release(key))
        self.dropped += lost

    def _drop_oldest(self) -> None:
        """Drop the oldest held event. Called only at the total limit, so one exists."""
        while True:
            number, key = self._arrivals.popleft()
            held = self._by_key.get(key)
            if held and held[0][0] == number:
                held.popleft()
                if not held:
                    del self._by_key[key]
                self.total -= 1
                self.dropped += 1
                return


class HeadSync:
    """Orders head and event writes for one component (see the module docstring)."""

    def __init__(  # noqa: PLR0913 - targets, limits and collaborators
        self,
        writer: BatchWriter,
        targets: HeadTargets,
        *,
        update_interval_seconds: float,
        label: str,
        max_keys: int = DEFAULT_MAX_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._writer = writer
        self._targets = targets
        self._interval = update_interval_seconds
        self._label = label
        self._clock = clock
        self._known: KeyedState[list[float]] = KeyedState(
            lambda: [-float("inf")], max_keys=max_keys, clock=clock
        )
        """Known keys, each with the time its head was last written."""
        self._unknown: KeyedState[_Unknown] = KeyedState(
            _Unknown, max_keys=max_keys, on_evict=self._unknown_evicted, clock=clock
        )
        self._updates: OrderedDict[str, JSONObject] = OrderedDict()
        self._withheld = _Withheld(WITHHELD_PER_KEY, WITHHELD_TOTAL)
        self._dropped_updates = 0
        self._flush: asyncio.Task[frozenset[str]] | None = None

    @property
    def withheld(self) -> int:
        """Events waiting for their key's head."""
        return self._withheld.total

    @property
    def pending_updates(self) -> int:
        """Best-effort head updates not yet sent."""
        return len(self._updates)

    def mark_known(self, keys: Iterable[str]) -> None:
        """Record that heads exist for `keys`, as found by a preload."""
        for key in keys:
            self._known.get(key)

    def forget_update(self, key: str) -> None:
        """Drop `key`'s pending best-effort update, if any."""
        self._updates.pop(key, None)

    def is_known(self, key: str) -> bool:
        """Whether a head is known to exist for `key`."""
        return key in self._known

    async def publish(self, heads: Sequence[JSONObject], events: Sequence[JSONObject]) -> None:
        """Write `heads` (the newest per key), then `events` in order per key."""
        self._collect_flush()
        for head in heads:
            key = self._key(head)
            if key is None:
                continue
            if key in self._known:
                self._stage_update(key, head)
            else:
                self._unknown.get(key).head = head
        ready = await self._create_unknown_heads()
        for event in events:
            key = self._key(event)
            if key is None:
                continue
            if key in self._known:
                self._known.get(key)
                ready.append(event)
            else:
                self._withhold(key, event)
        await self._write_events(ready)
        self._start_flush()

    async def tick(self) -> None:
        """Retry creates and release their events, expire idle keys, send due updates."""
        self._collect_flush()
        self._known.expire()
        self._unknown.expire()
        await self._write_events(await self._create_unknown_heads())
        self._start_flush()
        self._report_drops()

    async def close(self) -> None:
        """Write what can still be written, including updates not yet due."""
        await self.tick()
        if self._flush is not None:
            await asyncio.wait({self._flush})
            self._collect_flush()
        await self._send_updates(self._take_updates(due_only=False))
        self._report_drops()

    def _stage_update(self, key: str, head: JSONObject) -> None:
        self._known.get(key)
        if key in self._updates:
            self._updates.move_to_end(key)
        elif len(self._updates) >= PENDING_UPDATES_TOTAL:
            self._updates.popitem(last=False)
            self._dropped_updates += 1
        self._updates[key] = head

    async def _create_unknown_heads(self) -> list[JSONObject]:
        """Create the pending heads; return the events released by the creates."""
        pending = {
            key: state
            for key in self._unknown
            if (state := self._unknown.peek(key)) is not None and state.head is not None
        }
        if not pending:
            return []
        outcome = await self._writer.write_heads(
            self._targets.heads,
            [state.head for state in pending.values() if state.head is not None],
            key=self._targets.key,
            label=self._label,
        )
        released: list[JSONObject] = []
        now = self._clock()
        for key in outcome.written:
            self._unknown.pop(key)
            self._known.get(key)[0] = now
            released.extend(self._withheld.release(key))
        return released

    def _withhold(self, key: str, event: JSONObject) -> None:
        self._unknown.get(key)
        self._withheld.add(key, event)

    async def _write_events(self, events: list[JSONObject]) -> None:
        if events:
            await self._writer.post_in_waves(
                self._targets.events,
                events,
                key=self._targets.key,
                write_class=WriteClass.AUTHORITATIVE,
                label=self._label,
            )

    def _start_flush(self) -> None:
        if self._flush is None:
            due = self._take_updates(due_only=True)
            if due:
                self._flush = asyncio.create_task(self._send_updates(due))

    def _take_updates(self, *, due_only: bool) -> list[JSONObject]:
        """Remove and return the pending updates to send now."""
        now = self._clock()
        due: list[JSONObject] = []
        for key in list(self._updates):
            last = self._known.peek(key)
            if due_only and last is not None and now - last[0] < self._interval:
                continue
            due.append(self._updates.pop(key))
            if last is not None:
                last[0] = now
        return due

    async def _send_updates(self, heads: list[JSONObject]) -> frozenset[str]:
        """Send best-effort updates; return the keys whose update failed."""
        if not heads:
            return frozenset[str]()
        outcome = await self._writer.update(
            self._targets.heads,
            heads,
            key=self._targets.key,
            write_class=WriteClass.BEST_EFFORT,
            label=self._label,
        )
        return frozenset(key for record in outcome.failed if (key := self._key(record)) is not None)

    def _collect_flush(self) -> None:
        """Apply a finished flush: keys whose update failed become unknown."""
        if self._flush is None or not self._flush.done():
            return
        flush, self._flush = self._flush, None
        for key in flush.result():
            self._known.pop(key)

    def _key(self, record: JSONObject) -> str | None:
        return record_key(record, self._targets.key)

    def _unknown_evicted(self, key: str, state: _Unknown) -> None:
        del state
        self._withheld.discard(key)

    def _report_drops(self) -> None:
        if self._withheld.dropped:
            logger.warning(
                "%sDropped %d withheld event(s) at the buffer limits or with an idle key",
                self._label,
                self._withheld.dropped,
            )
            self._withheld.dropped = 0
        if self._dropped_updates:
            logger.warning(
                "%sDropped %d pending best-effort head update(s) at the limit",
                self._label,
                self._dropped_updates,
            )
            self._dropped_updates = 0


async def preload_heads(
    search: SearchClient, dataset: str, *, limit: int, setting: str
) -> list[JSONObject]:
    """The newest `limit` heads of `dataset`, or none if the query fails (best effort, §5.4).

    `setting` names the configuration key that raises the limit, for the
    warning logged when the limit is reached.
    """
    table = identifier(dataset)
    query = f"SELECT * FROM {table} ORDER BY {table}.crucibleHeader.updatedDate DESC LIMIT {limit}"  # noqa: S608
    try:
        heads = await search.search(query)
    except CrucibleError as error:
        logger.warning(
            "Head preload from %s failed (%s); starting with empty state", dataset, error
        )
        return []
    if len(heads) >= limit:
        logger.warning(
            "Head preload from %s hit its %d-row limit; older heads were not loaded. "
            "Raise %s or CRUCIBLE_HEAD_PRELOAD_LIMIT.",
            dataset,
            limit,
            setting,
        )
    return heads
