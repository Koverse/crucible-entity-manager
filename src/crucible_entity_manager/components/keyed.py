"""Per-key state shared by the tracker and the fuser (DESIGN.md §5.3, D2, D4).

`KeyedState` bounds per-key state by idle time and an LRU cap. `AppliedInputs`
remembers the inputs a key has already applied, so a repeated input is dropped.
"""

import hashlib
import json
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from typing import Final

from crucible_entity_manager.core.aliases import JSONObject, JSONValue

DEFAULT_MAX_KEYS: Final = 200_000
"""LRU cap on keys per kind of state (``max_tracked_keys``)."""

DEFAULT_IDLE_SECONDS: Final = 30 * 60.0
"""Idle expiry: twice the 15-minute reset horizon, so nothing useful is lost."""

APPLIED_INPUTS_PER_KEY: Final = 64

type InputIdentity = tuple[JSONValue, ...]


class KeyedState[V]:
    """A map from key to state, bounded by idle time and a least-recently-used cap.

    Reading or writing a key marks it used. Keys idle longer than
    `idle_seconds` are removed by `expire`, and inserting past `max_keys`
    removes the least recently used key. Every removal is reported to
    `on_evict`, so callers can count what was lost.
    """

    def __init__(
        self,
        factory: Callable[[], V],
        *,
        max_keys: int = DEFAULT_MAX_KEYS,
        idle_seconds: float | None = DEFAULT_IDLE_SECONDS,
        on_evict: Callable[[str, V], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_keys < 1:
            msg = f"max_keys must be at least 1, got {max_keys}"
            raise ValueError(msg)
        self._factory = factory
        self._max_keys = max_keys
        self._idle_seconds = idle_seconds
        self._on_evict = on_evict
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, V]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def peek(self, key: str) -> V | None:
        """Return `key`'s state without marking it used."""
        entry = self._entries.get(key)
        return entry[1] if entry else None

    def get(self, key: str) -> V:
        """Return `key`'s state, creating it if absent, and mark it used."""
        entry = self._entries.pop(key, None)
        value = entry[1] if entry else self._factory()
        self._entries[key] = (self._clock(), value)
        if entry is None:
            while len(self._entries) > self._max_keys:
                evicted, (_, state) = self._entries.popitem(last=False)
                self._evicted(evicted, state)
        return value

    def pop(self, key: str) -> V | None:
        """Remove `key` without reporting it as evicted."""
        entry = self._entries.pop(key, None)
        return entry[1] if entry else None

    def expire(self) -> int:
        """Remove keys idle longer than `idle_seconds`; return how many."""
        if self._idle_seconds is None:
            return 0
        cutoff = self._clock() - self._idle_seconds
        expired = 0
        while self._entries:
            key, (used, state) = next(iter(self._entries.items()))
            if used > cutoff:
                break
            del self._entries[key]
            self._evicted(key, state)
            expired += 1
        return expired

    def _evicted(self, key: str, state: V) -> None:
        if self._on_evict is not None:
            self._on_evict(key, state)


def input_fingerprint(record: JSONObject) -> str:
    """SHA-256 of `record` without ``crucibleHeader``, as canonical JSON (D4).

    Keys are sorted, separators are compact, and floats use Python's shortest
    round-trip form, so equal content always gives the same fingerprint.
    """
    content = {key: value for key, value in record.items() if key != "crucibleHeader"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class AppliedInputs:
    """The identities of the last inputs a key applied, newest last (D4)."""

    __slots__ = ("_order", "_seen")

    def __init__(self) -> None:
        self._seen: set[InputIdentity] = set()
        self._order: deque[InputIdentity] = deque()

    def __contains__(self, identity: InputIdentity) -> bool:
        return identity in self._seen

    def add(self, identity: InputIdentity) -> None:
        """Remember `identity`, forgetting the oldest beyond the window."""
        if identity in self._seen:
            return
        self._seen.add(identity)
        self._order.append(identity)
        if len(self._order) > APPLIED_INPUTS_PER_KEY:
            self._seen.discard(self._order.popleft())
