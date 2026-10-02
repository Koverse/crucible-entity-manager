"""What the runtime needs from a component."""

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from crucible_entity_manager.config.perspective import PerspectiveConfig, WriteSettings
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.protocols import CrucibleService
from crucible_entity_manager.crucible.writer import BatchWriter


class RecordSource(Protocol):
    """A stream of record batches with the signals the health check reads.

    `crucible.sse.SseSource` is the production implementation.
    """

    name: str
    last_activity_monotonic: float | None
    connected_since_monotonic: float | None

    @property
    def watchdog_seconds(self) -> float:
        """How long the source may be silent before it reconnects; 0 means no limit."""
        ...

    def __aiter__(self) -> AsyncIterator[list[JSONObject]]:
        """Yield record batches until cancelled."""
        ...


@dataclass(frozen=True, slots=True)
class Subscription:
    """A source a component consumes, and which of its records this partition owns.

    `owns` runs on the source's task, not the owner task, so it must depend
    only on the record (normally a stable hash of its key).
    """

    name: str
    source: RecordSource
    owns: Callable[[JSONObject], bool]


class Component(Protocol):
    """One pipeline component.

    Calls never overlap. `prepare` runs first; then `handle`, `tick` and
    `close` run on the single owner task.
    """

    name: str
    queue_max_records: int
    """Bound on records waiting between the sources and `handle` (decision D1)."""

    def subscriptions(self) -> Sequence[Subscription]:
        """The sources to consume. Called once, before `prepare`."""
        ...

    async def prepare(self) -> None:
        """Rebuild state from Crucible before any records are handled."""
        ...

    async def handle(self, subscription: str, records: list[JSONObject]) -> None:
        """Process owned records from `subscription`, in arrival order."""
        ...

    async def tick(self) -> None:
        """Do timed work: retries, expiry, polling. Runs even when no input arrives."""
        ...

    async def close(self) -> None:
        """Flush what can still be written. Called once, after the last `handle`."""
        ...


class Services(Protocol):
    """What a component is built from. `runtime.context.Context` provides it."""

    @property
    def settings(self) -> RuntimeSettings:
        """This process's command-line settings."""
        ...

    @property
    def perspective(self) -> PerspectiveConfig:
        """The perspective's validated configuration."""
        ...

    @property
    def crucible(self) -> CrucibleService:
        """Crucible searches and writes."""
        ...

    @property
    def clock(self) -> Callable[[], datetime]:
        """The current UTC time."""
        ...

    def writer(self, writes: WriteSettings) -> BatchWriter:
        """A writer with `writes` limits that shares the process's ledger and budget."""
        ...

    def source(self, name: str, query: str) -> RecordSource:
        """A subscription to the records `query` selects."""
        ...


def owns_everything(record: JSONObject) -> bool:
    """`Subscription.owns` for a source that one partition consumes entirely."""
    del record
    return True
