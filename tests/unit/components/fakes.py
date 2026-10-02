"""In-memory stand-ins for what components are built from."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from crucible_entity_manager.config.perspective import PerspectiveConfig, WriteSettings
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.core.records import clone_record
from crucible_entity_manager.crucible.writer import (
    BatchWriter,
    DrainBudget,
    WriteLedger,
    record_key,
)
from tests.unit.runtime.fakes import FakeSource

if TYPE_CHECKING:
    from crucible_entity_manager.crucible.protocols import CrucibleError

type Responder = Callable[[str], list[JSONObject]]


def no_rows(sql: str) -> list[JSONObject]:
    del sql
    return []


class FakeCrucible:
    """An in-memory Crucible.

    Writes are recorded by dataset, and in order in `log`. A PUT fails for keys
    in `put_fails`; an existence query (``SELECT trackId FROM ... IN (...)``) finds the keys in
    `exists`; a POST to a dataset in `post_errors` raises. Other searches are
    answered by `responder`.
    """

    def __init__(self, responder: Responder = no_rows) -> None:
        self.responder = responder
        self.queries: list[str] = []
        self.posted: dict[str, list[JSONObject]] = {}
        self.updated: dict[str, list[JSONObject]] = {}
        self.log: list[tuple[str, str, list[str]]] = []
        self.put_fails: set[str] = set()
        self.exists: set[str] = set()
        self.post_errors: dict[str, CrucibleError] = {}

    async def search(self, sql: str, *, auto_backtick: bool = True) -> list[JSONObject]:
        del auto_backtick
        self.queries.append(sql)
        if sql.startswith("SELECT trackId FROM") and " IN (" in sql:
            return [{"trackId": key} for key in sorted(self.exists) if f"'{key}'" in sql]
        return self.responder(sql)

    async def access_token(self) -> str:
        return "token"

    def write_batch(self, dataset: str, records: list[JSONObject]) -> None:
        self.log.append(("post", dataset, _keys(records)))
        if dataset in self.post_errors:
            raise self.post_errors[dataset]
        self.posted.setdefault(dataset, []).extend(clone_record(record) for record in records)

    def update_batch(self, dataset: str, records: list[JSONObject]) -> list[JSONObject]:
        self.log.append(("put", dataset, _keys(records)))
        failed = [record for record in records if record_key(record, "trackId") in self.put_fails]
        self.updated.setdefault(dataset, []).extend(
            clone_record(record) for record in records if record not in failed
        )
        return failed


def _keys(records: list[JSONObject]) -> list[str]:
    return [str(record_key(record, "trackId")) for record in records]


@dataclass
class FakeServices:
    """`Services` over a `FakeCrucible`, creating `FakeSource`s on request."""

    settings: RuntimeSettings
    perspective: PerspectiveConfig
    crucible: FakeCrucible = field(default_factory=FakeCrucible)
    clock: Callable[[], datetime] = lambda: datetime(2026, 9, 30, tzinfo=UTC)
    ledger: WriteLedger = field(default_factory=WriteLedger)
    sources: dict[str, tuple[str, FakeSource]] = field(default_factory=dict)
    writes: list[WriteSettings] = field(default_factory=list)

    def writer(self, writes: WriteSettings) -> BatchWriter:
        self.writes.append(writes)
        return BatchWriter(
            self.crucible,
            self.crucible,
            writes,
            ledger=self.ledger,
            budget=DrainBudget(request_seconds=15.0),
        )

    def source(self, name: str, query: str) -> FakeSource:
        source = FakeSource(name)
        self.sources[name] = (query, source)
        return source
