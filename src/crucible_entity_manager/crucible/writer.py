"""Writing records to Crucible: chunking, retries, ordering and accounting.

Endpoint semantics (Crucible's v2 write API):

- POST (``write_batch``) upserts into an ENTITY dataset and appends to an EVENT
  dataset.
- PUT (``update_batch``) partially updates existing ENTITY records and returns the
  records that failed, without saying whether a record was missing or invalid.

Failure handling per chunk, as at ``1b534df``:

- a transient failure (no response, or 504) is retried once;
- an authorization failure fails the whole chunk, with no fallback;
- any other rejection is retried as sub-chunks of ``min(50, chunk_size // 4)`` to
  isolate the bad records. A transient failure there gives up on the remaining
  sub-chunks.

Every request first checks the `DrainBudget`. Authoritative writes are recorded
in the `WriteLedger`, so shutdown can report what was left unwritten.
"""

import asyncio
import enum
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from crucible_entity_manager.config.perspective import WriteSettings
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.identity import is_track_id
from crucible_entity_manager.core.records import MISSING, get_path
from crucible_entity_manager.crucible.protocols import (
    AuthorizationError,
    CrucibleError,
    RequestError,
    SearchClient,
    TransientError,
    WriteClient,
)
from crucible_entity_manager.crucible.sql import identifier, string_list

logger = logging.getLogger(__name__)

SUB_CHUNK_LIMIT: Final = 50
EXISTENCE_QUERY_CHUNK: Final = 200


class WriteClass(enum.Enum):
    """Whether losing a write must be reported as a failed shutdown."""

    AUTHORITATIVE = enum.auto()
    """Events, and head creates: losing them loses data."""

    BEST_EFFORT = enum.auto()
    """Updates to existing heads and association stamps: later writes supersede them."""


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """Authoritative records outstanding, and those lost while draining, per dataset.

    *Outstanding* records were submitted and their outcome isn't known yet,
    including writes interrupted by cancellation, which may still land. *Lost*
    records failed or were refused after draining began.
    """

    outstanding: MappingProxyType[str, int] = field(default_factory=lambda: MappingProxyType({}))
    lost: MappingProxyType[str, int] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def total_outstanding(self) -> int:
        """Records whose outcome is unknown."""
        return sum(self.outstanding.values())

    @property
    def total_lost(self) -> int:
        """Records that failed or were refused while draining."""
        return sum(self.lost.values())

    @property
    def clean(self) -> bool:
        """Whether nothing authoritative is outstanding or lost."""
        return not self.outstanding and not self.lost


class WriteLedger:
    """Tracks authoritative records from submission until their outcome is known.

    The current state is one immutable `LedgerSnapshot`, replaced with a single
    reference assignment. A reader in another thread therefore always sees a
    consistent snapshot. Records are added before they are submitted. Settling
    them removes them from *outstanding* and, in the same replacement, adds any
    that failed while draining to *lost*. So a snapshot can over-count unwritten
    records but never under-count them (DESIGN.md §5.7).
    """

    def __init__(self) -> None:
        self.snapshot = LedgerSnapshot()

    def add(self, dataset: str, count: int) -> None:
        """Record `count` authoritative records about to be written to `dataset`."""
        current = self.snapshot
        self.snapshot = LedgerSnapshot(_adjusted(current.outstanding, dataset, count), current.lost)

    def settle(self, dataset: str, count: int, *, lost: int = 0) -> None:
        """Record that `count` records' outcomes are known, `lost` of them lost."""
        current = self.snapshot
        self.snapshot = LedgerSnapshot(
            _adjusted(current.outstanding, dataset, -count), _adjusted(current.lost, dataset, lost)
        )


def _adjusted(
    counts: MappingProxyType[str, int], dataset: str, delta: int
) -> MappingProxyType[str, int]:
    if delta == 0:
        return counts
    updated = dict(counts)
    remaining = updated.get(dataset, 0) + delta
    if remaining:
        updated[dataset] = remaining
    else:
        updated.pop(dataset, None)
    return MappingProxyType(updated)


class DrainBudget:
    """Decides whether a request may still start.

    Unlimited until `start` is called at shutdown. After that, a request starts
    only if the remaining time covers its worst case, `request_seconds`.
    """

    def __init__(self, request_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._request_seconds = request_seconds
        self._clock = clock
        self._deadline: float | None = None

    def start(self, deadline: float) -> None:
        """Begin draining: requests must finish by `deadline`, a time on this budget's clock."""
        self._deadline = deadline

    @property
    def draining(self) -> bool:
        """Whether shutdown has started."""
        return self._deadline is not None

    def allows_request(self) -> bool:
        """Whether a request started now would finish before the deadline."""
        return self._deadline is None or self._deadline - self._clock() >= self._request_seconds


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    """Records that could not be written: rejected, timed out, or never sent."""

    failed: tuple[JSONObject, ...] = ()


@dataclass(frozen=True, slots=True)
class HeadOutcome:
    """Result of writing heads through the unknown-key path (D11), by key."""

    written: frozenset[str]
    failed: frozenset[str]


def record_key(record: JSONObject, path: str) -> str | None:
    """Return a record's key at `path`, unwrapping ``{"uuid": ...}`` values."""
    value: JSONValue | object = get_path(record, path)
    if isinstance(value, dict):
        value = value.get("uuid")
    if value is MISSING or value is None:
        return None
    return str(value)


def waves(records: Sequence[JSONObject], key: str) -> list[list[JSONObject]]:
    """Split records into waves holding at most one record per key, in order.

    Writing the waves one after another keeps each key's records in order even
    though the chunks within a wave are written concurrently. Records without a
    key go in the first wave.
    """
    result: list[list[JSONObject]] = []
    depth: dict[str, int] = {}
    for record in records:
        record_id = record_key(record, key)
        level = 0 if record_id is None else depth.get(record_id, 0)
        if record_id is not None:
            depth[record_id] = level + 1
        if level == len(result):
            result.append([])
        result[level].append(record)
    return result


class BatchWriter:
    """Writes record batches with bounded concurrency and the baseline's failure contract."""

    def __init__(
        self,
        client: WriteClient,
        search: SearchClient,
        settings: WriteSettings,
        *,
        ledger: WriteLedger,
        budget: DrainBudget,
    ) -> None:
        self._client = client
        self._search = search
        self._settings = settings
        self._ledger = ledger
        self._budget = budget
        self._slots = asyncio.Semaphore(settings.max_concurrent)

    async def post(
        self,
        dataset: str,
        records: Sequence[JSONObject],
        *,
        write_class: WriteClass,
        label: str = "",
    ) -> WriteOutcome:
        """POST records in concurrent chunks of ``chunk_size``."""
        return await self._accounted(
            dataset, records, write_class, lambda: self._post_chunks(dataset, records, label)
        )

    async def post_in_waves(
        self,
        dataset: str,
        records: Sequence[JSONObject],
        *,
        key: str,
        write_class: WriteClass,
        label: str = "",
    ) -> WriteOutcome:
        """POST records so that each key's records are written in their given order."""

        async def write() -> list[JSONObject]:
            failed: list[JSONObject] = []
            for wave in waves(records, key):
                failed.extend(await self._post_chunks(dataset, wave, label))
            return failed

        return await self._accounted(dataset, records, write_class, write)

    async def update(
        self,
        dataset: str,
        records: Sequence[JSONObject],
        *,
        key: str,
        write_class: WriteClass,
        label: str = "",
    ) -> WriteOutcome:
        """PUT partial updates in chunks of ``update_chunk_size``.

        The records the server reports as failed are matched back to the sent
        records by `key`.
        """
        return await self._accounted(
            dataset, records, write_class, lambda: self._update_chunks(dataset, records, key, label)
        )

    async def write_heads(
        self, dataset: str, heads: Sequence[JSONObject], *, key: str, label: str = ""
    ) -> HeadOutcome:
        """Write heads whose existence is unknown, without overwriting existing ones (D11).

        1. PUT every head. PUT is partial, so fields that other writers own
           (such as ``associatedPrincipalTrack``) survive.
        2. For the heads the PUT reports as failed, ask Crucible which exist.
           This fails closed: if the query fails, nothing is created.
        3. Create the heads that don't exist (POST, an upsert). Heads that do
           exist could not be updated, most likely because they failed
           validation. They are logged and reported as failed, never upserted.

        Every head is authoritative.

        Raises:
            ValueError: If a head has no `key`. Components always set one.
        """
        keyed: dict[str, JSONObject] = {}
        for head in heads:
            head_id = record_key(head, key)
            if head_id is None:
                msg = f"a head for {dataset} has no {key}"
                raise ValueError(msg)
            keyed[head_id] = head

        async def write() -> frozenset[str]:
            put_failed = _keys(
                await self._update_chunks(dataset, list(keyed.values()), key, label), key
            )
            existing, unknown = await self._which_exist(dataset, key, put_failed, label)
            if existing:
                logger.error(
                    "%s%d head(s) in %s exist but could not be updated (rejected or not sent); "
                    "they were not overwritten and will be retried: %s",
                    label,
                    len(existing),
                    dataset,
                    sorted(existing)[:5],
                )
            missing = sorted(put_failed - existing - unknown)
            created = await self._post_chunks(
                dataset, [keyed[head_id] for head_id in missing], label
            )
            return frozenset(existing | unknown | _keys(created, key))

        failed = await self._accounted_call(
            dataset, len(keyed), WriteClass.AUTHORITATIVE, write, len
        )
        return HeadOutcome(written=frozenset(keyed) - failed, failed=failed)

    async def _accounted(
        self,
        dataset: str,
        records: Sequence[JSONObject],
        write_class: WriteClass,
        write: Callable[[], Awaitable[list[JSONObject]]],
    ) -> WriteOutcome:
        failed = await self._accounted_call(dataset, len(records), write_class, write, len)
        return WriteOutcome(failed=tuple(failed))

    async def _accounted_call[T](
        self,
        dataset: str,
        count: int,
        write_class: WriteClass,
        write: Callable[[], Awaitable[T]],
        failures: Callable[[T], int],
    ) -> T:
        """Run `write`, keeping its authoritative records in the ledger until known.

        A cancelled write is left outstanding: its requests may still land, so
        shutdown reports them as uncertain.
        """
        if write_class is WriteClass.BEST_EFFORT:
            return await write()
        self._ledger.add(dataset, count)
        try:
            result = await write()
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._ledger.settle(dataset, count, lost=count if self._budget.draining else 0)
            raise
        self._ledger.settle(dataset, count, lost=failures(result) if self._budget.draining else 0)
        return result

    async def _post_chunks(
        self, dataset: str, records: Sequence[JSONObject], label: str
    ) -> list[JSONObject]:
        async def send(chunk: list[JSONObject]) -> list[JSONObject]:
            await asyncio.to_thread(self._client.write_batch, dataset, chunk)
            return []

        return await self._chunked(dataset, records, self._settings.chunk_size, send, label)

    async def _update_chunks(
        self, dataset: str, records: Sequence[JSONObject], key: str, label: str
    ) -> list[JSONObject]:
        async def send(chunk: list[JSONObject]) -> list[JSONObject]:
            reported = await asyncio.to_thread(self._client.update_batch, dataset, chunk)
            failed_ids = _keys(reported, key)
            if len(failed_ids) < len(reported) or not failed_ids <= _keys(chunk, key):
                logger.warning(
                    "%sCrucible reported failed updates to %s that don't match a sent %s; "
                    "treating the chunk as failed",
                    label,
                    dataset,
                    key,
                )
                return chunk
            return [record for record in chunk if record_key(record, key) in failed_ids]

        return await self._chunked(dataset, records, self._settings.update_chunk_size, send, label)

    async def _chunked(
        self,
        dataset: str,
        records: Sequence[JSONObject],
        chunk_size: int,
        send: Callable[[list[JSONObject]], Awaitable[list[JSONObject]]],
        label: str,
    ) -> list[JSONObject]:
        if not records:
            return []
        chunks = [
            list(records[start : start + chunk_size])
            for start in range(0, len(records), chunk_size)
        ]
        results = await asyncio.gather(
            *(self._send_chunk(dataset, chunk, chunk_size, send, label) for chunk in chunks)
        )
        failed = [record for result in results for record in result]
        if failed:
            logger.warning(
                "%s%d of %d record(s) to %s failed", label, len(failed), len(records), dataset
            )
        return failed

    async def _send_chunk(
        self,
        dataset: str,
        chunk: list[JSONObject],
        chunk_size: int,
        send: Callable[[list[JSONObject]], Awaitable[list[JSONObject]]],
        label: str,
    ) -> list[JSONObject]:
        failed = chunk
        try:
            failed = await self._request(send, chunk)
        except TransientError as error:
            logger.warning(
                "%sChunk of %d to %s timed out (%s); retrying once",
                label,
                len(chunk),
                dataset,
                error,
            )
            failed = await self._retry_once(dataset, chunk, send, label)
        except AuthorizationError as error:
            logger.warning(
                "%sChunk of %d to %s was not authorized: %s", label, len(chunk), dataset, error
            )
        except _NotStartedError:
            pass
        except RequestError as error:
            logger.warning(
                "%sChunk of %d to %s was rejected (%s); retrying in sub-chunks. Server: %s",
                label,
                len(chunk),
                dataset,
                error,
                error.body,
            )
            failed = await self._send_sub_chunks(dataset, chunk, chunk_size, send, label)
        return failed

    async def _retry_once(
        self,
        dataset: str,
        chunk: list[JSONObject],
        send: Callable[[list[JSONObject]], Awaitable[list[JSONObject]]],
        label: str,
    ) -> list[JSONObject]:
        try:
            return await self._request(send, chunk)
        except _NotStartedError:
            return chunk
        except CrucibleError as error:
            logger.warning(
                "%sChunk of %d to %s failed again: %s", label, len(chunk), dataset, error
            )
            return chunk

    async def _send_sub_chunks(
        self,
        dataset: str,
        chunk: list[JSONObject],
        chunk_size: int,
        send: Callable[[list[JSONObject]], Awaitable[list[JSONObject]]],
        label: str,
    ) -> list[JSONObject]:
        size = max(1, min(SUB_CHUNK_LIMIT, chunk_size // 4))
        failed: list[JSONObject] = []
        for start in range(0, len(chunk), size):
            sub_chunk = chunk[start : start + size]
            try:
                failed.extend(await self._request(send, sub_chunk))
            except (TransientError, _NotStartedError) as error:
                logger.warning(
                    "%sSub-chunk to %s stopped (%s); giving up on the rest", label, dataset, error
                )
                failed.extend(chunk[start:])
                break
            except CrucibleError as error:
                body = error.body if isinstance(error, RequestError) else ""
                logger.error(
                    "%s%d record(s) rejected by %s: %s. Server: %s",
                    label,
                    len(sub_chunk),
                    dataset,
                    error,
                    body,
                )
                failed.extend(sub_chunk)
        return failed

    async def _request(
        self,
        send: Callable[[list[JSONObject]], Awaitable[list[JSONObject]]],
        chunk: list[JSONObject],
    ) -> list[JSONObject]:
        """Send one request while holding a concurrency slot.

        The slot is released when the request finishes, not when the awaiting
        task is cancelled: a worker thread keeps sending after cancellation,
        and it still counts against the bound.
        """
        await self._slots.acquire()
        if not self._budget.allows_request():
            self._slots.release()
            raise _NotStartedError
        try:
            request = asyncio.ensure_future(send(chunk))
        except BaseException:
            self._slots.release()
            raise
        request.add_done_callback(self._request_finished)
        return await asyncio.shield(request)

    def _request_finished(self, request: asyncio.Future[list[JSONObject]]) -> None:
        self._slots.release()
        if not request.cancelled():
            # Retrieve the exception so an abandoned failure isn't reported as unhandled.
            request.exception()

    async def _which_exist(
        self, dataset: str, key: str, ids: Iterable[str], label: str
    ) -> tuple[set[str], set[str]]:
        """Split `ids` into those that exist and those whose existence is unknown.

        IDs that are not 32-character lowercase hex, or whose query fails, are
        unknown. Everything else that the queries don't return is missing.
        """
        valid = sorted(head_id for head_id in ids if is_track_id(head_id))
        unknown = set(ids) - set(valid)
        existing: set[str] = set()
        table, column = identifier(dataset), identifier(key)
        for start in range(0, len(valid), EXISTENCE_QUERY_CHUNK):
            batch = valid[start : start + EXISTENCE_QUERY_CHUNK]
            sql = f"SELECT {column} FROM {table} WHERE {column} IN {string_list(batch)}"  # noqa: S608 - validated
            try:
                rows = await self._search.search(sql)
            except CrucibleError as error:
                logger.warning(
                    "%sCould not check which heads exist in %s: %s", label, dataset, error
                )
                unknown.update(batch)
                continue
            existing.update(
                head_id for row in rows if (head_id := record_key(row, key)) is not None
            )
        return existing & set(valid), unknown


def _keys(records: Iterable[JSONObject], key: str) -> set[str]:
    return {record_id for record in records if (record_id := record_key(record, key)) is not None}


class _NotStartedError(Exception):
    """The drain budget did not allow the request to start."""
