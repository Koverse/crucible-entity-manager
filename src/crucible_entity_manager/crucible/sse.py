"""Server-sent events from Crucible's streaming search.

`SseSource` subscribes to a SQL query and yields each event's records,
reconnecting forever with backoff. Crucible SSE has no replay: a reconnect
resumes at the current moment. So every outage is a data gap, and each one is
logged with its window (DESIGN.md, decision D9).

The source parses SSE framing itself, over aiohttp. ``aiohttp-sse-client``
(used at ``1b534df``) reconnects internally with the original headers, which
would bypass token refresh and gap accounting here.
"""

import asyncio
import json
import logging
import math
import random
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Self

import aiohttp

from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.core.records import parse_records
from crucible_entity_manager.core.timeutil import format_timestamp, utc_now

logger = logging.getLogger(__name__)

SSE_PATH: Final = "/api/v1/read/search/sse"
PERSISTENT_FAILURE_ATTEMPTS: Final = 10
"""After every this many consecutive failures, log at CRITICAL."""

MIN_RECONNECT_SECONDS: Final = 1.0
"""Wait before reconnecting after the server ends a stream cleanly."""

_TRUE_TEXT: Final = frozenset({"true", "1", "yes", "on"})
_ERROR_BODY_BYTES: Final = 500

type SseItem = str | None
"""An event's data, or ``None`` for a keep-alive comment (activity without data)."""

type Connect = Callable[[Mapping[str, str]], AbstractAsyncContextManager[AsyncIterator[SseItem]]]
"""Open one subscription with the given headers."""


class SseResponseError(Exception):
    """The SSE endpoint answered with something other than an event stream."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"SSE subscription failed with HTTP {status}: {detail}")
        self.status = status


@dataclass(frozen=True, slots=True)
class SseSettings:
    """Connection settings for Crucible SSE subscriptions."""

    url: str
    read_buffer_bytes: int = 256 * 1024 * 1024
    """aiohttp's read buffer. Lines up to twice this size are accepted; a single
    event holding a snapshot of many tracks can be large."""

    read_timeout_seconds: float = 180.0
    """Reconnect after this long without an event or keep-alive; 0 disables it."""

    connect_timeout_seconds: float = 10.0
    """Limit on connecting and receiving the response headers."""
    verify_tls: bool = False
    max_backoff_seconds: float = 60.0

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> Self:
        """Read the settings the baseline read from the environment.

        ``CRUCIBLE_SSE_URL`` overrides the endpoint built from
        ``CRUCIBLE_SERVICES_HOST``. ``CRUCIBLE_SSE_READ_BUFSIZE``,
        ``CRUCIBLE_SSE_READ_TIMEOUT`` and ``CRUCIBLE_SSL_VERIFY`` tune it.

        Raises:
            KeyError: If neither ``CRUCIBLE_SSE_URL`` nor
                ``CRUCIBLE_SERVICES_HOST`` is set.
            ValueError: If a numeric setting is not a number.
        """
        url = (
            environ.get("CRUCIBLE_SSE_URL")
            or f"https://{environ['CRUCIBLE_SERVICES_HOST']}{SSE_PATH}"
        )
        defaults = cls(url=url)
        return cls(
            url=url,
            read_buffer_bytes=int(
                environ.get("CRUCIBLE_SSE_READ_BUFSIZE", defaults.read_buffer_bytes)
            ),
            read_timeout_seconds=float(
                environ.get("CRUCIBLE_SSE_READ_TIMEOUT", defaults.read_timeout_seconds)
            ),
            verify_tls=environ.get("CRUCIBLE_SSL_VERIFY", "false").strip().lower() in _TRUE_TEXT,
        )


async def parse_events(lines: AsyncIterable[bytes]) -> AsyncIterator[SseItem]:
    """Yield each event's data, and ``None`` for each keep-alive comment.

    Follows the SSE framing rules: a leading byte-order mark is skipped; ``data``
    lines accumulate and are joined with newlines; a blank line dispatches the
    event; other fields are ignored. Lines must end in LF or CRLF; bare CR line
    endings are not supported.
    """
    data: list[str] = []
    first = True
    async for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if first:
            line = line.removeprefix("\ufeff")
            first = False
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            yield None
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value.removeprefix(" "))


class SseSource:
    """One SSE subscription that reconnects until cancelled.

    Iterate it (``async for records in source``) to receive each event's records.
    """

    def __init__(  # noqa: PLR0913 - collaborators are injected for testing
        self,
        name: str,
        query: str,
        settings: SseSettings,
        token: Callable[[], Awaitable[str]],
        *,
        connect: Connect | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        """Prepare a subscription; nothing connects until iteration starts.

        Args:
            name: Label for logs, such as the origin dataset.
            query: The SQL to stream.
            settings: Connection settings.
            token: Returns a fresh access token; called before every connect.
            connect: Opens a subscription. Defaults to an aiohttp connection.
            sleep: Waits between reconnects.
            clock: The wall clock used in gap reports.
        """
        self.name = name
        self.query = query
        self._settings = settings
        self._token = token
        self._connect = connect or self._aiohttp_connect
        self._sleep = sleep
        self._clock = clock
        self._random = random.Random()  # noqa: S311 - jitter, not security
        self._last_seen_at: datetime | None = None
        self._gap_start: datetime | None = None
        """Start of the current outage; ``None`` until a first subscription is lost."""
        self.last_activity_monotonic: float | None = None
        """When the stream last delivered an event or keep-alive, for the health check."""
        self.connected_since_monotonic: float | None = None
        """When the current subscription opened, or ``None`` while disconnected."""

    @property
    def watchdog_seconds(self) -> float:
        """How long the stream may be silent before it reconnects; 0 means no limit."""
        return self._settings.read_timeout_seconds

    def __aiter__(self) -> AsyncIterator[list[JSONObject]]:
        return self.batches()

    async def batches(self) -> AsyncGenerator[list[JSONObject]]:
        """Yield the records of every event, forever.

        A gap runs from the last activity on a lost connection (or from its
        opening, if it saw none) to the next successful subscription, and is
        logged once, when that subscription succeeds. Gap state belongs to the
        source, so it survives the consumer stopping and iterating again.
        """
        failures = 0
        while True:
            connected = False
            try:
                headers = {
                    "Authorization": f"Bearer {await self._token()}",
                    "Content-Type": "text/plain",
                }
                async with self._connect(headers) as items:
                    connected = True
                    self._on_subscribed()
                    async for item in self._watched(items):
                        self._on_activity()
                        records = self._decode(item) if item is not None else []
                        if records:
                            failures = 0
                            yield records
                logger.warning("[%s] SSE stream ended; reconnecting", self.name)
            except _WatchdogExpiredError:
                failures += 1
                logger.warning(
                    "[%s] No SSE activity for %.0fs; reconnecting",
                    self.name,
                    self._settings.read_timeout_seconds,
                )
            except Exception as error:  # any failure means reconnect
                failures += 1
                self._report_failure(error)
            finally:
                if connected:
                    self._on_disconnected()
            await self._sleep(self._delay(failures))

    def _on_subscribed(self) -> None:
        now = self._clock()
        if self._gap_start is None:
            logger.info("[%s] SSE subscribed at %s", self.name, format_timestamp(now))
        else:
            logger.warning(
                "[%s] SSE gap: no data from %s to %s (%.1fs); events in that window are lost",
                self.name,
                format_timestamp(self._gap_start),
                format_timestamp(now),
                (now - self._gap_start).total_seconds(),
            )
        self._gap_start = None
        self._last_seen_at = now
        self.connected_since_monotonic = time.monotonic()

    def _on_activity(self) -> None:
        self._last_seen_at = self._clock()
        self.last_activity_monotonic = time.monotonic()

    def _on_disconnected(self) -> None:
        self._gap_start = self._last_seen_at
        self.connected_since_monotonic = None

    def _decode(self, data: str) -> list[JSONObject]:
        try:
            return parse_records(json.loads(data))
        except (json.JSONDecodeError, TypeError) as error:
            logger.warning("[%s] Skipping malformed SSE event (%s): %.200s", self.name, error, data)
            return []

    async def _watched(self, items: AsyncIterator[SseItem]) -> AsyncIterator[SseItem]:
        """Re-yield `items`, raising `_WatchdogExpiredError` if one takes too long."""
        timeout = self._settings.read_timeout_seconds
        while True:
            try:
                if timeout > 0:
                    item = await asyncio.wait_for(anext(items), timeout)
                else:
                    item = await anext(items)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise _WatchdogExpiredError from None
            yield item

    def _delay(self, failures: int) -> float:
        """Exponential backoff with jitter after failures; a short pause otherwise."""
        if failures == 0:
            return MIN_RECONNECT_SECONDS
        if failures % PERSISTENT_FAILURE_ATTEMPTS == 0:
            logger.critical("[%s] %d consecutive SSE failures; check Crucible", self.name, failures)
        backoff = min(math.pow(2, failures), self._settings.max_backoff_seconds)
        return backoff + self._random.uniform(0, 1)

    def _report_failure(self, error: Exception) -> None:
        if isinstance(error, SseResponseError) and error.status == 401:  # noqa: PLR2004
            logger.error("[%s] SSE rejected the token despite a refresh; retrying", self.name)
        elif isinstance(error, aiohttp.ClientError | TimeoutError | ValueError | SseResponseError):
            logger.error("[%s] SSE connection failed: %s", self.name, error)
        else:
            logger.error("[%s] Unexpected SSE failure", self.name, exc_info=error)

    @asynccontextmanager
    async def _aiohttp_connect(
        self, headers: Mapping[str, str]
    ) -> AsyncIterator[AsyncIterator[SseItem]]:
        settings = self._settings
        async with aiohttp.ClientSession(
            read_bufsize=settings.read_buffer_bytes,
            connector=aiohttp.TCPConnector(ssl=settings.verify_tls),
            timeout=aiohttp.ClientTimeout(total=None),
        ) as session:
            request = session.get(
                settings.url,
                params={"query": self.query},
                headers={**headers, "Accept": "text/event-stream", "Cache-Control": "no-cache"},
            )
            response = await asyncio.wait_for(request, settings.connect_timeout_seconds)
            async with response:
                if response.status != 200:  # noqa: PLR2004
                    raise SseResponseError(response.status, await self._error_detail(response))
                if response.content_type != "text/event-stream":
                    content_type = response.headers.get("Content-Type")
                    raise SseResponseError(
                        response.status, f"unexpected Content-Type {content_type!r}"
                    )
                yield parse_events(response.content)

    async def _error_detail(self, response: aiohttp.ClientResponse) -> str:
        """Read the start of an error body, without waiting on a stalled server."""
        try:
            raw = await asyncio.wait_for(
                response.content.read(_ERROR_BODY_BYTES), self._settings.connect_timeout_seconds
            )
        except TimeoutError:
            return "<no body received>"
        return raw.decode("utf-8", errors="replace")


class _WatchdogExpiredError(Exception):
    """No activity arrived within the read timeout."""
