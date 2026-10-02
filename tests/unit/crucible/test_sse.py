import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import aiohttp
import pytest
from aiohttp import web

from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.sse import (
    MIN_RECONNECT_SECONDS,
    SSE_PATH,
    SseItem,
    SseResponseError,
    SseSettings,
    SseSource,
    parse_events,
)

SETTINGS = SseSettings(url="https://crucible.example/sse", read_timeout_seconds=5.0)
T0 = datetime(2026, 9, 30, 12, tzinfo=UTC)


async def lines(*items: bytes) -> AsyncIterator[bytes]:
    for item in items:
        yield item


async def collect(stream: AsyncIterator[SseItem]) -> list[SseItem]:
    return [item async for item in stream]


class TestParseEvents:
    async def test_blank_line_dispatches_accumulated_data(self) -> None:
        stream = lines(b"data: first\n", b"data: second\n", b"\n", b"data:third\r\n", b"\r\n")
        assert await collect(parse_events(stream)) == ["first\nsecond", "third"]

    async def test_comments_are_reported_as_keep_alives(self) -> None:
        stream = lines(b": keep-alive\n", b"data: x\n", b"\n")
        assert await collect(parse_events(stream)) == [None, "x"]

    async def test_other_fields_are_ignored(self) -> None:
        stream = lines(b"event: update\n", b"id: 7\n", b"retry: 10\n", b"data: x\n", b"\n", b"\n")
        assert await collect(parse_events(stream)) == ["x"]

    async def test_a_leading_byte_order_mark_is_skipped(self) -> None:
        stream = lines("﻿data: x\n".encode(), b"\n")
        assert await collect(parse_events(stream)) == ["x"]

    async def test_an_unterminated_event_is_not_dispatched(self) -> None:
        assert await collect(parse_events(lines(b"data: partial\n"))) == []


class StopSourceError(BaseException):
    """Ends an otherwise endless source.

    A `BaseException`, because the source rightly treats every `Exception` as a
    reason to reconnect.
    """


type Outcome = list[SseItem] | Exception | str


class Script:
    """Plays one scripted outcome per connection attempt, on a fake clock.

    An outcome is a list of items to stream (``None`` is a keep-alive), an
    exception raised on connect, or "hang" for a connection that never sends
    anything. Each streamed item advances the clock 10 s; each sleep advances it
    by the slept time.
    """

    def __init__(self, *outcomes: Outcome, max_sleeps: int = 10, item_delay: float = 0.0) -> None:
        self.outcomes = list(outcomes)
        self.headers: list[Mapping[str, str]] = []
        self.sleeps: list[float] = []
        self.max_sleeps = max_sleeps
        self.item_delay = item_delay
        self.tokens = 0
        self.open_connections = 0
        self.now = T0

    def clock(self) -> datetime:
        return self.now

    async def token(self) -> str:
        self.tokens += 1
        return f"token-{self.tokens}"

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)
        if len(self.sleeps) >= self.max_sleeps:
            raise StopSourceError

    @asynccontextmanager
    async def connect(self, headers: Mapping[str, str]) -> AsyncIterator[AsyncIterator[SseItem]]:
        self.headers.append(dict(headers))
        if not self.outcomes:
            raise StopSourceError
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        self.open_connections += 1
        try:
            yield self._stream(outcome)
        finally:
            self.open_connections -= 1

    async def _stream(self, outcome: list[SseItem] | str) -> AsyncIterator[SseItem]:
        if outcome == "hang":
            await asyncio.Event().wait()
        for item in outcome:
            if self.item_delay:
                await asyncio.sleep(self.item_delay)
            self.now += timedelta(seconds=10)
            yield item


def source(script: Script, settings: SseSettings = SETTINGS) -> SseSource:
    return SseSource(
        "AIS",
        "SELECT * FROM AIS",
        settings,
        script.token,
        connect=script.connect,
        sleep=script.sleep,
        clock=script.clock,
    )


async def drain(sse: SseSource) -> list[list[JSONObject]]:
    """Collect batches until the script stops the source."""
    batches: list[list[JSONObject]] = []

    async def consume() -> None:
        async for batch in sse:
            batches.append(batch)  # noqa: PERF401 - batches before the stop must be kept

    with pytest.raises(StopSourceError):
        await consume()
    return batches


def gap_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if "SSE gap" in record.getMessage()]


class TestEvents:
    async def test_yields_the_records_of_each_event(self) -> None:
        script = Script(['{"a": 1}', '[{"b": 2}, {"c": 3}]'])
        assert await drain(source(script)) == [[{"a": 1}], [{"b": 2}, {"c": 3}]]

    async def test_refreshes_the_token_before_every_connect(self) -> None:
        script = Script(['{"a": 1}'], ['{"a": 2}'])
        await drain(source(script))
        assert [headers["Authorization"] for headers in script.headers] == [
            "Bearer token-1",
            "Bearer token-2",
            "Bearer token-3",
        ]

    async def test_skips_malformed_and_empty_events(self, caplog: pytest.LogCaptureFixture) -> None:
        script = Script(["{not json", "[]", "7", '{"a": 1}'])
        assert await drain(source(script)) == [[{"a": 1}]]
        assert "Skipping malformed SSE event" in caplog.text

    async def test_closing_the_iterator_closes_the_connection(self) -> None:
        script = Script(['{"a": 1}', '{"a": 2}'])
        batches = source(script).batches()
        assert await anext(batches) == [{"a": 1}]
        assert script.open_connections == 1
        await batches.aclose()
        assert script.open_connections == 0


class TestReconnects:
    async def test_a_clean_end_reconnects_after_a_short_pause(self) -> None:
        script = Script(['{"a": 1}'])
        await drain(source(script))
        assert script.sleeps[0] == MIN_RECONNECT_SECONDS

    async def test_failures_back_off_exponentially_and_reset_on_data(self) -> None:
        refused = aiohttp.ClientConnectionError("refused")
        script = Script(refused, refused, refused, ['{"a": 1}'], refused, max_sleeps=5)
        await drain(source(script))
        assert [int(delay) for delay in script.sleeps] == [2, 4, 8, int(MIN_RECONNECT_SECONDS), 2]
        jitter = [delay - int(delay) for delay in script.sleeps if delay != MIN_RECONNECT_SECONDS]
        assert all(0 <= value < 1 for value in jitter)

    async def test_keep_alives_do_not_reset_the_failure_count(self) -> None:
        refused = aiohttp.ClientConnectionError("refused")
        script = Script(refused, [None], refused, max_sleeps=3)
        await drain(source(script))
        assert [int(delay) for delay in script.sleeps] == [2, 2, 4]

    async def test_backoff_is_capped(self) -> None:
        script = Script(*[aiohttp.ClientConnectionError("down")] * 8, max_sleeps=8)
        await drain(source(script, SseSettings(url="x", max_backoff_seconds=10.0)))
        assert max(script.sleeps) < 11.0

    async def test_persistent_failures_are_critical(self, caplog: pytest.LogCaptureFixture) -> None:
        script = Script(*[aiohttp.ClientConnectionError("down")] * 10, max_sleeps=10)
        with caplog.at_level(logging.CRITICAL):
            await drain(source(script))
        assert "10 consecutive SSE failures" in caplog.text

    async def test_rejected_token_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        await drain(source(Script(SseResponseError(401, "expired"), max_sleeps=1)))
        assert "rejected the token" in caplog.text

    async def test_unexpected_failures_are_logged_with_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        await drain(source(Script(RuntimeError("bug"), max_sleeps=1)))
        assert "Unexpected SSE failure" in caplog.text
        assert "RuntimeError: bug" in caplog.text


class TestWatchdog:
    async def test_reconnects_a_silent_stream(self, caplog: pytest.LogCaptureFixture) -> None:
        script = Script("hang", ['{"a": 1}'])
        batches = await drain(source(script, SseSettings(url="x", read_timeout_seconds=0.01)))
        assert batches == [[{"a": 1}]]
        assert "No SSE activity for" in caplog.text
        assert 2 <= script.sleeps[0] < 3

    async def test_keep_alives_keep_a_quiet_stream_open(self) -> None:
        script = Script([None, None, None, '{"a": 1}'], item_delay=0.03)
        settings = SseSettings(url="x", read_timeout_seconds=0.05)
        assert await drain(source(script, settings)) == [[{"a": 1}]]
        assert script.sleeps[0] == MIN_RECONNECT_SECONDS

    async def test_can_be_disabled(self) -> None:
        script = Script(['{"a": 1}'])
        assert await drain(source(script, SseSettings(url="x", read_timeout_seconds=0))) == [
            [{"a": 1}]
        ]


class TestGaps:
    async def test_first_subscription_is_not_a_gap(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            await drain(source(Script(['{"a": 1}'], max_sleeps=1)))
        assert "[AIS] SSE subscribed at 2026-09-30T12:00:00.000Z" in caplog.text
        assert gap_warnings(caplog) == []

    async def test_gap_runs_from_last_activity_to_resubscription(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        script = Script(['{"a": 1}'], aiohttp.ClientConnectionError("down"), ['{"b": 2}'])
        await drain(source(script))
        (warning,) = gap_warnings(caplog)
        last_activity = "2026-09-30T12:00:10.000Z"
        resubscribed = T0 + timedelta(seconds=10 + script.sleeps[0] + script.sleeps[1])
        assert f"no data from {last_activity} to {resubscribed:%Y-%m-%dT%H:%M:%S}" in warning

    async def test_each_gap_is_reported_once_even_without_data(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        down = aiohttp.ClientConnectionError("down")
        script = Script([], down, down, [], ['{"a": 1}'])
        with caplog.at_level(logging.INFO):
            await drain(source(script))
        assert len(gap_warnings(caplog)) == 2
        assert sum("SSE subscribed" in record.getMessage() for record in caplog.records) == 1

    async def test_resuming_iteration_still_reports_the_gap(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        script = Script(['{"a": 1}'], ['{"b": 2}'])
        sse = source(script)
        first = sse.batches()
        assert await anext(first) == [{"a": 1}]
        await first.aclose()
        assert await anext(sse.batches()) == [{"b": 2}]
        assert len(gap_warnings(caplog)) == 1


class TestHealthSignals:
    async def test_events_and_keep_alives_are_activity(self) -> None:
        sse = source(Script([None]))
        assert sse.last_activity_monotonic is None
        await drain(sse)
        assert sse.last_activity_monotonic is not None

    async def test_connecting_alone_is_not_activity(self) -> None:
        sse = source(Script([]))
        await drain(sse)
        assert sse.last_activity_monotonic is None

    async def test_connected_since_tracks_the_open_subscription(self) -> None:
        sse = source(Script(['{"a": 1}']))
        batches = sse.batches()
        assert sse.connected_since_monotonic is None
        await anext(batches)
        assert sse.connected_since_monotonic is not None
        await batches.aclose()
        assert sse.connected_since_monotonic is None


class TestSettings:
    def test_builds_the_endpoint_from_the_services_host(self) -> None:
        settings = SseSettings.from_environ({"CRUCIBLE_SERVICES_HOST": "crucible.example"})
        assert settings.url == f"https://crucible.example{SSE_PATH}"
        assert settings.read_timeout_seconds == 180.0
        assert not settings.verify_tls

    def test_environment_overrides(self) -> None:
        settings = SseSettings.from_environ(
            {
                "CRUCIBLE_SSE_URL": "https://other/sse",
                "CRUCIBLE_SSE_READ_BUFSIZE": "1024",
                "CRUCIBLE_SSE_READ_TIMEOUT": "0",
                "CRUCIBLE_SSL_VERIFY": "true",
            }
        )
        assert (settings.url, settings.read_buffer_bytes, settings.read_timeout_seconds) == (
            "https://other/sse",
            1024,
            0.0,
        )
        assert settings.verify_tls

    def test_requires_an_endpoint(self) -> None:
        with pytest.raises(KeyError):
            SseSettings.from_environ({})


@pytest.fixture
async def server() -> AsyncIterator[tuple[str, list[web.Request]]]:
    """A local endpoint whose behavior is chosen by the query.

    "fail" answers 503, "plain" answers 200 with the wrong content type, "stall"
    never sends headers, "error-stall" sends 500 headers but no body, and
    anything else streams two events.
    """
    requests: list[web.Request] = []

    async def handle(request: web.Request) -> web.StreamResponse:
        requests.append(request)
        query = request.query["query"]
        if query == "fail":
            return web.Response(status=503, text="unavailable")
        if query == "plain":
            return web.Response(text='data: {"a": 1}\n\n')
        if query == "stall":
            await asyncio.sleep(1)
        if query == "error-stall":
            stalled = web.StreamResponse(status=500)
            stalled.content_length = 100
            await stalled.prepare(request)
            await asyncio.sleep(1)
            return stalled
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for payload in ({"a": 1}, [{"b": 2}]):
            await response.write(f": keep-alive\ndata: {json.dumps(payload)}\n\n".encode())
        return response

    app = web.Application()
    app.router.add_get("/sse", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    _, port = runner.addresses[0]
    try:
        yield f"http://127.0.0.1:{port}/sse", requests
    finally:
        await runner.cleanup()


def live_source(
    url: str, query: str, script: Script, connect_timeout_seconds: float = 10.0
) -> SseSource:
    settings = SseSettings(url=url, connect_timeout_seconds=connect_timeout_seconds)
    return SseSource("AIS", query, settings, script.token, sleep=script.sleep)


class TestAiohttpTransport:
    async def test_streams_events_with_the_query_and_headers(
        self, server: tuple[str, list[web.Request]]
    ) -> None:
        url, requests = server
        query = "SELECT * FROM AIS WHERE a = 'x&y'"
        assert await drain(live_source(url, query, Script(max_sleeps=1))) == [
            [{"a": 1}],
            [{"b": 2}],
        ]
        (request,) = requests
        assert request.query["query"] == query
        assert request.headers["Authorization"] == "Bearer token-1"
        assert request.headers["Accept"] == "text/event-stream"

    @pytest.mark.parametrize(
        ("query", "message"),
        [("fail", "HTTP 503: unavailable"), ("plain", "unexpected Content-Type 'text/plain")],
    )
    async def test_rejects_responses_that_are_not_event_streams(
        self,
        server: tuple[str, list[web.Request]],
        caplog: pytest.LogCaptureFixture,
        query: str,
        message: str,
    ) -> None:
        url, _ = server
        assert await drain(live_source(url, query, Script(max_sleeps=1))) == []
        assert message in caplog.text

    async def test_a_stalled_error_body_does_not_block_reconnecting(
        self, server: tuple[str, list[web.Request]], caplog: pytest.LogCaptureFixture
    ) -> None:
        url, _ = server
        sse = live_source(url, "error-stall", Script(max_sleeps=1), connect_timeout_seconds=0.1)
        assert await drain(sse) == []
        assert "HTTP 500: <no body received>" in caplog.text

    async def test_a_server_that_never_answers_times_out(
        self, server: tuple[str, list[web.Request]], caplog: pytest.LogCaptureFixture
    ) -> None:
        url, _ = server
        sse = live_source(url, "stall", Script(max_sleeps=1), connect_timeout_seconds=0.1)
        assert await drain(sse) == []
        assert "SSE connection failed" in caplog.text
