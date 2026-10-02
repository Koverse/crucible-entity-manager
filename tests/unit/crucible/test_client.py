"""The cruciblelib adapter's error translation and response checks.

Constructing a `CrucibleClient` authenticates against Crucible, so these tests
drive the module's translation functions directly.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from cruciblelib.exceptions import DetailedHTTPError

from crucible_entity_manager.crucible import client
from crucible_entity_manager.crucible.protocols import (
    AuthorizationError,
    CrucibleError,
    RequestError,
    TransientError,
)

REQUEST = httpx.Request("POST", "https://crucible.example/api/v2/write")


def status_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    response = httpx.Response(status, request=REQUEST, text=body)
    return httpx.HTTPStatusError(f"{status}", request=REQUEST, response=response)


def raising(error: Exception) -> Callable[[], object]:
    def call() -> object:
        raise error

    return call


@pytest.mark.parametrize(
    ("error", "translated"),
    [
        (DetailedHTTPError(status_error(504), "gateway"), TransientError),
        (DetailedHTTPError(httpx.ReadTimeout("slow", request=REQUEST), "slow"), TransientError),
        (DetailedHTTPError(status_error(401), "expired"), AuthorizationError),
        (DetailedHTTPError(status_error(403), "denied"), AuthorizationError),
        (
            DetailedHTTPError(
                status_error(400, "Error processing query"), "Error processing query"
            ),
            RequestError,
        ),
        (httpx.ConnectError("refused", request=REQUEST), TransientError),
        (status_error(500, "oops"), RequestError),
    ],
)
def test_failures_are_translated(error: Exception, translated: type[Exception]) -> None:
    with pytest.raises(translated) as raised:
        client._translated(raising(error))
    assert raised.value.__cause__ is error


def test_request_errors_keep_status_and_body() -> None:
    with pytest.raises(RequestError) as raised:
        client._translated(
            raising(DetailedHTTPError(status_error(400, "bad"), "Error processing query"))
        )
    assert raised.value.status == 400
    assert raised.value.body == "Error processing query"


def test_successful_call_returns_its_value() -> None:
    assert client._translated(lambda: 204) == 204


def test_rows_accepts_a_list_of_records() -> None:
    assert client._rows([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize("result", [{"a": 1}, [{"a": 1}, 2], None])
def test_rows_rejects_anything_else(result: object) -> None:
    with pytest.raises(RequestError, match="expected"):
        client._rows(result)


class FakeController:
    """Stands in for cruciblelib's v2 controllers, recording how they are used."""

    instances: list["FakeController"] = []  # noqa: RUF012 - per-test registry, reset by the fixture

    def __init__(self, *, token_provider: object, config: object) -> None:
        self.token_provider = token_provider
        self.config = config
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.result: object = None
        FakeController.instances.append(self)

    def _record(self, name: str, *args: object, **kwargs: object) -> object:
        self.calls.append((name, args, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def search(self, *args: object, **kwargs: object) -> object:
        return self._record("search", *args, **kwargs)

    def write_record_batch_by_name(self, *args: object, **kwargs: object) -> object:
        return self._record("write_record_batch_by_name", *args, **kwargs)

    def update_entity_record_batch_by_name(self, *args: object, **kwargs: object) -> object:
        return self._record("update_entity_record_batch_by_name", *args, **kwargs)


class FakeToken:
    access_token = "token-value"


class FakeAuthenticator:
    def get_token(self) -> FakeToken:
        return FakeToken()


@pytest.fixture
def wired(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[client.CrucibleClient, FakeController, FakeController]:
    FakeController.instances = []
    monkeypatch.setattr(client, "Authenticator", FakeAuthenticator)
    monkeypatch.setattr(client, "ReadController", FakeController)
    monkeypatch.setattr(client, "WriteController", FakeController)
    crucible = client.CrucibleClient(
        client.ClientSettings(
            request_timeout_seconds=12.0, connect_timeout_seconds=4.0, transport_retries=1
        )
    )
    reader, writer = FakeController.instances
    return crucible, reader, writer


class TestCrucibleClient:
    def test_controllers_share_timeouts_and_tokens(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        _, reader, writer = wired
        for controller in (reader, writer):
            config = controller.config
            assert isinstance(config, client.ClientConfig)
            assert (config.timeout.connect, config.timeout.read, config.timeout.pool) == (
                4.0,
                12.0,
                4.0,
            )
            assert config.max_retries == 1
        assert reader.token_provider is writer.token_provider

    async def test_token_source_adapts_the_authenticator(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        _, reader, _ = wired
        source = reader.token_provider
        assert isinstance(source, client._TokenSource)
        assert source.get_token().access_token == "token-value"
        assert (await source.get_token_async()).access_token == "token-value"

    async def test_access_token(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        crucible, _, _ = wired
        assert await crucible.access_token() == "token-value"

    async def test_search_passes_backtick_choice_and_checks_rows(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        crucible, reader, _ = wired
        reader.result = [{"a": 1}]
        assert await crucible.search("SELECT 1", auto_backtick=False) == [{"a": 1}]
        assert reader.calls == [
            ("search", ("SELECT 1",), {"format": "json", "auto_backtick": False})
        ]

    async def test_search_failures_are_translated(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        crucible, reader, _ = wired
        reader.result = DetailedHTTPError(status_error(401), "expired")
        with pytest.raises(AuthorizationError):
            await crucible.search("SELECT 1")

    def test_write_batch(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        crucible, _, writer = wired
        writer.result = 204
        crucible.write_batch("Heads", [{"trackId": "t"}])
        assert writer.calls == [("write_record_batch_by_name", ("Heads", [{"trackId": "t"}]), {})]

    def test_update_batch_requests_failed_records(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        crucible, _, writer = wired
        writer.result = [{"trackId": "missing"}]
        assert crucible.update_batch("Heads", [{"trackId": "missing"}]) == [{"trackId": "missing"}]
        assert writer.calls == [
            (
                "update_entity_record_batch_by_name",
                ("Heads", [{"trackId": "missing"}]),
                {"include_failed_records": True},
            )
        ]

    def test_non_string_error_bodies_are_kept_readably(
        self, wired: tuple[client.CrucibleClient, FakeController, FakeController]
    ) -> None:
        crucible, _, writer = wired
        writer.result = DetailedHTTPError(
            status_error(422, '{"error": "bad field"}'), '{"error": "bad field"}'
        )
        with pytest.raises(RequestError) as raised:
            crucible.write_batch("Heads", [{"trackId": "t"}])
        assert "bad field" in raised.value.body


class TestConstruction:
    def test_missing_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing() -> object:
            raise KeyError("CRUCIBLE_CLIENT_SECRET")

        monkeypatch.setattr(client, "Authenticator", missing)
        with pytest.raises(CrucibleError, match="CRUCIBLE_CLIENT_SECRET is not set"):
            client.CrucibleClient(client.ClientSettings())

    def test_rejected_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def rejected() -> object:
            raise status_error(401)

        monkeypatch.setattr(client, "Authenticator", rejected)
        with pytest.raises(AuthorizationError):
            client.CrucibleClient(client.ClientSettings())


def test_invalid_json_is_transient() -> None:
    def garbled() -> object:
        return json.loads("{truncated")

    with pytest.raises(TransientError, match="not valid JSON"):
        client._translated(garbled)
