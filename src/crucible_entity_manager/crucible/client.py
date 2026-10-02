"""The cruciblelib-backed implementation of the Crucible protocols.

This is the only module that calls cruciblelib's controllers. It uses their
synchronous methods, run in worker threads: cruciblelib's async path re-raises
failures as a bare ``httpx.HTTPError``, losing the status code and the response
body that error handling depends on. Failures are translated into the
`protocols` exceptions.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx
from cruciblelib.authenticator import Authenticator, Token
from cruciblelib.controllers.v2.controller import ClientConfig, TimeoutConfig
from cruciblelib.controllers.v2.read_controller import ReadController
from cruciblelib.controllers.v2.write_controller import WriteController
from cruciblelib.exceptions import DetailedHTTPError

from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.protocols import (
    AuthorizationError,
    CrucibleError,
    RequestError,
    TransientError,
)

_GATEWAY_TIMEOUT = 504
_AUTH_STATUSES = frozenset({401, 403})


@dataclass(frozen=True, slots=True)
class ClientSettings:
    """HTTP behavior for every request."""

    request_timeout_seconds: float = 15.0
    """Read timeout: the longest wait for response data."""

    connect_timeout_seconds: float = 10.0
    transport_retries: int = 3
    """Connection attempts retried by the transport before a request fails."""


class CrucibleClient:
    """Searches and writes through cruciblelib's v2 controllers."""

    def __init__(self, settings: ClientSettings) -> None:
        """Authenticate and build the controllers.

        Credentials and endpoints come from the ``CRUCIBLE_*`` environment
        variables, which cruciblelib reads.

        Raises:
            CrucibleError: If an environment variable is missing or the first
                token cannot be fetched.
        """
        config = ClientConfig(
            timeout=TimeoutConfig(
                connect=settings.connect_timeout_seconds,
                read=settings.request_timeout_seconds,
                write=settings.request_timeout_seconds,
                pool=settings.connect_timeout_seconds,
            ),
            max_retries=settings.transport_retries,
        )
        try:
            tokens = _translated(lambda: _TokenSource(Authenticator()))
            self._tokens = tokens
            self._reader = _translated(lambda: ReadController(token_provider=tokens, config=config))
            self._writer = _translated(
                lambda: WriteController(token_provider=tokens, config=config)
            )
        except KeyError as error:
            msg = f"environment variable {error.args[0]} is not set"
            raise CrucibleError(msg) from error

    async def access_token(self) -> str:
        """Return a current access token, refreshing it if it has expired."""
        token = await asyncio.to_thread(lambda: _translated(self._tokens.get_token))
        return token.access_token

    async def search(self, sql: str, *, auto_backtick: bool = True) -> list[JSONObject]:
        """Return the rows of a SQL query."""
        result = await asyncio.to_thread(
            _translated,
            lambda: self._reader.search(sql, format="json", auto_backtick=auto_backtick),
        )
        return _rows(result)

    def write_batch(self, dataset: str, records: list[JSONObject]) -> None:
        """POST records: an upsert on ENTITY datasets, an append on EVENT datasets."""
        _translated(lambda: self._writer.write_record_batch_by_name(dataset, _as_dataset(records)))

    def update_batch(self, dataset: str, records: list[JSONObject]) -> list[JSONObject]:
        """PUT partial updates and return the records the server reports as failed."""
        failed = _translated(
            lambda: self._writer.update_entity_record_batch_by_name(
                dataset, _as_dataset(records), include_failed_records=True
            )
        )
        return _rows(failed)


class _TokenSource:
    """Adapts ``Authenticator`` to cruciblelib's ``TokenProvider`` protocol.

    The protocol requires ``get_token_async``, which ``Authenticator`` lacks.
    """

    def __init__(self, authenticator: Authenticator) -> None:
        self._authenticator = authenticator

    def get_token(self) -> Token:
        return self._authenticator.get_token()

    async def get_token_async(self) -> Token:
        return await asyncio.to_thread(self._authenticator.get_token)


def _translated[T](call: Callable[[], T]) -> T:
    """Run `call`, translating transport and HTTP failures into `CrucibleError`."""
    try:
        return call()
    except DetailedHTTPError as error:
        raise _classify(error.status_code, str(error), _body(error.response_text)) from error
    except httpx.TransportError as error:
        msg = f"no response from Crucible: {error}"
        raise TransientError(msg) from error
    except httpx.HTTPStatusError as error:
        raise _classify(error.response.status_code, str(error), error.response.text) from error
    except json.JSONDecodeError as error:
        msg = f"Crucible returned a response that is not valid JSON: {error}"
        raise TransientError(msg) from error


def _classify(status: int, message: str, body: str) -> CrucibleError:
    if status in {0, _GATEWAY_TIMEOUT}:
        return TransientError(message)
    if status in _AUTH_STATUSES:
        return AuthorizationError(message)
    return RequestError(message, status=status, body=body)


def _body(response_text: object) -> str:
    return response_text if isinstance(response_text, str) else repr(response_text)


def _rows(result: object) -> list[JSONObject]:
    """Check that a response is a list of records."""
    if not isinstance(result, list):
        msg = f"expected a list of records from Crucible, got {type(result).__name__}"
        raise RequestError(msg, status=200, body=repr(result)[:500])
    rows: list[JSONObject] = []
    for row in result:
        if not isinstance(row, dict):
            msg = f"expected records from Crucible, got a {type(row).__name__}"
            raise RequestError(msg, status=200, body=repr(row)[:500])
        rows.append(cast("JSONObject", row))
    return rows


def _as_dataset(records: list[JSONObject]) -> Any:
    """Pass records to cruciblelib, whose ``Dataset`` alias is ``List[Record]``."""
    return records
