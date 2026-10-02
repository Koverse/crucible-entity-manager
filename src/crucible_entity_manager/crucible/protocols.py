"""The Crucible operations the pipeline uses, as structural types.

Components and the writer depend on these protocols, not on cruciblelib, so
tests substitute in-memory fakes.
"""

from typing import Protocol

from crucible_entity_manager.core.aliases import JSONObject


class CrucibleError(Exception):
    """A Crucible request failed."""


class TransientError(CrucibleError):
    """No usable response: a timeout, a dropped connection, or a 504."""


class AuthorizationError(CrucibleError):
    """The server rejected the credentials (401 or 403)."""


class RequestError(CrucibleError):
    """The server rejected the request."""

    def __init__(self, message: str, *, status: int, body: str) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class SearchClient(Protocol):
    """Runs SQL searches."""

    async def search(self, sql: str, *, auto_backtick: bool = True) -> list[JSONObject]:
        """Return the rows of a SQL query.

        Raises:
            CrucibleError: If the request fails.
        """
        ...


class WriteClient(Protocol):
    """Writes records. Calls block, so callers run them in a worker thread."""

    def write_batch(self, dataset: str, records: list[JSONObject]) -> None:
        """POST records: an upsert on ENTITY datasets, an append on EVENT datasets.

        Raises:
            CrucibleError: If the request fails.
        """
        ...

    def update_batch(self, dataset: str, records: list[JSONObject]) -> list[JSONObject]:
        """PUT partial updates to existing ENTITY records.

        Returns:
            The records the server reports as failed, because they were not
            found or failed validation. The response does not say which.

        Raises:
            CrucibleError: If the request fails.
        """
        ...


class CrucibleService(SearchClient, WriteClient, Protocol):
    """Everything a component needs from Crucible."""

    async def access_token(self) -> str:
        """Return a current access token, for SSE subscriptions.

        Raises:
            CrucibleError: If no token can be obtained.
        """
        ...
