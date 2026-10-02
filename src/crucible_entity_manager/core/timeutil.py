"""UTC timestamp parsing and formatting.

Every datetime in this package is timezone-aware UTC.
"""

from datetime import UTC, datetime

from crucible_entity_manager.core.aliases import JSONValue


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def parse_timestamp(value: JSONValue | datetime) -> datetime | None:
    """Parse an ISO 8601 timestamp into an aware UTC datetime.

    Accepts a ``Z`` suffix or a numeric offset, and any number of fractional
    digits. Naive inputs are taken to be UTC.

    Returns:
        The parsed time, or ``None`` if `value` is not a parseable timestamp.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_timestamp(moment: datetime) -> str:
    """Format an aware datetime as ``YYYY-MM-DDTHH:MM:SS.mmmZ`` in UTC.

    Raises:
        ValueError: If `moment` is naive.
    """
    if moment.tzinfo is None:
        msg = "cannot format a naive datetime"
        raise ValueError(msg)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
