"""Building Crucible SQL safely from configured names and values."""

import re
from collections.abc import Iterable
from typing import Final

_IDENTIFIER: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def identifier(name: str) -> str:
    """Return `name` for use as a dataset name in SQL.

    Raises:
        ValueError: If `name` is not a plain identifier. Configured names reach
            SQL unquoted, so anything else is rejected rather than escaped.
    """
    if not _IDENTIFIER.fullmatch(name):
        msg = f"{name!r} is not a valid dataset name"
        raise ValueError(msg)
    return name


def string_literal(value: str) -> str:
    """Return `value` as a single-quoted SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def string_list(values: Iterable[str]) -> str:
    """Return `values` as a parenthesized list of SQL string literals."""
    return "(" + ", ".join(string_literal(value) for value in values) + ")"
