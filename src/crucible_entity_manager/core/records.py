"""Operations on nested JSON records addressed by dotted paths.

Records stay nested from ingestion to output. A dotted path such as
``geodetic.latitude`` addresses ``record["geodetic"]["latitude"]``.
"""

import enum
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Final, cast

import numpy as np

from crucible_entity_manager.core.aliases import JSONObject, JSONValue


class Missing(enum.Enum):
    """Type of the `MISSING` sentinel, which marks an absent path."""

    MISSING = enum.auto()


MISSING: Final = Missing.MISSING


def get_path(record: JSONObject, path: str) -> JSONValue | Missing:
    """Return the value at `path`, or `MISSING` if any segment is absent.

    An explicit ``None`` is returned as ``None``; only absence yields `MISSING`.
    """
    current: JSONValue = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def set_path(record: JSONObject, path: str, value: JSONValue) -> None:
    """Set the value at `path`, creating or replacing intermediate objects.

    A non-object value found at an intermediate segment is replaced by an empty
    object. `value` is stored by reference, not copied.
    """
    *parents, leaf = path.split(".")
    current = record
    for part in parents:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[leaf] = value


def remove_path(record: JSONObject, path: str) -> None:
    """Remove the value at `path` and prune intermediate objects left empty."""
    *parents, leaf = path.split(".")
    chain: list[tuple[JSONObject, str]] = []
    current = record
    for part in parents:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        chain.append((current, part))
        current = child
    current.pop(leaf, None)
    for parent, part in reversed(chain):
        if parent[part]:
            break
        del parent[part]


def clone_value(value: JSONValue) -> JSONValue:
    """Return a deep copy of a JSON value."""
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    return value


def clone_record(record: JSONObject) -> JSONObject:
    """Return a deep copy of a record."""
    return {key: clone_value(item) for key, item in record.items()}


def compact_record(record: Mapping[str, object]) -> JSONObject:
    """Return a JSON-native copy of `record` without empty or non-finite values.

    Rules, as at ``1b534df``:

    - ``None``, ``""`` and non-finite floats are removed from objects, and from
      lists that are object values.
    - Objects inside such lists are compacted; objects and lists left empty are
      removed.
    - A list inside a list is kept as it is, so positions within it hold. Only
      its values are converted, and a non-finite float in it becomes ``None``.
    - NumPy scalars become Python scalars; dates and datetimes become ISO strings.

    Raises:
        TypeError: If a value is not representable as JSON, including tuples and
            objects with non-string keys.
    """
    result: JSONObject = {}
    for key, value in _require_string_keys(record).items():
        compacted = _compact(value)
        if compacted is not MISSING:
            result[key] = compacted
    return result


def parse_records(payload: str | bytes | JSONValue) -> list[JSONObject]:
    """Normalize one event payload into a list of records.

    A JSON string is decoded. A single object becomes a one-element list. Objects
    passed in already decoded are deep-copied so the caller's data is not shared.

    Raises:
        json.JSONDecodeError: If a string payload is not valid JSON.
        TypeError: If the payload is not an object or an array of objects, or an
            object passed in already decoded has a non-string key.
    """
    if isinstance(payload, str | bytes):
        parsed = cast("JSONValue", json.loads(payload))
        copy = False
    else:
        parsed = payload
        copy = True
    if isinstance(parsed, dict):
        return [_copy_checked(parsed) if copy else parsed]
    if isinstance(parsed, list):
        records: list[JSONObject] = []
        for item in parsed:
            if not isinstance(item, dict):
                msg = "every event in a payload array must be a JSON object"
                raise TypeError(msg)
            records.append(_copy_checked(item) if copy else item)
        return records
    msg = "event payload must be a JSON object or an array of objects"
    raise TypeError(msg)


def _copy_checked(record: JSONObject) -> JSONObject:
    """Copy a record received already decoded, checking its keys at runtime."""
    _require_string_keys(record)
    return clone_record(record)


def _compact(value: object) -> JSONValue | Missing:
    if isinstance(value, Mapping):
        nested = compact_record(_require_string_keys(value))
        return nested or MISSING
    if isinstance(value, list):
        items = [
            compacted
            for item in value
            if (compacted := _verbatim(item) if isinstance(item, list) else _compact(item))
            is not MISSING
        ]
        return items or MISSING
    return _compact_scalar(value)


def _verbatim(value: object) -> JSONValue:
    """Convert `value` to JSON without removing anything."""
    if isinstance(value, Mapping):
        return {key: _verbatim(item) for key, item in _require_string_keys(value).items()}
    if isinstance(value, list):
        return [_verbatim(item) for item in value]
    return _json_scalar(value)


def _require_string_keys[K](mapping: Mapping[K, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            msg = f"object key {key!r} is not a string"
            raise TypeError(msg)
        result[key] = item
    return result


def _compact_scalar(value: object) -> JSONValue | Missing:
    scalar = _json_scalar(value)
    return MISSING if scalar is None or scalar == "" else scalar


def _json_scalar(value: object) -> JSONValue:
    """Convert a scalar to JSON; a non-finite float becomes ``None``.

    Raises:
        TypeError: If `value` is not representable as JSON.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    msg = f"value of type {type(value).__name__} is not representable as JSON"
    raise TypeError(msg)
