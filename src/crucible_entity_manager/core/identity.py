"""Deterministic track identifiers.

A component track's ID is derived from its origin dataset and identity fields,
and a principal track's ID from its supersede root. Every process computes the
same IDs, so restarts and partitions never need to coordinate.
"""

import json
import math
import re
import uuid
from collections.abc import Sequence
from typing import Final

import numpy as np

from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.records import MISSING, get_path

IDENTITY_WILDCARD: Final = "identity.*"
"""Field path that expands to every field of the record's ``identity`` object."""

DEFAULT_TRACK_ID_FIELDS: Final = (IDENTITY_WILDCARD,)


def canonical_identity_value(value: object) -> str:
    """Render one identity value as a canonical string.

    Equal identities must render equally whatever their JSON representation:
    ``7``, ``7.0`` and ``"7.0"`` all render as ``"7"``. Missing and non-finite
    values render as ``""`` and are omitted from the custom ID.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, str):
        return _canonical_numeric_string(value)
    if isinstance(value, list):
        return json.dumps([canonical_identity_value(item) for item in value], separators=(",", ":"))
    return str(value)


def identity_custom_id(
    record: JSONObject, field_paths: Sequence[str] = DEFAULT_TRACK_ID_FIELDS
) -> str:
    """Build the custom ID from the selected identity fields of `record`.

    `IDENTITY_WILDCARD` expands to every field of ``record["identity"]``; any
    other entry is a dotted path. Fields are ordered by path, and each non-empty
    value contributes ``"<label>:<value>"``, where the label is the path without
    its ``identity.`` prefix. Parts are joined with ``"-"``.
    """
    values: dict[str, JSONValue] = {}
    for path in field_paths:
        if path == IDENTITY_WILDCARD:
            identity = record.get("identity")
            if isinstance(identity, dict):
                for key, value in identity.items():
                    values[f"identity.{key}"] = value
            continue
        value = get_path(record, path)
        values[path] = None if value is MISSING else value

    parts: list[str] = []
    for path, value in sorted(values.items()):
        canonical = canonical_identity_value(value)
        if canonical:
            parts.append(f"{path.removeprefix('identity.')}:{canonical}")
    return "-".join(parts)


_TRACK_ID: Final = re.compile(r"[0-9a-f]{32}")


def is_track_id(value: str) -> bool:
    """Whether `value` has the form of a track ID: 32 lowercase hex digits."""
    return _TRACK_ID.fullmatch(value) is not None


def component_track_id(origin_dataset: str, custom_id: str) -> str:
    """Return the component track ID for a custom ID within one origin dataset."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{origin_dataset}:{custom_id}").hex


def principal_track_id(supersede_root: str) -> str:
    """Return the principal track ID for a supersede-chain root."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"entity_principal_track:{supersede_root}").hex


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return str(int(value)) if value.is_integer() else str(value)


def _canonical_numeric_string(value: str) -> str:
    """Normalize a decimal string that holds a whole number, e.g. ``"7.0"``."""
    if "." not in value:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return value
