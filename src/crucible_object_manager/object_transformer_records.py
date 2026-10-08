"""Nested-record helpers for parsing, write compaction, and batched kinematics."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

MISSING = object()
JSONObject = dict[str, Any]
WGS84_A = 6_378_137.0
WGS84_E2 = 6.6943799901413165e-3
SIGMA_95 = 2.448


def get_path(record: Mapping[str, Any], path: str) -> object:
    current: object = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def set_path(record: JSONObject, path: str, value: Any) -> None:
    parts = path.split(".")
    current = record
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def remove_path(record: JSONObject, path: str) -> None:
    parts = path.split(".")
    parents: list[tuple[JSONObject, str]] = []
    current = record
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        parents.append((current, part))
        current = child
    current.pop(parts[-1], None)
    for parent, part in reversed(parents):
        child = parent.get(part)
        if isinstance(child, dict) and not child:
            parent.pop(part, None)


def clone_record(record: Mapping[str, Any]) -> JSONObject:
    """Deep-copy at ownership boundaries so nested dicts/lists are not aliased."""
    return copy.deepcopy(dict(record))


def parse_records(payload: object) -> list[JSONObject]:
    """Decode one/list SSE payload and return worker-owned records."""
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(parsed, Mapping):
        return [copy.deepcopy(dict(parsed))]
    if isinstance(parsed, list) and all(isinstance(row, Mapping) for row in parsed):
        return [copy.deepcopy(dict(row)) for row in parsed]
    raise ValueError("event payload must be an object or an array of objects")


def _compact(value: Any) -> Any:
    if value is None or value == "":
        return MISSING
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            compacted = _compact(item)
            if compacted is not MISSING:
                result[key] = compacted
        return result if result else MISSING
    if isinstance(value, list):
        return [compacted for item in value if (compacted := _compact(item)) is not MISSING]
    if isinstance(value, float) and not math.isfinite(value):
        return MISSING
    item = getattr(value, "item", None)
    if callable(item):
        return _compact(item())
    return value


def compact_record(record: Mapping[str, Any]) -> JSONObject:
    """Drop empty and non-finite values before a record is written to Crucible."""
    compacted = _compact(dict(record))
    return compacted if isinstance(compacted, dict) else {}


def _number(value: object, default: float = math.nan) -> float:
    if value is MISSING or value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _geodetic_to_ecef(
    latitude: np.ndarray, longitude: np.ndarray, altitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (prime_vertical + altitude) * cos_lat * np.cos(longitude)
    y = (prime_vertical + altitude) * cos_lat * np.sin(longitude)
    z = (prime_vertical * (1.0 - WGS84_E2) + altitude) * sin_lat
    return x, y, z


def _enu_velocity_to_ecef(
    latitude: np.ndarray,
    longitude: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up
    return dx, dy, dz


def _ellipse_to_ecef_covariance(
    latitude: np.ndarray,
    longitude: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
    orientation: np.ndarray,
) -> np.ndarray:
    major_sigma = major / SIGMA_95
    minor_sigma = minor / SIGMA_95
    cos_angle, sin_angle = np.cos(orientation), np.sin(orientation)
    count = len(latitude)
    covariance = np.zeros((count, 3, 3), dtype=np.float64)
    covariance[:, 0, 0] = major_sigma**2 * sin_angle**2 + minor_sigma**2 * cos_angle**2
    covariance[:, 0, 1] = (major_sigma**2 - minor_sigma**2) * sin_angle * cos_angle
    covariance[:, 1, 0] = covariance[:, 0, 1]
    covariance[:, 1, 1] = major_sigma**2 * cos_angle**2 + minor_sigma**2 * sin_angle**2
    covariance[:, 2, 2] = 10_000.0

    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    rotation = np.zeros((count, 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = -sin_lon
    rotation[:, 0, 1] = -sin_lat * cos_lon
    rotation[:, 0, 2] = cos_lat * cos_lon
    rotation[:, 1, 0] = cos_lon
    rotation[:, 1, 1] = -sin_lat * sin_lon
    rotation[:, 1, 2] = cos_lat * sin_lon
    rotation[:, 2, 1] = cos_lat
    rotation[:, 2, 2] = sin_lat
    return np.einsum("nij,njk,nlk->nil", rotation, covariance, rotation)


def _update_nested(record: JSONObject, key: str, values: Mapping[str, float]) -> None:
    nested = record.get(key)
    if not isinstance(nested, dict):
        nested = {}
        record[key] = nested
    nested.update(values)


def add_object_ecef_kinematics(
    records: Sequence[JSONObject],
    *,
    has_velocity: bool,
    has_position_covariance: bool,
    copy_records: bool = True,
) -> list[JSONObject]:
    """Batch numeric transforms, then scatter results into nested event records.

    With copy_records=False, the caller transfers ownership for this mutating step.
    """
    result = (
        [copy.deepcopy(record) for record in records]
        if copy_records
        else list(records)
    )
    indices: list[int] = []
    latitudes: list[float] = []
    longitudes: list[float] = []
    altitudes: list[float] = []
    for index, record in enumerate(result):
        timestamp = get_path(record, "estimatedKinematics.kinematicsTimestamp")
        latitude = _number(get_path(record, "estimatedKinematics.position.latitude"))
        longitude = _number(get_path(record, "estimatedKinematics.position.longitude"))
        if timestamp in (MISSING, None, "") or not np.isfinite(latitude + longitude):
            continue
        indices.append(index)
        latitudes.append(latitude)
        longitudes.append(longitude)
        altitudes.append(_number(get_path(record, "estimatedKinematics.position.altitude"), 0.0))
    if not indices:
        return []

    # Convert the selected scalar fields once, run vectorized math, then update records.
    latitude = np.asarray(latitudes, dtype=np.float64)
    longitude = np.asarray(longitudes, dtype=np.float64)
    altitude = np.nan_to_num(np.asarray(altitudes, dtype=np.float64), nan=0.0)
    x, y, z = _geodetic_to_ecef(latitude, longitude, altitude)

    if has_velocity:
        selected = [result[index] for index in indices]
        east = np.asarray([
            _number(get_path(record, "estimatedKinematics.velocity.eastSpeed"))
            for record in selected
        ])
        north = np.asarray([
            _number(get_path(record, "estimatedKinematics.velocity.northSpeed"))
            for record in selected
        ])
        down = np.asarray([
            _number(get_path(record, "estimatedKinematics.velocity.downSpeed"), 0.0)
            for record in selected
        ])
        dx, dy, dz = _enu_velocity_to_ecef(
            latitude, longitude, east, north, -np.nan_to_num(down, nan=0.0)
        )
        valid_velocity = np.isfinite(east) & np.isfinite(north)

    if has_position_covariance:
        selected = [result[index] for index in indices]
        major = np.asarray([
            _number(get_path(
                record,
                "estimatedKinematics.uncertainty.uncertaintyEllipse.semiMajorAxisLength",
            ))
            for record in selected
        ])
        minor = np.asarray([
            _number(get_path(
                record,
                "estimatedKinematics.uncertainty.uncertaintyEllipse.semiMinorAxisLength",
            ),
            1000.0)
            for record in selected
        ])
        orientation = np.asarray([
            _number(get_path(
                record,
                "estimatedKinematics.uncertainty.uncertaintyEllipse.orientation",
            ),
            0.0)
            for record in selected
        ])
        covariance = _ellipse_to_ecef_covariance(
            latitude,
            longitude,
            np.nan_to_num(major, nan=1000.0),
            np.nan_to_num(minor, nan=1000.0),
            np.nan_to_num(orientation),
        )

    covariance_fields = (
        ("xx", 0, 0), ("xy", 0, 1), ("xz", 0, 2),
        ("yy", 1, 1), ("yz", 1, 2), ("zz", 2, 2),
    )
    for offset, record_index in enumerate(indices):
        record = result[record_index]
        set_path(record, "estimatedKinematics.position.altitude", float(altitude[offset]))
        _update_nested(record, "ecefPosition", {
            axis: float(value)
            for axis, value in zip(("x", "y", "z"), (x[offset], y[offset], z[offset]))
        })
        if has_velocity and valid_velocity[offset]:
            _update_nested(record, "ecefVelocity", {
                axis: float(value)
                for axis, value in zip(("dx", "dy", "dz"), (dx[offset], dy[offset], dz[offset]))
            })
        if has_position_covariance:
            _update_nested(record, "positionCovariance", {
                field: float(covariance[offset, row, column])
                for field, row, column in covariance_fields
            })
    return [result[index] for index in indices]