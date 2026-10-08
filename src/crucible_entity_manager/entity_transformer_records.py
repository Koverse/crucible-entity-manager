"""Record-native helpers for the entity transformer hot path."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import numpy as np


MISSING = object()
JSONObject = dict[str, Any]

WGS84_A = 6_378_137.0
WGS84_E2 = 6.6943799901413165e-3
SIGMA_95 = 2.448


def clone_record(record: JSONObject) -> JSONObject:
    """Deep-copy one JSON dictionary, including all nested dicts and lists."""
    return copy.deepcopy(record)


def get_value(record: JSONObject, path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def set_value(record: JSONObject, path: str, value: Any) -> None:
    parts = path.split(".")
    current = record
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def remove_value(record: JSONObject, path: str) -> None:
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


def parse_records(value: object) -> list[JSONObject]:
    """Normalize one decoded JSON object or array into the internal list[dict]."""
    parsed = json.loads(value) if isinstance(value, str) else value
    if isinstance(parsed, dict):
        if not all(isinstance(key, str) for key in parsed):
            raise ValueError("every event must have string keys")
        return [clone_record(parsed)]
    if isinstance(parsed, list):
        if not all(isinstance(record, dict) for record in parsed):
            raise ValueError("every event must be an object")
        return [clone_record(record) for record in parsed]
    raise ValueError("event payload must be a JSON object or an array of objects")


def compact_record(record: JSONObject) -> JSONObject:
    result: JSONObject = {}
    for key, value in record.items():
        if isinstance(value, dict):
            compacted = compact_record(value)
            if compacted:
                result[key] = compacted
        elif isinstance(value, list):
            compacted_list = [
                compact_record(item) if isinstance(item, dict) else _native_scalar(item)
                for item in value
            ]
            compacted_list = [item for item in compacted_list if item not in (None, "", {})]
            if compacted_list:
                result[key] = compacted_list
        else:
            native = _native_scalar(value)
            if native is not None and native != "":
                result[key] = native
    return result


def _native_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return value


def _number(value: Any, default: float = np.nan) -> float:
    if value is MISSING or value is None or isinstance(value, (bool, np.bool_)):
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


def add_track_wgs84_kinematics(records: list[JSONObject]) -> list[JSONObject]:
    """Add nested WGS84 geodetic coordinates from nested ECEF positions."""
    result = [clone_record(record) for record in records]
    indices: list[int] = []
    positions: list[tuple[float, float, float]] = []
    for index, record in enumerate(result):
        x = _number(get_value(record, "ecefPosition.x"))
        y = _number(get_value(record, "ecefPosition.y"))
        z = _number(get_value(record, "ecefPosition.z"))
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            indices.append(index)
            positions.append((x, y, z))
    if not indices:
        return result

    xyz = np.asarray(positions, dtype=np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    semi_minor = WGS84_A * np.sqrt(1.0 - WGS84_E2)
    second_eccentricity = (WGS84_A**2 - semi_minor**2) / semi_minor**2
    horizontal = np.hypot(x, y)
    theta = np.arctan2(WGS84_A * z, semi_minor * horizontal)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(
        z + second_eccentricity * semi_minor * sin_theta**3,
        horizontal - WGS84_E2 * WGS84_A * cos_theta**3,
    )
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(latitude) ** 2)
    altitude = horizontal / np.cos(latitude) - prime_vertical

    for offset, record_index in enumerate(indices):
        record = result[record_index]
        set_value(record, "geodetic.latitude", float(np.degrees(latitude[offset])))
        set_value(record, "geodetic.longitude", float(np.degrees(longitude[offset])))
        set_value(record, "geodetic.altitude", float(altitude[offset]))
    return result


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
    vertical_variance: float,
) -> np.ndarray:
    major_sigma = major / SIGMA_95
    minor_sigma = minor / SIGMA_95
    cos_theta, sin_theta = np.cos(orientation), np.sin(orientation)
    count = len(latitude)
    covariance = np.zeros((count, 3, 3), dtype=np.float64)
    covariance[:, 0, 0] = major_sigma**2 * sin_theta**2 + minor_sigma**2 * cos_theta**2
    covariance[:, 0, 1] = (major_sigma**2 - minor_sigma**2) * sin_theta * cos_theta
    covariance[:, 1, 0] = covariance[:, 0, 1]
    covariance[:, 1, 1] = major_sigma**2 * cos_theta**2 + minor_sigma**2 * sin_theta**2
    covariance[:, 2, 2] = vertical_variance

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


def add_report_ecef_kinematics(records: list[JSONObject]) -> list[JSONObject]:
    """Enrich entity report records using batched NumPy operations."""
    result = [clone_record(record) for record in records]
    valid_indices: list[int] = []
    latitudes: list[float] = []
    longitudes: list[float] = []
    altitudes: list[float] = []
    for index, record in enumerate(result):
        latitude = _number(get_value(record, "geodetic.latitude"))
        longitude = _number(get_value(record, "geodetic.longitude"))
        timestamp = get_value(record, "estimatedKinematics.kinematicsTimestamp")
        if timestamp in (MISSING, None, "") or not np.isfinite(latitude + longitude):
            continue
        valid_indices.append(index)
        latitudes.append(latitude)
        longitudes.append(longitude)
        altitudes.append(_number(get_value(record, "geodetic.altitude"), 0.0))
    if not valid_indices:
        return result

    latitude_array = np.asarray(latitudes, dtype=np.float64)
    longitude_array = np.asarray(longitudes, dtype=np.float64)
    altitude_array = np.asarray(altitudes, dtype=np.float64)
    altitude_array = np.where(np.isfinite(altitude_array), altitude_array, 0.0)
    x, y, z = _geodetic_to_ecef(latitude_array, longitude_array, altitude_array)

    selected = [result[index] for index in valid_indices]
    speed = np.asarray([_number(get_value(record, "speed")) for record in selected])
    heading = np.asarray([_number(get_value(record, "heading")) for record in selected])
    valid_velocity = np.isfinite(speed) & np.isfinite(heading)
    safe_speed = np.where(valid_velocity, speed, 0.0)
    safe_heading = np.where(valid_velocity, heading, 0.0)
    east = safe_speed * np.sin(safe_heading)
    north = safe_speed * np.cos(safe_heading)
    dx, dy, dz = _enu_velocity_to_ecef(
        latitude_array, longitude_array, east, north, np.zeros(len(selected))
    )

    major = np.asarray([
        _number(get_value(record, "uncertainty.uncertaintyEllipse.semiMajorAxisLength"))
        for record in selected
    ])
    minor = np.asarray([
        _number(get_value(record, "uncertainty.uncertaintyEllipse.semiMinorAxisLength"), 100.0)
        for record in selected
    ])
    orientation = np.asarray([
        _number(get_value(record, "uncertainty.uncertaintyEllipse.orientation"), 0.0)
        for record in selected
    ])
    valid_position_covariance = np.isfinite(major) & (major >= 0.0)
    safe_major = np.where(valid_position_covariance, major, 100.0)
    safe_minor = np.where(np.isfinite(minor) & (minor >= 0.0), minor, 100.0)
    safe_orientation = np.where(np.isfinite(orientation), orientation, 0.0)
    position_covariance = _ellipse_to_ecef_covariance(
        latitude_array,
        longitude_array,
        safe_major,
        safe_minor,
        safe_orientation,
        10_000.0,
    )
    velocity_covariance = None
    if valid_velocity.any():
        velocity_indices = np.flatnonzero(valid_velocity)
        velocity_covariance = _ellipse_to_ecef_covariance(
            latitude_array[velocity_indices],
            longitude_array[velocity_indices],
            np.full(len(velocity_indices), 2.0),
            np.full(len(velocity_indices), 1.0),
            safe_heading[velocity_indices],
            4.0,
        )

    covariance_paths = ("xx", "xy", "xz", "yy", "yz", "zz")
    covariance_indices = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    velocity_offset = 0
    for offset, record_index in enumerate(valid_indices):
        record = result[record_index]
        for axis, value in zip(("x", "y", "z"), (x[offset], y[offset], z[offset])):
            set_value(record, f"ecefPosition.{axis}", float(value))
        if valid_velocity[offset] and velocity_covariance is not None:
            for axis, value in zip(("x", "y", "z"), (dx[offset], dy[offset], dz[offset])):
                set_value(record, f"ecefVelocity.{axis}", float(value))
            covariance = velocity_covariance[velocity_offset]
            velocity_offset += 1
            for suffix, indices in zip(("dxdx", "dxdy", "dxdz", "dydy", "dydz", "dzdz"), covariance_indices):
                set_value(record, f"velocityCovariance.{suffix}", float(covariance[indices]))
        if valid_position_covariance[offset]:
            covariance = position_covariance[offset]
            for suffix, indices in zip(covariance_paths, covariance_indices):
                set_value(record, f"positionCovariance.{suffix}", float(covariance[indices]))
    return result