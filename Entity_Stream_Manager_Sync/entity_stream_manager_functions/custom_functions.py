"""Record-native custom hooks for Entity Stream Manager configs."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any


def _get(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _set(record: dict[str, Any], path: str, value: Any) -> None:
    target = record
    parts = path.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def placeholder_function(records, config):
    return records


def set_edh(records, config):
    for record in records:
        record["defaultEDH"] = ["CLS:U"]
    return records


def set_measurement_errors(records, config):
    defaults = {
        "position_semimajor": 1000.0,
        "position_semiminor": 700.0,
        "position_orientation": 0.0,
    }
    for record in records:
        for path, default in defaults.items():
            if _get(record, path) is None:
                _set(record, path, default)
    return records


def get_observation_time(records, config):
    for record in records:
        try:
            timestamp = (float(_get(record, "now_ms")) - 1000.0 * float(_get(record, "seen_pos"))) / 1000.0
            record["observation_time_iso"] = datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except (TypeError, ValueError, OSError):
            pass
    return records


def velocity_components_to_speed_heading(records, config):
    north_path = config.get("velocity_north_column")
    east_path = config.get("velocity_east_column")
    for record in records:
        try:
            north = float(_get(record, north_path))
            east = float(_get(record, east_path))
        except (TypeError, ValueError):
            continue
        record["entityComputedSpeed"] = math.hypot(north, east)
        record["entityComputedHeading"] = math.atan2(east, north) % (2.0 * math.pi)
    return records


def set_description(records, config):
    candidates = (
        "platformType", "emitterType", "unit", "unitEchelon",
        "kinematicallyInferredOrigin", "kinematicallyInferredDestination",
    )
    for record in records:
        parts = [str(value).strip() for path in candidates
                 if (value := _get(record, path)) not in (None, "")]
        record["set_description"] = " | ".join(parts)[:50] or "UNKNOWN"
    return records


def failed_destroyed(records, config):
    return records


def set_mode(records, config):
    for record in records:
        record["set_mode"] = "LIVE"
    return records


def set_standardIdentity(records, config):
    for record in records:
        record["set_standard"] = "ASSUMED_FRIEND"
    return records


def set_environment_air(records, config):
    for record in records:
        record["set_env"] = "AIR"
    return records
