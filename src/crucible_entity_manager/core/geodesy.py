"""WGS84 geodesy and the kinematic enrichment of report and track records.

Angles are in radians throughout, including the geodetic fields written to
records. Distances are in meters, velocities in meters per second.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np

from crucible_entity_manager.core.aliases import FloatArray, JSONObject, JSONValue
from crucible_entity_manager.core.records import MISSING, Missing, get_path, set_path

WGS84_A: Final = 6_378_137.0
"""WGS84 semi-major axis, in meters."""

WGS84_E2: Final = 6.6943799901413165e-3
"""WGS84 first eccentricity squared."""

SIGMA_95: Final = 2.448
"""Ratio of a 2-D 95% confidence ellipse axis to its one-sigma axis."""

DEFAULT_ELLIPSE_AXIS_M: Final = 100.0
"""Ellipse axis assumed when a report gives none."""

REPORT_VERTICAL_VARIANCE_M2: Final = 10_000.0
"""Vertical position variance assumed for reports, which carry 2-D ellipses."""

VELOCITY_ELLIPSE_AXES_MPS: Final = (2.0, 1.0)
"""95% axes (along-track, cross-track) of the velocity uncertainty for reports."""

VELOCITY_VERTICAL_VARIANCE: Final = 4.0
"""Vertical velocity variance assumed for reports, in (m/s)^2."""

_UPPER_TRIANGLE: Final = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
POSITION_COVARIANCE_FIELDS: Final = ("xx", "xy", "xz", "yy", "yz", "zz")
VELOCITY_COVARIANCE_FIELDS: Final = ("dxdx", "dxdy", "dxdz", "dydy", "dydz", "dzdz")


def geodetic_to_ecef(
    latitude: FloatArray, longitude: FloatArray, altitude: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert geodetic coordinates to ECEF ``(x, y, z)``."""
    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (prime_vertical + altitude) * cos_lat * np.cos(longitude)
    y = (prime_vertical + altitude) * cos_lat * np.sin(longitude)
    z = (prime_vertical * (1.0 - WGS84_E2) + altitude) * sin_lat
    return x, y, z


def ecef_to_geodetic(
    x: FloatArray, y: FloatArray, z: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert ECEF coordinates to geodetic ``(latitude, longitude, altitude)``.

    Latitude uses Bowring's closed-form approximation, accurate to well under a
    millimeter for terrestrial and airborne positions. Altitude uses the height
    formula that stays exact on the polar axis.
    """
    semi_minor = WGS84_A * np.sqrt(1.0 - WGS84_E2)
    second_eccentricity_sq = (WGS84_A**2 - semi_minor**2) / semi_minor**2
    horizontal = np.hypot(x, y)
    theta = np.arctan2(WGS84_A * z, semi_minor * horizontal)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(
        z + second_eccentricity_sq * semi_minor * np.sin(theta) ** 3,
        horizontal - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    sin_lat = np.sin(latitude)
    altitude = (
        horizontal * np.cos(latitude)
        + z * sin_lat
        - WGS84_A * np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    )
    return latitude, longitude, altitude


def enu_to_ecef_rotation(latitude: FloatArray, longitude: FloatArray) -> FloatArray:
    """Return the ``(n, 3, 3)`` rotations from local East-North-Up to ECEF."""
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    rotation = np.zeros((len(latitude), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = -sin_lon
    rotation[:, 0, 1] = -sin_lat * cos_lon
    rotation[:, 0, 2] = cos_lat * cos_lon
    rotation[:, 1, 0] = cos_lon
    rotation[:, 1, 1] = -sin_lat * sin_lon
    rotation[:, 1, 2] = cos_lat * sin_lon
    rotation[:, 2, 1] = cos_lat
    rotation[:, 2, 2] = sin_lat
    return rotation


def ellipse_to_ecef_covariance(  # noqa: PLR0913 - each argument is one physical quantity
    latitude: FloatArray,
    longitude: FloatArray,
    *,
    major_axis_95: FloatArray,
    minor_axis_95: FloatArray,
    orientation: FloatArray,
    vertical_variance: float,
) -> FloatArray:
    """Convert 95% horizontal error ellipses to ``(n, 3, 3)`` ECEF covariances.

    Args:
        latitude: Geodetic latitude of each ellipse center.
        longitude: Longitude of each ellipse center.
        major_axis_95: Major axis length of each 95% ellipse.
        minor_axis_95: Minor axis length of each 95% ellipse.
        orientation: Azimuth of each major axis, clockwise from north.
        vertical_variance: Variance assigned to the local vertical axis.
    """
    major_sigma = major_axis_95 / SIGMA_95
    minor_sigma = minor_axis_95 / SIGMA_95
    sin_o, cos_o = np.sin(orientation), np.cos(orientation)
    local = np.zeros((len(latitude), 3, 3), dtype=np.float64)
    local[:, 0, 0] = major_sigma**2 * sin_o**2 + minor_sigma**2 * cos_o**2
    local[:, 0, 1] = (major_sigma**2 - minor_sigma**2) * sin_o * cos_o
    local[:, 1, 0] = local[:, 0, 1]
    local[:, 1, 1] = major_sigma**2 * cos_o**2 + minor_sigma**2 * sin_o**2
    local[:, 2, 2] = vertical_variance
    rotation = enu_to_ecef_rotation(latitude, longitude)
    return np.einsum("nij,njk,nlk->nil", rotation, local, rotation)


def set_report_ecef_kinematics(records: Sequence[JSONObject]) -> None:
    """Add ECEF position, velocity and covariances to report records, in place.

    Only reports with a finite latitude and longitude and a
    ``estimatedKinematics.kinematicsTimestamp`` are enriched:

    - ``ecefPosition`` is always written. A missing altitude counts as 0.
    - ``ecefVelocity`` and ``velocityCovariance`` are written when ``speed`` and
      ``heading`` are finite. Heading is clockwise from north.
    - ``positionCovariance`` is written when the uncertainty ellipse has a
      major axis. A missing minor axis or orientation defaults to
      `DEFAULT_ELLIPSE_AXIS_M` or 0.
    """
    selected = [
        record
        for record in records
        if get_path(record, "estimatedKinematics.kinematicsTimestamp") not in (MISSING, None, "")
        and np.isfinite(
            _number(get_path(record, "geodetic.latitude"))
            + _number(get_path(record, "geodetic.longitude"))
        )
    ]
    if not selected:
        return

    latitude = _numbers(selected, "geodetic.latitude")
    longitude = _numbers(selected, "geodetic.longitude")
    altitude = np.nan_to_num(_numbers(selected, "geodetic.altitude", default=0.0), nan=0.0)
    x, y, z = geodetic_to_ecef(latitude, longitude, altitude)

    speed = _numbers(selected, "speed")
    heading = _numbers(selected, "heading")
    velocity_valid = np.isfinite(speed) & np.isfinite(heading)
    rotation = enu_to_ecef_rotation(latitude, longitude)
    enu_velocity = np.stack(
        [speed * np.sin(heading), speed * np.cos(heading), np.zeros(len(selected))], axis=1
    )
    ecef_velocity = np.einsum("nij,nj->ni", rotation, enu_velocity)
    major_along, minor_cross = VELOCITY_ELLIPSE_AXES_MPS
    velocity_covariance = ellipse_to_ecef_covariance(
        latitude,
        longitude,
        major_axis_95=np.full(len(selected), major_along),
        minor_axis_95=np.full(len(selected), minor_cross),
        orientation=np.nan_to_num(heading),
        vertical_variance=VELOCITY_VERTICAL_VARIANCE,
    )

    major = _numbers(selected, "uncertainty.uncertaintyEllipse.semiMajorAxisLength")
    minor = _numbers(
        selected,
        "uncertainty.uncertaintyEllipse.semiMinorAxisLength",
        default=DEFAULT_ELLIPSE_AXIS_M,
    )
    orientation = _numbers(selected, "uncertainty.uncertaintyEllipse.orientation", default=0.0)
    position_covariance_valid = np.isfinite(major)
    position_covariance = ellipse_to_ecef_covariance(
        latitude,
        longitude,
        major_axis_95=np.nan_to_num(major, nan=DEFAULT_ELLIPSE_AXIS_M),
        minor_axis_95=np.nan_to_num(minor, nan=DEFAULT_ELLIPSE_AXIS_M),
        orientation=np.nan_to_num(orientation),
        vertical_variance=REPORT_VERTICAL_VARIANCE_M2,
    )

    for index, record in enumerate(selected):
        for axis, value in zip("xyz", (x[index], y[index], z[index]), strict=True):
            set_path(record, f"ecefPosition.{axis}", float(value))
        if velocity_valid[index]:
            for axis, value in zip("xyz", ecef_velocity[index], strict=True):
                set_path(record, f"ecefVelocity.{axis}", float(value))
            _set_covariance(
                record, "velocityCovariance", VELOCITY_COVARIANCE_FIELDS, velocity_covariance[index]
            )
        if position_covariance_valid[index]:
            _set_covariance(
                record, "positionCovariance", POSITION_COVARIANCE_FIELDS, position_covariance[index]
            )


def set_track_geodetic(records: Sequence[JSONObject]) -> None:
    """Derive ``geodetic`` latitude, longitude and altitude from ``ecefPosition``, in place.

    Records without a finite ``ecefPosition`` are left unchanged.
    """
    selected = [
        record
        for record in records
        if all(np.isfinite(_number(get_path(record, f"ecefPosition.{axis}"))) for axis in "xyz")
    ]
    if not selected:
        return
    latitude, longitude, altitude = ecef_to_geodetic(
        _numbers(selected, "ecefPosition.x"),
        _numbers(selected, "ecefPosition.y"),
        _numbers(selected, "ecefPosition.z"),
    )
    for index, record in enumerate(selected):
        set_path(record, "geodetic.latitude", float(latitude[index]))
        set_path(record, "geodetic.longitude", float(longitude[index]))
        set_path(record, "geodetic.altitude", float(altitude[index]))


def _number(value: JSONValue | Missing, default: float = np.nan) -> float:
    """Read a JSON value as a float; booleans, missing and invalid values give `default`."""
    if value is MISSING or value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _numbers(records: Sequence[JSONObject], path: str, default: float = np.nan) -> FloatArray:
    return np.asarray([_number(get_path(record, path), default) for record in records])


def _set_covariance(
    record: JSONObject, prefix: str, names: Sequence[str], covariance: FloatArray
) -> None:
    """Write the six unique entries of a symmetric 3x3 covariance under `prefix`."""
    for name, (row, col) in zip(names, _UPPER_TRIANGLE, strict=True):
        set_path(record, f"{prefix}.{name}", float(covariance[row, col]))
