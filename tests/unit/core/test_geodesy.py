import math

import numpy as np
import pytest

from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.geodesy import (
    SIGMA_95,
    WGS84_A,
    WGS84_E2,
    ecef_to_geodetic,
    ellipse_to_ecef_covariance,
    enu_to_ecef_rotation,
    geodetic_to_ecef,
    set_report_ecef_kinematics,
    set_track_geodetic,
)
from crucible_entity_manager.core.records import MISSING, get_path

WGS84_B = WGS84_A * math.sqrt(1.0 - WGS84_E2)
RNG = np.random.default_rng(20260930)


def arr(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


class TestCoordinateConversion:
    def test_equator_at_prime_meridian(self) -> None:
        x, y, z = geodetic_to_ecef(arr(0.0), arr(0.0), arr(0.0))
        assert (x[0], y[0], z[0]) == pytest.approx((WGS84_A, 0.0, 0.0))

    def test_north_pole(self) -> None:
        x, y, z = geodetic_to_ecef(arr(math.pi / 2), arr(0.0), arr(0.0))
        assert (x[0], y[0], z[0]) == pytest.approx((0.0, 0.0, WGS84_B), abs=1e-6)

    @pytest.mark.parametrize(("z", "latitude"), [(WGS84_B, math.pi / 2), (-WGS84_B, -math.pi / 2)])
    def test_poles_have_zero_altitude(self, z: float, latitude: float) -> None:
        back_latitude, _, altitude = ecef_to_geodetic(arr(0.0), arr(0.0), arr(z))
        assert back_latitude[0] == pytest.approx(latitude)
        assert altitude[0] == pytest.approx(0.0, abs=1e-6)

    def test_altitude_above_the_pole(self) -> None:
        _, _, altitude = ecef_to_geodetic(arr(0.0), arr(0.0), arr(WGS84_B + 10_000.0))
        assert altitude[0] == pytest.approx(10_000.0, abs=1e-6)

    def test_round_trip(self) -> None:
        latitude = RNG.uniform(-math.pi / 2, math.pi / 2, 500)
        longitude = RNG.uniform(-math.pi, math.pi, 500)
        altitude = RNG.uniform(-500.0, 15_000.0, 500)
        back = ecef_to_geodetic(*geodetic_to_ecef(latitude, longitude, altitude))
        np.testing.assert_allclose(back[0], latitude, atol=1e-9)
        np.testing.assert_allclose(back[1], longitude, atol=1e-9)
        np.testing.assert_allclose(back[2], altitude, atol=1e-3)


class TestCovariance:
    def test_rotations_are_proper_orthonormal(self) -> None:
        rotation = enu_to_ecef_rotation(RNG.uniform(-1.5, 1.5, 50), RNG.uniform(-3.1, 3.1, 50))
        identity = np.einsum("nji,njk->nik", rotation, rotation)
        np.testing.assert_allclose(identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0)

    def test_ellipse_axes_become_one_sigma_variances_along_the_axes(self) -> None:
        latitude, longitude, orientation = arr(0.7), arr(-1.2), arr(0.4)
        covariance = ellipse_to_ecef_covariance(
            latitude,
            longitude,
            major_axis_95=arr(490.0),
            minor_axis_95=arr(122.4),
            orientation=orientation,
            vertical_variance=25.0,
        )[0]
        rotation = enu_to_ecef_rotation(latitude, longitude)[0]
        major_direction = rotation @ np.array([math.sin(0.4), math.cos(0.4), 0.0])
        minor_direction = rotation @ np.array([math.cos(0.4), -math.sin(0.4), 0.0])
        up = rotation @ np.array([0.0, 0.0, 1.0])
        assert major_direction @ covariance @ major_direction == pytest.approx(
            (490.0 / SIGMA_95) ** 2
        )
        assert minor_direction @ covariance @ minor_direction == pytest.approx(
            (122.4 / SIGMA_95) ** 2
        )
        assert up @ covariance @ up == pytest.approx(25.0)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-9)


def report(**fields: JSONValue) -> JSONObject:
    record: JSONObject = {
        "estimatedKinematics": {"kinematicsTimestamp": "2026-09-30T12:00:00.000Z"},
        "geodetic": {"latitude": 0.6, "longitude": -1.3, "altitude": 100.0},
    }
    record.update(fields)
    return record


class TestSetReportEcefKinematics:
    def test_writes_position_matching_the_geodetic_fields(self) -> None:
        record = report()
        set_report_ecef_kinematics([record])
        x, y, z = geodetic_to_ecef(arr(0.6), arr(-1.3), arr(100.0))
        assert get_path(record, "ecefPosition") == pytest.approx({"x": x[0], "y": y[0], "z": z[0]})

    def test_missing_altitude_counts_as_zero(self) -> None:
        record = report(geodetic={"latitude": 0.6, "longitude": -1.3})
        set_report_ecef_kinematics([record])
        x, _, _ = geodetic_to_ecef(arr(0.6), arr(-1.3), arr(0.0))
        assert get_path(record, "ecefPosition.x") == pytest.approx(x[0])

    def test_velocity_is_horizontal_with_the_reported_speed(self) -> None:
        record = report(speed=50.0, heading=1.0)
        set_report_ecef_kinematics([record])
        velocity = np.array(
            [get_path(record, f"ecefVelocity.{axis}") for axis in "xyz"], dtype=float
        )
        up = enu_to_ecef_rotation(arr(0.6), arr(-1.3))[0] @ np.array([0.0, 0.0, 1.0])
        assert np.linalg.norm(velocity) == pytest.approx(50.0)
        assert velocity @ up == pytest.approx(0.0, abs=1e-9)
        assert get_path(record, "velocityCovariance.dxdx") is not MISSING

    @pytest.mark.parametrize("fields", [{"speed": 50.0}, {"heading": 1.0}, {}])
    def test_no_velocity_without_speed_and_heading(self, fields: dict[str, float]) -> None:
        record = report(**fields)
        set_report_ecef_kinematics([record])
        assert get_path(record, "ecefVelocity") is MISSING
        assert get_path(record, "velocityCovariance") is MISSING

    def test_position_covariance_requires_a_major_axis(self) -> None:
        with_ellipse = report(uncertainty={"uncertaintyEllipse": {"semiMajorAxisLength": 300.0}})
        without = report()
        set_report_ecef_kinematics([with_ellipse, without])
        assert get_path(with_ellipse, "positionCovariance.xx") is not MISSING
        assert get_path(without, "positionCovariance") is MISSING

    @pytest.mark.parametrize(
        "record",
        [
            {"geodetic": {"latitude": 0.6, "longitude": -1.3}},
            report(estimatedKinematics={"kinematicsTimestamp": ""}),
            report(geodetic={"latitude": 0.6}),
            report(geodetic={"latitude": "north", "longitude": -1.3}),
            report(geodetic={"latitude": [0.6], "longitude": -1.3}),
        ],
    )
    def test_leaves_unusable_reports_unchanged(self, record: JSONObject) -> None:
        before = repr(record)
        set_report_ecef_kinematics([record])
        assert repr(record) == before

    def test_keeps_unrelated_fields_of_existing_objects(self) -> None:
        record = report(ecefPosition={"source": "sensor"})
        set_report_ecef_kinematics([record])
        assert get_path(record, "ecefPosition.source") == "sensor"


class TestSetTrackGeodetic:
    def test_writes_radians_and_meters(self) -> None:
        x, y, z = geodetic_to_ecef(arr(0.6), arr(-1.3), arr(250.0))
        record: JSONObject = {
            "ecefPosition": {"x": float(x[0]), "y": float(y[0]), "z": float(z[0])}
        }
        set_track_geodetic([record])
        assert get_path(record, "geodetic") == pytest.approx(
            {"latitude": 0.6, "longitude": -1.3, "altitude": 250.0}
        )

    @pytest.mark.parametrize(
        "record",
        [
            {},
            {"ecefPosition": {"x": 1.0, "y": 2.0}},
            {"ecefPosition": {"x": None, "y": 2.0, "z": 3.0}},
        ],
    )
    def test_skips_records_without_a_full_position(self, record: JSONObject) -> None:
        set_track_geodetic([record])
        assert get_path(record, "geodetic") is MISSING
