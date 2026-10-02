"""Parity of the core modules with the baseline, using fixtures from generate.py."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

from crucible_entity_manager.core import geodesy, identity, kalman, partition, smoother
from crucible_entity_manager.core.records import clone_record

FIXTURES: Final = Path(__file__).resolve().parent.parent / "fixtures" / "parity"
RTOL: Final = 1e-9
T0: Final = datetime(2026, 9, 30, tzinfo=UTC)
ALTITUDE_ATOL_M: Final = 1e-4
"""The baseline altitude formula errs by up to ~0.1 mm below 50 km; the port is exact."""


def load(name: str) -> dict[str, Any]:
    with (FIXTURES / f"{name}.json").open() as handle:
        return json.load(handle)


KALMAN = load("kalman")
SMOOTHER = load("smoother")
CORE = load("core")


def matrix(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def assert_close(actual: np.ndarray, expected: list[Any], *, atol: float = 0.0) -> None:
    np.testing.assert_allclose(actual, matrix(expected), rtol=RTOL, atol=atol)


@pytest.mark.parametrize("case", KALMAN["predict"])
def test_predict(case: dict[str, Any]) -> None:
    start = kalman.GaussianState(matrix(case["mean"]), matrix(case["covariance"]), T0)
    predicted = kalman.predict(start, T0 + timedelta(seconds=case["dt"]), case["q"])
    assert_close(predicted.mean, case["expected_mean"])
    assert_close(predicted.covariance, case["expected_covariance"], atol=1e-6)


@pytest.mark.parametrize("case", KALMAN["update"])
def test_update(case: dict[str, Any]) -> None:
    mean, covariance = kalman.update(
        matrix(case["mean"]),
        matrix(case["covariance"]),
        matrix(case["measurement"]),
        matrix(case["noise"]),
        case["observed"],
    )
    assert_close(mean, case["expected_mean"])
    assert_close(covariance, case["expected_covariance"], atol=1e-6)


@pytest.mark.parametrize("case", KALMAN["covariance_intersection"])
def test_covariance_intersection(case: dict[str, Any]) -> None:
    fused = kalman.covariance_intersection(
        matrix(case["mean_a"]),
        matrix(case["cov_a"]),
        matrix(case["mean_b"]),
        matrix(case["cov_b"]),
        objective=kalman.CiObjective[case["objective"]],
        omega=case["omega"],
    )
    assert fused is not None
    assert_close(fused[0], case["expected_mean"])
    assert_close(fused[1], case["expected_covariance"], atol=1e-6)


@pytest.mark.parametrize("case", SMOOTHER["tracks"])
def test_smooth_track(case: dict[str, Any]) -> None:
    result = smoother.smooth_track(
        matrix(case["times"]),
        matrix(case["positions"]),
        matrix(case["velocities"]) if case["velocities"] is not None else None,
        matrix(case["position_covariances"]) if case["position_covariances"] is not None else None,
    )
    assert_close(result.times, case["expected_times"])
    assert_close(result.positions, case["expected_positions"])
    assert_close(result.velocities, case["expected_velocities"], atol=1e-9)
    assert_close(result.position_covariances, case["expected_position_covariances"], atol=1e-6)


@pytest.mark.parametrize("case", CORE["identity"])
def test_track_identity(case: dict[str, Any]) -> None:
    paths = case["paths"] or identity.DEFAULT_TRACK_ID_FIELDS
    custom_id = identity.identity_custom_id(case["record"], paths)
    assert custom_id == case["expected_custom_id"]
    assert identity.component_track_id("FEED", custom_id) == case["expected_component_track_id"]
    assert identity.principal_track_id(custom_id) == case["expected_principal_track_id"]


def test_stable_shard() -> None:
    for case in CORE["stable_shard"]:
        assert partition.stable_shard(case["key"], case["count"]) == case["expected"]


@pytest.mark.parametrize("case", CORE["report_ecef"])
def test_report_ecef_kinematics(case: dict[str, Any]) -> None:
    record = clone_record(case["input"])
    geodesy.set_report_ecef_kinematics([record])
    assert_records_close(record, case["expected"])


@pytest.mark.parametrize("case", CORE["track_geodetic"])
def test_track_geodetic(case: dict[str, Any]) -> None:
    record = clone_record(case["input"])
    geodesy.set_track_geodetic([record])
    assert_records_close(record, case["expected"])


def assert_records_close(actual: object, expected: object, path: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_records_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, float):
        atol = ALTITUDE_ATOL_M if path.endswith("altitude") else 1e-6
        assert isinstance(actual, float)
        assert math.isclose(actual, expected, rel_tol=RTOL, abs_tol=atol), path
    else:
        assert actual == expected, path
