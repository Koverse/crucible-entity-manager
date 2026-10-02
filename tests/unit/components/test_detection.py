from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crucible_entity_manager.components import detection
from crucible_entity_manager.components.detection import (
    PARAMS_BY_ENVIRONMENT,
    DetectionParams,
    TrackHistory,
    build_histories,
    candidate_pairs,
    choose_survivor,
    creation_time,
    evaluate_pair,
    evaluation_grid,
    find_duplicates,
    horizontal_mahalanobis,
    interpolate,
)
from crucible_entity_manager.core.aliases import JSONObject, JSONValue

T0 = datetime(2026, 9, 30, tzinfo=UTC)
ORIGIN = np.array([4_000_000.0, 1_000_000.0, 4_800_000.0])


def stamp(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def event(track: str, seconds: float, offset: float = 0.0, **extra: JSONValue) -> JSONObject:
    position = ORIGIN + np.array([offset + 10.0 * seconds, 0.0, 0.0])
    record: JSONObject = {
        "trackId": track,
        "interceptTimestamp": stamp(seconds),
        "environment": "SEA_SURFACE",
        "ecefPosition": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
        "ecefVelocity": {"x": 10.0, "y": 0.0, "z": 0.0},
    }
    return {**record, **extra}


def history(
    track: str, times: list[float], environment: str = "SEA_SURFACE", **positions: object
) -> TrackHistory:
    count = len(times)
    start = np.asarray(positions.get("start", ORIGIN), dtype=float)
    velocity = np.asarray(positions.get("velocity", [10.0, 0.0, 0.0]), dtype=float)
    return TrackHistory(
        track_id=track,
        environment=environment,
        times=np.array(times, dtype=float),
        positions=np.array([start + velocity * time for time in times]),
        velocities=np.tile(velocity, (count, 1)),
        covariances=np.stack([np.eye(3) * 100.0] * count),
        measurement_times=count,
    )


class TestBuildHistories:
    def test_builds_sorted_smoothed_histories(self) -> None:
        events = [event("a", seconds) for seconds in (30.0, 0.0, 10.0, 20.0)]
        (built,) = build_histories(events, set()).values()
        assert built.track_id == "a"
        assert built.environment == "SEA_SURFACE"
        relative = built.times - T0.timestamp()
        np.testing.assert_array_equal(relative, [0.0, 10.0, 20.0, 30.0])
        assert built.measurement_times == 4
        np.testing.assert_allclose(built.positions[:, 0], ORIGIN[0] + 10.0 * relative, atol=1.0)

    def test_excluded_and_unusable_events_are_skipped(self) -> None:
        events = [
            event("a", 0.0),
            event("a", 10.0),
            event("protected", 0.0),
            event("b", 5.0, ecefPosition={"x": 0, "y": 0, "z": 0}),
            event("c", 5.0, ecefPosition={"x": "north", "y": 1.0, "z": 1.0}),
            event("d", 5.0, interceptTimestamp="soon"),
            event("", 5.0),
            event("e", 5.0, ecefPosition={"x": float("inf"), "y": 1.0, "z": 1.0}),
            event("f", 5.0, ecefPosition={"x": {"value": 1.0}, "y": 1.0, "z": 1.0}),
        ]
        assert sorted(build_histories(events, {"protected"})) == ["a"]

    def test_column_presence_follows_the_baseline(self) -> None:
        assert build_histories([], set()) == {}
        assert build_histories([{"trackId": "a", "ecefPosition": {"x": 1.0}}], set()) == {}
        timeless = event("a", 0.0)
        del timeless["interceptTimestamp"]
        assert build_histories([timeless], set()) == {}
        created = event("a", 0.0)
        del created["interceptTimestamp"]
        created["crucibleHeader"] = {"createdDate": stamp(3.0)}
        (built,) = build_histories([created], set()).values()
        assert built.times[0] == pytest.approx((T0 + timedelta(seconds=3)).timestamp())

    def test_velocity_and_covariance_are_optional(self) -> None:
        bad_velocity = event("a", 0.0, ecefVelocity={"x": "fast", "y": 0.0, "z": 0.0})
        covariance: JSONObject = {"xx": 4.0, "xy": 0.0, "xz": 0.0, "yy": 4.0, "yz": 0.0, "zz": 4.0}
        events = [
            bad_velocity,
            event("a", 10.0, positionCovariance=covariance),
            event("a", 20.0, positionCovariance={**covariance, "zz": "x"}),
            event("b", 0.0, positionCovariance=covariance),
            event("b", 10.0, positionCovariance={**covariance, "xy": None}),
            event("c", 0.0, positionCovariance={**covariance, "xx": float("nan")}),
        ]
        histories = build_histories(events, set())
        assert set(histories) == {"a", "b", "c"}


def test_a_track_that_cannot_be_smoothed_keeps_its_raw_states(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def singular(*args: object) -> None:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(detection, "smooth_track", singular)
    events = [event("a", seconds) for seconds in (20.0, 0.0, 10.0)]
    (built,) = build_histories(events, set()).values()
    np.testing.assert_array_equal(built.times - T0.timestamp(), [0.0, 10.0, 20.0])
    np.testing.assert_allclose(built.positions[:, 0] - ORIGIN[0], [0.0, 100.0, 200.0])
    np.testing.assert_array_equal(built.covariances[0], np.eye(3) * 100.0**2)
    no_velocity = [{**item, "ecefVelocity": None} for item in events]
    (still,) = build_histories(no_velocity, set()).values()
    np.testing.assert_array_equal(still.velocities, np.zeros((3, 3)))
    assert "Smoothing track a failed (singular); using its raw states" in caplog.text


class TestEvaluation:
    def test_grid_and_interpolation(self) -> None:
        first = history("a", [0.0, 10.0, 20.0])
        second = history("b", [5.0, 15.0, 25.0])
        grid = evaluation_grid(first, second, 10.0, 5.0)
        assert grid is not None
        np.testing.assert_allclose(grid, [0.0, 12.5, 25.0])
        states = interpolate(first, np.array([-10.0, 4.0, 6.0, 26.0]), 5.0)
        assert states[0] is None
        assert states[3] is None
        assert states[1] is not None
        assert states[2] is not None
        np.testing.assert_allclose(states[1][0], ORIGIN + np.array([40.0, 0.0, 0.0]))
        np.testing.assert_allclose(states[1][2], np.eye(3) * (100.0 + 16.0))
        np.testing.assert_allclose(states[2][0], ORIGIN + np.array([60.0, 0.0, 0.0]))

    def test_no_overlap_gives_no_grid(self) -> None:
        assert (
            evaluation_grid(history("a", [0.0, 1.0]), history("b", [100.0, 101.0]), 1.0, 5.0)
            is None
        )

    def test_identical_tracks_are_duplicates(self) -> None:
        times = [float(seconds) for seconds in range(0, 200, 20)]
        duplicate = evaluate_pair(history("a", times), history("b", times), DetectionParams())
        assert duplicate is not None
        assert duplicate.mean_mahalanobis == pytest.approx(0.0)
        assert duplicate.confidence == pytest.approx(1.0)
        assert duplicate.time_overlap_seconds == pytest.approx(180.0)

    @pytest.mark.parametrize(
        ("second", "params"),
        [
            pytest.param(
                history("a", [0.0, 20.0, 40.0, 60.0, 80.0]), DetectionParams(), id="same track"
            ),
            pytest.param(
                history("b", [0.0, 20.0, 40.0, 60.0, 80.0], "AIR"),
                DetectionParams(),
                id="other environment",
            ),
            pytest.param(history("b", [0.0, 20.0]), DetectionParams(), id="too few times"),
            pytest.param(
                history("b", [1_000.0, 1_020.0, 1_040.0, 1_060.0, 1_080.0]),
                DetectionParams(),
                id="no overlap",
            ),
            pytest.param(
                history(
                    "b", [0.0, 20.0, 40.0, 60.0, 80.0], start=ORIGIN + np.array([0.0, 900.0, 0.0])
                ),
                DetectionParams(max_separation_m=500.0),
                id="too far apart",
            ),
            pytest.param(
                history(
                    "b", [0.0, 20.0, 40.0, 60.0, 80.0], start=ORIGIN + np.array([0.0, 200.0, 0.0])
                ),
                DetectionParams(),
                id="too many sigma",
            ),
            pytest.param(
                history("b", [0.0, 20.0, 40.0, 60.0, 80.0], velocity=[-10.0, 0.0, 0.0]),
                DetectionParams(distance_threshold_sigma=1e9),
                id="different velocity",
            ),
            pytest.param(
                history("b", [60.0, 80.0, 100.0, 120.0, 140.0]),
                DetectionParams(eval_interval_seconds=100.0),
                id="too few grid points",
            ),
        ],
    )
    def test_non_duplicates(self, second: TrackHistory, params: DetectionParams) -> None:
        first = history("a", [0.0, 20.0, 40.0, 60.0, 80.0])
        assert evaluate_pair(first, second, params) is None

    def test_stationary_tracks_ignore_velocity(self) -> None:
        still = history("a", [0.0, 20.0, 40.0, 60.0, 80.0], velocity=[0.0, 0.0, 0.0])
        other = history("b", [0.0, 20.0, 40.0, 60.0, 80.0], velocity=[0.0, 0.0, 0.0])
        duplicate = evaluate_pair(still, other, DetectionParams(velocity_threshold_mps=0.5))
        assert duplicate is not None
        assert duplicate.mean_velocity_difference == 0.0


class TestMahalanobis:
    def test_horizontal_distance_ignores_altitude(self) -> None:
        up = ORIGIN / np.linalg.norm(ORIGIN)
        assert horizontal_mahalanobis(
            ORIGIN, ORIGIN + up * 500.0, np.eye(3), np.eye(3)
        ) == pytest.approx(0.0, abs=1e-6)

    def test_noise_floor_and_singular_fallback(self) -> None:
        east = np.cross([0.0, 0.0, 1.0], ORIGIN / np.linalg.norm(ORIGIN))
        east /= np.linalg.norm(east)
        apart = ORIGIN + east * 30.0
        zero = np.zeros((3, 3))
        assert horizontal_mahalanobis(ORIGIN, apart, zero, zero, 15.0) == pytest.approx(
            30.0 / np.sqrt(2 * 15.0**2)
        )
        assert horizontal_mahalanobis(ORIGIN, apart, zero, zero) == pytest.approx(0.3)

    def test_near_the_origin_and_the_pole(self) -> None:
        assert (
            horizontal_mahalanobis(np.zeros(3), np.array([0.5, 0.0, 0.0]), np.eye(3), np.eye(3)) > 0
        )
        pole = np.array([0.0, 0.0, 6_356_752.0])
        assert horizontal_mahalanobis(
            pole, pole + np.array([3.0, 4.0, 0.0]), np.eye(3), np.eye(3)
        ) == pytest.approx(5.0 / np.sqrt(2.0))


class TestCandidates:
    def test_needs_enough_nearby_points_in_one_environment(self) -> None:
        times = [float(seconds) for seconds in range(0, 100, 10)]
        histories = {
            "a": history("a", times),
            "b": history("b", times, start=ORIGIN + np.array([0.0, 50.0, 0.0])),
            "c": history("c", times, "AIR"),
            "d": history("d", [0.0, 10.0], start=ORIGIN + np.array([0.0, 10.0, 0.0])),
            "e": history(
                "e",
                [5_000.0 + time for time in times],
                start=ORIGIN - np.array([50_000.0, 0.0, 0.0]),
            ),
        }
        assert candidate_pairs(
            histories, radius_m=100.0, time_alignment_seconds=5.0, min_points=5
        ) == {("a", "b")}

    def test_tracks_must_overlap_in_time(self) -> None:
        histories = {
            "a": history("a", [0.0, 1.0, 2.0], velocity=[0.0, 0.0, 0.0]),
            "b": history("b", [100.0, 101.0, 102.0], velocity=[0.0, 0.0, 0.0]),
        }
        assert (
            candidate_pairs(histories, radius_m=10.0, time_alignment_seconds=5.0, min_points=1)
            == set()
        )
        assert (
            candidate_pairs(
                {"a": histories["a"]}, radius_m=10.0, time_alignment_seconds=5.0, min_points=1
            )
            == set()
        )
        later_first = {"a": histories["b"], "b": histories["a"]}
        assert (
            candidate_pairs(later_first, radius_m=10.0, time_alignment_seconds=5.0, min_points=1)
            == set()
        )
        nan = history("n", [0.0], velocity=[0.0, 0.0, 0.0])
        nan.positions[0] = np.nan
        assert (
            candidate_pairs(
                {"n": nan, "a": history("a", [0.0])},
                radius_m=10.0,
                time_alignment_seconds=5.0,
                min_points=1,
            )
            == set()
        )

    def test_find_duplicates_uses_environment_parameters_and_ranks(self) -> None:
        times = [float(seconds) for seconds in range(0, 300, 30)]
        histories = {
            "a": history("a", times),
            "b": history("b", times, start=ORIGIN + np.array([0.0, 5.0, 0.0])),
            "x": history("x", times, "", start=ORIGIN + np.array([0.0, 20_000.0, 0.0])),
            "y": history("y", times[:6], "", start=ORIGIN + np.array([0.0, 20_005.0, 0.0])),
        }
        found = find_duplicates(histories, DetectionParams())
        assert [(item.track_id_1, item.track_id_2) for item in found] == [("a", "b"), ("x", "y")]
        assert found[0].confidence >= found[1].confidence
        assert "SEA_SURFACE" in PARAMS_BY_ENVIRONMENT


class TestSurvivor:
    def test_the_earlier_created_track_survives(self) -> None:
        created = {"a": T0, "b": T0 + timedelta(seconds=1)}
        assert choose_survivor("a", "b", created.get) == ("b", "a")
        assert choose_survivor("b", "a", created.get) == ("b", "a")

    def test_ties_and_unknowns_fall_back_to_the_lesser_id(self) -> None:
        same = {"a": T0, "b": T0}
        assert choose_survivor("b", "a", same.get) == ("b", "a")
        assert choose_survivor("a", "b", {}.get) == ("b", "a")

    def test_creation_time(self) -> None:
        assert creation_time({"trackOriginatedTimestamp": stamp(1.0)}) == T0 + timedelta(seconds=1)
        assert creation_time(
            {"trackOriginatedTimestamp": "x", "crucibleHeader": {"createdDate": stamp(2.0)}}
        ) == (T0 + timedelta(seconds=2))
        assert creation_time({}) is None
