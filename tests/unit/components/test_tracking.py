from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crucible_entity_manager.components.keyed import KeyedState
from crucible_entity_manager.components.tracking import (
    DEFAULT_PROCESS_NOISE,
    INITIAL_VARIANCE,
    NOISE_FLOOR,
    NOISE_LIMIT,
    Fusion,
    KalmanTracker,
    Measurement,
    Outcome,
    TrackerParams,
    TrackState,
    environment_of,
    head_from_event,
    kalman_track_event,
    measurement_from_report,
    passthrough_track_event,
    state_from_head,
)
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.kalman import (
    CiObjective,
    GaussianState,
    covariance_intersection,
    predict,
)
from crucible_entity_manager.core.records import get_path

T0 = datetime(2026, 9, 30, tzinfo=UTC)
POSITION: JSONObject = {"x": 1_000.0, "y": 2_000.0, "z": 3_000.0}


def report(**extra: JSONValue) -> JSONObject:
    return {"ecefPosition": dict(POSITION), **extra}


def tracker(
    fusion: Fusion = Fusion.KALMAN, q: float = DEFAULT_PROCESS_NOISE
) -> tuple[KalmanTracker, KeyedState[TrackState]]:
    tracks: KeyedState[TrackState] = KeyedState(TrackState)
    return KalmanTracker(TrackerParams(fusion, q), tracks, "[test] "), tracks


def position_measurement(seconds: float, x: float = 1_000.0) -> Measurement:
    return Measurement(
        np.array([x, 2_000.0, 3_000.0]), np.eye(3) * 100.0, T0 + timedelta(seconds=seconds)
    )


class TestMeasurementFromReport:
    def test_position_only_uses_defaults_for_missing_covariance(self) -> None:
        measurement = measurement_from_report(report(), T0, 25.0)
        assert measurement is not None
        assert measurement.observed == (0, 2, 4)
        np.testing.assert_array_equal(measurement.values, [1_000.0, 2_000.0, 3_000.0])
        np.testing.assert_array_equal(measurement.noise, np.eye(3) * INITIAL_VARIANCE)

    def test_position_covariance_defaults_apply_per_term(self) -> None:
        measurement = measurement_from_report(
            report(positionCovariance={"xx": 4.0, "xy": 1.0, "yz": "2"}), T0, 25.0
        )
        assert measurement is not None
        np.testing.assert_array_equal(
            measurement.noise,
            [[4.0, 1.0, 0.0], [1.0, INITIAL_VARIANCE, 2.0], [0.0, 2.0, INITIAL_VARIANCE]],
        )

    def test_full_velocity_makes_a_six_state_measurement(self) -> None:
        measurement = measurement_from_report(
            report(ecefVelocity={"x": 1.0, "y": 2.0, "z": 3.0}), T0, 25.0
        )
        assert measurement is not None
        assert measurement.observed == (0, 1, 2, 3, 4, 5)
        np.testing.assert_array_equal(
            measurement.values, [1_000.0, 1.0, 2_000.0, 2.0, 3_000.0, 3.0]
        )
        np.testing.assert_array_equal(np.diag(measurement.noise)[1::2], [25.0, 25.0, 25.0])

    def test_velocity_noise_is_sanitized(self) -> None:
        measurement = measurement_from_report(
            report(
                ecefVelocity={"x": 1.0, "y": 2.0, "z": 3.0},
                velocityCovariance={"dxdx": -1.0, "dydy": 5e10, "dzdz": 4.0, "dxdy": 1e12},
            ),
            T0,
            25.0,
        )
        assert measurement is not None
        assert np.diag(measurement.noise)[1::2].tolist() == [NOISE_FLOOR, NOISE_LIMIT, 4.0]
        assert measurement.noise[1, 3] == 1e10

    def test_partial_velocity_is_ignored(self) -> None:
        measurement = measurement_from_report(report(ecefVelocity={"x": 1.0}), T0, 25.0)
        assert measurement is not None
        assert measurement.values.shape == (3,)

    @pytest.mark.parametrize(
        "position",
        [{"x": 1.0, "y": 2.0}, {"x": 1.0, "y": "north", "z": 3.0}, {"x": None, "y": 2.0, "z": 3.0}],
    )
    def test_an_incomplete_position_gives_no_measurement(self, position: JSONObject) -> None:
        assert measurement_from_report({"ecefPosition": position}, T0, 25.0) is None
        assert (
            measurement_from_report(
                {"ecefPosition": position, "ecefVelocity": {"x": 1.0, "y": 1.0, "z": 1.0}}, T0, 25.0
            )
            is None
        )


class TestKalmanTracker:
    def test_a_new_track_starts_and_updates_with_its_first_measurement(self) -> None:
        kalman, _ = tracker()
        outcome, state = kalman.apply("t", position_measurement(0))
        assert outcome is Outcome.UPDATED
        assert state.timestamp == T0
        assert state.covariance[0, 0] == pytest.approx(
            100.0 * INITIAL_VARIANCE / (100.0 + INITIAL_VARIANCE)
        )

    def test_a_stale_measurement_changes_nothing(self) -> None:
        kalman, tracks = tracker()
        kalman.apply("t", position_measurement(10))
        before = tracks.get("t").prior
        outcome, state = kalman.apply("t", position_measurement(5, x=9_999.0))
        assert outcome is Outcome.STALE
        assert state is before

    def test_a_long_gap_resets_to_a_fresh_prior(self, caplog: pytest.LogCaptureFixture) -> None:
        kalman, _ = tracker()
        kalman.apply("t", position_measurement(0))
        outcome, state = kalman.apply("t", position_measurement(16 * 60, x=5_000.0))
        assert outcome is Outcome.RESET
        assert state.mean[0] == 5_000.0
        assert state.covariance[0, 0] == INITIAL_VARIANCE
        assert "Resetting the filter for track t: large time jump" in caplog.text

    @pytest.mark.parametrize(
        ("prior", "reason"),
        [
            (GaussianState(np.full(6, np.nan), np.eye(6), T0), "state contains NaN"),
            (GaussianState(np.zeros(6), np.full((6, 6), np.nan), T0), "state contains NaN"),
            (GaussianState(np.zeros(6), np.eye(6) * 1e16, T0), "covariance explosion"),
            (GaussianState(np.zeros(6), np.diag([np.inf] * 6), T0), "prediction produced NaN"),
        ],
    )
    def test_numerical_trouble_resets(
        self, prior: GaussianState, reason: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        kalman, tracks = tracker()
        tracks.get("t").prior = prior
        outcome, _ = kalman.apply("t", position_measurement(1))
        assert outcome is Outcome.RESET
        assert reason in caplog.text

    def test_a_singular_update_resets(self, caplog: pytest.LogCaptureFixture) -> None:
        kalman, tracks = tracker()
        tracks.get("t").prior = GaussianState(np.zeros(6), np.zeros((6, 6)), T0)
        singular = Measurement(np.array([1.0, 2.0, 3.0]), np.zeros((3, 3)), T0)
        outcome, _ = kalman.apply("t", singular)
        assert outcome is Outcome.RESET
        assert "singular update" in caplog.text

    def test_a_nan_update_resets(self, caplog: pytest.LogCaptureFixture) -> None:
        kalman, _ = tracker()
        kalman.apply("t", position_measurement(0))
        poisoned = Measurement(np.array([np.nan, 2.0, 3.0]), np.eye(3), T0 + timedelta(seconds=1))
        outcome, state = kalman.apply("t", poisoned)
        assert outcome is Outcome.RESET
        assert state.mean[0] == 0.0
        assert "update produced NaN" in caplog.text

    def test_process_noise_follows_the_first_reported_environment(self) -> None:
        kalman, tracks = tracker(q=7.0)
        kalman.note_environment("air", {"identity": {"environment": "air"}})
        kalman.note_environment("air", {"identity": {"environment": "SPACE"}})
        kalman.note_environment("other", {"environment": "MARS"})
        kalman.note_environment("none", {})
        for track in ("air", "other", "none"):
            kalman.apply(track, position_measurement(0))
        assert tracks.get("air").process_noise == 5.0
        assert tracks.get("other").process_noise == 7.0
        assert tracks.get("none").process_noise == 7.0
        assert kalman.velocity_variance("air") == 25.0
        assert kalman.velocity_variance("unknown-track") == 25.0

    def test_a_restored_track_takes_its_noise_lazily(self) -> None:
        kalman, tracks = tracker()
        head: JSONObject = {
            "ecefPosition": dict(POSITION),
            "identity": {"environment": "SEA_SURFACE"},
        }
        kalman.restore("t", head, T0)
        assert tracks.get("t").process_noise is None
        outcome, _ = kalman.apply("t", position_measurement(1))
        assert outcome is Outcome.UPDATED
        assert tracks.get("t").process_noise == 0.5

    def test_covariance_intersection_lifts_position_measurements(self) -> None:
        kalman, tracks = tracker(Fusion.COVARIANCE_INTERSECTION)
        prior = GaussianState(np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0]), np.eye(6) * 1_000.0, T0)
        tracks.get("t").prior = prior
        kalman.note_environment("t", {"environment": "GROUND"})
        tracks.get("t").process_noise = 3.0
        measurement = position_measurement(1)
        outcome, state = kalman.apply("t", measurement)
        prediction = predict(prior, measurement.timestamp, 3.0)
        lifted_mean = np.array(
            [1_000.0, prediction.mean[1], 2_000.0, prediction.mean[3], 3_000.0, prediction.mean[5]]
        )
        lifted_noise = np.diag([100.0, 9.0, 100.0, 9.0, 100.0, 9.0])
        expected = covariance_intersection(
            prediction.mean,
            prediction.covariance,
            lifted_mean,
            lifted_noise,
            objective=CiObjective.FULL_TRACE,
        )
        assert expected is not None
        assert outcome is Outcome.UPDATED
        np.testing.assert_allclose(state.mean, expected[0], rtol=1e-12)
        np.testing.assert_allclose(state.covariance, expected[1], rtol=1e-12)

    def test_covariance_intersection_falls_back_to_kalman_when_singular(self) -> None:
        kalman, tracks = tracker(Fusion.COVARIANCE_INTERSECTION)
        tracks.get("t").prior = GaussianState(np.zeros(6), np.zeros((6, 6)), T0)
        measurement = position_measurement(0)
        outcome, state = kalman.apply("t", measurement)
        assert outcome is Outcome.UPDATED
        np.testing.assert_array_equal(state.covariance, np.zeros((6, 6)))


def simulate(
    kalman: KalmanTracker,
    *,
    steps: int = 60,
    noise_m: float = 30.0,
    with_velocity: bool | None = False,
    seed: int = 7,
) -> tuple[list[GaussianState], np.ndarray, np.ndarray]:
    """Track a constant-velocity target; return posteriors, truths and measurements.

    `with_velocity` None alternates position-only and full measurements.
    """
    rng = np.random.default_rng(seed)
    velocity = np.array([12.0, -4.0, 1.5])
    start = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
    posteriors: list[GaussianState] = []
    truths, measured = [], []
    for step in range(steps):
        truth = start + velocity * step
        observed = truth + rng.normal(0.0, noise_m, 3)
        full = with_velocity if with_velocity is not None else step % 2 == 0
        if full:
            values = np.array(
                [observed[0], velocity[0], observed[1], velocity[1], observed[2], velocity[2]]
            )
            noise = np.diag([noise_m**2, 1.0] * 3)
        else:
            values, noise = observed, np.eye(3) * noise_m**2
        _, posterior = kalman.apply("t", Measurement(values, noise, T0 + timedelta(seconds=step)))
        posteriors.append(posterior)
        truths.append(truth)
        measured.append(observed)
    return posteriors, np.array(truths), np.array(measured)


def rmse(estimates: np.ndarray, truths: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((estimates - truths) ** 2, axis=1))))


class TestBehavior:
    """Behavioral cases carried over from the baseline's tracker test suites."""

    @pytest.mark.parametrize(
        ("fusion", "with_velocity"),
        [
            (Fusion.KALMAN, False),
            (Fusion.KALMAN, True),
            (Fusion.KALMAN, None),
            (Fusion.COVARIANCE_INTERSECTION, False),
            (Fusion.COVARIANCE_INTERSECTION, True),
        ],
    )
    def test_filtered_error_is_below_measurement_error(
        self, *, fusion: Fusion, with_velocity: bool | None
    ) -> None:
        """Covariance intersection is conservative, so it is held to this only with one
        kind of measurement, as the baseline's tests were."""
        kalman, _ = tracker(fusion)
        posteriors, truths, measured = simulate(kalman, with_velocity=with_velocity)
        estimates = np.array([state.mean[[0, 2, 4]] for state in posteriors])
        settled = slice(10, None)
        assert rmse(estimates[settled], truths[settled]) < rmse(measured[settled], truths[settled])

    def test_velocity_is_inferred_from_positions(self) -> None:
        kalman, _ = tracker()
        posteriors, _, _ = simulate(kalman, steps=80, noise_m=5.0)
        velocity = posteriors[-1].mean[[1, 3, 5]]
        assert velocity[0] > 0 > velocity[1]
        np.testing.assert_allclose(velocity, [12.0, -4.0, 1.5], atol=2.5)

    def test_uncertainty_converges(self) -> None:
        kalman, _ = tracker()
        posteriors, _, _ = simulate(kalman)
        traces = [float(np.trace(state.covariance)) for state in posteriors]
        assert traces[-1] < traces[0] / 1_000
        assert max(traces[-10:]) / min(traces[-10:]) < 1.5

    def test_covariance_intersection_is_more_conservative(self) -> None:
        kalman, _ = tracker(Fusion.KALMAN)
        intersection, _ = tracker(Fusion.COVARIANCE_INTERSECTION)
        kalman_states, _, _ = simulate(kalman)
        ci_states, _, _ = simulate(intersection)
        assert np.trace(ci_states[-1].covariance) > np.trace(kalman_states[-1].covariance)

    @pytest.mark.parametrize(
        ("environment", "process_noise", "velocity_variance"),
        [
            ("AIR", 5.0, 25.0),
            ("GROUND", 3.0, 9.0),
            ("SEA_SURFACE", 0.5, 4.0),
            ("SEA_SUBSURFACE", 0.1, 4.0),
            ("SPACE", 0.001, 100.0),
            ("UNKNOWN", 3.0, 25.0),
            ("sea_surface", 0.5, 4.0),
            ("LAVA", DEFAULT_PROCESS_NOISE, 25.0),
        ],
    )
    def test_environment_noise_table(
        self, environment: str, process_noise: float, velocity_variance: float
    ) -> None:
        kalman, tracks = tracker()
        kalman.note_environment("t", {"identity": {"environment": environment}})
        kalman.apply("t", position_measurement(0))
        assert tracks.get("t").process_noise == process_noise
        assert kalman.velocity_variance("t") == velocity_variance


class TestHeads:
    def test_state_from_a_head(self) -> None:
        head: JSONObject = {
            "ecefPosition": dict(POSITION),
            "ecefVelocity": {"x": 1.0},
            "positionCovariance": {"xx": 4.0, "xy": 1.0},
            "positionVelocityCovariance": {"xdx": 0.5, "zdy": 0.25},
        }
        state = state_from_head(head, T0)
        np.testing.assert_array_equal(state.mean, [1_000.0, 1.0, 2_000.0, 0.0, 3_000.0, 0.0])
        assert state.covariance[0, 0] == 4.0
        assert state.covariance[0, 2] == state.covariance[2, 0] == 1.0
        assert state.covariance[0, 1] == state.covariance[1, 0] == 0.5
        assert state.covariance[4, 3] == state.covariance[3, 4] == 0.25
        assert state.covariance[2, 2] == INITIAL_VARIANCE

    def test_a_head_without_a_position_starts_at_the_origin(self) -> None:
        state = state_from_head({"ecefPosition": {"x": 1.0, "y": 2.0}}, T0)
        np.testing.assert_array_equal(state.mean, np.zeros(6))

    def test_nan_in_a_head_is_replaced(self) -> None:
        head: JSONObject = {
            "ecefPosition": {"x": "bad", "y": float("nan"), "z": 3.0},
            "positionCovariance": {"xx": float("nan")},
        }
        state = state_from_head(head, T0)
        np.testing.assert_array_equal(state.mean, [0.0, 0.0, 0.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(state.covariance, np.eye(6) * INITIAL_VARIANCE)

    def test_head_from_event(self) -> None:
        event: JSONObject = {
            "trackId": "t",
            "interceptTimestamp": "T",
            "trackQuality": 0.5,
            "x": {"y": 1},
        }
        head = head_from_event(event, "STALE")
        assert head == {
            "trackId": "t",
            "trackUpdatedTimestamp": "T",
            "x": {"y": 1},
            "stale": "STALE",
        }
        assert head["x"] is not event["x"]


class TestTrackEvents:
    def test_kalman_event_fields(self) -> None:
        source: JSONObject = {
            "identity": {"standard": "FRIEND", "environment": "AIR", "callsign": "A"},
            "crucibleHeader": {"createdDate": "C"},
            "trackQuality": 0.9,
            "estimatedKinematics": {"kinematicsTimestamp": "T"},
            "mode": "LIVE",
            "edhControlSet": {"classification": "U"},
        }
        covariance = np.arange(36.0).reshape(6, 6)
        event = kalman_track_event(source, "t", GaussianState(np.arange(6.0), covariance, T0))
        assert event["trackId"] == "t"
        assert event["standardIdentity"] == "FRIEND"
        assert event["environment"] == "AIR"
        assert event["trackOriginatedTimestamp"] == "C"
        assert event["interceptTimestamp"] == "T"
        assert event["identity"] == source["identity"]
        assert event["identity"] is not source["identity"]
        assert event["ecefPosition"] == {"x": 0.0, "y": 2.0, "z": 4.0}
        assert event["ecefVelocity"] == {"x": 1.0, "y": 3.0, "z": 5.0}
        assert event["positionCovariance"] == {
            "xx": 0.0,
            "xy": 2.0,
            "xz": 4.0,
            "yy": 14.0,
            "yz": 16.0,
            "zz": 28.0,
        }
        assert get_path(event, "positionVelocityCovariance.zdy") == covariance[4, 3]

    def test_kalman_event_without_identity(self) -> None:
        event = kalman_track_event({}, "t", GaussianState(np.zeros(6), np.eye(6), T0))
        assert event["identity"] == {}
        assert event["standardIdentity"] is None

    def test_passthrough_copies_kinematics(self) -> None:
        source: JSONObject = {
            "identity": {"callsign": "A"},
            "geodetic": {"latitude": 0.1},
            "ecefPosition": dict(POSITION),
            "uncertainty": "not an object",
            "speed": 3.0,
        }
        event = passthrough_track_event(source, "t")
        assert event["geodetic"] == {"latitude": 0.1}
        assert event["ecefPosition"] == POSITION
        assert event["speed"] == 3.0
        assert "uncertainty" not in event
        assert "heading" not in event


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"identity": {"environment": "AIR"}, "environment": "SEA"}, "AIR"),
        ({"identity": {"environment": ""}, "environment": "SEA"}, "SEA"),
        ({}, None),
    ],
)
def test_environment_of(record: JSONObject, expected: str | None) -> None:
    assert environment_of(record) == expected
