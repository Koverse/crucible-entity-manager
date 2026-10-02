from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crucible_entity_manager.core import kalman
from crucible_entity_manager.core.kalman import (
    CiObjective,
    GaussianState,
    constant_velocity,
    covariance_intersection,
    predict,
    update,
)

T0 = datetime(2026, 9, 30, tzinfo=UTC)
RNG = np.random.default_rng(42)


def spd(size: int = 6, scale: float = 10.0) -> np.ndarray:
    factor = RNG.normal(size=(size, size))
    return scale * (factor @ factor.T) + np.eye(size)


class TestConstantVelocity:
    def test_positions_advance_by_velocity_times_dt(self) -> None:
        transition, _ = constant_velocity(2.0, 1.0)
        state = np.array([0.0, 3.0, 10.0, -1.0, 5.0, 0.5])
        assert transition @ state == pytest.approx([6.0, 3.0, 8.0, -1.0, 6.0, 0.5])

    def test_process_noise_blocks_for_white_acceleration(self) -> None:
        _, noise = constant_velocity(2.0, 3.0)
        block = 3.0 * np.array([[8.0 / 3.0, 2.0], [2.0, 2.0]])
        for start in (0, 2, 4):
            np.testing.assert_allclose(noise[start : start + 2, start : start + 2], block)
        assert noise[0, 2] == noise[1, 3] == 0.0

    def test_backward_prediction_still_adds_uncertainty(self) -> None:
        _, forward = constant_velocity(5.0, 1.0)
        _, backward = constant_velocity(-5.0, 1.0)
        np.testing.assert_allclose(backward, forward)


class TestPredict:
    def test_zero_interval_is_identity(self) -> None:
        state = GaussianState(RNG.normal(size=6), spd(), T0)
        predicted = predict(state, T0, q=3.0)
        np.testing.assert_allclose(predicted.mean, state.mean)
        np.testing.assert_allclose(predicted.covariance, state.covariance)

    def test_carries_the_new_timestamp(self) -> None:
        state = GaussianState(np.zeros(6), spd(), T0)
        assert predict(state, T0 + timedelta(seconds=4), q=1.0).timestamp == T0 + timedelta(
            seconds=4
        )


class TestUpdate:
    def test_position_measurement_pulls_the_estimate_and_shrinks_uncertainty(self) -> None:
        covariance = np.eye(6) * 100.0
        mean, updated = update(
            np.zeros(6), covariance, np.array([10.0, 20.0, 30.0]), np.eye(3) * 100.0, (0, 2, 4)
        )
        assert mean[[0, 2, 4]] == pytest.approx([5.0, 10.0, 15.0])
        assert mean[[1, 3, 5]] == pytest.approx([0.0, 0.0, 0.0])
        assert np.trace(updated) < np.trace(covariance)

    def test_exact_measurement_pins_the_observed_components(self) -> None:
        measurement = RNG.normal(size=6)
        mean, covariance = update(
            RNG.normal(size=6), spd(), measurement, np.eye(6) * 1e-12, range(6)
        )
        np.testing.assert_allclose(mean, measurement, atol=1e-6)
        np.testing.assert_allclose(covariance, 0.0, atol=1e-6)


def indefinite_pair(singular_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Covariances whose fused information is singular exactly at `singular_weight`.

    Real covariances are positive definite, but numerical drift in non-Joseph
    updates can make them indefinite. CI must survive that.
    """
    b = 1.0
    a = b * (1.0 - singular_weight) / singular_weight
    cov_a = np.eye(6)
    cov_a[1, 1] = -1.0 / a
    return cov_a, np.eye(6) / b


class TestCovarianceIntersection:
    def test_identical_estimates_fuse_to_themselves(self) -> None:
        mean, covariance = RNG.normal(size=6), spd()
        fused = covariance_intersection(
            mean, covariance, mean, covariance, objective=CiObjective.FULL_TRACE
        )
        assert fused is not None
        np.testing.assert_allclose(fused[0], mean)
        np.testing.assert_allclose(fused[1], covariance, rtol=1e-9)

    @pytest.mark.parametrize(("omega", "winner"), [(0.99, "a"), (0.01, "b")])
    def test_fixed_weight_favors_the_weighted_estimate(self, omega: float, winner: str) -> None:
        mean_a, mean_b = np.zeros(6), np.full(6, 10.0)
        fused = covariance_intersection(
            mean_a, np.eye(6), mean_b, np.eye(6), objective=CiObjective.FULL_TRACE, omega=omega
        )
        assert fused is not None
        expected = mean_a if winner == "a" else mean_b
        np.testing.assert_allclose(fused[0], expected, atol=0.11)

    def test_result_is_symmetric(self) -> None:
        fused = covariance_intersection(
            RNG.normal(size=6),
            spd(),
            RNG.normal(size=6),
            spd(),
            objective=CiObjective.POSITION_TRACE,
        )
        assert fused is not None
        np.testing.assert_array_equal(fused[1], fused[1].T)

    def test_objectives_choose_by_their_own_criterion(self) -> None:
        # a is precise in position, b in velocity: each objective prefers a different blend.
        cov_a = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
        cov_b = np.diag([1e2, 1.0, 1e2, 1.0, 1e2, 1.0])
        mean = np.zeros(6)
        position = covariance_intersection(
            mean, cov_a, mean, cov_b, objective=CiObjective.POSITION_TRACE
        )
        full = covariance_intersection(mean, cov_a, mean, cov_b, objective=CiObjective.FULL_TRACE)
        assert position is not None
        assert full is not None
        pos = list(kalman.POSITION_INDICES)
        assert np.trace(position[1][np.ix_(pos, pos)]) < np.trace(full[1][np.ix_(pos, pos)])
        assert np.trace(full[1]) < np.trace(position[1])

    def test_singular_input_covariance_gives_none(self) -> None:
        singular = np.zeros((6, 6))
        assert (
            covariance_intersection(
                np.zeros(6), singular, np.zeros(6), np.eye(6), objective=CiObjective.FULL_TRACE
            )
            is None
        )

    @pytest.mark.parametrize("objective", list(CiObjective))
    def test_skips_singular_candidate_weights(self, objective: CiObjective) -> None:
        cov_a, cov_b = indefinite_pair(singular_weight=0.49)
        fused = covariance_intersection(np.zeros(6), cov_a, np.ones(6), cov_b, objective=objective)
        assert fused is not None
        assert np.isfinite(fused[1]).all()

    def test_singular_fixed_weight_gives_none(self) -> None:
        cov_a, cov_b = indefinite_pair(singular_weight=0.49)
        fused = covariance_intersection(
            np.zeros(6), cov_a, np.ones(6), cov_b, objective=CiObjective.FULL_TRACE, omega=0.49
        )
        assert fused is None

    def test_no_usable_weight_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kalman, "OMEGA_GRID", np.array([0.49]))
        cov_a, cov_b = indefinite_pair(singular_weight=0.49)
        fused = covariance_intersection(
            np.zeros(6), cov_a, np.ones(6), cov_b, objective=CiObjective.FULL_TRACE
        )
        assert fused is None
