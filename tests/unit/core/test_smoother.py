import numpy as np
import pytest

from crucible_entity_manager.core.smoother import SmootherParams, smooth_track

RNG = np.random.default_rng(7)
VELOCITY = np.array([120.0, -40.0, 5.0])


def trajectory(count: int, noise_m: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times, noisy positions and true positions of a constant-velocity track."""
    times = 1.79e9 + np.cumsum(RNG.uniform(2.0, 10.0, count))
    truth = np.array([4.0e6, 1.0e6, 4.5e6]) + (times - times[0])[:, None] * VELOCITY
    return times, truth + RNG.normal(scale=noise_m, size=(count, 3)), truth


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1))))


def test_reduces_position_error() -> None:
    times, measured, truth = trajectory(60)
    result = smooth_track(times, measured)
    assert rmse(result.positions, truth) < 0.6 * rmse(measured, truth)


def test_recovers_velocity_from_positions_alone() -> None:
    times, measured, _ = trajectory(60)
    result = smooth_track(times, measured)
    assert np.median(result.velocities, axis=0) == pytest.approx(VELOCITY, abs=5.0)


def test_velocity_observations_improve_the_estimate() -> None:
    times, measured, truth = trajectory(30, noise_m=150.0)
    velocities = np.tile(VELOCITY, (30, 1)) + RNG.normal(scale=1.0, size=(30, 3))
    without = smooth_track(times, measured)
    with_velocity = smooth_track(times, measured, velocities)
    assert rmse(with_velocity.positions, truth) < rmse(without.positions, truth)


def test_result_is_independent_of_input_order() -> None:
    # The baseline returned sorted states but its caller kept the input order.
    times, measured, _ = trajectory(25)
    order = RNG.permutation(25)
    reference = smooth_track(times, measured)
    shuffled = smooth_track(times[order], measured[order])
    np.testing.assert_array_equal(shuffled.times, reference.times)
    np.testing.assert_allclose(shuffled.positions, reference.positions)
    np.testing.assert_allclose(shuffled.position_covariances, reference.position_covariances)


def test_reversed_input_comes_back_ascending_with_its_times() -> None:
    times, measured, _ = trajectory(10)
    result = smooth_track(times[::-1], measured[::-1])
    assert np.all(np.diff(result.times) > 0)
    nearest = np.argmin(np.linalg.norm(result.positions[:, None] - measured[None], axis=2), axis=1)
    np.testing.assert_array_equal(nearest, np.arange(10))


def test_a_long_gap_splits_independent_segments() -> None:
    times, measured, _ = trajectory(20)
    times[10:] += 5_000.0
    joined = smooth_track(times, measured)
    first_half = smooth_track(times[:10], measured[:10])
    np.testing.assert_allclose(joined.positions[:10], first_half.positions)


def test_single_measurement_is_returned_unsmoothed() -> None:
    position = np.array([[1.0, 2.0, 3.0]])
    result = smooth_track(np.array([5.0]), position, params=SmootherParams(position_noise_m=7.0))
    np.testing.assert_array_equal(result.positions, position)
    np.testing.assert_array_equal(result.velocities, np.zeros((1, 3)))
    np.testing.assert_array_equal(result.position_covariances, [np.eye(3) * 49.0])


def test_isolated_measurement_after_a_gap_is_unsmoothed() -> None:
    times, measured, _ = trajectory(6)
    times[5] += 10_000.0
    result = smooth_track(times, measured)
    np.testing.assert_array_equal(result.positions[5], measured[5])


@pytest.mark.parametrize("tied", [[0, 1], [3, 4]])
def test_identical_timestamps_stay_finite(tied: list[int]) -> None:
    # A zero step is clamped (MIN_INITIAL_STEP_SECONDS for the first pair,
    # MIN_STEP_SECONDS after it) instead of dividing by zero.
    times, measured, _ = trajectory(6)
    times[tied[1]] = times[tied[0]]
    result = smooth_track(times, measured)
    assert np.isfinite(result.positions).all()
    assert np.isfinite(result.velocities).all()
    assert np.isfinite(result.position_covariances).all()


def test_slow_velocity_on_a_single_measurement_is_ignored() -> None:
    result = smooth_track(
        np.array([5.0]), np.array([[1.0, 2.0, 3.0]]), np.array([[0.05, 0.0, 0.0]])
    )
    np.testing.assert_array_equal(result.velocities, np.zeros((1, 3)))


def test_slow_velocity_observations_are_ignored() -> None:
    times, measured, _ = trajectory(15)
    still = np.full((15, 3), 0.01)
    np.testing.assert_allclose(
        smooth_track(times, measured, still).positions, smooth_track(times, measured).positions
    )


def test_per_measurement_covariances_weight_the_fit() -> None:
    times, measured, truth = trajectory(30, noise_m=5.0)
    measured[15] += 2_000.0
    covariances = np.broadcast_to(np.eye(3) * 25.0, (30, 3, 3)).copy()
    covariances[15] = np.eye(3) * 1e8
    result = smooth_track(times, measured, position_covariances=covariances)
    assert np.linalg.norm(result.positions[15] - truth[15]) < 100.0


def test_mismatched_covariances_fall_back_to_the_default_noise() -> None:
    times, measured, _ = trajectory(10)
    wrong_shape = np.broadcast_to(np.eye(3), (3, 3, 3)).copy()
    np.testing.assert_allclose(
        smooth_track(times, measured, position_covariances=wrong_shape).positions,
        smooth_track(times, measured).positions,
    )
