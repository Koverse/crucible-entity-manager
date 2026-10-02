"""Rauch-Tung-Striebel smoothing of a track history.

This is a NumPy port of the baseline's filterpy-based smoother. It uses the
same constant-velocity model, gap segmentation and Joseph-form update, and it
produces the same results. States here are grouped ``[x, y, z, vx, vy, vz]``,
as in the baseline smoother.

Results are always in ascending time order and carry their own timestamps.
The baseline returned sorted states but no times, so its caller zipped them
with timestamps in the original, unsorted order (DESIGN.md §8).
"""

import itertools
from dataclasses import dataclass
from typing import Final

import numpy as np

from crucible_entity_manager.core.aliases import FloatArray

MIN_STEP_SECONDS: Final = 0.001
"""Smallest time step used, so repeated timestamps stay well conditioned."""

MIN_INITIAL_STEP_SECONDS: Final = 0.01
"""Smallest step used to difference the first two positions into a velocity."""

MOVING_SPEED_MPS: Final = 0.1
"""Velocity observations count only if one exceeds this speed."""


@dataclass(frozen=True, slots=True)
class SmootherParams:
    """Model and noise settings for `smooth_track`."""

    process_noise_q: float = 1.0
    """White-acceleration spectral density, in m²/s³."""

    position_noise_m: float = 100.0
    """Position measurement standard deviation, used without per-point covariances."""

    velocity_noise_mps: float = 10.0
    """Velocity measurement standard deviation."""

    max_gap_seconds: float = 900.0
    """A gap longer than this starts an independently smoothed segment."""


@dataclass(frozen=True, slots=True)
class SmoothedTrack:
    """Smoothed states in ascending time order."""

    times: FloatArray
    """Epoch seconds, shape ``(n,)``."""

    positions: FloatArray
    """ECEF positions, shape ``(n, 3)``."""

    velocities: FloatArray
    """ECEF velocities, shape ``(n, 3)``."""

    position_covariances: FloatArray
    """Position covariances, shape ``(n, 3, 3)``."""


def smooth_track(
    times: FloatArray,
    positions: FloatArray,
    velocities: FloatArray | None = None,
    position_covariances: FloatArray | None = None,
    params: SmootherParams = SmootherParams(),  # noqa: B008 - frozen, so sharing is safe
) -> SmoothedTrack:
    """Smooth one track's measurements.

    Args:
        times: Measurement times in epoch seconds, in any order, shape ``(n,)``.
        positions: ECEF positions, shape ``(n, 3)``.
        velocities: Optional ECEF velocities, shape ``(n, 3)``. They are used
            only if one exceeds `MOVING_SPEED_MPS`.
        position_covariances: Optional per-measurement position covariances,
            shape ``(n, 3, 3)``. Otherwise ``params.position_noise_m`` applies.
        params: Model and noise settings.

    Returns:
        The smoothed track. A single measurement, or a segment of one
        measurement, is returned unsmoothed with the default position variance.
    """
    order = np.argsort(times, kind="stable")
    times = np.asarray(times, dtype=np.float64)[order]
    positions = np.asarray(positions, dtype=np.float64)[order]
    observed_velocities = None
    if velocities is not None and (np.linalg.norm(velocities, axis=1) > MOVING_SPEED_MPS).any():
        observed_velocities = np.asarray(velocities, dtype=np.float64)[order]
    covariances = (
        np.asarray(position_covariances, dtype=np.float64)[order]
        if position_covariances is not None and position_covariances.shape == (len(times), 3, 3)
        else None
    )

    count = len(times)
    smoothed_pos = positions.copy()
    smoothed_vel = (
        observed_velocities.copy() if observed_velocities is not None else np.zeros((count, 3))
    )
    smoothed_cov = np.broadcast_to(np.eye(3) * params.position_noise_m**2, (count, 3, 3)).copy()

    for start, stop in _segments(times, params.max_gap_seconds):
        if stop - start < 2:  # noqa: PLR2004 - smoothing needs two measurements
            continue
        means, covs = _smooth_segment(
            times[start:stop],
            positions[start:stop],
            observed_velocities[start:stop] if observed_velocities is not None else None,
            covariances[start:stop] if covariances is not None else None,
            params,
        )
        smoothed_pos[start:stop] = means[:, 0:3]
        smoothed_vel[start:stop] = means[:, 3:6]
        smoothed_cov[start:stop] = covs[:, 0:3, 0:3]

    return SmoothedTrack(times, smoothed_pos, smoothed_vel, smoothed_cov)


def _segments(times: FloatArray, max_gap_seconds: float) -> list[tuple[int, int]]:
    """Split sorted `times` into ``[start, stop)`` runs with no gap over the limit."""
    breaks = (np.flatnonzero(np.diff(times) > max_gap_seconds) + 1).tolist()
    bounds = [0, *breaks, len(times)]
    return list(itertools.pairwise(bounds))


def _smooth_segment(
    times: FloatArray,
    positions: FloatArray,
    velocities: FloatArray | None,
    position_covariances: FloatArray | None,
    params: SmootherParams,
) -> tuple[FloatArray, FloatArray]:
    """Forward-filter then RTS-smooth one segment; return 6-state means and covariances."""
    count = len(times)
    observed = 6 if velocities is not None else 3
    selection = np.eye(observed, 6)

    def measurement_noise(index: int) -> FloatArray:
        position_noise = (
            position_covariances[index]
            if position_covariances is not None
            else np.eye(3) * params.position_noise_m**2
        )
        if velocities is None:
            return position_noise
        noise = np.zeros((6, 6))
        noise[0:3, 0:3] = position_noise
        noise[3:6, 3:6] = np.eye(3) * params.velocity_noise_mps**2
        return noise

    if velocities is not None:
        initial_velocity = velocities[0]
    else:
        first_step = max(times[1] - times[0], MIN_INITIAL_STEP_SECONDS)
        initial_velocity = (positions[1] - positions[0]) / first_step
    mean = np.concatenate([positions[0], initial_velocity])
    covariance = np.diag([measurement_noise(0)[0, 0]] * 3 + [params.velocity_noise_mps**2] * 3)

    filtered_means = np.empty((count, 6))
    filtered_covs = np.empty((count, 6, 6))
    filtered_means[0], filtered_covs[0] = mean, covariance
    for index in range(1, count):
        transition, process = _grouped_model(times[index] - times[index - 1], params)
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + process
        measurement = (
            np.concatenate([positions[index], velocities[index]])
            if velocities is not None
            else positions[index]
        )
        mean, covariance = _joseph_update(
            mean, covariance, measurement, measurement_noise(index), selection
        )
        filtered_means[index], filtered_covs[index] = mean, covariance

    smoothed_means = filtered_means.copy()
    smoothed_covs = filtered_covs.copy()
    for index in range(count - 2, -1, -1):
        transition, process = _grouped_model(times[index + 1] - times[index], params)
        predicted_cov = transition @ filtered_covs[index] @ transition.T + process
        gain = filtered_covs[index] @ transition.T @ np.linalg.inv(predicted_cov)
        smoothed_means[index] = filtered_means[index] + gain @ (
            smoothed_means[index + 1] - transition @ filtered_means[index]
        )
        smoothed_covs[index] = (
            filtered_covs[index] + gain @ (smoothed_covs[index + 1] - predicted_cov) @ gain.T
        )
    return smoothed_means, smoothed_covs


def _grouped_model(step: float, params: SmootherParams) -> tuple[FloatArray, FloatArray]:
    """Constant-velocity ``F`` and ``Q`` for the grouped state layout."""
    dt = max(step, MIN_STEP_SECONDS)
    transition = np.eye(6)
    transition[0:3, 3:6] = np.eye(3) * dt
    q = params.process_noise_q
    process = np.zeros((6, 6))
    process[0:3, 0:3] = np.eye(3) * (dt**3 / 3.0) * q
    process[0:3, 3:6] = process[3:6, 0:3] = np.eye(3) * (dt**2 / 2.0) * q
    process[3:6, 3:6] = np.eye(3) * dt * q
    return transition, process


def _joseph_update(
    mean: FloatArray,
    covariance: FloatArray,
    measurement: FloatArray,
    noise: FloatArray,
    selection: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Kalman update in Joseph form, as filterpy's ``KalmanFilter.update`` does."""
    innovation_cov = selection @ covariance @ selection.T + noise
    gain = covariance @ selection.T @ np.linalg.inv(innovation_cov)
    updated_mean = mean + gain @ (measurement - selection @ mean)
    keep = np.eye(6) - gain @ selection
    updated_cov = keep @ covariance @ keep.T + gain @ noise @ gain.T
    return updated_mean, updated_cov
