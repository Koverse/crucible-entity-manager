"""Six-state constant-velocity Kalman filtering and covariance intersection.

States are interleaved ``[x, vx, y, vy, z, vz]`` in ECEF meters and meters per
second. These are the numeric primitives only. The tracker and fuser apply
their own guards (stale measurements, resets, noise tables) around them.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np

from crucible_entity_manager.core.aliases import FloatArray

STATE_DIM: Final = 6
POSITION_INDICES: Final = (0, 2, 4)
VELOCITY_INDICES: Final = (1, 3, 5)
FULL_STATE_INDICES: Final = (0, 1, 2, 3, 4, 5)

OMEGA_GRID: Final = np.linspace(0.01, 0.99, 50)
"""Candidate covariance-intersection weights searched when no weight is fixed."""


@dataclass(frozen=True, slots=True)
class GaussianState:
    """A state estimate: interleaved mean, its covariance, and its time."""

    mean: FloatArray
    covariance: FloatArray
    timestamp: datetime


class CiObjective(enum.Enum):
    """What covariance intersection minimizes when choosing its weight.

    The tracker and the fuser historically use different objectives
    (DESIGN.md, decision D6).
    """

    FULL_TRACE = enum.auto()
    """Trace of the whole 6x6 covariance (tracker)."""

    POSITION_TRACE = enum.auto()
    """Trace of the position block only (fuser)."""


def constant_velocity(dt: float, q: float) -> tuple[FloatArray, FloatArray]:
    """Return the transition ``F`` and process noise ``Q`` over `dt` seconds.

    `q` is the continuous white-acceleration spectral density. ``Q`` uses
    ``|dt|``, so predicting backwards still adds uncertainty.
    """
    transition = np.eye(STATE_DIM)
    transition[0, 1] = transition[2, 3] = transition[4, 5] = dt
    span = abs(dt)
    block = q * np.array([[span**3 / 3.0, span**2 / 2.0], [span**2 / 2.0, span]])
    noise = np.zeros((STATE_DIM, STATE_DIM))
    noise[0:2, 0:2] = noise[2:4, 2:4] = noise[4:6, 4:6] = block
    return transition, noise


def predict(state: GaussianState, timestamp: datetime, q: float) -> GaussianState:
    """Propagate `state` to `timestamp` under the constant-velocity model."""
    transition, noise = constant_velocity((timestamp - state.timestamp).total_seconds(), q)
    return GaussianState(
        mean=transition @ state.mean,
        covariance=transition @ state.covariance @ transition.T + noise,
        timestamp=timestamp,
    )


def update(
    mean: FloatArray,
    covariance: FloatArray,
    measurement: FloatArray,
    noise: FloatArray,
    observed: Sequence[int],
) -> tuple[FloatArray, FloatArray]:
    """Apply a linear measurement of the state components in `observed`.

    Uses the standard (non-Joseph) form ``P - K S Kᵀ``, as the baseline did.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular.
    """
    selection = np.zeros((len(observed), STATE_DIM))
    selection[np.arange(len(observed)), list(observed)] = 1.0
    innovation_cov = selection @ covariance @ selection.T + noise
    gain = covariance @ selection.T @ np.linalg.inv(innovation_cov)
    updated_mean = mean + gain @ (measurement - selection @ mean)
    updated_cov = covariance - gain @ innovation_cov @ gain.T
    return updated_mean, updated_cov


def covariance_intersection(  # noqa: PLR0913 - two estimates plus how to weigh them
    mean_a: FloatArray,
    cov_a: FloatArray,
    mean_b: FloatArray,
    cov_b: FloatArray,
    *,
    objective: CiObjective,
    omega: float | None = None,
) -> tuple[FloatArray, FloatArray] | None:
    """Fuse two estimates without assuming their errors are independent.

    The fused information is ``ω·Pa⁻¹ + (1-ω)·Pb⁻¹``. With `omega` unset, ω is
    the `OMEGA_GRID` value that minimizes `objective`; weights whose fused
    information is singular are skipped.

    Returns:
        The fused mean and symmetrized covariance, or ``None`` if either input
        covariance or every candidate fused information matrix is singular.
    """
    try:
        info_a = np.linalg.inv(cov_a)
        info_b = np.linalg.inv(cov_b)
    except np.linalg.LinAlgError:
        return None
    weight = omega if omega is not None else _best_weight(info_a, info_b, objective)
    if weight is None:
        return None
    try:
        fused_cov = np.linalg.inv(weight * info_a + (1.0 - weight) * info_b)
    except np.linalg.LinAlgError:
        return None
    fused_cov = (fused_cov + fused_cov.T) / 2.0
    fused_mean = fused_cov @ (weight * info_a @ mean_a + (1.0 - weight) * info_b @ mean_b)
    return fused_mean, fused_cov


def _best_weight(info_a: FloatArray, info_b: FloatArray, objective: CiObjective) -> float | None:
    weights = OMEGA_GRID[:, None, None]
    try:
        candidates = np.linalg.inv(weights * info_a + (1.0 - weights) * info_b)
    except np.linalg.LinAlgError:
        candidates = _invert_each(weights * info_a + (1.0 - weights) * info_b)
    scores = _objective_scores(candidates, objective)
    if not np.isfinite(scores).any():
        return None
    return float(OMEGA_GRID[int(np.nanargmin(scores))])


def _invert_each(matrices: FloatArray) -> FloatArray:
    """Invert each matrix, leaving NaN for the singular ones."""
    inverses = np.full_like(matrices, np.nan)
    for index, matrix in enumerate(matrices):
        try:
            inverses[index] = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
    return inverses


def _objective_scores(covariances: FloatArray, objective: CiObjective) -> FloatArray:
    if objective is CiObjective.FULL_TRACE:
        return np.trace(covariances, axis1=1, axis2=2)
    position = list(POSITION_INDICES)
    return covariances[:, position, position].sum(axis=1)
