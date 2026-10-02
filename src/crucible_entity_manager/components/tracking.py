"""Turning report events into component track states (DESIGN.md §5.11).

`KalmanTracker` applies the baseline tracker's filtering policy around the
primitives in `core.kalman`: environment-dependent noise, a fresh prior for a
new track, and resets on a long time jump or on numerical trouble. The track
rows it feeds are built by `kalman_track_event` and `passthrough_track_event`.
"""

import enum
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final

import numpy as np

from crucible_entity_manager.components.keyed import AppliedInputs, KeyedState
from crucible_entity_manager.core.aliases import FloatArray, JSONObject, JSONValue
from crucible_entity_manager.core.kalman import (
    FULL_STATE_INDICES,
    POSITION_INDICES,
    STATE_DIM,
    CiObjective,
    GaussianState,
    covariance_intersection,
    predict,
    update,
)
from crucible_entity_manager.core.records import MISSING, clone_value, get_path

logger = logging.getLogger(__name__)

PROCESS_NOISE: Final[Mapping[str, float]] = {
    "AIR": 5.0,
    "GROUND": 3.0,
    "SEA_SURFACE": 0.5,
    "SEA_SUBSURFACE": 0.1,
    "SPACE": 0.001,
    "UNKNOWN": 3.0,
}
"""Continuous white-acceleration density ``q`` by environment (decision D6)."""
DEFAULT_PROCESS_NOISE: Final = 3.0

VELOCITY_VARIANCE: Final[Mapping[str, float]] = {
    "AIR": 25.0,
    "GROUND": 9.0,
    "SEA_SURFACE": 4.0,
    "SEA_SUBSURFACE": 4.0,
    "SPACE": 100.0,
    "UNKNOWN": 25.0,
}
"""Velocity variance assumed when a report has velocity but no velocity covariance."""
DEFAULT_VELOCITY_VARIANCE: Final = 25.0

INITIAL_VARIANCE: Final = 1.0e7
"""Variance of every state component in a fresh prior, and the default in heads."""
RESET_HORIZON: Final = timedelta(minutes=15)
COVARIANCE_LIMIT: Final = 1.0e15
NOISE_LIMIT: Final = 1.0e10
NOISE_FLOOR: Final = 1.0e6
"""Replaces a measurement-noise variance that is not positive or exceeds `NOISE_LIMIT`."""

TIMESTAMP_PATH: Final = "estimatedKinematics.kinematicsTimestamp"


class Fusion(enum.Enum):
    """How a measurement is combined with the prediction."""

    KALMAN = enum.auto()
    COVARIANCE_INTERSECTION = enum.auto()


@dataclass(frozen=True, slots=True)
class Measurement:
    """One report's position (and velocity, when complete) with its noise."""

    values: FloatArray
    """``[x, y, z]``, or ``[x, vx, y, vy, z, vz]`` when velocity is present."""
    noise: FloatArray
    timestamp: datetime

    @property
    def observed(self) -> tuple[int, ...]:
        """The state components this measurement observes."""
        return FULL_STATE_INDICES if self.values.shape[0] == STATE_DIM else POSITION_INDICES


def environment_of(record: JSONObject) -> str | None:
    """``identity.environment``, else ``environment``, if either is set."""
    for path in ("identity.environment", "environment"):
        value = get_path(record, path)
        if value not in (MISSING, None, ""):
            return str(value)
    return None


def measurement_from_report(
    report: JSONObject, timestamp: datetime, velocity_variance: float
) -> Measurement | None:
    """Build the measurement in `report`, or ``None`` if its position is incomplete.

    Velocity is used only when all three components are present. Missing
    covariance terms take the baseline defaults; a report with velocity but no
    velocity covariance assumes `velocity_variance` on each axis.
    """

    def number(path: str, default: float = math.nan) -> float:
        return _number(get_path(report, path), default)

    position = [number(f"ecefPosition.{axis}") for axis in "xyz"]
    velocity = [number(f"ecefVelocity.{axis}") for axis in "xyz"]
    if math.isnan(number("positionCovariance.xx")):
        pxx = pyy = pzz = INITIAL_VARIANCE
        pxy = pxz = pyz = 0.0
    else:
        pxx = number("positionCovariance.xx")
        pyy = number("positionCovariance.yy", INITIAL_VARIANCE)
        pzz = number("positionCovariance.zz", INITIAL_VARIANCE)
        pxy = number("positionCovariance.xy", 0.0)
        pxz = number("positionCovariance.xz", 0.0)
        pyz = number("positionCovariance.yz", 0.0)

    if any(math.isnan(component) for component in velocity):
        values = np.array(position)
        if np.isnan(values).any():
            return None
        noise = np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
        return Measurement(values, noise, timestamp)

    if math.isnan(number("velocityCovariance.dxdx")):
        vxx = vyy = vzz = velocity_variance
        vxy = vxz = vyz = 0.0
    else:
        vxx = number("velocityCovariance.dxdx")
        vyy = number("velocityCovariance.dydy", INITIAL_VARIANCE)
        vzz = number("velocityCovariance.dzdz", INITIAL_VARIANCE)
        vxy = number("velocityCovariance.dxdy", 0.0)
        vxz = number("velocityCovariance.dxdz", 0.0)
        vyz = number("velocityCovariance.dydz", 0.0)
    noise = np.zeros((STATE_DIM, STATE_DIM))
    noise[0, 0], noise[2, 2], noise[4, 4] = pxx, pyy, pzz
    noise[0, 2] = noise[2, 0] = pxy
    noise[0, 4] = noise[4, 0] = pxz
    noise[2, 4] = noise[4, 2] = pyz
    noise[1, 1], noise[3, 3], noise[5, 5] = vxx, vyy, vzz
    noise[1, 3] = noise[3, 1] = vxy
    noise[1, 5] = noise[5, 1] = vxz
    noise[3, 5] = noise[5, 3] = vyz
    noise = np.clip(np.nan_to_num(noise, nan=0.0), -NOISE_LIMIT, NOISE_LIMIT)
    diagonal = np.diag(noise).copy()
    diagonal[(diagonal <= 0) | (diagonal > NOISE_LIMIT)] = NOISE_FLOOR
    np.fill_diagonal(noise, diagonal)
    values = np.array(
        [position[0], velocity[0], position[1], velocity[1], position[2], velocity[2]]
    )
    if np.isnan(values).any():
        return None
    return Measurement(values, noise, timestamp)


def state_from_head(head: JSONObject, timestamp: datetime) -> GaussianState:
    """The prior a component head stores, as the baseline restores it.

    Missing velocity counts as zero; a head without a full position starts at
    the origin. NaN state is zeroed, and a covariance with NaN is replaced by
    the fresh-prior covariance.
    """
    position = [get_path(head, f"ecefPosition.{axis}") for axis in "xyz"]
    if all(value not in (MISSING, None) for value in position):
        velocity = [_number(get_path(head, f"ecefVelocity.{axis}"), 0.0) for axis in "xyz"]
        mean = np.array(
            [
                _number(position[0]),
                velocity[0],
                _number(position[1]),
                velocity[1],
                _number(position[2]),
                velocity[2],
            ]
        )
    else:
        mean = np.zeros(STATE_DIM)
    covariance = covariance_from_head(head)
    if np.isnan(covariance).any():
        covariance = np.diag([INITIAL_VARIANCE] * STATE_DIM)
    return GaussianState(np.nan_to_num(mean, nan=0.0), covariance, timestamp)


def covariance_from_head(head: JSONObject) -> FloatArray:
    """The 6x6 covariance stored in a head's covariance blocks."""

    def number(path: str, default: float) -> float:
        return _number(get_path(head, path), default)

    covariance = np.zeros((STATE_DIM, STATE_DIM))
    covariance[0, 0] = number("positionCovariance.xx", INITIAL_VARIANCE)
    covariance[2, 2] = number("positionCovariance.yy", INITIAL_VARIANCE)
    covariance[4, 4] = number("positionCovariance.zz", INITIAL_VARIANCE)
    covariance[1, 1] = number("velocityCovariance.dxdx", INITIAL_VARIANCE)
    covariance[3, 3] = number("velocityCovariance.dydy", INITIAL_VARIANCE)
    covariance[5, 5] = number("velocityCovariance.dzdz", INITIAL_VARIANCE)
    for (row, column), path in _OFF_DIAGONAL_PATHS.items():
        covariance[row, column] = covariance[column, row] = number(path, 0.0)
    return covariance


_OFF_DIAGONAL_PATHS: Final[Mapping[tuple[int, int], str]] = {
    (0, 2): "positionCovariance.xy",
    (0, 4): "positionCovariance.xz",
    (2, 4): "positionCovariance.yz",
    (1, 3): "velocityCovariance.dxdy",
    (1, 5): "velocityCovariance.dxdz",
    (3, 5): "velocityCovariance.dydz",
    (0, 1): "positionVelocityCovariance.xdx",
    (0, 3): "positionVelocityCovariance.xdy",
    (0, 5): "positionVelocityCovariance.xdz",
    (2, 1): "positionVelocityCovariance.ydx",
    (2, 3): "positionVelocityCovariance.ydy",
    (2, 5): "positionVelocityCovariance.ydz",
    (4, 1): "positionVelocityCovariance.zdx",
    (4, 3): "positionVelocityCovariance.zdy",
    (4, 5): "positionVelocityCovariance.zdz",
}


@dataclass(slots=True)
class TrackState:
    """What the tracker keeps per component track."""

    prior: GaussianState | None = None
    process_noise: float | None = None
    """``q``, fixed when the prior is created; ``None`` until first needed."""
    environment: str | None = None
    applied: AppliedInputs = field(default_factory=AppliedInputs)


class Outcome(enum.Enum):
    """What applying a measurement did."""

    UPDATED = enum.auto()
    RESET = enum.auto()
    """The track was reset to a fresh prior at this measurement."""
    STALE = enum.auto()
    """Older than the track's state; nothing changed and nothing is emitted."""


@dataclass(frozen=True, slots=True)
class TrackerParams:
    """A feed's filtering policy."""

    fusion: Fusion
    default_process_noise: float = DEFAULT_PROCESS_NOISE


class KalmanTracker:
    """Filters one feed's component tracks."""

    def __init__(
        self,
        params: TrackerParams,
        tracks: KeyedState[TrackState],
        label: str,
    ) -> None:
        self._params = params
        self._tracks = tracks
        self._label = label

    def restore(self, track_id: str, head: JSONObject, timestamp: datetime) -> None:
        """Seed `track_id`'s prior from its stored head (preload)."""
        state = self._tracks.get(track_id)
        state.environment = environment_of(head)
        state.prior = state_from_head(head, timestamp)

    def note_environment(self, track_id: str, report: JSONObject) -> None:
        """Remember the first environment a track reports; it selects the noise."""
        state = self._tracks.get(track_id)
        if state.environment is None:
            state.environment = environment_of(report)

    def velocity_variance(self, track_id: str) -> float:
        """The velocity variance assumed for `track_id`'s environment."""
        state = self._tracks.peek(track_id)
        environment = state.environment if state else None
        return _by_environment(VELOCITY_VARIANCE, environment, DEFAULT_VELOCITY_VARIANCE)

    def apply(self, track_id: str, measurement: Measurement) -> tuple[Outcome, GaussianState]:
        """Predict `track_id` to the measurement and update it, with the baseline guards.

        A new track starts from a fresh prior at the measurement, which is then
        updated with the same measurement, as at ``1b534df``. A large time jump,
        a NaN state, a NaN or exploding prediction, a NaN update or a singular
        update resets the track to a fresh prior, without an update.
        """
        state = self._tracks.get(track_id)
        current = state.prior or self._start(state, measurement)
        if current.timestamp > measurement.timestamp:
            return Outcome.STALE, current
        posterior = self._filtered(state, current, measurement)
        if isinstance(posterior, str):
            logger.warning(
                "%sResetting the filter for track %s: %s", self._label, track_id, posterior
            )
            return Outcome.RESET, self._start(state, measurement)
        state.prior = posterior
        return Outcome.UPDATED, posterior

    def _filtered(
        self, state: TrackState, current: GaussianState, measurement: Measurement
    ) -> GaussianState | str:
        """The posterior, or why the filter must be reset instead.

        Non-finite intermediate values are expected here and caught by the
        checks, so NumPy's floating-point warnings are silenced.
        """
        with np.errstate(all="ignore"):
            return self._checked_posterior(state, current, measurement)

    def _checked_posterior(
        self, state: TrackState, current: GaussianState, measurement: Measurement
    ) -> GaussianState | str:
        if measurement.timestamp - current.timestamp > RESET_HORIZON:
            return "large time jump"
        if _has_nan(current.mean, current.covariance):
            return "state contains NaN"
        if state.process_noise is None:
            state.process_noise = self._process_noise(state)
        prediction = predict(current, measurement.timestamp, state.process_noise)
        problem = _prediction_problem(prediction)
        if problem is not None:
            return problem
        try:
            mean, covariance = self._combine(state, prediction, measurement)
        except np.linalg.LinAlgError:
            return "singular update"
        if _has_nan(mean, covariance):
            return "update produced NaN"
        return GaussianState(mean, covariance, measurement.timestamp)

    def _combine(
        self, state: TrackState, prediction: GaussianState, measurement: Measurement
    ) -> tuple[FloatArray, FloatArray]:
        if self._params.fusion is Fusion.COVARIANCE_INTERSECTION:
            mean, noise = self._lifted(state, prediction, measurement)
            fused = covariance_intersection(
                prediction.mean,
                prediction.covariance,
                mean,
                noise,
                objective=CiObjective.FULL_TRACE,
            )
            if fused is not None:
                return fused
        return update(
            prediction.mean,
            prediction.covariance,
            measurement.values,
            measurement.noise,
            measurement.observed,
        )

    def _lifted(
        self, state: TrackState, prediction: GaussianState, measurement: Measurement
    ) -> tuple[FloatArray, FloatArray]:
        """A full-state measurement for covariance intersection.

        A position-only measurement borrows the predicted velocity, with the
        environment's velocity variance, so the fusion mostly ignores it.
        """
        if measurement.values.shape[0] == STATE_DIM:
            return measurement.values, measurement.noise
        x, y, z = measurement.values
        mean = np.array([x, prediction.mean[1], y, prediction.mean[3], z, prediction.mean[5]])
        noise = np.zeros((STATE_DIM, STATE_DIM))
        position = list(POSITION_INDICES)
        noise[np.ix_(position, position)] = measurement.noise
        variance = _by_environment(VELOCITY_VARIANCE, state.environment, DEFAULT_VELOCITY_VARIANCE)
        noise[1, 1] = noise[3, 3] = noise[5, 5] = variance
        return mean, noise

    def _start(self, state: TrackState, measurement: Measurement) -> GaussianState:
        values = np.nan_to_num(measurement.values, nan=0.0)
        if values.shape[0] == STATE_DIM:
            mean = values.copy()
        else:
            mean = np.zeros(STATE_DIM)
            mean[list(POSITION_INDICES)] = values
        state.prior = GaussianState(
            mean, np.diag([INITIAL_VARIANCE] * STATE_DIM), measurement.timestamp
        )
        state.process_noise = self._process_noise(state)
        return state.prior

    def _process_noise(self, state: TrackState) -> float:
        return _by_environment(PROCESS_NOISE, state.environment, self._params.default_process_noise)


def kalman_track_event(report: JSONObject, track_id: str, posterior: GaussianState) -> JSONObject:
    """A component track event: the report's identity fields and the filtered state."""
    mean, covariance = posterior.mean, posterior.covariance
    identity = report.get("identity")
    event = _track_fields(report, track_id)
    event["edhControlSet"] = clone_value(report.get("edhControlSet"))
    event["identity"] = clone_value(identity) if isinstance(identity, dict) else {}
    event["ecefPosition"] = {"x": mean[0], "y": mean[2], "z": mean[4]}
    event["ecefVelocity"] = {"x": mean[1], "y": mean[3], "z": mean[5]}
    event["positionCovariance"] = {
        "xx": covariance[0, 0],
        "xy": covariance[0, 2],
        "xz": covariance[0, 4],
        "yy": covariance[2, 2],
        "yz": covariance[2, 4],
        "zz": covariance[4, 4],
    }
    event["velocityCovariance"] = {
        "dxdx": covariance[1, 1],
        "dxdy": covariance[1, 3],
        "dxdz": covariance[1, 5],
        "dydy": covariance[3, 3],
        "dydz": covariance[3, 5],
        "dzdz": covariance[5, 5],
    }
    event["positionVelocityCovariance"] = {
        suffix: covariance[row, column]
        for suffix, (row, column) in _POSITION_VELOCITY_SUFFIXES.items()
    }
    return event


_POSITION_VELOCITY_SUFFIXES: Final[Mapping[str, tuple[int, int]]] = {
    "xdx": (0, 1),
    "xdy": (0, 3),
    "xdz": (0, 5),
    "ydx": (2, 1),
    "ydy": (2, 3),
    "ydz": (2, 5),
    "zdx": (4, 1),
    "zdy": (4, 3),
    "zdz": (4, 5),
}

_PASSTHROUGH_OBJECTS: Final = (
    "identity",
    "geodetic",
    "ecefPosition",
    "ecefVelocity",
    "positionCovariance",
    "velocityCovariance",
    "positionVelocityCovariance",
    "uncertainty",
)


def passthrough_track_event(report: JSONObject, track_id: str) -> JSONObject:
    """A component track event that copies the report's kinematics unfiltered."""
    event = _track_fields(report, track_id)
    event["edhControlSet"] = clone_value(report.get("edhControlSet"))
    for key in _PASSTHROUGH_OBJECTS:
        value = report.get(key)
        if isinstance(value, dict):
            event[key] = clone_value(value)
    for key in ("speed", "heading"):
        if key in report:
            event[key] = clone_value(report[key])
    return event


def head_from_event(event: JSONObject, stale: str) -> JSONObject:
    """A component head from a track event: its time becomes ``trackUpdatedTimestamp``."""
    head = {key: clone_value(value) for key, value in event.items()}
    head["trackUpdatedTimestamp"] = head.pop("interceptTimestamp", None)
    head.pop("trackQuality", None)
    head["stale"] = stale
    return head


def _track_fields(report: JSONObject, track_id: str) -> JSONObject:
    return {
        "trackId": track_id,
        "standardIdentity": _value(report, "identity.standard"),
        "environment": _value(report, "identity.environment"),
        "trackOriginatedTimestamp": _value(report, "crucibleHeader.createdDate"),
        "trackQuality": _value(report, "trackQuality"),
        "interceptTimestamp": _value(report, TIMESTAMP_PATH),
        "mode": _value(report, "mode"),
    }


def _value(record: JSONObject, path: str) -> JSONValue:
    value = get_path(record, path)
    return None if value is MISSING else clone_value(value)


def _prediction_problem(prediction: GaussianState) -> str | None:
    if _has_nan(prediction.mean, prediction.covariance):
        return "prediction produced NaN"
    if np.max(np.abs(prediction.covariance)) > COVARIANCE_LIMIT:
        return "covariance explosion"
    return None


def _has_nan(*arrays: FloatArray) -> bool:
    return any(bool(np.isnan(array).any()) for array in arrays)


def _by_environment(table: Mapping[str, float], environment: str | None, default: float) -> float:
    if not environment:
        return default
    return table.get(environment.upper(), default)


def _number(value: JSONValue | object, default: float = math.nan) -> float:
    """``float(value)``, or `default` when it is missing, null or not numeric."""
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
