"""Fusing component tracks into principal tracks (DESIGN.md §5.11, D6).

`PrincipalFilter` applies the baseline fuser's filtering policy around the
primitives in `core.kalman`. It differs from the tracker's on purpose
(decision D6): its own noise table, a position-trace objective for
covariance intersection, and full-state measurements whose missing velocity
counts as zero.

`IdentityFusion` keeps each principal track's fused identity: the union of
non-empty ``identity.*`` values across its component tracks, where a
superseded track outranks the survivor and, at equal rank, the newest value
wins.
"""

import enum
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final

import numpy as np

from crucible_entity_manager.components.keyed import KeyedState
from crucible_entity_manager.core.aliases import FloatArray, JSONObject, JSONValue
from crucible_entity_manager.core.kalman import (
    FULL_STATE_INDICES,
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
    "SEA_SUBSURFACE": 0.5,
    "SPACE": 0.001,
    "UNKNOWN": 5.0,
}
"""The fuser's ``q`` by environment, which differs from the tracker's (decision D6)."""
DEFAULT_PROCESS_NOISE: Final = 0.5

VELOCITY_VARIANCE: Final[Mapping[str, float]] = {
    "AIR": 25.0,
    "GROUND": 9.0,
    "SEA_SURFACE": 4.0,
    "SEA_SUBSURFACE": 4.0,
    "SPACE": 100.0,
    "UNKNOWN": 25.0,
}
DEFAULT_VELOCITY_VARIANCE: Final = 25.0
POSITION_VARIANCE: Final = 1.0e7
"""Position variance assumed when a component track has no position covariance."""

RESET_HORIZON: Final = timedelta(minutes=15)
COVARIANCE_LIMIT: Final = 1.0e15
OMEGA_BOUNDS: Final = (0.01, 0.99)


class Fusion(enum.Enum):
    """How a component track is combined with the principal track's prediction."""

    KALMAN = enum.auto()
    COVARIANCE_INTERSECTION = enum.auto()


@dataclass(frozen=True, slots=True)
class FusionParams:
    """The fuser's command-line filtering options."""

    fusion: Fusion = Fusion.COVARIANCE_INTERSECTION
    omega: float | None = None
    """A fixed covariance-intersection weight, clipped to `OMEGA_BOUNDS`; ``None`` searches."""


@dataclass(slots=True)
class PrincipalState:
    """What the fuser keeps per principal track."""

    prior: GaussianState | None = None
    process_noise: float | None = None
    environment: str | None = None


@dataclass(frozen=True, slots=True)
class ComponentMeasurement:
    """A component track's state, as the measurement of its principal track."""

    values: FloatArray
    noise: FloatArray
    timestamp: datetime


def component_measurement(
    component: JSONObject, timestamp: datetime, velocity_variance: float
) -> ComponentMeasurement | None:
    """The component track's full state, or ``None`` if its position is incomplete.

    A missing velocity component counts as zero, with `velocity_variance`.
    Position-velocity cross terms are not used, as at ``1b534df``.
    """

    def number(path: str, default: float = math.nan) -> float:
        return _number(get_path(component, path), default)

    values = np.array(
        [
            number("ecefPosition.x"),
            number("ecefVelocity.x", 0.0),
            number("ecefPosition.y"),
            number("ecefVelocity.y", 0.0),
            number("ecefPosition.z"),
            number("ecefVelocity.z", 0.0),
        ]
    )
    if np.isnan(values).any():
        return None
    noise = np.zeros((STATE_DIM, STATE_DIM))
    noise[0, 0] = number("positionCovariance.xx", POSITION_VARIANCE)
    noise[2, 2] = number("positionCovariance.yy", POSITION_VARIANCE)
    noise[4, 4] = number("positionCovariance.zz", POSITION_VARIANCE)
    noise[1, 1] = number("velocityCovariance.dxdx", velocity_variance)
    noise[3, 3] = number("velocityCovariance.dydy", velocity_variance)
    noise[5, 5] = number("velocityCovariance.dzdz", velocity_variance)
    for (row, column), path in _CROSS_TERMS.items():
        noise[row, column] = noise[column, row] = number(path, 0.0)
    return ComponentMeasurement(values, noise, timestamp)


_CROSS_TERMS: Final[Mapping[tuple[int, int], str]] = {
    (0, 2): "positionCovariance.xy",
    (0, 4): "positionCovariance.xz",
    (2, 4): "positionCovariance.yz",
    (1, 3): "velocityCovariance.dxdy",
    (1, 5): "velocityCovariance.dxdz",
    (3, 5): "velocityCovariance.dydz",
}


class PrincipalFilter:
    """Fuses component tracks into principal track states."""

    def __init__(
        self, params: FusionParams, principals: KeyedState[PrincipalState], label: str = ""
    ) -> None:
        omega = params.omega
        self._omega = None if omega is None else float(np.clip(omega, *OMEGA_BOUNDS))
        self._fusion = params.fusion
        self._principals = principals
        self._label = label

    def restore(self, principal: str, head: JSONObject, timestamp: datetime) -> None:
        """Seed a principal's prior from its stored head (preload).

        The noise is taken from the default, as the baseline does at preload.
        """
        measurement = component_measurement(head, timestamp, DEFAULT_VELOCITY_VARIANCE)
        if measurement is None:
            return
        state = self._principals.get(principal)
        state.prior = GaussianState(measurement.values, measurement.noise, timestamp)
        state.process_noise = DEFAULT_PROCESS_NOISE

    def note_environment(self, principal: str, environment: str | None) -> None:
        """Remember the latest non-empty environment a principal's components report."""
        if environment:
            self._principals.get(principal).environment = environment

    def velocity_variance(self, principal: str) -> float:
        """The velocity variance assumed for `principal`'s environment."""
        state = self._principals.peek(principal)
        environment = state.environment if state else None
        return _by_environment(VELOCITY_VARIANCE, environment, DEFAULT_VELOCITY_VARIANCE)

    def forget(self, principal: str) -> None:
        """Drop a principal's filter state."""
        self._principals.pop(principal)

    def apply(self, principal: str, measurement: ComponentMeasurement) -> GaussianState | None:
        """Fuse a component state into `principal`; ``None`` if it is stale.

        A new principal starts at the measurement and is then updated with it,
        as at ``1b534df``. A gap over 15 minutes, a NaN state or prediction, an
        exploding prediction or a singular update resets the principal to the
        measurement itself.
        """
        state = self._principals.get(principal)
        if state.prior is None:
            state.prior = GaussianState(
                measurement.values, measurement.noise, measurement.timestamp
            )
            state.process_noise = self._process_noise(state)
        current = state.prior
        if measurement.timestamp < current.timestamp:
            return None
        with np.errstate(all="ignore"):
            posterior = self._fused(state, current, measurement)
        if isinstance(posterior, str):
            logger.debug("%sResetting principal %s: %s", self._label, principal, posterior)
            posterior = GaussianState(measurement.values, measurement.noise, measurement.timestamp)
        state.prior = posterior
        return posterior

    def _fused(
        self, state: PrincipalState, current: GaussianState, measurement: ComponentMeasurement
    ) -> GaussianState | str:
        """The posterior, or why the principal must be reset instead."""
        if measurement.timestamp - current.timestamp > RESET_HORIZON:
            return "large time jump"
        if np.isnan(current.mean).any() or np.isnan(current.covariance).any():
            return "state contains NaN"
        process_noise = (
            state.process_noise if state.process_noise is not None else DEFAULT_PROCESS_NOISE
        )
        prediction = predict(current, measurement.timestamp, process_noise)
        if np.isnan(prediction.mean).any() or np.isnan(prediction.covariance).any():
            return "prediction produced NaN"
        if np.max(np.abs(prediction.covariance)) > COVARIANCE_LIMIT:
            return "covariance explosion"
        try:
            mean, covariance = self._combine(prediction, measurement)
        except np.linalg.LinAlgError:
            return "singular update"
        return GaussianState(mean, covariance, measurement.timestamp)

    def _combine(
        self, prediction: GaussianState, measurement: ComponentMeasurement
    ) -> tuple[FloatArray, FloatArray]:
        if self._fusion is Fusion.COVARIANCE_INTERSECTION:
            fused = covariance_intersection(
                prediction.mean,
                prediction.covariance,
                measurement.values,
                measurement.noise,
                objective=CiObjective.POSITION_TRACE,
                omega=self._omega,
            )
            if fused is not None:
                return fused
        return update(
            prediction.mean,
            prediction.covariance,
            measurement.values,
            measurement.noise,
            FULL_STATE_INDICES,
        )

    def _process_noise(self, state: PrincipalState) -> float:
        return _by_environment(PROCESS_NOISE, state.environment, DEFAULT_PROCESS_NOISE)


def state_fields(posterior: GaussianState) -> JSONObject:
    """The principal state fields a fused event and head carry."""
    mean, covariance = posterior.mean, posterior.covariance
    return {
        "ecefPosition": {"x": float(mean[0]), "y": float(mean[2]), "z": float(mean[4])},
        "ecefVelocity": {"x": float(mean[1]), "y": float(mean[3]), "z": float(mean[5])},
        "positionCovariance": {
            "xx": float(covariance[0, 0]),
            "xy": float(covariance[0, 2]),
            "xz": float(covariance[0, 4]),
            "yy": float(covariance[2, 2]),
            "yz": float(covariance[2, 4]),
            "zz": float(covariance[4, 4]),
        },
        "velocityCovariance": {
            "dxdx": float(covariance[1, 1]),
            "dxdy": float(covariance[1, 3]),
            "dxdz": float(covariance[1, 5]),
            "dydy": float(covariance[3, 3]),
            "dydz": float(covariance[3, 5]),
            "dzdz": float(covariance[5, 5]),
        },
    }


_COPIED_PATHS: Final = (
    "geodetic",
    "ecefPosition",
    "ecefVelocity",
    "positionCovariance",
    "velocityCovariance",
    "positionVelocityCovariance",
    "uncertainty",
    "identity",
    "speed",
    "heading",
)


def principal_record(
    component: JSONObject, principal: str, *, head: bool, stale: str
) -> JSONObject:
    """A principal track event (or head) copied from a component track event."""
    record: JSONObject = {
        "trackId": principal,
        "standardIdentity": _value(component, "standardIdentity"),
        "environment": _value(component, "environment"),
        "trackOriginatedTimestamp": _value(component, "trackOriginatedTimestamp"),
        "edhControlSet": _value(component, "edhControlSet"),
        "mode": _value(component, "mode"),
    }
    if head:
        record["trackUpdatedTimestamp"] = _value(component, "interceptTimestamp")
        record["stale"] = stale
    else:
        record["trackQuality"] = _value(component, "trackQuality")
        record["interceptTimestamp"] = _value(component, "interceptTimestamp")
    for path in _COPIED_PATHS:
        value = get_path(component, path)
        if value is not MISSING:
            record[path] = clone_value(value)
    return record


@dataclass(slots=True)
class _Fused:
    value: JSONValue
    rank: int


@dataclass(slots=True)
class FusedIdentity:
    """A principal track's fused identity, by ``identity.*`` field."""

    fields: dict[str, _Fused] = field(default_factory=dict)

    def merge(self, identity: JSONValue, *, superseded: bool) -> None:
        """Take `identity`'s non-empty values that outrank or equal what is held."""
        if not isinstance(identity, dict):
            return
        rank = 1 if superseded else 0
        for name, value in identity.items():
            if value is None or value == "":
                continue
            held = self.fields.get(name)
            if held is None or rank >= held.rank:
                self.fields[name] = _Fused(clone_value(value), rank)

    def values(self) -> dict[str, JSONValue]:
        """The fused ``identity`` object."""
        return {name: clone_value(fused.value) for name, fused in self.fields.items()}


def _value(record: JSONObject, path: str) -> JSONValue:
    value = get_path(record, path)
    return None if value is MISSING else clone_value(value)


def _by_environment(table: Mapping[str, float], environment: str | None, default: float) -> float:
    return table.get(str(environment).upper(), default)


def _number(value: JSONValue | object, default: float) -> float:
    """``float(value)``, or `default` when it is missing, null or not numeric."""
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
