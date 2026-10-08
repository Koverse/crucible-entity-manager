"""Native record/NumPy principal-track filter for the entity fuser."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

try:
    from entity_transformer_records import MISSING, get_value
except ImportError:
    from .entity_transformer_records import MISSING, get_value

logger = logging.getLogger(__name__)


@dataclass
class GaussianState:
    state_vector: np.ndarray
    covar: np.ndarray
    timestamp: datetime

    def __post_init__(self) -> None:
        self.state_vector = np.asarray(self.state_vector, dtype=float).reshape(-1)
        self.covar = np.asarray(self.covar, dtype=float)


def _value(record: dict, path: str, default: Any = None) -> Any:
    """Read a nested field via a dotted path; missing and explicit None differ."""
    value = get_value(record, path)
    return default if value is MISSING else value


def _number(record: dict, path: str, default: float = np.nan) -> float:
    """Read a nested field as float, using default when it is absent or invalid."""
    value = _value(record, path, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp(record: dict) -> datetime:
    """Use the first available track timestamp, falling back to current UTC."""
    value = (_value(record, "interceptTimestamp")
             or _value(record, "trackUpdatedTimestamp")
             or _value(record, "crucibleHeader.createdDate"))
    for pattern in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(str(value), pattern).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    return datetime.now(timezone.utc)


class EntityPrincipalTrackFilter:
    """Principal-track ownership and six-state Kalman fusion without adapters."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.id_field = "trackId"
        self.vel_write_cols = ("x", "y", "z")
        self.q_by_environment = {
            "AIR": 5.0, "GROUND": 3.0, "SEA_SURFACE": 0.5,
            "SEA_SUBSURFACE": 0.5, "SPACE": 0.001, "UNKNOWN": 5.0,
        }
        self.vel_var_by_environment = {
            "AIR": 25.0, "GROUND": 9.0, "SEA_SURFACE": 4.0,
            "SEA_SUBSURFACE": 4.0, "SPACE": 100.0, "UNKNOWN": 25.0,
        }
        self.q_default = 0.5
        self.vel_var_default = 25.0
        self.priors: dict[str, GaussianState] = {}
        self.environment_by_track: dict[str, str] = {}
        self.id_to_principal_trackid: dict[str, str] = {}
        self.principal_trackid_to_id: dict[str, str] = {}
        self.component_trackids_by_principal: dict[str, set[str]] = {}
        self.component_trackid_to_principal_trackid: dict[str, str] = {}
        self.new_principal_trackids: set[str] = set()
        self.known_principal_trackids: set[str] = set()
        self.recently_restored_principal_trackids: set[str] = set()
        self.supersede_map: dict[str, str] = {}
        self.fusion_method = "kalman"
        self.ci_omega: Optional[float] = None
        self._q_by_track: dict[str, float] = {}
        self.stats = {
            "successful_updates": 0, "nan_resets": 0,
            "time_jump_resets": 0, "stale_measurements_skipped": 0,
            "covariance_explosion_resets": 0,
            "prediction_nan_resets": 0, "exception_resets": 0,
            "measurements_skipped_nan": 0,
        }

    def initialize(self, supersede_map: dict[str, str],
                   principal_heads: list[dict] | None = None,
                   component_heads: list[dict] | None = None) -> None:
        self.supersede_map = dict(supersede_map or {})
        for head in principal_heads or []:
            principal_track_id = _value(head, "trackId")
            principal_id = _value(head, "principalId", principal_track_id)
            if not principal_track_id:
                continue
            principal_track_id = str(principal_track_id)
            principal_id = str(principal_id)
            self.id_to_principal_trackid[principal_id] = principal_track_id
            self.principal_trackid_to_id[principal_track_id] = principal_id
            self.known_principal_trackids.add(principal_track_id)
            measurement = self.measurement_from_record(head, principal_track_id)
            if measurement is not None:
                state, covariance, timestamp = measurement
                self.priors[principal_track_id] = GaussianState(state, covariance, timestamp)
                self._q_by_track[principal_track_id] = self._get_q(
                    self.environment_by_track.get(principal_track_id))
        for head in component_heads or []:
            component_id = _value(head, "trackId")
            principal_track_id = _value(head, "associatedPrincipalTrack")
            if component_id and principal_track_id:
                component_id = str(component_id)
                principal_track_id = str(principal_track_id)
                self.component_trackid_to_principal_trackid[component_id] = principal_track_id
                self.component_trackids_by_principal.setdefault(
                    principal_track_id, set()).add(component_id)

    def _get_q(self, environment: Optional[str]) -> float:
        return self.q_by_environment.get(str(environment).upper(), self.q_default)

    def get_vel_var_fallback(self, environment: Optional[str]) -> float:
        return self.vel_var_by_environment.get(
            str(environment).upper(), self.vel_var_default)

    def set_fusion_method(self, method: str = "kalman",
                          omega: Optional[float] = None) -> None:
        self.fusion_method = method if method in {"kalman", "ci"} else "kalman"
        self.ci_omega = None if omega is None else float(np.clip(omega, 0.01, 0.99))

    def set_track_environment(self, principal_track_id: str,
                              environment: Optional[str]) -> None:
        if environment:
            self.environment_by_track[principal_track_id] = str(environment)

    def get_principal_id(self, item_id: str) -> str:
        current = item_id
        visited = set()
        while current in self.supersede_map and current not in visited:
            visited.add(current)
            current = self.supersede_map[current]
        return current

    @staticmethod
    def _deterministic_principal_trackid(track_key: str) -> str:
        return uuid.uuid5(
            uuid.NAMESPACE_DNS, f"entity_principal_track:{track_key}"
        ).hex

    def get_or_create_principal_track(
        self, item_id: str, component_track_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> tuple[str, str, bool, bool]:
        principal_id = self.get_principal_id(item_id)
        if component_track_id in self.component_trackid_to_principal_trackid:
            principal_track_id = self.component_trackid_to_principal_trackid[component_track_id]
            return principal_track_id, self.principal_trackid_to_id.get(
                principal_track_id, principal_id), False, False
        principal_track_id = self.id_to_principal_trackid.get(principal_id)
        if principal_track_id is None:
            principal_track_id = self._deterministic_principal_trackid(principal_id)
            is_new = principal_track_id not in self.known_principal_trackids
            self.id_to_principal_trackid[principal_id] = principal_track_id
            self.principal_trackid_to_id[principal_track_id] = principal_id
            self.component_trackids_by_principal.setdefault(principal_track_id, set())
            self.known_principal_trackids.add(principal_track_id)
            if is_new:
                self.new_principal_trackids.add(principal_track_id)
        else:
            is_new = False
        if component_track_id:
            self.component_trackid_to_principal_trackid[component_track_id] = principal_track_id
        if environment:
            self.environment_by_track.setdefault(principal_track_id, str(environment))
        return principal_track_id, principal_id, is_new, component_track_id is not None

    def add_component_track(self, principal_track_id: str,
                            component_track_id: str) -> None:
        self.component_trackids_by_principal.setdefault(
            principal_track_id, set()).add(component_track_id)
        self.component_trackid_to_principal_trackid[component_track_id] = principal_track_id

    def _discard_principal(self, principal_track_id: str) -> None:
        for component_id in self.component_trackids_by_principal.pop(
                principal_track_id, set()):
            self.component_trackid_to_principal_trackid.pop(component_id, None)
        for component_id, mapped in list(self.component_trackid_to_principal_trackid.items()):
            if mapped == principal_track_id:
                self.component_trackid_to_principal_trackid.pop(component_id, None)
        for item_id, mapped in list(self.id_to_principal_trackid.items()):
            if mapped == principal_track_id:
                self.id_to_principal_trackid.pop(item_id, None)
        self.principal_trackid_to_id.pop(principal_track_id, None)
        self.priors.pop(principal_track_id, None)
        self._q_by_track.pop(principal_track_id, None)
        self.environment_by_track.pop(principal_track_id, None)
        self.new_principal_trackids.discard(principal_track_id)

    def update_supersede_map(self, new_map: dict[str, str]) -> list[dict[str, str]]:
        restored = set(self.supersede_map) - set(new_map)
        restored_principals = {
            self.component_trackid_to_principal_trackid[track_id]
            for track_id in restored
            if track_id in self.component_trackid_to_principal_trackid
        }
        restored_principals.update(
            self._deterministic_principal_trackid(track_id) for track_id in restored)
        self.recently_restored_principal_trackids = restored_principals
        for principal_track_id in restored_principals:
            self._discard_principal(principal_track_id)
        self.supersede_map = dict(new_map)
        reassociations = []
        for component_id in list(self.component_trackid_to_principal_trackid):
            root = self.get_principal_id(component_id)
            expected = self.id_to_principal_trackid.get(root)
            if expected and expected != self.component_trackid_to_principal_trackid[component_id]:
                self.component_trackid_to_principal_trackid[component_id] = expected
                self.component_trackids_by_principal.setdefault(expected, set()).add(component_id)
                reassociations.append({
                    "trackId": component_id,
                    "associatedPrincipalTrack": expected,
                })
        return reassociations

    @staticmethod
    def _transition(delta_seconds: float, q: float) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(6)
        transition[0, 1] = transition[2, 3] = transition[4, 5] = delta_seconds
        duration = abs(delta_seconds)
        block = q * np.array([
            [duration**3 / 3.0, duration**2 / 2.0],
            [duration**2 / 2.0, duration],
        ])
        process = np.zeros((6, 6))
        process[0:2, 0:2] = process[2:4, 2:4] = process[4:6, 4:6] = block
        return transition, process

    def _covariance_intersection(self, pred_mean: np.ndarray,
                                 pred_cov: np.ndarray, measurement: np.ndarray,
                                 measurement_cov: np.ndarray
                                 ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            pred_inverse = np.linalg.inv(pred_cov)
            measurement_inverse = np.linalg.inv(measurement_cov)
            weights = (np.asarray([self.ci_omega]) if self.ci_omega is not None
                       else np.linspace(0.01, 0.99, 50))
            information = (weights[:, None, None] * pred_inverse
                           + (1.0 - weights[:, None, None]) * measurement_inverse)
            covariances = np.linalg.inv(information)
            traces = covariances[:, 0, 0] + covariances[:, 2, 2] + covariances[:, 4, 4]
            index = int(np.argmin(traces))
            covariance = covariances[index]
            weight = weights[index]
            mean = covariance @ (
                weight * pred_inverse @ pred_mean
                + (1.0 - weight) * measurement_inverse @ measurement
            )
            return mean, (covariance + covariance.T) / 2.0
        except np.linalg.LinAlgError:
            return None, None

    def measurement_from_record(
        self, record: dict, principal_track_id: str
    ) -> Optional[tuple[np.ndarray, np.ndarray, datetime]]:
        velocity_variance = self.get_vel_var_fallback(
            self.environment_by_track.get(principal_track_id))
        state = np.array([
            _number(record, "ecefPosition.x"),
            _number(record, "ecefVelocity.x", 0.0),
            _number(record, "ecefPosition.y"),
            _number(record, "ecefVelocity.y", 0.0),
            _number(record, "ecefPosition.z"),
            _number(record, "ecefVelocity.z", 0.0),
        ])
        if np.isnan(state).any():
            self.stats["measurements_skipped_nan"] += 1
            return None
        covariance = np.zeros((6, 6))
        covariance[0, 0] = _number(record, "positionCovariance.xx", 1e7)
        covariance[0, 2] = covariance[2, 0] = _number(record, "positionCovariance.xy", 0.0)
        covariance[0, 4] = covariance[4, 0] = _number(record, "positionCovariance.xz", 0.0)
        covariance[2, 2] = _number(record, "positionCovariance.yy", 1e7)
        covariance[2, 4] = covariance[4, 2] = _number(record, "positionCovariance.yz", 0.0)
        covariance[4, 4] = _number(record, "positionCovariance.zz", 1e7)
        covariance[1, 1] = _number(record, "velocityCovariance.dxdx", velocity_variance)
        covariance[1, 3] = covariance[3, 1] = _number(record, "velocityCovariance.dxdy", 0.0)
        covariance[1, 5] = covariance[5, 1] = _number(record, "velocityCovariance.dxdz", 0.0)
        covariance[3, 3] = _number(record, "velocityCovariance.dydy", velocity_variance)
        covariance[3, 5] = covariance[5, 3] = _number(record, "velocityCovariance.dydz", 0.0)
        covariance[5, 5] = _number(record, "velocityCovariance.dzdz", velocity_variance)
        return state, covariance, _timestamp(record)

    def update_record(self, principal_track_id: str,
                      record: dict) -> Optional[GaussianState]:
        measurement = self.measurement_from_record(record, principal_track_id)
        if measurement is None:
            return None
        state, measurement_cov, timestamp = measurement
        if principal_track_id not in self.priors:
            self.priors[principal_track_id] = GaussianState(
                state, measurement_cov, timestamp)
            self._q_by_track[principal_track_id] = self._get_q(
                self.environment_by_track.get(principal_track_id))
        current = self.priors[principal_track_id]
        delta = (timestamp - current.timestamp).total_seconds()
        if delta < 0:
            self.stats["stale_measurements_skipped"] += 1
            return None
        if delta > 900 or np.isnan(current.state_vector).any() or np.isnan(current.covar).any():
            self.stats["time_jump_resets" if delta > 900 else "nan_resets"] += 1
            post = GaussianState(state, measurement_cov, timestamp)
            self.priors[principal_track_id] = post
            return post
        try:
            transition, process = self._transition(
                delta, self._q_by_track.get(principal_track_id, self.q_default))
            pred_mean = transition @ current.state_vector
            pred_cov = transition @ current.covar @ transition.T + process
            if np.isnan(pred_mean).any() or np.isnan(pred_cov).any():
                self.stats["prediction_nan_resets"] += 1
                post = GaussianState(state, measurement_cov, timestamp)
            elif np.max(np.abs(pred_cov)) > 1e15:
                self.stats["covariance_explosion_resets"] += 1
                post = GaussianState(state, measurement_cov, timestamp)
            elif self.fusion_method == "ci":
                mean, covariance = self._covariance_intersection(
                    pred_mean, pred_cov, state, measurement_cov)
                if mean is None:
                    mean, covariance = self._standard_update(
                        pred_mean, pred_cov, state, measurement_cov)
                post = GaussianState(mean, covariance, timestamp)
            else:
                mean, covariance = self._standard_update(
                    pred_mean, pred_cov, state, measurement_cov)
                post = GaussianState(mean, covariance, timestamp)
            self.priors[principal_track_id] = post
            self.stats["successful_updates"] += 1
            return post
        except Exception as error:
            self.stats["exception_resets"] += 1
            logger.error("Principal-track update failed for %s: %s",
                         principal_track_id, error)
            post = GaussianState(state, measurement_cov, timestamp)
            self.priors[principal_track_id] = post
            return post

    @staticmethod
    def _standard_update(pred_mean: np.ndarray, pred_cov: np.ndarray,
                         measurement: np.ndarray,
                         measurement_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        innovation = pred_cov + measurement_cov
        gain = pred_cov @ np.linalg.inv(innovation)
        mean = pred_mean + gain @ (measurement - pred_mean)
        covariance = pred_cov - gain @ innovation @ gain.T
        return mean, covariance

    def state_record(self, post: GaussianState, principal_track_id: str,
                     principal_id: str) -> dict:
        state = post.state_vector
        covariance = post.covar
        record = {
            self.id_field: principal_id,
            "trackId": principal_track_id,
            "ecefPosition": {"x": float(state[0]), "y": float(state[2]), "z": float(state[4])},
            "ecefVelocity": {"x": float(state[1]), "y": float(state[3]), "z": float(state[5])},
            "positionCovariance": {
                "xx": float(covariance[0, 0]), "xy": float(covariance[0, 2]),
                "xz": float(covariance[0, 4]), "yy": float(covariance[2, 2]),
                "yz": float(covariance[2, 4]), "zz": float(covariance[4, 4]),
            },
            "velocityCovariance": {
                "dxdx": float(covariance[1, 1]), "dxdy": float(covariance[1, 3]),
                "dxdz": float(covariance[1, 5]), "dydy": float(covariance[3, 3]),
                "dydz": float(covariance[3, 5]), "dzdz": float(covariance[5, 5]),
            },
        }
        return record

    def log_stats(self) -> None:
        logger.info("Entity principal-track filter stats: %s", self.stats)
