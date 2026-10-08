#!/usr/bin/env python
"""Entity tracker.

Reads Report_Events and groups them into component tracks using a deterministic
trackId derived from the origin dataset and identity fields. Component track
events and heads retain the full identity payload.
"""
from __future__ import annotations

from multiprocessing import Process, Queue, set_start_method
from datetime import datetime as dt
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import traceback
import argparse
import asyncio
import logging
import signal
import uuid
import os
import re

try:
    from cruciblelib.decorators import async_retry
except ImportError:
    from .decorators import async_retry

try:
    # Bare import supports direct script execution; relative import supports
    # package imports.
    from entity_utils import (
        ensure_api_controllers,
        identity_custom_id,
        find_and_validate_configs,
        terminate, write_batch_chunked, SSE_listener,
        best_effort_update_track_records,
        collect_finished_track_update_tasks,
        buffer_latest_records_by_track,
        coalesce_existing_head_updates,
        get_current_timestamp_string,
        last_record_by_track,
        load_all_records_for_preload,
        skip_head_preload,
        stable_shard,
    )
    from entity_transformer_records import (
        add_track_wgs84_kinematics, clone_record, compact_record,
    )
except ImportError:
    from .entity_utils import (
        ensure_api_controllers,
        identity_custom_id,
        find_and_validate_configs,
        terminate, write_batch_chunked, SSE_listener,
        best_effort_update_track_records,
        collect_finished_track_update_tasks,
        buffer_latest_records_by_track,
        coalesce_existing_head_updates,
        get_current_timestamp_string,
        last_record_by_track,
        load_all_records_for_preload,
        skip_head_preload,
        stable_shard,
    )
    from .entity_transformer_records import (
        add_track_wgs84_kinematics, clone_record, compact_record,
    )


# Lightweight in-file compatibility shim: the active tracker path is NumPy-based
# and operates on dict rows / arrays. No external StoneSoup dependency is
# required at runtime.
class Detection:
    def __init__(self, state_vector: Any, timestamp: Any = None,
                 measurement_model: Any = None) -> None:
        self.state_vector = np.asarray(state_vector, dtype=float).reshape(-1)
        self.timestamp = timestamp
        self.measurement_model = measurement_model


class SingleHypothesis:
    def __init__(self, prediction: Any, measurement: Any) -> None:
        self.prediction = prediction
        self.measurement = measurement


class CombinedLinearGaussianTransitionModel:
    def __init__(self, models: List[Any]) -> None:
        self.models = list(models)


class ConstantVelocity:
    def __init__(self, q: float) -> None:
        self.q = float(q)


class KalmanPredictor:
    def __init__(self, transition_model: Any) -> None:
        self.transition_model = transition_model

    def predict(self, current_state: Any, timestamp: Any) -> Any:
        mean = np.asarray(current_state.state_vector, dtype=float).reshape(-1)
        cov = np.asarray(current_state.covar, dtype=float)
        dt = (timestamp - current_state.timestamp).total_seconds()
        q = float(self.transition_model.models[0].q)
        F = np.eye(6)
        F[0, 1] = F[2, 3] = F[4, 5] = dt
        d = abs(dt)
        qb = q * np.array([[d ** 3 / 3.0, d ** 2 / 2.0],
                           [d ** 2 / 2.0, d]])
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = qb
        Q[2:4, 2:4] = qb
        Q[4:6, 4:6] = qb
        return GaussianState((F @ mean).reshape(-1, 1), F @ cov @ F.T + Q,
                             timestamp=timestamp)


class KalmanUpdater:
    def __init__(self, measurement_model: Any) -> None:
        self.measurement_model = measurement_model

    def update(self, hypothesis: SingleHypothesis) -> Any:
        pred = hypothesis.prediction
        meas = hypothesis.measurement
        pred_mean = np.asarray(pred.state_vector, dtype=float).reshape(-1)
        pred_cov = np.asarray(pred.covar, dtype=float)
        z = np.asarray(meas.state_vector, dtype=float).reshape(-1)
        H = np.zeros((len(meas.measurement_model.mapping), 6))
        H[np.arange(len(meas.measurement_model.mapping)), list(meas.measurement_model.mapping)] = 1.0
        R = np.asarray(meas.measurement_model.noise_covar, dtype=float)
        S = H @ pred_cov @ H.T + R
        K = pred_cov @ H.T @ np.linalg.inv(S)
        post_mean = pred_mean + K @ (z - H @ pred_mean)
        post_cov = pred_cov - K @ S @ K.T
        return GaussianState(post_mean.reshape(-1, 1), post_cov,
                             timestamp=meas.timestamp)


class LinearGaussian:
    """Minimal measurement model compatible with the legacy tracker interface."""
    def __init__(self, ndim_state: Any, mapping: Any, noise_covar: Any) -> None:
        self.ndim_state = int(ndim_state)
        self.mapping = tuple(mapping)
        self.noise_covar = np.asarray(noise_covar, dtype=float)

# Global variables for controllers (will be initialized in the child process)
rc = wc = auth = None
_DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS = float(
    os.getenv('CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS', '15'))


async def _best_effort_update_track_heads(records: List[dict], track_head_dataset: str,
                                          heads_chunk_size: int, origin_dataset: str,
                                          max_concurrent: Optional[int]) -> List[dict]:
    return await best_effort_update_track_records(
        records, track_head_dataset, heads_chunk_size,
        wc.update_entity_record_batch_by_name,
        write_batch_chunked,
        label=f' [{origin_dataset}] existing-head best-effort: ',
        log_prefix=f' [{origin_dataset}]: ',
        token_refresher=lambda: setattr(wc, 'token', auth.get_token()),
        max_concurrent_writes=max_concurrent,
    )


def _collect_finished_head_update_tasks(filter_manager: Any, origin_dataset: str) -> None:
    pending = getattr(filter_manager, 'pending_head_update_tasks', [])
    filter_manager.pending_head_update_tasks = collect_finished_track_update_tasks(
        pending, 'existing track-head update',
        filter_manager.pending_head_updates_by_track,
        log_prefix=f' [{origin_dataset}]: ',
    )


def _head_update_interval_seconds(config: Dict[str, Any]) -> float:
    value = config.get('head_update_interval_seconds',
                       _DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS

def _parse_tracker_modes(crucible_tracker: Any) -> tuple[bool, bool, bool]:
    """Return passthrough, Kalman, and CI flags for a tracker mode value."""
    value = str(crucible_tracker or '').strip().lower()
    use_ci = 'covariance intersection' in value or 'ci' in value.split()
    use_kalman = 'kalman' in value or use_ci
    use_passthrough = 'passthrough' in value and not use_kalman
    return use_passthrough, use_kalman, use_ci


class NumpyGaussianState:
    """Minimal StoneSoup-compatible state container used by the numpy tracker."""

    def __init__(self, state_vector: Any, covar: Any, timestamp: Any = None) -> None:
        self.state_vector = np.asarray(state_vector, dtype=float).reshape(-1)
        self.covar = np.asarray(covar, dtype=float)
        self.timestamp = timestamp


class Track(list):
    """Minimal list-backed replacement for stonesoup.types.track.Track."""


class LinearGaussianModel:
    """Minimal measurement-model container for mapping/noise covariance."""

    def __init__(self, mapping: Any, noise_covar: Any) -> None:
        self.mapping = tuple(mapping)
        self.noise_covar = np.asarray(noise_covar, dtype=float)


# State type used by the in-file NumPy kernel.
GaussianState = NumpyGaussianState

_MISSING = object()


def _isna(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return False
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _notna(value: Any) -> bool:
    return not _isna(value)


def _record_value(
    record: dict,
    keys: str | tuple[str, ...],
    default: Any = None,
) -> Any:
    if isinstance(keys, str):
        return record.get(keys, default)
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _write_records(records: List[dict]) -> List[dict]:
    return [compact_record(record) for record in records]


def _covariance_from_record(record: dict, velocity_variance: float = 1.e7) -> np.ndarray:
    def number(keys: tuple[str, ...], default: float) -> float:
        value = _record_value(record, keys, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    covariance = np.zeros((6, 6), dtype=float)
    covariance[0, 0] = number(('positionCovariance', 'xx'), 1.e7)
    covariance[0, 2] = covariance[2, 0] = number(('positionCovariance', 'xy'), 0.0)
    covariance[0, 4] = covariance[4, 0] = number(('positionCovariance', 'xz'), 0.0)
    covariance[2, 2] = number(('positionCovariance', 'yy'), 1.e7)
    covariance[2, 4] = covariance[4, 2] = number(('positionCovariance', 'yz'), 0.0)
    covariance[4, 4] = number(('positionCovariance', 'zz'), 1.e7)
    covariance[1, 1] = number(('velocityCovariance', 'dxdx'), velocity_variance)
    covariance[1, 3] = covariance[3, 1] = number(('velocityCovariance', 'dxdy'), 0.0)
    covariance[1, 5] = covariance[5, 1] = number(('velocityCovariance', 'dxdz'), 0.0)
    covariance[3, 3] = number(('velocityCovariance', 'dydy'), velocity_variance)
    covariance[3, 5] = covariance[5, 3] = number(('velocityCovariance', 'dydz'), 0.0)
    covariance[5, 5] = number(('velocityCovariance', 'dzdz'), velocity_variance)
    position_velocity = (
        ('xdx', 0, 1), ('xdy', 0, 3), ('xdz', 0, 5),
        ('ydx', 2, 1), ('ydy', 2, 3), ('ydz', 2, 5),
        ('zdx', 4, 1), ('zdy', 4, 3), ('zdz', 4, 5),
    )
    for suffix, position_index, velocity_index in position_velocity:
        value = number(('positionVelocityCovariance', suffix), 0.0)
        covariance[position_index, velocity_index] = value
        covariance[velocity_index, position_index] = value
    return covariance

# ---------------------------------------------------------------------------
# Track-id helpers.
#
# Each report event is grouped into a component track by a deterministic trackId built directly from
# the identity fields: customID is the sorted concatenation of all identity.*
# fields, and the trackId is uuid5(NAMESPACE_DNS, "{origin_dataset}:{customID}")
# -- stable across runs and processes, and unique per data feed.
# ---------------------------------------------------------------------------

TRACK_ID_FIELDS = ["identity.*"]
TRACK_ID_FIELDS_CONFIG_KEY = "track_id_fields"


def configured_track_id_fields(config: dict) -> List[str]:
    """Return configured track-ID paths or the schema-flexible default list."""
    configured = config.get(TRACK_ID_FIELDS_CONFIG_KEY)
    if configured is None:
        return list(TRACK_ID_FIELDS)
    if not isinstance(configured, list):
        raise ValueError(f"{TRACK_ID_FIELDS_CONFIG_KEY} must be a list of field paths")
    fields = [field.strip() for field in configured if isinstance(field, str) and field.strip()]
    if len(fields) != len(configured) or not fields:
        raise ValueError(
            f"{TRACK_ID_FIELDS_CONFIG_KEY} must contain one or more non-empty strings"
        )
    return fields


def _track_id_from_custom_id(origin_dataset: str, custom_id: str) -> str:
    """Deterministic trackId = uuid5(origin_dataset : identity customID). Stable
    across runs and processes, and unique per data feed."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{origin_dataset}:{custom_id}").hex


def assign_track_id(
    event: dict,
    origin_dataset: str,
    field_paths: Optional[List[str]] = None,
) -> str:
    """Resolve a deterministic trackId for a raw report event from its identity
    fields using the (origin_dataset, identity customID) pair."""
    custom_id = identity_custom_id(event, field_paths or TRACK_ID_FIELDS)
    return _track_id_from_custom_id(origin_dataset, custom_id)


def _shard_for_track(track_id: Any, n_shards: int) -> int:
    """Compatibility wrapper for deterministic track sharding."""
    return stable_shard(track_id, n_shards)


class KalmanFilterManager:
    """Manages Kalman filters and priors for each track.
        There is one instance per data feed.
    """
    
    def __init__(self, config: dict) -> None:
        self.config = config
        self.origin_dataset = config.get('origin_dataset', '')
        self.predictor = {}  # trackId -> KalmanPredictor
        self.updater = {}    # trackId -> KalmanUpdater
        self.measurement_model = {}  # trackId -> LinearGaussian
        self.priors = {}   # trackId -> GaussianState
        self.tracks = {}   # trackId -> Track
        self.initialized = False
        self.known_trackids = set()  # trackIds already seen (created or loaded)
        self.new_trackIds = set()  # TrackIds that are newly created in this run
        self.pending_head_update_tasks = []  # best-effort existing-head PUTs
        self.pending_head_updates_by_track = {}
        self.existing_head_update_emit_time_by_track = {}
        self.passthrough_tracker = False
        self.ecef_kalman_filter = False
        self.fusion_method = 'kalman'  # 'kalman' or 'ci'
        self.ci_omega = None  # None = auto-optimize, float = fixed weight
        self.environment_by_track = {}  # trackId -> environment string

        # Environment-aware process noise (continuous-time acceleration PSD).
        self.q_by_environment = {
            'AIR': 5.0,
            'GROUND': 3.0,
            'SEA_SURFACE': 0.5,
            'SEA_SUBSURFACE': 0.1,
            'SPACE': 0.001,
            'UNKNOWN': 3.0,
        }
        self.q_default = 3.0

        # Velocity variance fallback when raw velocity is provided without
        # velocity covariance.
        self.vel_var_by_environment = {
            'AIR': 25.0,
            'GROUND': 9.0,
            'SEA_SURFACE': 4.0,
            'SEA_SUBSURFACE': 4.0,
            'SPACE': 100.0,
            'UNKNOWN': 25.0,
        }
        self.vel_var_default = 25.0

    def _get_q(self, track_id: Optional[Any] = None) -> float:
        env = self.environment_by_track.get(track_id)
        if env:
            return self.q_by_environment.get(env.upper(), self.q_default)
        return self.q_default

    def _get_vel_var(self, track_id: Optional[Any] = None) -> float:
        env = self.environment_by_track.get(track_id)
        if env:
            return self.vel_var_by_environment.get(env.upper(), self.vel_var_default)
        return self.vel_var_default
    
    async def initialize_from_heads(self, crucible_kalman_config_string: str,
                                   preloaded_heads: Optional[List[dict]] = None) -> None:
        """Initialize filters and priors from the component track head dataset.

        When *preloaded_heads* is provided, its native nested records are used directly and
        no per-shard Crucible query is issued — avoiding the N-shard duplicate
        reads that 504'd on large head datasets. When None, falls back to the
        direct record query.
        """
        if self.initialized:
            return
            
        try:
            
            # Extract q value from crucible_kalman_config_string as override
            match = re.search(r'q\s*=\s*(\d+(\.\d+)?)', crucible_kalman_config_string)
            if match:
                q_override = float(match.group(1))
                self.q_default = q_override
                logging.info(f"Using q={q_override} from config as default")

            if preloaded_heads is not None:
                heads = list(preloaded_heads)
            else:
                head_dataset = self.config.get('component_track_head_dataset')
                if not isinstance(head_dataset, str) or not head_dataset:
                    raise ValueError("component_track_head_dataset must be configured")
                heads = await asyncio.to_thread(
                    load_all_records_for_preload,
                    head_dataset,
                    self.config,
                    rc,
                    auth,
                    "tracker_head_preload_limit",
                )

            if heads:
                track_id_fields = configured_track_id_fields(self.config)
                for head in heads:
                    if not isinstance(head, dict):
                        continue
                    track_id = _record_value(head, 'trackId')
                    custom_id = identity_custom_id(head, track_id_fields)
                    expected_track_id = _track_id_from_custom_id(self.origin_dataset, custom_id)
                    if track_id != expected_track_id:
                        continue

                    self.known_trackids.add(track_id)
                    if not self.passthrough_tracker:
                        env = (_record_value(head, ('identity', 'environment'))
                               or _record_value(head, 'environment'))
                        if env:
                            self.environment_by_track[track_id] = str(env)

                        position = [_record_value(head, ('ecefPosition', axis))
                                    for axis in ('x', 'y', 'z')]
                        velocity = [_record_value(head, ('ecefVelocity', axis))
                                    for axis in ('x', 'y', 'z')]
                        if all(value is not None for value in position):
                            velocity = [0.0 if value is None else value for value in velocity]
                            initial_state_vector = np.asarray([
                                position[0], velocity[0], position[1], velocity[1],
                                position[2], velocity[2],
                            ], dtype=float)
                        else:
                            initial_state_vector = np.zeros(6, dtype=float)

                        # Replace NaN values with 0 and use high-uncertainty covariance
                        if np.isnan(initial_state_vector).any():
                            logging.warning(f" [{self.origin_dataset}]: Track {track_id} initial state contains NaN - replacing with zeros and resetting covariance")
                            initial_state_vector = np.nan_to_num(initial_state_vector, nan=0.0)

                        initial_covar = _covariance_from_record(head)

                        # Replace NaN covariance values with high-uncertainty default
                        if np.isnan(initial_covar).any():
                            logging.warning(f" [{self.origin_dataset}]: Track {track_id} initial covariance contains NaN - using default high-uncertainty covariance")
                            initial_covar = np.diag([1.e7, 1.e7, 1.e7, 1.e7, 1.e7, 1.e7])

                        # Create and store prior
                        timestamp_str = _record_value(
                            head, 'trackUpdatedTimestamp', dt.utcnow().isoformat() + 'Z')
                        try:
                            timestamp = dt.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                        except Exception:
                            timestamp = dt.utcnow()
                        prior = GaussianState(
                            initial_state_vector,
                            initial_covar,
                            timestamp=timestamp
                        )
                        self.priors[track_id] = prior
                        self.tracks[track_id] = Track([prior])

                if not self.passthrough_tracker:
                    logging.info(f" [{self.origin_dataset}]: Initialized Kalman filters for {len(self.priors)} tracks from track heads")
                else:
                    logging.info(f" [{self.origin_dataset}]: Loaded {len(self.known_trackids)} known trackIds from track heads (passthrough mode)")
            else:
                logging.info(f" [{self.origin_dataset}]: No existing track heads found - filters will be initialized as new tracks are encountered")
            
            self.initialized = True
            
        except Exception as e:
            logging.error(f" [{self.origin_dataset}]: Error initializing Kalman filters: {e}")
            logging.error(traceback.format_exc())
            self.initialized = True  # Mark as initialized to prevent retry loops
    
    def _reset_filter(self, track_id: Any, measurement: Any) -> None:
        """Reset a filter to the current measurement state when numerical issues occur."""
        for store in (self.priors, self.predictor, self.updater, self.measurement_model, self.tracks):
            store.pop(track_id, None)
        self.get_or_create_filter(track_id, measurement)
        logging.warning(f" [{self.origin_dataset}]: Reset Kalman filter for track {track_id} due to numerical issues")

    def _covariance_intersection(self, pred_mean: np.ndarray, pred_covar: np.ndarray,
                                  meas_mean: np.ndarray, meas_covar: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Covariance Intersection: fuse predicted state and measurement
        without assuming statistical independence.

        Omega interpretation:
            ~0   → measurement-dominated (prediction covariance >> measurement)
            ~0.5 → balanced (comparable covariances, real fusion)
            ~1   → prediction-dominated (measurement covariance >> prediction)

        Returns (x_fused, P_fused) or (None, None) on failure.
        """
        try:
            P1_inv = np.linalg.inv(pred_covar)
            P2_inv = np.linalg.inv(meas_covar)
        except np.linalg.LinAlgError:
            return None, None

        if self.ci_omega is not None:
            best_w = self.ci_omega
        else:
            # Vectorized omega grid search: build all 50 candidate fused
            # information matrices and invert them in one batched np.linalg.inv,
            # then argmin the (full) trace. Batched inv is element-wise identical
            # to the per-omega loop and argmin returns the first minimum, matching
            # the old strict `t < best_trace` loop — result unchanged, ~50x fewer
            # Python-level inversions. Rare singular case falls back to the loop.
            ws = np.linspace(0.01, 0.99, 50)
            try:
                M = ws[:, None, None] * P1_inv + (1.0 - ws)[:, None, None] * P2_inv
                P_all = np.linalg.inv(M)
                traces = np.trace(P_all, axis1=1, axis2=2)
                best_w = ws[int(np.argmin(traces))]
            except np.linalg.LinAlgError:
                best_w = 0.5
                best_trace = np.inf
                for w in np.linspace(0.01, 0.99, 50):
                    try:
                        P_fused = np.linalg.inv(w * P1_inv + (1 - w) * P2_inv)
                        t = np.trace(P_fused)
                        if t < best_trace:
                            best_trace = t
                            best_w = w
                    except np.linalg.LinAlgError:
                        continue

        try:
            P_fused_inv = best_w * P1_inv + (1 - best_w) * P2_inv
            P_fused = np.linalg.inv(P_fused_inv)
            P_fused = (P_fused + P_fused.T) / 2

            x1 = np.asarray(pred_mean).flatten()
            x2 = np.asarray(meas_mean).flatten()
            x_fused = P_fused @ (best_w * P1_inv @ x1
                                  + (1 - best_w) * P2_inv @ x2)
            return x_fused, P_fused
        except np.linalg.LinAlgError:
            return None, None

    def get_or_create_filter(self, track_id: Any, measurement: Any) -> None:
        """Get existing filter components or create new ones for a track.
        
        Always succeeds: NaN measurements are replaced with zeros and
        high-uncertainty covariance before filter creation.
        """
        if track_id not in self.priors:
            # Replace NaN in measurement with 0 and use high-uncertainty covariance
            if np.isnan(measurement.state_vector).any():
                logging.warning(f" [{self.origin_dataset}]: Measurement contains NaN for track {track_id} - replacing with zeros and using high-uncertainty covariance")
                measurement = Detection(
                    np.nan_to_num(measurement.state_vector, nan=0.0),
                    timestamp=measurement.timestamp,
                    measurement_model=measurement.measurement_model
                )
            
            # Create new filter components with environment-aware q
            q = self._get_q(track_id)
            transition_model = CombinedLinearGaussianTransitionModel([
                ConstantVelocity(q),
                ConstantVelocity(q),
                ConstantVelocity(q)
            ])
            
            # Clip noise covariance to reasonable bounds
            noise_covar = measurement.measurement_model.noise_covar.copy()
            noise_covar = np.clip(noise_covar, -1e10, 1e10)
            for i in range(noise_covar.shape[0]):
                if noise_covar[i, i] <= 0 or noise_covar[i, i] > 1e10:
                    noise_covar[i, i] = 1e6
            
            measurement_model = LinearGaussian(
                ndim_state=6,
                mapping=measurement.measurement_model.mapping,
                noise_covar=noise_covar
            )
            
            predictor = KalmanPredictor(transition_model)
            updater = KalmanUpdater(measurement_model)
            
            # Create new track
            self.tracks[track_id] = Track()
            
            # Create new prior from measurement
            if len(measurement.state_vector) == 6:
                # 6D measurement: position + velocity
                initial_state_vector = np.array([
                    measurement.state_vector[0],
                    measurement.state_vector[1],
                    measurement.state_vector[2],
                    measurement.state_vector[3],
                    measurement.state_vector[4],
                    measurement.state_vector[5]
                ])
            else:
                # Position-only measurement: map into interleaved state, velocity = 0
                initial_state_vector = np.array([
                    measurement.state_vector[0],  # x
                    0.,                            # vx
                    measurement.state_vector[1],  # y
                    0.,                            # vy
                    measurement.state_vector[2],  # z
                    0.                             # vz
                ])
            prior = GaussianState(
                initial_state_vector,
                np.diag([1.e7, 1.e7, 1.e7, 1.e7, 1.e7, 1.e7]),
                timestamp=measurement.timestamp
            )
            self.priors[track_id] = prior
            self.tracks[track_id].append(prior)
            self.predictor[track_id] = predictor
            self.updater[track_id] = updater
            self.measurement_model[track_id] = measurement_model

    def _predict(self, track_id: Any, current_state: Any, timestamp: Any) -> Any:
        """Kalman time-update. StoneSoup implementation; overridden by
        NumpyKalmanFilterManager with an equivalent numpy kernel."""
        return self.predictor[track_id].predict(current_state, timestamp=timestamp)

    def _standard_update(self, track_id: Any, prediction: Any, measurement: Any) -> Any:
        """Kalman measurement-update. StoneSoup implementation; overridden by
        NumpyKalmanFilterManager with an equivalent numpy kernel."""
        hypothesis = SingleHypothesis(prediction, measurement)
        return self.updater[track_id].update(hypothesis)

    def predict_and_update(self, track_id: Any, measurement: Any) -> Any:
        """Predict forward to measurement time, then update with measurement."""
        self.get_or_create_filter(track_id, measurement)
        
        current_state = self.priors[track_id]
        
        # Check if measurement is too old compared to current state
        time_diff = (current_state.timestamp - measurement.timestamp).total_seconds()
        if time_diff > 0:
            logging.warning(f" [{self.origin_dataset}]: Skipping measurement for track {track_id} - too old: {time_diff:.1f} seconds behind current state")
            return current_state
        
        # Check for large time jumps that can cause numerical instability
        abs_time_diff = abs((measurement.timestamp - current_state.timestamp).total_seconds())
        if abs_time_diff > 15 * 60:  # 15 minutes
            logging.warning(f" [{self.origin_dataset}]: Large time jump ({abs_time_diff:.1f}s) for track {track_id} - resetting filter")
            self._reset_filter(track_id, measurement)
            return self.priors[track_id]
        
        # Check if current state has NaN (corrupted filter) - reset if so
        if np.isnan(current_state.state_vector).any() or np.isnan(current_state.covar).any():
            logging.warning(f" [{self.origin_dataset}]: Current state contains NaN for track {track_id} - resetting filter")
            self._reset_filter(track_id, measurement)
            return self.priors[track_id]
        
        try:
            # Predict and update
            prediction = self._predict(track_id, self.priors[track_id], measurement.timestamp)
            
            # Check prediction for NaN or covariance explosion
            if np.isnan(prediction.state_vector).any() or np.isnan(prediction.covar).any():
                logging.warning(f" [{self.origin_dataset}]: Prediction produced NaN for track {track_id} - resetting filter")
                self._reset_filter(track_id, measurement)
                return self.priors[track_id]
            
            max_covar = np.max(np.abs(prediction.covar))
            if max_covar > 1e15:
                logging.warning(f" [{self.origin_dataset}]: Covariance explosion ({max_covar:.2e}) for track {track_id} - resetting filter")
                self._reset_filter(track_id, measurement)
                return self.priors[track_id]
            
            if self.fusion_method == 'ci':
                pred_sv = np.asarray(prediction.state_vector).flatten()
                pred_cov = np.asarray(prediction.covar)

                if len(measurement.state_vector) == 6:
                    ci_meas_mean = np.asarray(measurement.state_vector).flatten()
                    ci_meas_covar = np.asarray(measurement.measurement_model.noise_covar)
                else:
                    # Lift 3D position-only to 6D: borrow predicted velocity
                    # with env-aware variance so CI mostly ignores it
                    vel_var = self._get_vel_var(track_id)
                    m = np.asarray(measurement.state_vector).flatten()
                    R3 = np.asarray(measurement.measurement_model.noise_covar)
                    ci_meas_mean = np.array([
                        m[0], pred_sv[1],
                        m[1], pred_sv[3],
                        m[2], pred_sv[5]
                    ])
                    ci_meas_covar = np.zeros((6, 6))
                    ci_meas_covar[0, 0] = R3[0, 0]
                    ci_meas_covar[0, 2] = ci_meas_covar[2, 0] = R3[0, 1]
                    ci_meas_covar[0, 4] = ci_meas_covar[4, 0] = R3[0, 2]
                    ci_meas_covar[2, 2] = R3[1, 1]
                    ci_meas_covar[2, 4] = ci_meas_covar[4, 2] = R3[1, 2]
                    ci_meas_covar[4, 4] = R3[2, 2]
                    ci_meas_covar[1, 1] = vel_var
                    ci_meas_covar[3, 3] = vel_var
                    ci_meas_covar[5, 5] = vel_var

                x_fused, P_fused = self._covariance_intersection(
                    pred_sv, pred_cov, ci_meas_mean, ci_meas_covar)
                if x_fused is not None:
                    post = GaussianState(
                        state_vector=x_fused.reshape(-1, 1),
                        covar=P_fused,
                        timestamp=measurement.timestamp)
                else:
                    post = self._standard_update(track_id, prediction, measurement)
            else:
                post = self._standard_update(track_id, prediction, measurement)
            
            # Check for NaN in result
            if np.isnan(post.state_vector).any() or np.isnan(post.covar).any():
                logging.warning(f" [{self.origin_dataset}]: Kalman update produced NaN for track {track_id} - resetting filter")
                self._reset_filter(track_id, measurement)
                return self.priors[track_id]
            
            # Update prior
            self.priors[track_id] = post
            return post
            
        except Exception as e:
            logging.error(f" [{self.origin_dataset}]: Exception in Kalman update for track {track_id}: {e}")
            self._reset_filter(track_id, measurement)
            return self.priors[track_id]


class NumpyKalmanFilterManager(KalmanFilterManager):
    """Drop-in replacement for :class:`KalmanFilterManager` that runs the Kalman
    predict/update with plain numpy instead of StoneSoup.

    Entity-tracker analogue of object_manager's NumpyKalmanFilterManager. The
    predict/update is a small fixed-size 6-state constant-velocity Kalman filter;
    StoneSoup's per-event object plumbing dominates tracker cost. This subclass
    overrides ONLY the numeric kernel (get_or_create_filter / _predict /
    _standard_update); all control flow, guards (stale/out-of-order, 15-min
    time-jump reset, NaN and covariance-explosion resets), environment-aware
    process noise, and the covariance-intersection fusion path are inherited
    unchanged, so behaviour matches StoneSoup by construction (verified in
    tests/unit/test_entity_tracker_kalman_equivalence.py). State is stored as
    StoneSoup ``GaussianState`` in ``self.priors`` exactly as the base class does.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Process-noise PSD fixed per track at filter-creation time, mirroring
        # StoneSoup building the transition model once in get_or_create_filter.
        self._q_by_track: Dict[Any, float] = {}

    @staticmethod
    def _cv_transition(dt: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
        """Constant-velocity transition F and process noise Q for interleaved
        state [x, vx, y, vy, z, vz] over interval dt — the block-diagonal form of
        StoneSoup's CombinedLinearGaussianTransitionModel([CV(q)] * 3)."""
        F = np.eye(6)
        F[0, 1] = F[2, 3] = F[4, 5] = dt
        d = abs(dt)  # StoneSoup ConstantVelocity.covar uses abs(dt)
        qb = q * np.array([[d ** 3 / 3.0, d ** 2 / 2.0],
                           [d ** 2 / 2.0, d]])
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = qb
        Q[2:4, 2:4] = qb
        Q[4:6, 4:6] = qb
        return F, Q

    @staticmethod
    def _measurement_matrix(mapping) -> np.ndarray:
        """Selection matrix H mapping the 6-state vector onto the measured dims."""
        H = np.zeros((len(mapping), 6))
        H[np.arange(len(mapping)), list(mapping)] = 1.0
        return H

    def get_or_create_filter(self, track_id: Any, measurement: Any) -> None:
        """Create numpy filter state for a new track. Mirrors the base StoneSoup
        version: NaN->0 measurement, diag(1e7) initial covariance, velocity 0 for
        position-only measurements."""
        if track_id not in self.priors:
            noise_covar = np.asarray(measurement.measurement_model.noise_covar, dtype=float)
            noise_covar = np.clip(noise_covar, -1e10, 1e10)
            for i in range(noise_covar.shape[0]):
                if noise_covar[i, i] <= 0 or noise_covar[i, i] > 1e10:
                    noise_covar[i, i] = 1e6
            self.measurement_model[track_id] = LinearGaussianModel(
                measurement.measurement_model.mapping, noise_covar)
            m = np.asarray(measurement.state_vector, dtype=float).reshape(-1)
            if np.isnan(m).any():
                logging.warning(f" [{self.origin_dataset}]: Measurement contains NaN for track {track_id} - replacing with zeros and using high-uncertainty covariance")
                m = np.nan_to_num(m, nan=0.0)
            self._q_by_track[track_id] = self._get_q(track_id)
            self.tracks[track_id] = Track()
            if m.shape[0] == 6:
                mean = m
            else:
                mean = np.array([m[0], 0., m[1], 0., m[2], 0.])
            prior = GaussianState(mean.reshape(-1, 1), np.diag([1.e7] * 6),
                                  timestamp=measurement.timestamp)
            self.priors[track_id] = prior
            self.tracks[track_id].append(prior)

    def _predict(self, track_id: Any, current_state: Any, timestamp: Any) -> Any:
        mean = np.asarray(current_state.state_vector, dtype=float).reshape(-1)
        cov = np.asarray(current_state.covar, dtype=float)
        dt = (timestamp - current_state.timestamp).total_seconds()
        q = self._q_by_track.get(track_id)
        if q is None:  # preloaded track — derive lazily from its environment
            q = self._q_by_track[track_id] = self._get_q(track_id)
        F, Q = self._cv_transition(dt, q)
        return GaussianState((F @ mean).reshape(-1, 1), F @ cov @ F.T + Q,
                             timestamp=timestamp)

    def _standard_update(self, track_id: Any, prediction: Any, measurement: Any) -> Any:
        pred_mean = np.asarray(prediction.state_vector, dtype=float).reshape(-1)
        pred_cov = np.asarray(prediction.covar, dtype=float)
        z = np.asarray(measurement.state_vector, dtype=float).reshape(-1)
        H = self._measurement_matrix(measurement.measurement_model.mapping)
        R = np.asarray(measurement.measurement_model.noise_covar, dtype=float)
        # Non-Joseph Kalman update, matching stonesoup KalmanUpdater exactly:
        #   S = H P Hᵀ + R;  K = P Hᵀ S⁻¹;  P⁺ = P − K S Kᵀ
        S = H @ pred_cov @ H.T + R
        K = pred_cov @ H.T @ np.linalg.inv(S)
        post_mean = pred_mean + K @ (z - H @ pred_mean)
        post_cov = pred_cov - K @ S @ K.T
        return GaussianState(post_mean.reshape(-1, 1), post_cov,
                             timestamp=measurement.timestamp)

    # ------------------------------------------------------------------
    # Stage 2: array-native batched path (no per-row Detection/LinearGaussian)
    # ------------------------------------------------------------------
    def _fresh_prior(self, track_id: Any, meas_vec: np.ndarray, timestamp: Any) -> GaussianState:
        """Create (or reset to) a fresh prior from a measurement array: diag(1e7)
        covariance, velocity 0 for position-only measurements, NaN->0. Array
        analogue of get_or_create_filter's initial state, used for the in-line
        resets in _update_core."""
        m = np.nan_to_num(np.asarray(meas_vec, dtype=float).reshape(-1), nan=0.0)
        if m.shape[0] == 6:
            mean = m
        else:
            mean = np.array([m[0], 0., m[1], 0., m[2], 0.])
        prior = GaussianState(mean.reshape(-1, 1), np.diag([1.e7] * 6), timestamp=timestamp)
        self.priors[track_id] = prior
        self.tracks[track_id] = Track()
        self.tracks[track_id].append(prior)
        self._q_by_track[track_id] = self._get_q(track_id)
        return prior

    def _reset_filter_np(self, track_id: Any, meas_vec: np.ndarray, timestamp: Any) -> GaussianState:
        """Array-based reset mirroring the base _reset_filter: drop the filter's
        stores, recreate a fresh prior from the current measurement, and return
        it (the base then returns self.priors[track_id])."""
        for store in (self.priors, self.predictor, self.updater,
                      self.measurement_model, self.tracks):
            store.pop(track_id, None)
        self._q_by_track.pop(track_id, None)
        prior = self._fresh_prior(track_id, meas_vec, timestamp)
        logging.warning(f" [{self.origin_dataset}]: Reset Kalman filter for track {track_id} due to numerical issues")
        return prior

    def _ci_update(self, track_id: Any, pred_mean: np.ndarray, pred_cov: np.ndarray,
                   m: np.ndarray, mapping, noise_covar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Covariance-Intersection fusion mirroring the base predict_and_update CI
        branch (3D measurements lifted to 6D by borrowing the predicted velocity
        with an env-aware variance). Falls back to a standard Kalman update if CI
        fails, matching the base."""
        if m.shape[0] == 6:
            ci_meas_mean = m
            ci_meas_covar = np.asarray(noise_covar, dtype=float)
        else:
            vel_var = self._get_vel_var(track_id)
            R3 = np.asarray(noise_covar, dtype=float)
            ci_meas_mean = np.array([m[0], pred_mean[1], m[1], pred_mean[3], m[2], pred_mean[5]])
            ci_meas_covar = np.zeros((6, 6))
            ci_meas_covar[0, 0] = R3[0, 0]
            ci_meas_covar[0, 2] = ci_meas_covar[2, 0] = R3[0, 1]
            ci_meas_covar[0, 4] = ci_meas_covar[4, 0] = R3[0, 2]
            ci_meas_covar[2, 2] = R3[1, 1]
            ci_meas_covar[2, 4] = ci_meas_covar[4, 2] = R3[1, 2]
            ci_meas_covar[4, 4] = R3[2, 2]
            ci_meas_covar[1, 1] = vel_var
            ci_meas_covar[3, 3] = vel_var
            ci_meas_covar[5, 5] = vel_var
        x_fused, P_fused = self._covariance_intersection(pred_mean, pred_cov, ci_meas_mean, ci_meas_covar)
        if x_fused is not None:
            return x_fused, P_fused
        # CI failed -> standard Kalman fallback (matches base)
        H = self._measurement_matrix(mapping)
        R = np.asarray(noise_covar, dtype=float)
        S = H @ pred_cov @ H.T + R
        K = pred_cov @ H.T @ np.linalg.inv(S)
        return pred_mean + K @ (m - H @ pred_mean), pred_cov - K @ S @ K.T

    def _update_core(self, track_id: Any, meas_vec: np.ndarray, mapping,
                     noise_covar: np.ndarray, timestamp: Any) -> GaussianState:
        """Array-native predict+update mirroring the inherited predict_and_update
        (same guard order and thresholds) without constructing a StoneSoup
        Detection/LinearGaussian, so it can be driven from bulk-extracted arrays by
        process_measurements_batch. Returns the posterior GaussianState and updates
        self.priors exactly as predict_and_update does."""
        origin_dataset = self.origin_dataset
        m = np.asarray(meas_vec, dtype=float).reshape(-1)
        if track_id not in self.priors:
            self._fresh_prior(track_id, m, timestamp)

        current = self.priors[track_id]
        cur_mean = np.asarray(current.state_vector, dtype=float).reshape(-1)
        cur_cov = np.asarray(current.covar, dtype=float)
        cur_ts = current.timestamp

        # Out-of-order / stale duplicate: a newer measurement already advanced this
        # track's filter (aggregated per batch, logged once by the caller).
        stale_diff = (cur_ts - timestamp).total_seconds()
        if stale_diff > 0:
            logging.debug(f" [{origin_dataset}]: Skipping measurement for track {track_id} - too old: {stale_diff:.1f} seconds behind current state")
            self._stale_skip_count = getattr(self, '_stale_skip_count', 0) + 1
            if stale_diff > getattr(self, '_stale_skip_max', 0.0):
                self._stale_skip_max = stale_diff
            return current

        abs_time_diff = abs((timestamp - cur_ts).total_seconds())
        if abs_time_diff > 15 * 60:
            logging.warning(f" [{origin_dataset}]: Large time jump ({abs_time_diff:.1f}s) for track {track_id} - resetting filter")
            return self._reset_filter_np(track_id, m, timestamp)

        if np.isnan(cur_mean).any() or np.isnan(cur_cov).any():
            logging.warning(f" [{origin_dataset}]: Current state contains NaN for track {track_id} - resetting filter")
            return self._reset_filter_np(track_id, m, timestamp)

        try:
            prediction = self._predict(track_id, current, timestamp)
            pred_mean = np.asarray(prediction.state_vector, dtype=float).reshape(-1)
            pred_cov = np.asarray(prediction.covar, dtype=float)

            if np.isnan(pred_mean).any() or np.isnan(pred_cov).any():
                logging.warning(f" [{origin_dataset}]: Prediction produced NaN for track {track_id} - resetting filter")
                return self._reset_filter_np(track_id, m, timestamp)
            max_covar = np.max(np.abs(pred_cov))
            if max_covar > 1e15:
                logging.warning(f" [{origin_dataset}]: Covariance explosion ({max_covar:.2e}) for track {track_id} - resetting filter")
                return self._reset_filter_np(track_id, m, timestamp)

            if self.fusion_method == 'ci':
                post_mean, post_cov = self._ci_update(track_id, pred_mean, pred_cov, m, mapping, noise_covar)
            else:
                H = self._measurement_matrix(mapping)
                R = np.asarray(noise_covar, dtype=float)
                # Non-Joseph Kalman update, matching stonesoup KalmanUpdater exactly:
                #   S = H P Hᵀ + R;  K = P Hᵀ S⁻¹;  P⁺ = P − K S Kᵀ
                S = H @ pred_cov @ H.T + R
                K = pred_cov @ H.T @ np.linalg.inv(S)
                post_mean = pred_mean + K @ (m - H @ pred_mean)
                post_cov = pred_cov - K @ S @ K.T

            if np.isnan(post_mean).any() or np.isnan(post_cov).any():
                logging.warning(f" [{origin_dataset}]: Kalman update produced NaN for track {track_id} - resetting filter")
                return self._reset_filter_np(track_id, m, timestamp)

            post = GaussianState(post_mean.reshape(-1, 1), post_cov, timestamp=timestamp)
            self.priors[track_id] = post
            return post
        except Exception as e:
            logging.error(f" [{origin_dataset}]: Exception in Kalman update for track {track_id}: {e}")
            return self._reset_filter_np(track_id, m, timestamp)

    def process_measurements_batch(self, records: List[dict],
                                   config: Dict[str, Any]) -> Dict[int, GaussianState]:
        """Vectorized equivalent of the per-row measurement-build + predict_and_update
        loop in process_with_kalman for the numpy backend.

        Extracts needed fields from JSON records, parses timestamps, then applies
        each measurement via _update_core. Rows with NaN measured positions are
        skipped, exactly like the previous tracker path.
        """
        posteriors: Dict[int, GaussianState] = {}
        n = len(records)
        if n == 0:
            return posteriors
        origin_dataset = config.get('origin_dataset', 'unknown')
        self._stale_skip_count = 0
        self._stale_skip_max = 0.0

        now_dt = dt.utcnow()

        ebt = self.environment_by_track
        map6 = (0, 1, 2, 3, 4, 5)
        map3 = (0, 2, 4)
        for i in range(n):
            record = records[i]
            track_id = _record_value(record, 'trackId')

            # Set environment for this track (first non-empty), before the q /
            # vel_var lookups, exactly as the per-row loop does.
            e = _record_value(record, ('identity', 'environment'))
            if not e:
                e = _record_value(record, 'environment')
            if e and track_id not in ebt:
                ebt[track_id] = str(e)

            def number(keys: tuple[str, ...], default: float = np.nan) -> float:
                value = _record_value(record, keys, default)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            x = number(('ecefPosition', 'x'))
            y = number(('ecefPosition', 'y'))
            z = number(('ecefPosition', 'z'))
            _vx = number(('ecefVelocity', 'x'))
            _vy = number(('ecefVelocity', 'y'))
            _vz = number(('ecefVelocity', 'z'))
            has_vel = not (_vx != _vx or _vy != _vy or _vz != _vz)

            pcxx = number(('positionCovariance', 'xx'))
            if pcxx == pcxx:
                pxx = pcxx
                pyy = number(('positionCovariance', 'yy'), 1.e7)
                pzz = number(('positionCovariance', 'zz'), 1.e7)
                pxy = number(('positionCovariance', 'xy'), 0.0)
                pxz = number(('positionCovariance', 'xz'), 0.0)
                pyz = number(('positionCovariance', 'yz'), 0.0)
            else:
                pxx = pyy = pzz = 1.e7
                pxy = pxz = pyz = 0.0

            timestamp_value = _record_value(
                record, ('estimatedKinematics', 'kinematicsTimestamp'))
            try:
                ts_i = dt.strptime(str(timestamp_value), "%Y-%m-%dT%H:%M:%S.%fZ")
            except (TypeError, ValueError):
                ts_i = now_dt
            if has_vel:
                vcxx = number(('velocityCovariance', 'dxdx'))
                if vcxx == vcxx:
                    vxx = vcxx
                    vyy = number(('velocityCovariance', 'dydy'), 1.e7)
                    vzz = number(('velocityCovariance', 'dzdz'), 1.e7)
                    vxy = number(('velocityCovariance', 'dxdy'), 0.0)
                    vxz = number(('velocityCovariance', 'dxdz'), 0.0)
                    vyz = number(('velocityCovariance', 'dydz'), 0.0)
                else:
                    vv = self._get_vel_var(track_id)
                    vxx = vyy = vzz = vv
                    vxy = vxz = vyz = 0.0
                nc = np.zeros((6, 6))
                nc[0, 0] = pxx; nc[2, 2] = pyy; nc[4, 4] = pzz
                nc[0, 2] = nc[2, 0] = pxy; nc[0, 4] = nc[4, 0] = pxz; nc[2, 4] = nc[4, 2] = pyz
                nc[1, 1] = vxx; nc[3, 3] = vyy; nc[5, 5] = vzz
                nc[1, 3] = nc[3, 1] = vxy; nc[1, 5] = nc[5, 1] = vxz; nc[3, 5] = nc[5, 3] = vyz
                # Entity-specific noise sanitization (matches the 6D branch of
                # process_with_kalman): NaN->0, clip, and floor bad diagonals.
                nc = np.nan_to_num(nc, nan=0.0)
                nc = np.clip(nc, -1e10, 1e10)
                for k in range(6):
                    if nc[k, k] <= 0 or nc[k, k] > 1e10:
                        nc[k, k] = 1e6
                meas = np.array([x, _vx, y, _vy, z, _vz])
                if np.isnan(meas).any():
                    logging.warning(f" [{origin_dataset}]: Skipping measurement for track {track_id} - contains NaN values")
                    continue
                posteriors[i] = self._update_core(track_id, meas, map6, nc, ts_i)
            else:
                pc = np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
                meas = np.array([x, y, z])
                if np.isnan(meas).any():
                    logging.warning(f" [{origin_dataset}]: Skipping measurement for track {track_id} - contains NaN values")
                    continue
                posteriors[i] = self._update_core(track_id, meas, map3, pc, ts_i)

        if self._stale_skip_count:
            logging.info(f" [{origin_dataset}]: Skipped {self._stale_skip_count} stale/out-of-order "
                         f"measurement(s) this batch (max {self._stale_skip_max:.1f}s behind)")
        return posteriors


NUM_TRACKER_WORKERS = 2


# Records per fan-out message (bounds pickle size through the shard queues).
_HEAD_FANOUT_CHUNK = 5000

# Bound the shard queues so a slow/dead shard or a large head preload can't pile
# an unbounded copy of the head data into the queue: .put() blocks (backpressure)
# until the shard drains. The cap is expressed in RECORDS
# (CRUCIBLE_SHARD_QUEUE_MAX_RECORDS, default 100000) and converted to a queue
# depth in MESSAGES via the fan-out chunk size, so the two knobs stay consistent
# (in-flight per shard ≈ maxsize × _HEAD_FANOUT_CHUNK). CRUCIBLE_SHARD_QUEUE_MAXSIZE
# overrides the message depth directly.
_SHARD_QUEUE_MAX_RECORDS = int(os.getenv('CRUCIBLE_SHARD_QUEUE_MAX_RECORDS', '100000'))
_SHARD_QUEUE_MAXSIZE = int(os.getenv(
    'CRUCIBLE_SHARD_QUEUE_MAXSIZE',
    str(max(4, -(-_SHARD_QUEUE_MAX_RECORDS // _HEAD_FANOUT_CHUNK)))))


def _fanout_head_preload_tracker(shard_queues: List[Any], config: dict, n_shards: int,
                                 head_cache: Optional[Dict[str, List[dict]]] = None) -> None:
    """Download component track heads ONCE and fan them out to the tracker shard
    workers' queues (pre-bucketed by trackId to match the dispatcher routing).

    Replaces the per-shard ``select * ... limit 10000`` query in
    ``initialize_from_heads`` (which N-duplicated the read load and 504'd on
    large head datasets). The parent buckets by ``_shard_for_track(trackId)``
    (the same stable md5 routing ``shard_dispatcher`` uses); each shard still
    filters to its own feed via the identity-derived trackId. Ends with a
    ``done`` marker so a shard that owns no heads still initializes.
    ``head_cache`` avoids re-downloading a dataset shared by multiple feeds.
    """
    if skip_head_preload(config):
        for s in range(n_shards):
            shard_queues[s].put({'type': 'init_heads', 'heads': [], 'done': True})
        logging.info(
            f"Head preload SKIPPED (skip_head_preload): {n_shards} tracker shard(s) start "
            f"with empty Kalman state; deterministic trackIds keep heads consistent.")
        return

    head_ds = config.get('component_track_head_dataset')
    buckets: List[List[dict]] = [[] for _ in range(n_shards)]

    if head_ds:
        if head_cache is not None and head_ds in head_cache:
            recs = head_cache[head_ds]
        else:
            recs = load_all_records_for_preload(
                head_ds, config, rc, auth, "tracker_head_preload_limit"
            )
            if head_cache is not None:
                head_cache[head_ds] = recs
        if recs:
            for row in recs:
                if not isinstance(row, dict):
                    continue
                tid = _record_value(row, 'trackId')
                if isinstance(tid, dict):
                    tid = tid.get('uuid')
                if _isna(tid):
                    continue
                shard = _shard_for_track(tid, n_shards)
                buckets[shard].append(row)
            logging.info(
                f"Head preload: {sum(len(b) for b in buckets)} track heads from "
                f"{head_ds} bucketed across {n_shards} shards")

    for s in range(n_shards):
        b = buckets[s]
        for i in range(0, len(b), _HEAD_FANOUT_CHUNK):
            shard_queues[s].put({'type': 'init_heads',
                                 'heads': b[i:i + _HEAD_FANOUT_CHUNK], 'done': False})
        shard_queues[s].put({'type': 'init_heads', 'heads': [], 'done': True})
        buckets[s] = []  # release after queueing so the parent doesn't hold all shards' heads


def _run_tracker_shard(shard_queue: Any, config: dict) -> None:
    """Launch a sharded tracker worker that processes a subset of trackIds."""
    asyncio.run(tracker(shard_queue, config))


async def shard_dispatcher(event_queue: Any, shard_queues: List[Any], config: dict) -> None:
    """
    Reads raw Report_Events from the main event_queue, ASSIGNS a deterministic
    trackId to each from its identity fields, then partitions by trackId and
    routes to the appropriate shard queue.

    trackId is a stable uuid5 of "{origin_dataset}:{identity customID}".
    Sharding by trackId keeps each
    track's Kalman state in exactly one shard.
    Runs as an asyncio task in the parent process alongside SSE listeners.
    """
    n_shards = len(shard_queues)
    origin_dataset = config.get('origin_dataset', 'unknown')
    track_id_fields = configured_track_id_fields(config)

    while True:
        # Blocking get on multiprocessing Queue — wrap in to_thread
        events = await asyncio.to_thread(event_queue.get)

        if not events:
            continue

        # Drain entire backlog
        while not event_queue.empty():
            try:
                extra_events = event_queue.get_nowait()
                if isinstance(extra_events, list):
                    events.extend(extra_events)
                else:
                    events.append(extra_events)
            except Exception:
                break

        # Assign a deterministic trackId then partition into shard buckets
        shard_buckets = [[] for _ in range(n_shards)]
        for event in events:
            if not isinstance(event, dict):
                continue
            # Configured dotted paths select fields from the nested report.
            track_id = assign_track_id(event, origin_dataset, track_id_fields)
            event['trackId'] = track_id
            shard_idx = _shard_for_track(track_id, n_shards) if track_id else 0
            shard_buckets[shard_idx].append(event)

        # Push each bucket to its shard queue
        for i, bucket in enumerate(shard_buckets):
            if bucket:
                shard_queues[i].put(bucket)


@async_retry
async def tracker(event_queue: Any,
                  config: dict) -> None:
    # Reinitialize controllers in subprocess to avoid SSL issues with forked connections
    global rc, wc, auth
    rc = None
    wc = None
    auth = None
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)

    if config.get('disabled'):
        logging.info(f" [{config.get('origin_dataset')}]: No tracking for dataset {config.get('origin_dataset')}")
        logging.info(f" [{config.get('origin_dataset')}]:   as it has Entity Stream Manager configuration option disabled set to True")
        return

    # The live tracker is
    # StoneSoup-free: Kalman mode always uses the numpy backend. The previous
    # StoneSoup implementation is preserved in entity_tracker_stonesoup_deprecated.py.
    _use_numpy_kalman = config.get(
        'use_numpy_kalman',
        os.getenv('CRUCIBLE_USE_NUMPY_KALMAN', 'false').strip().lower() in ('1', 'true', 'yes'))
    if not _use_numpy_kalman:
        logging.warning(f" [{config.get('origin_dataset')}]: use_numpy_kalman is false, but the tracker is StoneSoup-free; using numpy backend")
    _manager_cls = NumpyKalmanFilterManager
    filter_manager = _manager_cls(config)
    logging.info(f" [{config.get('origin_dataset')}]: Kalman backend = numpy")

    tracker_mode = config.get('crucible_tracker') or ''
    use_passthrough, use_kalman, use_ci = _parse_tracker_modes(tracker_mode)
    filter_manager.passthrough_tracker = use_passthrough

    if use_kalman:
        ci_label = " with Covariance Intersection" if use_ci else ""
        logging.info(f" [{config.get('origin_dataset')}]: Starting ECEF Kalman Filter{ci_label} for {config.get('origin_dataset')}")
        filter_manager.ecef_kalman_filter = True
    else:
        filter_manager.ecef_kalman_filter = False

    if use_ci:
        filter_manager.fusion_method = 'ci'

    # Receive the one-time head preload fanned out by the parent (run()) instead
    # of querying Crucible per-shard (which N-duplicated the load and 504'd on
    # large head datasets). Event buckets are plain lists; init_heads messages
    # are dicts, so they are distinguishable. Stash any event bucket that arrives
    # before init completes (handled first in the loop below).
    preloaded_heads: List[dict] = []
    stashed_events: List[Any] = []
    _init_done = False
    while not _init_done:
        msg = event_queue.get()
        if isinstance(msg, dict) and msg.get('type') == 'init_heads':
            preloaded_heads.extend(msg.get('heads') or [])
            if msg.get('done'):
                _init_done = True
        elif msg is not None:
            stashed_events.append(msg)

    await filter_manager.initialize_from_heads(config.get('crucible_tracker'), preloaded_heads=preloaded_heads)

    if filter_manager.ecef_kalman_filter:
        logging.info(f" [{config.get('origin_dataset')}]:   fusion={filter_manager.fusion_method}, q_default={filter_manager.q_default}")

        
    if filter_manager.passthrough_tracker:
        logging.info(f" [{config.get('origin_dataset')}]: Starting passthrough tracker for {config.get('origin_dataset')}")
       
    # Event buckets received before the head preload finished (unusual) are
    # processed on the first loop pass.
    startup_events: List[Any] = stashed_events

    while True:

        try:
            if startup_events:
                events = []
                for _b in startup_events:
                    if isinstance(_b, list):
                        events.extend(_b)
                    else:
                        events.append(_b)
                startup_events = []
            else:
                # get the backlog of data received from the SSE listener
                events = event_queue.get()
            # aggregate the events into a single list if the queue has a backlog
            while not event_queue.empty():
                extra_events = event_queue.get()
                if isinstance(extra_events, list):
                    events.extend(extra_events)
                else:
                    events.append(extra_events)
            
            
            # Belt-and-suspenders: a report event with no kinematics timestamp is a
            # position with no observation time. Drop those rows so no timeless
            # track is produced. The transformer normally filters them, but events
            # may arrive from other producers if the transformer is bypassed.
            event_records = [event for event in events if isinstance(event, dict)]
            before_count = len(event_records)
            event_records = [
                event for event in event_records
                if _record_value(
                    event, ('estimatedKinematics', 'kinematicsTimestamp')) not in (None, '')
            ]
            dropped_count = before_count - len(event_records)
            if dropped_count:
                logging.warning(f" [{config.get('origin_dataset')}]: dropped {dropped_count} "
                                f"report event(s) missing estimatedKinematics.kinematicsTimestamp")
            if not event_records:
                continue

            event_records.sort(key=lambda record: str(
                _record_value(record, ('estimatedKinematics', 'kinematicsTimestamp'), '')))
            # The dispatcher already assigned a deterministic trackId to each event.
            # Flag any trackId not seen before (this run or loaded from heads) as new.
            track_ids = {_record_value(event, 'trackId') for event in event_records}
            for track_id in track_ids:
                if track_id not in filter_manager.known_trackids:
                    filter_manager.known_trackids.add(track_id)
                    filter_manager.new_trackIds.add(track_id)

            track_events: List[dict] = []
            track_heads: List[dict] = []
            if filter_manager.passthrough_tracker is True:
                track_events, track_heads = copy_kinematics_to_track(
                    event_records, config, filter_manager)
  

            if filter_manager.ecef_kalman_filter is True and filter_manager.passthrough_tracker is False:
                track_events = process_with_kalman(
                    event_records, config, filter_manager)
                track_heads = []
                for event in track_events:
                    head = clone_record(event)
                    head['trackUpdatedTimestamp'] = head.pop('interceptTimestamp', None)
                    head.pop('trackQuality', None)
                    head['stale'] = get_current_timestamp_string(10)
                    track_heads.append(head)
                track_heads = last_record_by_track(track_heads)
                track_events = add_track_wgs84_kinematics(track_events)
                track_heads = add_track_wgs84_kinematics(track_heads)

            if not filter_manager.ecef_kalman_filter and not filter_manager.passthrough_tracker:
                logging.warning(f" [{config.get('origin_dataset')}]: No tracker mode configured for {config.get('origin_dataset')} - skipping batch")
                continue
                
            if filter_manager.passthrough_tracker is True:
                track_heads = last_record_by_track(track_heads)

            if filter_manager.ecef_kalman_filter is True or filter_manager.passthrough_tracker is True:

                _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
                event_chunk_size = int(config['batch_write_chunk_size'])
                heads_chunk_size = int(config['batch_update_chunk_size'])
                max_concurrent = config.get('batch_write_max_concurrent')
                _collect_finished_head_update_tasks(filter_manager, config.get('origin_dataset'))
                head_dataset = config['component_track_head_dataset']
                if (not filter_manager.pending_head_update_tasks
                        and filter_manager.pending_head_updates_by_track):
                    coalesced_updates = list(
                        filter_manager.pending_head_updates_by_track.values())
                    filter_manager.pending_head_updates_by_track.clear()
                    logging.info(
                        f" [{config.get('origin_dataset')}]: scheduled coalesced background "
                        f"existing track-head update: records={len(coalesced_updates)}")
                    filter_manager.pending_head_update_tasks.append(
                        asyncio.create_task(_best_effort_update_track_heads(
                            coalesced_updates, head_dataset, heads_chunk_size,
                            config.get('origin_dataset'), max_concurrent))
                    )

                new_tracks = [
                    head for head in track_heads
                    if _record_value(head, 'trackId') in filter_manager.new_trackIds
                ]
                existing_tracks = [
                    head for head in track_heads
                    if _record_value(head, 'trackId') not in filter_manager.new_trackIds
                ]

                # Write NEW track heads FIRST so we never emit component track
                # events that reference a head that failed to create. Any head
                # that fails is kept in new_trackIds for retry and its events are
                # withheld this cycle (re-sent once the head lands next cycle).
                failed_trackIds = set()
                if new_tracks:
                    new_tracks_json = _write_records(new_tracks)
                    failed_records, _ = await write_batch_chunked(
                        new_tracks_json,
                        head_dataset,
                        wc.write_record_batch_by_name,
                        heads_chunk_size,
                        label=f' [{config.get("origin_dataset")}]: ',
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent,
                    )
                    for rec in failed_records:
                        tid = rec.get('trackId')
                        if tid:
                            failed_trackIds.add(tid)
                    if failed_trackIds:
                        logging.warning(f' [{config.get("origin_dataset")}]: {len(failed_trackIds)} new track heads failed to write; withholding their events and keeping in new_trackIds for retry')

                # Only clear new_trackIds for heads that were actually written
                successfully_written = {
                    _record_value(record, 'trackId') for record in new_tracks
                }
                successfully_written -= failed_trackIds
                filter_manager.new_trackIds -= successfully_written

                # Existing head updates remain background/best-effort so they
                # do not hold up the live event stream, but failed records are
                # returned to the per-track pending buffer for retry.
                if existing_tracks:
                    existing_tracks = coalesce_existing_head_updates(
                        existing_tracks,
                        filter_manager.existing_head_update_emit_time_by_track,
                        _head_update_interval_seconds(config),
                        label='component',
                        log_prefix=f" [{config.get('origin_dataset')}]: ")
                    existing_tracks_json = _write_records(existing_tracks)
                    coalesced = buffer_latest_records_by_track(
                        filter_manager.pending_head_updates_by_track,
                        existing_tracks_json)
                    if coalesced:
                        logging.info(
                            f" [{config.get('origin_dataset')}]: coalesced buffered existing "
                            f"track-head updates; coalesced={coalesced} "
                            f"pending={len(filter_manager.pending_head_updates_by_track)}")
                    if (not filter_manager.pending_head_update_tasks
                            and filter_manager.pending_head_updates_by_track):
                        coalesced_updates = list(
                            filter_manager.pending_head_updates_by_track.values())
                        filter_manager.pending_head_updates_by_track.clear()
                        filter_manager.pending_head_update_tasks.append(
                            asyncio.create_task(_best_effort_update_track_heads(
                                coalesced_updates, head_dataset, heads_chunk_size,
                                config.get('origin_dataset'), max_concurrent))
                        )

                # Now write track events, EXCLUDING any whose new head failed to
                # create this cycle (avoids orphan events with no head).
                writable_events = []
                for event, source in zip(track_events, event_records):
                    output = clone_record(event)
                    source_uuid = _record_value(source, ('source', 'uuid'))
                    if source_uuid is not None:
                        output['reportIds'] = [source_uuid]
                    if _record_value(output, 'trackId') not in failed_trackIds:
                        writable_events.append(output)

                pending = writable_events
                while pending:
                    seen = set()
                    unique_records = []
                    remainder = []
                    for event in pending:
                        track_id = _record_value(event, 'trackId')
                        if track_id in seen:
                            remainder.append(event)
                        else:
                            seen.add(track_id)
                            unique_records.append(event)
                    unique_json = _write_records(unique_records)
                    dataset_name = config['component_track_event_dataset']
                    await write_batch_chunked(
                        unique_json,
                        dataset_name,
                        wc.write_record_batch_by_name,
                        event_chunk_size,
                        label=f' [{config.get("origin_dataset")}]: ',
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent,
                    )
                    pending = remainder

                logging.info(f" [{config.get('origin_dataset')}]: Wrote track events for {len(track_heads)} track heads"
                             f"{f' ({len(failed_trackIds)} new heads withheld)' if failed_trackIds else ''}")

        except Exception as e:
            logging.error(f" [{config.get('origin_dataset')}]: An error occurred in tracker: {e}")
            logging.error(traceback.format_exc())
            continue

async def sse_listener_launcher(event_queue: Any,
                                config: dict) -> None:
    global auth, rc, wc
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)

    coroutine_list = []
    # SQL query that filters on origin dataset--this allows us to treat each source differently.
    # The dispatcher assigns a deterministic trackId to raw Report_Events.
    sql_query = "SELECT * FROM "+config.get('report_event_dataset')+" WHERE "+config.get('report_event_dataset')+".source.datasetName = '"+config.get('origin_dataset')+"'"

    logging.info(f" [{config.get('origin_dataset')}]: Tracker SSE Listener SQL Query: "+sql_query)
    coroutine_list.append(
         SSE_listener(sql_query ,event_queue, auth)
    )

    # note: nothing is returned from these coroutines except for exceptions
    excepts = await asyncio.gather(*coroutine_list, return_exceptions=True) # this is a blocking call
    for exc in excepts:
        if isinstance(exc, Exception):
            logging.error("An error occurred during an API call: ")
            logging.error(f"{exc}")


def process_with_kalman(records: List[dict], config: Dict[str, Any],
                        filter_manager: KalmanFilterManager) -> List[dict]:
    """Process native nested report records with the NumPy Kalman filter."""
    if not isinstance(filter_manager, NumpyKalmanFilterManager):
        raise RuntimeError("entity_tracker.py requires NumpyKalmanFilterManager")
    posteriors = filter_manager.process_measurements_batch(records, config)
    track_rows: List[dict] = []
    for index, posterior in posteriors.items():
        source = records[index]
        identity = _record_value(source, 'identity', {})
        track_row: Dict[str, Any] = {
            'trackId': _record_value(source, 'trackId'),
            'standardIdentity': _record_value(source, ('identity', 'standard')),
            'environment': _record_value(source, ('identity', 'environment')),
            'trackOriginatedTimestamp': _record_value(source, ('crucibleHeader', 'createdDate')),
            'trackQuality': _record_value(source, 'trackQuality'),
            'interceptTimestamp': _record_value(
                source, ('estimatedKinematics', 'kinematicsTimestamp')),
            'edhControlSet': clone_record({'value': _record_value(source, 'edhControlSet')})['value'],
            'mode': _record_value(source, 'mode'),
            'entityStatus': _record_value(source, 'entityStatus'),
            'identity': clone_record(identity) if isinstance(identity, dict) else {},
        }
        state = np.asarray(posterior.state_vector, dtype=float).reshape(-1)
        covariance = np.asarray(posterior.covar, dtype=float)
        track_row['ecefPosition'] = {
            'x': state[0], 'y': state[2], 'z': state[4],
        }
        track_row['ecefVelocity'] = {
            'x': state[1], 'y': state[3], 'z': state[5],
        }
        track_row['positionCovariance'] = {
            'xx': covariance[0, 0], 'xy': covariance[0, 2],
            'xz': covariance[0, 4], 'yy': covariance[2, 2],
            'yz': covariance[2, 4], 'zz': covariance[4, 4],
        }
        track_row['velocityCovariance'] = {
            'dxdx': covariance[1, 1], 'dxdy': covariance[1, 3],
            'dxdz': covariance[1, 5], 'dydy': covariance[3, 3],
            'dydz': covariance[3, 5], 'dzdz': covariance[5, 5],
        }
        track_row['positionVelocityCovariance'] = {
            'xdx': covariance[0, 1], 'xdy': covariance[0, 3],
            'xdz': covariance[0, 5], 'ydx': covariance[2, 1],
            'ydy': covariance[2, 3], 'ydz': covariance[2, 5],
            'zdx': covariance[4, 1], 'zdy': covariance[4, 3],
            'zdz': covariance[4, 5],
        }
        track_rows.append(track_row)
    return track_rows

def copy_kinematics_to_track(event_rows: List[dict], config: dict,
                             filter_manager: Optional[KalmanFilterManager] = None
                             ) -> Tuple[List[dict], List[dict]]:
    """Copy nested kinematics directly without filtering."""
    track_rows: List[dict] = []
    for row in event_rows:
        track_row: Dict[str, Any] = {
            'trackId': _record_value(row, 'trackId'),
            'standardIdentity': _record_value(row, ('identity', 'standard')),
            'environment': _record_value(row, ('identity', 'environment')),
            'trackOriginatedTimestamp': _record_value(row, ('crucibleHeader', 'createdDate')),
            'trackQuality': _record_value(row, 'trackQuality'),
            'interceptTimestamp': _record_value(
                row, ('estimatedKinematics', 'kinematicsTimestamp')),
            'edhControlSet': _record_value(row, 'edhControlSet'),
            'mode': _record_value(row, 'mode'),
            'entityStatus': _record_value(row, 'entityStatus'),
        }
        for key in ('identity', 'geodetic', 'ecefPosition', 'ecefVelocity',
                    'positionCovariance', 'velocityCovariance',
                    'positionVelocityCovariance', 'uncertainty'):
            value = row.get(key)
            if isinstance(value, dict):
                track_row[key] = clone_record(value)
        for key in ('speed', 'heading'):
            if key in row:
                track_row[key] = row[key]
        track_rows.append(track_row)

    track_head_rows = []
    for row in track_rows:
        head_row = clone_record(row)
        head_row['trackUpdatedTimestamp'] = row.get('interceptTimestamp')
        head_row.pop('trackQuality', None)
        head_row.pop('interceptTimestamp', None)
        head_row['stale'] = get_current_timestamp_string(10)
        head_row.setdefault('speed', 0.0)
        head_row.setdefault('heading', 0.0)
        track_head_rows.append(head_row)

    return track_rows, track_head_rows

def run(stream_manager_perspective: str) -> None:
    """
    Main function to run the tracker.
    
    Args:
        stream_manager_perspective: Perspective name for Stream Manager
    """
    
    # Register the signal handler
    global auth, rc, wc
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)

    # Initialize controllers before running
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)

    result = find_and_validate_configs(stream_manager_perspective, include_scripts=False, rc_instance=rc)
    config_list = result['datafeed_configs']
    perspective_config = result['perspective_config']

    # Merge perspective config (dataset names) into each datafeed config
    for config in config_list:
        for key, value in perspective_config.items():
            if key not in config:
                config[key] = value

    processes = []
    sse_coros = []
    # Cache of downloaded head datasets so multiple feeds sharing the same
    # component_track_head_dataset don't each re-download it.
    head_cache: Dict[str, List[dict]] = {}

    for config in config_list:
        if config.get('disabled'):
            logging.info(f"Skipping disabled config for {config.get('origin_dataset')}")
            continue

        event_queue = Queue()

        # Check if 'crucible_tracker' is set and valid
        crucible_tracker = config.get('crucible_tracker', '')
        if not crucible_tracker:
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - 'crucible_tracker' not set in config")
            continue
        crucible_tracker_lower = crucible_tracker.lower()
        if 'skip' in crucible_tracker_lower:
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - 'crucible_tracker' contains 'skip'")
            continue
        if '3rd party' in crucible_tracker_lower or 'third party' in crucible_tracker_lower:
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - 'crucible_tracker' contains '3rd party' or 'third party'")
            continue
        # Ensure component track datasets are set
        if not config.get('component_track_head_dataset') or not config.get('component_track_event_dataset'):
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - component track datasets not set")
            continue

        # Spawn N sharded tracker workers, each with its own queue
        n_shards = int(config.get('num_tracker_workers', NUM_TRACKER_WORKERS))
        shard_queues = [Queue(maxsize=_SHARD_QUEUE_MAXSIZE) for _ in range(n_shards)]

        for shard_idx in range(n_shards):
            shard_proc = Process(
                target=_run_tracker_shard,
                args=(shard_queues[shard_idx], config),
                daemon=True
            )
            shard_proc.start()
            processes.append(shard_proc)

        # Download this feed's track heads ONCE and fan them out to its shard
        # workers (pre-bucketed by trackId) so each shard initializes from its
        # queue instead of independently querying Crucible. Runs before the
        # dispatcher (started below in run_all_sse) so heads are consumed first.
        _fanout_head_preload_tracker(shard_queues, config, n_shards, head_cache)

        # Collect SSE listener coroutines + dispatcher coroutine
        sse_coros.append(sse_listener_launcher(event_queue, config))
        sse_coros.append(shard_dispatcher(event_queue, shard_queues, config))

    # Run all SSE listeners + dispatchers in a single event loop
    async def run_all_sse() -> None:
        await asyncio.gather(*sse_coros)

    try:
        asyncio.run(run_all_sse())
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except BaseException:
        logging.critical("Entity Tracker event loop exited unexpectedly")
        logging.critical(traceback.format_exc())
        raise
    finally:
        for proc in processes:
            if proc.is_alive():
                try:
                    proc.kill()
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                proc.join(timeout=3)

if __name__ == '__main__':

    # get command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("Stream_Manager_Perspective", help='Name of perspective in Stream Manager Configuration', default='Blue')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--use-fork', action='store_true', help='Use fork start method for multiprocessing') # needed for future multiprocessing support on Macs
    args = parser.parse_args()

    if args.use_fork:
        set_start_method('forkserver', force=True)
        logging.info("  Using the fork start method for multiprocessing  ")
    
    log_level = args.log_level
        # set up logging:
    if log_level is None:
        log_level = 'info'

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    
    # Configure root logger - this ensures all modules (utils, etc.) log consistently
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s: %(levelname)s %(name)s %(module)s Func: %(funcName)s:%(lineno)d-%(message)s'
    )

    stream_manager_perspective = args.Stream_Manager_Perspective

    run(stream_manager_perspective)



 



