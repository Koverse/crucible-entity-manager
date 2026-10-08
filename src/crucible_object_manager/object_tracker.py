#!/usr/bin/env python
"""
Object Tracker for V1 Object Schema

This tracker processes object events and writes to track datasets using the v1 Object schema.
It supports both passthrough mode (direct copy) and Kalman filter mode.

Usage:
    python object_tracker.py <Stream Manager perspective> --log=DEBUG
    e.g., python object_tracker.py Live_POV --log=DEBUG
"""
from __future__ import annotations

from multiprocessing import Process, Queue, set_start_method
from datetime import timezone as tz
from datetime import datetime as dt
from datetime import timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import traceback
import argparse
import asyncio
import logging
import signal
import uuid
import hashlib
import os
import json
import re
import time

try:
    from cruciblelib import authenticator, read_controller, write_controller, utils, log_utils
    from cruciblelib.decorators import async_retry
except ImportError:
    from . import authenticator, read_controller, write_controller, utils, log_utils
    from .decorators import async_retry

try:
    from object_utils import find_and_validate_configs, instantiate_api_controllers, write_batch_chunked, _fast_df_to_nested_json, _get_object_id, normalize_uuid, SSE_listener
except ImportError:
    from .object_utils import find_and_validate_configs, instantiate_api_controllers, write_batch_chunked, _fast_df_to_nested_json, _get_object_id, normalize_uuid, SSE_listener

# Global variables for controllers (initialized per-process via instantiate_api_controllers)
rc = wc = auth = None

# --- [PERF] timing switch (single toggle for ALL per-stage timing logs) ---
# Flip to False (or set env CRUCIBLE_PERF_TIMING=false) to silence every [PERF]
# line and make the timers no-ops.
PERF_TIMING = os.getenv('CRUCIBLE_PERF_TIMING', 'true').lower() not in ('false', '0', 'no')
_DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS = float(
    os.getenv('CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS', '15'))


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

    def __init__(self, mapping: Any, noise_covar: Any, ndim_state: Any = None) -> None:
        self.ndim_state = ndim_state
        self.mapping = tuple(mapping)
        self.noise_covar = np.asarray(noise_covar, dtype=float)


class Detection:
    """Minimal detection container accepted by the numpy update wrapper."""

    def __init__(self, state_vector: Any, timestamp: Any = None,
                 measurement_model: Any = None) -> None:
        self.state_vector = np.asarray(state_vector, dtype=float)
        self.timestamp = timestamp
        self.measurement_model = measurement_model


class _DeprecatedStoneSoupBackend:
    """Placeholder for legacy backend classes retained only in unreachable code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "object_tracker.py is StoneSoup-free; use "
            "object_tracker_stonesoup_deprecated.py for the legacy backend")


CombinedLinearGaussianTransitionModel = _DeprecatedStoneSoupBackend
ConstantVelocity = _DeprecatedStoneSoupBackend
LinearGaussian = LinearGaussianModel
KalmanPredictor = _DeprecatedStoneSoupBackend
KalmanUpdater = _DeprecatedStoneSoupBackend
SingleHypothesis = _DeprecatedStoneSoupBackend


GaussianState = NumpyGaussianState


def _perf_disabled() -> float:
    """Zero-cost stand-in for time.perf_counter when PERF_TIMING is off."""
    return 0.0

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def terminate(signum: int, frame: Any) -> None:
    """Signal handler for graceful shutdown — kills entire process group."""
    logging.info(f"Object Tracker: Received signal {signum}, killing process group")
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def initialize_controllers() -> None:
    """Initialize the read/write controllers and authenticator.
    
    Uses instantiate_api_controllers() from transformer (forkserver-safe).
    Call this once per process before using rc, wc, or auth.
    """
    global rc, wc, auth
    auth, rc, wc = instantiate_api_controllers()


async def _best_effort_update_track_heads(records: List[dict], track_head_dataset: str,
                                          heads_chunk_size: int, origin_dataset: str,
                                          max_concurrent: Optional[int]) -> tuple:
    if not records:
        return [], 0.0
    _t0 = time.perf_counter()
    _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
    try:
        failed_records, missing_object_ids = await write_batch_chunked(
            records,
            track_head_dataset,
            wc.update_entity_record_batch_by_name,
            heads_chunk_size,
            label=f' [{origin_dataset}] existing-head best-effort: ',
            token_refresher=_refresh_token,
            max_concurrent_writes=max_concurrent,
        )
        if failed_records or missing_object_ids:
            logging.warning(
                f" [{origin_dataset}]: deferred existing track-head update "
                f"{len(failed_records)} failed record(s), {len(missing_object_ids)} missing object(s)")
        failed_track_ids = {
            str(record.get('trackId'))
            for record in failed_records
            if record.get('trackId') is not None
        }
        missing_object_ids = {str(object_id) for object_id in missing_object_ids}
        retry_records = [
            record for record in records
            if (record.get('trackId') is not None
                and str(record.get('trackId')) in failed_track_ids)
            or (_get_object_id(record) is not None
                and str(_get_object_id(record)) in missing_object_ids)
        ]
        return retry_records, time.perf_counter() - _t0
    except Exception as head_err:
        logging.warning(
            f" [{origin_dataset}]: best-effort existing track-head update failed; "
            f"deferring {len(records)} record(s) for retry: {head_err}")
        return records, time.perf_counter() - _t0


def _collect_finished_head_update_tasks(filter_manager: Any, origin_dataset: str) -> float:
    pending = getattr(filter_manager, 'pending_head_update_tasks', [])
    if not pending:
        return 0.0
    still_pending = []
    completed = 0
    elapsed = 0.0
    for task in pending:
        if task.done():
            completed += 1
            try:
                retry_records, task_elapsed = task.result()
                elapsed += float(task_elapsed or 0.0)
                for record in retry_records or []:
                    track_id = record.get('trackId')
                    if isinstance(track_id, dict):
                        track_id = track_id.get('uuid')
                    if track_id is not None:
                        filter_manager.pending_head_updates_by_track[str(track_id)] = record
            except Exception as exc:
                logging.warning(
                    f" [{origin_dataset}]: existing track-head update task failed; "
                    f"records will be retried by the next batch: {exc}")
        else:
            still_pending.append(task)
    filter_manager.pending_head_update_tasks = still_pending
    if completed:
        logging.info(
            f" [{origin_dataset}]: completed {completed} background existing track-head update task(s); "
            f"still_running={len(still_pending)}")
    return elapsed


def _head_update_interval_seconds(config: Dict[str, Any]) -> float:
    value = config.get(
        'track_head_update_interval',
        config.get('head_update_interval_seconds',
                   _DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS))
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS

def _coalesce_existing_head_updates(head_df: pd.DataFrame, last_emit_by_track: Dict[str, float],
                                    interval_seconds: float, origin_dataset: str,
                                    label: str = 'component') -> pd.DataFrame:
    if head_df.empty or interval_seconds <= 0 or 'trackId' not in head_df.columns:
        return head_df
    now = time.monotonic()
    track_ids = head_df['trackId'].astype(str)
    keep = track_ids.map(lambda tid: (now - last_emit_by_track.get(tid, -1e30)) >= interval_seconds)
    kept = head_df[keep.to_numpy()].copy()
    for tid in kept['trackId'].astype(str):
        last_emit_by_track[tid] = now
    coalesced = len(head_df) - len(kept)
    if coalesced:
        logging.info(
            f" [{origin_dataset}]: coalesced existing {label} track-head update(s); "
            f"coalesced={coalesced} kept={len(kept)} interval={interval_seconds:g}s")
    return kept


def _parse_tracker_modes(crucible_tracker: Any) -> tuple[bool, bool, bool]:
    """Return passthrough, Kalman, and CI flags for a tracker mode value."""
    value = str(crucible_tracker or '').strip().lower()
    use_ci = 'covariance intersection' in value or 'ci' in value.split()
    use_kalman = 'kalman' in value or use_ci
    use_passthrough = 'passthrough' in value and not use_kalman
    return use_passthrough, use_kalman, use_ci


class ObjectKalmanFilterManager:
    """Manages Kalman filters and priors for each object (v1 schema)."""
    
    def __init__(self, config: dict) -> None:
        self.config = config
        self.origin_dataset = config.get('origin_dataset', '')
        self.predictor = {}  # objectId -> KalmanPredictor
        self.updater = {}    # objectId -> KalmanUpdater
        self.measurement_model = {}  # objectId -> LinearGaussian
        self.priors = {}   # objectId -> GaussianState
        self.tracks = {}   # objectId -> Track
        self.initialized = False
        self.objectid_to_trackid = {}  # objectId -> trackId
        self.new_trackIds = set()  # TrackIds that are newly created in this run
        self.pending_head_update_tasks = []  # best-effort existing-head PUTs
        self.pending_head_updates_by_track = {}
        self.existing_head_update_emit_time_by_track = {}
        # Derive passthrough mode from the canonical 'crucible_tracker' config
        # key. (The legacy 'passthrough_tracker' boolean key is no longer used.)
        self.passthrough_tracker, _, _ = _parse_tracker_modes(
            config.get('crucible_tracker', ''))
        self.fusion_method = 'kalman'  # 'kalman' or 'ci'
        self.ci_omega = None  # None = auto-optimize, float = fixed weight
        self.environment_by_object = {}  # objectId -> environment string
  
        # Environment-aware continuous-time white-acceleration noise intensity.
        # q has units m^2/s^3; over dt seconds it adds velocity variance q * dt,
        # so sigma_delta_v = sqrt(q * dt).
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
        # velocity covariance. Keep this to a few m/s around the reported
        # value so velocity uncertainty does not dominate position uncertainty.
        self.vel_var_by_environment = {
            'AIR': 25.0,             # sigma 5 m/s
            'GROUND': 9.0,           # sigma 3 m/s
            'SEA_SURFACE': 4.0,      # sigma 2 m/s
            'SEA_SUBSURFACE': 4.0,   # sigma 2 m/s
            'SPACE': 100.0,          # sigma 10 m/s
            'UNKNOWN': 25.0,         # sigma 5 m/s
        }
        self.vel_var_default = 25.0
    
    def _get_q(self, object_id: Optional[str] = None) -> float:
        """Return environment-appropriate process noise for an object."""
        env = self.environment_by_object.get(object_id)
        if env:
            return self.q_by_environment.get(env.upper(), self.q_default)
        return self.q_default

    def _get_vel_var(self, object_id: Optional[str] = None) -> float:
        """Return environment-appropriate velocity variance fallback."""
        env = self.environment_by_object.get(object_id)
        if env:
            return self.vel_var_by_environment.get(env.upper(), self.vel_var_default)
        return self.vel_var_default

    def _generate_track_id(self, object_id: str) -> str:
        """Generate a deterministic trackId from origin_dataset + objectId.
        This ensures the same feed always produces the same trackId for a given object,
        even across restarts."""
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.origin_dataset}:{object_id}").hex

    async def initialize_from_heads(self, crucible_kalman_config_string: str,
                                   preloaded_heads: Optional[List[dict]] = None) -> None:
        """Initialize filters and priors from the track head dataset.

        When *preloaded_heads* is provided (a list of already-flattened head
        records fanned out ONCE by the parent run()), it is used directly and
        no per-shard Crucible query is issued — avoiding the N-shard duplicate
        reads that 504'd on large head datasets. When None, falls back to the
        legacy per-shard query.
        """
        if self.initialized:
            return
            
        try:
            # Extract q value from crucible_kalman_config_string as override
            if crucible_kalman_config_string:
                match = re.search(r'q\s*=\s*(\d+(\.\d+)?)', crucible_kalman_config_string)
                if match:
                    q_override = float(match.group(1))
                    self.q_default = q_override
                    logging.info(f"Using q={q_override} from config as default")

            if preloaded_heads is not None:
                # Records already flattened by the parent's fan-out.
                heads_df = pd.DataFrame(preloaded_heads) if preloaded_heads else pd.DataFrame()
            else:
                # Get track head dataset name from config
                track_head_dataset = self.config.get('track_head_dataset', 
                                                      self.config.get('perspective', 'Live_POV') + '_ComponentTrackHeads')

                # Fallback path (used only when the parent didn't fan out heads,
                # e.g. a direct/non-sharded call). Use the same robust, unlimited
                # download-once loader instead of a capped search that 504s on
                # large head datasets.
                heads = await asyncio.to_thread(_load_all_records, track_head_dataset, self.config)

                heads_df = utils.flatten_crucible_dataset(heads)
            
            if len(heads_df) > 0:
                for idx, head in heads_df.iterrows():
                    object_id = head.get('objectId') or head.get('objectId.uuid')
                    track_id = head.get('trackId')
                    expected_track_id = self._generate_track_id(object_id)
                    
                    # Only load heads that belong to this feed
                    if track_id != expected_track_id:
                        continue
                    
                    self.objectid_to_trackid[object_id] = track_id

                    # Only initialize Kalman filter if passthrough_tracker is False
                    if not self.passthrough_tracker:
                        # Set environment from head data
                        env = head.get('identity.environment.environment') or head.get('environment')
                        if isinstance(env, str) and env:
                            self.environment_by_object[object_id] = env
                        
                        # Create initial state from head data - using interleaved [x, vx, y, vy, z, vz] format
                        if all(col in head for col in [
                            'ecefPosition.x', 'ecefPosition.y', 'ecefPosition.z',
                            'ecefVelocity.x', 'ecefVelocity.y', 'ecefVelocity.z'
                        ]):
                            initial_state_vector = np.array([
                                head['ecefPosition.x'], 
                                head['ecefVelocity.x'],
                                head['ecefPosition.y'], 
                                head['ecefVelocity.y'],
                                head['ecefPosition.z'],
                                head['ecefVelocity.z'],
                            ])
                        elif all(col in head for col in [
                            'ecefPosition.x', 'ecefPosition.y', 'ecefPosition.z',
                        ]):
                            initial_state_vector = np.array([
                                head['ecefPosition.x'], 
                                0.,
                                head['ecefPosition.y'], 
                                0.,
                                head['ecefPosition.z'],
                                0.
                            ])
                        else:
                            initial_state_vector = np.array([
                                0., 0.,
                                0., 0.,
                                0., 0.
                            ])
                        
                        # Replace NaN values with 0 and use high-uncertainty covariance
                        if np.isnan(initial_state_vector).any():
                            origin_ds = self.config.get('origin_dataset', 'unknown')
                            logging.warning(f" [{origin_ds}]: Object {object_id} initial state contains NaN - replacing with zeros and resetting covariance")
                            initial_state_vector = np.nan_to_num(initial_state_vector, nan=0.0)
                            
                        # Create initial covariance from head data
                        initial_covar = np.zeros((6, 6))
                        if 'positionVelocityCovariance.xdx' in head:
                            # Full position-velocity covariance terms
                            initial_covar[0, 0] = head['positionCovariance.xx']
                            initial_covar[0, 1] = head['positionVelocityCovariance.xdx']
                            initial_covar[0, 2] = head['positionCovariance.xy']
                            initial_covar[0, 3] = head['positionVelocityCovariance.xdy']
                            initial_covar[0, 4] = head['positionCovariance.xz']
                            initial_covar[0, 5] = head['positionVelocityCovariance.xdz']
                            
                            initial_covar[1, 0] = head['positionVelocityCovariance.xdx']
                            initial_covar[1, 1] = head['velocityCovariance.dxdx']
                            initial_covar[1, 2] = head['positionVelocityCovariance.ydx']
                            initial_covar[1, 3] = head['velocityCovariance.dxdy']
                            initial_covar[1, 4] = head['positionVelocityCovariance.zdx']
                            initial_covar[1, 5] = head['velocityCovariance.dxdz']
                            
                            initial_covar[2, 0] = head['positionCovariance.xy']
                            initial_covar[2, 1] = head['positionVelocityCovariance.ydx']
                            initial_covar[2, 2] = head['positionCovariance.yy']
                            initial_covar[2, 3] = head['positionVelocityCovariance.ydy']
                            initial_covar[2, 4] = head['positionCovariance.yz']
                            initial_covar[2, 5] = head['positionVelocityCovariance.ydz']
                            
                            initial_covar[3, 0] = head['positionVelocityCovariance.xdy']
                            initial_covar[3, 1] = head['velocityCovariance.dxdy']
                            initial_covar[3, 2] = head['positionVelocityCovariance.ydy']
                            initial_covar[3, 3] = head['velocityCovariance.dydy']
                            initial_covar[3, 4] = head['positionVelocityCovariance.zdy']
                            initial_covar[3, 5] = head['velocityCovariance.dydz']
                            
                            initial_covar[4, 0] = head['positionCovariance.xz']
                            initial_covar[4, 1] = head['positionVelocityCovariance.zdx']
                            initial_covar[4, 2] = head['positionCovariance.yz']
                            initial_covar[4, 3] = head['positionVelocityCovariance.zdy']
                            initial_covar[4, 4] = head['positionCovariance.zz']
                            initial_covar[4, 5] = head['positionVelocityCovariance.zdz']
                            
                            initial_covar[5, 0] = head['positionVelocityCovariance.xdz']
                            initial_covar[5, 1] = head['velocityCovariance.dxdz']
                            initial_covar[5, 2] = head['positionVelocityCovariance.ydz']
                            initial_covar[5, 3] = head['velocityCovariance.dydz']
                            initial_covar[5, 4] = head['positionVelocityCovariance.zdz']
                            initial_covar[5, 5] = head['velocityCovariance.dzdz']
                        else:
                            # Default covariance
                            initial_covar = np.diag([1.e7, 1.e7, 1.e7, 1.e7, 1.e7, 1.e7])
                        
                        # Replace NaN covariance values with high-uncertainty default
                        if np.isnan(initial_covar).any():
                            origin_ds = self.config.get('origin_dataset', 'unknown')
                            logging.warning(f" [{origin_ds}]: Object {object_id} initial covariance contains NaN - using default high-uncertainty covariance")
                            initial_covar = np.diag([1.e7, 1.e7, 1.e7, 1.e7, 1.e7, 1.e7])
                        
                        # Create and store prior
                        timestamp_str = head.get('trackUpdatedTimestamp', dt.utcnow().isoformat() + 'Z')
                        try:
                            timestamp = dt.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                        except:
                            timestamp = dt.utcnow()
                            
                        prior = GaussianState(
                            initial_state_vector,
                            initial_covar,
                            timestamp=timestamp
                        )
                        self.priors[object_id] = prior
                        self.tracks[object_id] = Track([prior])
                
                origin_dataset = self.config.get('origin_dataset', 'unknown')
                if not self.passthrough_tracker:
                    logging.info(f" [{origin_dataset}]: Initialized Kalman filters for {len(self.priors)} objects from track heads")
                else:
                    logging.info(f" [{origin_dataset}]: Populated trackId to objectId mapping for {len(self.objectid_to_trackid)} objects from track heads (passthrough mode)")
            else:
                origin_dataset = self.config.get('origin_dataset', 'unknown')
                logging.info(f" [{origin_dataset}]: No existing track heads found - filters will be initialized as new objects are encountered")
            
            self.initialized = True
            
        except Exception as e:
            origin_dataset = self.config.get('origin_dataset', 'unknown')
            logging.error(f" [{origin_dataset}]: Error initializing Kalman filters: {e}")
            logging.error(traceback.format_exc())
            self.initialized = True  # Mark as initialized to prevent retry loops
    
    def _reset_filter(self, object_id: str, measurement: Detection) -> None:
        """Reset a filter to the current measurement state when numerical issues occur."""
        origin_dataset = self.config.get('origin_dataset', 'unknown')
        for store in (self.priors, self.predictor, self.updater, self.measurement_model, self.tracks):
            store.pop(object_id, None)
        self.get_or_create_filter(object_id, measurement)
        logging.warning(f" [{origin_dataset}]: Reset Kalman filter for object {object_id} due to numerical issues")

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

    def get_or_create_filter(self, object_id: str, measurement: Detection) -> bool:
        """Get existing filter components or create new ones for an object.
        
        Always succeeds: NaN measurements are replaced with zeros and
        high-uncertainty covariance before filter creation.
        """
        origin_dataset = self.config.get('origin_dataset', 'unknown')
        if object_id not in self.priors:
            # Replace NaN in measurement with 0 and use high-uncertainty covariance
            if np.isnan(measurement.state_vector).any():
                logging.warning(f" [{origin_dataset}]: Measurement contains NaN for object {object_id} - replacing with zeros and using high-uncertainty covariance")
                measurement = Detection(
                    np.nan_to_num(measurement.state_vector, nan=0.0),
                    timestamp=measurement.timestamp,
                    measurement_model=measurement.measurement_model
                )
            
            # Create new filter components with environment-aware q
            q = self._get_q(object_id)
            transition_model = CombinedLinearGaussianTransitionModel([
                ConstantVelocity(q),
                ConstantVelocity(q),
                ConstantVelocity(q)
            ])
            
            # Clip noise covariance to reasonable bounds to prevent numerical issues
            noise_covar = measurement.measurement_model.noise_covar.copy()
            noise_covar = np.clip(noise_covar, -1e10, 1e10)
            # Ensure diagonal elements are positive and reasonable
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
            self.tracks[object_id] = Track()
            
            # Create new prior from measurement with reasonable initial covariance
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
            self.priors[object_id] = prior
            self.tracks[object_id].append(prior)
            self.predictor[object_id] = predictor
            self.updater[object_id] = updater
            self.measurement_model[object_id] = measurement_model
        return True

    def _predict(self, object_id: str, current_state: GaussianState, timestamp) -> GaussianState:
        """Kalman time-update. StoneSoup implementation; overridden by
        NumpyKalmanFilterManager with an equivalent numpy kernel."""
        return self.predictor[object_id].predict(current_state, timestamp=timestamp)

    def _standard_update(self, object_id: str, prediction: GaussianState,
                         measurement: Detection) -> GaussianState:
        """Kalman measurement-update. StoneSoup implementation; overridden by
        NumpyKalmanFilterManager with an equivalent numpy kernel."""
        hypothesis = SingleHypothesis(prediction, measurement)
        return self.updater[object_id].update(hypothesis)

    def update_filter(self, object_id: str, measurement: Detection) -> Optional[GaussianState]:
        """Update filter state for an object with a new measurement."""
        origin_dataset = self.config.get('origin_dataset', 'unknown')
        if not self.get_or_create_filter(object_id, measurement):
            # Filter creation failed (e.g., due to NaN values)
            return None
        
        current_state = self.priors[object_id]
        
        # Check if measurement is too old compared to current state
        stale_diff = (current_state.timestamp - measurement.timestamp).total_seconds()
        if stale_diff > 0:
            # Out-of-order/stale duplicate: a newer measurement already advanced
            # this object's filter, so dropping this one loses nothing. Logged at
            # DEBUG (high-volume + benign); process_with_kalman emits a single
            # per-batch aggregate count instead of one WARNING per measurement.
            logging.debug(f" [{origin_dataset}]: Skipping measurement for object {object_id} - too old: {stale_diff:.1f} seconds behind current state")
            self._stale_skip_count = getattr(self, '_stale_skip_count', 0) + 1
            if stale_diff > getattr(self, '_stale_skip_max', 0.0):
                self._stale_skip_max = stale_diff
            return current_state
        
        # Check for large time jumps that can cause numerical instability
        time_diff = abs((measurement.timestamp - current_state.timestamp).total_seconds())
        max_time_jump = 15*60 # 15 minutes - reset filter if time jump is too large
        
        if time_diff > max_time_jump:
            logging.warning(f" [{origin_dataset}]: Large time jump ({time_diff:.1f}s) for object {object_id} - resetting filter")
            self._reset_filter(object_id, measurement)
            return self.priors[object_id]
        
        # Check if current state has NaN (corrupted filter) - reset if so
        if np.isnan(current_state.state_vector).any() or np.isnan(current_state.covar).any():
            logging.warning(f" [{origin_dataset}]: Current state contains NaN for object {object_id} - resetting filter")
            self._reset_filter(object_id, measurement)
            return self.priors[object_id]
        
        try:
            # Predict to measurement time
            prediction = self._predict(object_id, current_state, measurement.timestamp)
            
            # Check prediction for NaN or extremely large values
            if np.isnan(prediction.state_vector).any() or np.isnan(prediction.covar).any():
                logging.warning(f" [{origin_dataset}]: Prediction produced NaN for object {object_id} - resetting filter")
                self._reset_filter(object_id, measurement)
                return self.priors[object_id]
            
            # Check for exploding covariance (numerical instability)
            max_covar = np.max(np.abs(prediction.covar))
            if max_covar > 1e15:
                logging.warning(f" [{origin_dataset}]: Covariance explosion ({max_covar:.2e}) for object {object_id} - resetting filter")
                self._reset_filter(object_id, measurement)
                return self.priors[object_id]
            
            # Update with measurement
            if self.fusion_method == 'ci':
                pred_sv = np.asarray(prediction.state_vector).flatten()
                pred_cov = np.asarray(prediction.covar)

                if len(measurement.state_vector) == 6:
                    # Full 6D measurement — use directly
                    ci_meas_mean = np.asarray(measurement.state_vector).flatten()
                    ci_meas_covar = np.asarray(measurement.measurement_model.noise_covar)
                else:
                    # 3D position-only: lift to 6D by borrowing predicted
                    # velocity with env-aware variance so CI mostly ignores it
                    vel_var = self._get_vel_var(object_id)
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
                    # CI failed, fall back to standard Kalman
                    post = self._standard_update(object_id, prediction, measurement)
            else:
                post = self._standard_update(object_id, prediction, measurement)
            
            # Check for NaN in result
            if np.isnan(post.state_vector).any() or np.isnan(post.covar).any():
                logging.warning(f" [{origin_dataset}]: Kalman update produced NaN for object {object_id} - resetting filter")
                self._reset_filter(object_id, measurement)
                return self.priors[object_id]
            
            # Update prior
            self.priors[object_id] = post
            
            logging.debug(f" [{origin_dataset}]: Kalman filter update successful for object {object_id} at {measurement.timestamp}")
            
            return post
            
        except Exception as e:
            logging.error(f" [{origin_dataset}]: Exception in Kalman update for object {object_id}: {e}")
            self._reset_filter(object_id, measurement)
            return self.priors[object_id]


class NumpyKalmanFilterManager(ObjectKalmanFilterManager):
    """Drop-in replacement for :class:`ObjectKalmanFilterManager` that runs the
    Kalman predict/update with plain numpy instead of StoneSoup.

    StoneSoup's per-event predict/update (object allocation + property plumbing)
    dominates tracker cost (~18 ms/event). The underlying math is a small,
    fixed-size 6-state constant-velocity Kalman filter, so a direct numpy
    implementation is orders of magnitude cheaper. Only the numeric kernel is
    overridden: ALL control flow, guards (stale/out-of-order, 15-min time-jump
    reset, NaN and covariance-explosion resets), environment-aware process
    noise, and the covariance-intersection fusion path are inherited unchanged,
    so behavior matches StoneSoup by construction (verified by
    tests/unit/test_tracker_kalman_equivalence.py).

    State is stored as StoneSoup ``GaussianState`` in ``self.priors`` exactly as
    the base class does, so the rest of the pipeline and the head-preload path
    are unaffected.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Process-noise PSD fixed per object at filter-creation time, mirroring
        # StoneSoup building the transition model once in get_or_create_filter.
        self._q_by_object: Dict[str, float] = {}

    @staticmethod
    def _cv_transition(dt: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
        """Constant-velocity transition F and process noise Q for interleaved
        state [x, vx, y, vy, z, vz] over interval ``dt`` — the block-diagonal
        form of StoneSoup's CombinedLinearGaussianTransitionModel([CV(q)] * 3).
        """
        F = np.eye(6)
        F[0, 1] = F[2, 3] = F[4, 5] = dt
        d = abs(dt)  # StoneSoup ConstantVelocity.covar uses abs(dt)
        q_block = q * np.array([[d ** 3 / 3.0, d ** 2 / 2.0],
                                [d ** 2 / 2.0, d]])
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = q_block
        Q[2:4, 2:4] = q_block
        Q[4:6, 4:6] = q_block
        return F, Q

    @staticmethod
    def _measurement_matrix(mapping) -> np.ndarray:
        """Selection matrix H mapping the 6-state vector onto the measured dims."""
        H = np.zeros((len(mapping), 6))
        H[np.arange(len(mapping)), list(mapping)] = 1.0
        return H

    def _fresh_prior(self, object_id: str, meas_vec: np.ndarray, timestamp) -> GaussianState:
        """Create (or reset to) a fresh prior from a measurement: diag(1e7)
        covariance, velocity 0 for position-only measurements, NaN->0. Mirrors
        the base StoneSoup get_or_create_filter / _reset_filter initial state.
        Also used for the in-line resets in _update_core."""
        m = np.nan_to_num(np.asarray(meas_vec, dtype=float).reshape(-1), nan=0.0)
        if m.shape[0] == 6:
            mean = m
        else:
            mean = np.array([m[0], 0., m[1], 0., m[2], 0.])
        prior = GaussianState(mean.reshape(-1, 1), np.diag([1.e7] * 6), timestamp=timestamp)
        self.priors[object_id] = prior
        self.tracks[object_id] = Track()
        self.tracks[object_id].append(prior)
        self._q_by_object[object_id] = self._get_q(object_id)
        return prior

    def get_or_create_filter(self, object_id: str, measurement: Detection) -> bool:
        """Create numpy filter state for a new object (delegates to _fresh_prior).
        Kept for the inherited _reset_filter path, which calls this with a
        Detection."""
        if object_id not in self.priors:
            noise_covar = np.asarray(measurement.measurement_model.noise_covar, dtype=float)
            noise_covar = np.clip(noise_covar, -1e10, 1e10)
            for i in range(noise_covar.shape[0]):
                if noise_covar[i, i] <= 0 or noise_covar[i, i] > 1e10:
                    noise_covar[i, i] = 1e6
            self.measurement_model[object_id] = LinearGaussianModel(
                measurement.measurement_model.mapping, noise_covar)
            self._fresh_prior(object_id,
                              np.asarray(measurement.state_vector, dtype=float).reshape(-1),
                              measurement.timestamp)
        return True

    def _ci_update(self, object_id, pred_mean, pred_cov, m, mapping, noise_covar):
        """Covariance-Intersection fusion of the prediction with the measurement,
        matching the base ObjectKalmanFilterManager.update_filter CI branch (3D
        measurements are lifted to 6D by borrowing the predicted velocity with an
        env-aware variance). Falls back to a standard Kalman update if CI fails."""
        if m.shape[0] == 6:
            ci_meas_mean = m
            ci_meas_covar = np.asarray(noise_covar, dtype=float)
        else:
            vel_var = self._get_vel_var(object_id)
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

    def _update_core(self, object_id, meas_vec, mapping, noise_covar, timestamp):
        """Array-native Kalman update: the numpy equivalent of the base
        update_filter body (same guard order and thresholds), operating on plain
        arrays so it can be driven either from a StoneSoup Detection
        (update_filter) or from bulk-extracted arrays (process_measurements_batch)
        without constructing a Detection/LinearGaussian per event."""
        origin_dataset = self.config.get('origin_dataset', 'unknown')
        m = np.asarray(meas_vec, dtype=float).reshape(-1)
        if object_id not in self.priors:
            self._fresh_prior(object_id, m, timestamp)

        current = self.priors[object_id]
        cur_mean = np.asarray(current.state_vector, dtype=float).reshape(-1)
        cur_cov = np.asarray(current.covar, dtype=float)
        cur_ts = current.timestamp

        # Out-of-order / stale duplicate: a newer measurement already advanced
        # this object's filter (aggregated into one per-batch line by the caller).
        stale_diff = (cur_ts - timestamp).total_seconds()
        if stale_diff > 0:
            logging.debug(f" [{origin_dataset}]: Skipping measurement for object {object_id} - too old: {stale_diff:.1f} seconds behind current state")
            self._stale_skip_count = getattr(self, '_stale_skip_count', 0) + 1
            if stale_diff > getattr(self, '_stale_skip_max', 0.0):
                self._stale_skip_max = stale_diff
            return current

        time_diff = abs((timestamp - cur_ts).total_seconds())
        if time_diff > 15 * 60:
            logging.warning(f" [{origin_dataset}]: Large time jump ({time_diff:.1f}s) for object {object_id} - resetting filter")
            return self._fresh_prior(object_id, m, timestamp)

        if np.isnan(cur_mean).any() or np.isnan(cur_cov).any():
            logging.warning(f" [{origin_dataset}]: Current state contains NaN for object {object_id} - resetting filter")
            return self._fresh_prior(object_id, m, timestamp)

        try:
            q = self._q_by_object.get(object_id)
            if q is None:  # preloaded object (head fan-out) — derive lazily
                q = self._q_by_object[object_id] = self._get_q(object_id)
            F, Q = self._cv_transition(time_diff, q)
            pred_mean = F @ cur_mean
            pred_cov = F @ cur_cov @ F.T + Q

            if np.isnan(pred_mean).any() or np.isnan(pred_cov).any():
                logging.warning(f" [{origin_dataset}]: Prediction produced NaN for object {object_id} - resetting filter")
                return self._fresh_prior(object_id, m, timestamp)
            _max_covar = np.max(np.abs(pred_cov))
            if _max_covar > 1e15:
                logging.warning(f" [{origin_dataset}]: Covariance explosion ({_max_covar:.2e}) for object {object_id} - resetting filter")
                return self._fresh_prior(object_id, m, timestamp)

            if self.fusion_method == 'ci':
                post_mean, post_cov = self._ci_update(object_id, pred_mean, pred_cov, m, mapping, noise_covar)
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
                logging.warning(f" [{origin_dataset}]: Kalman update produced NaN for object {object_id} - resetting filter")
                return self._fresh_prior(object_id, m, timestamp)

            post = GaussianState(post_mean.reshape(-1, 1), post_cov, timestamp=timestamp)
            self.priors[object_id] = post
            return post
        except Exception as e:
            logging.error(f" [{origin_dataset}]: Exception in Kalman update for object {object_id}: {e}")
            return self._fresh_prior(object_id, m, timestamp)

    def update_filter(self, object_id: str, measurement: Detection) -> Optional[GaussianState]:
        """Update filter state for an object with a new measurement. Extracts the
        arrays from the StoneSoup Detection and delegates to the array-native
        _update_core (the same kernel used by the batched path)."""
        return self._update_core(
            object_id,
            np.asarray(measurement.state_vector, dtype=float).reshape(-1),
            measurement.measurement_model.mapping,
            np.asarray(measurement.measurement_model.noise_covar, dtype=float),
            measurement.timestamp,
        )

    def process_measurements_batch(self, event_df: pd.DataFrame, object_id_col: str,
                                   config: Dict[str, Any]) -> None:
        """Vectorized equivalent of the per-row measurement-build + update loop in
        process_with_kalman for the numpy backend.

        Extracts every needed column as a numpy array once (no DataFrame
        .iterrows / per-cell .get), parses timestamps in bulk, then applies each
        measurement via _update_core (no per-event StoneSoup Detection /
        LinearGaussian). The caller pre-sorts rows ascending by timestamp, so
        applying them in row order preserves the exact sequential filter
        semantics; self.priors ends up identical to the update_filter loop.
        """
        n = len(event_df)
        if n == 0:
            return
        cols = event_df.columns
        origin_dataset = config.get('origin_dataset', 'unknown')

        def _col(name, default):
            return event_df[name].to_numpy() if name in cols else np.full(n, default)

        def _col_fallback(primary, secondary, default):
            if primary in cols:
                return event_df[primary].to_numpy()
            if secondary in cols:
                return event_df[secondary].to_numpy()
            return np.full(n, default)

        oid = event_df[object_id_col].to_numpy()

        px = _col('ecefPosition.x', 0.0).astype(float)
        py = _col('ecefPosition.y', 0.0).astype(float)
        pz = _col('ecefPosition.z', 0.0).astype(float)
        vx = _col_fallback('ecefVelocity.x', 'ecefVelocity.dx', np.nan).astype(float)
        vy = _col_fallback('ecefVelocity.y', 'ecefVelocity.dy', np.nan).astype(float)
        vz = _col_fallback('ecefVelocity.z', 'ecefVelocity.dz', np.nan).astype(float)

        has_pos_cov_col = 'positionCovariance.xx' in cols
        pcxx = _col('positionCovariance.xx', np.nan).astype(float)
        pcyy = _col('positionCovariance.yy', 1.e7).astype(float)
        pczz = _col('positionCovariance.zz', 1.e7).astype(float)
        pcxy = _col('positionCovariance.xy', 0.0).astype(float)
        pcxz = _col('positionCovariance.xz', 0.0).astype(float)
        pcyz = _col('positionCovariance.yz', 0.0).astype(float)

        has_vel_cov_col = 'velocityCovariance.dxdx' in cols
        vcxx = _col('velocityCovariance.dxdx', np.nan).astype(float)
        vcyy = _col('velocityCovariance.dydy', 1.e7).astype(float)
        vczz = _col('velocityCovariance.dzdz', 1.e7).astype(float)
        vcxy = _col('velocityCovariance.dxdy', 0.0).astype(float)
        vcxz = _col('velocityCovariance.dxdz', 0.0).astype(float)
        vcyz = _col('velocityCovariance.dydz', 0.0).astype(float)

        # Environment: first non-empty value per object (only if not already known),
        # so the env-aware q / vel_var lookups work for new objects.
        env_arr = None
        if 'identity.environment.environment' in cols:
            env_arr = event_df['identity.environment.environment'].to_numpy()
        elif 'environment' in cols:
            env_arr = event_df['environment'].to_numpy()
        if env_arr is not None:
            ebo = self.environment_by_object
            for i in range(n):
                e = env_arr[i]
                o = oid[i]
                if isinstance(e, str) and e and o not in ebo:
                    ebo[o] = e

        # Timestamps: kinematicsTimestamp or createdDate or now, parsed in bulk.
        kts = (event_df['estimatedKinematics.kinematicsTimestamp'].to_numpy(dtype=object)
               if 'estimatedKinematics.kinematicsTimestamp' in cols else np.full(n, None, dtype=object))
        cds = (event_df['crucibleHeader.createdDate'].to_numpy(dtype=object)
               if 'crucibleHeader.createdDate' in cols else np.full(n, None, dtype=object))
        now_str = get_current_timestamp_string()

        def _falsy(v):
            return v is None or v == '' or (isinstance(v, float) and v != v)

        ts_raw = np.empty(n, dtype=object)
        for i in range(n):
            v = kts[i]
            if _falsy(v):
                v = cds[i]
            if _falsy(v):
                v = now_str
            ts_raw[i] = v
        parsed = pd.to_datetime(ts_raw, format="%Y-%m-%dT%H:%M:%S.%fZ", errors='coerce')
        now_dt = dt.utcnow()
        ts_list = [(p.to_pydatetime() if not pd.isna(p) else now_dt) for p in parsed]

        map6 = (0, 1, 2, 3, 4, 5)
        map3 = (0, 2, 4)
        n_skipped_nan = 0
        for i in range(n):
            x = px[i]; y = py[i]; z = pz[i]
            if x != x or y != y or z != z:  # NaN position -> measured vector has NaN, skip
                n_skipped_nan += 1
                continue
            _vx = vx[i]; _vy = vy[i]; _vz = vz[i]
            has_vel = not (_vx != _vx or _vy != _vy or _vz != _vz)

            if has_pos_cov_col and pcxx[i] == pcxx[i]:  # column present and this row notna
                pxx = pcxx[i]; pyy = pcyy[i]; pzz = pczz[i]
                pxy = pcxy[i]; pxz = pcxz[i]; pyz = pcyz[i]
            else:
                pxx = pyy = pzz = 1.e7
                pxy = pxz = pyz = 0.0

            oid_i = oid[i]
            ts_i = ts_list[i]
            if has_vel:
                if has_vel_cov_col and vcxx[i] == vcxx[i]:
                    vxx = vcxx[i]; vyy = vcyy[i]; vzz = vczz[i]
                    vxy = vcxy[i]; vxz = vcxz[i]; vyz = vcyz[i]
                else:
                    vv = self._get_vel_var(oid_i)
                    vxx = vyy = vzz = vv
                    vxy = vxz = vyz = 0.0
                nc = np.zeros((6, 6))
                nc[0, 0] = pxx; nc[2, 2] = pyy; nc[4, 4] = pzz
                nc[0, 2] = nc[2, 0] = pxy; nc[0, 4] = nc[4, 0] = pxz; nc[2, 4] = nc[4, 2] = pyz
                nc[1, 1] = vxx; nc[3, 3] = vyy; nc[5, 5] = vzz
                nc[1, 3] = nc[3, 1] = vxy; nc[1, 5] = nc[5, 1] = vxz; nc[3, 5] = nc[5, 3] = vyz
                meas = np.array([x, _vx, y, _vy, z, _vz])
                self._update_core(oid_i, meas, map6, nc, ts_i)
            else:
                pc = np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
                meas = np.array([x, y, z])
                self._update_core(oid_i, meas, map3, pc, ts_i)

        if n_skipped_nan:
            logging.warning(f" [{origin_dataset}]: Skipped {n_skipped_nan} measurement(s) "
                            f"with NaN position this batch")


def get_current_timestamp_string() -> str:
    """Get current UTC timestamp as ISO string."""
    datetime_format_string = "%Y-%m-%dT%H:%M:%S.%f"
    current_time = dt.now(tz=tz.utc).strftime(datetime_format_string)[:-3] + 'Z'
    return current_time


def get_current_timestamp_string_with_offset(offset_hours: int = 0) -> str:
    """Get UTC timestamp with offset as ISO string."""
    datetime_format_string = "%Y-%m-%dT%H:%M:%S.%f"
    time_with_offset = dt.now(tz=tz.utc) + timedelta(hours=offset_hours)
    time_string = time_with_offset.strftime(datetime_format_string)[:-3] + 'Z'
    return time_string


# Number of worker processes per data feed for parallel objectId processing
NUM_TRACKER_WORKERS = 4  # Default parallel tracker workers (override per data feed with num_tracker_workers)


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


def _tracker_shard_for_object(object_id: Any, n_shards: int) -> int:
    """Deterministically map an objectId to a tracker shard.

    Python's built-in ``hash()`` is salted per process, so parent-side head
    preload bucketing and event dispatch are not restart-stable. A stable hash
    keeps object ownership aligned across preload, dispatch, and restarts.
    """
    if not object_id or n_shards <= 1:
        return 0
    digest = hashlib.md5(
        str(object_id).encode('utf-8'), usedforsecurity=False).hexdigest()
    return int(digest, 16) % n_shards


def _load_all_records(dataset_name: str, config: dict) -> List[dict]:
    """Pull ALL records of a dataset once via the search API for the head preload.

    Uses a single high-LIMIT ``SELECT *`` search and NEVER
    ``download_dataset_by_name`` (whose raw byte stream can hand back a whole
    JSON array on one line, which then breaks ``json_normalize``). Head
    datasets are scalar-keyed, so search returns them as a list of records.
    Returns [] on failure so shards start with no preloaded heads rather than
    crashing.
    """
    try:
        rc.token = auth.get_token()
        limit = int(config.get('tracker_head_preload_limit',
                               os.getenv('CRUCIBLE_HEAD_PRELOAD_LIMIT', '100000')))
        # ORDER BY crucibleHeader.updatedDate DESC so that when there are more
        # heads than `limit`, we keep the most recently updated ones (freshest
        # state) and only the oldest are dropped.
        raw = rc.search(
            f"SELECT * FROM {dataset_name} "
            f"ORDER BY {dataset_name}.crucibleHeader.updatedDate DESC LIMIT {limit}"
        ) or []
        if len(raw) >= limit:
            logging.warning(
                f"Head preload for {dataset_name} hit the {limit}-row LIMIT; the oldest "
                f"heads were dropped. Raise tracker_head_preload_limit / CRUCIBLE_HEAD_PRELOAD_LIMIT.")
        return raw
    except Exception as e:
        logging.warning(
            f"Head preload: search for {dataset_name} failed "
            f"({e}); shards will start with no preloaded heads")
        return []


def _skip_head_preload(config: dict) -> bool:
    """Return True when the head preload should be skipped entirely
    (config ``skip_head_preload`` or env ``CRUCIBLE_SKIP_HEAD_PRELOAD``).

    Safe because trackIds are deterministic: starting a shard with empty Kalman
    state re-derives the same head ids (UPSERT, no orphan explosion), and stale
    Kalman states are low value since they age out quickly. Trades a brief
    re-initialization on restart for eliminating the head-preload memory spike.
    """
    val = config.get('skip_head_preload', os.getenv('CRUCIBLE_SKIP_HEAD_PRELOAD', 'false'))
    return str(val).strip().lower() in ('1', 'true', 'yes', 'on')


def _fanout_head_preload_tracker(shard_queues: List[Queue], config: dict, n_shards: int,
                                 head_cache: Optional[Dict[str, List[dict]]] = None) -> None:
    """Download component track heads ONCE and fan them out to the tracker shard
    workers' queues (pre-bucketed by objectId to match the dispatcher routing).

    Replaces the per-shard ``SELECT * ... LIMIT 10000`` query in
    ``initialize_from_heads`` (which N-duplicated the read load and 504'd on
    large head datasets). The parent buckets by the same stable objectId shard
    helper used by ``shard_dispatcher``; each shard still filters to its own
    feed via its trackId. Ends with a ``done`` marker so a shard that owns no
    heads still initializes.
    ``head_cache`` avoids re-downloading a dataset shared by multiple feeds.
    """
    if _skip_head_preload(config):
        for s in range(n_shards):
            shard_queues[s].put({'type': 'init_heads', 'heads': [], 'done': True})
        logging.info(
            f"Head preload SKIPPED (skip_head_preload): {n_shards} tracker shard(s) start "
            f"with empty Kalman state; deterministic trackIds keep heads consistent.")
        return

    head_ds = config.get('track_head_dataset') or (
        config.get('perspective', 'Live_POV') + '_ComponentTrackHeads')

    if head_cache is not None and head_ds in head_cache:
        recs = head_cache[head_ds]
    else:
        recs = _load_all_records(head_ds, config)
        if head_cache is not None:
            head_cache[head_ds] = recs

    buckets: List[List[dict]] = [[] for _ in range(n_shards)]
    if recs:
        hdf = utils.flatten_crucible_dataset(recs)
        for row in hdf.to_dict('records'):
            oid = row.get('objectId')
            if oid is None or (isinstance(oid, float) and pd.isna(oid)):
                oid = row.get('objectId.uuid')
            if oid is None or (isinstance(oid, float) and pd.isna(oid)):
                continue
            shard = _tracker_shard_for_object(oid, n_shards)
            buckets[shard].append(row)
        logging.info(
            f"Head preload: {sum(len(b) for b in buckets)} track heads from "
            f"{head_ds} bucketed across {n_shards} shards")
        del hdf  # free the flattened frame; recs may be shared via head_cache

    for s in range(n_shards):
        b = buckets[s]
        for i in range(0, len(b), _HEAD_FANOUT_CHUNK):
            shard_queues[s].put({'type': 'init_heads',
                                 'heads': b[i:i + _HEAD_FANOUT_CHUNK], 'done': False})
        shard_queues[s].put({'type': 'init_heads', 'heads': [], 'done': True})
        buckets[s] = []  # release after queueing so the parent doesn't hold all shards' heads


def _run_tracker_shard(shard_queue: Queue, config: dict) -> None:
    """Launch a sharded tracker worker that processes a subset of objectIds."""
    asyncio.run(tracker(shard_queue, config))


async def shard_dispatcher(event_queue: Queue, shard_queues: List[Queue], config: dict) -> None:
    """
    Reads events from the main event_queue, partitions by objectId,
    and routes to the appropriate shard queue (hash-based assignment).
    Runs as an asyncio task in the parent process alongside SSE listeners.
    """
    n_shards = len(shard_queues)
    origin_dataset = config.get('origin_dataset', 'unknown')

    while True:
        # Blocking get on multiprocessing Queue — wrap in to_thread
        events = await asyncio.to_thread(event_queue.get)

        if not events:
            continue

        # Aggregate backlog
        while not event_queue.empty():
            try:
                extra_events = event_queue.get_nowait()
                if isinstance(extra_events, list):
                    events.extend(extra_events)
                else:
                    events.append(extra_events)
            except Exception:
                break

        # Partition events by objectId into shard buckets
        shard_buckets = [[] for _ in range(n_shards)]
        for event in events:
            obj_id = None
            if isinstance(event, dict):
                obj_id = event.get('objectId', {})
                if isinstance(obj_id, dict):
                    obj_id = obj_id.get('uuid')
            if obj_id:
                shard_idx = _tracker_shard_for_object(obj_id, n_shards)
            else:
                shard_idx = 0  # fallback: send to first shard
            shard_buckets[shard_idx].append(event)

        # Push each bucket to its shard queue
        for i, bucket in enumerate(shard_buckets):
            if bucket:
                shard_queues[i].put(bucket)


@async_retry
async def tracker(event_queue: Queue, config: dict) -> None:
    """
    Main tracker coroutine that processes object events.
    
    Args:
        event_queue: Queue containing incoming object events
        config: Stream Manager configuration dictionary
    """
    # Initialize controllers in child process (forkserver children start fresh)
    global rc, wc, auth
    # Configure logging in child (forkserver children don't inherit parent's config)
    if '_log_level' in config:
        log_utils.get_logger(log_type='transformer', log_level=config['_log_level'])
    auth, rc, wc = instantiate_api_controllers()

    origin_dataset = config.get('origin_dataset', 'unknown')
    if config.get('disabled'):
        logging.info(f" [{origin_dataset}]: No tracking for dataset {origin_dataset}")
        logging.info(f" [{origin_dataset}]:   as it has Stream Manager configuration option disabled set to True")
        return


    crucible_tracker = config.get('crucible_tracker', '').lower()
    # Skip if '3rd party' or 'third party' in crucible_tracker
    if '3rd party' in crucible_tracker or 'third party' in crucible_tracker:
        logging.info(f" [{origin_dataset}]: Skipping tracker for dataset {origin_dataset} - 'crucible_tracker' contains '3rd party' or 'third party'")
        return

    use_passthrough, use_kalman, use_ci = _parse_tracker_modes(crucible_tracker)

    # Only proceed if either passthrough or kalman is selected
    if not (use_passthrough or use_kalman):
        logging.info(f" [{origin_dataset}]: No valid tracking mode selected. Skipping.")
        return

    # Initialize Kalman filter manager if needed. The live tracker is now
    # StoneSoup-free: Kalman mode always uses the numpy backend. The previous
    # StoneSoup implementation is preserved in object_tracker_stonesoup_deprecated.py.
    use_numpy_kalman = config.get(
        'use_numpy_kalman',
        os.getenv('CRUCIBLE_USE_NUMPY_KALMAN', 'false').strip().lower() in ('1', 'true', 'yes'))
    if not use_numpy_kalman:
        logging.warning(f" [{origin_dataset}]: use_numpy_kalman is false, but the live tracker is StoneSoup-free; using numpy backend")
    manager_cls = NumpyKalmanFilterManager
    filter_manager = manager_cls(config)
    if use_ci:
        filter_manager.fusion_method = 'ci'
    logging.info(f" [{origin_dataset}]: Kalman backend = numpy")

    # Receive the one-time track-head preload fanned out by the parent (run())
    # instead of querying Crucible per-shard (which N-duplicated the load and
    # 504'd on large head datasets). Event buckets are plain lists; init_heads
    # messages are dicts, so they are distinguishable. Stash any event bucket
    # that arrives before init completes (handled first in the loop below).
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

    await filter_manager.initialize_from_heads(crucible_tracker, preloaded_heads=preloaded_heads)
    filter_manager.initialized = True


    if use_passthrough:
        logging.info(f" [{origin_dataset}]: Starting passthrough tracker")
    if use_kalman:
        ci_label = " with Covariance Intersection" if use_ci else ""
        logging.info(f" [{origin_dataset}]: Starting Kalman tracker{ci_label}")
        logging.info(f" [{origin_dataset}]:   fusion={filter_manager.fusion_method}, q_default={filter_manager.q_default}")

    # Get dataset names from config (always define, not inside if)
    track_event_dataset = config.get('component_track_event_dataset',)
    track_head_dataset = config.get('component_track_head_dataset',)

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
                # Get the backlog of data from SSE listener
                events = event_queue.get()

            # Aggregate events if queue has backlog
            while not event_queue.empty():
                extra_events = event_queue.get()
                if isinstance(extra_events, list):
                    events.extend(extra_events)
                else:
                    events.append(extra_events)

            _perf = time.perf_counter if PERF_TIMING else _perf_disabled
            _t0 = _perf(); _t = _t0; _timings = {}
            event_df = utils.flatten_crucible_dataset(events)

            # Filter out state updates - only process kinematic updates
            if 'eventType' in event_df.columns:
                kinematic_df = event_df[event_df['eventType'] == 'KINEMATIC_UPDATE']
                if len(kinematic_df) == 0:
                    logging.debug(f" [{origin_dataset}]: No kinematic updates in this batch - skipping")
                    continue
                event_df = kinematic_df
                logging.debug(f" [{origin_dataset}]: Filtered to {len(event_df)} kinematic updates (excluded state updates)")

            # Drop records missing estimatedKinematics.kinematicsTimestamp
            if 'estimatedKinematics.kinematicsTimestamp' not in event_df.columns:
                logging.debug(f" [{origin_dataset}]: No estimatedKinematics.kinematicsTimestamp column in this batch - skipping")
                continue
            pre_filter_count = len(event_df)
            event_df = event_df[event_df['estimatedKinematics.kinematicsTimestamp'].notna()]
            dropped = pre_filter_count - len(event_df)
            if dropped:
                logging.debug(f" [{origin_dataset}]: Dropped {dropped} records missing estimatedKinematics.kinematicsTimestamp")
            if len(event_df) == 0:
                logging.debug(f" [{origin_dataset}]: No records with estimatedKinematics.kinematicsTimestamp in this batch - skipping")
                continue

            # Handle different column names for object ID
            if 'objectId.uuid' in event_df.columns:
                object_id_col = 'objectId.uuid'
            elif 'objectId' in event_df.columns:
                object_id_col = 'objectId'
            else:
                logging.error(f" [{origin_dataset}]: No objectId column found in event data")
                continue

            # Sort by timestamp if available
            timestamp_col = None
            for col in ['estimatedKinematics.kinematicsTimestamp', 'crucibleHeader.createdDate', 'timestamp']:
                if col in event_df.columns:
                    timestamp_col = col
                    break
            if timestamp_col:
                event_df = event_df.sort_values(by=timestamp_col, ignore_index=True, ascending=True)

            # Generate trackIds for new objects
            object_ids = event_df[object_id_col].unique()
            for object_id in object_ids:
                if object_id and object_id not in filter_manager.objectid_to_trackid:
                    track_id = filter_manager._generate_track_id(object_id)
                    filter_manager.objectid_to_trackid[object_id] = track_id
                    filter_manager.new_trackIds.add(track_id)
                    logging.debug(f" [{origin_dataset}]: Created new trackId {track_id} for objectId {object_id}")

            if use_passthrough:
                track_event_df, track_head_df = copy_kinematics_to_track(
                    event_df, config, filter_manager, object_id_col
                )
            elif use_kalman:
                track_event_df = process_with_kalman(
                    event_df, config, filter_manager, object_id_col
                )
                # Create track head DataFrame
                track_head_df = track_event_df.copy()
                # trackUpdatedTimestamp must reflect the genuine observation
                # (intercept) time; never fabricate it from the current time.
                # interceptTimestamp is set from estimatedKinematics.kinematicsTimestamp
                # and the event loop already dropped records missing it, so it is
                # present and non-null here.
                track_head_df['trackUpdatedTimestamp'] = track_event_df['interceptTimestamp']

                # Drop columns not in track heads schema
                track_head_df.drop(columns=['interceptTimestamp', 'reportIds'], inplace=True, errors='ignore')
                track_head_df['stale'] = get_current_timestamp_string_with_offset(10)
                track_head_df = track_head_df.groupby('trackId', as_index=False).agg('last')
            else:
                # Should not reach here, but just in case
                continue


            # Add flag for newly-created trackIds
            track_head_df['isNewTrack'] = track_head_df['trackId'].isin(filter_manager.new_trackIds)

            if use_passthrough:
                track_head_df = track_head_df.groupby('objectId', as_index=False).agg('last')

            # --- Fix: Remove 'stale' column if present ---
            if 'stale' in track_head_df.columns:
                track_head_df = track_head_df.drop(columns=['stale'])

            _timings['process'] = _perf() - _t; _t = _perf()
            # Write to Crucible
            # Split track events to handle multiple events per object
            unique_dataframes = []
            track_event_df_copy = track_event_df.copy()

            while not track_event_df_copy.empty:
                unique_df = track_event_df_copy.drop_duplicates(subset=['objectId'], keep='first')
                unique_dataframes.append(unique_df)
                track_event_df_copy = track_event_df_copy[~track_event_df_copy['objectId'].isin(unique_df['objectId'])]

            # Handle track heads - split into new and existing
            new_tracks = track_head_df[track_head_df['isNewTrack']]
            existing_tracks = track_head_df[~track_head_df['isNewTrack']]
            new_tracks = new_tracks.drop(columns=['isNewTrack'])
            existing_tracks = existing_tracks.drop(columns=['isNewTrack'])

            # Token refresher for write operations
            _refresh_token = lambda: setattr(wc, 'token', auth.get_token())

            # Write track heads BEFORE events so that when the SSE fires for
            # the event, track_fusion can immediately find the head to stamp
            # associatedPrincipalTrack on (eliminates the race condition).

            failed_new_trackIds = set()
            heads_chunk_size = int(config['batch_update_chunk_size'])
            max_concurrent = config.get('batch_write_max_concurrent')
            # Track whether the head writes completed without an unexpected
            # error for NEW head creates. Existing-head PUT updates are
            # best-effort because live fusion consumes ComponentTrackEvents; a
            # slow/stuck head update must not block the event stream.
            heads_ok = True

            _timings['update_heads'] = _collect_finished_head_update_tasks(filter_manager, origin_dataset)
            if (not filter_manager.pending_head_update_tasks
                    and filter_manager.pending_head_updates_by_track):
                coalesced_updates = list(
                    filter_manager.pending_head_updates_by_track.values())
                filter_manager.pending_head_updates_by_track.clear()
                logging.info(
                    f" [{origin_dataset}]: scheduled coalesced background existing "
                    f"track-head update: records={len(coalesced_updates)}")
                filter_manager.pending_head_update_tasks.append(
                    asyncio.create_task(_best_effort_update_track_heads(
                        coalesced_updates, track_head_dataset,
                        heads_chunk_size, origin_dataset, max_concurrent))
                )

            # Upsert new track heads with batch→sub-chunk fallback.
            _t_ih = _perf()
            if not new_tracks.empty:
                try:
                    new_tracks_json = _fast_df_to_nested_json(new_tracks)
                    failed_records, _ = await write_batch_chunked(
                        new_tracks_json,
                        track_head_dataset,
                        wc.write_record_batch_by_name,
                        heads_chunk_size,
                        label=f' [{origin_dataset}]: ',
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent,
                    )
                    for record in failed_records:
                        track_id = record.get('trackId')
                        if track_id:
                            failed_new_trackIds.add(track_id)
                except Exception as head_err:
                    heads_ok = False
                    logging.error(f" [{origin_dataset}]: Unexpected error writing new track heads: {head_err}")
                    logging.error(traceback.format_exc())
            _timings['upsert_heads'] = _perf() - _t_ih

            # Only clear new_trackIds for records that were actually written successfully.
            # Keep failed trackIds in new_trackIds so they get re-inserted (not updated) next time.
            successfully_written_trackIds = set(new_tracks['trackId'].tolist()) if not new_tracks.empty else set()
            successfully_written_trackIds -= failed_new_trackIds
            filter_manager.new_trackIds -= successfully_written_trackIds

            # Existing head updates remain background/best-effort so they do
            # not hold up the live event stream, but failed records are
            # returned to the per-track pending buffer for retry.
            _t_sh = _perf()
            if not existing_tracks.empty:
                try:
                    existing_tracks = _coalesce_existing_head_updates(
                        existing_tracks,
                        filter_manager.existing_head_update_emit_time_by_track,
                        _head_update_interval_seconds(config),
                        origin_dataset,
                        label='component')
                    existing_tracks_json = _fast_df_to_nested_json(existing_tracks)
                    for head in existing_tracks_json:
                        track_id = head.get('trackId')
                        if isinstance(track_id, dict):
                            track_id = track_id.get('uuid')
                        if track_id is not None:
                            filter_manager.pending_head_updates_by_track[str(track_id)] = head
                    if (not filter_manager.pending_head_update_tasks
                            and filter_manager.pending_head_updates_by_track):
                        coalesced_updates = list(
                            filter_manager.pending_head_updates_by_track.values())
                        filter_manager.pending_head_updates_by_track.clear()
                        filter_manager.pending_head_update_tasks.append(
                            asyncio.create_task(_best_effort_update_track_heads(
                                coalesced_updates, track_head_dataset,
                                heads_chunk_size, origin_dataset, max_concurrent))
                        )
                except Exception as head_err:
                    logging.warning(
                        f" [{origin_dataset}]: Could not schedule best-effort existing track-head update: {head_err}")
            _timings['schedule_update_heads'] = _perf() - _t_sh
            # Track events are the reliable live input to object_track_fusion.
            # New head creates are awaited above; existing head updates are
            # best-effort and must not hold up event writes.
            _t_we = _perf()
            event_chunk_size = int(config['batch_write_chunk_size'])
            if not heads_ok:
                # A head write raised unexpectedly; skip event writes this cycle
                # so we never emit an event that references a missing head.
                logging.warning(
                    f" [{origin_dataset}]: Skipping track-event writes this cycle "
                    f"because a track-head write failed (avoids orphan events)"
                )
            else:
                for unique_df in unique_dataframes:
                    # Isolate each event sub-batch so one bad sub-batch does not
                    # abort the remaining event writes for this cycle.
                    try:
                        unique_json = _fast_df_to_nested_json(unique_df)
                        if len(unique_json) > 0:
                            await write_batch_chunked(
                                unique_json,
                                track_event_dataset,
                                wc.write_record_batch_by_name,
                                event_chunk_size,
                                label=f' [{origin_dataset}]: ',
                                token_refresher=_refresh_token,
                                max_concurrent_writes=max_concurrent,
                            )
                    except Exception as ev_err:
                        logging.error(f" [{origin_dataset}]: Unexpected error writing a track-event sub-batch: {ev_err}")
                        logging.error(traceback.format_exc())
                        continue

            _timings['write_events'] = _perf() - _t_we
            if PERF_TIMING:
                _timings['total'] = _perf() - _t0
                _timings['other'] = _timings['total'] - (
                    _timings.get('process', 0.0)
                    + _timings.get('upsert_heads', 0.0)
                    + _timings.get('update_heads', 0.0)
                    + _timings.get('schedule_update_heads', 0.0)
                    + _timings.get('write_events', 0.0))
                logging.info(
                    f" [{origin_dataset}]: [PERF tracker] "
                    f"total={_timings['total']:.3f}s "
                    f"process={_timings.get('process', 0.0):.3f}s "
                    f"upsert_heads={_timings.get('upsert_heads', 0.0):.3f}s "
                    f"update_heads={_timings.get('update_heads', 0.0):.3f}s "
                    f"schedule_update_heads={_timings.get('schedule_update_heads', 0.0):.3f}s "
                    f"write_events={_timings.get('write_events', 0.0):.3f}s "
                    f"other={_timings['other']:.3f}s "
                    f"| event_rows={len(event_df)} track_events={len(track_event_df)} "
                    f"track_heads={len(track_head_df)}")
            logging.info(f" [{origin_dataset}]: Wrote {len(track_event_df)} track events, {len(track_head_df)} track heads")

        except Exception as e:
            # Last-resort net so the tracker loop survives an unexpected error in
            # the processing phase. Head/event write failures are handled above
            # with phase-level isolation; reaching here drops only this batch.
            logging.error(f" [{origin_dataset}]: Unrecoverable error processing this batch (skipping it): {e}")
            logging.error(traceback.format_exc())
            continue


def process_with_kalman(event_df: pd.DataFrame, config: Dict[str, Any], 
                        filter_manager: ObjectKalmanFilterManager, 
                        object_id_col: str) -> pd.DataFrame:
    """
    Process kinematic data using StoneSoup Kalman filter.
    
    Args:
        event_df: DataFrame containing kinematic data
        config: Configuration dictionary
        filter_manager: ObjectKalmanFilterManager instance
        object_id_col: Name of the object ID column
        
    Returns:
        DataFrame: track event DataFrame
    """
    # Reset the per-batch out-of-order (too-old) measurement counters; they are
    # incremented in filter_manager.update_filter and reported in aggregate below.
    filter_manager._stale_skip_count = 0
    filter_manager._stale_skip_max = 0.0

    # The numpy backend applies the whole batch through a vectorized array path
    # (no per-row DataFrame access or StoneSoup Detection/LinearGaussian objects);
    # the StoneSoup backend falls through to the per-row loop below. Both leave
    # filter_manager.priors in the same state (validated by
    # test_tracker_kalman_equivalence.py).
    _numpy_batch = isinstance(filter_manager, NumpyKalmanFilterManager)
    if not _numpy_batch:
        raise RuntimeError(
            "object_tracker.py is StoneSoup-free; use NumpyKalmanFilterManager "
            "or object_tracker_stonesoup_deprecated.py for the legacy backend")
    if _numpy_batch:
        filter_manager.process_measurements_batch(event_df, object_id_col, config)

    # Process each measurement (StoneSoup backend)
    for idx, row in (() if _numpy_batch else event_df.iterrows()):
        object_id = row[object_id_col]

        # Set environment for this object so q and vel_var lookups work
        env = row.get('identity.environment.environment') or row.get('environment')
        if isinstance(env, str) and env and object_id not in filter_manager.environment_by_object:
            filter_manager.environment_by_object[object_id] = env
        
        # Build noise covariance from position/velocity covariance if available
        if 'positionCovariance.xx' in row and pd.notna(row.get('positionCovariance.xx')):
            noise_pos_covar = np.array([
                [row.get('positionCovariance.xx', 1e7), row.get('positionCovariance.xy', 0), row.get('positionCovariance.xz', 0)],
                [row.get('positionCovariance.xy', 0), row.get('positionCovariance.yy', 1e7), row.get('positionCovariance.yz', 0)],
                [row.get('positionCovariance.xz', 0), row.get('positionCovariance.yz', 0), row.get('positionCovariance.zz', 1e7)]
            ])
        else:
            noise_pos_covar = np.diag([1e7, 1e7, 1e7])
        
        # Check if velocity values are present (not NaN)
        _vx = row.get('ecefVelocity.x', row.get('ecefVelocity.dx', float('nan')))
        _vy = row.get('ecefVelocity.y', row.get('ecefVelocity.dy', float('nan')))
        _vz = row.get('ecefVelocity.z', row.get('ecefVelocity.dz', float('nan')))
        has_velocity_values = not (pd.isna(_vx) or pd.isna(_vy) or pd.isna(_vz))

        # Get timestamp
        ts = (row.get('estimatedKinematics.kinematicsTimestamp')
              or row.get('crucibleHeader.createdDate')
              or get_current_timestamp_string())
        try:
            time = dt.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
        except:
            time = dt.utcnow()

        if has_velocity_values:
            # 6D measurement: position + velocity
            has_velocity_covariance = 'velocityCovariance.dxdx' in row and pd.notna(row.get('velocityCovariance.dxdx'))
            if has_velocity_covariance:
                noise_vel_covar = np.array([
                    [row.get('velocityCovariance.dxdx', 1e7), row.get('velocityCovariance.dxdy', 0), row.get('velocityCovariance.dxdz', 0)],
                    [row.get('velocityCovariance.dxdy', 0), row.get('velocityCovariance.dydy', 1e7), row.get('velocityCovariance.dydz', 0)],
                    [row.get('velocityCovariance.dxdz', 0), row.get('velocityCovariance.dydz', 0), row.get('velocityCovariance.dzdz', 1e7)]
                ])
            else:
                # No velocity covariance — use environment-appropriate fallback
                vel_var = filter_manager._get_vel_var(object_id)
                noise_vel_covar = np.diag([vel_var, vel_var, vel_var])

            # Build interleaved covariance matrix for StoneSoup [x, vx, y, vy, z, vz]
            noise_covar = np.zeros((6, 6))
            noise_covar[0, 0] = noise_pos_covar[0, 0]  # xx
            noise_covar[2, 2] = noise_pos_covar[1, 1]  # yy
            noise_covar[4, 4] = noise_pos_covar[2, 2]  # zz
            noise_covar[0, 2] = noise_covar[2, 0] = noise_pos_covar[0, 1]  # xy
            noise_covar[0, 4] = noise_covar[4, 0] = noise_pos_covar[0, 2]  # xz
            noise_covar[2, 4] = noise_covar[4, 2] = noise_pos_covar[1, 2]  # yz
            noise_covar[1, 1] = noise_vel_covar[0, 0]  # dxdx
            noise_covar[3, 3] = noise_vel_covar[1, 1]  # dydy
            noise_covar[5, 5] = noise_vel_covar[2, 2]  # dzdz
            noise_covar[1, 3] = noise_covar[3, 1] = noise_vel_covar[0, 1]  # dxdy
            noise_covar[1, 5] = noise_covar[5, 1] = noise_vel_covar[0, 2]  # dxdz
            noise_covar[3, 5] = noise_covar[5, 3] = noise_vel_covar[1, 2]  # dydz

            measurement_model = LinearGaussian(
                ndim_state=6,
                mapping=(0, 1, 2, 3, 4, 5),
                noise_covar=noise_covar
            )
            measured_state_vector = np.array([
                row.get('ecefPosition.x', 0),
                _vx,
                row.get('ecefPosition.y', 0),
                _vy,
                row.get('ecefPosition.z', 0),
                _vz
            ])
        else:
            # 3D position-only measurement — velocity inferred by transition model
            measurement_model = LinearGaussian(
                ndim_state=6,
                mapping=(0, 2, 4),
                noise_covar=noise_pos_covar
            )
            measured_state_vector = np.array([
                row.get('ecefPosition.x', 0),
                row.get('ecefPosition.y', 0),
                row.get('ecefPosition.z', 0)
            ])

        measurement = Detection(
            measured_state_vector,
            timestamp=time,
            measurement_model=measurement_model
        )

        # Skip if measurement contains NaN values
        if np.isnan(measured_state_vector).any():
            logging.warning(f" [{config.get('origin_dataset', 'unknown')}]: Skipping measurement for object {object_id} - contains NaN values: {measured_state_vector}")
            continue
        
        # Update filter
        filter_manager.update_filter(object_id, measurement)

    # Aggregate the (benign, high-volume) out-of-order measurement skips into a
    # single per-batch line instead of one WARNING per measurement.  These are
    # stale/duplicate reports whose track was already updated by a newer one.
    _stale_n = getattr(filter_manager, '_stale_skip_count', 0)
    if _stale_n:
        logging.info(f" [{config.get('origin_dataset', 'unknown')}]: skipped {_stale_n} "
                     f"out-of-order (too-old) measurement(s) this batch "
                     f"(max {getattr(filter_manager, '_stale_skip_max', 0.0):.0f}s behind); "
                     f"newer measurements already updated these tracks")

    # Build output DataFrame
    track_df = pd.DataFrame()
    track_df['objectId'] = event_df[object_id_col]
    track_df['trackId'] = track_df['objectId'].map(filter_manager.objectid_to_trackid)
    
    # Copy identity fields
    if 'identity.standard.standardIdentity' in event_df.columns:
        track_df['standardIdentity'] = event_df['identity.standard.standardIdentity']
    elif 'standardIdentity' in event_df.columns:
        track_df['standardIdentity'] = event_df['standardIdentity']
    else:
        track_df['standardIdentity'] = 'UNKNOWN'
        
    if 'identity.environment.environment' in event_df.columns:
        track_df['environment'] = event_df['identity.environment.environment']
    elif 'environment' in event_df.columns:
        track_df['environment'] = event_df['environment']
        
    track_df['trackOriginatedTimestamp'] = event_df.get('crucibleHeader.createdDate', get_current_timestamp_string())
    track_df['interceptTimestamp'] = event_df['estimatedKinematics.kinematicsTimestamp']
    
    if 'edhControlSet' in event_df.columns:
        track_df['edhControlSet'] = event_df['edhControlSet']
    if 'mode' in event_df.columns:
        track_df['mode'] = event_df['mode']
    
    # Add reportIds from source.uuid
    track_df['reportIds'] = event_df['crucibleHeader.uuid'].map(
        lambda value: [normalize_uuid(value)] if value else [])

    # Add filtered positions/velocities/covariances from the Kalman posterior.
    # Collect each object's state into arrays in a light Python loop (dict +
    # numpy only), then assign all 27 output columns in bulk — was 27 per-cell
    # track_df.at[i, col] pandas scalar sets per row (~67k scalar sets per
    # Open_Sky batch).
    _n = len(event_df)
    _oids_out = event_df[object_id_col].to_numpy()
    _idx_out = event_df.index.to_numpy()
    _sv = np.full((_n, 6), np.nan)
    _cov = np.full((_n, 6, 6), np.nan)
    rows_to_drop = []
    _n_no_prior = 0
    _n_nan_state = 0
    for _pos in range(_n):
        object_id = _oids_out[_pos]
        state = filter_manager.priors.get(object_id)
        if state is None:
            _n_no_prior += 1
            rows_to_drop.append(_idx_out[_pos])
            continue
        _v = np.asarray(state.state_vector).reshape(-1)
        if np.isnan(_v).any():
            _n_nan_state += 1
            rows_to_drop.append(_idx_out[_pos])
            continue
        _sv[_pos] = _v
        _cov[_pos] = np.asarray(state.covar)
    if _n_no_prior:
        logging.warning(f" [{config.get('origin_dataset', 'unknown')}]: {_n_no_prior} object(s) "
                        f"had no filter state - dropped from output")
    if _n_nan_state:
        logging.warning(f" [{config.get('origin_dataset', 'unknown')}]: {_n_nan_state} object(s) "
                        f"had a NaN Kalman state - dropped from output")

    # State [x, vx, y, vy, z, vz]
    track_df['ecefPosition.x'] = _sv[:, 0]
    track_df['ecefPosition.y'] = _sv[:, 2]
    track_df['ecefPosition.z'] = _sv[:, 4]
    track_df['ecefVelocity.dx'] = _sv[:, 1]
    track_df['ecefVelocity.dy'] = _sv[:, 3]
    track_df['ecefVelocity.dz'] = _sv[:, 5]
    track_df['positionCovariance.xx'] = _cov[:, 0, 0]
    track_df['positionCovariance.xy'] = _cov[:, 0, 2]
    track_df['positionCovariance.xz'] = _cov[:, 0, 4]
    track_df['positionCovariance.yy'] = _cov[:, 2, 2]
    track_df['positionCovariance.yz'] = _cov[:, 2, 4]
    track_df['positionCovariance.zz'] = _cov[:, 4, 4]
    track_df['velocityCovariance.dxdx'] = _cov[:, 1, 1]
    track_df['velocityCovariance.dxdy'] = _cov[:, 1, 3]
    track_df['velocityCovariance.dxdz'] = _cov[:, 1, 5]
    track_df['velocityCovariance.dydy'] = _cov[:, 3, 3]
    track_df['velocityCovariance.dydz'] = _cov[:, 3, 5]
    track_df['velocityCovariance.dzdz'] = _cov[:, 5, 5]
    track_df['positionVelocityCovariance.xdx'] = _cov[:, 0, 1]
    track_df['positionVelocityCovariance.xdy'] = _cov[:, 0, 3]
    track_df['positionVelocityCovariance.xdz'] = _cov[:, 0, 5]
    track_df['positionVelocityCovariance.ydx'] = _cov[:, 2, 1]
    track_df['positionVelocityCovariance.ydy'] = _cov[:, 2, 3]
    track_df['positionVelocityCovariance.ydz'] = _cov[:, 2, 5]
    track_df['positionVelocityCovariance.zdx'] = _cov[:, 4, 1]
    track_df['positionVelocityCovariance.zdy'] = _cov[:, 4, 3]
    track_df['positionVelocityCovariance.zdz'] = _cov[:, 4, 5]

    # Drop rows with NaN states
    if rows_to_drop:
        track_df = track_df.drop(rows_to_drop)
        logging.info(f" [{config.get('origin_dataset', 'unknown')}]: Dropped {len(rows_to_drop)} rows with NaN/missing states from track output")
    
    return track_df


def copy_kinematics_to_track(event_df: pd.DataFrame, config: Dict[str, Any],
                              filter_manager: ObjectKalmanFilterManager,
                              object_id_col: str) -> tuple:
    """
    Copy kinematic data directly without filtering (passthrough mode).
    
    Args:
        event_df: DataFrame containing kinematic data
        config: Configuration dictionary
        filter_manager: ObjectKalmanFilterManager instance
        object_id_col: Name of the object ID column
        
    Returns:
        tuple: (track_event_df, track_head_df)
    """
    track_df = pd.DataFrame()
    track_df['objectId'] = event_df[object_id_col]
    track_df['trackId'] = track_df['objectId'].map(filter_manager.objectid_to_trackid)
    
    # Copy identity fields
    if 'identity.standard.standardIdentity' in event_df.columns:
        track_df['standardIdentity'] = event_df['identity.standard.standardIdentity']
    elif 'standardIdentity' in event_df.columns:
        track_df['standardIdentity'] = event_df['standardIdentity']
    else:
        track_df['standardIdentity'] = 'UNKNOWN'
        
    if 'identity.environment.environment' in event_df.columns:
        track_df['environment'] = event_df['identity.environment.environment']
    elif 'environment' in event_df.columns:
        track_df['environment'] = event_df['environment']
    
    track_df['trackOriginatedTimestamp'] = event_df.get('crucibleHeader.createdDate', get_current_timestamp_string())
    track_df['interceptTimestamp'] = event_df['estimatedKinematics.kinematicsTimestamp']
    
    if 'edhControlSet' in event_df.columns:
        track_df['edhControlSet'] = event_df['edhControlSet']
    if 'mode' in event_df.columns:
        track_df['mode'] = event_df['mode']
    
    # Add reportIds from crucibleHeader.uuid
    if 'crucibleHeader.uuid' in event_df.columns:
        track_df['reportIds'] = event_df['crucibleHeader.uuid'].map(
            lambda value: [normalize_uuid(value)] if value else [])
    else:
        track_df['reportIds'] = [[] for _ in range(len(event_df))]
    
    # Copy all kinematic columns in ONE assignment (avoids fragmenting track_df
    # with a separate column insert per matched column).
    _kin_prefixes = ('ecefPosition.', 'ecefVelocity.', 'positionCovariance.',
                     'velocityCovariance.', 'positionVelocityCovariance.', 'geodetic.')
    _kin_cols = [c for c in event_df.columns if c.startswith(_kin_prefixes)]
    if _kin_cols:
        track_df[_kin_cols] = event_df[_kin_cols]
    
    # Create track head DataFrame
    track_head_df = track_df.copy()
    track_head_df['trackUpdatedTimestamp'] = track_df['interceptTimestamp']
    track_head_df.drop(columns=['interceptTimestamp', 'reportIds'], inplace=True, errors='ignore')
    track_head_df['stale'] = get_current_timestamp_string_with_offset(10)
    
    return track_df, track_head_df


async def sse_listener_launcher(event_queue: Queue, config: dict) -> None:
    """Launch SSE listener for object events."""
    coroutine_list = []
    
    # Build SQL query for object events
    object_event_dataset = config.get('object_event_dataset', 
                                       config.get('perspective', 'Live_POV') + '_Object_Events')
    
    # Filter by origin dataset if specified
    origin_dataset = config.get('origin_dataset')
    if origin_dataset:
        sql_query = f"SELECT * FROM {object_event_dataset} WHERE {object_event_dataset}.`source`.datasetName = '{origin_dataset}'"
    else:
        sql_query = f"SELECT * FROM {object_event_dataset}"

    logging.info(f" [{origin_dataset}]: Object Tracker SSE Listener SQL Query: {sql_query}")
    coroutine_list.append(
        SSE_listener(sql_query, event_queue, auth)
    )

    excepts = await asyncio.gather(*coroutine_list, return_exceptions=True)
    for exc in excepts:
        if isinstance(exc, Exception):
            logging.error(f"An error occurred during SSE listener: {exc}")


def run(stream_manager_perspective: str, log_level: str = None) -> None:
    """
    Main function to run the object tracker.
    
    Args:
        stream_manager_perspective: Perspective name for Stream Manager (e.g., 'Live_POV')
        log_level: Logging level (INFO, DEBUG, WARN, ERROR)
    """
    # Set up logging
    if log_level is None:
        log_level = 'INFO'
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register signal handlers
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    
    logging.info(f"Starting Object Tracker for perspective: {stream_manager_perspective}")
    
    # Initialize controllers in parent process
    initialize_controllers()
    
    # Load configurations from Stream Manager
    config_list = find_and_validate_configs(stream_manager_perspective, include_scripts=False, rc_instance=rc,)
    
    if not config_list:
        logging.error(f"No configurations found for perspective: {stream_manager_perspective}")
        return
    
    processes = []
    sse_coros = []
    # Cache of downloaded head datasets so multiple feeds sharing the same
    # component_track_head_dataset don't each re-download it.
    head_cache: Dict[str, List[dict]] = {}

    for config in config_list:
        if config.get('disabled'):
            logging.info(f"Skipping disabled config for {config.get('origin_dataset')}")
            continue

        if config.get('skip_tracker'):
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} as per configuration")
            continue
        # Check if 'crucible_tracker' is set and valid
        crucible_tracker = config.get('crucible_tracker', '')
        if not crucible_tracker:
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - 'crucible_tracker' not set in config")
            continue
        crucible_tracker_lower = crucible_tracker.lower()
        if 'skip' in crucible_tracker_lower:
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - 'crucible_tracker' contains 'skip'")
            continue
        if '3rd party' in crucible_tracker_lower or 'third-party' in crucible_tracker_lower:
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - 'crucible_tracker' contains '3rd party' or 'third-party'")
            continue
        # Ensure component track datasets are set
        if not config.get('component_track_head_dataset') or not config.get('component_track_event_dataset'):
            logging.info(f"Skipping tracker for dataset {config.get('origin_dataset')} - component track datasets not set")
            continue
        # Store log level so child process can configure its own logging
        config['_log_level'] = numeric_level

        # Create event queue for this config (SSE listener writes here)
        event_queue = Queue()

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
        # workers (pre-bucketed by objectId) so each shard initializes from its
        # queue instead of independently querying Crucible. Runs before the
        # dispatcher (started below in run_all_sse) so heads are consumed first.
        _fanout_head_preload_tracker(shard_queues, config, n_shards, head_cache)

        # Collect SSE listener coroutines + dispatcher coroutine
        sse_coros.append(sse_listener_launcher(event_queue, config))
        sse_coros.append(shard_dispatcher(event_queue, shard_queues, config))

    # Run all SSE listeners in a single event loop
    async def run_all_sse() -> None:
        await asyncio.gather(*sse_coros)

    try:
        asyncio.run(run_all_sse())
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except BaseException:
        logging.critical("Object Tracker event loop exited unexpectedly")
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
    parser = argparse.ArgumentParser(description='Object Tracker for V1 Schema')
    parser.add_argument('stream_manager_perspective', 
                        help='Name of perspective in Stream Manager Configuration (e.g., Live_POV)')
    parser.add_argument('--log', 
                        help='Logging level: INFO (default), WARN, ERROR, or DEBUG',
                        default='INFO')
    parser.add_argument('--use-fork', action='store_true',
                        help='Use the fork start method for multiprocessing')
    args = parser.parse_args()
    
    if args.use_fork:
        set_start_method('forkserver', force=True)
        logging.info("  Using the forkserver start method for multiprocessing  ")
    
    run(args.stream_manager_perspective, log_level=args.log)
