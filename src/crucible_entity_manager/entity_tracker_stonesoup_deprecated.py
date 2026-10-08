#!/usr/bin/env python
"""
entity_beta.entity_tracker -- TRACKER (entity_beta pipeline)

Differences from entity_manager.entity_tracker:
- Reads raw Report_Events directly (the report_event_with_entityId dataset is gone).
- Does NOT assign an entityId to tracks. Each report event is grouped into a
  component track by a deterministic trackId generated directly from the
  (origin_dataset, identity customID) pair -- uuid5(origin_dataset:customID).
  (entityId may be assigned to principal track heads by a downstream stage in
  the future; the tracker no longer produces it.)
- Writes the full identity.* fields into the component track event/head datasets,
  making component track heads the store relating trackId <-> identity fields.
The Kalman / passthrough tracking logic is otherwise identical to the original.
"""
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
import os
import re
import hashlib

# StoneSoup imports
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.types.state import GaussianState

try:
    from cruciblelib import utils
    from cruciblelib.decorators import async_retry
except ImportError:
    from . import utils
    from .decorators import async_retry

try:
    # entity_beta is self-contained: use the entity_beta copy of the shared
    # utilities (the same module the fuser and duplicate identifier use). Bare
    # import works when entity_beta is on the path; relative import works when
    # imported as the entity_beta.entity_tracker package.
    from entity_utils import (
        instantiate_api_controllers,
        find_and_validate_configs,
        terminate, write_batch_chunked, write_entity_updates_with_create_fallback,
        SSE_listener,
    )
    from entity_transformer_records import (
        add_track_wgs84_kinematics, compact_record,
    )
except ImportError:
    from .entity_utils import (
        instantiate_api_controllers,
        find_and_validate_configs,
        terminate, write_batch_chunked, write_entity_updates_with_create_fallback,
        SSE_listener,
    )
    from .entity_transformer_records import (
        add_track_wgs84_kinematics, compact_record,
    )

# Global variables for controllers (will be initialized in the child process)
rc = wc = auth = None

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def _flatten_record(record: dict, prefix: str = '') -> dict:
    flattened = {}
    for key, value in record.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_record(value, path))
        else:
            flattened[path] = value
    return flattened


def _unflatten_record(record: dict) -> dict:
    nested = {}
    for path, value in record.items():
        if value is None:
            continue
        if not isinstance(value, (list, tuple, dict, np.ndarray)):
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
        if hasattr(value, 'item'):
            value = value.item()
        current = nested
        parts = path.split('.')
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def _fast_df_to_nested_json(df: pd.DataFrame) -> List[dict]:
    return [compact_record(_unflatten_record(record)) for record in df.to_dict('records')]


def update_timestamps(df: pd.DataFrame, timestamp_columns: List[str]) -> pd.DataFrame:
    result = df.copy()
    for column in timestamp_columns:
        if column not in result:
            continue
        timestamps = pd.to_datetime(result[column], utc=True, format='ISO8601', errors='coerce')
        result[column] = timestamps.dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str[:-3] + 'Z'
        result.loc[timestamps.isna(), column] = None
    return result


def add_WGS84_kinematics(df: pd.DataFrame, drop_geodetic: bool = False) -> pd.DataFrame:
    records = [_unflatten_record(record) for record in df.to_dict('records')]
    enriched = add_track_wgs84_kinematics(records)
    result = pd.DataFrame.from_records([_flatten_record(record) for record in enriched])
    if drop_geodetic:
        result.drop(
            columns=[column for column in result if column.startswith('geodetic.')],
            inplace=True,
        )
    return result


# ---------------------------------------------------------------------------
# Track-id helpers.
#
# entity_beta no longer assigns an entityId to tracks. Each report event is
# grouped into a component track by a deterministic trackId built directly from
# the identity fields: customID is the sorted concatenation of all identity.*
# fields, and the trackId is uuid5(NAMESPACE_DNS, "{origin_dataset}:{customID}")
# -- stable across runs and processes, and unique per data feed.
# ---------------------------------------------------------------------------

def safe_str(x: Any) -> str:
    """Convert a value to a string for customID building (ported from
    entity_manager). Returns '' for NA/None and renders whole-number floats as
    ints so the customID matches the entity_manager identity concatenation."""
    if pd.isna(x):
        return ''
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)) and x == int(x):
        return str(int(x))
    if hasattr(x, 'dtype') and 'int' in str(x.dtype).lower():
        return str(int(x))
    if isinstance(x, str) and '.' in x:
        try:
            f = float(x)
            if f == int(f):
                return str(int(f))
        except (ValueError, OverflowError):
            pass
    return str(x)


def string_concat_ID_fields(row: pd.Series) -> str:
    """Build the customID from a flattened report/entity row (identity.* columns).
    Single source of truth with _identity_custom_id_from_dict."""
    id_key_value = {}
    for col, val in row.items():
        if 'identity.' in col:
            trunc_col = col.replace('identity.', '')
            id_key_value[trunc_col] = safe_str(val)
    parts = [f"{col}:{val}" for col, val in sorted(id_key_value.items()) if val != '']
    return '-'.join(parts)


def _identity_custom_id_from_dict(identity: dict) -> str:
    """Build the customID from a nested identity dict (raw SSE event shape).
    Produces the same string as string_concat_ID_fields for the same identity."""
    if not isinstance(identity, dict):
        return ''
    id_key_value = {col: safe_str(val) for col, val in identity.items()}
    parts = [f"{col}:{val}" for col, val in sorted(id_key_value.items()) if val != '']
    return '-'.join(parts)


def _track_id_from_custom_id(origin_dataset: str, custom_id: str) -> str:
    """Deterministic trackId = uuid5(origin_dataset : identity customID). Stable
    across runs and processes, and unique per data feed."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{origin_dataset}:{custom_id}").hex


def assign_track_id(event: dict, origin_dataset: str) -> str:
    """Resolve a deterministic trackId for a raw report event from its identity
    fields. entity_beta no longer assigns an entityId to tracks -- the trackId is
    generated directly from the (origin_dataset, identity customID) pair."""
    custom_id = _identity_custom_id_from_dict(event.get('identity', {}))
    return _track_id_from_custom_id(origin_dataset, custom_id)


def _shard_for_track(track_id: Any, n_shards: int) -> int:
    """Stable shard assignment for a trackId. Uses md5 (not the builtin hash,
    which is PYTHONHASHSEED-salted and would move tracks between shards across
    restarts)."""
    digest = hashlib.md5(
        str(track_id).encode('utf-8'), usedforsecurity=False).hexdigest()
    return int(digest, 16) % n_shards


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
            match = re.search(r'q\s*=\s*(\d+(\.\d+)?)', crucible_kalman_config_string)
            if match:
                q_override = float(match.group(1))
                self.q_default = q_override
                logging.info(f"Using q={q_override} from config as default")

            if preloaded_heads is not None:
                # Records already flattened by the parent's fan-out.
                heads_df = pd.DataFrame(preloaded_heads) if preloaded_heads else pd.DataFrame()
            else:
                # Fallback path (used only when the parent didn't fan out heads,
                # e.g. a direct/non-sharded call). Use the same robust, unlimited
                # download-once loader instead of a capped search that 504s on
                # large head datasets.
                heads = await asyncio.to_thread(
                    _load_all_records, self.config.get('component_track_head_dataset'), self.config)

                heads_df = utils.flatten_crucible_dataset(heads)
            
            if len(heads_df) > 0:
                heads_df = update_timestamps(heads_df, ['trackUpdatedTimestamp'])
                
                for idx,head in heads_df.iterrows():
                    track_id = head.get('trackId')
                    # Feed-ownership check: recompute the deterministic trackId from
                    # the head's identity.* fields and keep only heads for this feed.
                    custom_id = string_concat_ID_fields(head)
                    expected_track_id = _track_id_from_custom_id(self.origin_dataset, custom_id)
                    
                    # Only load heads that belong to this feed
                    if track_id != expected_track_id:
                        continue
                    
                    self.known_trackids.add(track_id)
                    
                    # Only initialize Kalman filter if passthrough_tracker is False
                    if not self.passthrough_tracker:
                        # Set environment from head data
                        env = head.get('identity.environment') or head.get('environment')
                        if env:
                            self.environment_by_track[track_id] = str(env)

                        # Create filter components with environment-aware q
                        q = self._get_q(track_id)
                        transition_model = CombinedLinearGaussianTransitionModel([
                            ConstantVelocity(q),
                            ConstantVelocity(q),
                            ConstantVelocity(q)
                        ])
                        
                        noise_covar = np.diag([1.e7, 1.e7, 1.e7, 1.e7, 1.e7, 1.e7])
                        measurement_model = LinearGaussian(
                            ndim_state=6,
                            mapping=(0, 1, 2, 3, 4, 5),
                            noise_covar=noise_covar
                        )
                        
                        updater = KalmanUpdater(measurement_model)
                        
                        # Create predictor and updater
                        predictor = KalmanPredictor(transition_model)
                        
                        # Create initial state from head data
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
                            logging.warning(f" [{self.origin_dataset}]: Track {track_id} initial state contains NaN - replacing with zeros and resetting covariance")
                            initial_state_vector = np.nan_to_num(initial_state_vector, nan=0.0)
                            
                        # Create initial covariance from head data
                        initial_covar = np.zeros((6, 6))
                        if 'positionVelocityCovariance.xdx' in head:
                            # Position-velocity covariance terms
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
                            logging.warning(f" [{self.origin_dataset}]: Track {track_id} initial covariance contains NaN - using default high-uncertainty covariance")
                            initial_covar = np.diag([1.e7, 1.e7, 1.e7, 1.e7, 1.e7, 1.e7])

                        # Create and store prior
                        timestamp_str = head.get('trackUpdatedTimestamp', dt.utcnow().isoformat() + 'Z')
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
                        self.predictor[track_id] = predictor
                        self.updater[track_id] = updater
                        self.measurement_model[track_id] = measurement_model
                
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

    def process_measurements_batch(self, event_df: pd.DataFrame,
                                   config: Dict[str, Any]) -> Dict[Any, GaussianState]:
        """Vectorized equivalent of the per-row measurement-build + predict_and_update
        loop in process_with_kalman for the numpy backend.

        Extracts every needed column as a numpy array once (no DataFrame
        .iterrows / per-cell .get), parses the interceptTimestamps in bulk, then
        applies each measurement via _update_core (no per-event StoneSoup
        Detection / LinearGaussian). Returns ``{row_index -> posterior
        GaussianState}`` mirroring the per-row ``posteriors`` the StoneSoup loop
        records, so the downstream output build is unchanged. Rows with a NaN
        measured position are skipped (no posterior), exactly like the per-row
        loop's ``continue``.
        """
        posteriors: Dict[Any, GaussianState] = {}
        n = len(event_df)
        if n == 0:
            return posteriors
        cols = event_df.columns
        origin_dataset = config.get('origin_dataset', 'unknown')
        index = event_df.index.to_numpy()
        self._stale_skip_count = 0
        self._stale_skip_max = 0.0

        def _col(name, default):
            return event_df[name].to_numpy() if name in cols else np.full(n, default)

        tid = event_df['trackId'].to_numpy()

        px = _col('ecefPosition.x', np.nan).astype(float)
        py = _col('ecefPosition.y', np.nan).astype(float)
        pz = _col('ecefPosition.z', np.nan).astype(float)
        vx = _col('ecefVelocity.x', np.nan).astype(float)
        vy = _col('ecefVelocity.y', np.nan).astype(float)
        vz = _col('ecefVelocity.z', np.nan).astype(float)

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

        # Environment: identity.environment then environment (matches per-row
        # `row.get('identity.environment') or row.get('environment')`).
        ident_env = (event_df['identity.environment'].to_numpy()
                     if 'identity.environment' in cols else None)
        plain_env = (event_df['environment'].to_numpy()
                     if 'environment' in cols else None)

        # interceptTimestamp parsed in bulk (naive), fallback now for unparseable.
        its = (event_df['interceptTimestamp'].to_numpy(dtype=object)
               if 'interceptTimestamp' in cols else np.full(n, None, dtype=object))
        parsed = pd.to_datetime(its, format="%Y-%m-%dT%H:%M:%S.%fZ", errors='coerce')
        now_dt = dt.utcnow()
        ts_list = [(p.to_pydatetime() if not pd.isna(p) else now_dt) for p in parsed]

        ebt = self.environment_by_track
        map6 = (0, 1, 2, 3, 4, 5)
        map3 = (0, 2, 4)
        for i in range(n):
            track_id = tid[i]

            # Set environment for this track (first non-empty), before the q /
            # vel_var lookups, exactly as the per-row loop does.
            e = ident_env[i] if ident_env is not None else None
            if not e and plain_env is not None:
                e = plain_env[i]
            if e and track_id not in ebt:
                ebt[track_id] = str(e)

            x = px[i]; y = py[i]; z = pz[i]
            _vx = vx[i]; _vy = vy[i]; _vz = vz[i]
            has_vel = not (_vx != _vx or _vy != _vy or _vz != _vz)

            if has_pos_cov_col and pcxx[i] == pcxx[i]:  # column present and row notna
                pxx = pcxx[i]; pyy = pcyy[i]; pzz = pczz[i]
                pxy = pcxy[i]; pxz = pcxz[i]; pyz = pcyz[i]
            else:
                pxx = pyy = pzz = 1.e7
                pxy = pxz = pyz = 0.0

            ts_i = ts_list[i]
            if has_vel:
                if has_vel_cov_col and vcxx[i] == vcxx[i]:
                    vxx = vcxx[i]; vyy = vcyy[i]; vzz = vczz[i]
                    vxy = vcxy[i]; vxz = vcxz[i]; vyz = vcyz[i]
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
                posteriors[index[i]] = self._update_core(track_id, meas, map6, nc, ts_i)
            else:
                pc = np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
                meas = np.array([x, y, z])
                if np.isnan(meas).any():
                    logging.warning(f" [{origin_dataset}]: Skipping measurement for track {track_id} - contains NaN values")
                    continue
                posteriors[index[i]] = self._update_core(track_id, meas, map3, pc, ts_i)

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
    if _skip_head_preload(config):
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
            recs = _load_all_records(head_ds, config)
            if head_cache is not None:
                head_cache[head_ds] = recs
        if recs:
            hdf = utils.flatten_crucible_dataset(recs)
            for row in hdf.to_dict('records'):
                tid = row.get('trackId')
                if tid is None or (isinstance(tid, float) and pd.isna(tid)):
                    tid = row.get('trackId.uuid')
                if tid is None or (isinstance(tid, float) and pd.isna(tid)):
                    continue
                shard = _shard_for_track(tid, n_shards)
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


def _run_tracker_shard(shard_queue: Any, config: dict) -> None:
    """Launch a sharded tracker worker that processes a subset of trackIds."""
    asyncio.run(tracker(shard_queue, config))


async def shard_dispatcher(event_queue: Any, shard_queues: List[Any], config: dict) -> None:
    """
    Reads raw Report_Events from the main event_queue, ASSIGNS a deterministic
    trackId to each from its identity fields, then partitions by trackId and
    routes to the appropriate shard queue.

    trackId is a stable uuid5 of "{origin_dataset}:{identity customID}". entity_beta
    no longer assigns an entityId to tracks. Sharding by trackId keeps each
    track's Kalman state in exactly one shard.
    Runs as an asyncio task in the parent process alongside SSE listeners.
    """
    n_shards = len(shard_queues)
    origin_dataset = config.get('origin_dataset', 'unknown')

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
            # Deterministic trackId from the identity.* fields.
            track_id = assign_track_id(event, origin_dataset)
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
    initialize_controllers()

    if config.get('disabled'):
        logging.info(f" [{config.get('origin_dataset')}]: No tracking for dataset {config.get('origin_dataset')}")
        logging.info(f" [{config.get('origin_dataset')}]:   as it has Entity Stream Manager configuration option disabled set to True")
        return

    # Initialize Kalman filter manager. The numpy backend is a behaviourally-
    # identical (verified in test_entity_tracker_kalman_equivalence.py), faster
    # drop-in for the StoneSoup filter; opt in per feed via the 'use_numpy_kalman'
    # config key or cluster-wide via CRUCIBLE_USE_NUMPY_KALMAN.
    _use_numpy_kalman = config.get(
        'use_numpy_kalman',
        os.getenv('CRUCIBLE_USE_NUMPY_KALMAN', 'false').strip().lower() in ('1', 'true', 'yes'))
    _manager_cls = NumpyKalmanFilterManager if _use_numpy_kalman else KalmanFilterManager
    filter_manager = _manager_cls(config)
    logging.info(f" [{config.get('origin_dataset')}]: Kalman backend = {'numpy' if _use_numpy_kalman else 'stonesoup'}")

    tracker_mode = config.get('crucible_tracker') or ''
    tracker_mode_lower = tracker_mode.lower()

    use_ci = 'covariance intersection' in tracker_mode_lower or 'ci' in tracker_mode_lower.split()

    if 'passthrough' in tracker_mode_lower:
        filter_manager.passthrough_tracker = True
    else:
        filter_manager.passthrough_tracker = False

    if ('ECEF' in tracker_mode or 'kalman' in tracker_mode_lower or use_ci) and filter_manager.passthrough_tracker is False:
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
            
            
            event_df = utils.flatten_crucible_dataset(events)

            # Belt-and-suspenders: a report event with no interceptTimestamp is a
            # position with no observation time. Drop those rows so no timeless
            # track is produced. The transformer normally filters them, but events
            # may arrive from other producers if the transformer is bypassed.
            if 'interceptTimestamp' in event_df.columns:
                _n_before = len(event_df)
                event_df = event_df[event_df['interceptTimestamp'].notna()]
                _n_dropped = _n_before - len(event_df)
                if _n_dropped:
                    logging.warning(f" [{config.get('origin_dataset')}]: dropped {_n_dropped} "
                                    f"report event(s) missing interceptTimestamp")
            else:
                logging.warning(f" [{config.get('origin_dataset')}]: no interceptTimestamp "
                                f"column in this batch - skipping")
                continue
            if len(event_df) == 0:
                continue

            event_df = event_df.sort_values(by='interceptTimestamp',ignore_index=True,ascending=True)
            # The dispatcher already assigned a deterministic trackId to each event.
            # Flag any trackId not seen before (this run or loaded from heads) as new.
            track_ids = event_df['trackId'].unique()
            for track_id in track_ids:
                if track_id not in filter_manager.known_trackids:
                    filter_manager.known_trackids.add(track_id)
                    filter_manager.new_trackIds.add(track_id)
    
            if filter_manager.passthrough_tracker is True:
                    # Passthrough mode - just copy the data
                    track_event_df, track_head_df = \
                        copy_kinematics_to_track(event_df, config, filter_manager)
  

            if filter_manager.ecef_kalman_filter is True and filter_manager.passthrough_tracker is False:
                # Kalman filter mode
                event_df = update_timestamps(event_df, ['interceptTimestamp'])
                track_event_df  = \
                        process_with_kalman(event_df, config, filter_manager)
                #add WGS84 kinematics
            
                # Create track head DataFrame
                track_head_df = track_event_df.copy()
                track_head_df['trackUpdatedTimestamp'] = track_event_df['interceptTimestamp']
                drop_cols = [c for c in ['trackQuality', 'interceptTimestamp'] if c in track_head_df.columns]
                track_head_df.drop(columns=drop_cols, inplace=True)
                track_head_df['stale'] = get_current_timestamp_string_with_offset(10)
                track_head_df=track_head_df.groupby('trackId',as_index=False).agg('last')
                
                
                track_event_df = add_WGS84_kinematics(track_event_df, drop_geodetic=False)
                track_head_df = add_WGS84_kinematics(track_head_df, drop_geodetic=False)

            if not filter_manager.ecef_kalman_filter and not filter_manager.passthrough_tracker:
                logging.warning(f" [{config.get('origin_dataset')}]: No tracker mode configured for {config.get('origin_dataset')} - skipping batch")
                continue
                
            # Add flag for newly-created trackIds
            track_head_df['isNewTrack'] = track_head_df['trackId'].isin(filter_manager.new_trackIds)

            if filter_manager.passthrough_tracker is True:
                track_head_df = track_head_df.groupby('trackId',as_index=False).agg('last') # this happens earlier for kalman processed data

            if filter_manager.ecef_kalman_filter is True or filter_manager.passthrough_tracker is True:

                _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
                event_chunk_size = int(config['batch_write_chunk_size'])
                heads_chunk_size = int(config['batch_update_chunk_size'])
                max_concurrent = config.get('batch_write_max_concurrent')

                # Handle track heads separately - split into new and existing tracks
                new_tracks = track_head_df[track_head_df['isNewTrack']]
                existing_tracks = track_head_df[~track_head_df['isNewTrack']]
                new_tracks = new_tracks.drop(columns=['isNewTrack'])
                existing_tracks = existing_tracks.drop(columns=['isNewTrack'])

                # Write NEW track heads FIRST so we never emit component track
                # events that reference a head that failed to create. Any head
                # that fails is kept in new_trackIds for retry and its events are
                # withheld this cycle (re-sent once the head lands next cycle).
                failed_trackIds = set()
                if not new_tracks.empty:
                    new_tracks_json = _fast_df_to_nested_json(new_tracks)
                    head_dataset = config['component_track_head_dataset']
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
                successfully_written = set(new_tracks['trackId'].tolist()) if not new_tracks.empty else set()
                successfully_written -= failed_trackIds
                filter_manager.new_trackIds -= successfully_written

                # Update existing tracks via entity-aware partial update.
                # POST would fully overwrite the head and wipe the
                # associatedPrincipalTrack field owned by the fuser; an update
                # only sets the kinematic/head fields we send here. Any head the
                # server reports as not-found (e.g. expired) is re-created.
                if not existing_tracks.empty:
                    existing_tracks_json = _fast_df_to_nested_json(existing_tracks)
                    head_dataset = config['component_track_head_dataset']
                    await write_entity_updates_with_create_fallback(
                        existing_tracks_json,
                        head_dataset,
                        wc,
                        update_chunk_size=heads_chunk_size,
                        create_chunk_size=heads_chunk_size,
                        create_records=existing_tracks_json,
                        key_field='trackId',
                        label=f' [{config.get("origin_dataset")}]: ',
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent,
                    )

                # Now write track events, EXCLUDING any whose new head failed to
                # create this cycle (avoids orphan events with no head).
                track_event_df_copy = track_event_df.copy()

                # Add reportIds for component tracks only
                track_event_df_copy['reportIds'] = event_df['source.uuid'].map(lambda x: [x])

                if failed_trackIds:
                    track_event_df_copy = track_event_df_copy[~track_event_df_copy['trackId'].isin(failed_trackIds)]

                # Split track_event_df into multiple DataFrames, each with unique trackIds
                unique_dataframes = []
                while not track_event_df_copy.empty:
                    # Keep only the first occurrence of each trackId
                    unique_df = track_event_df_copy.drop_duplicates(subset=['trackId'], keep='first')
                    unique_dataframes.append(unique_df)

                    # Remove the rows with trackIds already processed
                    track_event_df_copy = track_event_df_copy[~track_event_df_copy['trackId'].isin(unique_df['trackId'])]

                # Write all unique DataFrames to Crucible with chunked concurrent execution
                for unique_df in unique_dataframes:
                    unique_json = _fast_df_to_nested_json(unique_df)
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

                logging.info(f" [{config.get('origin_dataset')}]: Wrote track events for {len(track_head_df)} track heads"
                             f"{f' ({len(failed_trackIds)} new heads withheld)' if failed_trackIds else ''}")

        except Exception as e:
            logging.error(f" [{config.get('origin_dataset')}]: An error occurred in tracker: {e}")
            logging.error(traceback.format_exc())
            continue

async def sse_listener_launcher(event_queue: Any,
                                config: dict) -> None:

    coroutine_list = []
    # SQL query that filters on origin dataset--this allows us to treat each source differently.
    # entity_beta reads the raw Report_Events dataset directly (no
    # report_event_with_entityId stage); the dispatcher assigns a deterministic trackId.
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


def process_with_kalman(event_df: pd.DataFrame, config: Dict[str, Any], filter_manager: KalmanFilterManager) -> pd.DataFrame:
    """
    Process kinematic data using StoneSoup Kalman filter.
    
    Args:
        event_df: DataFrame containing kinematic data
        config: Configuration dictionary
        filter_manager: KalmanFilterManager instance
        
    Returns:
        DataFrame: track event DataFrame
    """



    # Process each measurement
    posteriors = {}  # idx -> GaussianState posterior for each row

    # The numpy backend applies the whole batch through a vectorized array path
    # (no per-row DataFrame access or StoneSoup Detection/LinearGaussian objects)
    # and returns the same per-row posteriors dict; the StoneSoup backend falls
    # through to the per-row loop below. Both produce identical posteriors
    # (validated by test_entity_tracker_batch_equivalence.py).
    _numpy_batch = isinstance(filter_manager, NumpyKalmanFilterManager)
    if _numpy_batch:
        posteriors = filter_manager.process_measurements_batch(event_df, config)

    for idx, row in (() if _numpy_batch else event_df.iterrows()):
        track_id = row['trackId']

        # Set environment for this track so q and vel_var lookups work
        env = row.get('identity.environment') or row.get('environment')
        if env and track_id not in filter_manager.environment_by_track:
            filter_manager.environment_by_track[track_id] = str(env)

        # Create noise covariance matrix from position covariance
        # these covariances come from reported uncertainty ellipses in the event data
        # and assumed error ellipses for velocity. In contrast, covariances in the heads
        # are assumed to be state uncertainty, not measurement uncertainty.
        if 'positionCovariance.xx' in row and pd.notna(row.get('positionCovariance.xx')):
            noise_pos_covar = np.array([
                [row.get('positionCovariance.xx', 1e7), row.get('positionCovariance.xy', 0), row.get('positionCovariance.xz', 0)],
                [row.get('positionCovariance.xy', 0), row.get('positionCovariance.yy', 1e7), row.get('positionCovariance.yz', 0)],
                [row.get('positionCovariance.xz', 0), row.get('positionCovariance.yz', 0), row.get('positionCovariance.zz', 1e7)]
            ])
        else:
            noise_pos_covar = np.diag([1e7, 1e7, 1e7])

        # Check if velocity values are present (not NaN)
        _vx = row.get('ecefVelocity.x', float('nan'))
        _vy = row.get('ecefVelocity.y', float('nan'))
        _vz = row.get('ecefVelocity.z', float('nan'))
        has_velocity_values = not (pd.isna(_vx) or pd.isna(_vy) or pd.isna(_vz))

        # Convert timestamp
        ts = row['interceptTimestamp']
        try:
            time = dt.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
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
                vel_var = filter_manager._get_vel_var(track_id)
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

            # Sanitize noise covariance: replace NaN and clip to reasonable bounds
            noise_covar = np.nan_to_num(noise_covar, nan=0.0)
            noise_covar = np.clip(noise_covar, -1e10, 1e10)
            for k in range(noise_covar.shape[0]):
                if noise_covar[k, k] <= 0 or noise_covar[k, k] > 1e10:
                    noise_covar[k, k] = 1e6

            measurement_model = LinearGaussian(
                ndim_state=6,
                mapping=(0, 1, 2, 3, 4, 5),
                noise_covar=noise_covar
            )
            measured_state_vector = np.array([
                row['ecefPosition.x'],
                _vx,
                row['ecefPosition.y'],
                _vy,
                row['ecefPosition.z'],
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
                row['ecefPosition.x'],
                row['ecefPosition.y'],
                row['ecefPosition.z']
            ])

        measurement = Detection(
            measured_state_vector,
            timestamp=time,
            measurement_model=measurement_model
        )

        # Skip if measurement contains NaN values
        if np.isnan(measured_state_vector).any():
            logging.warning(f" [{config.get('origin_dataset', 'unknown')}]: Skipping measurement for track {track_id} - contains NaN values")
            continue

        # Update filter for this track
        post = filter_manager.predict_and_update(track_id, measurement)
        posteriors[idx] = post
    
    # Convert track states to DataFrame format
    track_df = pd.DataFrame()
    track_df['trackId'] = event_df['trackId']
    track_df['standardIdentity'] = event_df['identity.standard']
    track_df['environment'] = event_df['identity.environment']
    # entity_beta: carry the full identity.* fields so component track
    # events/heads become the store relating trackId <-> identity (one
    # assignment, not a per-column insert).
    _id_cols = [c for c in event_df.columns if c.startswith('identity.')]
    if _id_cols:
        track_df[_id_cols] = event_df[_id_cols]
    track_df['trackOriginatedTimestamp'] = event_df['crucibleHeader.createdDate']
    if 'trackQuality' in event_df.columns:
        track_df['trackQuality'] = event_df['trackQuality']
    track_df['interceptTimestamp'] = event_df['interceptTimestamp']
    track_df['edhControlSet'] = event_df['edhControlSet']
    track_df['mode'] = event_df['mode']

    
    # Add filtered positions and velocities from per-row posteriors.
    # Each row gets its own posterior (not just the last one per entity), so
    # multi-measurement batches produce distinct filtered states. Rows with no
    # posterior (NaN measurements skipped above) are dropped. Vectorized: the
    # kept posteriors are stacked once into (n, 6) state / (n, 6, 6) covariance
    # arrays and each output column is assigned in bulk, replacing a per-row
    # .at[] loop over ~27 columns.
    kept_idx = [i for i in track_df.index if i in posteriors]
    track_df = track_df.loc[kept_idx].reset_index(drop=True)

    if kept_idx:
        sv = np.stack([np.asarray(posteriors[i].state_vector, dtype=float).reshape(-1)
                       for i in kept_idx])          # (n, 6): [x, vx, y, vy, z, vz]
        cv = np.stack([np.asarray(posteriors[i].covar, dtype=float)
                       for i in kept_idx])          # (n, 6, 6)

        track_df['ecefPosition.x'] = sv[:, 0]
        track_df['ecefPosition.y'] = sv[:, 2]
        track_df['ecefPosition.z'] = sv[:, 4]
        track_df['ecefVelocity.x'] = sv[:, 1]
        track_df['ecefVelocity.y'] = sv[:, 3]
        track_df['ecefVelocity.z'] = sv[:, 5]

        # Position covariance
        track_df['positionCovariance.xx'] = cv[:, 0, 0]
        track_df['positionCovariance.xy'] = cv[:, 0, 2]
        track_df['positionCovariance.xz'] = cv[:, 0, 4]
        track_df['positionCovariance.yy'] = cv[:, 2, 2]
        track_df['positionCovariance.yz'] = cv[:, 2, 4]
        track_df['positionCovariance.zz'] = cv[:, 4, 4]

        # Velocity covariance
        track_df['velocityCovariance.dxdx'] = cv[:, 1, 1]
        track_df['velocityCovariance.dxdy'] = cv[:, 1, 3]
        track_df['velocityCovariance.dxdz'] = cv[:, 1, 5]
        track_df['velocityCovariance.dydy'] = cv[:, 3, 3]
        track_df['velocityCovariance.dydz'] = cv[:, 3, 5]
        track_df['velocityCovariance.dzdz'] = cv[:, 5, 5]

        # Position-velocity cross covariance
        track_df['positionVelocityCovariance.xdx'] = cv[:, 0, 1]
        track_df['positionVelocityCovariance.xdy'] = cv[:, 0, 3]
        track_df['positionVelocityCovariance.xdz'] = cv[:, 0, 5]
        track_df['positionVelocityCovariance.ydx'] = cv[:, 2, 1]
        track_df['positionVelocityCovariance.ydy'] = cv[:, 2, 3]
        track_df['positionVelocityCovariance.ydz'] = cv[:, 2, 5]
        track_df['positionVelocityCovariance.zdx'] = cv[:, 4, 1]
        track_df['positionVelocityCovariance.zdy'] = cv[:, 4, 3]
        track_df['positionVelocityCovariance.zdz'] = cv[:, 4, 5]

    return track_df

def get_current_timestamp_string_with_offset(offset_hours: int = 0) -> str:
    """
    Gets the current time in UTC with a format like the following:
       2021-03-04T15:00:00.000Z

    Note: put this into utils when possible
    Returns:
         str
    """
    # Get the current time in UTC and format it in the way dictated by the schema
    datetime_format_string = "%Y-%m-%dT%H:%M:%S.%f"
    time_with_offset = dt.now(tz=tz.utc) + timedelta(hours=offset_hours)
    time_string = time_with_offset.strftime(datetime_format_string)[:-3] + 'Z'
    return time_string

def copy_kinematics_to_track(event_df: pd.DataFrame, config: dict, filter_manager: Optional[KalmanFilterManager] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Copy kinematic data directly without filtering (passthrough mode).
    
    Args:
        event_df: DataFrame containing kinematic data
        config: Configuration dictionary
        filter_manager: Optional KalmanFilterManager instance (unused; kept for signature parity)
    Returns:
        tuple: (track_event_json, track_head_json)
    """



    # first copy the event_df info to track events
    track_df = pd.DataFrame()
    track_df['trackId'] = event_df['trackId']
    track_df['standardIdentity'] = event_df['identity.standard']
    track_df['environment'] = event_df['identity.environment']
    # entity_beta: carry the full identity.* fields so component track
    # events/heads become the store relating trackId <-> identity (one
    # assignment, not a per-column insert).
    _id_cols = [c for c in event_df.columns if c.startswith('identity.')]
    if _id_cols:
        track_df[_id_cols] = event_df[_id_cols]
    track_df['trackOriginatedTimestamp'] = event_df['crucibleHeader.createdDate']
    if 'trackQuality' in event_df.columns:
        track_df['trackQuality'] = event_df['trackQuality']
    track_df['interceptTimestamp'] = event_df['interceptTimestamp']
    track_df['edhControlSet'] = event_df['edhControlSet']
    track_df['mode'] = event_df['mode']
    
    # Copy all kinematic data in ONE assignment (avoids fragmenting track_df
    # with a separate column insert per matched column).
    _kin_prefixes = ("geodetic.", "ecefPosition.", "ecefVelocity.",
                     "positionCovariance.", "velocityCovariance.",
                     "positionVelocityCovariance.", "uncertainty")
    _kin_cols = [c for c in event_df.columns if c.startswith(_kin_prefixes)]
    if _kin_cols:
        track_df[_kin_cols] = event_df[_kin_cols]
    if 'speed' in event_df.columns:
        track_df['speed'] = event_df['speed']
    if 'heading' in event_df.columns:
        track_df['heading'] = event_df['heading']
    
    # now copy the event_df info to track heads
    track_head_df = track_df.copy()
    track_head_df['trackUpdatedTimestamp'] = event_df['interceptTimestamp']
    if 'trackQuality' in track_head_df.columns:
        track_head_df.drop(columns=['trackQuality'], inplace=True)
    if 'interceptTimestamp' in track_head_df.columns:
        track_head_df.drop(columns=['interceptTimestamp'], inplace=True)
    track_head_df['stale'] = get_current_timestamp_string_with_offset(10)
    if 'speed' not in track_head_df.columns:
        track_head_df['speed'] = 0.0
    if 'heading' not in track_head_df.columns:
        track_head_df['heading'] = 0.0

    return track_df, track_head_df

def initialize_controllers() -> None:
    """Initialize the read/write controllers and authenticator.
    
    Call this function before using rc, wc, or auth.
    """
    global rc, wc, auth
    if rc is None or wc is None or auth is None:
        auth, rc, wc = instantiate_api_controllers()

def run(stream_manager_perspective: str) -> None:
    """
    Main function to run the tracker.
    
    Args:
        stream_manager_perspective: Perspective name for Stream Manager
    """
    
    # Register the signal handler
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)

    # Initialize controllers before running
    initialize_controllers()

    result = find_and_validate_configs(stream_manager_perspective, include_scripts=False, rc_instance=rc)
    config_list = result['datafeed_configs']
    perspective_config = result['perspective_config']

    # Merge perspective config (dataset names) into each datafeed config
    for config in config_list:
        for key, value in perspective_config.items():
            if key not in config:
                config[key] = value

    # entity_beta no longer assigns an entityId to tracks; the dispatcher derives a
    # deterministic trackId from each event's identity fields at dispatch time.

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



 



