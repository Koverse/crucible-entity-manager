'''
Object Duplicate Identifier for SUPERSEDE Actions

Advanced duplicate detection system for Objects (v1 schema) that:
1. Reads position history from component track events (not just track heads)
2. Identifies duplicate Objects by finding tracks with similar kinematics over time
3. Smooths trajectories with an RTS Kalman smoother for interpolation/extrapolation
4. Aligns staggered measurements onto a common time grid using the smoother's
   own state estimates (forward/backward Kalman prediction + fusion)
5. Compares both positions AND velocities using Mahalanobis distance with propagated covariances
6. Creates SUPERSEDE actions for object_manager.py to merge duplicates

Usage:
    python object_duplicate_identifier.py Live_POV --log=DEBUG
'''

import os
import sys
import time
import argparse
import logging
import traceback
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional, Set, DefaultDict, Any
from datetime import datetime as dt
from datetime import timedelta
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from cruciblelib import read_controller, write_controller, authenticator
except ImportError:
    from . import read_controller, write_controller, authenticator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from track_smoother import smooth_track_states

try:
    from object_utils import find_and_validate_configs, get_duplicate_protected_ids, get_supersede_map, instantiate_api_controllers
except ImportError:
    from .object_utils import find_and_validate_configs, get_duplicate_protected_ids, get_supersede_map, instantiate_api_controllers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Environment-dependent lookback time (hours).
# Air tracks are fast-moving so a shorter window suffices; surface/subsurface
# tracks are slower and benefit from a longer history.
LOOKBACK_HOURS_BY_ENVIRONMENT: Dict[str, float] = {
    'AIR':           0.25,
    'GROUND':        2.0,
    'SEA_SURFACE':   2.0,
    'SEA_SUBSURFACE': 2.0,
    'UNKNOWN':       0.5,
}
# Rows read per environment each cycle; queried separately so high-rate
# feeds cannot crowd slower environments out of their lookback.
TRACK_EVENT_LIMIT_PER_ENVIRONMENT = 50000


@dataclass
class DetectionParams:
    """Per-environment duplicate detection thresholds."""
    # Coarse candidate-search threshold. This only controls which track pairs
    # reach final evaluation; it does not by itself classify a duplicate.
    candidate_search_radius_m: float
    # Final duplicate-evaluation thresholds.
    distance_threshold_sigma: float      # Max Mahalanobis distance (sigma)
    velocity_threshold_mps: float        # Max mean velocity difference (m/s)
    time_alignment_seconds: float        # Max extrapolation beyond endpoints (s)
    eval_interval_seconds: float         # Evaluation grid spacing (s)
    min_matching_points: int             # Min distinct raw measurements per track
    min_confidence: float                # Final confidence floor
    # Absolute same-time separation floor (meters).  A pair whose median
    # same-time Euclidean separation exceeds this is never merged, regardless
    # of how large (covariance-inflated) the Mahalanobis test allows.  Guards
    # against merging closely-spaced-but-distinct tracks (convoys/followers).
    # None disables the floor.
    max_separation_m: Optional[float]    # Final median separation ceiling
    # Sensor position one-sigma (meters).  A fixed-interval smoother can drive
    # the *state* covariance far below the per-measurement noise, but the
    # residual between two independent noisy tracks of the same object is still
    # governed by the sensor one-sigma.  This floor is added to the combined
    # covariance in the Mahalanobis test so real sensor noise (e.g. 150 m AIS)
    # is not read as a many-sigma mismatch.  None/0 disables the floor.
    measurement_noise_floor_m: Optional[float]  # Final Mahalanobis noise floor
    # RESTORE when the recent median same-time separation exceeds this.
    restore_separation_m: float
    restore_window_seconds: float        # Recent span compared for RESTORE


DETECTION_PARAMS_BY_ENVIRONMENT: Dict[str, DetectionParams] = {
    # Airborne duplicates are common false positives: aircraft in trail,
    # formation flights, and holding patterns can sit close together while
    # remaining distinct.  Keep the gates tight — a small Mahalanobis sigma,
    # a narrow same-time separation floor, and a strict velocity match — so
    # only two feeds of the *same* aircraft are fused.
    'AIR': DetectionParams(
        candidate_search_radius_m=300.0,
        distance_threshold_sigma=2.0,
        velocity_threshold_mps=20.0,
        time_alignment_seconds=5.0,
        eval_interval_seconds=5.0,
        min_matching_points=5,
        min_confidence=0.6,
        max_separation_m=250.0,
        measurement_noise_floor_m=30.0,
        restore_separation_m=500.0,
        restore_window_seconds=120.0,
    ),
    # Surface/subsurface vessels are slow and report sparsely (AIS can be
    # minutes apart), and the two feeds for one vessel often do NOT overlap in
    # time.  The larger pre-filter radius (4 km) and extrapolation buffer (5 min)
    # let two staggered/gapped tracks still be bridged onto a common grid.
    # Trade-off: a wider radius + longer extrapolation makes the coarse pass and
    # the (covariance-inflated) Mahalanobis test more permissive, so distinct
    # vessels a few km apart during a long gap are more likely to be evaluated.
    # The max_separation_m floor caps how far apart two "merged" tracks can be.
    'SEA_SURFACE': DetectionParams(
        candidate_search_radius_m=4000.0,
        distance_threshold_sigma=3.0,
        velocity_threshold_mps=5.0,
        time_alignment_seconds=300.0,
        eval_interval_seconds=30.0,
        min_matching_points=5,
        min_confidence=0.5,
        max_separation_m=500.0,
        measurement_noise_floor_m=150.0,
        restore_separation_m=1500.0,
        restore_window_seconds=1800.0,
    ),
    'SEA_SUBSURFACE': DetectionParams(
        candidate_search_radius_m=4000.0,
        distance_threshold_sigma=3.0,
        velocity_threshold_mps=5.0,
        time_alignment_seconds=300.0,
        eval_interval_seconds=60.0,
        min_matching_points=5,
        min_confidence=0.5,
        max_separation_m=500.0,
        measurement_noise_floor_m=150.0,
        restore_separation_m=1500.0,
        restore_window_seconds=3600.0,
    ),
    # Ground vehicles legitimately travel very close together for long
    # stretches — buses share routes, queue at stops/lights, and follow each
    # other in traffic — so proximity alone is a poor duplicate signal here.
    # Keep the gates TIGHT (mirroring AIR) so only two feeds of the *same*
    # ground vehicle fuse: a small pre-filter radius and same-time separation
    # floor, a strict velocity match, a low sensor-noise floor (GPS ~15 m, not
    # the 150 m of AIS), more matching points, and higher confidence.
    'GROUND': DetectionParams(
        candidate_search_radius_m=150.0,
        distance_threshold_sigma=2.0,
        velocity_threshold_mps=5.0,
        time_alignment_seconds=5.0,
        eval_interval_seconds=10.0,
        min_matching_points=5,
        min_confidence=0.6,
        max_separation_m=30.0,
        measurement_noise_floor_m=15.0,
        restore_separation_m=150.0,
        restore_window_seconds=300.0,
    ),
    # Used for any environment not listed above.
    'UNKNOWN': DetectionParams(
        candidate_search_radius_m=500.0,
        distance_threshold_sigma=3.0,
        velocity_threshold_mps=10.0,
        time_alignment_seconds=5.0,
        eval_interval_seconds=10.0,
        min_matching_points=5,
        min_confidence=0.5,
        max_separation_m=500.0,
        measurement_noise_floor_m=None,
        restore_separation_m=1000.0,
        restore_window_seconds=600.0,
    ),
}


def detection_params_for(environment: Any) -> DetectionParams:
    return DETECTION_PARAMS_BY_ENVIRONMENT.get(
        str(environment or 'UNKNOWN').upper(), DETECTION_PARAMS_BY_ENVIRONMENT['UNKNOWN'])

# Seconds a restored ID is shielded from re-supersession and repeat RESTOREs.
DEFAULT_RESTORE_COOLDOWN_SECONDS = 300

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

# Global variables for controllers (lazy initialization)
rc = wc = auth = None

# Global tracking of processed SUPERSEDE actions
processed_supersedes: Set[str] = set()
active_supersedes: Dict[str, Any] = {}
_recent_restores: Dict[str, float] = {}


@dataclass
class DuplicateCandidate:
    """Represents a candidate duplicate pair with supporting evidence."""
    object_id_1: str
    object_id_2: str
    avg_position_distance: float
    avg_velocity_difference: float
    num_matching_points: int
    time_overlap_seconds: float
    confidence_score: float
    reason: str = ''

    def __repr__(self) -> str:
        return (f"DuplicateCandidate({self.object_id_1} <-> {self.object_id_2}, "
                f"mahal_dist={self.avg_position_distance:.2f}σ, "
                f"vel_diff={self.avg_velocity_difference:.1f}m/s, "
                f"points={self.num_matching_points}, "
                f"confidence={self.confidence_score:.2f})")


@dataclass
class TrackHistory:
    """Stores position and velocity history for an object."""
    object_id: str
    track_id: str
    timestamps: List[dt] = field(default_factory=list)
    positions_ecef: List[Tuple[float, float, float]] = field(default_factory=list)
    velocities_ecef: List[Tuple[float, float, float]] = field(default_factory=list)
    position_covariances: List[np.ndarray] = field(default_factory=list)
    environment: str = None
    # Full smoothed 6-state estimates (sorted by time), used to evaluate the
    # smoothed track at arbitrary query times via Kalman prediction/fusion.
    smoothed_epochs: Optional[np.ndarray] = None      # (N,) epoch seconds
    smoothed_means: Optional[np.ndarray] = None       # (N, 6) [x,y,z,vx,vy,vz]
    smoothed_covs: Optional[np.ndarray] = None        # (N, 6, 6)

    def add_point(self, timestamp: dt, position: Tuple[float, float, float],
                  velocity: Tuple[float, float, float],
                  position_covariance: np.ndarray = None) -> None:
        self.timestamps.append(timestamp)
        self.positions_ecef.append(position)
        self.velocities_ecef.append(velocity)
        self.position_covariances.append(position_covariance)

    def get_time_range(self) -> Tuple[Optional[dt], Optional[dt]]:
        if not self.timestamps:
            return None, None
        return min(self.timestamps), max(self.timestamps)

    def __len__(self) -> int:
        return len(self.timestamps)


# ---------------------------------------------------------------------------
# Distance / velocity helpers
# ---------------------------------------------------------------------------

def calculate_ecef_distance(pos1: Tuple[float, float, float],
                            pos2: Tuple[float, float, float]) -> float:
    try:
        return np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
    except (TypeError, ValueError):
        return float('inf')


def _horizontal_projection(pos1: Tuple[float, float, float],
                           pos2: Tuple[float, float, float]) -> np.ndarray:
    """Build a 3x2 matrix whose columns span the local horizontal plane.

    The radial (altitude) direction is approximated as the unit vector
    from the Earth's center to the midpoint of the two ECEF positions.
    Two orthonormal horizontal vectors are derived via cross products.
    """
    mid = (np.array(pos1, dtype=float) + np.array(pos2, dtype=float)) / 2.0
    r_norm = np.linalg.norm(mid)
    if r_norm < 1.0:  # degenerate (near origin)
        return np.eye(3)[:, :2]  # just use x, y
    up = mid / r_norm
    # Pick a seed vector that isn't parallel to up
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    e = np.cross(up, seed)
    e /= np.linalg.norm(e)
    n = np.cross(up, e)
    n /= np.linalg.norm(n)
    return np.column_stack([e, n])  # (3, 2)


def calculate_mahalanobis_distance(
    pos1: Tuple[float, float, float],
    pos2: Tuple[float, float, float],
    cov1: Optional[np.ndarray],
    cov2: Optional[np.ndarray],
    pos_noise_floor_m: float = 0.0,
) -> float:
    """Compute horizontal-plane Mahalanobis distance between two ECEF positions.

    Because altitude uncertainty is typically unavailable in report events,
    the distance is computed in the local horizontal (East-North) plane:

        H  = [e, n]          (3x2 horizontal basis at the midpoint)
        d  = sqrt( dh^T Ch^-1 dh )

    where dh = H^T delta  and  Ch = H^T (P1+P2) H  are the 2-D projections.

    *pos_noise_floor_m* is the sensor position one-sigma.  A fixed-interval
    smoother can drive the state covariance well below the per-measurement
    noise, so ``floor**2`` is added to the horizontal covariance for EACH
    track (``2*floor**2`` total).  This keeps genuine sensor noise (e.g. the
    150 m spread between two AIS feeds of one vessel) from reading as a
    many-sigma mismatch.

    Falls back to horizontal Euclidean distance / (sqrt(2)*floor) — or / 100 m
    when no floor is given — if covariances are missing or the projection is
    singular.
    """
    delta = np.array(pos1, dtype=float) - np.array(pos2, dtype=float)
    H = _horizontal_projection(pos1, pos2)
    dh = H.T @ delta  # (2,)

    floor_var = float(pos_noise_floor_m) ** 2
    fallback_scale = (np.sqrt(2.0) * pos_noise_floor_m) if floor_var > 0.0 else 100.0

    if cov1 is None or cov2 is None:
        return float(np.sqrt(dh @ dh)) / fallback_scale
    try:
        combined = cov1 + cov2
        Ch = H.T @ combined @ H  # (2, 2)
        if floor_var > 0.0:
            Ch = Ch + np.eye(2) * (2.0 * floor_var)
        return float(np.sqrt(dh @ np.linalg.solve(Ch, dh)))
    except np.linalg.LinAlgError:
        return float(np.sqrt(dh @ dh)) / fallback_scale


def calculate_velocity_difference(vel1: Tuple[float, float, float],
                                   vel2: Tuple[float, float, float]) -> float:
    try:
        return np.sqrt(sum((a - b)**2 for a, b in zip(vel1, vel2)))
    except (TypeError, ValueError):
        return float('inf')


def calculate_velocity_magnitude(vel: Tuple[float, float, float]) -> float:
    try:
        return np.sqrt(sum(v**2 for v in vel))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------

def _environment_filter(dataset: str, environment: str) -> str:
    column = f"{dataset}.`environment`"
    if environment != 'UNKNOWN':
        return f"{column} = '{environment}'"
    listed = ', '.join(f"'{env}'" for env in LOOKBACK_HOURS_BY_ENVIRONMENT if env != 'UNKNOWN')
    return f"({column} IS NULL OR {column} NOT IN ({listed}))"


def get_track_events(dataset_config: dict) -> pd.DataFrame:
    """Retrieve each environment's component track events over its own lookback."""
    track_event_dataset = dataset_config.get('component_track_event_dataset')
    if not track_event_dataset:
        logging.warning("No component_track_event_dataset configured")
        return pd.DataFrame()

    frames = []
    for environment, hours in LOOKBACK_HOURS_BY_ENVIRONMENT.items():
        query = f"""
        SELECT * FROM {track_event_dataset}
        WHERE {track_event_dataset}.`ecefPosition`.x IS NOT NULL
        AND {_environment_filter(track_event_dataset, environment)}
        AND {track_event_dataset}.crucibleHeader.createdDate > TIMESTAMP_OFFSET(-{int(hours * 3600)},'seconds')
        ORDER BY {track_event_dataset}.interceptTimestamp DESC
        LIMIT {TRACK_EVENT_LIMIT_PER_ENVIRONMENT}
        """
        try:
            rc.token = auth.get_token()
            result = rc.search(query, format='dataframe', auto_backtick=False)
        except Exception as e:
            logging.error(f"Error querying {environment} track events: {e}")
            logging.error(traceback.format_exc())
            continue
        if result is None or result.empty:
            continue
        if len(result) >= TRACK_EVENT_LIMIT_PER_ENVIRONMENT:
            logging.warning(f" {environment} track events hit the {TRACK_EVENT_LIMIT_PER_ENVIRONMENT}-row "
                            f"limit; effective lookback is shorter than {hours}h")
        frames.append(result)

    if not frames:
        logging.info(" No track events found")
        return pd.DataFrame()
    track_events = pd.concat(frames, ignore_index=True)
    logging.info(f" Found {len(track_events)} track events from {track_event_dataset}")
    return track_events


def get_track_heads(dataset_config: dict) -> pd.DataFrame:
    """Retrieve track heads for determining which object to keep."""
    try:
        track_head_dataset = dataset_config.get('component_track_head_dataset')
        if not track_head_dataset:
            logging.warning("No component_track_head_dataset configured")
            return pd.DataFrame()

        query = f"""
        SELECT * FROM {track_head_dataset}
        WHERE {track_head_dataset}.`ecefPosition`.x IS NOT NULL
        LIMIT 10000
        """

        logging.info(f" Querying track heads from {track_head_dataset}")
        rc.token = auth.get_token()
        result = rc.search(query, format='dataframe', auto_backtick=False)
        if not result.empty:
            logging.info(f" Found {len(result)} track heads")
        else:
            logging.info(" No track heads found")
        return result if not result.empty else pd.DataFrame()
    except Exception as e:
        logging.error(f"Error in get_track_heads: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()


def get_confirmed_object_ids(dataset_config: dict) -> Set[str]:
    """Return the set of objectId UUIDs whose entityStatus is CONFIRMED."""
    try:
        objects_dataset = dataset_config.get('object_dataset', 'Live_POV_Objects')
        query = f"""
        SELECT {objects_dataset}.objectId.uuid
        FROM {objects_dataset}
        WHERE {objects_dataset}.entityStatus = 'CONFIRMED'
        LIMIT 10000
        """
        rc.token = auth.get_token()
        result = rc.search(query, format='dataframe', auto_backtick=False)
        if result.empty:
            return set()
        uuid_col = 'uuid' if 'uuid' in result.columns else 'objectId.uuid'
        return set(result[uuid_col].dropna().astype(str).tolist())
    except Exception as e:
        logging.error(f"Error querying CONFIRMED objects: {e}")
        logging.error(traceback.format_exc())
        return set()


# ---------------------------------------------------------------------------
# Build track histories (with Kalman smoothing)
# ---------------------------------------------------------------------------

def build_track_histories(track_events_df: pd.DataFrame,
                          dataset_config: dict,
                          lookback_hours_by_env: Dict[str, float] = None,
                          exclude_protected: bool = True) -> Dict[str, TrackHistory]:
    """Build smoothed track histories from track events DataFrame.

    If *lookback_hours_by_env* is provided, each object's points are trimmed
    to the lookback window appropriate for its environment.
    """
    histories: Dict[str, TrackHistory] = {}
    if track_events_df.empty:
        return histories

    pos_cols = ['ecefPosition.x', 'ecefPosition.y', 'ecefPosition.z']
    # Object tracker writes velocity as .dx/.dy/.dz; fall back to .x/.y/.z
    vel_cols_dx = ['ecefVelocity.dx', 'ecefVelocity.dy', 'ecefVelocity.dz']
    vel_cols_x = ['ecefVelocity.x', 'ecefVelocity.y', 'ecefVelocity.z']

    missing_pos = [c for c in pos_cols if c not in track_events_df.columns]
    if missing_pos:
        logging.warning(f" Missing position columns: {missing_pos}")
        return histories

    if all(c in track_events_df.columns for c in vel_cols_dx):
        vel_cols = vel_cols_dx
    elif all(c in track_events_df.columns for c in vel_cols_x):
        vel_cols = vel_cols_x
    else:
        vel_cols = None
    has_velocity = vel_cols is not None

    timestamp_col = None
    for col in ['interceptTimestamp', 'trackOriginatedTimestamp', 'crucibleHeader.createdDate']:
        if col in track_events_df.columns:
            timestamp_col = col
            break
    if not timestamp_col:
        logging.warning(" No timestamp column found")
        return histories

    # Check for real position covariance columns from the transformer
    cov_cols = ['positionCovariance.xx', 'positionCovariance.xy',
                'positionCovariance.xz', 'positionCovariance.yy',
                'positionCovariance.yz', 'positionCovariance.zz']
    has_cov = all(c in track_events_df.columns for c in cov_cols)
    if has_cov:
        logging.info(" Using real position covariances from track events")

    # Collect raw points per object
    raw: Dict[str, dict] = {}  # object_id -> {track_id, ts, pos, vel, cov}
    for _, row in track_events_df.iterrows():
        object_id = row.get('objectId')
        if not object_id:
            continue
        try:
            timestamp = pd.to_datetime(row.get(timestamp_col))
        except Exception:
            continue
        try:
            position = (float(row['ecefPosition.x']),
                        float(row['ecefPosition.y']),
                        float(row['ecefPosition.z']))
            if all(p == 0 for p in position):
                continue
            if not all(np.isfinite(p) for p in position):
                continue
        except (TypeError, ValueError, KeyError):
            continue
        if has_velocity:
            try:
                velocity = (float(row.get(vel_cols[0], 0) or 0),
                            float(row.get(vel_cols[1], 0) or 0),
                            float(row.get(vel_cols[2], 0) or 0))
            except (TypeError, ValueError):
                velocity = (0.0, 0.0, 0.0)
        else:
            velocity = (0.0, 0.0, 0.0)

        # Reconstruct 3×3 position covariance if available
        meas_cov = None
        if has_cov:
            try:
                xx = float(row['positionCovariance.xx'])
                xy = float(row['positionCovariance.xy'])
                xz = float(row['positionCovariance.xz'])
                yy = float(row['positionCovariance.yy'])
                yz = float(row['positionCovariance.yz'])
                zz = float(row['positionCovariance.zz'])
                meas_cov = np.array([[xx, xy, xz],
                                     [xy, yy, yz],
                                     [xz, yz, zz]])
                if not np.all(np.isfinite(meas_cov)):
                    meas_cov = None
            except (TypeError, ValueError, KeyError):
                meas_cov = None

        if object_id not in raw:
            raw[object_id] = {
                'track_id': row.get('trackId', object_id),
                'env': row.get('environment'),
                'ts': [], 'pos': [], 'vel': [], 'cov': [],
            }
        raw[object_id]['ts'].append(timestamp)
        raw[object_id]['pos'].append(position)
        raw[object_id]['vel'].append(velocity)
        raw[object_id]['cov'].append(meas_cov)

    # Filter out already-superseded objects
    superseded_ids = raw.keys() & processed_supersedes if exclude_protected else set()
    if superseded_ids:
        for sid in list(superseded_ids):
            del raw[sid]
        logging.info(f" Filtered out {len(superseded_ids)} already-superseded/deleted objects")

    # Apply per-environment lookback filtering
    if lookback_hours_by_env:
        now = dt.utcnow()
        for object_id, data in list(raw.items()):
            env_val = data.get('env')
            env_key = (str(env_val).upper() if env_val is not None and not (isinstance(env_val, float) and pd.isna(env_val)) else 'UNKNOWN')
            env_lookback = lookback_hours_by_env.get(env_key, lookback_hours_by_env['UNKNOWN'])
            cutoff = now - timedelta(hours=env_lookback)
            keep = [i for i, ts in enumerate(data['ts'])
                    if ts.replace(tzinfo=None) >= cutoff]
            if not keep:
                del raw[object_id]
                continue
            data['ts'] = [data['ts'][i] for i in keep]
            data['pos'] = [data['pos'][i] for i in keep]
            data['vel'] = [data['vel'][i] for i in keep]
            data['cov'] = [data['cov'][i] for i in keep]

    # Smooth and build TrackHistory objects
    for object_id, data in raw.items():
        # Sort this object's measurements by time so the stored per-point
        # arrays and the smoothed state arrays share a consistent ordering.
        pt_epochs = np.array([_epoch(t) for t in data['ts']])
        pt_order = np.argsort(pt_epochs)
        ts_sorted = [data['ts'][i] for i in pt_order]
        pos_sorted = [data['pos'][i] for i in pt_order]
        vel_sorted = [data['vel'][i] for i in pt_order]
        cov_sorted = [data['cov'][i] for i in pt_order]

        vels = vel_sorted if has_velocity else None
        # Build per-measurement covariance array if all points have real covs
        meas_covs = None
        if all(c is not None for c in cov_sorted):
            meas_covs = np.stack(cov_sorted)
        try:
            s_epochs, s_means, s_covs = smooth_track_states(
                ts_sorted, pos_sorted, vels,
                measurement_covariances=meas_covs)
            s_pos = s_means[:, 0:3]
            s_vel = s_means[:, 3:6]
            s_poscov = s_covs[:, 0:3, 0:3]
        except Exception:
            s_pos = np.array(pos_sorted)
            s_vel = (np.array(vel_sorted) if has_velocity
                     else np.zeros((len(pos_sorted), 3)))
            s_poscov = np.stack([np.eye(3) * 100.0**2] * len(pos_sorted))
            s_epochs = np.array([_epoch(t) for t in ts_sorted])
            s_means = np.hstack([s_pos, s_vel])
            s_covs = np.stack([np.eye(6) * 100.0**2] * len(pos_sorted))

        history = TrackHistory(object_id=object_id, track_id=data['track_id'],
                               environment=data.get('env'))
        for i in range(len(ts_sorted)):
            history.add_point(ts_sorted[i],
                              tuple(s_pos[i]),
                              tuple(s_vel[i]),
                              s_poscov[i])
        history.smoothed_epochs = s_epochs
        history.smoothed_means = s_means
        history.smoothed_covs = s_covs
        histories[object_id] = history

    logging.info(f" Built smoothed track histories for {len(histories)} objects")
    return histories


# ---------------------------------------------------------------------------
# Kinematic interpolation helpers
# ---------------------------------------------------------------------------

def _epoch(t: Any) -> float:
    """Convert a datetime to epoch seconds."""
    if hasattr(t, 'timestamp') and callable(t.timestamp):
        return t.timestamp()
    return (t.replace(tzinfo=None) - dt(1970, 1, 1)).total_seconds()


def _cv_transition(dt_sec: float) -> np.ndarray:
    """Constant-velocity 6x6 state transition for time step *dt_sec*."""
    F = np.eye(6)
    F[0, 3] = dt_sec
    F[1, 4] = dt_sec
    F[2, 5] = dt_sec
    return F


def _cv_process_noise(dt_sec: float, q: float) -> np.ndarray:
    """6x6 process-noise matrix for the constant-velocity model (|dt| used)."""
    dt_sec = abs(dt_sec)
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = np.eye(3) * (dt_sec**3 / 3.0) * q
    Q[0:3, 3:6] = np.eye(3) * (dt_sec**2 / 2.0) * q
    Q[3:6, 0:3] = np.eye(3) * (dt_sec**2 / 2.0) * q
    Q[3:6, 3:6] = np.eye(3) * dt_sec * q
    return Q


def _predict_state(mean: np.ndarray, cov: np.ndarray, dt_sec: float,
                   q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Kalman-predict a 6-state estimate forward/backward by *dt_sec* seconds."""
    F = _cv_transition(dt_sec)
    pred_mean = F @ mean
    pred_cov = F @ cov @ F.T + _cv_process_noise(dt_sec, q)
    return pred_mean, pred_cov


def interpolate_track_at_times(history: TrackHistory,
                               eval_times: List[float],
                               process_noise_q: float = 1.0,
                               max_extrapolate_seconds: float = 30.0,
                               ) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Evaluate a smoothed track at arbitrary times using the RTS smoother output.

    The track's full 6-state RTS-smoothed estimates (``history.smoothed_*``)
    are used as anchors, and the state at each query time is obtained by
    Kalman prediction from the smoothed states:

    * **Boundary (extrapolation)** — for query times before the first / after
      the last smoothed state, predict from that single boundary state.  This
      is exactly what a fixed-interval smoother reduces to outside the
      measurement span.
    * **Interior (interpolation)** — for query times between two smoothed
      states, forward-predict from the left state *and* backward-predict from
      the right state, then fuse the two estimates by inverse-covariance
      weighting.  This is bidirectional (uses information from both sides) and
      covariance-consistent, unlike a one-sided constant-velocity extrapolation.

    Args:
        history: TrackHistory populated with smoothed_epochs/means/covs.
        eval_times: List of epoch-seconds at which to evaluate.
        process_noise_q: Process-noise spectral density used for the
                         prediction covariance growth.
        max_extrapolate_seconds: Maximum extrapolation beyond the smoothed
                                 track endpoints; query times outside that
                                 buffer return None.

    Returns:
        List (same length as eval_times) of either:
            (position_3, velocity_3, covariance_3x3)  — evaluated state
            None  — if eval_time is outside the track + max_extrapolate buffer
    """
    epochs = history.smoothed_epochs
    means = history.smoothed_means
    covs = history.smoothed_covs
    if epochs is None or means is None or covs is None or len(epochs) == 0:
        # Fallback: derive anchor states from the stored per-point arrays
        # (e.g. histories built without the smoother precomputed).  The
        # measurement-time positions/velocities/covariances are used directly
        # as the 6-state anchors.
        if not history.timestamps:
            return [None] * len(eval_times)
        ep = np.array([_epoch(t) for t in history.timestamps])
        order = np.argsort(ep)
        epochs = ep[order]
        pos = np.array(history.positions_ecef, dtype=float)[order]
        vel = (np.array(history.velocities_ecef, dtype=float)[order]
               if history.velocities_ecef else np.zeros_like(pos))
        means = np.hstack([pos, vel])
        covs = np.empty((len(epochs), 6, 6))
        for k, i in enumerate(order):
            C = np.eye(6) * 100.0**2
            if (i < len(history.position_covariances)
                    and history.position_covariances[i] is not None):
                C[0:3, 0:3] = history.position_covariances[i]
            covs[k] = C

    t_min = epochs[0] - max_extrapolate_seconds
    t_max = epochs[-1] + max_extrapolate_seconds

    results: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for t_eval in eval_times:
        if t_eval < t_min or t_eval > t_max:
            results.append(None)
            continue

        if len(epochs) == 1 or t_eval <= epochs[0]:
            m, P = _predict_state(means[0], covs[0], t_eval - epochs[0],
                                  process_noise_q)
        elif t_eval >= epochs[-1]:
            m, P = _predict_state(means[-1], covs[-1], t_eval - epochs[-1],
                                  process_noise_q)
        else:
            idx = int(np.clip(np.searchsorted(epochs, t_eval, side='right') - 1,
                              0, len(epochs) - 2))
            mL, PL = _predict_state(means[idx], covs[idx],
                                    t_eval - epochs[idx], process_noise_q)
            mR, PR = _predict_state(means[idx + 1], covs[idx + 1],
                                    t_eval - epochs[idx + 1], process_noise_q)
            try:
                iL = np.linalg.inv(PL)
                iR = np.linalg.inv(PR)
                P = np.linalg.inv(iL + iR)
                m = P @ (iL @ mL + iR @ mR)
            except np.linalg.LinAlgError:
                m, P = mL, PL

        results.append((m[0:3], m[3:6], P[0:3, 0:3]))

    return results


def _compute_evaluation_grid(history1: TrackHistory,
                             history2: TrackHistory,
                             eval_interval_seconds: float = 2.0,
                             max_extrapolate_seconds: float = 30.0,
                             ) -> Optional[Tuple[List[float], float]]:
    """Compute a common evaluation time grid over the overlap region.

    Returns (eval_times, overlap_seconds) or None if no overlap.
    """
    t1_min, t1_max = history1.get_time_range()
    t2_min, t2_max = history2.get_time_range()
    if t1_min is None or t2_min is None:
        return None

    # Allow some extrapolation into the gap between tracks
    t1_min_ext = _epoch(t1_min) - max_extrapolate_seconds
    t1_max_ext = _epoch(t1_max) + max_extrapolate_seconds
    t2_min_ext = _epoch(t2_min) - max_extrapolate_seconds
    t2_max_ext = _epoch(t2_max) + max_extrapolate_seconds

    overlap_start = max(t1_min_ext, t2_min_ext)
    overlap_end = min(t1_max_ext, t2_max_ext)

    if overlap_end <= overlap_start:
        return None

    overlap_seconds = overlap_end - overlap_start
    n_points = max(2, int(overlap_seconds / eval_interval_seconds) + 1)
    # Cap grid size to avoid excessive computation
    n_points = min(n_points, 500)
    eval_times = np.linspace(overlap_start, overlap_end, n_points).tolist()
    return eval_times, overlap_seconds


# ---------------------------------------------------------------------------
# Kinematic duplicate detection (cKDTree + interpolation-based evaluation)
# ---------------------------------------------------------------------------

def find_time_aligned_pairs(history1: TrackHistory,
                            history2: TrackHistory,
                            max_time_diff_seconds: float = 5.0) -> List[Tuple[int, int]]:
    """Legacy: find pairs of indices with matching timestamps.

    Kept for backward compatibility but no longer used in the main
    evaluation path (which uses interpolation instead).
    """
    if not history1.timestamps or not history2.timestamps:
        return []
    times1 = np.array([_epoch(t) for t in history1.timestamps])
    times2 = np.array([_epoch(t) for t in history2.timestamps])
    time_diffs = np.abs(times1[:, np.newaxis] - times2[np.newaxis, :])
    matches = np.where(time_diffs <= max_time_diff_seconds)
    return list(zip(matches[0].tolist(), matches[1].tolist()))


def evaluate_duplicate_candidate(history1: TrackHistory,
                                  history2: TrackHistory,
                                  final_mahalanobis_threshold_sigma: float,
                                  velocity_threshold_mps: float,
                                  min_matching_points: int,
                                  time_alignment_seconds: float,
                                  eval_interval_seconds: float = 2.0,
                                  process_noise_q: float = 1.0,
                                  max_separation_m: Optional[float] = None,
                                  measurement_noise_floor_m: Optional[float] = None,
                                  ) -> Optional[DuplicateCandidate]:
    """Evaluate whether two tracks are duplicates using interpolation.

    Instead of requiring measurements at nearly the same timestamps,
    both tracks are interpolated/extrapolated to a common evaluation
    grid using constant-velocity prediction from the smoothed states.
    This allows comparison of staggered measurements from different
    sensors.

    *final_mahalanobis_threshold_sigma* is a final duplicate-classification
    threshold in Mahalanobis sigma units (e.g. 3.0); it is unrelated to the
    cKDTree candidate search radius.
    *time_alignment_seconds* controls the maximum extrapolation beyond
    each track's last point.
    """
    if history1.object_id == history2.object_id:
        return None

    pair = f"{history1.object_id} <-> {history2.object_id}"

    # Skip if environments don't match.  Two known, different environments
    # are never duplicates.  A known env paired with an unknown/missing env
    # is also rejected — better to miss a duplicate than merge a ship with
    # a plane.
    env1 = (history1.environment or '').upper()
    env2 = (history2.environment or '').upper()
    if env1 != env2:
        logging.info(f" REJECT {pair}: environment mismatch "
                      f"({env1 or 'NONE'} vs {env2 or 'NONE'})")
        return None

    # Compute common evaluation grid over the time overlap
    max_extrapolate = time_alignment_seconds
    grid_result = _compute_evaluation_grid(
        history1, history2,
        eval_interval_seconds=eval_interval_seconds,
        max_extrapolate_seconds=max_extrapolate)
    if grid_result is None:
        logging.info(f" REJECT {pair}: no time overlap within "
                      f"{time_alignment_seconds:.0f}s extrapolation buffer")
        return None
    eval_times, overlap_seconds = grid_result

    measurements1 = len({_epoch(timestamp) for timestamp in history1.timestamps})
    measurements2 = len({_epoch(timestamp) for timestamp in history2.timestamps})
    independent_points = min(measurements1, measurements2)
    if independent_points < min_matching_points:
        logging.info(
            f" REJECT {pair}: only {independent_points} distinct raw measurements "
            f"per limiting track (< min_matching_points={min_matching_points})")
        return None

    # Interpolate both tracks to the evaluation grid
    states1 = interpolate_track_at_times(
        history1, eval_times,
        process_noise_q=process_noise_q,
        max_extrapolate_seconds=max_extrapolate)
    states2 = interpolate_track_at_times(
        history2, eval_times,
        process_noise_q=process_noise_q,
        max_extrapolate_seconds=max_extrapolate)

    # Compare at each evaluation point where both tracks have valid states
    position_distances = []
    velocity_differences = []
    velocity_deltas = []   # per-point (v1 - v2) vectors
    position_separations = []
    noise_floor = float(measurement_noise_floor_m or 0.0)
    for s1, s2 in zip(states1, states2):
        if s1 is None or s2 is None:
            continue
        pos1, vel1, cov1 = s1
        pos2, vel2, cov2 = s2
        position_distances.append(
            calculate_mahalanobis_distance(tuple(pos1), tuple(pos2), cov1, cov2,
                                           pos_noise_floor_m=noise_floor))
        position_separations.append(
            calculate_ecef_distance(tuple(pos1), tuple(pos2)))
        velocity_deltas.append(np.asarray(vel1, dtype=float) - np.asarray(vel2, dtype=float))
        velocity_differences.append(
            calculate_velocity_difference(tuple(vel1), tuple(vel2)))

    if len(position_distances) < min_matching_points:
        logging.info(f" REJECT {pair}: only {len(position_distances)} overlapping "
                      f"eval points (< min_matching_points={min_matching_points})")
        return None

    # Absolute separation floor: regardless of how inflated the covariance is
    # (e.g. from long-gap extrapolation), two tracks whose typical same-time
    # separation exceeds *max_separation_m* are not treated as duplicates.
    # This guards against merging closely-spaced-but-distinct tracks (convoys,
    # trailing vessels) that the Mahalanobis test alone might accept.
    median_separation = float(np.median(position_separations))
    if max_separation_m is not None and median_separation > max_separation_m:
        logging.info(f" REJECT {pair}: median same-time separation "
                      f"{median_separation:.0f}m > max_separation_m={max_separation_m:.0f}m")
        return None

    avg_pos_dist = np.mean(position_distances)
    # Velocity comparison uses the mean velocity *vector* difference, not the
    # mean of per-point magnitudes.  Two independent noisy estimates of the same
    # true velocity give a large, positively-biased mean |v1-v2| (a folded
    # magnitude), whereas their vector differences are zero-mean — so averaging
    # the vectors first cancels the estimation noise and leaves only a genuine
    # systematic velocity difference.
    mean_velocity_delta = np.mean(np.stack(velocity_deltas), axis=0)
    avg_vel_diff = float(np.linalg.norm(mean_velocity_delta))

    if avg_pos_dist > final_mahalanobis_threshold_sigma:
        logging.info(f" REJECT {pair}: Mahalanobis {avg_pos_dist:.2f}σ > "
                      f"threshold {final_mahalanobis_threshold_sigma:.2f}σ "
                      f"(median sep {median_separation:.0f}m, pts={len(position_distances)})")
        return None

    has_meaningful_velocity = any(
        calculate_velocity_magnitude(v) > 0.1
        for v in history1.velocities_ecef + history2.velocities_ecef)
    if has_meaningful_velocity and avg_vel_diff > velocity_threshold_mps:
        logging.info(f" REJECT {pair}: velocity diff {avg_vel_diff:.1f}m/s > "
                      f"threshold {velocity_threshold_mps:.1f}m/s")
        return None

    pos_std = np.std(position_distances) if len(position_distances) > 1 else 0
    vel_std = np.std(velocity_differences) if len(velocity_differences) > 1 else 0
    # Evidence factor: count REAL measurements (not interpolated grid samples)
    # that fall within the evaluated window, using the sparser/limiting track.
    # Interpolated points over a short overlap are highly correlated and must
    # not inflate confidence; only genuine measurements add independent looks.
    num_points_factor = min(1.0, independent_points / 10.0)
    pos_consistency_factor = 1.0 / (1.0 + pos_std / final_mahalanobis_threshold_sigma)
    vel_consistency_factor = (1.0 / (1.0 + vel_std / max(velocity_threshold_mps, 1.0))
                              if has_meaningful_velocity else 1.0)
    confidence = (num_points_factor * 0.4 +
                  pos_consistency_factor * 0.4 +
                  vel_consistency_factor * 0.2)

    t1_min, t1_max = history1.get_time_range()
    t2_min, t2_max = history2.get_time_range()
    if t1_min and t2_min:
        overlap_start = max(_epoch(t1_min), _epoch(t2_min))
        overlap_end = min(_epoch(t1_max), _epoch(t2_max))
        time_overlap = max(0.0, overlap_end - overlap_start)
    else:
        time_overlap = 0

    return DuplicateCandidate(
        object_id_1=history1.object_id,
        object_id_2=history2.object_id,
        avg_position_distance=avg_pos_dist,
        avg_velocity_difference=avg_vel_diff,
        num_matching_points=independent_points,
        time_overlap_seconds=time_overlap,
        confidence_score=confidence,
        reason=f"kinematic: mahal={avg_pos_dist:.2f}σ sep={median_separation:.0f}m vel={avg_vel_diff:.1f}m/s pts={independent_points}"
    )


def find_candidate_pairs_kdtree(histories: Dict[str, TrackHistory],
                                 candidate_search_radius_m: float,
                                 time_alignment_seconds: float,
                                 min_matching_points: int) -> Set[Tuple[str, str]]:
    """Use cKDTree (spatial-only) to find object pairs with nearby positions.

    Since tracks from different sensors may have staggered timestamps,
    the pre-filter uses a 3-D spatial tree (no time dimension).  A pair
    is a candidate if:
        1. At least *min_matching_points* positions from different objects
            are within *candidate_search_radius_m* of each other (spatially).
      2. The two tracks have overlapping time coverage (with a buffer of
         *time_alignment_seconds* for extrapolation).
    """
    # Build environment and time-range lookup
    env_by_object: Dict[str, str] = {}
    time_range_by_object: Dict[str, Tuple[float, float]] = {}
    for object_id, history in histories.items():
        env_by_object[object_id] = (history.environment or '').upper()
        if history.timestamps:
            epochs = [_epoch(t) for t in history.timestamps]
            time_range_by_object[object_id] = (min(epochs), max(epochs))

    all_points = []
    all_object_ids = []
    all_timestamps = []

    for object_id, history in histories.items():
        for timestamp, pos in zip(history.timestamps, history.positions_ecef):
            all_points.append((pos[0], pos[1], pos[2]))
            all_object_ids.append(object_id)
            all_timestamps.append(_epoch(timestamp))

    if len(all_points) < 2:
        return set()

    points_array = np.array(all_points)
    finite_mask = np.isfinite(points_array).all(axis=1)
    if not finite_mask.all():
        n_bad = (~finite_mask).sum()
        logging.warning(f" Dropping {n_bad} non-finite points before building cKDTree")
        good_idx = np.where(finite_mask)[0]
        points_array = points_array[good_idx]
        all_object_ids = [all_object_ids[i] for i in good_idx]
        all_timestamps = [all_timestamps[i] for i in good_idx]
        if len(points_array) < 2:
            return set()

    tree = cKDTree(points_array)

    pair_measurements: DefaultDict[Tuple[str, str], Tuple[Set[float], Set[float]]] = defaultdict(
        lambda: (set(), set()))
    pairs = tree.query_pairs(r=candidate_search_radius_m, output_type='ndarray')

    for idx1, idx2 in pairs:
        obj1 = all_object_ids[idx1]
        obj2 = all_object_ids[idx2]
        if obj1 == obj2:
            continue
        # Skip cross-environment pairs early (AIR vs SEA_SURFACE etc.)
        if env_by_object[obj1] != env_by_object[obj2]:
            continue
        pair_key = tuple(sorted([obj1, obj2]))
        timestamps1, timestamps2 = pair_measurements[pair_key]
        if obj1 == pair_key[0]:
            timestamps1.add(all_timestamps[idx1])
            timestamps2.add(all_timestamps[idx2])
        else:
            timestamps1.add(all_timestamps[idx2])
            timestamps2.add(all_timestamps[idx1])

    # Filter by minimum spatial proximity count AND time overlap
    candidates = set()
    for pair, (timestamps1, timestamps2) in pair_measurements.items():
        if min(len(timestamps1), len(timestamps2)) < min_matching_points:
            continue
        obj_a, obj_b = pair
        # Check time overlap (with extrapolation buffer)
        tr_a = time_range_by_object.get(obj_a)
        tr_b = time_range_by_object.get(obj_b)
        if tr_a is None or tr_b is None:
            continue
        # Extend each track's range by the extrapolation buffer
        a_min, a_max = tr_a[0] - time_alignment_seconds, tr_a[1] + time_alignment_seconds
        b_min, b_max = tr_b[0] - time_alignment_seconds, tr_b[1] + time_alignment_seconds
        if a_max < b_min or b_max < a_min:
            continue  # No time overlap even with extrapolation
        candidates.add(pair)

    return candidates


def find_duplicate_tracks(track_events_df: pd.DataFrame,
                          track_heads_df: pd.DataFrame,
                          dataset_config: dict) -> List[DuplicateCandidate]:
    """Find duplicate tracks using smoothed position history and velocity comparison."""
    duplicates: List[DuplicateCandidate] = []
    try:
        histories = build_track_histories(track_events_df, dataset_config,
                                          lookback_hours_by_env=LOOKBACK_HOURS_BY_ENVIRONMENT)
        if len(histories) < 2:
            logging.info(" Not enough track histories for comparison")
            return duplicates

        n_objects = len(histories)
        total_points = sum(len(h) for h in histories.values())
        logging.info(f" Analyzing {n_objects} objects with {total_points} total track points")
        logging.info(f" Building spatial index (cKDTree)...")

        # Use the loosest thresholds across all environments for the pre-filter
        all_params = list(DETECTION_PARAMS_BY_ENVIRONMENT.values())
        max_candidate_search_radius_m = max(
            p.candidate_search_radius_m for p in all_params)
        max_time_align = max(p.time_alignment_seconds for p in all_params)
        min_pts = min(p.min_matching_points for p in all_params)

        candidate_pairs = find_candidate_pairs_kdtree(
            histories,
            candidate_search_radius_m=max_candidate_search_radius_m,
            time_alignment_seconds=max_time_align,
            min_matching_points=min_pts,
        )
        logging.info(f" Found {len(candidate_pairs)} candidate pairs "
                    f"(reduced from {n_objects * (n_objects - 1) // 2} total pairs)")

        for obj_id_1, obj_id_2 in candidate_pairs:
            params = detection_params_for(histories[obj_id_1].environment)

            candidate = evaluate_duplicate_candidate(
                histories[obj_id_1], histories[obj_id_2],
                final_mahalanobis_threshold_sigma=params.distance_threshold_sigma,
                velocity_threshold_mps=params.velocity_threshold_mps,
                min_matching_points=params.min_matching_points,
                time_alignment_seconds=params.time_alignment_seconds,
                eval_interval_seconds=params.eval_interval_seconds,
                max_separation_m=params.max_separation_m,
                measurement_noise_floor_m=params.measurement_noise_floor_m,
            )
            if candidate and candidate.confidence_score >= params.min_confidence:
                logging.info(f" Found duplicate candidate: {candidate}")
                duplicates.append(candidate)
            elif candidate is not None:
                logging.info(f" REJECT {candidate.object_id_1} <-> {candidate.object_id_2}: "
                              f"confidence {candidate.confidence_score:.2f} < "
                              f"min_confidence={params.min_confidence:.2f} ({candidate.reason})")

        duplicates.sort(key=lambda x: x.confidence_score, reverse=True)
        logging.info(f" Found {len(duplicates)} confirmed duplicates")
        return duplicates
    except Exception as e:
        logging.error(f" Error in find_duplicate_tracks: {e}")
        logging.error(traceback.format_exc())
        return duplicates


# ---------------------------------------------------------------------------
# Supersede action helpers
# ---------------------------------------------------------------------------

def determine_object_to_keep(candidate: DuplicateCandidate,
                              track_heads_df: pd.DataFrame,
                              confirmed_ids: Set[str] = None) -> Tuple[str, str]:
    """Determine (superseded_id, superseding_id). Keep the older object.

    Priority:
    1. CONFIRMED objects are always kept (never superseded).
    2. Keep the older object (by trackOriginatedTimestamp or createdDate).
    3. Fallback: lexicographic (deterministic).
    """
    obj1 = candidate.object_id_1
    obj2 = candidate.object_id_2

    # CONFIRMED objects must always be the superseding (kept) side
    if confirmed_ids:
        obj1_confirmed = obj1 in confirmed_ids
        obj2_confirmed = obj2 in confirmed_ids
        if obj1_confirmed and not obj2_confirmed:
            return (obj2, obj1)  # Keep obj1 (CONFIRMED)
        if obj2_confirmed and not obj1_confirmed:
            return (obj1, obj2)  # Keep obj2 (CONFIRMED)
        # If both are CONFIRMED, fall through to age-based logic;
        # caller will skip the pair.

    obj1_info = track_heads_df[track_heads_df['objectId'] == obj1] if not track_heads_df.empty else pd.DataFrame()
    obj2_info = track_heads_df[track_heads_df['objectId'] == obj2] if not track_heads_df.empty else pd.DataFrame()

    obj1_created = obj2_created = None
    for info, target in [(obj1_info, 'obj1'), (obj2_info, 'obj2')]:
        if not info.empty:
            for col in ['trackOriginatedTimestamp', 'crucibleHeader.createdDate']:
                if col in info.columns:
                    try:
                        val = pd.to_datetime(info.iloc[0][col])
                        if target == 'obj1':
                            obj1_created = val
                        else:
                            obj2_created = val
                        break
                    except Exception:
                        pass

    if obj1_created and obj2_created:
        if obj1_created < obj2_created:
            return (obj2, obj1)
        elif obj2_created < obj1_created:
            return (obj1, obj2)

    # Fallback: lexicographic
    if obj1 < obj2:
        return (obj2, obj1)
    return (obj1, obj2)


def create_supersede_action(superseded_id: str,
                            superseding_id: str,
                            candidate: DuplicateCandidate,
                            dataset_config: dict) -> bool:
    """Write a SUPERSEDE management event to Crucible.

    Hard rule: a CONFIRMED object may NEVER be superseded.
    """
    try:
        if superseded_id in processed_supersedes:
            logging.debug(f" Already processed: {superseded_id}")
            return False

        # Absolute safeguard: never supersede a CONFIRMED object
        confirmed_ids = get_confirmed_object_ids(dataset_config)
        if superseded_id in confirmed_ids:
            logging.warning(f" BLOCKED: cannot supersede CONFIRMED object {superseded_id}")
            return False

        reason = candidate.reason[:100] if candidate.reason else ''
        management_event = {
            "action": "SUPERSEDE",
            "objectId": superseded_id,
            "supersededBy": superseding_id,
            "source": "MANAGER",
            "edhControlSet": ["CLS:U"],
        }

        wc.token = auth.get_token()
        management_dataset = dataset_config.get('object_management_event_dataset',
                                                 'Live_POV_ObjectManagementEvents')
        logging.info(f" Writing SUPERSEDE to {management_dataset}: "
                    f"{superseded_id} -> {superseding_id} ({reason})")
        wc.write_record_batch_by_name(management_dataset, [management_event])
        # Track both sides to prevent circular re-detection within this cycle
        processed_supersedes.add(superseded_id)
        processed_supersedes.add(superseding_id)
        return True
    except Exception as e:
        logging.error(f" Error creating SUPERSEDE action: {e}")
        logging.error(traceback.format_exc())
        return False


def process_kinematic_duplicates(duplicates: List[DuplicateCandidate],
                                  track_heads_df: pd.DataFrame,
                                  dataset_config: dict) -> int:
    """Process kinematic duplicates — determine which to keep, then supersede."""
    confirmed_ids = get_confirmed_object_ids(dataset_config)
    actions = 0
    for candidate in duplicates:
        superseded_id, superseding_id = determine_object_to_keep(
            candidate, track_heads_df, confirmed_ids)
        if superseded_id in processed_supersedes:
            continue
        if superseded_id in confirmed_ids:
            # Both objects are CONFIRMED — cannot supersede either
            logging.info(f" Skipping pair: both {superseded_id} and {superseding_id} are CONFIRMED")
            continue
        if create_supersede_action(superseded_id, superseding_id, candidate, dataset_config):
            actions += 1
    if actions > 0:
        logging.info(f" Created {actions} kinematic SUPERSEDE actions")
    return actions


def find_divergent_supersedes(histories: Dict[str, TrackHistory],
                              supersede_map: Dict[str, Any],
                              min_points: int) -> List[str]:
    """Return superseded objectIds whose tracks recently separated from their survivor."""
    restores = []
    for superseded_id, survivor_id in supersede_map.items():
        history1 = histories.get(superseded_id)
        history2 = histories.get(survivor_id) if survivor_id else None
        if history1 is None or history2 is None:
            continue
        params = detection_params_for(history1.environment)
        grid = _compute_evaluation_grid(
            history1, history2, params.eval_interval_seconds,
            params.time_alignment_seconds)
        if grid is None:
            continue
        window_start = grid[0][-1] - params.restore_window_seconds
        if any(sum(_epoch(ts) >= window_start for ts in history.timestamps) < min_points
               for history in (history1, history2)):
            continue
        recent_times = [t for t in grid[0] if t >= window_start]
        states1 = interpolate_track_at_times(
            history1, recent_times, max_extrapolate_seconds=params.time_alignment_seconds)
        states2 = interpolate_track_at_times(
            history2, recent_times, max_extrapolate_seconds=params.time_alignment_seconds)
        separations = [
            calculate_ecef_distance(tuple(state1[0]), tuple(state2[0]))
            for state1, state2 in zip(states1, states2)
            if state1 is not None and state2 is not None]
        if (len(separations) >= min_points
                and float(np.median(separations)) > params.restore_separation_m):
            restores.append(superseded_id)
    return restores


def create_restore_action(object_id: str, dataset_config: dict) -> bool:
    """Write a RESTORE management event for a superseded object."""
    last_restore = _recent_restores.get(object_id)
    if (last_restore is not None
            and time.monotonic() - last_restore < DEFAULT_RESTORE_COOLDOWN_SECONDS):
        return False
    management_event = {
        "action": "RESTORE",
        "objectId": object_id,
        "source": "MANAGER",
        "edhControlSet": ["CLS:U"],
    }
    try:
        management_dataset = dataset_config.get('object_management_event_dataset',
                                                 'Live_POV_ObjectManagementEvents')
        wc.token = auth.get_token()
        wc.write_record_batch_by_name(management_dataset, [management_event])
    except Exception as e:
        logging.error(f" Error creating RESTORE action for {object_id}: {e}")
        return False
    _recent_restores[object_id] = time.monotonic()
    active_supersedes.pop(object_id, None)
    processed_supersedes.add(object_id)
    logging.info(f" Created RESTORE action: {object_id}")
    return True


def process_restores(track_events_df: pd.DataFrame, dataset_config: dict,
                     min_points: int) -> int:
    """Emit RESTORE events for active supersedes whose tracks have diverged."""
    pairs = {object_id: survivor for object_id, survivor in active_supersedes.items()
             if survivor}
    if not pairs or track_events_df.empty or 'objectId' not in track_events_df.columns:
        return 0
    pair_ids = set(pairs) | set(pairs.values())
    pair_events = track_events_df[track_events_df['objectId'].astype(str).isin(pair_ids)]
    histories = build_track_histories(
        pair_events, dataset_config,
        lookback_hours_by_env=LOOKBACK_HOURS_BY_ENVIRONMENT,
        exclude_protected=False)
    return sum(create_restore_action(object_id, dataset_config)
               for object_id in find_divergent_supersedes(histories, pairs, min_points))


def refresh_protected_ids(dataset_config: dict) -> None:
    """Refresh the duplicate detector's local protected-ID cache."""
    global processed_supersedes, active_supersedes
    try:
        supersede_map = get_supersede_map(
            dataset_config, rc_instance=rc, auth_instance=auth)
        processed_supersedes = get_duplicate_protected_ids(
            dataset_config, rc, auth, DEFAULT_RESTORE_COOLDOWN_SECONDS,
            supersede_map=supersede_map)
        active_supersedes = supersede_map
        logging.info(f"  Loaded {len(processed_supersedes)} protected object ID(s)")
    except Exception as e:
        logging.warning(f"  Could not load existing SUPERSEDE actions: {e}")
        logging.warning(traceback.format_exc())


def initialize_controllers() -> None:
    """Lazy-initialize the read/write controllers and authenticator."""
    global rc, wc, auth
    if rc is None or wc is None or auth is None:
        auth, rc, wc = instantiate_api_controllers()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(stream_manager_perspective: str,
        log_level: str = None,
        poll_interval: Optional[int] = None,
        enable_kinematic: bool = True,
        restore_min_points: Optional[int] = None) -> None:
    parser = build_arg_parser()
    if poll_interval is None:
        poll_interval = parser.get_default('poll_interval')
    if restore_min_points is None:
        restore_min_points = parser.get_default('restore_min_points')
    if log_level is None:
        log_level = 'info'
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s: %(levelname)s %(name)s %(module)s Func: %(funcName)s:%(lineno)d-%(message)s',
        force=True,
    )

    logging.info("=" * 60)
    logging.info("Starting Object Duplicate Identifier")
    logging.info("=" * 60)
    modes = []
    if enable_kinematic:
        modes.append('kinematic')
    logging.info(f"  Detection modes: {', '.join(modes) or 'NONE'}")
    logging.info(f"  Lookback hours: {LOOKBACK_HOURS_BY_ENVIRONMENT}")
    logging.info(f"  Restore min points: {restore_min_points}")
    logging.info(f"  Poll interval: {poll_interval}s")

    initialize_controllers()

    config_list = find_and_validate_configs(stream_manager_perspective,
                                            include_scripts=False, rc_instance=rc)
    if not config_list:
        logging.error(f"No configuration found for perspective: {stream_manager_perspective}")
        return

    # Find config with track datasets
    main_config = None
    for config in config_list:
        if config.get('disabled'):
            continue
        if (config.get('component_track_head_dataset') or
                config.get('component_track_event_dataset')):
            main_config = config
            break
    if main_config is None:
        for config in config_list:
            if not config.get('disabled'):
                main_config = config
                break
    if main_config is None:
        logging.error(f"No valid configuration found for: {stream_manager_perspective}")
        return

    management_dataset = main_config.get('object_management_event_dataset',
                                          'Live_POV_ObjectManagementEvents')
    logging.info(f" Writing SUPERSEDE actions to: {management_dataset}")

    refresh_protected_ids(main_config)

    while True:
        try:
            logging.info('-' * 50)
            # Refresh supersede map each iteration
            refresh_protected_ids(main_config)

            if enable_kinematic:
                logging.info(" Running kinematic duplicate detection...")
                track_events_df = get_track_events(main_config)
                track_heads_df = get_track_heads(main_config)
                if not track_events_df.empty:
                    duplicates = find_duplicate_tracks(
                        track_events_df=track_events_df,
                        track_heads_df=track_heads_df,
                        dataset_config=main_config,
                    )
                    if duplicates:
                        process_kinematic_duplicates(duplicates, track_heads_df, main_config)
                    else:
                        logging.info(" No kinematic duplicates found")
                    process_restores(track_events_df, main_config, restore_min_points)
                else:
                    logging.info(" No track events available for analysis")

            logging.info(f" Search completed. Waiting {poll_interval}s...")
            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logging.info("Shutting down...")
            break
        except Exception as err:
            logging.error('-' * 50)
            logging.error(f"Error from main loop: {err}")
            logging.error(traceback.format_exc())
            time.sleep(5)
            continue


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Object Duplicate Identifier for SUPERSEDE Actions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python object_duplicate_identifier.py Live_POV
    python object_duplicate_identifier.py Live_POV --log=DEBUG
        ''',
    )
    parser.add_argument("stream_manager_perspective",
                        help='Name of perspective in Stream Manager Configuration (e.g., Live_POV)')
    parser.add_argument("--log", default='INFO',
                        help='Logging level: INFO (default), WARN, ERROR, or DEBUG')
    parser.add_argument("--poll-interval", type=int, default=60,
                        help='Interval in seconds between searches (default: 60)')
    parser.add_argument("--restore-min-points", type=int, default=3,
                        help='Min points per track and aligned comparisons to RESTORE (default: 3)')
    # Detection mode flags
    parser.add_argument("--kinematic", action='store_true', default=True, dest='kinematic',
                        help='Enable kinematic duplicate detection (default: ON)')
    parser.add_argument("--no-kinematic", action='store_false', dest='kinematic',
                        help='Disable kinematic duplicate detection')

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    run(
        args.stream_manager_perspective,
        log_level=args.log,
        poll_interval=args.poll_interval,
        enable_kinematic=args.kinematic,
        restore_min_points=args.restore_min_points,
    )
