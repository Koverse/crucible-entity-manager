#!/usr/bin/env python
'''
Entity Duplicate Identifier for SUPERSEDE Actions

Advanced duplicate detection system for component tracks (v2 schema) that:
1. Reads position history from component_track_events (not just track heads)
2. Identifies duplicate component tracks by finding tracks with similar kinematics over time
3. Smooths trajectories with an RTS Kalman smoother before comparing
4. Compares both positions AND velocities for accurate duplicate detection
5. Creates SUPERSEDE actions (keyed by trackId) consumed by entity_track_fuser.py to fuse duplicates

Duplicate detection and SUPERSEDE actions operate on component trackIds.
Management events record the superseded track in ``trackId`` and the surviving
track in ``supersededBy``.

Usage:
    python entity_duplicate_identifier.py Live --log=DEBUG
'''


import os
import time
import sys
import argparse
import logging
import traceback
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional, Set, DefaultDict, Any
from datetime import datetime as dt
from datetime import timedelta, timezone
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from track_smoother import smooth_track

try:
    from entity_utils import ensure_api_controllers, find_and_validate_configs, get_duplicate_protected_ids, get_supersede_map
except ImportError:
    from .entity_utils import ensure_api_controllers, find_and_validate_configs, get_duplicate_protected_ids, get_supersede_map

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Environment-dependent lookback time (hours), matching the object detector.
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
    max_separation_m: Optional[float]    # Final median separation ceiling
    measurement_noise_floor_m: Optional[float]  # Final Mahalanobis noise floor
    # RESTORE when the recent median same-time separation exceeds this.
    restore_separation_m: float
    restore_window_seconds: float        # Recent span compared for RESTORE


DETECTION_PARAMS_BY_ENVIRONMENT: Dict[str, DetectionParams] = {
    # Airborne duplicates are common false positives: aircraft in trail,
    # formation flights, and holding patterns can sit close together while
    # remaining distinct.  Keep the gates tight — a small Mahalanobis sigma,
    # a strict velocity match, more required overlapping points, and a higher
    # confidence floor — so only two feeds of the *same* aircraft are fused.
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

# Global variables for controllers (will be initialized in the child process)
rc = wc = auth = None

# Global tracking of processed SUPERSEDE actions
processed_supersedes: Set[str] = set()
active_supersedes: Dict[str, Any] = {}
_recent_restores: Dict[str, float] = {}


def _value(record: dict, path: str, default: Any = None) -> Any:
    current: Any = record
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _parse_timestamp(value: Any) -> Optional[dt]:
    if isinstance(value, dt):
        return value
    if not value:
        return None
    try:
        parsed = dt.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except (TypeError, ValueError):
        return None

@dataclass
class DuplicateCandidate:
    """Represents a candidate duplicate pair with supporting evidence."""
    track_id_1: str
    track_id_2: str
    avg_position_distance: float
    avg_velocity_difference: float
    num_matching_points: int
    time_overlap_seconds: float
    confidence_score: float
    environment: str = None
    
    def __repr__(self) -> str:
        return (f"DuplicateCandidate({self.track_id_1} <-> {self.track_id_2}, "
                f"mahal_dist={self.avg_position_distance:.2f}\u03c3, "
                f"vel_diff={self.avg_velocity_difference:.1f}m/s, "
                f"points={self.num_matching_points}, "
                f"confidence={self.confidence_score:.2f})")


@dataclass
class TrackHistory:
    """Stores position and velocity history for an entity."""
    entity_id: str
    track_id: str
    timestamps: List[dt] = field(default_factory=list)
    positions_ecef: List[Tuple[float, float, float]] = field(default_factory=list)
    velocities_ecef: List[Tuple[float, float, float]] = field(default_factory=list)
    position_covariances: List[np.ndarray] = field(default_factory=list)
    environment: str = None
    standard_identity: str = None
    
    def add_point(self, timestamp: dt, position: Tuple[float, float, float], 
                  velocity: Tuple[float, float, float],
                  position_covariance: np.ndarray = None) -> None:
        """Add a position/velocity point to the history."""
        self.timestamps.append(timestamp)
        self.positions_ecef.append(position)
        self.velocities_ecef.append(velocity)
        self.position_covariances.append(position_covariance)
    
    def get_time_range(self) -> Tuple[dt, dt]:
        """Return (min_time, max_time) for this track."""
        if not self.timestamps:
            return None, None
        return min(self.timestamps), max(self.timestamps)
    
    def __len__(self) -> int:
        return len(self.timestamps)


def calculate_ecef_distance(pos1: Tuple[float, float, float], 
                            pos2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between two ECEF positions.
    
    Args:
        pos1: (x, y, z) ECEF position in meters
        pos2: (x, y, z) ECEF position in meters
        
    Returns:
        float: Distance in meters
    """
    try:
        return np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
    except (TypeError, ValueError):
        return float('inf')


def _horizontal_projection(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> np.ndarray:
    """Build a 3x2 matrix whose columns span the local horizontal plane.

    The radial (altitude) direction is approximated as the unit vector
    from the Earth's centre to the midpoint of the two ECEF positions.
    Two orthonormal horizontal vectors are derived via cross products.
    """
    mid = (np.array(pos1, dtype=float) + np.array(pos2, dtype=float)) / 2.0
    r_norm = np.linalg.norm(mid)
    if r_norm < 1.0:  # degenerate (near origin)
        return np.eye(3)[:, :2]  # just use x, y
    up = mid / r_norm
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

    Falls back to horizontal Euclidean distance / 100 m when covariances are
    missing, or to full 3-D Euclidean / 100 m if the projection is singular.
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
            Ch = Ch + 2.0 * floor_var * np.eye(2)
        return float(np.sqrt(dh @ np.linalg.solve(Ch, dh)))
    except np.linalg.LinAlgError:
        return float(np.sqrt(dh @ dh)) / fallback_scale


def calculate_velocity_difference(vel1: Tuple[float, float, float], 
                                   vel2: Tuple[float, float, float]) -> float:
    """
    Calculate the magnitude of velocity difference (m/s).
    
    Args:
        vel1: (vx, vy, vz) ECEF velocity in m/s
        vel2: (vx, vy, vz) ECEF velocity in m/s
        
    Returns:
        float: Velocity difference magnitude in m/s
    """
    try:
        return np.sqrt(sum((a - b)**2 for a, b in zip(vel1, vel2)))
    except (TypeError, ValueError):
        return float('inf')


def calculate_velocity_magnitude(vel: Tuple[float, float, float]) -> float:
    """Calculate velocity magnitude from ECEF components."""
    try:
        return np.sqrt(sum(v**2 for v in vel))
    except (TypeError, ValueError):
        return 0.0


def get_principal_track_events(dataset_config: dict) -> List[dict]:
    """
    Retrieve component track events from the component_track_event_dataset.
    This gives us position history for accurate duplicate detection.

    The tracker creates deterministic trackIds; duplicate detection runs
    over component tracks (whose trackIds are the fuser's routing keys) so the
    SUPERSEDE actions it emits can be applied by the fuser.
    
    Args:
        dataset_config (dict): Configuration containing dataset names
        
    Returns:
        list[dict]: JSON track-event records
    """
    track_event_dataset = (dataset_config.get('component_track_event_dataset')
                           or dataset_config.get('principal_track_event_dataset'))
    if not track_event_dataset:
        logging.warning("No track event dataset configured")
        return []

    track_events: List[dict] = []
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
            result = rc.search(query, format='json', auto_backtick=False) or []
        except Exception as e:
            logging.error(f"Error querying {environment} track events: {e}")
            logging.error(traceback.format_exc())
            continue
        if len(result) >= TRACK_EVENT_LIMIT_PER_ENVIRONMENT:
            logging.warning(f" {environment} track events hit the {TRACK_EVENT_LIMIT_PER_ENVIRONMENT}-row "
                            f"limit; effective lookback is shorter than {hours}h")
        track_events.extend(result)

    logging.info(f" Found {len(track_events)} track events from {track_event_dataset}")
    return track_events


def _environment_filter(dataset: str, environment: str) -> str:
    column = f"{dataset}.`environment`"
    if environment != 'UNKNOWN':
        return f"{column} = '{environment}'"
    listed = ', '.join(f"'{env}'" for env in LOOKBACK_HOURS_BY_ENVIRONMENT if env != 'UNKNOWN')
    return f"({column} IS NULL OR {column} NOT IN ({listed}))"


def get_track_heads(dataset_config: dict) -> List[dict]:
    """
    Retrieve track heads from the principal track head dataset for current state.
    
    Args:
        dataset_config (dict): Configuration containing dataset names
        
    Returns:
        list[dict]: JSON track-head records
    """
    try:
        # Duplicate detection and fusion are keyed by component trackId.
        track_head_dataset = dataset_config.get('component_track_head_dataset')
        if not track_head_dataset:
            track_head_dataset = dataset_config.get('principal_track_head_dataset')
        
        if not track_head_dataset:
            logging.warning("No track head dataset configured")
            return []
        
        origin_dataset = dataset_config.get('origin_dataset', 'unknown')
        
        # Query for all track heads with ECEF positions
        query = f"""
        SELECT * FROM {track_head_dataset} 
        WHERE {track_head_dataset}.`ecefPosition`.x IS NOT NULL
        LIMIT 10000
        """
        
        logging.info(f" [using config for {origin_dataset}]: Querying track heads from {track_head_dataset}")
        
        rc.token = auth.get_token()
        result = rc.search(query, format='json', auto_backtick=False) or []
        
        if result:
            logging.info(f" [using config for {origin_dataset}]: Found {len(result)} track heads")
            return result
        else:
            logging.info(f" [using config for {origin_dataset}]: No track heads found")
            return []
    
    except Exception as e:
        logging.error(f"An error occurred in get_track_heads: {e}")
        logging.error(traceback.format_exc())
        return []


def build_track_histories(track_events: List[dict],
                          dataset_config: dict,
                          lookback_hours_by_env: Optional[Dict[str, float]] = None,
                          exclude_protected: bool = True) -> Dict[str, TrackHistory]:
    """
    Build track histories from JSON track-event records.
    
    Args:
        track_events: JSON track-event records
        dataset_config: Configuration dict
        
    Returns:
        Dict mapping trackId -> TrackHistory
    """
    histories: Dict[str, TrackHistory] = {}
    origin_dataset = dataset_config.get('origin_dataset', 'unknown')
    
    if not track_events:
        return histories
    
    # Required columns for position
    pos_cols = ['ecefPosition.x', 'ecefPosition.y', 'ecefPosition.z']
    vel_cols = ['ecefVelocity.x', 'ecefVelocity.y', 'ecefVelocity.z']
    
    # Check for required position columns
    missing_pos = [col for col in pos_cols
                   if not any(_value(record, col) is not None for record in track_events)]
    if missing_pos:
        logging.warning(f" Missing position columns: {missing_pos}")
        return histories
    
    # Check for velocity columns (optional but preferred)
    has_velocity = all(
        any(_value(record, col) is not None for record in track_events)
        for col in vel_cols)
    if not has_velocity:
        logging.info(f" Velocity columns not found, will use position-only comparison")
    
    # Determine timestamp column
    timestamp_col = None
    for col in ['interceptTimestamp', 'trackOriginatedTimestamp', 'crucibleHeader.createdDate']:
        if any(_value(record, col) is not None for record in track_events):
            timestamp_col = col
            break
    
    if not timestamp_col:
        logging.warning(" No timestamp column found")
        return histories
    
    # Check for real position covariance columns from the transformer
    cov_cols = ['positionCovariance.xx', 'positionCovariance.xy',
                'positionCovariance.xz', 'positionCovariance.yy',
                'positionCovariance.yz', 'positionCovariance.zz']
    has_cov = all(
        any(_value(record, col) is not None for record in track_events)
        for col in cov_cols)
    if has_cov:
        logging.info(" Using real position covariances from track events")

    # Collect raw points per component track
    raw: Dict[str, dict] = {}  # track_id -> {track_id, env, si, ts, pos, vel, cov}
    for row in track_events:
        track_id = _value(row, 'trackId')
        if not track_id:
            continue
        try:
            timestamp = _parse_timestamp(_value(row, timestamp_col))
            if timestamp is None:
                continue
        except Exception:
            continue
        try:
            position = (
                float(_value(row, 'ecefPosition.x', 0) or 0),
                float(_value(row, 'ecefPosition.y', 0) or 0),
                float(_value(row, 'ecefPosition.z', 0) or 0)
            )
            if all(p == 0 for p in position):
                continue
            if not all(np.isfinite(p) for p in position):
                continue
        except (TypeError, ValueError):
            continue
        if has_velocity:
            try:
                velocity = (
                    float(_value(row, 'ecefVelocity.x', 0) or 0),
                    float(_value(row, 'ecefVelocity.y', 0) or 0),
                    float(_value(row, 'ecefVelocity.z', 0) or 0)
                )
            except (TypeError, ValueError):
                velocity = (0.0, 0.0, 0.0)
        else:
            velocity = (0.0, 0.0, 0.0)

        # Reconstruct 3×3 position covariance if available
        meas_cov = None
        if has_cov:
            try:
                xx = float(_value(row, 'positionCovariance.xx'))
                xy = float(_value(row, 'positionCovariance.xy'))
                xz = float(_value(row, 'positionCovariance.xz'))
                yy = float(_value(row, 'positionCovariance.yy'))
                yz = float(_value(row, 'positionCovariance.yz'))
                zz = float(_value(row, 'positionCovariance.zz'))
                meas_cov = np.array([[xx, xy, xz],
                                     [xy, yy, yz],
                                     [xz, yz, zz]])
                if not np.all(np.isfinite(meas_cov)):
                    meas_cov = None
            except (TypeError, ValueError, KeyError):
                meas_cov = None

        if track_id not in raw:
            raw[track_id] = {
                'track_id': track_id,
                'env': _value(row, 'environment'),
                'si': _value(row, 'standardIdentity'),
                'ts': [], 'pos': [], 'vel': [], 'cov': [],
            }
        raw[track_id]['ts'].append(timestamp)
        raw[track_id]['pos'].append(position)
        raw[track_id]['vel'].append(velocity)
        raw[track_id]['cov'].append(meas_cov)

    # Filter out already-superseded component tracks
    superseded_ids = raw.keys() & processed_supersedes if exclude_protected else set()
    if superseded_ids:
        for sid in list(superseded_ids):
            del raw[sid]
        logging.info(f" Filtered out {len(superseded_ids)} already-superseded/deleted tracks")

    if lookback_hours_by_env:
        now = dt.now(timezone.utc).replace(tzinfo=None)
        for track_id, data in list(raw.items()):
            env_key = str(data.get('env') or 'UNKNOWN').upper()
            cutoff = now - timedelta(hours=lookback_hours_by_env.get(
                env_key, lookback_hours_by_env['UNKNOWN']))
            keep = [i for i, ts in enumerate(data['ts'])
                    if ts.replace(tzinfo=None) >= cutoff]
            if not keep:
                del raw[track_id]
                continue
            for key in ('ts', 'pos', 'vel', 'cov'):
                data[key] = [data[key][i] for i in keep]

    # Smooth and build TrackHistory objects
    for track_id, data in raw.items():
        vels = data['vel'] if has_velocity else None
        # Build per-measurement covariance array if all points have real covs
        meas_covs = None
        if all(c is not None for c in data['cov']):
            meas_covs = np.stack(data['cov'])
        try:
            s_pos, s_vel, s_cov = smooth_track(
                data['ts'], data['pos'], vels,
                return_covariances=True,
                measurement_covariances=meas_covs)
        except Exception:
            s_pos = np.array(data['pos'])
            s_vel = np.array(data['vel']) if has_velocity else np.zeros((len(data['pos']), 3))
            s_cov = np.stack([np.eye(3) * 100.0**2] * len(data['pos']))

        history = TrackHistory(
            entity_id=track_id,
            track_id=data['track_id'] or track_id,
            environment=data['env'],
            standard_identity=data['si'],
        )
        for i in range(len(data['ts'])):
            history.add_point(data['ts'][i], tuple(s_pos[i]), tuple(s_vel[i]), s_cov[i])
        histories[track_id] = history

    logging.info(f" Built smoothed track histories for {len(histories)} tracks")
    return histories


# ---------------------------------------------------------------------------
# Kinematic interpolation helpers
# ---------------------------------------------------------------------------

def _epoch(t: Any) -> float:
    """Convert a datetime to epoch seconds."""
    if hasattr(t, 'timestamp') and callable(t.timestamp):
        return t.timestamp()
    if t.tzinfo is None:
        return (t - dt(1970, 1, 1)).total_seconds()
    return (t.replace(tzinfo=None) - dt(1970, 1, 1)).total_seconds()


def interpolate_track_at_times(history: TrackHistory,
                               eval_times: List[float],
                               process_noise_q: float = 1.0,
                               max_extrapolate_seconds: float = 30.0,
                               ) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Interpolate/extrapolate a smoothed track to arbitrary evaluation times.

    Uses constant-velocity kinematic prediction from the nearest bracketing
    smoothed state:

        pos(t) = pos(t_k) + vel(t_k) * dt
        P(t)   = P(t_k) + Q * |dt|

    where Q = process_noise_q * |dt|^2 * I_3  (position uncertainty grows
    with time in a random-walk acceleration model).

    Args:
        history: Smoothed TrackHistory with positions, velocities, covariances.
        eval_times: List of epoch-seconds at which to evaluate.
        process_noise_q: Process noise spectral density (m^2/s^3) for
                         covariance growth during extrapolation.
        max_extrapolate_seconds: Maximum extrapolation beyond track endpoints.

    Returns:
        List (same length as eval_times) of either:
            (position_3, velocity_3, covariance_3x3)  — interpolated state
            None  — if eval_time is outside the track + max_extrapolate buffer
    """
    if not history.timestamps:
        return [None] * len(eval_times)

    # Build sorted arrays of track states
    track_epochs = np.array([_epoch(t) for t in history.timestamps])
    sort_idx = np.argsort(track_epochs)
    track_epochs = track_epochs[sort_idx]
    positions = np.array(history.positions_ecef)[sort_idx]
    velocities = np.array(history.velocities_ecef)[sort_idx]
    covariances = [history.position_covariances[i] for i in sort_idx]

    t_min = track_epochs[0] - max_extrapolate_seconds
    t_max = track_epochs[-1] + max_extrapolate_seconds

    results = []
    for t_eval in eval_times:
        if t_eval < t_min or t_eval > t_max:
            results.append(None)
            continue

        # Find bracketing index: rightmost track point <= t_eval
        idx = np.searchsorted(track_epochs, t_eval, side='right') - 1
        idx = np.clip(idx, 0, len(track_epochs) - 1)

        # If we're between two points, pick the closer one for prediction
        if idx < len(track_epochs) - 1:
            dt_left = t_eval - track_epochs[idx]
            dt_right = track_epochs[idx + 1] - t_eval
            if dt_right < dt_left:
                idx = idx + 1

        dt_pred = t_eval - track_epochs[idx]
        pred_pos = positions[idx] + velocities[idx] * dt_pred
        pred_vel = velocities[idx].copy()

        # Covariance growth: P(t) = P(t_k) + q * |dt|^2 * I
        base_cov = covariances[idx]
        if base_cov is not None:
            pred_cov = base_cov + process_noise_q * dt_pred**2 * np.eye(3)
        else:
            pred_cov = np.eye(3) * (100.0**2 + process_noise_q * dt_pred**2)

        results.append((pred_pos, pred_vel, pred_cov))

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
    """Evaluate whether two track histories represent duplicates using interpolation.

    Instead of requiring measurements at nearly the same timestamps,
    both tracks are interpolated/extrapolated to a common evaluation
    grid using constant-velocity prediction from the smoothed states.
    This allows comparison of staggered measurements from different
    sensors.

    *final_mahalanobis_threshold_sigma* is a final duplicate-classification
    threshold in Mahalanobis sigma units (e.g. 3.0); it is unrelated to the
    cKDTree candidate search radius.
    *time_alignment_seconds* controls the maximum extrapolation beyond
    each track's endpoints (renamed semantically but kept for API compat).
    """
    # Skip if same entity
    if history1.entity_id == history2.entity_id:
        return None
    
    env1 = (history1.environment or '').upper()
    env2 = (history2.environment or '').upper()
    if env1 != env2:
        return None
    
    # Compute common evaluation grid over the time overlap
    max_extrapolate = time_alignment_seconds
    grid_result = _compute_evaluation_grid(
        history1, history2,
        eval_interval_seconds=eval_interval_seconds,
        max_extrapolate_seconds=max_extrapolate)
    if grid_result is None:
        return None
    eval_times, overlap_seconds = grid_result

    measurements1 = len({_epoch(timestamp) for timestamp in history1.timestamps})
    measurements2 = len({_epoch(timestamp) for timestamp in history2.timestamps})
    independent_points = min(measurements1, measurements2)
    if independent_points < min_matching_points:
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
    velocity_deltas = []
    position_separations = []
    noise_floor = float(measurement_noise_floor_m or 0.0)
    for s1, s2 in zip(states1, states2):
        if s1 is None or s2 is None:
            continue
        pos1, vel1, cov1 = s1
        pos2, vel2, cov2 = s2
        position_distances.append(
            calculate_mahalanobis_distance(
                tuple(pos1), tuple(pos2), cov1, cov2,
                pos_noise_floor_m=noise_floor))
        position_separations.append(calculate_ecef_distance(tuple(pos1), tuple(pos2)))
        velocity_deltas.append(np.asarray(vel1, dtype=float) - np.asarray(vel2, dtype=float))
        velocity_differences.append(
            calculate_velocity_difference(tuple(vel1), tuple(vel2)))

    if len(position_distances) < min_matching_points:
        return None
    
    median_separation = float(np.median(position_separations))
    if max_separation_m is not None and median_separation > max_separation_m:
        return None

    avg_pos_dist = np.mean(position_distances)
    avg_vel_diff = float(np.linalg.norm(np.mean(np.stack(velocity_deltas), axis=0)))
    
    if avg_pos_dist > final_mahalanobis_threshold_sigma:
        return None
    
    # Velocity check (only if velocity data is meaningful)
    has_meaningful_velocity = any(calculate_velocity_magnitude(v) > 0.1 
                                   for v in history1.velocities_ecef + history2.velocities_ecef)
    
    if has_meaningful_velocity and avg_vel_diff > velocity_threshold_mps:
        return None
    
    pos_std = np.std(position_distances) if len(position_distances) > 1 else 0
    vel_std = np.std(velocity_differences) if len(velocity_differences) > 1 else 0
    
    num_points_factor = min(1.0, independent_points / 10.0)
    pos_consistency_factor = 1.0 / (1.0 + pos_std / final_mahalanobis_threshold_sigma)
    vel_consistency_factor = 1.0 / (1.0 + vel_std / max(velocity_threshold_mps, 1.0)) if has_meaningful_velocity else 1.0
    
    confidence = (num_points_factor * 0.4 + 
                  pos_consistency_factor * 0.4 + 
                  vel_consistency_factor * 0.2)
    
    # Calculate time overlap
    t1_min, t1_max = history1.get_time_range()
    t2_min, t2_max = history2.get_time_range()
    if t1_min and t2_min:
        overlap_start = max(_epoch(t1_min), _epoch(t2_min))
        overlap_end = min(_epoch(t1_max), _epoch(t2_max))
        time_overlap = max(0.0, overlap_end - overlap_start)
    else:
        time_overlap = 0
    
    return DuplicateCandidate(
        track_id_1=history1.entity_id,
        track_id_2=history2.entity_id,
        avg_position_distance=avg_pos_dist,
        avg_velocity_difference=avg_vel_diff,
        num_matching_points=independent_points,
        time_overlap_seconds=time_overlap,
        confidence_score=confidence,
        environment=history1.environment or history2.environment
    )


def find_candidate_pairs_kdtree(histories: Dict[str, TrackHistory],
                                 candidate_search_radius_m: float,
                                 time_alignment_seconds: float,
                                 min_matching_points: int) -> Set[Tuple[str, str]]:
    """Use cKDTree (spatial-only) to find entity pairs with nearby positions.

    Since tracks from different sensors may have staggered timestamps,
    the pre-filter uses a 3-D spatial tree (no time dimension).  A pair
    is a candidate if:
        1. At least *min_matching_points* positions from different entities
            are within *candidate_search_radius_m* of each other (spatially).
      2. The two tracks have overlapping time coverage (with a buffer of
         *time_alignment_seconds* for extrapolation).
    """
    # Build environment and time-range lookup
    env_by_entity: Dict[str, str] = {}
    time_range_by_entity: Dict[str, Tuple[float, float]] = {}
    for entity_id, history in histories.items():
        env_by_entity[entity_id] = (history.environment or '').upper()
        if history.timestamps:
            epochs = [_epoch(t) for t in history.timestamps]
            time_range_by_entity[entity_id] = (min(epochs), max(epochs))

    all_points = []
    all_entity_ids = []
    all_timestamps = []

    for entity_id, history in histories.items():
        for timestamp, pos in zip(history.timestamps, history.positions_ecef):
            if not all(np.isfinite(p) for p in pos):
                continue
            all_points.append((pos[0], pos[1], pos[2]))
            all_entity_ids.append(entity_id)
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
        all_entity_ids = [all_entity_ids[i] for i in good_idx]
        all_timestamps = [all_timestamps[i] for i in good_idx]
        if len(points_array) < 2:
            return set()

    tree = cKDTree(points_array)

    pair_measurements: DefaultDict[Tuple[str, str], Tuple[Set[float], Set[float]]] = defaultdict(
        lambda: (set(), set()))
    pairs = tree.query_pairs(r=candidate_search_radius_m, output_type='ndarray')

    for idx1, idx2 in pairs:
        entity1 = all_entity_ids[idx1]
        entity2 = all_entity_ids[idx2]
        if entity1 == entity2:
            continue
        # Skip cross-environment pairs early
        if env_by_entity[entity1] != env_by_entity[entity2]:
            continue
        pair_key = tuple(sorted([entity1, entity2]))
        timestamps1, timestamps2 = pair_measurements[pair_key]
        if entity1 == pair_key[0]:
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
        ent_a, ent_b = pair
        # Check time overlap (with extrapolation buffer)
        tr_a = time_range_by_entity.get(ent_a)
        tr_b = time_range_by_entity.get(ent_b)
        if tr_a is None or tr_b is None:
            continue
        a_min, a_max = tr_a[0] - time_alignment_seconds, tr_a[1] + time_alignment_seconds
        b_min, b_max = tr_b[0] - time_alignment_seconds, tr_b[1] + time_alignment_seconds
        if a_max < b_min or b_max < a_min:
            continue  # No time overlap even with extrapolation
        candidates.add(pair)

    return candidates


def find_duplicate_tracks(track_events: List[dict],
                          track_heads: List[dict],
                          dataset_config: dict) -> List[DuplicateCandidate]:
    """Find duplicate tracks using smoothed position history and velocity comparison."""
    duplicates: List[DuplicateCandidate] = []
    origin_dataset = dataset_config.get('origin_dataset', 'unknown')
    
    try:
        # Build track histories from events
        histories = build_track_histories(
            track_events, dataset_config,
            lookback_hours_by_env=LOOKBACK_HOURS_BY_ENVIRONMENT)
        
        if len(histories) < 2:
            logging.info(" Not enough track histories for comparison")
            return duplicates
        
        n_entities = len(histories)
        total_points = sum(len(h) for h in histories.values())
        logging.info(f" Analyzing {n_entities} entities with {total_points} total track points")
        
        # Use the loosest thresholds across all environments for the pre-filter
        all_params = list(DETECTION_PARAMS_BY_ENVIRONMENT.values())
        max_candidate_search_radius_m = max(
            p.candidate_search_radius_m for p in all_params)
        max_time_align = max(p.time_alignment_seconds for p in all_params)
        min_pts = min(p.min_matching_points for p in all_params)

        logging.info(f" Building spatial index (cKDTree) for fast duplicate detection...")
        candidate_pairs = find_candidate_pairs_kdtree(
            histories,
            candidate_search_radius_m=max_candidate_search_radius_m,
            time_alignment_seconds=max_time_align,
            min_matching_points=min_pts,
        )
        
        logging.info(f" Found {len(candidate_pairs)} candidate pairs to evaluate "
                    f"(reduced from {n_entities * (n_entities - 1) // 2} total pairs)")
        
        for track_id_1, track_id_2 in candidate_pairs:
            params = detection_params_for(histories[track_id_1].environment)

            candidate = evaluate_duplicate_candidate(
                histories[track_id_1], histories[track_id_2],
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
        
        # Sort by confidence (highest first)
        duplicates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logging.info(f" Found {len(duplicates)} confirmed duplicates")
        return duplicates
    
    except Exception as e:
        logging.error(f" Error in find_duplicate_tracks: {e}")
        logging.error(traceback.format_exc())
        return duplicates


def get_confirmed_track_ids(track_heads: List[dict]) -> Set[str]:
    """Return component trackIds whose latest head is CONFIRMED."""
    return {
        str(_value(record, 'trackId')) for record in track_heads
        if _value(record, 'trackId') is not None
        and str(_value(record, 'entityStatus', '')).upper() == 'CONFIRMED'}


def determine_entity_to_keep(candidate: DuplicateCandidate,
                              track_heads: List[dict],
                              confirmed_ids: Optional[Set[str]] = None) -> Tuple[str, str]:
    """
    Determine which entity to keep (superseding) and which to supersede.
    
    CONFIRMED tracks are always kept; otherwise keep the earlier component
    track, falling back to lexical order.
    
    Args:
        candidate: DuplicateCandidate
        track_heads: Current track heads
        confirmed_ids: CONFIRMED component trackIds
    Returns:
        Tuple of (entity_to_supersede, entity_to_keep)
    """
    entity1 = candidate.track_id_1
    entity2 = candidate.track_id_2

    if confirmed_ids:
        if entity1 in confirmed_ids and entity2 not in confirmed_ids:
            return (entity2, entity1)
        if entity2 in confirmed_ids and entity1 not in confirmed_ids:
            return (entity1, entity2)
    
    # Try to get track counts from track heads
    entity1_info = next(
        (record for record in track_heads
         if str(_value(record, 'trackId')) == entity1), None)
    entity2_info = next(
        (record for record in track_heads
         if str(_value(record, 'trackId')) == entity2), None)
    
    # Get creation dates if available
    entity1_created = None
    entity2_created = None
    
    if entity1_info is not None:
        for col in ['trackOriginatedTimestamp', 'crucibleHeader.createdDate']:
            entity1_created = _parse_timestamp(_value(entity1_info, col))
            if entity1_created is not None:
                break
    
    if entity2_info is not None:
        for col in ['trackOriginatedTimestamp', 'crucibleHeader.createdDate']:
            entity2_created = _parse_timestamp(_value(entity2_info, col))
            if entity2_created is not None:
                break
    
    # Decision logic: keep the older entity
    if entity1_created and entity2_created:
        if entity1_created < entity2_created:
            return (entity2, entity1)  # Keep entity1
        elif entity2_created < entity1_created:
            return (entity1, entity2)  # Keep entity2
    
    # Fall back to lexicographic comparison for determinism
    if entity1 < entity2:
        return (entity2, entity1)  # Keep entity1
    else:
        return (entity1, entity2)  # Keep entity2


def create_supersede_action(superseded_entity_id: str, 
                            superseding_entity_id: str, 
                            candidate: DuplicateCandidate,
                            dataset_config: dict) -> bool:
    """
    Create a SUPERSEDE action to merge duplicate component tracks.

    Both IDs are component trackIds. The management event records the
    superseded track in ``trackId`` and the surviving track in ``supersededBy``.

    Args:
        superseded_entity_id: trackId of the component track to be superseded
        superseding_entity_id: trackId of the component track that supersedes
        candidate: DuplicateCandidate with supporting evidence
        dataset_config: Configuration containing dataset names
        
    Returns:
        bool: True if action was created successfully
    """
    origin_dataset = dataset_config.get('origin_dataset', 'unknown')
    
    try:
        # Check if we've already processed this SUPERSEDE
        if superseded_entity_id in processed_supersedes:
            logging.debug(f" SUPERSEDE action already processed: {superseded_entity_id}")
            return False
        
        # Build reason with evidence
        reason = (f"Duplicate entity detected: avg_pos_diff={candidate.avg_position_distance:.1f}m, "
                  f"avg_vel_diff={candidate.avg_velocity_difference:.1f}m/s, "
                  f"matching_points={candidate.num_matching_points}, "
                  f"confidence={candidate.confidence_score:.2f}")
        reason = reason[:100] # Truncate to 100 chars if needed
        management_event = {
            "action": "SUPERSEDE",
            "trackId": superseded_entity_id,
            "supersededBy": superseding_entity_id,
            "source": {
                "datasetName": "entity_duplicate_identifier",
                "uuid": '0'*32
            },
            "sourceRaw": {
                "datasetName": "entity_duplicate_identifier",
                "uuid": '0'*32
            },
            "mechanism": "TRACKER",
            "status": "COMPLETED",
            "edhControlSet": ["CLS:U"],
            "reason": reason
        }
        
        json_event = [management_event]
        
        wc.token = auth.get_token()
        
        management_dataset = dataset_config.get('entity_management_event_dataset', 
                                                 'EntityManagementEvents')
        
        # Lookup callsigns from track_heads_df if available
        callsign_superseded = None
        callsign_superseding = None
        track_heads = dataset_config.get('track_heads', [])
        info_superseded = next(
            (record for record in track_heads
             if str(_value(record, 'trackId')) == superseded_entity_id), None)
        info_superseding = next(
            (record for record in track_heads
             if str(_value(record, 'trackId')) == superseding_entity_id), None)
        if info_superseded is not None:
            callsign_superseded = _value(
                info_superseded, 'callsign', _value(info_superseded, 'identity.callsign'))
        if info_superseding is not None:
            callsign_superseding = _value(
                info_superseding, 'callsign', _value(info_superseding, 'identity.callsign'))
        
        logging.info(f" Writing SUPERSEDE action to {management_dataset}: "
                    f"{superseded_entity_id} ({callsign_superseded}) -> {superseding_entity_id} ({callsign_superseding})")
        logging.info(f" Reason: {reason}")

        wc.token = auth.get_token()
        wc.write_record_batch_by_name(management_dataset, json_event)

        # Mark both sides as processed to prevent circular re-detection
        processed_supersedes.add(superseded_entity_id)
        processed_supersedes.add(superseding_entity_id)

        logging.info(f" Created SUPERSEDE action: "
                    f"{superseded_entity_id} ({callsign_superseded}) -> {superseding_entity_id} ({callsign_superseding})")
        return True
    
    except Exception as e:
        logging.error(f" Error creating SUPERSEDE action: {e}")
        logging.error(traceback.format_exc())
        return False


def process_duplicates(duplicates: List[DuplicateCandidate],
                       track_heads: List[dict],
                       dataset_config: dict) -> int:
    """
    Process all found duplicates and create SUPERSEDE actions.
    
    Args:
        duplicates: List of DuplicateCandidate objects
        track_heads: Current track heads
        dataset_config: Configuration dict
        
    Returns:
        int: Number of SUPERSEDE actions created
    """
    origin_dataset = dataset_config.get('origin_dataset', 'unknown')
    actions_created = 0
    confirmed_ids = get_confirmed_track_ids(track_heads)
    
    for candidate in duplicates:
        superseded_id, superseding_id = determine_entity_to_keep(
            candidate, track_heads, confirmed_ids)
        
        # Skip if already processed
        if superseded_id in processed_supersedes:
            continue
        if superseded_id in confirmed_ids:
            logging.info(f" Skipping pair: both {superseded_id} and {superseding_id} are CONFIRMED")
            continue
        
        if create_supersede_action(superseded_id, superseding_id, candidate, dataset_config):
            actions_created += 1
    
    if actions_created > 0:
        logging.info(f" Created {actions_created} SUPERSEDE actions")
    
    return actions_created


def find_divergent_supersedes(histories: Dict[str, TrackHistory],
                              supersede_map: Dict[str, Any],
                              min_points: int) -> List[str]:
    """Return superseded trackIds whose tracks recently separated from their survivor."""
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


def create_restore_action(track_id: str, dataset_config: dict) -> bool:
    """Write a RESTORE management event for a superseded component track."""
    last_restore = _recent_restores.get(track_id)
    if (last_restore is not None
            and time.monotonic() - last_restore < DEFAULT_RESTORE_COOLDOWN_SECONDS):
        return False
    management_event = {
        "action": "RESTORE",
        "trackId": track_id,
        "source": {"datasetName": "entity_duplicate_identifier", "uuid": '0'*32},
        "sourceRaw": {"datasetName": "entity_duplicate_identifier", "uuid": '0'*32},
        "mechanism": "TRACKER",
        "status": "COMPLETED",
        "edhControlSet": ["CLS:U"],
        "reason": "Superseded track diverged from its survivor",
    }
    try:
        management_dataset = dataset_config.get('entity_management_event_dataset',
                                                'EntityManagementEvents')
        wc.token = auth.get_token()
        wc.write_record_batch_by_name(management_dataset, [management_event])
    except Exception as e:
        logging.error(f" Error creating RESTORE action for {track_id}: {e}")
        return False
    _recent_restores[track_id] = time.monotonic()
    active_supersedes.pop(track_id, None)
    processed_supersedes.add(track_id)
    logging.info(f" Created RESTORE action: {track_id}")
    return True


def process_restores(track_events: List[dict], dataset_config: dict,
                     min_points: int) -> int:
    """Emit RESTORE events for active supersedes whose tracks have diverged."""
    pairs = {track_id: survivor for track_id, survivor in active_supersedes.items()
             if survivor}
    if not pairs:
        return 0
    pair_ids = set(pairs) | set(pairs.values())
    pair_events = [event for event in track_events
                   if str(_value(event, 'trackId')) in pair_ids]
    histories = build_track_histories(
        pair_events, dataset_config,
        lookback_hours_by_env=LOOKBACK_HOURS_BY_ENVIRONMENT,
        exclude_protected=False)
    return sum(create_restore_action(track_id, dataset_config)
               for track_id in find_divergent_supersedes(histories, pairs, min_points))


def refresh_protected_ids(perspective_config: dict) -> None:
    """Refresh the duplicate detector's local protected-ID cache."""
    global processed_supersedes, active_supersedes
    try:
        supersede_map = get_supersede_map(
            perspective_config, rc_instance=rc, auth_instance=auth)
        processed_supersedes = get_duplicate_protected_ids(
            perspective_config, rc, auth, DEFAULT_RESTORE_COOLDOWN_SECONDS,
            supersede_map=supersede_map)
        active_supersedes = supersede_map
        logging.info(f"  Loaded {len(processed_supersedes)} protected track ID(s)")
    
    except Exception as e:
        logging.warning(f" Could not load existing SUPERSEDE actions: {e}")
        logging.warning(traceback.format_exc())



def run(stream_manager_perspective: str, 
        log_level: str = None, 
        poll_interval: Optional[int] = None,
        restore_min_points: Optional[int] = None) -> None:
    parser = build_arg_parser()
    if poll_interval is None:
        poll_interval = parser.get_default('poll_interval')
    if restore_min_points is None:
        restore_min_points = parser.get_default('restore_min_points')
    # Set up logging
    if log_level is None:
        log_level = 'info'
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s: %(levelname)s %(name)s %(module)s Func: %(funcName)s:%(lineno)d-%(message)s',
        force=True
    )
    
    logging.info("=" * 60)
    logging.info("Starting Entity Duplicate Identifier")
    logging.info("=" * 60)
    logging.info(f"  Perspective: {stream_manager_perspective}")
    logging.info(f"  Lookback hours: {LOOKBACK_HOURS_BY_ENVIRONMENT}")
    logging.info(f"  Restore min points: {restore_min_points}")
    logging.info(f"  Poll interval: {poll_interval}s")
    
    # Initialize controllers first
    global auth, rc, wc
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)
    
    # Get config dictionaries from Crucible stream manager datasets
    result = find_and_validate_configs(stream_manager_perspective, include_scripts=False, rc_instance=rc)
    perspective_config = result['perspective_config']

    if not perspective_config:
        logging.error(f"No perspective config found for perspective: {stream_manager_perspective}")
        return

    
    origin_dataset = perspective_config.get('origin_dataset', 'unknown')
    track_event_dataset = perspective_config.get('component_track_event_dataset',
                                           perspective_config.get('principal_track_event_dataset', 'ComponentTrackEvents'))
    track_head_dataset = perspective_config.get('component_track_head_dataset',
                                          perspective_config.get('principal_track_head_dataset', 'ComponentTrackHeads'))
    management_dataset = perspective_config.get('entity_management_event_dataset', 'EntityManagementEvents')
    
    logging.info(f" [{origin_dataset}]: Using track event dataset: {track_event_dataset}")
    logging.info(f" [{origin_dataset}]: Using track head dataset: {track_head_dataset}")
    logging.info(f" [{origin_dataset}]: Writing SUPERSEDE actions to: {management_dataset}")
    
    # Load existing SUPERSEDE actions to avoid duplicates
    refresh_protected_ids(perspective_config)
    
    # Run main program loop
    while True:
        try:
            logging.info('-' * 50)
            # Refresh supersede map each iteration
            refresh_protected_ids(perspective_config)

            logging.info(" Running kinematic duplicate detection...")
            track_events = get_principal_track_events(perspective_config)
            track_heads = get_track_heads(perspective_config)
            if track_events:
                duplicates = find_duplicate_tracks(
                    track_events=track_events,
                    track_heads=track_heads,
                    dataset_config=perspective_config,
                )
                if duplicates:
                    process_duplicates(duplicates, track_heads, perspective_config)
                else:
                    logging.info(" No kinematic duplicates found")
                process_restores(track_events, perspective_config, restore_min_points)
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
        description='Entity Duplicate Identifier for SUPERSEDE Actions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python entity_duplicate_identifier.py Live
        ''',
    )
    parser.add_argument("stream_manager_perspective", 
                        help='Name of perspective in Entity Stream Manager Configuration (e.g., Live)')
    parser.add_argument("--log", default='INFO',
                        help='Logging level: INFO (default), WARN, ERROR, or DEBUG')
    parser.add_argument("--poll-interval", type=int, default=30,
                        help='Interval in seconds between searches (default: 30)')
    parser.add_argument("--restore-min-points", type=int, default=3,
                        help='Min points per track and aligned comparisons to RESTORE (default: 3)')

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    
    run(
        args.stream_manager_perspective, 
        log_level=args.log, 
        poll_interval=args.poll_interval,
        restore_min_points=args.restore_min_points,
    )
