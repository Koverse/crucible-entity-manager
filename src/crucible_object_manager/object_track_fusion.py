#!/usr/bin/env python
"""
Track Fusion for Principal Track Creation

This script creates a track fusion system that:
1. Reads component track events from object_tracker.py output
2. For non-fused objects: passes kinematics directly to principal track events/heads
3. For superseded objects (per object_duplicate_identifier.py): fuses kinematics using Covariance Intersection (CI)

Usage:
    python track_fusion.py <Stream Manager perspective> --log=DEBUG
    e.g., python track_fusion.py Live_POV --log=DEBUG

The script will:
- Load configuration from Stream_Manager_Configuration dataset
- Listen for component track events via SSE
- Check supersede map for objects that should be fused
- For non-superseded objects: pass through kinematics directly
- For superseded objects: fuse kinematics using Covariance Intersection into the superseding object's principal track
- Write principal track events and heads to their respective datasets

Principal Track Logic:
- Each unique object gets one principal track
- When objects are superseded, their component tracks are fused into the superseding object's principal track
- Covariance Intersection (CI) is used by default to combine measurements from multiple component tracks
  (use --no-ci to fall back to standard Kalman update, or --passthrough to skip filtering entirely)
"""

import os
import sys
import re
import json
import uuid
import signal
import hashlib
import argparse
import logging
import asyncio
import time
import traceback
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue as MPQueue
from typing import Dict, List, Any, Optional
from datetime import datetime as dt
from datetime import timezone as tz
from datetime import timedelta

# Add parent directory to path for principal_track_kalman module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from cruciblelib import read_controller, write_controller, authenticator, utils, log_utils

try:
    from object_utils import find_and_validate_configs, get_supersede_map, build_supersede_map, write_batch_chunked, instantiate_api_controllers, _fast_df_to_nested_json, normalize_uuid, SSE_listener
except ImportError:
    from .object_utils import find_and_validate_configs, get_supersede_map, build_supersede_map, write_batch_chunked, instantiate_api_controllers, _fast_df_to_nested_json, normalize_uuid, SSE_listener

from principal_track_kalman import (
    PrincipalTrackKalmanFilter, NumpyPrincipalTrackKalmanFilter,
    create_measurement_from_row, build_principal_state_dict)

# Global variables for controllers (will be initialized before use)
rc = wc = auth = None
logger = log_utils.get_logger(log_type='transformer', log_level=logging.INFO)

# --- [PERF] timing switch (single toggle for ALL per-stage timing logs) ---
# Flip to False (or set env CRUCIBLE_PERF_TIMING=false) to silence every [PERF]
# line and make the timers no-ops.
PERF_TIMING = os.getenv('CRUCIBLE_PERF_TIMING', 'true').lower() not in ('false', '0', 'no')
_DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS = float(
    os.getenv('CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS', '15'))


def _perf_disabled() -> float:
    """Zero-cost stand-in for time.perf_counter when PERF_TIMING is off."""
    return 0.0


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
                                    interval_seconds: float, label: str) -> pd.DataFrame:
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
            f"Coalesced existing {label} track-head write(s); "
            f"coalesced={coalesced} kept={len(kept)} interval={interval_seconds:g}s")
    return kept

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def terminate(signum: int, frame: Any) -> None:
    """Signal handler for graceful shutdown — kills entire process group."""
    logging.info(f"Track Fusion: Received signal {signum}, killing process group")
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def initialize_controllers() -> None:
    """Initialize the read/write controllers and authenticator."""
    global rc, wc, auth
    if rc is None or wc is None or auth is None:
        auth, rc, wc = instantiate_api_controllers()


def get_current_timestamp_string() -> str:
    """Get current UTC timestamp as ISO string."""
    return dt.now(tz=tz.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'


def get_current_timestamp_string_with_offset(offset_hours: int = 0) -> str:
    """Get UTC timestamp with offset as ISO string."""
    time_with_offset = dt.now(tz=tz.utc) + timedelta(hours=offset_hours)
    return time_with_offset.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'


def normalize_object_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize flattened objectId columns to flat names.

    Crucible stores objectId as ``{uuid: "..."}`` so after
    ``pd.json_normalize`` the column becomes ``objectId.uuid``.
    The supersede map and KF track management always use the bare
    ``objectId`` / ``supersededBy`` column names.  This helper renames
    the nested variants so all downstream code can use a single name.
    """
    renames = {}
    if 'objectId.uuid' in df.columns and 'objectId' not in df.columns:
        renames['objectId.uuid'] = 'objectId'
    if 'supersededBy.uuid' in df.columns and 'supersededBy' not in df.columns:
        renames['supersededBy.uuid'] = 'supersededBy'
    if renames:
        df = df.rename(columns=renames)
    return df


def process_with_fusion(event_df: pd.DataFrame,
                        config: Dict[str, Any],
                        kf: PrincipalTrackKalmanFilter,
                        object_id_col: str) -> tuple:
    """
    Process component track events with Kalman filter fusion for superseded objects.
    
    Args:
        event_df: DataFrame containing component track events
        config: Configuration dictionary
        kf: PrincipalTrackKalmanFilter instance
        object_id_col: Name of the object ID column
        
    Returns:
        DataFrame of principal track events
    """

    principal_tracks = []
    component_to_principal_associations = []
    
    for idx, row in event_df.iterrows():
        object_id = row[object_id_col]
        component_track_id = row.get('trackId')
        
        # Skip events for objects that have been deleted. A DELETE management
        # event maps the objectId to None in the supersede map, so the chain
        # resolves to None. Creating a principal track for it would produce a
        # record with a null objectId (stripped by df_to_formatted_JSON),
        # which the schema rejects with a 400.
        if kf.get_principal_id(object_id) is None:
            continue
        
        # Get or create principal track for this object (handles supersede)
        row_env = row.get('environment')
        row_env = row_env if isinstance(row_env, str) and row_env else None
        principal_track_id, principal_object_id, is_new, is_new_association = (
            kf.get_or_create_principal_track(object_id, component_track_id, environment=row_env))

        kf.set_track_environment(principal_track_id, row_env)
        
        # Track new associations for writeback to component track heads
        if is_new_association and component_track_id:
            component_to_principal_associations.append({
                'trackId': component_track_id,
                'associatedPrincipalTrack': principal_track_id
            })
        
        # Track component association
        if component_track_id:
            kf.add_component_track(principal_track_id, component_track_id)
        
        # Create measurement
        measurement = kf.create_measurement(row, principal_track_id)
        if measurement is None:
            continue
        
        # Update Kalman filter
        post = kf.update_filter(principal_track_id, measurement)
        if post is None:
            continue
        
        # Build principal track event from filtered state
        principal_event = build_principal_state_dict(
            post, principal_track_id, principal_object_id,
            kf.id_field, kf.vel_write_cols)
        principal_event['trackOriginatedTimestamp'] = row.get(
            'trackOriginatedTimestamp', row.get('crucibleHeader.createdDate', get_current_timestamp_string()))
        
        # Carry source (intercept) time through to the principal track event
        intercept_ts = row.get('interceptTimestamp')
        if pd.notna(intercept_ts):
            principal_event['interceptTimestamp'] = intercept_ts
        
        # Copy required identity fields (per schema: standardIdentity, environment, mode are required)
        for col in ['standardIdentity', 'environment', 'edhControlSet', 'mode']:
            if col in row.index:
                val = row[col]
                # Handle scalar vs array values for notna check
                if isinstance(val, (np.ndarray, list)):
                    if len(val) > 0:
                        try:
                            if not np.isnan(val).all():
                                principal_event[col] = val
                        except (TypeError, ValueError):
                            # Non-numeric arrays (e.g., strings) - just assign
                            principal_event[col] = val
                else:
                    try:
                        if pd.notna(val):
                            principal_event[col] = val
                    except ValueError:
                        # If notna check fails on array-like, assign if not None
                        if val is not None:
                            principal_event[col] = val
        
        # Ensure required fields have defaults if missing
        if 'standardIdentity' not in principal_event:
            principal_event['standardIdentity'] = 'UNKNOWN'
        if 'environment' not in principal_event:
            principal_event['environment'] = 'UNKNOWN'
        if 'mode' not in principal_event:
            principal_event['mode'] = 'LIVE'
        
        # componentTrackEvents - record id (crucibleHeader.uuid) of the source component track event
        principal_event['componentTrackEvents'] = [normalize_uuid(row['crucibleHeader.uuid'])] if pd.notna(row.get('crucibleHeader.uuid')) else []
        principal_tracks.append(principal_event)
    
    return pd.DataFrame(principal_tracks), component_to_principal_associations


def process_with_fusion_batch(event_df: pd.DataFrame,
                              config: Dict[str, Any],
                              kf: 'NumpyPrincipalTrackKalmanFilter',
                              object_id_col: str) -> tuple:
    """Batched (Stage-2) equivalent of :func:`process_with_fusion` for the numpy
    fuser kernel.

    Structurally identical to ``process_with_fusion`` — same per-row track
    management (supersede resolution, associations) and same output event dict —
    but the two hot per-event costs are removed: every column is bulk-extracted
    to a numpy array ONCE (no ``iterrows`` / per-cell ``row.get``), and the
    Kalman step runs through the array-native ``kf._update_core`` instead of
    constructing a StoneSoup ``Detection`` + ``LinearGaussian`` per event
    (``create_measurement`` + ``update_filter``). The caller pre-sorts rows
    ascending by timestamp, so applying them in row order preserves exact
    sequential semantics; ``kf.priors`` and the emitted events are identical to
    the per-row path (verified in tests/unit/test_fuser_batch_equivalence.py).
    """
    n = len(event_df)
    if n == 0:
        return pd.DataFrame(), []

    cols = event_df.columns

    def arr(name):
        return event_df[name].to_numpy(dtype=object) if name in cols else None

    oid_a = event_df[object_id_col].to_numpy(dtype=object)
    tid_a = arr('trackId')
    env_a = arr('environment')
    tot_a = arr('trackOriginatedTimestamp')
    cdate_a = arr('crucibleHeader.createdDate')
    intercept_a = arr('interceptTimestamp')
    uuid_a = arr('crucibleHeader.uuid')
    # Required identity/metadata fields copied straight through (schema requires
    # standardIdentity, environment, mode; edhControlSet carried when present).
    id_cols = [c for c in ('standardIdentity', 'environment', 'edhControlSet', 'mode')
               if c in cols]
    id_arrs = {c: event_df[c].to_numpy(dtype=object) for c in id_cols}

    prepared = kf.extract_measurement_columns(event_df)
    now_str = get_current_timestamp_string()

    principal_tracks = []
    component_to_principal_associations = []

    for i in range(n):
        object_id = oid_a[i]
        component_track_id = tid_a[i] if tid_a is not None else None

        # Skip events for objects deleted via a DELETE management event (their
        # supersede chain resolves to None) — a null-objectId principal record
        # would be stripped and 400.
        if kf.get_principal_id(object_id) is None:
            continue

        row_env = env_a[i] if env_a is not None else None
        row_env = row_env if isinstance(row_env, str) and row_env else None
        principal_track_id, principal_object_id, is_new, is_new_association = (
            kf.get_or_create_principal_track(object_id, component_track_id, environment=row_env))

        kf.set_track_environment(principal_track_id, row_env)

        if is_new_association and component_track_id:
            component_to_principal_associations.append({
                'trackId': component_track_id,
                'associatedPrincipalTrack': principal_track_id
            })

        if component_track_id:
            kf.add_component_track(principal_track_id, component_track_id)

        # Build the measurement from the pre-extracted arrays (no Detection) and
        # run the array-native kernel. vel_var_fallback mirrors
        # kf.create_measurement: env-appropriate, read AFTER set_track_environment.
        vel_var_fallback = kf.get_vel_var_fallback(
            kf.environment_by_track.get(principal_track_id))
        sv, cov, ts = kf.measurement_at(prepared, i, vel_var_fallback)
        if sv is None:
            continue

        post = kf._update_core(principal_track_id, sv, cov, ts)
        if post is None:
            continue

        principal_event = build_principal_state_dict(
            post, principal_track_id, principal_object_id,
            kf.id_field, kf.vel_write_cols)
        _tot = tot_a[i] if tot_a is not None else None
        if _tot is None or (isinstance(_tot, float) and _tot != _tot):
            _tot = (cdate_a[i] if cdate_a is not None else None)
            if _tot is None or (isinstance(_tot, float) and _tot != _tot):
                _tot = now_str
        principal_event['trackOriginatedTimestamp'] = _tot

        if intercept_a is not None:
            intercept_ts = intercept_a[i]
            if pd.notna(intercept_ts):
                principal_event['interceptTimestamp'] = intercept_ts

        for col in id_cols:
            val = id_arrs[col][i]
            if isinstance(val, (np.ndarray, list)):
                if len(val) > 0:
                    try:
                        if not np.isnan(val).all():
                            principal_event[col] = val
                    except (TypeError, ValueError):
                        principal_event[col] = val
            else:
                try:
                    if pd.notna(val):
                        principal_event[col] = val
                except ValueError:
                    if val is not None:
                        principal_event[col] = val

        if 'standardIdentity' not in principal_event:
            principal_event['standardIdentity'] = 'UNKNOWN'
        if 'environment' not in principal_event:
            principal_event['environment'] = 'UNKNOWN'
        if 'mode' not in principal_event:
            principal_event['mode'] = 'LIVE'

        _uuid = uuid_a[i] if uuid_a is not None else None
        principal_event['componentTrackEvents'] = [normalize_uuid(_uuid)] if pd.notna(_uuid) else []
        principal_tracks.append(principal_event)

    return pd.DataFrame(principal_tracks), component_to_principal_associations


def passthrough_to_principal(event_df: pd.DataFrame,
                             config: Dict[str, Any],
                             kf: PrincipalTrackKalmanFilter,
                             object_id_col: str) -> tuple:
    """
    Pass through component track events directly to principal tracks (no fusion).
    
    Args:
        event_df: DataFrame containing component track events
        config: Configuration dictionary
        kf: PrincipalTrackKalmanFilter instance
        object_id_col: Name of the object ID column
        
    Returns:
        tuple: (principal_track_events_df, principal_track_heads_df, associations)
    """

    principal_events = []
    component_to_principal_associations = []
    
    for idx, row in event_df.iterrows():
        object_id = row[object_id_col]
        component_track_id = row.get('trackId')
        
        # Skip events for objects that have been deleted. A DELETE management
        # event maps the objectId to None in the supersede map, so the chain
        # resolves to None. Creating a principal track for it would produce a
        # record with a null objectId (stripped by df_to_formatted_JSON),
        # which the schema rejects with a 400.
        if kf.get_principal_id(object_id) is None:
            continue
        
        # Get or create principal track
        row_env = row.get('environment')
        row_env = row_env if isinstance(row_env, str) and row_env else None
        principal_track_id, principal_object_id, is_new, is_new_association = (
            kf.get_or_create_principal_track(object_id, component_track_id, environment=row_env))

        kf.set_track_environment(principal_track_id, row_env)
        
        # Track new associations for writeback to component track heads
        if is_new_association and component_track_id:
            component_to_principal_associations.append({
                'trackId': component_track_id,
                'associatedPrincipalTrack': principal_track_id
            })
        
        # Track component association
        if component_track_id:
            kf.add_component_track(principal_track_id, component_track_id)
        
        # Build principal track event (passthrough - copy kinematics directly)
        principal_event = {
            'objectId': principal_object_id,
            'trackId': principal_track_id,
            'trackOriginatedTimestamp': row.get('trackOriginatedTimestamp', row.get('crucibleHeader.createdDate', get_current_timestamp_string())),
        }
        
        # Carry source (intercept) time through to the principal track event
        intercept_ts = row.get('interceptTimestamp')
        if pd.notna(intercept_ts):
            principal_event['interceptTimestamp'] = intercept_ts
        
        # Copy all kinematic columns directly
        kinematic_prefixes = ('ecefPosition.', 'ecefVelocity.', 'positionCovariance.', 
                             'velocityCovariance.', 'positionVelocityCovariance.')
        for col in row.index:
            if col.startswith(kinematic_prefixes) and pd.notna(row[col]):
                principal_event[col] = row[col]
        
        # Copy required identity and metadata fields (per schema)
        for col in ['standardIdentity', 'environment', 'edhControlSet', 'mode']:
            if col in row.index:
                val = row[col]
                # Handle scalar vs array values for notna check
                if isinstance(val, (np.ndarray, list)):
                    if len(val) > 0:
                        try:
                            if not np.isnan(val).all():
                                principal_event[col] = val
                        except (TypeError, ValueError):
                            # Non-numeric arrays (e.g., strings) - just assign
                            principal_event[col] = val
                else:
                    try:
                        if pd.notna(val):
                            principal_event[col] = val
                    except ValueError:
                        # If notna check fails on array-like, assign if not None
                        if val is not None:
                            principal_event[col] = val
        
        # Ensure required fields have defaults if missing
        if 'standardIdentity' not in principal_event:
            principal_event['standardIdentity'] = 'UNKNOWN'
        if 'environment' not in principal_event:
            principal_event['environment'] = 'UNKNOWN'
        if 'mode' not in principal_event:
            principal_event['mode'] = 'LIVE'
        
        # componentTrackEvents - array of component track IDs associated with this principal track
        component_track_ids = sorted(kf.component_trackids_by_principal.get(principal_track_id, set()))
        principal_event['componentTrackEvents'] = component_track_ids
        
        principal_events.append(principal_event)
    
    events_df = pd.DataFrame(principal_events)
    
    # Create heads DataFrame - remove fields not in PrincipalTrackHeads schema.
    # trackUpdatedTimestamp must reflect the genuine observation (intercept)
    # time; never fabricate it from the current time. Drop head rows that have
    # no interceptTimestamp instead of emitting a synthesized timestamp.
    heads_df = events_df.copy()
    if 'interceptTimestamp' in heads_df.columns:
        heads_df = heads_df[heads_df['interceptTimestamp'].notna()].copy()
        heads_df['trackUpdatedTimestamp'] = heads_df['interceptTimestamp']
    else:
        heads_df = heads_df.iloc[0:0]
    
    # Remove columns not in PrincipalTrackHeads schema (single drop, errors='ignore'
    # skips any that are absent).
    columns_to_remove = ['componentTrackEvents', 'reportIds', 'interceptTimestamp']
    heads_df = heads_df.drop(columns=columns_to_remove, errors='ignore')
    
    return events_df, heads_df, component_to_principal_associations


# --- Sharded multi-process track fusion ---

NUM_FUSION_WORKERS = 4  # Default parallel fusion workers (override perspective-wide with num_fusion_workers)


def _use_numpy_fusion(config: dict) -> bool:
    """Whether to fuse with the numpy principal-track Kalman kernel instead of
    StoneSoup. The numpy kernel is behaviourally identical (verified in
    tests/unit/test_fuser_kalman_equivalence.py) but far faster.

    Perspective-level: set the ``use_numpy_fusion`` config key (read from the
    perspective's representative config, like ``num_fusion_workers``) or the
    ``CRUCIBLE_USE_NUMPY_FUSION`` environment variable. Defaults to StoneSoup.
    """
    return str(config.get('use_numpy_fusion',
                          os.getenv('CRUCIBLE_USE_NUMPY_FUSION', 'false'))
               ).strip().lower() in ('1', 'true', 'yes', 'on')


def _find_supersede_root(object_id: str, supersede_map: Dict[str, str]) -> str:
    """Follow the supersede chain to find the root (superseding) objectId.
    If object_id is not superseded, it IS the root."""
    visited = set()
    current = object_id
    while current in supersede_map and current not in visited:
        visited.add(current)
        current = supersede_map[current]
    return current


def _shard_for_root(root: str, n_shards: int) -> int:
    """Deterministically map a supersede-group root to a shard index.

    The built-in ``hash()`` is salted per-process (``PYTHONHASHSEED``), so it is
    unstable across restarts and between the dispatcher and worker processes.
    A stable hash guarantees that a given supersede chain always routes to the
    same shard, so principal tracks reloaded on restart stay on their owning
    shard instead of becoming orphaned duplicates.
    """
    digest = hashlib.md5(
        str(root).encode('utf-8'), usedforsecurity=False).hexdigest()
    return int(digest, 16) % n_shards


def _fusion_shard_for_object(object_id: str, supersede_map: Dict[str, str], n_shards: int) -> int:
    """Return the shard that owns an object's supersede-group principal track."""
    if not object_id:
        return 0
    root = _find_supersede_root(object_id, supersede_map)
    return _shard_for_root(root, n_shards)


def _mark_migrated_supersede_roots(old_roots: Dict[str, str], supersede_map: Dict[str, str],
                                   shard_queues: List[Any], n_shards: int) -> None:
    """Notify old-owner shards when a supersede changes an object's root shard."""
    for object_id, old_root in old_roots.items():
        new_root = _find_supersede_root(object_id, supersede_map)
        old_shard = _shard_for_root(old_root, n_shards)
        new_shard = _shard_for_root(new_root, n_shards)
        if old_shard != new_shard:
            shard_queues[old_shard].put({
                'type': 'mark_stale',
                'data': {'objectId': object_id}
            })


def _apply_management_events_to_supersede_map(
        supersede_map: Dict[str, str],
        management_events: pd.DataFrame) -> Dict[str, str]:
    """Return the complete new map after incremental management events.

    The result must replace the caller's prior snapshot. Merging it with
    ``dict.update`` would retain keys removed by RESTORE actions.
    """
    return build_supersede_map(
        management_events, init_supersede_map=supersede_map)


def _run_fusion_shard(shard_queue: Any, config: dict, use_ci: bool, ci_omega: Optional[float], passthrough: bool, shard_idx: int, n_shards: int) -> None:
    """Launch a sharded track fuser worker in its own process."""
    asyncio.run(_fusion_shard_worker(
        shard_queue, config, use_ci, ci_omega, passthrough, shard_idx, n_shards))


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
        limit = int(config.get('fusion_head_preload_limit',
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
                f"heads were dropped. Raise fusion_head_preload_limit / CRUCIBLE_HEAD_PRELOAD_LIMIT.")
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


def _fanout_head_preload(shard_queues: List[Any], config: dict, n_shards: int,
                         supersede_map: Optional[Dict[str, str]] = None) -> None:
    """Download principal + component track heads ONCE and fan them out to the
    shard workers' queues (pre-bucketed to each shard's owned tracks).

    Mirrors the transformer's object-cache fan-out: a single parent-side read
    replaces N independent per-shard queries (which N-duplicated the load and
    504'd on large head datasets). Each shard receives only the heads it owns,
    terminated by a ``done`` marker so it can initialize (even if it owns none).
    """
    if _skip_head_preload(config):
        for s in range(n_shards):
            shard_queues[s].put({'type': 'init_heads', 'principal_heads': [],
                                 'component_heads': [], 'done': True})
        logging.info(
            f"Head preload SKIPPED (skip_head_preload): {n_shards} fusion shard(s) start "
            f"with empty Kalman state; deterministic principal trackIds keep heads consistent.")
        return

    principal_ds = config.get('principal_track_head_dataset')
    component_ds = config.get('component_track_head_dataset')

    supersede_map = supersede_map or {}
    principal_buckets: List[List[dict]] = [[] for _ in range(n_shards)]
    component_buckets: List[List[dict]] = [[] for _ in range(n_shards)]
    trackid_to_shard: Dict[str, int] = {}

    # ---- principal heads: bucket by owning shard; map trackId -> shard ----
    if principal_ds:
        recs = _load_all_records(principal_ds, config)
        if recs:
            pdf = normalize_object_id_columns(utils.flatten_crucible_dataset(recs))
            for row in pdf.to_dict('records'):
                oid = row.get('objectId')
                if oid is None or (isinstance(oid, float) and pd.isna(oid)):
                    continue
                shard = _fusion_shard_for_object(oid, supersede_map, n_shards)
                principal_buckets[shard].append(row)
                tid = row.get('trackId')
                if tid is not None and not (isinstance(tid, float) and pd.isna(tid)):
                    trackid_to_shard[tid] = shard
            logging.info(
                f"Head preload: {sum(len(b) for b in principal_buckets)} principal "
                f"heads bucketed across {n_shards} shards")
            del recs, pdf  # free the raw list + flattened frame once bucketed

    # ---- component heads: route by associatedPrincipalTrack's owning shard ----
    if component_ds:
        recs = _load_all_records(component_ds, config)
        if recs:
            cdf = utils.flatten_crucible_dataset(recs)
            if 'associatedPrincipalTrack' in cdf.columns:
                cdf = cdf[cdf['associatedPrincipalTrack'].notna()]
                for row in cdf.to_dict('records'):
                    shard = trackid_to_shard.get(row.get('associatedPrincipalTrack'))
                    if shard is not None:
                        component_buckets[shard].append(row)
            logging.info(
                f"Head preload: {sum(len(b) for b in component_buckets)} component "
                f"associations routed to shards")
            del recs, cdf  # free the raw list + flattened frame once bucketed

    # ---- fan out chunked init to each shard, ending with a done marker ----
    for s in range(n_shards):
        p, c = principal_buckets[s], component_buckets[s]
        for i in range(0, len(p), _HEAD_FANOUT_CHUNK):
            shard_queues[s].put({'type': 'init_heads',
                                 'principal_heads': p[i:i + _HEAD_FANOUT_CHUNK],
                                 'component_heads': [], 'done': False})
        for i in range(0, len(c), _HEAD_FANOUT_CHUNK):
            shard_queues[s].put({'type': 'init_heads',
                                 'principal_heads': [],
                                 'component_heads': c[i:i + _HEAD_FANOUT_CHUNK],
                                 'done': False})
        shard_queues[s].put({'type': 'init_heads', 'principal_heads': [],
                             'component_heads': [], 'done': True})
        # Release this shard's buckets now that they've been queued, so the
        # parent doesn't hold every shard's heads plus the queued copies at once.
        principal_buckets[s] = []
        component_buckets[s] = []
    logging.info(f"Head preload: fan-out complete to {n_shards} shard(s)")



async def _flush_associations(associations: List[Dict[str, Any]],
                              component_track_head_dataset: str,
                              config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], float]:
    """Write ``associatedPrincipalTrack`` stamps back to component track heads.

    Returns the list to retry (the same raw dicts) and elapsed wall time.
    Association stamps are needed for restart/preload correctness, so failures
    are retried even though the write is scheduled in a background task.
    """
    if not associations or not component_track_head_dataset:
        return [], 0.0

    started_at = time.perf_counter()

    assoc_df = pd.DataFrame(associations)
    assoc_df = assoc_df.groupby('trackId', as_index=False).agg({'associatedPrincipalTrack': 'last'})
    assoc_json = _fast_df_to_nested_json(assoc_df)
    if not assoc_json:
        return [], time.perf_counter() - started_at

    heads_chunk_size = int(config['batch_update_chunk_size'])
    max_concurrent = config.get('batch_write_max_concurrent')
    _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
    rc.token = auth.get_token()
    try:
        failed_records, missing_object_ids = await write_batch_chunked(
            assoc_json, component_track_head_dataset,
            wc.update_entity_record_batch_by_name, heads_chunk_size,
            token_refresher=_refresh_token,
            max_concurrent_writes=max_concurrent,
        )
        retry_track_ids = set()
        for record in failed_records:
            track_id = record.get('trackId')
            if isinstance(track_id, dict):
                track_id = track_id.get('uuid')
            if track_id:
                retry_track_ids.add(str(track_id))
        retry_track_ids.update(str(object_id) for object_id in missing_object_ids)
        if retry_track_ids:
            retry = [assoc for assoc in associations if str(assoc.get('trackId')) in retry_track_ids]
            logging.warning(
                f"Deferred {len(retry)} associatedPrincipalTrack stamps for retry "
                f"({len(failed_records)} failed, {len(missing_object_ids)} missing)")
            return retry, time.perf_counter() - started_at
        logging.info(
            f"Completed associatedPrincipalTrack update for {len(assoc_json)} component track head(s)")
        return [], time.perf_counter() - started_at
    except Exception as assoc_err:
        logging.warning(
            f"Deferred {len(associations)} associatedPrincipalTrack stamps for retry: {assoc_err}")
        return associations, time.perf_counter() - started_at


async def _best_effort_write_principal_heads(records: List[dict],
                                             principal_track_head_dataset: str,
                                             heads_chunk_size: int,
                                             max_concurrent: Optional[int]) -> List[dict]:
    if not records or not principal_track_head_dataset:
        return []
    _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
    try:
        failed_records, missing_ids = await write_batch_chunked(
            records, principal_track_head_dataset,
            wc.write_record_batch_by_name, heads_chunk_size,
            token_refresher=_refresh_token,
            max_concurrent_writes=max_concurrent,
        )
        if failed_records or missing_ids:
            logging.warning(
                f"Deferred best-effort principal track head write(s) to {principal_track_head_dataset}: "
                f"{len(failed_records)} failed, {len(missing_ids)} missing")
        else:
            logging.info(
                f"Completed best-effort principal track head write to {principal_track_head_dataset}: "
                f"records={len(records)}")
        failed_track_ids = {
            str(record.get('trackId'))
            for record in failed_records
            if record.get('trackId') is not None
        }
        missing_ids = {str(object_id) for object_id in missing_ids}
        return [
            record for record in records
            if (record.get('trackId') is not None
                and str(record.get('trackId')) in failed_track_ids)
            or (record.get('objectId') is not None
                and str(record.get('objectId')) in missing_ids)
        ]
    except Exception as head_err:
        logging.warning(
            f"Deferred best-effort principal track head write(s) to "
            f"{principal_track_head_dataset} for retry: {head_err}")
        return records


def _collect_finished_best_effort_tasks(
    tasks: List[Any], label: str,
    pending_by_track: Dict[str, Dict[str, Any]]) -> List[Any]:
    if not tasks:
        return []
    still_pending = []
    completed = 0
    for task in tasks:
        if task.done():
            completed += 1
            try:
                retry_records = task.result() or []
                for record in retry_records:
                    track_id = record.get('trackId')
                    if isinstance(track_id, dict):
                        track_id = track_id.get('uuid')
                    if track_id is not None:
                        pending_by_track[str(track_id)] = record
            except Exception as exc:
                logging.warning(
                    f"Best-effort {label} task failed; records will be retried "
                    f"by the next batch: {exc}")
        else:
            still_pending.append(task)
    if completed:
        logging.info(
            f"Completed {completed} background best-effort {label} task(s); "
            f"still_running={len(still_pending)}")
    return still_pending


def _collect_finished_association_tasks(tasks: List[Any]) -> tuple:
    if not tasks:
        return [], [], 0.0
    still_pending = []
    retry_associations: List[Dict[str, Any]] = []
    update_seconds = 0.0
    completed = 0
    for task in tasks:
        if task.done():
            completed += 1
            try:
                retry, elapsed = task.result()
                retry_associations.extend(retry or [])
                update_seconds += elapsed
            except Exception as exc:
                logging.warning(f"Background associatedPrincipalTrack update task failed: {exc}")
        else:
            still_pending.append(task)
    if completed:
        logging.info(
            f"Completed {completed} background associatedPrincipalTrack update task(s); "
            f"retry_queued={len(retry_associations)} still_running={len(still_pending)}")
    return still_pending, retry_associations, update_seconds


async def _fusion_shard_dispatcher(event_queue: asyncio.Queue, supersede_queue: asyncio.Queue,
                                    shard_queues: List[Any], supersede_map: Dict[str, str]) -> None:
    """
    Dispatcher coroutine that routes events to shard workers by supersede-group hash.
    Objects in the same supersede chain are guaranteed to go to the same shard.
    Also broadcasts supersede map updates to all shards.
    """
    n_shards = len(shard_queues)

    while True:
        # Wait for events
        events = await event_queue.get()

        # Drain entire backlog for better batching
        while not event_queue.empty():
            extra = await event_queue.get()
            if isinstance(extra, list):
                events.extend(extra)
            else:
                events.append(extra)

        # Process supersede updates: update local map and broadcast to all shards
        supersede_events = []
        while not supersede_queue.empty():
            try:
                extra = supersede_queue.get_nowait()
                if isinstance(extra, list):
                    supersede_events.extend(extra)
                else:
                    supersede_events.append(extra)
            except asyncio.QueueEmpty:
                break

        if supersede_events:
            supersede_df = utils.flatten_crucible_dataset(supersede_events)
            supersede_df = normalize_object_id_columns(supersede_df)
            if not supersede_df.empty:
                old_roots = {}
                if 'objectId' in supersede_df.columns:
                    for object_id in supersede_df['objectId'].dropna().unique():
                        old_roots[object_id] = _find_supersede_root(object_id, supersede_map)

                supersede_map = _apply_management_events_to_supersede_map(
                    supersede_map, supersede_df)

                # If a supersede changes an object's owning root, future events
                # route to a different shard. Tell the old shard to drop its
                # stale local principal mapping/state so it cannot retain an
                # orphan principal track for the superseded object.
                _mark_migrated_supersede_roots(old_roots, supersede_map, shard_queues, n_shards)

                # Broadcast the full supersede_map to all shards. Each shard's
                # update_supersede_map() merges any superseded object's component
                # tracks into the surviving principal track and re-wires
                # associations, so no dispatcher-driven track migration is needed.
                for sq in shard_queues:
                    sq.put({'type': 'supersede_update', 'data': dict(supersede_map)})

        # Partition events by supersede-group root hash. The root of a chain is
        # never itself superseded, so a chain's owning shard is stable and its
        # principal track never migrates between shards.
        shard_buckets = [[] for _ in range(n_shards)]
        for event in events:
            obj_id = None
            if isinstance(event, dict):
                obj_id = event.get('objectId', {})
                if isinstance(obj_id, dict):
                    obj_id = obj_id.get('uuid')
            if obj_id:
                shard_idx = _fusion_shard_for_object(obj_id, supersede_map, n_shards)
            else:
                shard_idx = 0
            shard_buckets[shard_idx].append(event)

        # Push to shard queues
        for i, bucket in enumerate(shard_buckets):
            if bucket:
                shard_queues[i].put({'type': 'events', 'data': bucket})


async def _fusion_shard_worker(shard_queue: Any, config: dict, use_ci: bool, ci_omega: Optional[float], passthrough: bool,
                               shard_idx: int, n_shards: int) -> None:
    """
    Async worker for a single fusion shard. Processes events from shard_queue.
    Each shard maintains its own PrincipalTrackKalmanFilter for its subset of objects.

    Messages on the queue are dicts with keys:
            - 'type': 'events' | 'supersede_update' | 'mark_stale'
            - 'data': list of event dicts, updated supersede_map dict, or stale objectId marker
    """
    initialize_controllers()

    if '_log_level' in config:
        log_utils.get_logger(log_type='transformer', log_level=config['_log_level'])

    principal_track_event_dataset = config.get('principal_track_event_dataset')
    principal_track_head_dataset = config.get('principal_track_head_dataset')
    component_track_head_dataset = config.get('component_track_head_dataset')

    # Initialize Kalman filter manager for this shard. The numpy kernel is a
    # behaviourally-identical, faster drop-in selected by the perspective-level
    # use_numpy_fusion config key.
    _fusion_cls = (NumpyPrincipalTrackKalmanFilter if _use_numpy_fusion(config)
                   else PrincipalTrackKalmanFilter)
    kf = _fusion_cls(
        config, id_field='objectId',
        vel_read_cols=('ecefVelocity.dx', 'ecefVelocity.dy', 'ecefVelocity.dz'),
        vel_write_cols=('ecefVelocity.dx', 'ecefVelocity.dy', 'ecefVelocity.dz'),
        vel_read_fallback=('ecefVelocity.x', 'ecefVelocity.y', 'ecefVelocity.z'),
    )

    # Load existing principal track heads and component associations so that
    # restarts reuse the same trackIds instead of creating orphan duplicates.
    # These are downloaded ONCE by the parent (run()) and fanned out to this
    # shard's queue as 'init_heads' messages (pre-bucketed to only the tracks
    # this shard owns) — instead of every shard independently querying Crucible,
    # which N-duplicated the read load and 504'd on large head datasets.
    # Accumulate the fan-out chunks until the 'done' marker; stash any other
    # message that happens to arrive meanwhile (handled after init).
    principal_head_records: List[dict] = []
    component_head_records: List[dict] = []
    stashed_msgs: List[Any] = []
    _init_done = False
    while not _init_done:
        msg = shard_queue.get()
        if isinstance(msg, dict) and msg.get('type') == 'init_heads':
            principal_head_records.extend(msg.get('principal_heads') or [])
            component_head_records.extend(msg.get('component_heads') or [])
            if msg.get('done'):
                _init_done = True
        elif msg is not None:
            stashed_msgs.append(msg)

    principal_heads_df_init = pd.DataFrame()
    component_heads_df_init = pd.DataFrame()
    if principal_head_records:
        # Records are already flattened + objectId-normalized by the parent.
        principal_heads_df_init = pd.DataFrame(principal_head_records)
    if component_head_records:
        component_heads_df_init = pd.DataFrame(component_head_records)
    logging.info(
        f"Shard {shard_idx}: initialized from fan-out with "
        f"{len(principal_heads_df_init)} principal heads, "
        f"{len(component_heads_df_init)} component associations")

    kf.initialize({}, principal_heads_df_init, component_heads_df_init)
    if use_ci:
        kf.set_fusion_method('ci', omega=ci_omega)

    pending_associations: List[Dict[str, Any]] = []
    pending_principal_head_tasks: List[Any] = []
    pending_principal_heads_by_track: Dict[str, Dict[str, Any]] = {}
    pending_association_tasks: List[Any] = []
    completed_assoc_update_seconds = 0.0
    principal_head_update_emit_time_by_track: Dict[str, float] = {}
    heads_chunk_size = int(config['batch_update_chunk_size'])
    max_concurrent = config.get('batch_write_max_concurrent')

    # Any non-init messages received while accumulating the init_heads fan-out
    # (e.g. the initial supersede broadcast) are handled on the first loop pass.
    startup_msgs: List[Any] = stashed_msgs

    while True:
        try:
            if startup_msgs:
                messages = startup_msgs
                startup_msgs = []
            else:
                # Blocking get on multiprocessing Queue
                msg = shard_queue.get()
                if msg is None:
                    continue

                # Drain backlog
                messages = [msg]
                while not shard_queue.empty():
                    try:
                        extra = shard_queue.get_nowait()
                        if extra is not None:
                            messages.append(extra)
                    except Exception:
                        break

            pending_principal_head_tasks = _collect_finished_best_effort_tasks(
                pending_principal_head_tasks, 'principal-head write',
                pending_principal_heads_by_track)
            pending_association_tasks, retry_associations, update_seconds = _collect_finished_association_tasks(
                pending_association_tasks)
            completed_assoc_update_seconds += update_seconds
            pending_associations.extend(retry_associations)
            if not pending_principal_head_tasks and pending_principal_heads_by_track:
                coalesced_heads = list(pending_principal_heads_by_track.values())
                pending_principal_heads_by_track.clear()
                logging.info(
                    f"Scheduled coalesced background principal track head write to "
                    f"{principal_track_head_dataset}: records={len(coalesced_heads)}")
                pending_principal_head_tasks.append(asyncio.create_task(
                    _best_effort_write_principal_heads(
                        coalesced_heads, principal_track_head_dataset,
                        heads_chunk_size, max_concurrent)))
            if (pending_associations and component_track_head_dataset
                    and not pending_association_tasks):
                associations_to_write = pending_associations
                pending_associations = []
                logging.info(
                    f"Scheduled coalesced background associatedPrincipalTrack update for "
                    f"{len(associations_to_write)} component track head(s)")
                pending_association_tasks.append(asyncio.create_task(
                    _flush_associations(
                        associations_to_write, component_track_head_dataset, config)))

            # Process supersede updates first, then events
            all_events = []
            stale_object_ids = []
            for m in messages:
                if m.get('type') == 'supersede_update':
                    updated_map = m['data']
                    # Re-associations from merging superseded component tracks
                    # into surviving principal tracks must be persisted; queue
                    # them so they flush with the next write (even if this batch
                    # carries no track events).
                    re_associations = kf.update_supersede_map(updated_map)
                    if re_associations:
                        pending_associations.extend(re_associations)
                elif m.get('type') == 'mark_stale':
                    stale_object_ids.append(m['data'].get('objectId'))
                elif m.get('type') == 'events':
                    all_events.extend(m['data'])

            if stale_object_ids:
                for object_id in stale_object_ids:
                    ptid = kf.id_to_principal_trackid.get(object_id)
                    if ptid:
                        kf.id_to_principal_trackid.pop(object_id, None)
                        kf.priors.pop(ptid, None)
                        kf.predictor.pop(ptid, None)
                        kf.updater.pop(ptid, None)
                        kf.measurement_model.pop(ptid, None)
                        logging.info(f"Shard {shard_idx}: removed stale principal track {ptid} for migrated object {object_id}")

            if not all_events:
                # No track events, but flush any pending re-associations so
                # supersede merges are persisted promptly.
                if (pending_associations and component_track_head_dataset
                        and not pending_association_tasks):
                    associations_to_write = pending_associations
                    pending_associations = []
                    logging.info(
                        f"Scheduled background associatedPrincipalTrack update for "
                        f"{len(associations_to_write)} component track head(s)")
                    pending_association_tasks.append(asyncio.create_task(
                        _flush_associations(
                            associations_to_write, component_track_head_dataset, config)))
                continue

            _perf = time.perf_counter if PERF_TIMING else _perf_disabled
            _t0 = _perf(); _t = _t0; _timings = {}
            event_df = utils.flatten_crucible_dataset(all_events)
            event_df = normalize_object_id_columns(event_df)

            if event_df.empty:
                continue

            if 'objectId' in event_df.columns:
                object_id_col = 'objectId'
            else:
                logging.error("No objectId column found in shard event data")
                continue

            # Sort by timestamp
            timestamp_col = None
            for col in ['interceptTimestamp', 'trackUpdatedTimestamp', 'crucibleHeader.createdDate']:
                if col in event_df.columns:
                    timestamp_col = col
                    break
            if timestamp_col:
                event_df = event_df.sort_values(by=timestamp_col, ignore_index=True, ascending=True)

            # Process events
            component_to_principal_associations = list(pending_associations)
            pending_associations = []

            if passthrough:
                principal_events_df, principal_heads_df, new_associations = passthrough_to_principal(
                    event_df, config, kf, object_id_col
                )
                component_to_principal_associations.extend(new_associations)
            else:
                _fuse = (process_with_fusion_batch
                         if isinstance(kf, NumpyPrincipalTrackKalmanFilter)
                         else process_with_fusion)
                principal_events_df, new_associations = _fuse(
                    event_df, config, kf, object_id_col
                )
                component_to_principal_associations.extend(new_associations)
                principal_heads_df = principal_events_df.copy()
                # trackUpdatedTimestamp must reflect the genuine observation
                # (intercept) time; never fabricate it from the current time. Drop
                # head rows with no interceptTimestamp instead of synthesizing one.
                if 'interceptTimestamp' in principal_heads_df.columns:
                    principal_heads_df = principal_heads_df[principal_heads_df['interceptTimestamp'].notna()].copy()
                    principal_heads_df['trackUpdatedTimestamp'] = principal_heads_df['interceptTimestamp']
                else:
                    principal_heads_df = principal_heads_df.iloc[0:0]
                columns_to_remove = ['componentTrackEvents', 'reportIds', 'interceptTimestamp']
                principal_heads_df = principal_heads_df.drop(columns=columns_to_remove, errors='ignore')

            if principal_events_df.empty:
                continue

            # Mark new vs existing tracks
            principal_heads_df['isNewTrack'] = principal_heads_df['trackId'].isin(kf.new_principal_trackids)
            principal_heads_df = principal_heads_df.groupby('trackId', as_index=False).agg('last')

            new_heads = principal_heads_df[principal_heads_df['isNewTrack']]
            existing_heads = principal_heads_df[~principal_heads_df['isNewTrack']]
            new_heads = new_heads.drop(columns=['isNewTrack'], errors='ignore')
            existing_heads = existing_heads.drop(columns=['isNewTrack'], errors='ignore')

            columns_to_drop = ['stale', 'componentTrackEvents', 'componentTrackIds', 'interceptTimestamp', 'reportIds']
            new_heads = new_heads.drop(columns=columns_to_drop, errors='ignore')
            existing_heads = existing_heads.drop(columns=columns_to_drop, errors='ignore')

            _timings['process'] = _perf() - _t
            # Write to Crucible
            rc.token = auth.get_token()
            _refresh_token = lambda: setattr(wc, 'token', auth.get_token())

            chunk_size = int(config['batch_write_chunk_size'])

            # Write principal track events
            _t_we = _perf()
            events_copy = principal_events_df.copy()
            while not events_copy.empty:
                unique_df = events_copy.drop_duplicates(subset=['objectId'], keep='first')
                events_json = _fast_df_to_nested_json(unique_df)
                if events_json:
                    await write_batch_chunked(
                        events_json, principal_track_event_dataset,
                        wc.write_record_batch_by_name, chunk_size,
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent,
                    )
                events_copy = events_copy[~events_copy['objectId'].isin(unique_df['objectId'])]

            _timings['write_events'] = _perf() - _t_we
            # Write new track heads
            _t_wh = _perf()
            failed_new_head_ids = set()
            if not new_heads.empty:
                new_heads_json = _fast_df_to_nested_json(new_heads)
                if new_heads_json:
                    failed_records, _ = await write_batch_chunked(
                        new_heads_json, principal_track_head_dataset,
                        wc.write_record_batch_by_name, heads_chunk_size,
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent,
                    )
                    for rec in failed_records:
                        tid = rec.get('trackId')
                        if tid:
                            failed_new_head_ids.add(tid)
                successfully_written = set(new_heads['trackId'].tolist())
                successfully_written -= failed_new_head_ids
                kf.new_principal_trackids -= successfully_written
                if failed_new_head_ids:
                    logging.warning(
                        f"{len(failed_new_head_ids)} new principal track head(s) failed; keeping for retry")

            # Existing principal head writes are best-effort. Principal track
            # events are the reliable live stream; heads support preload/query
            # state and should not block fuser progress on a contended dataset.
            if not existing_heads.empty:
                existing_heads = _coalesce_existing_head_updates(
                    existing_heads,
                    principal_head_update_emit_time_by_track,
                    _head_update_interval_seconds(config),
                    'principal')
                existing_heads_json = _fast_df_to_nested_json(existing_heads)
                if existing_heads_json:
                    for head in existing_heads_json:
                        track_id = head.get('trackId')
                        if isinstance(track_id, dict):
                            track_id = track_id.get('uuid')
                        if track_id is not None:
                            pending_principal_heads_by_track[str(track_id)] = head
                    if not pending_principal_head_tasks and pending_principal_heads_by_track:
                        coalesced_heads = list(pending_principal_heads_by_track.values())
                        pending_principal_heads_by_track.clear()
                        logging.info(
                            f"Scheduled background principal track head write to "
                            f"{principal_track_head_dataset}: records={len(coalesced_heads)}")
                        pending_principal_head_tasks.append(asyncio.create_task(
                            _best_effort_write_principal_heads(
                                coalesced_heads, principal_track_head_dataset,
                                heads_chunk_size, max_concurrent)))

            _timings['write_heads'] = _perf() - _t_wh
            # Write associatedPrincipalTrack back to component track heads
            _t_wa = _perf()
            if component_to_principal_associations and component_track_head_dataset:
                pending_associations.extend(component_to_principal_associations)
                if not pending_association_tasks:
                    associations_to_write = pending_associations
                    pending_associations = []
                    logging.info(
                        f"Scheduled background associatedPrincipalTrack update for "
                        f"{len(associations_to_write)} component track head(s)")
                    pending_association_tasks.append(asyncio.create_task(
                        _flush_associations(
                            associations_to_write, component_track_head_dataset, config)))

            _timings['write_assoc'] = _perf() - _t_wa
            if PERF_TIMING:
                _timings['update_assoc'] = completed_assoc_update_seconds
                completed_assoc_update_seconds = 0.0
                _timings['total'] = _perf() - _t0
                _timings['other'] = _timings['total'] - (
                    _timings.get('process', 0.0)
                    + _timings.get('write_events', 0.0)
                    + _timings.get('write_heads', 0.0)
                    + _timings.get('write_assoc', 0.0))
                logging.info(
                    f" [PERF fuser] shard={shard_idx} "
                    f"total={_timings['total']:.3f}s "
                    f"process={_timings.get('process', 0.0):.3f}s "
                    f"write_events={_timings.get('write_events', 0.0):.3f}s "
                    f"write_heads={_timings.get('write_heads', 0.0):.3f}s "
                    f"write_assoc={_timings.get('write_assoc', 0.0):.3f}s "
                    f"update_assoc={_timings.get('update_assoc', 0.0):.3f}s "
                    f"other={_timings['other']:.3f}s "
                    f"| event_rows={len(event_df)} principal_events={len(principal_events_df)} "
                    f"principal_heads={len(principal_heads_df)}")
            logging.info(f"Shard processed {len(event_df)} events -> {len(principal_events_df)} principal events")

        except Exception as e:
            logging.error(f"Error in fusion shard worker: {e}")
            logging.error(traceback.format_exc())
            continue


def run(stream_manager_perspective: str, log_level: Optional[str] = None, *,
        use_ci: bool = True, ci_omega: Optional[float] = None,
        passthrough: bool = False) -> None:
    """
    Main function to run the track fusion process.
    
    Args:
        stream_manager_perspective: Perspective name for Stream Manager (e.g., 'Live_POV')
        log_level: Logging level (INFO, DEBUG, WARN, ERROR)
        use_ci: If True, use Covariance Intersection instead of standard Kalman
        ci_omega: Fixed CI weight in (0, 1). None = auto-optimize.
                  ~0 = measurement-dominated, ~0.5 = balanced, ~1 = prediction-dominated.
        passthrough: If True, skip Kalman filtering and pass through kinematics directly.
    """
    # Set up logging
    if log_level is None:
        log_level = 'INFO'
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    # Register signal handlers
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    
    logging.info(f"Starting Track Fusion for perspective: {stream_manager_perspective}")
    
    # Initialize controllers
    initialize_controllers()
    
    # Load configurations
    config_list = find_and_validate_configs(stream_manager_perspective, include_scripts=False, rc_instance=rc,)
    
    if not config_list:
        logging.error(f"No configuration found for perspective: {stream_manager_perspective}")
        return
    
    # Find first valid config with required datasets
    config = None
    for cfg in config_list:
        if cfg.get('disabled'):
            continue
        if not cfg.get('component_track_event_dataset'):
            continue
        if not cfg.get('principal_track_event_dataset') or not cfg.get('principal_track_head_dataset'):
            logging.warning(f"Config missing principal track datasets - skipping")
            continue
        config = cfg
        break
    
    if not config:
        logging.error("No valid configuration found for track fusion")
        return
    

    component_track_event_dataset = config.get('component_track_event_dataset')
    
    object_management_event_dataset = config.get('object_management_event_dataset')
    
    logging.info(f"Starting track fusion, listening to {component_track_event_dataset}")

    # Store log level for child processes
    config['_log_level'] = numeric_level

    # Load initial supersede map (needed for routing decisions in dispatcher)
    supersede_map = get_supersede_map(config, rc_instance=rc, auth_instance=auth)
    logging.info(f"Loaded {len(supersede_map)} supersede mappings for routing")

    # Spawn sharded worker processes
    n_shards = int(config.get('num_fusion_workers', NUM_FUSION_WORKERS))
    # num_fusion_workers is a PERSPECTIVE-level setting: exactly one fuser runs
    # per perspective and reads this value from the first valid config. It is
    # stored per-feed, so warn if feed configs disagree — otherwise an arbitrary
    # config's value would silently win.
    _fusion_vals = set()
    for _c in config_list:
        if _c.get('disabled'):
            continue
        _v = _c.get('num_fusion_workers')
        if _v is None:
            continue
        try:
            _fusion_vals.add(int(_v))
        except (TypeError, ValueError):
            pass
    if len(_fusion_vals) > 1:
        logging.warning(
            f"num_fusion_workers differs across {stream_manager_perspective} feed configs "
            f"{sorted(_fusion_vals)}; using {n_shards} (from the first valid config). "
            f"num_fusion_workers is perspective-level — set it identically in every config.")

    # use_numpy_fusion is perspective-level too (each shard reads it from the same
    # representative config); warn if feed configs disagree, then log the backend.
    use_numpy_fusion = _use_numpy_fusion(config)
    _numpy_vals = {_use_numpy_fusion(_c) for _c in config_list
                   if not _c.get('disabled') and 'use_numpy_fusion' in _c}
    if len(_numpy_vals) > 1:
        logging.warning(
            f"use_numpy_fusion differs across {stream_manager_perspective} feed configs; "
            f"using {use_numpy_fusion} (from the first valid config). "
            f"use_numpy_fusion is perspective-level — set it identically in every config.")
    logging.info(f"Track fusion Kalman backend = {'numpy' if use_numpy_fusion else 'stonesoup'}")
    shard_queues = [MPQueue(maxsize=_SHARD_QUEUE_MAXSIZE) for _ in range(n_shards)]
    processes = []

    for shard_idx in range(n_shards):
        proc = Process(
            target=_run_fusion_shard,
            args=(shard_queues[shard_idx], config, use_ci, ci_omega, passthrough,
                  shard_idx, n_shards),
            daemon=True
        )
        proc.start()
        processes.append(proc)

    logging.info(f"Started {n_shards} fusion shard workers")

    # Download principal + component track heads ONCE and fan them out to the
    # shard workers (pre-bucketed) so each shard initializes from its queue
    # instead of independently querying Crucible. Must run BEFORE the supersede
    # broadcast / event dispatch so heads are consumed first.
    _fanout_head_preload(shard_queues, config, n_shards, supersede_map)

    # Broadcast the initial supersede map to all shard workers so they can
    # correctly fuse superseded objects from the first batch onward.
    if supersede_map:
        for sq in shard_queues:
            sq.put({'type': 'supersede_update', 'data': dict(supersede_map)})
        logging.info(f"Broadcast initial supersede map ({len(supersede_map)} entries) to {n_shards} shards")

    # Run SSE listeners + dispatcher in the main event loop
    async def run_async() -> None:
        event_queue = asyncio.Queue()
        supersede_queue = asyncio.Queue()
        
        coroutines = []
        
        # SSE listener for component track events
        sql_query = f"SELECT * FROM {component_track_event_dataset}"
        logging.info(f"Track Fusion SSE Listener SQL Query: {sql_query}")
        coroutines.append(SSE_listener(sql_query, event_queue, auth))
        
        # SSE listener for supersede/delete/restore management events
        if object_management_event_dataset:
            supersede_sql_query = (
                f"SELECT * FROM {object_management_event_dataset}"
                " WHERE action IN ('SUPERSEDE', 'DELETE', 'RESTORE')"
            )
            logging.info(f"Track Fusion Supersede SSE Listener SQL Query: {supersede_sql_query}")
            coroutines.append(SSE_listener(supersede_sql_query, supersede_queue, auth))
        else:
            logging.warning("No object_management_event_dataset configured; supersede SSE listener not started")
        
        # Dispatcher: routes events and supersede updates to shard workers
        coroutines.append(
            _fusion_shard_dispatcher(event_queue, supersede_queue, shard_queues, supersede_map)
        )
        
        await asyncio.gather(*coroutines)
    
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    except BaseException:
        logging.critical("Track Fusion event loop exited unexpectedly")
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
    parser = argparse.ArgumentParser(description='Track Fusion for Principal Track Creation')
    parser.add_argument('stream_manager_perspective',
                        help='Name of perspective in Stream Manager Configuration (e.g., Live_POV)')
    parser.add_argument('--log',
                        help='Logging level: INFO (default), WARN, ERROR, or DEBUG',
                        default='INFO')
    parser.add_argument('--no-ci', action='store_true',
                        help='Disable Covariance Intersection; use standard Kalman update instead')
    parser.add_argument('--ci-omega', type=float, default=None,
                        help='Fixed CI weight in (0,1). ~0=measurement-dominated, ~0.5=balanced, ~1=prediction-dominated. Omit for auto-optimization.')
    parser.add_argument('--passthrough', action='store_true',
                        help='Skip Kalman filtering, pass through kinematics directly')
    args = parser.parse_args()
    
    run(args.stream_manager_perspective, log_level=args.log,
        use_ci=not args.no_ci, ci_omega=args.ci_omega, passthrough=args.passthrough)
