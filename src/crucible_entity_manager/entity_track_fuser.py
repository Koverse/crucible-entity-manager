#!/usr/bin/env python
"""
Track Fusion for Principal Track Creation (Entity Pipeline)

This script creates a track fusion system that:
1. Reads component track events from entity_tracker.py output
2. For non-superseded component tracks: passes kinematics directly to principal track events/heads
3. For superseded component tracks (per entity_duplicate_identifier.py): fuses kinematics into the superseding component track's principal track

Usage:
    python entity_track_fuser.py <Stream Manager perspective> --log=DEBUG
    e.g., python entity_track_fuser.py Blue --log=DEBUG

The script will:
- Load configuration from Entity_Stream_Manager_Configurations dataset
- Listen for component track events via SSE
- Check supersede map for component tracks that should be fused
- For non-superseded component tracks: pass through kinematics directly
- For superseded component tracks: fuse kinematics into the superseding component track's principal track
- Write principal track events and heads to their respective datasets

Principal Track Logic:
- Each component track initially gets its own principal track
- When component tracks are superseded (kinematic duplicates), they are fused into the superseding component track's principal track
- Principal tracks are keyed by component trackId
"""
from multiprocessing import Process, Queue, set_start_method
from typing import Dict, Any, Optional, Callable, List
import traceback
import argparse
import asyncio
import logging
import signal
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from entity_utils import (
        ensure_api_controllers, find_and_validate_configs, terminate,
        get_supersede_map, write_batch_chunked, SSE_listener,
        buffer_latest_records_by_track, coalesce_existing_head_updates,
        get_current_timestamp_string, last_record_by_track,
        load_all_records_for_preload, skip_head_preload, stable_shard,
        best_effort_update_track_records, collect_finished_track_update_tasks,
    )
except ImportError:
    from .entity_utils import (
        ensure_api_controllers, find_and_validate_configs, terminate,
        get_supersede_map, write_batch_chunked, SSE_listener,
        buffer_latest_records_by_track, coalesce_existing_head_updates,
        get_current_timestamp_string, last_record_by_track,
        load_all_records_for_preload, skip_head_preload, stable_shard,
        best_effort_update_track_records, collect_finished_track_update_tasks,
    )

try:
    from entity_fusion_filter import EntityPrincipalTrackFilter
    from entity_transformer_records import (
        MISSING, add_track_wgs84_kinematics, clone_record, compact_record,
        get_value, set_value)
except ImportError:
    from .entity_fusion_filter import EntityPrincipalTrackFilter
    from .entity_transformer_records import (
        MISSING, add_track_wgs84_kinematics, clone_record, compact_record,
        get_value, set_value)

# Global variables for controllers (will be initialized in the child process)
rc = wc = auth = None
_DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS = float(
    os.getenv('CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS', '15'))


def head_update_interval_seconds(config: Dict[str, Any]) -> float:
    value = config.get('head_update_interval_seconds',
                       _DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS


def _value(record: dict, path: str, default=None):
    value = get_value(record, path)
    return default if value is MISSING else value


def _id_value(record: dict, path: str):
    value = _value(record, path)
    if isinstance(value, dict):
        value = value.get('uuid')
    return value


try:
    DEFAULT_WRITE_BATCH_SIZE = int(os.getenv('PRINCIPAL_TRACK_WRITE_BATCH_SIZE', '250'))
    if DEFAULT_WRITE_BATCH_SIZE <= 0:
        raise ValueError
except ValueError:
    DEFAULT_WRITE_BATCH_SIZE = 250


def resolve_batch_size(config: Dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key)
    if value is None:
        return fallback
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except (TypeError, ValueError):
        logging.warning(f"Invalid batch size {value} for {key}; using {fallback}")
        return fallback


async def write_records_in_batches(
    records: List[dict],
    dataset_name: str,
    write_func: Callable[[str, Any], Any],
    batch_size: int,
    *,
    token_refresher=None,
    max_concurrent_writes: int = None,
) -> None:
    if not records:
        return
    json_list = [compact_record(record) for record in records]
    json_list = [record for record in json_list if record]
    if not json_list:
        return
    await write_batch_chunked(
        json_list,
        dataset_name,
        write_func,
        batch_size,
        token_refresher=token_refresher,
        max_concurrent_writes=max_concurrent_writes,
    )


async def flush_component_associations(
    associations: Dict[str, str],
    component_track_head_dataset: str,
    batch_size: int,
    *,
    token_refresher=None,
    max_concurrent_writes: int = None,
) -> Dict[str, str]:
    """Write associatedPrincipalTrack stamps and return associations to retry.

    Association stamps are needed for restart/preload correctness, so failures
    are retried even though the write is scheduled in a background task.
    """
    if not associations or not component_track_head_dataset:
        return {}

    assoc_json = [
        {'trackId': tid, 'associatedPrincipalTrack': pid}
        for tid, pid in associations.items()
    ]
    if not assoc_json:
        return {}

    try:
        failed_records, dropped_assoc_ids = await write_batch_chunked(
            assoc_json,
            component_track_head_dataset,
            wc.update_entity_record_batch_by_name,
            batch_size,
            response_mode='entity_update',
            token_refresher=token_refresher,
            max_concurrent_writes=max_concurrent_writes,
        )
    except Exception as assoc_err:
        logging.warning(
            f"Deferred {len(associations)} component-principal association(s) "
            f"for retry: {assoc_err}")
        return associations

    failed_assoc_ids = set()
    for record in failed_records:
        track_id = record.get('trackId')
        if isinstance(track_id, dict):
            track_id = track_id.get('uuid')
        if track_id:
            failed_assoc_ids.add(str(track_id))
    failed_assoc_ids.update(str(track_id) for track_id in dropped_assoc_ids)
    retry = {tid: associations[tid] for tid in failed_assoc_ids if tid in associations}
    if retry:
        logging.warning(
            f"{len(retry)} component-principal association(s) deferred "
            f"({len(failed_records)} failed, {len(dropped_assoc_ids)} missing); will retry next cycle")
    else:
        logging.info(
            f"Completed associatedPrincipalTrack update for {len(assoc_json)} component track head(s)")
    return retry


async def best_effort_write_principal_heads(
    records: List[dict],
    principal_head_dataset: str,
    batch_size: int,
    *,
    token_refresher=None,
    max_concurrent_writes: int = None,
) -> List[dict]:
    return await best_effort_update_track_records(
        records, principal_head_dataset, batch_size,
        wc.update_entity_record_batch_by_name, write_batch_chunked,
        label='principal-head best-effort: ',
        log_prefix=f'[{principal_head_dataset}]: ',
        token_refresher=token_refresher,
        max_concurrent_writes=max_concurrent_writes,
    )


async def cleanup_restored_identity(
    kf: EntityPrincipalTrackFilter,
    identity_store: Dict[str, Dict[str, Any]],
    head_tasks: list,
    pending_heads: Dict[str, Dict[str, Any]],
    head_emit_times: Dict[str, float],
) -> None:
    restored_ids = kf.recently_restored_principal_trackids
    if not restored_ids:
        return

    # Thread-backed writes must finish before a replacement can clear old identity.
    if head_tasks:
        await asyncio.gather(*head_tasks, return_exceptions=True)
        collect_finished_track_update_tasks(
            head_tasks, 'principal-head write', pending_heads)
        head_tasks.clear()
    for principal_id in restored_ids:
        identity_store.pop(principal_id, None)
        pending_heads.pop(principal_id, None)
        head_emit_times.pop(principal_id, None)
        kf.known_principal_trackids.discard(principal_id)


def collect_finished_association_tasks(tasks: list) -> tuple:
    if not tasks:
        return [], {}
    still_pending = []
    retry_associations: Dict[str, str] = {}
    completed = 0
    for task in tasks:
        if task.done():
            completed += 1
            try:
                retry_associations.update(task.result() or {})
            except Exception as exc:
                logging.warning(f"Background associatedPrincipalTrack update task failed: {exc}")
        else:
            still_pending.append(task)
    if completed:
        logging.info(
            f"Completed {completed} background associatedPrincipalTrack update task(s); "
            f"retry_queued={len(retry_associations)} still_running={len(still_pending)}")
    return still_pending, retry_associations

def fuse_track_identity(identity_store: Dict[str, Dict[str, Any]],
                        principal_track_id: str,
                        component_track: dict,
                        is_superseded: bool = False) -> Dict[str, Any]:
    """
    Fuse identity.* fields for a principal track across all of its component
    tracks (including superseded duplicates merged in via the supersede map).

    The principal track's identity is the UNION of non-null identity.* values
    observed across its component tracks. Conflict resolution: a SUPERSEDED
    entity's identity takes precedence over the surviving (superseding) entity's
    identity. Among sources of equal precedence the most-recent non-null value
    wins (component tracks are processed in ascending interceptTimestamp order).

    Args:
        identity_store: Accumulator keyed by principal_track_id. Each entry maps
            an identity.* column to {'value': v, 'rank': r}; a higher rank wins.
        principal_track_id: The principal track being fused.
        component_track: A single component track event row.
        is_superseded: True when this component track's trackId has been
            superseded (i.e., it is a duplicate merged into the surviving
            principal track). Superseded sources outrank the surviving track.

    Returns:
        The accumulated fused identity dict ({identity.col: value}) for this
        principal track.
    """
    store = identity_store.setdefault(str(principal_track_id), {})
    rank = 1 if is_superseded else 0
    identity = _value(component_track, 'identity', {})
    if not isinstance(identity, dict):
        return {col: entry['value'] for col, entry in store.items()}
    for field, val in identity.items():
        if val is None or val == '':
            continue
        col = f'identity.{field}'
        existing = store.get(col)
        # Overwrite only when the new source has equal-or-higher precedence;
        # equal precedence -> most-recent wins (ascending timestamp order).
        if existing is None or rank >= existing['rank']:
            store[col] = {'value': val, 'rank': rank}
    return {col: entry['value'] for col, entry in store.items()}


def _copy_track_fields(component_track: dict, principal_track_id: str,
                       *, head: bool) -> Dict[str, Any]:
    record = {
        'trackId': principal_track_id,
        'standardIdentity': _value(component_track, 'standardIdentity'),
        'environment': _value(component_track, 'environment'),
        'trackOriginatedTimestamp': _value(component_track, 'trackOriginatedTimestamp'),
        'edhControlSet': clone_record(
            {'value': _value(component_track, 'edhControlSet')})['value'],
        'mode': _value(component_track, 'mode'),
    }
    if head:
        record['trackUpdatedTimestamp'] = _value(component_track, 'interceptTimestamp')
        record['stale'] = get_current_timestamp_string(10)
    else:
        record['trackQuality'] = _value(component_track, 'trackQuality')
        record['interceptTimestamp'] = _value(component_track, 'interceptTimestamp')
    for path in (
        'geodetic', 'ecefPosition', 'ecefVelocity', 'positionCovariance',
        'velocityCovariance', 'positionVelocityCovariance', 'uncertainty',
        'identity', 'speed', 'heading'):
        value = _value(component_track, path, MISSING)
        if value is not MISSING:
            set_value(record, path, clone_record({'value': value})['value'])
    return compact_record(record)


def create_principal_track_event(component_track: dict, principal_track_id: str) -> Dict[str, Any]:
    """
    Create a principal track event from a component track.
    
    Args:
        component_track: Component track data
        principal_track_id: Principal track ID
        
    Returns:
        Dict containing principal track event data
    """
    return _copy_track_fields(component_track, principal_track_id, head=False)

def create_principal_track_head(component_track: dict, principal_track_id: str) -> Dict[str, Any]:
    """
    Create a principal track head from a component track.
    
    Args:
        component_track: Component track data
        principal_track_id: Principal track ID
        
    Returns:
        Dict containing principal track head data
    """
    return _copy_track_fields(component_track, principal_track_id, head=True)

async def download_existing_principal_tracks_and_associations(config):
    """
    Download and display existing principal tracks and component associations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (principal_tracks_df, component_associations_df)
    """
    try:
        logging.info("Downloading existing principal tracks and component associations...")
        
        principal_tracks = []
        component_associations = []
        
        # Download existing principal track heads
        if 'principal_track_head_dataset' in config:
            principal_heads = await asyncio.to_thread(
                rc.search,
                f"SELECT * FROM {config['principal_track_head_dataset']} LIMIT 10000"
            )
            
            if principal_heads:
                principal_tracks = principal_heads
                track_ids = {_value(record, 'trackId') for record in principal_tracks}
                track_ids.discard(None)
                logging.info(
                    f"Downloaded {len(principal_tracks)} principal track heads "
                    f"covering {len(track_ids)} unique tracks")
        
        # Download existing component track heads with associations
        if 'component_track_head_dataset' in config:
            component_heads = await asyncio.to_thread(
                rc.search,
                f"SELECT * FROM {config['component_track_head_dataset']} WHERE associatedPrincipalTrack IS NOT NULL LIMIT 10000"
            )
            
            if component_heads:
                component_associations = component_heads
                unique_principals = {
                    _value(record, 'associatedPrincipalTrack')
                    for record in component_associations}
                unique_principals.discard(None)
                logging.info(
                    f"Downloaded {len(component_associations)} component tracks "
                    f"associated with {len(unique_principals)} principal tracks")
        
        return principal_tracks, component_associations
        
    except Exception as e:
        logging.error(f"Error downloading existing data: {e}")
        logging.error(traceback.format_exc())
        return None, None



NUM_FUSION_WORKERS = 4


def _find_supersede_root(track_id: str, supersede_map: Dict[str, str]) -> str:
    """Follow the supersede chain to find the root (superseding) trackId.
    If track_id is not superseded, it IS the root."""
    visited = set()
    current = track_id
    while current in supersede_map and current not in visited:
        visited.add(current)
        current = supersede_map[current]
    return current


def _fusion_shard_for_track(track_id: str, supersede_map: Dict[str, str], n_shards: int) -> int:
    """Return the shard that owns a component-track supersede group."""
    if not track_id:
        return 0
    root = _find_supersede_root(track_id, supersede_map)
    return stable_shard(root, n_shards)


def _apply_management_events_to_supersede_map(
        supersede_map: Dict[str, str],
    management_events: List[dict]) -> Dict[str, str]:
    """Return the complete new map after incremental management events.

    The result must replace the caller's prior snapshot. Merging it with
    ``dict.update`` would retain keys removed by RESTORE actions.
    """
    latest = {}
    for event in sorted(
            management_events,
            key=lambda record: str(_value(record, 'crucibleHeader.updatedDate', '')),
            reverse=True):
        track_id = _value(event, 'trackId')
        if track_id is not None:
            latest.setdefault(str(track_id), event)
    updated = dict(supersede_map)
    for track_id, event in latest.items():
        action = _value(event, 'action')
        if action == 'SUPERSEDE':
            superseded_by = _value(event, 'supersededBy')
            if isinstance(superseded_by, float) and superseded_by != superseded_by:
                superseded_by = None
            updated[track_id] = str(superseded_by) if superseded_by else None
        elif action == 'DELETE':
            updated[track_id] = None
        elif action == 'RESTORE':
            updated.pop(track_id, None)
    result = {}
    for track_id, target in updated.items():
        current = target
        visited = {track_id}
        while current in updated and current not in visited:
            visited.add(current)
            current = updated[current]
        result[track_id] = current if isinstance(current, str) else None
    return result


def _broadcast_initial_supersede_map(
        shard_queues: List[Any], supersede_map: Dict[str, str]) -> None:
    if not supersede_map:
        return
    for shard_queue in shard_queues:
        shard_queue.put({
            'type': 'supersede_update', 'data': dict(supersede_map)})


def _remove_stale_principal_track(
    kf: EntityPrincipalTrackFilter, track_id: str) -> Optional[str]:
    principal_track_id = kf.id_to_principal_trackid.pop(track_id, None)
    if not principal_track_id:
        return None
    kf.priors.pop(principal_track_id, None)
    kf.environment_by_track.pop(principal_track_id, None)
    return principal_track_id


def _select_fuser_config(config_list: list, perspective_config: dict) -> Optional[dict]:
    """Return the first enabled config with the datasets the fuser requires."""
    for cfg in config_list:
        if cfg.get('disabled'):
            continue
        if not cfg.get('component_track_event_dataset'):
            continue
        if not cfg.get('principal_track_event_dataset') or not cfg.get('principal_track_head_dataset'):
            logging.warning("Config missing principal track datasets - skipping")
            continue
        config = dict(cfg)
        for key, value in perspective_config.items():
            if key not in config:
                config[key] = value
        return config
    return None


def _run_fusion_shard(shard_queue, config, use_ci, ci_omega, passthrough):
    """Launch a sharded track fuser worker in its own process."""
    asyncio.run(_fusion_shard_worker(shard_queue, config, use_ci, ci_omega, passthrough))


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


def _head_field(row: dict, base: str):
    """Read a head field, unwrapping UUID-valued nested fields."""
    val = row.get(base)
    if isinstance(val, dict):
        val = val.get('uuid')
    return val


def _fanout_head_preload(shard_queues, config: dict, n_shards: int,
                         supersede_map: Dict[str, str]) -> None:
    """Download principal + component track heads ONCE and fan them out to the
    shard workers' queues (pre-bucketed to each shard's owned tracks).

    Mirrors the transformer's object-cache fan-out: a single parent-side read
    replaces N independent per-shard queries (which N-duplicated the load and
    504'd on large head datasets). Routing matches the dispatcher: a component
    head routes by the supersede-root hash of its (component) trackId; a
    principal head follows its associated components to the same shard.
    """
    if skip_head_preload(config):
        for s in range(n_shards):
            shard_queues[s].put({'type': 'init_heads', 'principal_heads': [],
                                 'component_heads': [], 'done': True})
        logging.info(
            f"Head preload SKIPPED (skip_head_preload): {n_shards} fusion shard(s) start "
            f"with empty Kalman state; deterministic principal trackIds keep heads consistent.")
        return

    principal_ds = config.get('principal_track_head_dataset')
    component_ds = config.get('component_track_head_dataset')

    principal_buckets = [[] for _ in range(n_shards)]
    component_buckets = [[] for _ in range(n_shards)]
    principal_to_shard: Dict[str, int] = {}

    # ---- component heads: route by component-trackId supersede-root hash ----
    if component_ds:
        recs = load_all_records_for_preload(
            component_ds, config, rc, auth, "fusion_head_preload_limit"
        )
        if recs:
            for row in recs:
                apt = _head_field(row, 'associatedPrincipalTrack')
                if apt is None:
                    continue
                comp_tid = _head_field(row, 'trackId')
                if comp_tid is None:
                    continue
                shard = _fusion_shard_for_track(str(comp_tid), supersede_map, n_shards)
                component_buckets[shard].append(row)
                principal_to_shard[str(apt)] = shard
            logging.info(
                f"Head preload: {sum(len(b) for b in component_buckets)} component "
                f"associations routed to {n_shards} shards")
            del recs

    # ---- principal heads: follow associated components to the same shard ----
    if principal_ds:
        recs = load_all_records_for_preload(
            principal_ds, config, rc, auth, "fusion_head_preload_limit"
        )
        if recs:
            for row in recs:
                ptid = _head_field(row, 'trackId')
                if ptid is None:
                    continue
                shard = principal_to_shard.get(str(ptid))
                if shard is None:
                    shard = _fusion_shard_for_track(str(ptid), supersede_map, n_shards)
                principal_buckets[shard].append(row)
            logging.info(
                f"Head preload: {sum(len(b) for b in principal_buckets)} principal "
                f"heads bucketed across {n_shards} shards")
            del recs

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


async def _fusion_shard_dispatcher(event_queue: asyncio.Queue, supersede_queue: asyncio.Queue,
                                    shard_queues, supersede_map: Dict[str, str]):
    """
    Dispatcher coroutine that routes events to shard workers by supersede-group hash.
    Component tracks in the same supersede chain are guaranteed to go to the same
    shard. Also broadcasts supersede map updates to all shards.
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
            if supersede_events:
                # Snapshot old routing before updating the trackId map.
                old_roots = {}
                for event in supersede_events:
                    track_id = _id_value(event, 'trackId')
                    if track_id is None:
                        continue
                    track_id = str(track_id)
                    old_roots[track_id] = _find_supersede_root(track_id, supersede_map)

                supersede_map = _apply_management_events_to_supersede_map(
                    supersede_map, supersede_events)

                # Detect routing changes: if a track's root changed, its old
                # shard may hold an orphaned principal track that should be stale-marked.
                for track_id, old_root in old_roots.items():
                    new_root = _find_supersede_root(track_id, supersede_map)
                    old_shard = stable_shard(old_root, n_shards)
                    new_shard = stable_shard(new_root, n_shards)
                    if old_shard != new_shard:
                        shard_queues[old_shard].put({
                            'type': 'mark_stale',
                            'data': {'trackId': track_id}
                        })

                # Broadcast the full supersede_map to all shards
                for sq in shard_queues:
                    sq.put({'type': 'supersede_update', 'data': dict(supersede_map)})

        # Partition events by supersede-group root hash
        shard_buckets = [[] for _ in range(n_shards)]
        for event in events:
            track_id = None
            if isinstance(event, dict):
                track_id = event.get('trackId', {})
                if isinstance(track_id, dict):
                    track_id = track_id.get('uuid')
            if track_id:
                shard_idx = _fusion_shard_for_track(track_id, supersede_map, n_shards)
            else:
                shard_idx = 0
            shard_buckets[shard_idx].append(event)

        # Push to shard queues
        for i, bucket in enumerate(shard_buckets):
            if bucket:
                shard_queues[i].put({'type': 'events', 'data': bucket})


async def _fusion_shard_worker(shard_queue, config, use_ci, ci_omega, passthrough):
    """
    Async worker for a single fusion shard. Processes events from shard_queue.
    Each shard maintains its own PrincipalTrackKalmanFilter for its subset of tracks.

    Messages on the queue are dicts with keys:
      - 'type': 'events' | 'supersede_update' | 'mark_stale'
      - 'data': list of event dicts, updated supersede_map dict, or {trackId: ...}
    """
    global auth, rc, wc
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)

    kf = EntityPrincipalTrackFilter(config)
    # Preloaded principal/component heads are fanned out ONCE by the parent
    # (run()) via the shard queue as 'init_heads' messages (pre-bucketed to the
    # tracks this shard owns) instead of each shard querying Crucible. Accumulate
    # the chunks until the 'done' marker; stash any other message that arrives
    # meanwhile (handled on the first loop pass). Deterministic principal
    # trackIds mean a shard that receives no preload still won't create orphan
    # heads — it just re-derives the same ids.
    principal_head_records = []
    component_head_records = []
    stashed_msgs = []
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

    logging.info(
        f"Shard: initialized from fan-out with {len(principal_head_records)} "
        f"principal heads, {len(component_head_records)} component associations")

    kf.initialize({}, principal_head_records, component_head_records)
    if use_ci:
        kf.set_fusion_method('ci', omega=ci_omega)

    # Fuse identity per principal track across all of its component
    # tracks (this shard's subset). principal_track_id -> {identity.field: value}.
    # Starts empty (like the kf state) and accumulates as events arrive.
    principal_identity_map: Dict[str, Dict[str, Any]] = {}

    # Component-principal association stamps and existing principal-head writes
    # are best-effort preload/query state; principal track events are the live
    # fuser output.  Keep background tasks so slow head datasets do not block
    # event writes.
    pending_associations: Dict[str, str] = {}
    pending_association_tasks: list = []
    pending_principal_head_tasks: list = []
    pending_principal_heads_by_track: Dict[str, Dict[str, Any]] = {}
    principal_head_update_emit_time_by_track: Dict[str, float] = {}

    default_batch_size = int(config['batch_write_chunk_size'])
    head_batch_size = int(config['batch_update_chunk_size'])
    assoc_batch_size = head_batch_size
    max_concurrent = config.get('batch_write_max_concurrent')
    principal_head_dataset = config.get('principal_track_head_dataset')
    component_head_dataset = config.get('component_track_head_dataset')
    _refresh_token = lambda: setattr(wc, 'token', auth.get_token())

    # Any non-init messages received while accumulating the init_heads fan-out
    # (e.g. the initial supersede broadcast) are handled on the first loop pass.
    startup_msgs = stashed_msgs

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

            pending_association_tasks, retry_associations = collect_finished_association_tasks(
                pending_association_tasks)
            stale_retries = 0
            for track_id, principal_track_id in retry_associations.items():
                if track_id in pending_associations:
                    stale_retries += 1
                    continue
                pending_associations[track_id] = principal_track_id
            if stale_retries:
                logging.info(
                    f"Coalesced stale associatedPrincipalTrack retries; "
                    f"coalesced={stale_retries} pending={len(pending_associations)}")
            pending_principal_head_tasks = collect_finished_track_update_tasks(
                pending_principal_head_tasks, 'principal-head write',
                pending_principal_heads_by_track)
            if not pending_principal_head_tasks and pending_principal_heads_by_track:
                coalesced_heads = list(pending_principal_heads_by_track.values())
                pending_principal_heads_by_track.clear()
                logging.info(
                    f"Scheduled coalesced background principal track head write to "
                    f"{principal_head_dataset}: records={len(coalesced_heads)}")
                pending_principal_head_tasks.append(asyncio.create_task(
                    best_effort_write_principal_heads(
                        coalesced_heads, principal_head_dataset, head_batch_size,
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent)))
            if pending_associations and component_head_dataset and not pending_association_tasks:
                associations_to_write = pending_associations
                pending_associations = {}
                logging.info(
                    f"Scheduled coalesced background associatedPrincipalTrack update for "
                    f"{len(associations_to_write)} component track head(s)")
                pending_association_tasks.append(asyncio.create_task(
                    flush_component_associations(
                        associations_to_write, component_head_dataset, assoc_batch_size,
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent)))

            # Process supersede updates first, then events
            all_events = []
            stale_track_ids = []
            for m in messages:
                if m.get('type') == 'supersede_update':
                    updated_map = m['data']
                    re_associations = kf.update_supersede_map(updated_map)
                    await cleanup_restored_identity(
                        kf, principal_identity_map, pending_principal_head_tasks,
                        pending_principal_heads_by_track,
                        principal_head_update_emit_time_by_track)
                    for assoc in re_associations:
                        pending_associations[assoc['trackId']] = assoc['associatedPrincipalTrack']
                elif m.get('type') == 'mark_stale':
                    stale_track_ids.append(m['data']['trackId'])
                elif m.get('type') == 'events':
                    all_events.extend(m['data'])

            # Remove orphaned principal tracks for component tracks that migrated shards
            if stale_track_ids:
                for track_id in stale_track_ids:
                    pt_id = _remove_stale_principal_track(kf, track_id)
                    if pt_id:
                        principal_identity_map.pop(pt_id, None)
                        pending_principal_heads_by_track.pop(pt_id, None)
                        logging.info(f"Removed orphaned principal track {pt_id} for migrated track {track_id}")

            if not all_events:
                if (pending_associations and component_head_dataset
                        and not pending_association_tasks):
                    associations_to_write = pending_associations
                    pending_associations = {}
                    logging.info(
                        f"Scheduled background associatedPrincipalTrack update for "
                        f"{len(associations_to_write)} component track head(s)")
                    pending_association_tasks.append(asyncio.create_task(
                        flush_component_associations(
                            associations_to_write,
                            component_head_dataset,
                            assoc_batch_size,
                            token_refresher=_refresh_token,
                            max_concurrent_writes=max_concurrent,
                        )))
                continue

            component_tracks = [
                event for event in all_events
                if isinstance(event, dict) and _value(event, 'interceptTimestamp') is not None
            ]
            dropped = len(all_events) - len(component_tracks)
            if dropped:
                logging.warning(
                    f"Dropped {dropped} component track(s) missing interceptTimestamp")
            component_tracks.sort(
                key=lambda record: str(_value(record, 'interceptTimestamp', '')))
            if not component_tracks:
                continue

            # Process each component track to create principal tracks
            principal_track_events = []
            principal_track_heads = []
            component_to_principal_associations = []

            for component_track in component_tracks:
                component_trackid = _id_value(component_track, 'trackId')
                routing_key = str(component_trackid)

                comp_env = _value(component_track, 'environment')
                principal_track_id, principal_id, is_new, is_new_association = (
                    kf.get_or_create_principal_track(routing_key, component_trackid, environment=comp_env))

                kf.set_track_environment(principal_track_id, comp_env)

                if is_new_association and component_trackid is not None:
                    component_to_principal_associations.append({
                        'trackId': component_trackid,
                        'associatedPrincipalTrack': principal_track_id
                    })

                if component_trackid:
                    kf.add_component_track(principal_track_id, component_trackid)

                if passthrough:
                    principal_event = create_principal_track_event(
                        component_track, principal_track_id)
                    principal_head = create_principal_track_head(
                        component_track, principal_track_id)
                else:
                    post = kf.update_record(principal_track_id, component_track)
                    principal_event = create_principal_track_event(
                        component_track, principal_track_id)
                    principal_head = create_principal_track_head(
                        component_track, principal_track_id)
                    if post is not None:
                        state_dict = kf.state_record(
                            post, principal_track_id, principal_id)
                        principal_event.update(state_dict)
                        principal_head.update(state_dict)

                # Overlay the fused identity (union of non-null
                # identity.* across this principal track's component tracks).
                # A superseded track's identity takes precedence over the
                # surviving track's; among equal-precedence sources the
                # most-recent non-null wins (ascending interceptTimestamp order).
                is_superseded = kf.supersede_map.get(routing_key) is not None
                fused_identity = fuse_track_identity(
                    principal_identity_map, principal_track_id, component_track,
                    is_superseded=is_superseded)
                for identity_path, identity_value in fused_identity.items():
                    set_value(principal_event, identity_path, identity_value)
                    set_value(principal_head, identity_path, identity_value)

                principal_track_events.append(principal_event)
                principal_track_heads.append(principal_head)

            if principal_track_events:
                principal_track_heads = last_record_by_track(principal_track_heads)
                add_track_wgs84_kinematics(principal_track_events)
                add_track_wgs84_kinematics(principal_track_heads)

                # Write principal HEADS first so we never emit principal track
                # events that reference a newly-created head that failed to write.
                failed_principal_trackIds = set()
                if 'principal_track_head_dataset' in config:
                    # Split principal heads into newly-created vs existing tracks,
                    # mirroring the tracker's new_tracks/existing_tracks split.
                    # New principal trackIds are fresh uuid4s (see
                    # PrincipalTrackKalmanFilter.get_or_create_principal_track) so
                    # nothing exists on the server yet -> POST create is safe.
                    # Existing heads carry FUSED identity accumulated in-memory
                    # (non-null fields only); a POST would overwrite the record and,
                    # after a restart with a partial accumulator, blank out identity
                    # already on the server. An entity-aware update only sets the
                    # non-null fields we send (df_to_formatted_JSON drops nulls),
                    # preserving server-side identity; not-found heads are re-created.
                    new_principal_heads = [
                        head for head in principal_track_heads
                        if str(_id_value(head, 'trackId')) in kf.new_principal_trackids
                    ]
                    existing_principal_heads = [
                        head for head in principal_track_heads
                        if str(_id_value(head, 'trackId')) not in kf.new_principal_trackids
                    ]

                    # Create new principal heads (POST), collecting any failures.
                    if new_principal_heads:
                        new_principal_json = [compact_record(head) for head in new_principal_heads]
                        if new_principal_json:
                            failed_records, _ = await write_batch_chunked(
                                new_principal_json,
                                principal_head_dataset,
                                wc.write_record_batch_by_name,
                                head_batch_size,
                                token_refresher=_refresh_token,
                                max_concurrent_writes=max_concurrent,
                            )
                            for rec in failed_records:
                                tid = _id_value(rec, 'trackId')
                                if tid:
                                    failed_principal_trackIds.add(tid)
                            if failed_principal_trackIds:
                                logging.warning(
                                    f"{len(failed_principal_trackIds)} new principal track heads "
                                    f"failed to write; withholding their events and keeping for retry"
                                )

                    # Only clear successfully-written new tracks; keep failures for
                    # the next cycle (mirrors filter_manager.new_trackIds in the tracker).
                    successfully_written = (
                        {str(_id_value(head, 'trackId')) for head in new_principal_heads}
                    )
                    successfully_written -= failed_principal_trackIds
                    kf.new_principal_trackids -= successfully_written

                    # Existing principal head writes are best-effort. Principal
                    # track events are the reliable live stream; heads support
                    # preload/query state and should not block fuser progress on
                    # a contended dataset.
                    if existing_principal_heads:
                        existing_principal_heads = coalesce_existing_head_updates(
                            existing_principal_heads,
                            principal_head_update_emit_time_by_track,
                            head_update_interval_seconds(config),
                            'principal')
                        existing_principal_json = [
                            compact_record(head) for head in existing_principal_heads]
                        coalesced = buffer_latest_records_by_track(
                            pending_principal_heads_by_track,
                            existing_principal_json)
                        if coalesced:
                            logging.info(
                                f"Coalesced buffered principal track-head writes; "
                                f"coalesced={coalesced} "
                                f"pending={len(pending_principal_heads_by_track)}")
                        if (not pending_principal_head_tasks
                                and pending_principal_heads_by_track):
                            coalesced_heads = list(pending_principal_heads_by_track.values())
                            pending_principal_heads_by_track.clear()
                            logging.info(
                                f"Scheduled background principal track head write to {principal_head_dataset}: "
                                f"records={len(coalesced_heads)}")
                            pending_principal_head_tasks.append(asyncio.create_task(
                                best_effort_write_principal_heads(
                                    coalesced_heads,
                                    principal_head_dataset,
                                    head_batch_size,
                                    token_refresher=_refresh_token,
                                    max_concurrent_writes=max_concurrent,
                                )))

                # Write principal EVENTS after heads, EXCLUDING any whose new head
                # failed to create this cycle (avoids orphan events with no head).
                if 'principal_track_event_dataset' in config:
                    if failed_principal_trackIds:
                        principal_track_events = [
                            event for event in principal_track_events
                            if _id_value(event, 'trackId') not in failed_principal_trackIds
                        ]
                    await write_records_in_batches(
                        principal_track_events,
                        config['principal_track_event_dataset'],
                        wc.write_record_batch_by_name,
                        default_batch_size,
                        token_refresher=_refresh_token,
                        max_concurrent_writes=max_concurrent,
                    )

                if component_head_dataset and (
                        component_to_principal_associations or pending_associations):
                    # Merge freshly-derived associations (emitted once, on
                    # is_new_association) with any still pending from prior cycles
                    # whose component head had not yet been created.
                    coalesced_associations = 0
                    for a in component_to_principal_associations:
                        if a['trackId'] in pending_associations:
                            coalesced_associations += 1
                        pending_associations[a['trackId']] = a['associatedPrincipalTrack']
                    if coalesced_associations:
                        logging.info(
                            f"Coalesced buffered associatedPrincipalTrack updates; "
                            f"coalesced={coalesced_associations} "
                            f"pending={len(pending_associations)}")
                    if not pending_association_tasks:
                        associations_to_write = pending_associations
                        pending_associations = {}
                        logging.info(
                            f"Scheduled background associatedPrincipalTrack update for "
                            f"{len(associations_to_write)} component track head(s)")
                        pending_association_tasks.append(asyncio.create_task(
                            flush_component_associations(
                                associations_to_write,
                                component_head_dataset,
                                assoc_batch_size,
                                token_refresher=_refresh_token,
                                max_concurrent_writes=max_concurrent,
                            )))

                logging.info(
                    f"Processed {len(principal_track_events)} principal track events, "
                    f"{len(principal_track_heads)} heads, "
                    f"{len(component_to_principal_associations)} component-principal associations"
                )

                if not passthrough:
                    kf.log_stats()

        except Exception as e:
            logging.error(f"An error occurred in fusion shard worker: {e}")
            logging.error(traceback.format_exc())
            continue


def run(stream_manager_perspective, *, use_ci: bool = True,
        ci_omega: float = None, passthrough: bool = False):
    """
    Main function to run the entity track fuser.
    
    This function:
    - Listens for component track events via SSE
    - Listens for supersede/delete management events via SSE
    - Checks supersede map for component tracks that should be fused (from entity_duplicate_identifier.py)
    - For non-superseded component tracks: passes through kinematics directly
    - For superseded component tracks: fuses kinematics into the superseding component track's principal track
    - Writes principal track events and heads to their respective datasets
    
    Args:
        stream_manager_perspective: Perspective name for Stream Manager (e.g., 'Blue')
        use_ci: If True, use Covariance Intersection instead of standard Kalman
        ci_omega: Fixed CI weight in (0, 1). None = auto-optimize.
                  ~0 = measurement-dominated, ~0.5 = balanced, ~1 = prediction-dominated.
        passthrough: If True, skip Kalman filtering and pass through kinematics directly.
    """
    global auth, rc, wc
    # Register the signal handler
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)

    result = find_and_validate_configs(stream_manager_perspective, include_scripts=False, rc_instance=rc)
    config_list = result['datafeed_configs']
    perspective_config = result['perspective_config']

    config = _select_fuser_config(config_list, perspective_config)
    if not config:
        logging.error("No valid configuration found for entity track fusion")
        return

    event_queue = Queue()
    supersede_queue = Queue()

    # Bootstrap the supersede map from existing management events
    rc.token = auth.get_token()
    supersede_map = get_supersede_map(config, rc_instance=rc, auth_instance=auth)
    logging.info(f"Loaded {len(supersede_map)} supersede mappings for routing")

    # Spawn sharded worker processes
    n_shards = int(config.get('num_fusion_workers', NUM_FUSION_WORKERS))
    # num_fusion_workers is a PERSPECTIVE-level setting: one fuser runs per
    # perspective. Warn if datafeed configs disagree so a silent per-feed
    # mismatch is visible instead of an arbitrary config's value winning.
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
            f"num_fusion_workers differs across {stream_manager_perspective} datafeed configs "
            f"{sorted(_fusion_vals)}; using {n_shards}. num_fusion_workers is perspective-level "
            f"\u2014 set it identically in every config.")
    logging.info("Track fusion Kalman backend = native numpy")
    shard_queues = [Queue(maxsize=_SHARD_QUEUE_MAXSIZE) for _ in range(n_shards)]
    processes = []

    for shard_idx in range(n_shards):
        proc = Process(
            target=_run_fusion_shard,
            args=(shard_queues[shard_idx], config, use_ci, ci_omega, passthrough),
            daemon=True
        )
        proc.start()
        processes.append(proc)

    logging.info(f"Started {n_shards} fusion shard workers")

    # Download principal + component track heads ONCE and fan them out to the
    # shard workers (pre-bucketed) so each shard initializes from its queue
    # instead of independently querying Crucible. Must run BEFORE the event
    # dispatch so heads are consumed first.
    _fanout_head_preload(shard_queues, config, n_shards, supersede_map)

    _broadcast_initial_supersede_map(shard_queues, supersede_map)
    if supersede_map:
        logging.info(
            f"Broadcast initial supersede map ({len(supersede_map)} entries) "
            f"to {n_shards} shards")

    # Run SSE listeners + dispatcher in the main event loop
    async def run_async():
        aio_event_queue = asyncio.Queue()
        aio_supersede_queue = asyncio.Queue()

        component_track_event_dataset = config['component_track_event_dataset']
        entity_management_event_dataset = config.get('entity_management_event_dataset')

        coroutines = []

        # SSE listener for component track events
        sql_query = f"SELECT * FROM {component_track_event_dataset}"
        logging.info(f"Track Fusion SSE Listener SQL Query: {sql_query}")
        coroutines.append(SSE_listener(sql_query, aio_event_queue, auth))

        # SSE listener for supersede/delete/restore management events
        if entity_management_event_dataset:
            supersede_sql_query = (
                f"SELECT * FROM {entity_management_event_dataset}"
                " WHERE action IN ('SUPERSEDE', 'DELETE', 'RESTORE')"
            )
            logging.info(f"Track Fusion Supersede SSE Listener SQL Query: {supersede_sql_query}")
            coroutines.append(SSE_listener(supersede_sql_query, aio_supersede_queue, auth))
        else:
            logging.warning("No entity_management_event_dataset configured; supersede SSE listener not started")

        # Dispatcher: routes events and supersede updates to shard workers
        coroutines.append(
            _fusion_shard_dispatcher(aio_event_queue, aio_supersede_queue, shard_queues, supersede_map)
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
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Track Fusion for Principal Track Creation (Entity Pipeline)')
    parser.add_argument("Stream_Manager_Perspective", help='Name of perspective in Entity Stream Manager Configuration (e.g., Blue)', default='Blue')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level: INFO (default), WARN, ERROR, or DEBUG')
    parser.add_argument('--use-fork', action='store_true', help='Use fork start method for multiprocessing') # needed for future multiprocessing support on Macs
    parser.add_argument('--download-only', action='store_true', help='Only download and display existing principal tracks and associations, then exit')
    parser.add_argument('--no-ci', action='store_true', help='Disable Covariance Intersection; use standard Kalman update instead')
    parser.add_argument('--ci-omega', type=float, default=None, help='Fixed CI weight in (0,1). ~0=measurement-dominated, ~0.5=balanced, ~1=prediction-dominated. Omit for auto-optimization.')
    parser.add_argument('--passthrough', action='store_true', help='Skip Kalman filtering, pass through kinematics directly')
    args = parser.parse_args()

    if args.use_fork:
        set_start_method('forkserver', force=True)
        logging.info("  Using the fork start method for multiprocessing  ")
    
    log_level = args.log_level
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
    
    if args.download_only:
        # Just download and display existing data
        async def download_and_exit():
            global auth, rc, wc
            auth, rc, wc = ensure_api_controllers(auth, rc, wc)
            result = find_and_validate_configs(stream_manager_perspective, include_scripts=False, rc_instance=rc)
            config_list = result['datafeed_configs']
            perspective_config = result['perspective_config']
            config = _select_fuser_config(config_list, perspective_config)
            if config:
                await download_existing_principal_tracks_and_associations(config)
            else:
                logging.error("No valid configuration found")
        
        asyncio.run(download_and_exit())
    else:
        # Run the full entity track fuser
        run(stream_manager_perspective, use_ci=not args.no_ci, ci_omega=args.ci_omega,
            passthrough=args.passthrough) 