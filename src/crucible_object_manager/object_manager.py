extra_diagnostics = False


'''

 A correlator is set running from the command line as follows:

     python object_manager.py <Stream Manager configuration dataset name>
        e.g.,
     python object_manager.py Blue_POV
     with optional arguments:
          --log=DEBUG (or INFO, WARNING, ERROR)
          INFO is the default
          --use-fork (only for Mac)


It can be run in a script with the following:
from cruciblelib import object_manager
object_manager.run(<Stream Manager configuration dataset name>,log_level='INFO')

KINEMATIC DATA ROUTING:
The object manager routes kinematic data through two pathways based on configuration.
There is one SSE dataset feeding tracker kinematic data and one feeding passthrough data.
Both SSE streams feed into the same message queue and are separated by column eventType.

1. TRACKED PATHWAY (format_tracker_kinematic_data):
    - used when 'track_event_dataset' or 'principal_track_event_dataset' is configured 
    - this can be either "passthrough" (crucible tracker runs in passthrough mode) or ANY Kalman filter
    - Triggered for records with eventType == 'empty' (missing eventType field)
    - Tracker records also lack the source.datasetName column (only present in object events)
    - AND the record must have valid ECEF position columns (ecefPosition.x/y/z)
    - Records missing ECEF columns are dropped with a warning (likely malformed passthrough)
    - Records with all-NaN ECEF values are dropped with a warning
    - Performs ECEF to WGS84 coordinate transformation and covariance matrix calculations

2. NON-TRACKED PATHWAY aka passthrough (format_non_tracked_kinematic_data):
    - copies kinematic data with minimal processing from Live Object Events that are not from tracked datasets
    - Triggered for records with eventType == 'KINEMATIC_UPDATE'
    - Records with eventType='KINEMATIC_UPDATE' from a tracked dataset: filtered out
    - Records with eventType='KINEMATIC_UPDATE' always have source.datasetName (object event origin)
    - Records from datasets in tracked_datasets list are filtered OUT (those are handled by tracker)
    - Records must have estimatedKinematics columns; otherwise returns empty
    - Performs minimal processing, primarily copying existing uncertainty ellipse data

EDGE CASES:
- Records with eventType='empty' but no ECEF data: dropped (not valid tracker output)
- Empty DataFrames at any stage: safely skipped without unnecessary error logging
- Missing timestamp columns in tracker data: falls back through priority chain, uses current time as last resort

IGNORED DATASETS:
- Cruciblelib has its own tracker, object_tracker.py, but this tracker ignores datasets with
  crucible_tracker containing '3rd party' or 'third party'. 
  These datasets can still go through the tracked pathway if the other conditions are met,
  but they will not be processed by the object_tracker.

BACKWARD COMPATIBILITY:
- Missing track_event_dataset key defaults to passthrough mode (non-tracked pathway)
- Missing crucible_tracker key defaults to tracked mode (if track_event_dataset exists)
- Explicit 'passthrough' in crucible_tracker with a track_event_dataset routes through the tracked pathway
  (tracker runs in passthrough mode, duplicate KINEMATIC_UPDATE object events are filtered out)
- bypass_tracker=true uses object-event kinematics even with track datasets configured,
  independently of crucible_tracker.
'''



import os
import signal
import json
import time
import hashlib
import argparse
import logging
import traceback
from typing import List, Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import asyncio
import aiohttp
from aiohttp_sse_client import client as sse_client
from multiprocessing import Process, Queue, Event, set_start_method
from concurrent.futures import ThreadPoolExecutor

# function for translating ECEF to WGS84 lat and lon
from pyproj import Transformer as pyproj_transformer
ecef_transformer = pyproj_transformer.from_crs( "epsg:4978", "epsg:4326") 



try:
    from object_utils import  \
        find_and_validate_configs, uses_object_event_kinematics, \
        coroutine_launcher,pandas_ecef_vel_to_wgs84_by_row, \
        rotate_covariance_matrix_ecef_to_enu,covariance_matrix_to_uncertainty_ellipse, \
        int_conversion,int_cols, instantiate_api_controllers, \
        get_supersede_map, get_management_events, build_supersede_map, write_batch_chunked, _fast_df_to_nested_json, run_sse_listener, _get_object_id
        
except ImportError:
    from .object_utils import \
        find_and_validate_configs, uses_object_event_kinematics, \
        coroutine_launcher,pandas_ecef_vel_to_wgs84_by_row, \
        rotate_covariance_matrix_ecef_to_enu,covariance_matrix_to_uncertainty_ellipse, \
        int_conversion,int_cols, instantiate_api_controllers, \
        get_supersede_map, get_management_events, build_supersede_map, write_batch_chunked, _fast_df_to_nested_json, run_sse_listener, _get_object_id

try:
    from cruciblelib import read_controller, write_controller,utils,log_utils,authenticator
except:
    from . import read_controller, write_controller,utils,log_utils, authenticator

try:
    from cruciblelib.exceptions import DetailedHTTPError
except:
    from .exceptions import DetailedHTTPError



# Controllers are initialized lazily (not at import time) so that
# forkserver child processes don't make HTTPS calls during module import.
rc = None
wc = None
auth = None


object_primary_key = 'objectId.uuid'
tracker_primary_key = 'objectId'

# --- [PERF] timing switch (single toggle for ALL per-stage timing logs) ---
# Flip to False (or set env CRUCIBLE_PERF_TIMING=false) to silence every [PERF]
# line and make the timers no-ops.
PERF_TIMING = os.getenv('CRUCIBLE_PERF_TIMING', 'true').lower() not in ('false', '0', 'no')


def _perf_disabled() -> float:
    """Zero-cost stand-in for time.perf_counter when PERF_TIMING is off."""
    return 0.0

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# CAP (ft^2) to TQ conversion from MIL-STD-6016G
cap_to_tq = {15:1080.,14:3600.,13:14400.,12:64800., \
                11:252_000.,10:1_011_600.,9:39_600_000.,8:158_400_000.,
                7:972_000_000.,6:3_960_000_000.,5:8_892_000_000.,4:15_804_000_000.,
                3:24_696_000_000.,2:99_180_000_000.,} 
# convert from ft^2 to m^2
cap_to_tq = {key: value * 0.092903 for key, value in cap_to_tq.items()}


def update_source_tally(source_tally: dict, event_df: pd.DataFrame,
                        tracked_datasets: Optional[List[str]] = None,
                        track_event_datasets: Optional[Union[List[str], str]] = None) -> None:
    """
    Updates the shared source_tally with new events.

    Structure: source_tally[obj_id] = {
        (dataset_name, upstream_source, event_type): (timestamp, collection_type),
        ...
    }

    Each key tracks the latest timestamp for that unique source combination.
    """
    if tracked_datasets is None:
        tracked_datasets = []
    if isinstance(track_event_datasets, list):
        tracker_dataset_label = track_event_datasets[0] if track_event_datasets else None
    else:
        tracker_dataset_label = track_event_datasets or None
    now = time.time()

    if event_df is None or len(event_df) == 0:
        return

    # --- Vectorised replacement for the per-row iterrows() loop ---
    # Pull only the columns we need; missing columns become all-None Series.
    n = len(event_df)
    idx = event_df.index

    def _col(name):
        return event_df[name] if name in event_df.columns else pd.Series([None] * n, index=idx)

    # Normalise a column to python str-or-None (single-column map, not a
    # per-row dict build).  Non-strings (NaN floats, ints) collapse to None.
    def _str_or_none(s):
        return s.map(lambda x: x if isinstance(x, str) and x != '' else None)

    w = pd.DataFrame({
        'oid': _str_or_none(_col(object_primary_key)),
        'e': _col('eventType'),
        'd': _str_or_none(_col('source.datasetName')),
        'u': _str_or_none(_col('upstreamSource')),
        'c': _str_or_none(_col('collectionType')),
    }, index=idx)

    # Must have a usable objectId (non-empty string).
    w = w[w['oid'].notna()]
    if w.empty:
        return

    empty_mask = w['e'].eq('empty')

    # eventType == 'empty' == tracker head output; it must NOT carry a
    # source.datasetName or upstreamSource (schema-locked).  Rows that do are
    # misclassified object events -> log + drop them from the tally.
    bad_dataset = empty_mask & w['d'].notna()
    if bool(bad_dataset.any()):
        logging.error(f"{int(bad_dataset.sum())} event(s) with eventType='empty' carry a "
                      f"source.datasetName (e.g. {w.loc[bad_dataset, 'd'].iloc[0]!r}); "
                      f"skipping from source tally.")
    bad_upstream = empty_mask & ~bad_dataset & w['u'].notna()
    if bool(bad_upstream.any()):
        logging.error(f"{int(bad_upstream.sum())} event(s) with eventType='empty' and no "
                      f"source.datasetName carry an upstreamSource "
                      f"(e.g. {w.loc[bad_upstream, 'u'].iloc[0]!r}); skipping from source tally.")

    # Valid 'empty' rows route to the tracker source, but only if the tracker
    # dataset label is known; otherwise they are dropped.
    empty_ok = empty_mask & ~bad_dataset & ~bad_upstream
    if tracker_dataset_label:
        w.loc[empty_ok, 'e'] = 'TRACKER_KINEMATIC_UPDATE'
        w.loc[empty_ok, 'd'] = tracker_dataset_label
        w.loc[empty_ok, 'u'] = None
    else:
        empty_ok = pd.Series(False, index=w.index)

    # Non-'empty' rows: tracked-feed KINEMATIC_UPDATE is handled by the tracker
    # (dropped here); remaining KINEMATIC_UPDATE becomes passthrough.
    kin_mask = w['e'].eq('KINEMATIC_UPDATE')
    kin_tracked = kin_mask & w['d'].notna() & w['d'].isin(tracked_datasets)
    w.loc[kin_mask & ~kin_tracked, 'e'] = 'PASSTHROUGH_KINEMATIC_UPDATE'

    # Keep: valid routed 'empty' rows, OR non-empty rows with a truthy eventType
    # that are not a tracked KINEMATIC_UPDATE.  Every kept row must have a
    # dataset name (mirrors the original `if not (dataset_name and str)` guard).
    non_empty = (~empty_mask) & w['e'].map(lambda x: bool(x)) & (~kin_tracked)
    keep = (empty_ok | non_empty) & w['d'].notna()
    w = w[keep]
    if w.empty:
        return

    # Collapse to one row per (objectId, dataset, upstream, eventType); the
    # FIRST occurrence's collectionType wins, matching the original loop where
    # equal-`now` timestamps mean a later duplicate never overwrites the first.
    w = w.drop_duplicates(subset=['oid', 'd', 'u', 'e'], keep='first')

    # Populate the nested tally over UNIQUE source rows (itertuples, fast) rather
    # than every event row.  Normalise upstream/collectionType to str-or-None at
    # read time (pandas can leave NaN in the object columns) and preserve the
    # original "only overwrite if newer" guard.
    for r in w.itertuples(index=False):
        u = r.u if isinstance(r.u, str) and r.u else None
        c = r.c if isinstance(r.c, str) and r.c else None
        key = (r.d, u, r.e)
        obj_sources = source_tally.get(r.oid)
        if obj_sources is None:
            obj_sources = {}
            source_tally[r.oid] = obj_sources
        existing = obj_sources.get(key)
        if existing is None or now > existing[0]:
            obj_sources[key] = (now, c)


def compute_latest_sources(source_tally: dict, obj_id: str) -> list:
    """
    Formats the source_tally entries for a given objectId into the sources array.
    Returns a list of dicts or None if no entries.
    """
    obj_sources = source_tally.get(obj_id)
    if not obj_sources:
        return None

    result = []
    for (dataset_name, upstream_source, event_type), (ts, collection_type) in obj_sources.items():
        ts_iso = pd.Timestamp(ts, unit='s', tz='UTC').strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        entry = {
            'datasetName': dataset_name,
            'latestUpdatedDate': ts_iso,
            'upstreamSource': upstream_source or '',
        }
        if collection_type:
            entry['collectionType'] = collection_type
        if event_type:
            entry['eventType'] = event_type
        result.append(entry)

    # Stable sort so index-based merge doesn't bleed fields across entries.
    result.sort(key=lambda x: (x.get('datasetName', ''), x.get('eventType', '')))
    return result[:20]


def _load_startup_snapshot(object_dataset: str) -> tuple:
    """Single startup download of the object dataset used to seed BOTH the
    ``source_tally`` and the ``identity_presence`` snapshot, so every object is
    downloaded only once.

    Returns ``(source_tally, identity_presence)``.  On query failure both are
    empty (correlator runs without either merge guard).
    """
    try:
        rc.token = auth.get_token()
        query = f"select objectId, `identity`, sources from {object_dataset}"
        existing = rc.search(query, format='json')
    except Exception as e:
        logging.warning(f"startup snapshot: could not query existing objects at "
                        f"startup; sources array may leave stale entries and the "
                        f"identity merge guard is disabled until each feed is "
                        f"re-seen: {e}")
        return {}, {}

    if not existing or not isinstance(existing, list):
        return {}, {}

    source_tally = _build_source_tally(existing)
    identity_presence = _build_identity_presence(existing)
    logging.info(f"startup snapshot: loaded {len(existing)} objects from "
                 f"{object_dataset} (sources seeded for {len(source_tally)}, "
                 f"identity presence for {len(identity_presence)})")
    return source_tally, identity_presence


def _apply_supersede_cache_updates(source_tally: Optional[dict],
                                   identity_presence: Optional[dict],
                                   drop_counts: Optional[dict],
                                   supersede_map: Optional[dict]) -> None:
    """Keep object-local caches compact when objects are superseded/deleted."""
    if not supersede_map:
        return

    for old_id, new_id in list(supersede_map.items()):
        old_id = str(old_id)
        if new_id is None:
            if source_tally is not None:
                source_tally.pop(old_id, None)
            if identity_presence is not None:
                identity_presence.pop(old_id, None)
            if drop_counts is not None:
                drop_counts.pop(old_id, None)
            continue

        new_id = str(new_id)
        if new_id == old_id:
            continue

        if source_tally is not None:
            old_sources = source_tally.pop(old_id, None)
            if old_sources:
                new_sources = source_tally.setdefault(new_id, {})
                for key, value in old_sources.items():
                    existing = new_sources.get(key)
                    if existing is None or value[0] > existing[0]:
                        new_sources[key] = value
        if identity_presence is not None:
            old_identity = identity_presence.pop(old_id, None)
            if old_identity:
                identity_presence.setdefault(new_id, set()).update(old_identity)
        if drop_counts is not None:
            drop_counts.pop(old_id, None)


def _build_source_tally(records: list) -> dict:
    """Rebuild the in-memory ``source_tally`` from the existing ``sources``
    array of every object.

    Crucible deep-merges the ``sources`` array by index on partial writes, so a
    freshly-started correlator that has only re-seen a subset of an object's
    feeds would otherwise emit a short ``sources`` array that leaves stale
    entries behind.  Seeding ``source_tally`` from what is already persisted
    means every write carries the full, current source list and the merge
    overwrites cleanly.  Later events (``time.time()`` timestamps) always win
    over these seeded startup entries.

    Returns the ``source_tally`` structure:
    ``{objectId.uuid: {(datasetName, upstreamSource, eventType): (timestamp, collectionType)}}``
    """
    source_tally: dict = {}
    for obj in records:
        if not isinstance(obj, dict):
            continue
        oid = obj.get('objectId')
        if isinstance(oid, dict):
            oid = oid.get('uuid')
        if not isinstance(oid, str) or not oid:
            continue
        sources = obj.get('sources')
        if not isinstance(sources, list) or not sources:
            continue

        obj_sources: dict = {}
        for entry in sources:
            if not isinstance(entry, dict):
                continue
            dataset_name = entry.get('datasetName')
            if not (isinstance(dataset_name, str) and dataset_name):
                continue
            # compute_latest_sources writes '' for a missing upstreamSource;
            # normalize back to None so keys match update_source_tally.
            upstream_source = entry.get('upstreamSource') or None
            event_type = entry.get('eventType') or None
            collection_type = entry.get('collectionType') or None
            ts_iso = entry.get('latestUpdatedDate')
            try:
                ts = pd.Timestamp(ts_iso).timestamp() if ts_iso else 0.0
            except (ValueError, TypeError):
                ts = 0.0
            key = (dataset_name, upstream_source, event_type)
            existing_entry = obj_sources.get(key)
            if existing_entry is None or ts > existing_entry[0]:
                obj_sources[key] = (ts, collection_type)

        if obj_sources:
            source_tally[oid] = obj_sources

    return source_tally


def _build_object_event_query(object_event_dataset: str,
                              tracked_datasets: List[str]) -> str:
    """Build the SSE query, excluding only tracked-feed kinematic events."""
    query = f"select * from {object_event_dataset}"
    if not tracked_datasets:
        return query

    tracked_dataset_literals = ', '.join(
        f"'{str(dataset).replace(chr(39), chr(39) * 2)}'"
        for dataset in sorted(set(tracked_datasets)))
    return (
        f"{query} where "
        f"{object_event_dataset}.eventType is null or "
        f"{object_event_dataset}.eventType <> 'KINEMATIC_UPDATE' or "
        f"{object_event_dataset}.`source`.datasetName is null or "
        f"{object_event_dataset}.`source`.datasetName "
        f"not in ({tracked_dataset_literals})")


def event_loop_manager(perspective_config: dict, 
                       tracked_datasets: Optional[List[str]] = None, 
                       track_event_datasets: Optional[Union[List[str], str]] = None,
                       ) -> None:
    """
        Creates asyncio coroutines, each of which
        concurrently waits for server-side event messages.
        Messages are sent to process_events()
        in a separate multiprocessing Process().

    Args:
        perspective_config: (dict)

        """

    if tracked_datasets is None:
        tracked_datasets = []

    logging.info("Object Manager: event_loop_manager starting...")

    message_queue = Queue()
    management_event_queue = Queue()
    management_signal = Event()  # signaled when management events arrive

    logging.info("Object Manager: Starting correlator process...")

    child_processes = []

    p = Process(
        target=launch_correlator_process,
        args=(
            message_queue,
            management_event_queue,
            perspective_config,
            tracked_datasets,
            track_event_datasets,
            management_signal,
        ),
        daemon=True)
    p.start()
    child_processes.append(p)

    if 'object_management_event_dataset' in perspective_config and \
        'superseded_object_dataset' in perspective_config:
        p2 = Process(
            target=remove_superseded_objects, 
                                args=(
                                perspective_config,
                                management_signal,
                                ),
            daemon=True)
        p2.start()
        child_processes.append(p2)
        
        p3 = Process(
            target=restore_superseded_objects, 
                                args=(
                                perspective_config,
                                management_signal,
                                ),
            daemon=True)
        p3.start()
        child_processes.append(p3)




    # start the SSE listener
    sse_listener_coroutine_list = []
    
    object_event_dataset = perspective_config['object_event_dataset']
    object_event_query = _build_object_event_query(
        object_event_dataset, tracked_datasets)
    logging.info(
        f"Object Manager: Setting up SSE listener for {object_event_dataset}: "
        f"{object_event_query}")
    sse_listener_coroutine_list.append(SSE_listener(
        message_queue, object_event_dataset, sql_query=object_event_query))
    # track_event_datasets can be a single dataset name or None
    if track_event_datasets:
        if isinstance(track_event_datasets, list):
            for track_dataset in track_event_datasets:
                
                logging.info(f"Object Manager: Setting up SSE listener for track dataset {track_dataset}")
                sse_listener_coroutine_list.append(SSE_listener(message_queue, 
                                                                track_dataset))
        else:
            # Single dataset string
            
            logging.info(f"Object Manager: Setting up SSE listener for track dataset {track_event_datasets}")
            sse_listener_coroutine_list.append(SSE_listener(message_queue, 
                                                            track_event_datasets))
    if 'object_management_event_dataset' in perspective_config:
        
        logging.info(f"Object Manager: Setting up SSE listener for management dataset {perspective_config['object_management_event_dataset']}")
        sse_listener_coroutine_list.append(SSE_listener(management_event_queue,
                                                    perspective_config['object_management_event_dataset']))
    
    
    logging.info(f"Object Manager: Starting {len(sse_listener_coroutine_list)} SSE listener coroutines...")
    try:
        asyncio.run(coroutine_launcher(sse_listener_coroutine_list)) # this is blocking
    finally:
        # Terminate child processes when SSE listeners exit (error or otherwise)
        for p in child_processes:
            if p.is_alive():
                logging.info(f"Object Manager: Terminating child process {p.pid}")
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    p.kill()
                p.join(timeout=3)

def launch_correlator_process(message_queue: Queue, 
                              management_event_queue: Queue,
                              dataset_config: dict,
                              tracked_datasets: List[str],
                              track_event_datasets: Optional[Union[List[str], str]] = None,
                              management_signal: Optional[Event] = None,
                              ) -> None:
    '''
    This function retrieves queued messages and sends them to process_events for processing.
    '''
    # Configure logging for this child process (forkserver children start fresh)
    _log_level = dataset_config.get('_log_level', logging.INFO)
    log_utils.get_logger(log_type='correlator', log_level=_log_level)

    global rc, wc, auth
    auth, rc, wc = instantiate_api_controllers()

    supersede_map = get_supersede_map(dataset_config, rc_instance=rc, auth_instance=auth)
    # Single startup download of the object dataset seeds BOTH the source_tally
    # (so partial writes don't leave stale entries behind Crucible's index-based
    # deep merge) and the identity_presence snapshot (fill-empty-only guard).
    source_tally, identity_presence = _load_startup_snapshot(dataset_config['object_dataset'])
    drop_counts = {}   # objectId -> consecutive drop count (for race-condition tolerance)
    DROP_THRESHOLD = 3  # require N consecutive drops before marking as deleted

    # >>> IDENTITY FIELD HANDLING (CHANGED) <<<
    # identity_presence is snapshotted ONCE at startup (above) then maintained
    # in-memory across cycles (see _update_identity_presence)

    logging.info(f"Object Manager: Correlator process started, waiting for SSE messages...")

    loop_count = 0
    while True:

        # refresh the access token if necessary
        rc.token = auth.get_token()
        wc.token = auth.get_token()

        loop_count += 1
        # Log queue status periodically (every 100 iterations = ~10 seconds)
        if loop_count % 100 == 0:
            
            logging.info(f"Object Manager: Queue check #{loop_count} - message_queue.empty()={message_queue.empty()}, management_queue.empty()={management_event_queue.empty()}")

        # Drain management events even when no SSE messages so the
        # remove/restore processes get signaled promptly.
        if not management_event_queue.empty():
            management_events = aggregate_SSE_messages(management_event_queue)
            if management_events:
                supersede_map = update_supersede_map(management_events, supersede_map)
                _apply_supersede_cache_updates(source_tally, identity_presence,
                                               drop_counts, supersede_map)
                if management_signal:
                    management_signal.set()
                logging.info(f"Object Manager: Processed {len(management_events)} management events (no SSE batch)")

        # if the queue is not empty, get all the messages in the queue 
        if message_queue.empty():
            # if the queue is empty, go back to the top of the loop
            time.sleep(0.1)
            continue

        logging.info(f"Object Manager: Found messages in queue, aggregating...")
        # collate all the messages in the queue for processing all at once
        SSE_msg = aggregate_SSE_messages(message_queue) 
        # Drain any remaining management events that arrived during aggregation
        management_events = aggregate_SSE_messages(management_event_queue)
        if management_events and management_signal:
            management_signal.set()
        logging.info(f"Object Manager: Aggregated {len(SSE_msg)} SSE messages, {len(management_events)} management events")
        
        
        # process the list of events
        supersede_map = asyncio.run(process_events(SSE_msg, 
                                   management_events,
                                    dataset_config, 
                                    tracked_datasets,
                                    supersede_map,
                                    source_tally,
                                    track_event_datasets,
                                    drop_counts=drop_counts,
                                    drop_threshold=DROP_THRESHOLD,
                                    identity_presence=identity_presence,
                                    )) # this is blocking


async def connect_with_timeout(url: str, session: "aiohttp.ClientSession", headers: dict, timeout: int) -> Optional[Any]:
    try:
        return await asyncio.wait_for( sse_client.EventSource(url, session=session, headers=headers, timeout=None).__aenter__(), timeout)
    except asyncio.TimeoutError:
        logging.error(f"Timeout error occurred: Initial connection to {url} could not be established within {timeout} seconds.")
        return None


async def SSE_listener(message_queue: Queue, dataset_name: str,
                       sql_query: Optional[str] = None) -> None:
    """
      Makes a client request to the server-side events endpoint to receive SSE
      messages and put the RAW events in the message_queue for subsequent
      processing.  Delegates the resilient connection/read/watchdog/backoff loop
      to object_utils.run_sse_listener so a large or stalled SSE payload can
      neither kill the listener nor hang it silently.

    Args:
        message_queue (Manager.Queue):
        dataset_name (str):
    """
    query = sql_query or "select * from " + dataset_name

    async def _on_event(event) -> None:
        message_queue.put(event)

    await run_sse_listener(query, auth, _on_event, label=f"Object Manager SSE [{dataset_name}]")


# Default number of worker threads for parallel objectId partitioning; override
# perspective-wide with the `num_object_manager_workers` config key.
NUM_OBJECT_MANAGER_WORKERS = 6


def _object_manager_shard_for_object(object_id: Any, n_shards: int) -> int:
    """Return a deterministic worker shard for an object primary key."""
    if n_shards <= 0:
        raise ValueError("n_shards must be greater than zero")
    if object_id is None:
        return 0
    digest = hashlib.md5(
        str(object_id).encode('utf-8'), usedforsecurity=False).hexdigest()
    return int(digest, 16) % n_shards


def _partition_events_by_object_shard(event_df: pd.DataFrame,
                                      primary_key: str,
                                      n_shards: int) -> List[tuple[int, pd.DataFrame]]:
    """Partition complete object groups into deterministic, non-empty shards."""
    if n_shards <= 0:
        raise ValueError("n_shards must be greater than zero")
    if event_df.empty:
        return [(0, event_df)]

    object_shards = event_df[primary_key].map(
        lambda object_id: _object_manager_shard_for_object(object_id, n_shards))
    return [
        (shard_id, event_df.loc[object_shards == shard_id])
        for shard_id in range(n_shards)
        if (object_shards == shard_id).any()
    ]


def _launch_partition(*args: Any) -> tuple:
    """Sync wrapper to run async process_object_partition in its own event loop (for ThreadPoolExecutor)."""
    return asyncio.run(process_object_partition(*args))


# ============================================================================
# >>> IDENTITY FIELD HANDLING (CHANGED) <<<
# ----------------------------------------------------------------------------
# WHAT CHANGED:
#   OLD behavior (single line, in _finalize_and_write):
#       df = df.loc[:, ~df.columns.str.contains('identity')]
#       -> dropped EVERY identity.* column on every write, so the object
#          manager could never set or update an object's identity at all.
#
#   NEW behavior (the three helpers + constant below):
#       "fill-empty-only" merge. An identity.* field is written only if the
#       object does not already have that field set. Once set, a field is
#       "write-once" and a later, conflicting value is dropped instead of
#       overwriting it. Brand-new objects still get all their identity fields.
#
# WHY:
#   Allows identities (e.g. dynamicIdentifier) to be populated and new
#   identifiers (e.g. link16TrackNumber) to be added, while preventing a
#   conflicting feed from flipping an established identity — which previously
#   caused supersede/delete/recreate churn.
#
# HOW (no per-cycle queries):
#   _load_startup_snapshot()    - one full-dataset download at startup, shared
#                                 with the source_tally seed
#   _build_identity_presence()  - builds the snapshot from that download
#   _update_identity_presence() - keeps that snapshot current in-memory
#   _apply_identity_fill_only() - blanks already-set fields before each write
# ============================================================================

# Identity fields allowed to overwrite an existing value despite the
# fill-empty-only policy in _apply_identity_fill_only().  Empty by default:
# every identity.* field is write-once (filled when absent, never replaced once
# set).  Add field names here (e.g. 'identity.standard.standardIdentity') to let
# specific fields update.
IDENTITY_OVERWRITABLE_FIELDS: set = set()


def _identity_value_present(value: Any) -> bool:
    """True only for identity values that should block later fill attempts."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip() != ''
    if isinstance(value, (list, tuple, set)):
        return any(_identity_value_present(item) for item in value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return True


def _build_identity_presence(records: list) -> dict:
    """Snapshot which identity.* fields each existing object already has set,
    built from the shared startup download (see ``_load_startup_snapshot``).

    Returns ``{objectId.uuid: {identity_col, ...}}``.  The result is maintained
    in-memory by ``_update_identity_presence`` as the correlator fills fields,
    so no per-cycle Crucible query is ever needed.  Records are normalized with
    ``pd.json_normalize`` — the same flattening the write path uses — so the
    dotted ``identity.*`` column names match exactly.
    """
    presence: dict = {}
    if not records:
        return presence

    existing = pd.json_normalize(records)
    if existing.empty or object_primary_key not in existing.columns:
        return presence

    identity_cols = [c for c in existing.columns if c.startswith('identity')]
    for col in identity_cols:
        owners = existing.loc[existing[col].map(_identity_value_present), object_primary_key]
        for oid in owners.astype(str):
            presence.setdefault(oid, set()).add(col)
    return presence


def _update_identity_presence(identity_presence: dict, event_df: pd.DataFrame,
                              skip_ids: Optional[set] = None) -> None:
    """Record identity.* fields seen in this cycle's events so the
    fill-empty-only guard stays write-once across cycles without re-querying.

    Any identity field an object carries in the incoming events is now either
    already present (no change) or just filled by this cycle's write, so adding
    the union of non-null identity fields keeps the snapshot current in-memory.
    """
    if identity_presence is None or object_primary_key not in event_df.columns:
        return
    identity_cols = [c for c in event_df.columns if c.startswith('identity')]
    if not identity_cols:
        return
    skip_ids = skip_ids or set()
    oid_str = event_df[object_primary_key].astype(str)
    for col in identity_cols:
        present_mask = event_df[col].map(_identity_value_present)
        for oid in oid_str[present_mask].unique():
            if oid in skip_ids:
                continue
            identity_presence.setdefault(oid, set()).add(col)


def _invert_identity_presence(identity_presence: dict) -> dict:
    """Invert ``{objectId: {identity_col, ...}}`` into
    ``{identity_col: {objectId, ...}}`` ONCE per cycle.

    ``_apply_identity_fill_only`` then does an O(1) dict lookup per column
    instead of rescanning every object for every identity column in every
    partition (previously O(identity_cols x total_objects) per partition, run in
    every worker thread while holding the GIL -> a hot spot).
    """
    owners_by_col: dict = {}
    if not identity_presence:
        return owners_by_col
    for oid, cols in identity_presence.items():
        for col in cols:
            owners_by_col.setdefault(col, set()).add(oid)
    return owners_by_col


def _apply_identity_fill_only(df: pd.DataFrame, identity_owners: dict) -> pd.DataFrame:
    """Fill-empty-only merge guard for identity fields.

    Blanks out (NaN) any ``identity.*`` field an object already has a value for,
    so those fields are not re-sent.  ``identity_owners`` is the pre-inverted
    ``{identity_col: {objectId, ...}}`` snapshot (see
    ``_invert_identity_presence``).  ``df_to_formatted_JSON`` drops NaN and the
    server-side partial update preserves the established value.  Identity fields
    the object does not yet have are left in place, so new identifiers (e.g.
    ``identity.link16TrackNumber``) are still merged in.

    This keeps anchored identifiers such as ``identity.dynamicIdentifier``
    stable, preventing conflicting feeds from flipping identity and triggering
    supersede/delete/recreate churn.
    """
    if not identity_owners or object_primary_key not in df.columns:
        return df

    identity_cols = [c for c in df.columns
                     if c.startswith('identity') and c not in IDENTITY_OVERWRITABLE_FIELDS]
    if not identity_cols:
        return df

    # Build a single (rows x identity_cols) blank mask via O(1) set lookups and
    # blank all already-set identity cells in ONE masked assignment, instead of a
    # per-column obj_id.isin(owners) + df.loc[...] = NaN assignment (each a
    # full-frame op over a wide object-dtype frame -> the #1 object_manager hot
    # spot on large state batches).
    oid = df[object_primary_key].astype(str).to_numpy()
    n = len(oid)
    blank = np.zeros((n, len(identity_cols)), dtype=bool)
    for j, col in enumerate(identity_cols):
        owners = identity_owners.get(col)
        if owners:
            blank[:, j] = np.fromiter((o in owners for o in oid), dtype=bool, count=n)
    if blank.any():
        df[identity_cols] = df[identity_cols].mask(blank)

    return df


async def process_object_partition(partition_df: pd.DataFrame,
                                   tracked_datasets: List[str],
                                   precomputed_sources: dict,
                                   enable_sources_array: bool,
                                   object_dataset: str,
                                   chunk_size: int,
                                   identity_owners: Optional[dict] = None,
                                   max_concurrent_writes: Optional[int] = None) -> tuple:
    """
    Process a partition of event_df through the full pipeline and write to Crucible:
    eventType routing -> format_* -> column cleanup -> groupby -> sources -> JSON -> write.

    Runs in a thread via _launch_partition for parallel CPU work (NumPy/pyproj release GIL).
    Writes use write_batch_chunked for chunking, retry, and failure tracking.

    Returns:
        tuple: (failed write UUIDs, UUIDs silently dropped because they were already deleted).
    """
    col_name_exclusions = ['crucibleHeader', 'source', 'eventType',
                           'positionCovariance', 'velocityCovariance',
                           'upstreamSource', 'latestSource', 'latestUpstreamSource',
                           'collectionType']

    _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
    partition_failed_ids = set()
    partition_deleted_ids = set()

    # [PERF] per-partition timing breakdown (gated by PERF_TIMING). The formerly
    # large "other" bucket is now split into coldrop/groupby/identity to pin the
    # object_manager hot spot.
    _pt = time.perf_counter if PERF_TIMING else _perf_disabled
    _pt_t = {'format': 0.0, 'json': 0.0, 'write': 0.0,
             'coldrop': 0.0, 'groupby': 0.0, 'identity': 0.0}
    _pt_start = _pt()

    async def _finalize_and_write(df: Optional[pd.DataFrame], pathway_name: str) -> None:
        """Column cleanup, groupby, sources, JSON conversion, then write."""
        if df is None or df.empty:
            return

        if 'crucibleHeader.edhControlSet' in df.columns:
            df['edhControlSet'] = df['crucibleHeader.edhControlSet']
        else:
            df['edhControlSet'] = ''

        # Single-pass column drop: OR the exclusion prefixes into ONE column mask
        # and slice once, instead of copying the whole (wide) frame once per
        # prefix (was N full-frame copies -> now 1).
        _tc = _pt()
        _excl = np.zeros(len(df.columns), dtype=bool)
        for col_name in col_name_exclusions:
            _excl = _excl | df.columns.str.startswith(col_name)
        df = df.loc[:, ~_excl]
        _pt_t['coldrop'] += _pt() - _tc

        if object_primary_key not in df.columns:
            return

        _tg = _pt()
        df = df.groupby(object_primary_key, as_index=False).agg('last')
        _pt_t['groupby'] += _pt() - _tg

        # >>> IDENTITY FIELD HANDLING (CHANGED) <<<
        # WAS (single line that wiped all identity columns on every write):
        #     df = df.loc[:, ~df.columns.str.contains('identity')]
        # NOW: fill-empty-only merge. Add new identity fields (e.g.
        # link16TrackNumber) but never overwrite an identity field the object
        # already has set (e.g. dynamicIdentifier). 
        _ti = _pt()
        df = _apply_identity_fill_only(df, identity_owners)
        _pt_t['identity'] += _pt() - _ti

        int_conversion(df, int_cols)

        if enable_sources_array and precomputed_sources is not None:
            df['sources'] = df[object_primary_key].map(precomputed_sources)

        _tj = _pt()
        JSON_object = _fast_df_to_nested_json(df)
        _pt_t['json'] += _pt() - _tj
        if not JSON_object:
            return

        _tw = _pt()
        _failed, deleted = await write_batch_chunked(
            JSON_object, object_dataset,
            wc.update_entity_record_batch_by_name, chunk_size,
            label=f' [{pathway_name}]: ',
            token_refresher=_refresh_token,
            max_concurrent_writes=max_concurrent_writes,
            transient_retry_attempts=2,
        )
        _pt_t['write'] += _pt() - _tw
        partition_deleted_ids.update(deleted)
        partition_failed_ids.update(
            object_id for object_id in (_get_object_id(record) for record in _failed)
            if object_id
        )

    # --- TRACKED pathway ---
    kin_df = partition_df[partition_df['eventType'] == 'empty']
    if not kin_df.empty:
        required_ecef_cols = ['ecefPosition.x', 'ecefPosition.y', 'ecefPosition.z']
        has_ecef = all(col in kin_df.columns for col in required_ecef_cols)
        if has_ecef:
            ecef_valid_mask = kin_df[required_ecef_cols].notna().any(axis=1)
            kin_df = kin_df[ecef_valid_mask]
        if has_ecef and not kin_df.empty:
            _tf = _pt()
            kin_df = format_tracker_kinematic_data(kin_df)
            _pt_t['format'] += _pt() - _tf
            await _finalize_and_write(kin_df, 'TRACKER KINEMATIC UPDATE')

    # --- STATE pathway ---
    state_df = partition_df[partition_df['eventType'] == 'STATE_UPDATE']
    if not state_df.empty:
        _tf = _pt()
        state_df = format_state_data(state_df)
        _pt_t['format'] += _pt() - _tf
        await _finalize_and_write(state_df, 'STATE UPDATE')

    # --- NON-TRACKED pathway ---
    kin_nt = partition_df[partition_df['eventType'] == 'KINEMATIC_UPDATE']
    if not kin_nt.empty:
        if 'source.datasetName' in kin_nt.columns and tracked_datasets:
            kin_nt = kin_nt[~kin_nt['source.datasetName'].isin(tracked_datasets)]
        if not kin_nt.empty:
            _tf = _pt()
            kin_nt = format_non_tracked_kinematic_data(kin_nt)
            _pt_t['format'] += _pt() - _tf
            await _finalize_and_write(kin_nt, 'NON-TRACKER KINEMATIC UPDATE')

    if PERF_TIMING:
        _pt_total = _pt() - _pt_start
        _pt_other = _pt_total - sum(_pt_t.values())
        logging.info(
            f" [PERF object_manager partition] "
            f"total={_pt_total:.3f}s format={_pt_t['format']:.3f}s "
            f"json={_pt_t['json']:.3f}s write={_pt_t['write']:.3f}s "
            f"coldrop={_pt_t['coldrop']:.3f}s groupby={_pt_t['groupby']:.3f}s "
            f"identity={_pt_t['identity']:.3f}s "
            f"other={_pt_other:.3f}s | rows={len(partition_df)}")
    return partition_failed_ids, partition_deleted_ids


async def process_events(event: List[Dict[Any, Any]],
                        management_events: List[Dict[Any, Any]],
                        perspective_config: Dict, 
                        tracked_datasets: List[str],
                        supersede_map: Dict[str, str],
                        source_tally: dict,
                        track_event_datasets: Optional[Union[List[str], str]] = None,
                        drop_counts: Optional[Dict[str, int]] = None,
                        drop_threshold: int = 3,
                        identity_presence: Optional[dict] = None,
                        ) -> Dict[str, str]:
    """
    Performs processing of events received from the
    SSE endpoint. SSE messages are accumulated in a 
    long list and processed in a single batch here.
    

    Args:
        event: (List[Dict[Any, Any]]):
        perspective_config (dict):
        tracked_datasets (List[str]):
        supersede_map (Dict[str, List[str]]):

    Returns:
        object: (not needed at the moment)
    """

    try:

        supersede_map = update_supersede_map(management_events,supersede_map)
        _apply_supersede_cache_updates(source_tally, identity_presence,
                                       drop_counts, supersede_map)

        msg_length = len(event)
        if msg_length >= 1:
            input_datasets = [perspective_config['object_event_dataset']]
            if isinstance(track_event_datasets, list):
                input_datasets.extend(track_event_datasets)
            elif track_event_datasets:
                input_datasets.append(track_event_datasets)
            info_string = (
                'Object Manager processing aggregated SSE batch from configured inputs: '
                + ', '.join(input_datasets))
            logging.info('-' * 50)
            logging.info(info_string)
            # convert incoming JSON message to DataFrame
            _perf = time.perf_counter if PERF_TIMING else _perf_disabled
            _t0 = _perf(); _t = _t0; _timings = {}
            event_df = pd.json_normalize(event)

            # tracker uses a different primary key than the rest of the
            # datasets, so we need to copy the tracker primary key to the 
            # object primary key

            if object_primary_key not in event_df.columns:
                event_df[object_primary_key] = pd.NA
                
            if tracker_primary_key in event_df.columns:
                # Vectorized primary key copy: fill objectId.uuid from objectId
                # where objectId.uuid is not already a string
                mask = ~event_df[object_primary_key].apply(lambda x: isinstance(x, str))
                event_df.loc[mask, object_primary_key] = event_df.loc[mask, tracker_primary_key]
                event_df.drop(columns=[tracker_primary_key], inplace=True)

            if supersede_map:
                logging.info(f"Applying supersede map to {len(event_df)} records")
                original_ids = event_df[object_primary_key].tolist()
                
                # Filter out records for deleted objects (those mapping to None)
                deleted_ids = [oid for oid in original_ids if oid in supersede_map and supersede_map[oid] is None]
                if deleted_ids:
                    logging.info(f"Filtering out {len(deleted_ids)} records for deleted objects: {deleted_ids[:10]}{'...' if len(deleted_ids) > 10 else ''}")
                    event_df = event_df[~event_df[object_primary_key].isin(deleted_ids)]
                    original_ids = event_df[object_primary_key].tolist()
                
                # Apply supersede mapping to remaining records
                event_df[object_primary_key]=event_df[object_primary_key].map( \
                    lambda x:rename_value(x,supersede_map))
                new_ids = event_df[object_primary_key].tolist()
                # Log any changes made by supersede map
                for orig, new in zip(original_ids, new_ids):
                    if orig != new:
                        logging.info(f"Supersede map applied: {orig} -> {new}")

            # fillna with placeholder string to avoid errors below
            if 'eventType' not in event_df.columns:
                event_df['eventType'] = 'empty'
            event_df['eventType'] = event_df['eventType'].fillna('empty')

            # --- Drop tracked-feed KINEMATIC_UPDATE events up front ---
            # For a tracked feed the direct KINEMATIC_UPDATE events are always
            # discarded downstream anyway: the NON-TRACKED write pathway filters
            # out any KINEMATIC_UPDATE whose source.datasetName is a tracked
            # dataset, and update_source_tally skips the same rows.  Removing
            # them here means the expensive single-threaded source-tally +
            # coalesce below never touch them.  Rows with a missing/NaN
            # source.datasetName are kept (they are not filtered downstream
            # either — .isin(...) yields False for NaN).
            if tracked_datasets and 'source.datasetName' in event_df.columns:
                _drop_tracked_kin = (
                    (event_df['eventType'] == 'KINEMATIC_UPDATE')
                    & event_df['source.datasetName'].isin(tracked_datasets)
                )
                _n_drop = int(_drop_tracked_kin.sum())
                if _n_drop:
                    event_df = event_df[~_drop_tracked_kin]
                    logging.info(f"Dropped {_n_drop} tracked-feed KINEMATIC_UPDATE "
                                 f"event(s) up front (discarded downstream anyway)")

            # --- Coalesce redundant events per object+source up front ---
            # The write pathways collapse to one row per objectId per eventType
            # (groupby.agg('last')), and update_source_tally only keeps the
            # latest entry per (datasetName, upstreamSource, eventType) key.  So
            # many high-rate events for the same object+source in one batch are
            # pure overhead for the single-threaded source tally (which iterates
            # row-by-row) and the per-partition formatting.  Collapse them to the
            # last row per (objectId, eventType, source.datasetName,
            # upstreamSource, collectionType) BEFORE the source tally so those
            # steps run over unique rows only.  dropna=False keeps track-head
            # rows whose source columns are NaN; agg('last') matches the
            # write-path last-non-null semantics; keying on the full source
            # identity preserves the sources array for multi-source objects.
            if len(event_df) > 0 and object_primary_key in event_df.columns:
                _coalesce_keys = [c for c in (object_primary_key, 'eventType',
                                              'source.datasetName', 'upstreamSource',
                                              'collectionType') if c in event_df.columns]
                _n_before = len(event_df)
                event_df = event_df.groupby(
                    _coalesce_keys, as_index=False, sort=False, dropna=False
                ).agg('last')
                _n_after = len(event_df)
                if _n_after < _n_before:
                    logging.info(f"Coalesced {_n_before} events -> {_n_after} unique "
                                 f"object/source rows before source-tally + processing")

            # Update source tally and set latestSource / latestUpstreamSource
            enable_sources_array = perspective_config.get('enable_sources_array', True)
            if enable_sources_array:
                update_source_tally(source_tally, event_df, tracked_datasets, track_event_datasets)
                unique_obj_ids = event_df[object_primary_key].dropna().unique()
                precomputed_sources = {
                    obj_id: compute_latest_sources(source_tally, obj_id)
                    for obj_id in unique_obj_ids
                    if isinstance(obj_id, str)
                }
            else:
                precomputed_sources = None
            if 'source.datasetName' in event_df.columns:
                event_df['latestSource'] = event_df['source.datasetName']
            if 'upstreamSource' in event_df.columns:
                event_df['latestUpstreamSource'] = event_df['upstreamSource']

            logging.info('')
            logging.info(f'# records from event: {len(event_df)}')
            # Log the eventType distribution to help debug what's coming through
            eventType_counts = event_df['eventType'].value_counts().to_dict()
            logging.info(f'eventType distribution: {eventType_counts}')
            logging.info('')


            if extra_diagnostics:
                # Diagnostic: Compare objectIds in event vs objectIds in dataset
                try:
                    event_object_ids = set(event_df[object_primary_key].unique())
                    object_dataset_query = f"SELECT {perspective_config['object_dataset']}.objectId.uuid FROM {perspective_config['object_dataset']}"
                    object_dataset_result = await asyncio.to_thread(rc.search, object_dataset_query, format='dataframe')
                    object_dataset_object_ids = set(object_dataset_result['uuid'].tolist()) if not object_dataset_result.empty else set()
                    
                    in_event_not_dataset = event_object_ids - object_dataset_object_ids
                    in_dataset_not_event = object_dataset_object_ids - event_object_ids
                    in_both = event_object_ids & object_dataset_object_ids
                    
                    logging.info(f"=== ObjectId Comparison ===")
                    logging.info(f"Total in event: {len(event_object_ids)}")
                    logging.info(f"Total in dataset: {len(object_dataset_object_ids)}")
                    logging.info(f"In both: {len(in_both)}")
                    logging.info(f"In event but NOT in dataset: {len(in_event_not_dataset)}")
                    if in_event_not_dataset:
                        logging.info(f"  IDs: {list(in_event_not_dataset)[:10]}{'...' if len(in_event_not_dataset) > 10 else ''}")
                    logging.info(f"===========================")
                except Exception as diag_error:
                    logging.warning(f"Diagnostic comparison failed: {diag_error}")


            _timings['preprocess'] = _perf() - _t; _t = _perf()
            # --- Partition by objectId and process + write in parallel ---
            unique_ids = event_df[object_primary_key].dropna().unique()
            configured_workers = max(1, int(perspective_config.get(
                'num_object_manager_workers', NUM_OBJECT_MANAGER_WORKERS)))
            shard_partitions = _partition_events_by_object_shard(
                event_df, object_primary_key, configured_workers)
            partitions = [partition for _, partition in shard_partitions]
            active_workers = len(partitions)
            chunk_size = int(perspective_config['batch_update_chunk_size'])
            object_dataset = perspective_config['object_dataset']
            max_concurrent_writes = int(perspective_config.get(
                'batch_write_max_concurrent',
                os.getenv('BATCH_WRITE_MAX_CONCURRENT', '2')))

            # identity_presence is snapshotted once at startup and maintained
            # in-memory; partitions read it (disjoint by objectId) and it is
            # refreshed below from this cycle's events. Invert it to
            # {identity_col: {objectId,...}} ONCE per cycle so each partition does
            # an O(1) per-column lookup instead of rescanning every object for
            # every identity column in every worker thread.
            identity_owners = _invert_identity_presence(identity_presence)

            if active_workers <= 1:
                # Single partition — run in-process directly
                failed_write_ids, deleted_write_ids = await process_object_partition(
                    partitions[0], tracked_datasets, precomputed_sources,
                    enable_sources_array, object_dataset, chunk_size,
                    identity_owners, max_concurrent_writes)
                failed_write_ids.update(deleted_write_ids)
            else:
                logging.info(
                    f"Partitioned {len(event_df)} records into {active_workers} active "
                    f"stable object shards of {configured_workers} configured "
                    f"({[(shard_id, len(partition)) for shard_id, partition in shard_partitions]})")

                # Each thread gets its own event loop via _launch_partition
                # → real parallelism for NumPy/pyproj C code (GIL released)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=active_workers) as executor:
                    futures = [
                        loop.run_in_executor(
                            executor,
                            _launch_partition,
                            partition,
                            tracked_datasets,
                            precomputed_sources,
                            enable_sources_array,
                            object_dataset,
                            chunk_size,
                            identity_owners,
                            max_concurrent_writes,
                        )
                        for partition in partitions
                    ]
                    results = await asyncio.gather(*futures)
                failed_write_ids = set()
                deleted_write_ids = set()
                for result in results:
                    if result:
                        failed_ids, deleted_ids = result
                        failed_write_ids.update(failed_ids)
                        deleted_write_ids.update(deleted_ids)
                failed_write_ids.update(deleted_write_ids)

            _timings['partition_exec'] = _perf() - _t
            if PERF_TIMING:
                _timings['total'] = _perf() - _t0
                _timings['other'] = _timings['total'] - (
                    _timings.get('preprocess', 0.0)
                    + _timings.get('partition_exec', 0.0))
                logging.info(
                    f" [PERF object_manager] "
                    f"total={_timings['total']:.3f}s "
                    f"preprocess={_timings.get('preprocess', 0.0):.3f}s "
                    f"partition_exec={_timings.get('partition_exec', 0.0):.3f}s "
                    f"other={_timings['other']:.3f}s "
                    f"| event_rows={len(event_df)} workers={active_workers} "
                    f"configured_shards={configured_workers} "
                    f"unique_objects={len(unique_ids)}")
            # Keep the in-memory identity snapshot current (write-once across
            # cycles) using this cycle's events, skipping objects that dropped.
            # >>> IDENTITY FIELD HANDLING (CHANGED) <<<
            _update_identity_presence(identity_presence, event_df, skip_ids=failed_write_ids)

            hard_failed_ids = failed_write_ids - deleted_write_ids
            if hard_failed_ids:
                logging.warning(
                    f"{len(hard_failed_ids)} object update(s) remain failed after "
                    f"the bounded transient retries: "
                    f"{list(hard_failed_ids)[:5]}"
                    f"{'...' if len(hard_failed_ids) > 5 else ''}")

            # Warn if records were rejected as not-found but aren't in supersede_map
            if deleted_write_ids:
                unknown_deletes = deleted_write_ids - set(supersede_map.keys())
                if unknown_deletes:
                    logging.warning(
                        f"{len(unknown_deletes)} record(s) not found in dataset but NOT in supersede_map "
                        f"(may have been deleted externally): {list(unknown_deletes)[:5]}"
                        f"{'...' if len(unknown_deletes) > 5 else ''}"
                    )
                known = deleted_write_ids - unknown_deletes
                if known:
                    logging.info(
                        f"{len(known)} record(s) not found in dataset (already superseded): "
                        f"{list(known)[:5]}{'...' if len(known) > 5 else ''}"
                    )

                # Increment drop counter for each dropped ID. Only mark as deleted
                # after consecutive drops exceed threshold (tolerates transient absence
                # during supersede/restore move operations).
                if drop_counts is not None:
                    for oid in deleted_write_ids:
                        drop_counts[oid] = drop_counts.get(oid, 0) + 1
                        if drop_counts[oid] >= drop_threshold:
                            prev = supersede_map.get(oid)
                            if prev is not None:
                                logging.info(
                                    f"  Updating supersede_map['{oid}']: {repr(prev)} -> None "
                                    f"(dropped {drop_counts[oid]} consecutive times)"
                                )
                                supersede_map[oid] = None
                        else:
                            logging.info(
                                f"  Drop count for '{oid}': {drop_counts[oid]}/{drop_threshold}"
                            )

            # Reset drop counters for objects that wrote successfully this cycle
            if drop_counts is not None:
                successful_ids = set()
                for partition in partitions:
                    for oid_val in partition[object_primary_key].dropna().unique():
                        if oid_val not in failed_write_ids:
                            successful_ids.add(oid_val)
                for oid in successful_ids:
                    if oid in drop_counts:
                        del drop_counts[oid]
    
        else:

            status_string = 'Empty message received from ' + \
                            perspective_config['object_event_dataset']
            logging.info(status_string)

        return supersede_map
    
    except Exception as e:
        logging.error(f"An error occurred in process_events: {e}")
        logging.error(traceback.format_exc())
        return supersede_map

def calculate_cap(row: pd.Series) -> float:
    '''
    This function is used to calculate the circular area probable (CAP) of an object
    based on the semi-major and semi-minor axes of the uncertainty ellipse.
    
    CAP radius = 0.75 * sqrt(a^2 + b^2)  [ref: JPL D-4710]
    CAP area = pi * (0.75)^2 * (a^2 + b^2) = pi * 0.5625 * (a^2 + b^2)
    '''
    semiMajorAxisLength = row['estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMajorAxisLength']
    semiMinorAxisLength = row['estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMinorAxisLength']
    # Handle missing or invalid values
    if pd.isna(semiMajorAxisLength) or pd.isna(semiMinorAxisLength):
        return np.nan
    return np.pi * 0.5625 * (semiMajorAxisLength**2 + semiMinorAxisLength**2)

def format_tracker_kinematic_data(kin_df: pd.DataFrame) -> pd.DataFrame:

    logging.info(f"Formatting {len(kin_df)} records through tracked kinematic pathway")
    # The observation time (estimatedKinematics.kinematicsTimestamp) must only
    # come from the genuine object-event estimatedKinematics.kinematicsTimestamp,
    # which the tracker/fuser carry through as interceptTimestamp. Principal
    # track HEADS strip interceptTimestamp and instead carry that same genuine
    # time as trackUpdatedTimestamp (the fuser drops head rows with no
    # interceptTimestamp and sets trackUpdatedTimestamp = interceptTimestamp,
    # with no current-time fallback), so trackUpdatedTimestamp on a head IS the
    # genuine observation time. Do NOT synthesize from trackOriginatedTimestamp,
    # crucibleHeader.updatedDate, or the current time — when neither a genuine
    # interceptTimestamp nor trackUpdatedTimestamp is present, leave
    # kinematicsTimestamp unset rather than fabricating one.
    if 'interceptTimestamp' in kin_df.columns:
        kin_df['estimatedKinematics.kinematicsTimestamp'] = kin_df['interceptTimestamp']
    elif 'trackUpdatedTimestamp' in kin_df.columns:
        kin_df['estimatedKinematics.kinematicsTimestamp'] = kin_df['trackUpdatedTimestamp']
    else:
        logging.warning("format_tracker_kinematic_data: no interceptTimestamp or "
                        "trackUpdatedTimestamp present - leaving "
                        "estimatedKinematics.kinematicsTimestamp unset (no synthesized timestamp)")



    # transform kinematics from ECEF to WGS84:
    # first, transform position
    kin_df['estimatedKinematics.position.latitude'], \
        kin_df['estimatedKinematics.position.longitude'], \
        kin_df['estimatedKinematics.position.altitude'] = \
        ecef_transformer.transform(kin_df['ecefPosition.x'],
                                 kin_df['ecefPosition.y'],
                                 kin_df['ecefPosition.z'],
                                 inplace=False,radians=True)

    # then, transform velocity (vectorized ECEF -> ENU)
    if 'ecefVelocity.dx' in kin_df.columns:
        _lat = pd.to_numeric(kin_df['estimatedKinematics.position.latitude'], errors='coerce').values
        _lon = pd.to_numeric(kin_df['estimatedKinematics.position.longitude'], errors='coerce').values
        _clon = np.cos(_lon); _slon = np.sin(_lon)
        _clat = np.cos(_lat); _slat = np.sin(_lat)

        _dx = pd.to_numeric(kin_df['ecefVelocity.dx'], errors='coerce').values
        _dy = pd.to_numeric(kin_df['ecefVelocity.dy'], errors='coerce').values
        _dz = pd.to_numeric(kin_df['ecefVelocity.dz'], errors='coerce').values

        _v_east = -_slon * _dx + _clon * _dy
        _v_north = -_slat * _clon * _dx - _slat * _slon * _dy + _clat * _dz
        _v_up = _clat * _clon * _dx + _clat * _slon * _dy + _slat * _dz

        kin_df['estimatedKinematics.velocity.eastSpeed'] = _v_east
        kin_df['estimatedKinematics.velocity.northSpeed'] = _v_north
        kin_df['estimatedKinematics.velocity.downSpeed'] = -_v_up


    # then transform covariance matrix to get uncertainty ellipse
    try:
        if 'positionCovariance.xx' in kin_df.columns:
            # Ensure covariance columns are numeric (SSE may return strings)
            cov_cols = ['positionCovariance.xx', 'positionCovariance.xy', 'positionCovariance.xz',
                        'positionCovariance.yy', 'positionCovariance.yz', 'positionCovariance.zz']
            for col in cov_cols:
                if col in kin_df.columns:
                    kin_df[col] = pd.to_numeric(kin_df[col], errors='coerce')

            # --- Vectorized covariance rotation + uncertainty ellipse ---
            # Replaces 4 row-by-row .apply() passes (build matrix, rotate
            # ECEF->ENU, ellipse, CAP) with batched numpy over all rows at once.
            _N = len(kin_df)

            # lat/lon (radians) trig for the ECEF->ENU rotation matrices
            _cov_lat = pd.to_numeric(kin_df['estimatedKinematics.position.latitude'], errors='coerce').values
            _cov_lon = pd.to_numeric(kin_df['estimatedKinematics.position.longitude'], errors='coerce').values
            _cclon = np.cos(_cov_lon); _cslon = np.sin(_cov_lon)
            _cclat = np.cos(_cov_lat); _cslat = np.sin(_cov_lat)

            # ECEF covariance matrices (N, 3, 3)
            _C = np.empty((_N, 3, 3))
            _C[:, 0, 0] = kin_df['positionCovariance.xx'].values
            _C[:, 0, 1] = _C[:, 1, 0] = kin_df['positionCovariance.xy'].values
            _C[:, 0, 2] = _C[:, 2, 0] = kin_df['positionCovariance.xz'].values
            _C[:, 1, 1] = kin_df['positionCovariance.yy'].values
            _C[:, 1, 2] = _C[:, 2, 1] = kin_df['positionCovariance.yz'].values
            _C[:, 2, 2] = kin_df['positionCovariance.zz'].values

            # ECEF->ENU rotation matrices (N, 3, 3); rows are [East, North, Up]
            _R = np.zeros((_N, 3, 3))
            _R[:, 0, 0] = -_cslon
            _R[:, 0, 1] =  _cclon
            _R[:, 1, 0] = -_cslat * _cclon
            _R[:, 1, 1] = -_cslat * _cslon
            _R[:, 1, 2] =  _cclat
            _R[:, 2, 0] =  _cclat * _cclon
            _R[:, 2, 1] =  _cclat * _cslon
            _R[:, 2, 2] =  _cslat

            # Rotated ENU covariance: R @ C @ R^T for all rows at once
            _enu = np.einsum('nij,njk,nlk->nil', _R, _C, _R)
            _cov2d = _enu[:, :2, :2]

            # Uncertainty ellipse via batched SVD (matches filterpy.covariance_ellipse:
            # orientation = atan2(U[1,0], U[0,0]); axes = 2.448 * sqrt(singular values)).
            _major = np.full(_N, np.nan)
            _minor = np.full(_N, np.nan)
            _azimuth = np.full(_N, np.nan)
            _valid = np.isfinite(_cov2d).all(axis=(1, 2))
            if _valid.any():
                _U, _s, _ = np.linalg.svd(_cov2d[_valid])
                _orientation = np.arctan2(_U[:, 1, 0], _U[:, 0, 0])
                _major[_valid] = 2.448 * np.sqrt(_s[:, 0])
                _minor[_valid] = 2.448 * np.sqrt(_s[:, 1])
                # filterpy angle is CCW from East; convert to azimuth CW from North
                _az = (np.pi / 2 - _orientation) % (2 * np.pi)
                _az = np.where(_az >= 6.283185, 0.0, _az)
                _azimuth[_valid] = _az

            kin_df["estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMajorAxisLength"] = _major
            kin_df["estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMinorAxisLength"] = _minor
            kin_df["estimatedKinematics.position.uncertainty.uncertaintyEllipse.orientation"] = _azimuth

            # calculate the CAP for 95% confidence (vectorized), then map to TQ
            _cap = np.pi * 0.5625 * (_major**2 + _minor**2)
            kin_df["trackQuality"] = pd.Series(_cap, index=kin_df.index).map(find_tq)

            # fill in NaNs in uncertainty ellipse with 0
            # this case was needed for the random data generator, which doesn't have valid covariance matrices
            kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.orientation'] = \
                kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.orientation'].fillna(0)
            kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMajorAxisLength'] = \
                kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMajorAxisLength'].fillna(0)
            kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMinorAxisLength'] = \
                kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMinorAxisLength'].fillna(0)
            
            # constrain the uncertainty ellipse to be valid, between 0 and 180 degrees
            kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.orientation'] = \
                kin_df['estimatedKinematics.position.uncertainty.uncertaintyEllipse.orientation'].apply(lambda x: x % 6.283185307179586)

    except Exception as e:
        logging.error(f"Covariance→uncertainty ellipse failed: {e}")
        logging.error(traceback.format_exc())
        if 'positionCovariance.xx' in kin_df.columns:
            logging.error(f"  positionCovariance.xx dtype: {kin_df['positionCovariance.xx'].dtype}, "
                          f"sample: {kin_df['positionCovariance.xx'].iloc[0]!r}")
        logging.error(' Skipping covariance/trackQuality step. ')
        logging.error(' ')

    # remove all columns that are not in the schema 
    keep_list = ['estimatedKinematics',
                 'objectId',
                 'trackId',
                 'edhControlSet',
                 'crucibleHeader.edhControlSet',
                 'trackQuality',
                ]
    drop_list = []

    for col in kin_df.columns:
        
        if any([keep in col for keep in keep_list]):
            pass
        else:
            drop_list.append(col)

    kin_df = kin_df.drop(columns=drop_list)
    kin_df.rename(columns={'trackId':'trackId.uuid'},inplace=True)

    return kin_df
  

def format_state_data(state_df: pd.DataFrame) -> pd.DataFrame:
    
    # kinematic object event updates are ignored (one combined column mask +
    # one slice, instead of a full-frame copy per prefix)
    state_df = state_df.loc[:, ~state_df.columns.str.startswith(('ecef', 'estimatedKinematics'))]

    return state_df

def format_non_tracked_kinematic_data(kin_df_nontracked: pd.DataFrame) -> pd.DataFrame:

    # Guard: verify we have at minimum objectId and some kinematic data
    has_kinematics = any('estimatedKinematics' in col for col in kin_df_nontracked.columns)
    if not has_kinematics:
        logging.warning("format_non_tracked_kinematic_data: No estimatedKinematics columns found - returning empty DataFrame")
        return pd.DataFrame(columns=[object_primary_key])

    # Keep only kinematic / objectId / edh columns: build the drop list once and
    # drop in a single call instead of one inplace drop per unwanted column.
    _keep_substrings = ('estimatedKinematics', 'objectId', 'edhControlSet')
    _drop_cols = [c for c in kin_df_nontracked.columns
                  if not any(k in c for k in _keep_substrings)]
    if _drop_cols:
        kin_df_nontracked = kin_df_nontracked.drop(columns=_drop_cols)



    if "estimatedKinematics.uncertainty.uncertaintyEllipse.semiMajorAxisLength" in kin_df_nontracked.columns:
        kin_df_nontracked.rename( \
            columns={"estimatedKinematics.uncertainty.uncertaintyEllipse.semiMajorAxisLength": "estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMajorAxisLength",
                                          "estimatedKinematics.uncertainty.uncertaintyEllipse.semiMinorAxisLength": "estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMinorAxisLength",
                                          "estimatedKinematics.uncertainty.uncertaintyEllipse.orientation": "estimatedKinematics.position.uncertainty.uncertaintyEllipse.orientation"}, 
                                          inplace=True)

        # calculate the CAP for 95% confidence
        if "estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMajorAxisLength" \
                    in kin_df_nontracked.columns \
                    and "estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMinorAxisLength" \
                    in kin_df_nontracked.columns:
            # Vectorized CAP (replaces df.apply(calculate_cap, axis=1)):
            # CAP = pi * 0.5625 * (a^2 + b^2); NaN in either axis -> NaN -> find_tq(NaN)=0,
            # matching calculate_cap's pd.isna guard.
            _maj = pd.to_numeric(
                kin_df_nontracked["estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMajorAxisLength"],
                errors='coerce')
            _min = pd.to_numeric(
                kin_df_nontracked["estimatedKinematics.position.uncertainty.uncertaintyEllipse.semiMinorAxisLength"],
                errors='coerce')
            _cap = np.pi * 0.5625 * (_maj ** 2 + _min ** 2)
            kin_df_nontracked["trackQuality"] = _cap.map(find_tq)

    return kin_df_nontracked

def fill_empty_cell_with_list(x: Any) -> Any:
    if x == 'empty':
        return []
    else:
        return x

def run(stream_manager_perspective: str, log_level: Optional[str] = None) -> None:
    """
    Sets logging, gathers config files, and calls event_loop_manager().

    Args:
        stream_manager_header_name (str): name of dataset with stream manager header config
        log_level (str)): 'INFO', 'WARN', or 'ERROR'
    """
    # set up logging:
    if log_level is None:
        log_level = 'info'
    # Convert log_level to upper case to allow the user to
    # specify --log=DEBUG or --log=debug

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    # Configure default logging
    log_utils.get_logger(log_type='correlator', log_level=numeric_level)

    # Initialize controllers for the parent process
    global rc, wc, auth
    auth, rc, wc = instantiate_api_controllers()

    # get config dictionaries from Crucible stream manager datasets:

    config_list \
        = find_and_validate_configs(stream_manager_perspective,include_scripts=True, rc_instance=rc, caller_globals=globals())
    
    # Get the perspective config
    # by choosing the first config that has a principal_track_event_dataset or track_event_dataset
    # and is not disabled.
    # Failing that, choose the first config that is not disabled.
    perspective_config=None
    for config in config_list:
        if 'disabled' not in config or \
                not config['disabled']:
            if 'principal_track_event_dataset' in config or 'track_event_dataset' in config:
                perspective_config = config
                break
    if perspective_config is None:
        for config in config_list:
            if 'disabled' not in config or \
                    not config['disabled']:
                perspective_config = config
                break
    logging.info(f"Grabbing Stream Manager config for the {perspective_config['origin_dataset']} dataset to find dataset names for the object manager.")

    # num_object_manager_workers is a PERSPECTIVE-level setting: exactly one
    # object manager runs per perspective and reads this from perspective_config.
    # It is stored per-feed, so warn if feed configs disagree — otherwise an
    # arbitrary config's value would silently win.
    _om_vals = set()
    for _c in config_list:
        if _c.get('disabled'):
            continue
        _v = _c.get('num_object_manager_workers')
        if _v is None:
            continue
        try:
            _om_vals.add(int(_v))
        except (TypeError, ValueError):
            pass
    if len(_om_vals) > 1:
        _om_used = int(perspective_config.get('num_object_manager_workers', NUM_OBJECT_MANAGER_WORKERS))
        logging.warning(
            f"num_object_manager_workers differs across {stream_manager_perspective} feed configs "
            f"{sorted(_om_vals)}; using {_om_used} (from perspective_config). "
            f"num_object_manager_workers is perspective-level — set it identically in every config.")

    tracked_datasets,track_event_datasets = find_tracked_datasets(config_list)
    logging.info("-" * 50)
    logging.info(" Origin datasets using tracker kinematics (track dataset defined and not in bypass mode):")
    logging.info(f"Tracked datasets: {tracked_datasets}")
    logging.info(f"Track event datasets: {track_event_datasets}")
    logging.info("-" * 50)

    # Store log level for child processes (forkserver children don't inherit logging config)
    perspective_config['_log_level'] = numeric_level

    # run main program and retry if there is a connection failure
    while True:
        try:
            event_loop_manager(perspective_config, \
                               tracked_datasets=tracked_datasets, \
                               track_event_datasets=track_event_datasets, \
                               )
            # note: exceptions thrown in event_loop_manager and functions
            # called within are caught here and the program will retry.
            # All tasks are cancelled after exiting asyncio.gather()
            # in event_loop_manager.
        except (KeyboardInterrupt, SystemExit):
            logging.info("Object Manager: Received shutdown signal, exiting...")
            break
        except BaseException as err:
            logging.error('-' * 50)
            logging.error("Error from __main__")
            logging.error(f"{err=}, {type(err)=}")
            logging.error(traceback.format_exc())
            logging.error(' ')
            logging.error(' pausing before restarting')
            logging.error('-' * 50)
            time.sleep(5)
            continue


def rename_value(cell: str, supersede_map: Dict[str, str]) -> str:
    '''
    This function is used to rename the objectId of an object that has been superseded.
    This needs to be defined here because it uses the supersede_map can't be
    passed as an argument to the map function.
    
    If the supersede_map maps the cell to None (deleted object), returns the original
    cell value unchanged - the record should not be updated with a None objectId.
    '''
    if cell in supersede_map:
        superseded_by = supersede_map[cell]
        # Only return the superseded value if it's not None (deleted)
        if superseded_by is not None:
            return superseded_by
    return cell

def find_tq(cap : float) -> int:
    '''
    This function is used to calculate the track quality (TQ) of an object
    based on the circular area probable ( CAP) of the object.
    The CAP is given in square meters, and the TQ is an integer from 0 to 15.
    This is based on MIL-STD-6016G.
    '''
    # Handle missing or invalid values
    if cap is None:
        return 0
    if pd.isna(cap):
        return 0
    try:
        # Check for negative or invalid values
        if cap < 0:
            return 0
        # Check for infinity or very large values
        if not np.isfinite(cap):
            return 0
        if cap >= cap_to_tq[2]:
            return 1
        for key, val in cap_to_tq.items():
            if cap < val:
                return key
        return 0
    except (TypeError, ValueError, OverflowError):
        return 0

def find_tracked_datasets(config_list: List[Dict[str, Any]]) -> List[str]:
    """
    Returns two lists:
    datasets that are being tracked by the stream manager and
    destination datasets for tracking events. (There may be more than one but it's unlikely.)
    
    Kinematic data routing logic:
        - BYPASS MODE: bypass_tracker=true uses KINEMATIC_UPDATE object events,
            regardless of crucible_tracker or configured track datasets.
    - PASSTHROUGH MODE (tracked pathway via passthrough): If key crucible_tracker contains 'passthrough' substring
      AND a track_event_dataset is defined, the dataset is added to tracked_datasets.
      The tracker runs in passthrough mode (no Kalman filter), and duplicate KINEMATIC_UPDATE
      object events are filtered out.
    - TRACKED MODE (tracked pathway): If track_event_dataset or principal_track_event_dataset is
      defined for ANY config, then kinematic data from that dataset is routed through
      format_tracker_kinematic_data().
    - DEFAULT PASSTHROUGH: If neither track_event_dataset is defined nor crucible_tracker is set,
      defaults to passthrough (non-tracked) mode for backward compatibility.
    
    A dataset is added to tracked_datasets only if:
    - It is not disabled
    - It has a principal_track_event_dataset or track_event_dataset defined
    - crucible_tracker='passthrough' is treated as tracked (tracker runs in passthrough mode)
    
    The SSE listener for tracker kinematics uses principal_track_event_dataset,
    falling back to track_event_dataset.
    """
    tracked_datasets = []
    tracker_dataset = None
    for config in config_list:
        origin_dataset = config.get('origin_dataset', 'unknown')
        
        # Skip disabled configs
        if config.get('disabled', False):
            logging.debug(f" [{origin_dataset}]: Skipping - config is disabled")
            continue

        if uses_object_event_kinematics(config):
            logging.info(
                f" [{origin_dataset}]: Bypass mode - using object-event kinematics")
            continue
        
        # Check crucible_tracker value for routing info
        crucible_tracker = config.get('crucible_tracker', '')
        is_passthrough = crucible_tracker and 'passthrough' in crucible_tracker.lower()
        
        # Check if this config has a track event dataset defined
        # Use principal_track_event_dataset if present, else fall back to track_event_dataset
        config_tracker_dataset = None
        if 'principal_track_event_dataset' in config:
            config_tracker_dataset = config['principal_track_event_dataset']
        elif 'track_event_dataset' in config:
            config_tracker_dataset = config['track_event_dataset']
        
        # Only consider this a tracked dataset if it has a track event dataset defined
        if config_tracker_dataset:
            if not tracker_dataset:
                # Prefer principal track EVENTS for SSE kinematic updates. Head
                # writes are best-effort/preload state and can lag under write
                # contention; events are the reliable live stream and carry the
                # interceptTimestamp needed below to set kinematicsTimestamp.
                tracker_dataset = config_tracker_dataset
            # Add to tracked datasets list
            if origin_dataset and origin_dataset != 'unknown':
                tracked_datasets.append(origin_dataset)
                mode_label = "passthrough" if is_passthrough else "tracked"
                logging.info(f" [{origin_dataset}]: {mode_label.capitalize()} mode - using track dataset: {tracker_dataset}, will use tracked kinematic pathway")
        else:
            logging.info(f" [{origin_dataset}]: Default passthrough mode - no track event dataset defined, will use non-tracked kinematic pathway")
    
    return tracked_datasets, tracker_dataset


def remove_superseded_objects(perspective_config: Dict[str, Any], management_signal: Optional[Event] = None) -> None:
    """
    Removes duplicate objects from the object dataset and writes them to the superseded object dataset.
    Waits for management_signal (set by correlator when new management events arrive) before each cycle.
    """
    # Configure logging for this child process (forkserver children start fresh)
    _log_level = perspective_config.get('_log_level', logging.INFO)
    log_utils.get_logger(log_type='correlator', log_level=_log_level)

    global rc, wc, auth
    auth, rc, wc = instantiate_api_controllers()
    # Build the historical map once at startup, then keep it warm with small
    # delta reads.  Rebuilding the full 30-day ObjectManagementEvents map every
    # 60s was repeatedly scanning LIMIT-sized (10k-row) history on active pods.
    supersede_map, management_cursor = get_supersede_map(
        perspective_config, rc_instance=rc, auth_instance=auth,
        return_latest_timestamp=True)
    try:
        id_cache_refresh_seconds = int(
            perspective_config.get('object_manager_id_cache_refresh_seconds', 300))
    except (TypeError, ValueError):
        id_cache_refresh_seconds = 300
    deleted_objectIds = set(get_superseded_object_ids(perspective_config))
    existing_objectIds = set(get_existing_object_ids(perspective_config))
    id_cache_last_refresh = time.monotonic()
    logging.info(
        f"remove_superseded_objects: cached {len(existing_objectIds)} live object ID(s) "
        f"and {len(deleted_objectIds)} superseded object ID(s); "
        f"refresh_interval={id_cache_refresh_seconds}s")

    while True:
        # Wait for a management event signal or timeout (periodic fallback)
        if management_signal:
            management_signal.wait(timeout=60)
            management_signal.clear()

        objectIds = []
        if (id_cache_refresh_seconds > 0
                and time.monotonic() - id_cache_last_refresh >= id_cache_refresh_seconds):
            deleted_objectIds = set(get_superseded_object_ids(perspective_config))
            existing_objectIds = set(get_existing_object_ids(perspective_config))
            id_cache_last_refresh = time.monotonic()
            logging.info(
                f"remove_superseded_objects: refreshed cached ID sets: "
                f"live={len(existing_objectIds)} superseded={len(deleted_objectIds)}")
        try:
            management_event_df, management_cursor = get_management_events(
                perspective_config, rc_instance=rc, auth_instance=auth,
                since_timestamp=management_cursor)
            if not management_event_df.empty:
                supersede_map = build_supersede_map(
                    management_event_df, init_supersede_map=supersede_map)
                logging.info(
                    f"remove_superseded_objects: applied {len(management_event_df)} "
                    f"new management event(s); supersede_map={len(supersede_map)}")
        except Exception as e:
            logging.error(f"remove_superseded_objects: failed to update incremental supersede map: {e}")

        if not supersede_map:
            continue
        objectIds = list(supersede_map.keys())
        objectIds = list(set(objectIds))

        # Skip self-mapped objects (circular chains like A→B→A resolve to self;
        # these objects are still active and should NOT be removed)
        objectIds = [oid for oid in objectIds if supersede_map[oid] != oid]
        # Skip deleted objects (mapped to None) — they don't need to be moved
        # to the superseded dataset, they're just gone
        objectIds = [oid for oid in objectIds if supersede_map[oid] is not None]

        # only add new objectIds to the list
        objectIds = list(set(objectIds) - set(deleted_objectIds))
        # only include objectIds in list existing_objectIds
        objectIds = [obj_id for obj_id in objectIds if obj_id in existing_objectIds]

        # Filter out superseded objects whose superseding objects don't exist
        valid_objectIds = []
        for obj_id in objectIds:
            superseding_obj_id = supersede_map[obj_id]
            if superseding_obj_id in existing_objectIds:
                valid_objectIds.append(obj_id)
            else:
                # logging.warning(f"Skipping {obj_id}: superseding object {superseding_obj_id} does not exist")
                pass

        objectIds = valid_objectIds

        # Never supersede/remove a CONFIRMED object
        if objectIds:
            try:
                rc.token = auth.get_token()
                confirmed_query = (
                    f"SELECT {perspective_config['object_dataset']}.objectId.uuid "
                    f"FROM {perspective_config['object_dataset']} "
                    f"WHERE {perspective_config['object_dataset']}.entityStatus = 'CONFIRMED'"
                )
                confirmed_df = rc.search(confirmed_query, format='dataframe')
                if not confirmed_df.empty:
                    uuid_col = 'uuid' if 'uuid' in confirmed_df.columns else 'objectId.uuid'
                    confirmed_ids = set(confirmed_df[uuid_col].dropna().astype(str).tolist())
                    blocked = [oid for oid in objectIds if oid in confirmed_ids]
                    if blocked:
                        logging.warning(f"BLOCKED: {len(blocked)} CONFIRMED object(s) cannot be superseded: {blocked[:5]}")
                        objectIds = [oid for oid in objectIds if oid not in confirmed_ids]
            except Exception as e:
                logging.error(f"Error checking CONFIRMED status: {e}")

        if objectIds:
            logging.info(f"Processing {len(objectIds)} objects for supersession")

        # Write each object to superseded dataset one at a time, tracking successes
        successful_writes = []
        for num,object_id in enumerate(objectIds):
            # refresh the access token if necessary
            rc.token = auth.get_token()
            wc.token = auth.get_token()
        
            # download the object from the object dataset
            query  = \
                "SELECT * FROM " \
                + perspective_config['object_dataset'] + \
                " WHERE " \
                + perspective_config['object_dataset'] + "."+"objectId.uuid = " + "'" + object_id + "'"
            
            try:
                object=rc.search(query)
                # NOTE: trackId is kept nested as {"uuid": ...}; the superseded
                # object schema is identical to live_pov_objects, so no flattening.
                if len(object) == 0:
                    # Object not in object_dataset - check if it's already in superseded_object_dataset
                    if object_id not in deleted_objectIds:
                        logging.warning(f"Object {object_id} not found in object_dataset or superseded_object_dataset")
                elif len(object) == 1:
                    # Remove crucibleHeader and sub-fields before writing
                    for obj in object:
                        if isinstance(obj, dict):
                            keys_to_remove = [k for k in obj if k == 'crucibleHeader' or k.startswith('crucibleHeader.')]
                            for k in keys_to_remove:
                                del obj[k]
                    # Mark the live Object DROPPED before archiving or deleting it.
                    try:
                        wc.update_entity_record_by_name(
                            perspective_config['object_dataset'],
                            object_id,
                            {'entityStatus': 'DROPPED'},
                        )
                    except Exception as e:
                        logging.error(
                            f"Failed to flag superseded object {object_id} as DROPPED; "
                            f"leaving it in {perspective_config['object_dataset']}: {e}")
                        continue

                    # Archive the pre-supersede snapshot so RESTORE recovers
                    # the Object's prior status rather than DROPPED.
                    logging.info(f"Writing superseded object {num+1}/{len(objectIds)}: {object_id}")
                    try:
                        wc.write_record_batch_by_name(perspective_config['superseded_object_dataset'], object)
                        logging.info(f"Successfully wrote object {num+1}")
                        successful_writes.append(object_id)
                    except Exception as e:
                        logging.error(f"Failed to write superseded object {object_id}: {e}")
                        continue
                else:
                    logging.warning(f"Unexpected: found {len(object)} objects for {object_id}")
            except Exception as e:
                logging.error(f"Failed to process superseded object {object_id}: {e}")
            deleted_objectIds.add(object_id)

        logging.info(f"Finished writing {len(successful_writes)}/{len(objectIds)} superseded objects")
        if successful_writes:
            deleted_objectIds.update(successful_writes)

        # Delete only successfully written objects from object_dataset
        if not successful_writes:
            if objectIds:
                logging.warning("No objects were successfully written to superseded dataset, skipping delete operations")
        else:
            # Allow consumers time to observe the DROPPED tombstone before deletion.
            time.sleep(1)
            delete_coroutines = []
            for object_id in successful_writes:
                delete_coroutines.append(
                    asyncio.to_thread(wc.delete_entity_record_by_name, perspective_config['object_dataset'], object_id))
            try:
                wc.token = auth.get_token()
                asyncio.run(coroutine_launcher(delete_coroutines))
                logging.info(f"Successfully completed {len(delete_coroutines)} delete operations")
                existing_objectIds.difference_update(successful_writes)
            except Exception as e:
                logging.error(f"Error executing delete operations: {e}")
                logging.error(traceback.format_exc())

    # nothing returned, as this function runs indefinitely


def restore_superseded_objects(perspective_config: Dict[str, Any], management_signal: Optional[Event] = None) -> None:
    """
    Restores superseded objects from the superseded_object_dataset back to the object_dataset
    when a RESTORE action is received in the object_management_event_dataset.
    Waits for management_signal (set by correlator when new management events arrive) before each cycle.
    """
    # Configure logging for this child process (forkserver children start fresh)
    _log_level = perspective_config.get('_log_level', logging.INFO)
    log_utils.get_logger(log_type='correlator', log_level=_log_level)

    global rc, wc, auth
    auth, rc, wc = instantiate_api_controllers()
    # Full map only once; subsequent loops read only ObjectManagementEvents rows
    # newer than this cursor.  RESTORE processing has its own cursor because it
    # must inspect historical RESTORE rows once at startup, then only deltas.
    supersede_map, management_cursor = get_supersede_map(
        perspective_config, rc_instance=rc, auth_instance=auth,
        return_latest_timestamp=True)
    restore_cursor = None

    while True:
        # Wait for a management event signal or timeout (periodic fallback)
        if management_signal:
            management_signal.wait(timeout=60)
            management_signal.clear()

        try:
            # Refresh access token
            rc.token = auth.get_token()
            wc.token = auth.get_token()

            # Query for RESTORE actions from the last period
            object_management_event_dataset = perspective_config.get('object_management_event_dataset')
            
            if not object_management_event_dataset:
                continue

            try:
                new_management_events, management_cursor = get_management_events(
                    perspective_config, rc_instance=rc, auth_instance=auth,
                    since_timestamp=management_cursor)
                if not new_management_events.empty:
                    supersede_map = build_supersede_map(
                        new_management_events, init_supersede_map=supersede_map)
                    logging.info(
                        f"restore_superseded_objects: applied {len(new_management_events)} "
                        f"new management event(s); supersede_map={len(supersede_map)}")
            except Exception as e:
                logging.error(f"restore_superseded_objects: failed to update incremental supersede map: {e}")

            try:
                management_event_df, restore_cursor = get_management_events(
                    perspective_config, rc_instance=rc, auth_instance=auth,
                    actions=('RESTORE',), since_timestamp=restore_cursor)
            except Exception as e:
                logging.error(f"Error occurred while fetching RESTORE events: {e}")
                management_event_df = pd.DataFrame()

            if management_event_df.empty:
                continue

            # Get list of objectIds that need to be restored
            restore_object_ids = management_event_df['objectId'].unique().tolist()
            
            # Filter out objects that are currently in supersede_map 
            # (meaning they have a more recent SUPERSEDE event after the RESTORE)
            restore_object_ids = [oid for oid in restore_object_ids if oid not in supersede_map]
            
            if not restore_object_ids:
                logging.debug("All RESTORE events have been superseded by more recent SUPERSEDE events")
                continue
                
            logging.info(f"Processing {len(restore_object_ids)} RESTORE events for potential restoration")

            # Download all superseded objects at once
            try:
                superseded_query = f"SELECT * FROM {perspective_config['superseded_object_dataset']}"
                all_superseded_objects = rc.search(superseded_query, format='json')
                logging.info(f"Downloaded superseded objects from {perspective_config['superseded_object_dataset']}")
            except Exception as e:
                logging.warning(f"Failed to fetch superseded objects: {e}")
                all_superseded_objects = []

            if not all_superseded_objects:
                continue

            # Filter superseded objects to only those that need to be restored
            objects_to_restore = []
            if isinstance(all_superseded_objects, list):
                for obj in all_superseded_objects:
                    if isinstance(obj, dict) and 'objectId' in obj:
                        obj_id = obj['objectId'].get('uuid') if isinstance(obj['objectId'], dict) else obj['objectId']
                        if obj_id in restore_object_ids:
                            # Remove trackId field - it causes schema validation issues
                            # The object_manager will re-associate trackIds as needed
                            if "trackId" in obj:
                                del obj["trackId"]
                            objects_to_restore.append(obj)

            if not objects_to_restore:
                logging.info(f"No superseded objects found matching RESTORE events")
                continue

            # Remove crucibleHeader and sub-fields from objects before write
            for obj in objects_to_restore:
                keys_to_remove = [k for k in obj if k == 'crucibleHeader' or k.startswith('crucibleHeader.')]
                for k in keys_to_remove:
                    del obj[k]

            # Write all objects to restore one at a time
            logging.info(f"Restoring {len(objects_to_restore)} objects to {perspective_config['object_dataset']}")
           
            
            # Try writing one at a time to isolate issues
            successful_restores = []
            for i, obj in enumerate(objects_to_restore):
                obj_uuid = obj.get('objectId', {}).get('uuid', 'unknown')
                logging.info(f"Writing object {i+1}/{len(objects_to_restore)}: {obj_uuid}")
                # logging.info(f"Object {i+1} JSON: {json.dumps(obj, indent=2, default=str)}")
                try:
                    wc.token = auth.get_token()
                    wc.write_record_batch_by_name(perspective_config['object_dataset'], [obj])
                    logging.info(f"Successfully wrote object {i+1}")
                    successful_restores.append(obj_uuid)
                except DetailedHTTPError as e:
                    logging.error(f"Failed to write object {i+1} ({obj_uuid}) - API Error: {e}")
                    # Continue to next object instead of stopping
                    continue
                except Exception as e:
                    logging.error(f"Failed to write object {i+1} ({obj_uuid}): {e}")
                    continue
            
            logging.info(f"Finished processing {len(objects_to_restore)} objects, {len(successful_restores)} successful")

            # Delete only successfully restored objects from superseded_object_dataset
            if not successful_restores:
                logging.info("No objects were successfully restored, skipping delete operations")
                continue
            
            # Collect all delete operations as coroutines
            delete_coroutines = []
            for object_id in successful_restores:
                delete_coroutines.append(
                    asyncio.to_thread(wc.delete_entity_record_by_name, perspective_config['superseded_object_dataset'], object_id))

            # Execute all delete operations concurrently
            if delete_coroutines:
                try:
                    wc.token = auth.get_token()
                    asyncio.run(coroutine_launcher(delete_coroutines))
                    for object_id in successful_restores:
                        logging.info(f"Deleted restored object {object_id} from {perspective_config['superseded_object_dataset']}")
                    logging.info(f"Successfully deleted {len(delete_coroutines)} objects from {perspective_config['superseded_object_dataset']}")
                except Exception as e:
                    logging.error(f"Error executing delete operations: {e}")
                    logging.error(traceback.format_exc())

        except Exception as e:
            logging.error(f"Error in restore_superseded_objects: {e}")
            logging.error(traceback.format_exc())

    # nothing returned, as this function runs indefinitely


def update_supersede_map(management_events: List[Dict[str, Any]], supersede_map: Dict[str, str]) -> Dict[str, str]:
    """
    Updates the supersede map with the most recent supersede events.
    Uses build_supersede_map for consistency in handling SUPERSEDE, DELETE, and RESTORE actions.
    """
    # Initialize supersede_map as empty dict if None
    if supersede_map is None:
        supersede_map = {}
    
    if not management_events:
        return supersede_map
    
    logging.info("supersede management events")
    for event in management_events:
        if event.get('action') == 'SUPERSEDE':
            logging.info(f"{event['objectId']}-> {event['supersededBy']}")
        elif event.get('action') == 'DELETE':
            logging.info(f"{event['objectId']} DELETED")
        elif event.get('action') == 'RESTORE':
            logging.info(f"{event['objectId']} RESTORED")
    
    # Convert list of events to DataFrame and use build_supersede_map
    management_event_df = pd.json_normalize(management_events)
    return build_supersede_map(management_event_df, init_supersede_map=supersede_map)

def find_dynamic_datasets(config_list: List[Dict[str, Any]]) -> List[str]:
    """
    Returns a list of all datasets that are being tracked by the stream manager.
    """
  
    dynamic_datasets = []
    for config in config_list:
        if 'dynamic' in config['correlation_type'] :
            dynamic_datasets.append(config['origin_dataset'])

        # we no longer need this code:
        # if 'precreated' in config['correlation_type'] or 'named' in config['correlation_type'] or 'predefined' in config['correlation_type']:
        #     precreated_datasets.append(config['origin_dataset'])
        # it was used as follows:
                # bool = state_df['source.datasetName'].isin(precreated_datasets)   
                # state_df.loc[bool,'entityStatus'] = 'CONFIRMED' 
    return dynamic_datasets

def get_superseded_object_ids(perspective_config: Dict[str, Any]) -> List[str]:
    """
    Retrieves deleted objectIds from the superseded_object dataset.
    
    Args:
        perspective_config: Configuration dictionary containing dataset names
        
    Returns:
        List of objectId.uuid values from the superseded_object dataset
    """
    if 'superseded_object_dataset' not in perspective_config:
        logging.warning("No superseded_object_dataset found in perspective config")
        return []
    
    try:
        # Refresh access token
        rc.token = auth.get_token()
        
        # Query the superseded_object dataset for all objectId.uuid values
        query = f"SELECT {perspective_config['superseded_object_dataset']}.objectId.uuid FROM {perspective_config['superseded_object_dataset']}"
        
        logging.info(f"Retrieving superseded object IDs from superseded_object dataset: {perspective_config['superseded_object_dataset']}")
        result = rc.search(query, format='dataframe')
        
        if result.empty:
            logging.info(f"No deleted objects found in {perspective_config['superseded_object_dataset']}")
            return []
        
        # Extract objectId.uuid values
        deleted_object_ids = result['uuid'].tolist()
        logging.info(f"Found {len(deleted_object_ids)} deleted object IDs")
        
        return deleted_object_ids
        
    except Exception as e:
        logging.error(f"Error retrieving superseded object IDs: {e}")
        logging.error(traceback.format_exc())
        return []


def get_existing_object_ids(perspective_config: Dict[str, Any]) -> List[str]:
    """
    Retrieves existing objectIds from the main object dataset.
    
    Args:
        perspective_config: Configuration dictionary containing dataset names
        
    Returns:
        List of objectId.uuid values from the object dataset
    """
    if 'object_dataset' not in perspective_config:
        logging.warning("No object_dataset found in perspective config")
        return []
    
    try:
        # Refresh access token
        rc.token = auth.get_token()
        
        # Query the object dataset for all objectId.uuid values
        query = f"SELECT {perspective_config['object_dataset']}.objectId.uuid FROM {perspective_config['object_dataset']}"
        
        logging.info(f"Retrieving existing object IDs from object dataset: {perspective_config['object_dataset']}")
        result = rc.search(query, format='dataframe')
        
        if result.empty:
            logging.info(f"No existing objects found in {perspective_config['object_dataset']}")
            return []
        
        # Extract objectId.uuid values
        existing_object_ids = result['uuid'].tolist()
        logging.info(f"Found {len(existing_object_ids)} existing object IDs")
        
        return existing_object_ids
        
    except Exception as e:
        logging.error(f"Error retrieving existing object IDs: {e}")
        logging.error(traceback.format_exc())
        return []


def aggregate_SSE_messages(message_queue: Queue) -> List[Dict[str, Any]]:
    """
    Aggregates messages from the Queue into a single list.
    Drains the entire queue for better batching; redundant events are coalesced downstream.
    """
    SSE_msg = []
    while not message_queue.empty():
        event = message_queue.get(timeout=1)
        if event.data == '[]':
            pass
        else:
            json_list = json.loads(event.data)
            #append the event to the list of events
            SSE_msg=SSE_msg+json_list
    return SSE_msg

if __name__ == "__main__":
    # get command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("stream_manager_perspective", help='Name of perspective in Stream Manager Configuration')
    parser.add_argument("--use-fork", action='store_true', help='Use the fork start method for multiprocessing')
    parser.add_argument("--log", help='INFO (default), WARN, ERROR, or DEBUG')
    args = parser.parse_args()

    if args.use_fork:
        set_start_method('forkserver', force=True)
        logging.info("  Using the forkserver start method for multiprocessing  ")

    # Ensure Ctrl-C kills the entire process group (forkserver + all workers)
    def _sigterm_handler(signum: int, frame: Any) -> None:
        logging.info(f"Object Manager: Received signal {signum}, killing process group")
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    log_level = args.log
    stream_manager_perspective = args.stream_manager_perspective

    run(stream_manager_perspective, log_level=log_level)
