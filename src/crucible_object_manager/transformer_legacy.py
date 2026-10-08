'''

 A transformer is set running from the command line as follows:
    python transformer.py <Stream Manager configuration dataset name>
      e.g.,
    python transformer.py Blue_POV
    with optional arguments:
        --log=DEBUG (or INFO, WARNING, ERROR)
        INFO is the default
        --use-fork (only for Mac)

It can be run in a script with the following:
from object_manager import transformer
transformer.run(<Stream Manager configuration dataset name>,log_level='INFO')
'''

try:
    from object_utils import (
        _import_cruciblelib_modules, is_timeout_error, instantiate_api_controllers,
        coroutine_launcher, int_conversion, int_cols, _script_sources, _load_module_from_source,
        import_script_files, find_and_validate_configs,
        enu_to_ecef_rotation_matrix, ecef_to_enu_rotation_matrix,
        enu_to_ecef_vector, ecef_to_enu_vector,
        rotate_covariance_matrix_enu_to_ecef, rotate_covariance_matrix_ecef_to_enu,
        covariance_matrix_to_uncertainty_ellipse, pandas_ecef_vel_to_wgs84_by_row,
        write_batch_chunked, _fast_df_to_nested_json, drop_records_missing_object_id, normalize_uuid, run_sse_listener,
    )
except ImportError:
    from .object_utils import (
        _import_cruciblelib_modules, is_timeout_error, instantiate_api_controllers,
        coroutine_launcher, int_conversion, int_cols, _script_sources, _load_module_from_source,
        import_script_files, find_and_validate_configs,
        enu_to_ecef_rotation_matrix, ecef_to_enu_rotation_matrix,
        enu_to_ecef_vector, ecef_to_enu_vector,
        rotate_covariance_matrix_enu_to_ecef, rotate_covariance_matrix_ecef_to_enu,
        covariance_matrix_to_uncertainty_ellipse, pandas_ecef_vel_to_wgs84_by_row,
        write_batch_chunked, _fast_df_to_nested_json, drop_records_missing_object_id, normalize_uuid, run_sse_listener,
    )


import os
import sys
import re
import uuid
import json
import time
import signal
import argparse
import logging
import traceback
import copy
import queue
import gc
from datetime import timezone as tz
from datetime import datetime as dt
from datetime import timedelta
from typing import List, Any, Dict, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from pyproj import Transformer as pyproj_transformer    
import asyncio
# aiohttp and aiohttp_sse_client are lazy-imported in the functions that need them
from multiprocessing import Process, Queue,set_start_method

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress SettingWithCopyWarning
# pd.options.mode.chained_assignment = None


# function for translating WGS 84 lat and lon to ECEF x, y, & z
ecef_transformer = pyproj_transformer.from_crs("epsg:4326", "epsg:4978") 

# instantiation is done in main() after parsing args to avoid issues when
#  other scripts import from transformer.py
auth = None
rc = None
wc = None

object_primary_key = 'objectId.uuid'
custom_ID_column_name = 'custom_ID_column' # this is a temporary column used to join event and object dataframes
# Local-only column recording when THIS process last created/wrote a row.
# Used by the reconcile safety window in _prune_missing_remote_uuids().  It is essential for
# freshly-created objects: their crucibleHeader is stripped before the upsert
# (Crucible assigns it server-side), so crucibleHeader.updatedDate is NaT
# locally until the object read-propagates back.  Never written to Crucible
# (included in temp_col_list so it's dropped from all write payloads).
local_write_time_column = '_local_write_time'
temp_col_list = ['updated', 'new_object',custom_ID_column_name, local_write_time_column]
_ROW_LIMIT = 100000       # must match retrieve_objects()'s LIMIT
_OBJECT_CACHE_FANOUT_CHUNK = int(os.getenv('CRUCIBLE_OBJECT_CACHE_FANOUT_CHUNK', '5000'))
_OBJECT_CACHE_REFRESH_DEFAULT_SECONDS = int(os.getenv('CRUCIBLE_OBJECT_CACHE_REFRESH_DEFAULT_SECONDS', '1800'))
_OBJECT_CACHE_RECONCILE_DEFAULT_SECONDS = int(os.getenv('CRUCIBLE_OBJECT_CACHE_RECONCILE_DEFAULT_SECONDS', '300'))
_OBJECT_CACHE_SNAPSHOT_MIN_RETAINED_FRACTION = float(os.getenv(
    'CRUCIBLE_OBJECT_CACHE_SNAPSHOT_MIN_RETAINED_FRACTION', '0.75'))
_OBJECT_CACHE_SNAPSHOT_CONFIRMATIONS = max(2, int(os.getenv(
    'CRUCIBLE_OBJECT_CACHE_SNAPSHOT_CONFIRMATIONS', '2')))
# Queue maxsize is message count, not record count. Full-refresh messages carry
# up to _OBJECT_CACHE_FANOUT_CHUNK object records each. Keep this small: the
# cache queue is a memory-safety valve, not a durable event log. Dropped live SSE
# updates are repaired by the periodic full refresh and UUID reconcile paths.
_OBJECT_CACHE_QUEUE_MAXSIZE = int(os.getenv('CRUCIBLE_OBJECT_CACHE_QUEUE_MAXSIZE', '20'))
_SOURCE_QUEUE_MAXSIZE_DEFAULT = int(os.getenv('CRUCIBLE_SOURCE_QUEUE_MAXSIZE', '100'))
_SOURCE_QUEUE_PRESSURE_FRACTION = float(os.getenv(
    'CRUCIBLE_SOURCE_QUEUE_PRESSURE_FRACTION', '0.75'))
_SOURCE_QUEUE_LOG_INTERVAL_SECONDS = float(os.getenv(
    'CRUCIBLE_SOURCE_QUEUE_LOG_INTERVAL_SECONDS', '30'))
_source_queue_stats: Dict[str, Dict[str, float]] = {}


def _interval_seconds(value: object) -> float:
    text = str(value).strip().lower()
    units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    suffix = text[-1:]
    multiplier = units.get(suffix, 1)
    seconds = float(text[:-1] if suffix in units else text) * multiplier
    if not np.isfinite(seconds):
        raise ValueError(f"invalid interval: {value!r}")
    return seconds


_MEMORY_TRIM_DEFAULT = True
_MEMORY_TRIM_INTERVAL_DEFAULT = 60.0
_last_memory_trim_time = 0.0
_malloc_trim = None
_malloc_trim_checked = False


def _trim_process_memory_if_due(enabled: bool = _MEMORY_TRIM_DEFAULT,
                                interval: float = _MEMORY_TRIM_INTERVAL_DEFAULT,
                                now: Optional[float] = None) -> bool:
    """Periodically return freed batch-allocation arenas to glibc."""
    global _last_memory_trim_time, _malloc_trim, _malloc_trim_checked

    if not enabled:
        return False
    if now is None:
        now = time.monotonic()
    if (_last_memory_trim_time > 0
            and now - _last_memory_trim_time < interval):
        return False
    _last_memory_trim_time = now

    gc.collect()
    if not _malloc_trim_checked:
        _malloc_trim_checked = True
        try:
            import ctypes
            _malloc_trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
            if _malloc_trim is not None:
                _malloc_trim.argtypes = [ctypes.c_size_t]
                _malloc_trim.restype = ctypes.c_int
        except (ImportError, OSError, AttributeError):
            _malloc_trim = None
    if _malloc_trim is None:
        return False
    try:
        return bool(_malloc_trim(0))
    except (OSError, ValueError):
        return False

# --- [PERF] timing switch (single toggle for ALL per-stage timing logs) -------
# Flip to False (or set env CRUCIBLE_PERF_TIMING=false) to silence every [PERF]
# line and make the timers no-ops. Kept as one variable so timing is trivially
# enabled/disabled per script.
PERF_TIMING = os.getenv('CRUCIBLE_PERF_TIMING', 'true').lower() not in ('false', '0', 'no')


def _perf_disabled() -> float:
    """Zero-cost stand-in for time.perf_counter when PERF_TIMING is off."""
    return 0.0


# Per-worker caches for STATE_UPDATE de-dup: last meaningful state SIGNATURE,
# last EMIT TIME, and unchanged-repeat count per object/source. Used to suppress
# repeated low-information state updates (e.g. a position feed re-sending the
# same identity/mode/edh on every kinematic tick) while still letting a small
# heartbeat sample through. Per worker process (each handles a single feed) ->
# no cross-feed contamination.
_state_signature_cache: Dict[Any, int] = {}
_state_emit_time_cache: Dict[Any, float] = {}
_state_repeat_count_cache: Dict[Any, int] = {}
_STATE_SIG_CACHE_CAP = 500000   # hard cap so a bloated object set can't grow it unbounded
# Even when state is unchanged, emit a duplicate STATE_UPDATE at most this often
# (seconds, per object) so state never goes fully silent. Config override:
# state_heartbeat_seconds (<= 0 disables the heartbeat = suppress all duplicates).
_STATE_HEARTBEAT_SECONDS_DEFAULT = 300.0
# For live transformer calls, prefer count-based heartbeat sampling: emit one
# unchanged repeated STATE_UPDATE every N repeats per object (config override:
# state_heartbeat_every_n). This spreads heartbeat writes with event flow instead
# of pushing many objects when a wall-clock interval comes due.
_STATE_HEARTBEAT_EVERY_N_DEFAULT = 100


def _canonical_numeric_str(s: pd.Series) -> pd.Series:
    """Vectorized canonical per-value string for a numeric column in the
    STATE_UPDATE de-dup signature. Whole numbers render with no trailing '.0'
    and NA renders as '' regardless of int/float/nullable dtype, so the SAME
    logical value yields the SAME signature even if the column's dtype drifts
    between batches (e.g. Int64 123 one event, float 123.0 the next). Values
    outside int64 range fall back to their float string (no overflow). No
    per-row Python — a few C-level passes over the column only."""
    n = pd.to_numeric(s, errors='coerce').astype('float64')
    out = pd.Series('', index=s.index, dtype=object)
    finite = np.isfinite(n)
    rounded = n.round(0)
    is_whole = finite & (n == rounded) & (n.abs() < 9e18)
    if is_whole.any():
        out[is_whole] = rounded[is_whole].astype('int64').astype(str)
    other = finite & ~is_whole
    if other.any():
        out[other] = n[other].astype(str)
    return out


def _filter_repeated_state_updates(state_df: pd.DataFrame, origin_dataset: str = '',
                                   heartbeat_seconds: float = _STATE_HEARTBEAT_SECONDS_DEFAULT,
                                   heartbeat_every_n: Optional[int] = None) -> pd.DataFrame:
    """Drop STATE_UPDATE rows whose meaningful state fields are unchanged since
    the last state update emitted for that object — EXCEPT let a small heartbeat
    sample through so state never goes fully silent.

    Time-varying / source-bookkeeping columns (timestamps, source.*, latest*)
    are excluded from the signature so ordinary per-event noise does not count
    as a state change. The cache key includes source provenance, ensuring each
    newly observed object/source combination passes through once. This greatly
    reduces repetitive, low-information writes to the event API for high-rate
    position feeds while keeping a periodic heartbeat per object/source.
    When ``heartbeat_every_n`` is a positive integer, one unchanged repeat is
    emitted every N repeats per object/source. The time heartbeat remains a
    maximum-silence backstop, so low-rate sources do not disappear from source
    provenance while waiting to accumulate N repeats. A non-positive setting
    disables its corresponding heartbeat mode.
    """
    if state_df.empty or object_primary_key not in state_df.columns:
        return state_df
    try:
        heartbeat_seconds = float(heartbeat_seconds)
    except (TypeError, ValueError):
        heartbeat_seconds = _STATE_HEARTBEAT_SECONDS_DEFAULT
    if heartbeat_every_n is not None:
        try:
            heartbeat_every_n = int(heartbeat_every_n)
        except (TypeError, ValueError):
            heartbeat_every_n = _STATE_HEARTBEAT_EVERY_N_DEFAULT
    sig_cols = [c for c in state_df.columns
                if c != object_primary_key
                and c != 'eventType'
                and 'imestamp' not in c
                and not c.startswith('source')
                and not c.startswith('latestSource')
                and not c.startswith('latestUpstreamSource')
                and not c.startswith('upstreamSource')
                and c != 'collectionType']
    if not sig_cols:
        return state_df
    # Stringify first so unhashable cells (e.g. the edhControlSet list) don't
    # break hashing; this is a cheap vectorized per-row signature. Numeric
    # columns are canonicalised (123 Int64 and 123.0 float -> same string,
    # any-dtype NA -> '') so a dtype flip for the SAME value is not mistaken for
    # a state change (which would leak a redundant STATE_UPDATE through).
    sub = state_df[sig_cols]
    str_sub = sub.astype(str)
    for _c in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[_c]) and not pd.api.types.is_bool_dtype(sub[_c]):
            str_sub[_c] = _canonical_numeric_str(sub[_c])
    sig = pd.util.hash_pandas_object(str_sub, index=False)
    ids = state_df[object_primary_key].astype(str)
    provenance_cols = [c for c in ('source.datasetName', 'upstreamSource', 'collectionType')
                       if c in state_df.columns]
    cache_keys = ids.to_numpy().tolist()
    if provenance_cols:
        provenance = state_df[provenance_cols].fillna('').astype(str).to_numpy()
        cache_keys = [tuple([oid, *values])
                      for oid, values in zip(cache_keys, provenance.tolist())]
    now = time.monotonic()
    sig_np = sig.to_numpy()  # uint64
    ids_np = ids.to_numpy()
    # Look up prior signatures as uint64 WITHOUT going through a float Series:
    # ids.map() yields NaN for objects not yet cached, which coerces the whole
    # Series to float64 and corrupts 64-bit hashes above 2**53 -> a repeat object
    # would falsely look "changed" whenever the same batch also contains a new
    # object (mixed batch), leaking duplicates through.
    in_cache = np.fromiter((key in _state_signature_cache for key in cache_keys),
                           dtype=bool, count=len(ids_np))
    prev_uint = np.fromiter((_state_signature_cache.get(key, 0) for key in cache_keys),
                            dtype=np.uint64, count=len(ids_np))
    is_new = ~in_cache
    sig_changed = in_cache & (prev_uint != sig_np)
    unchanged_repeat = in_cache & (~sig_changed)
    # Heartbeat: count-based sampling spreads writes with event flow, while the
    # time-based mode is a maximum-silence backstop for low-rate sources.
    heartbeat_due = np.zeros(len(sig_np), dtype=bool)
    next_repeat_count = np.zeros(len(sig_np), dtype=np.int64)
    if heartbeat_every_n is not None and heartbeat_every_n > 0:
        prev_repeat_count = np.fromiter((_state_repeat_count_cache.get(key, 0)
                         for key in cache_keys),
                                        dtype=np.int64, count=len(ids_np))
        next_repeat_count = prev_repeat_count + 1
        heartbeat_due = unchanged_repeat & (next_repeat_count >= heartbeat_every_n)
    if heartbeat_seconds > 0:
        prev_time = np.fromiter((_state_emit_time_cache.get(key, 0.0)
                     for key in cache_keys),
                                dtype=np.float64, count=len(ids_np))
        # Per-object jitter in [-0.25, +0.25) * heartbeat_seconds so objects first
        # seen together don't all heartbeat in the SAME batch (thundering herd
        # every heartbeat_seconds -> periodic write + object_manager identity-fill
        # spikes). Deterministic per objectId within a worker; the mean interval
        # stays heartbeat_seconds so total heartbeat volume is unchanged.
        jitter = np.fromiter(((hash(o) % 1000) / 1000.0 - 0.5 for o in ids_np),
                             dtype=np.float64, count=len(ids_np)) * (0.5 * heartbeat_seconds)
        heartbeat_due |= unchanged_repeat & ((now - prev_time) >= (heartbeat_seconds + jitter))
    keep = is_new | sig_changed | heartbeat_due
    n_kept = int(keep.sum())
    n_dropped = int((~keep).sum())
    n_heartbeat = int(heartbeat_due.sum())
    if keep.any():
        # Update both caches only for the rows we keep (new / changed / heartbeat).
        for index in np.flatnonzero(keep):
            key = cache_keys[index]
            _state_signature_cache[key] = int(sig_np[index])
            _state_emit_time_cache[key] = now
            _state_repeat_count_cache[key] = 0
    if heartbeat_every_n is not None and heartbeat_every_n > 0:
        dropped_repeats = unchanged_repeat & (~keep)
        for index in np.flatnonzero(dropped_repeats):
            _state_repeat_count_cache[cache_keys[index]] = int(next_repeat_count[index])
    # Bound cache growth (the live object set can bloat when age-off is behind).
    if len(_state_signature_cache) > _STATE_SIG_CACHE_CAP:
        _state_signature_cache.clear()
        _state_emit_time_cache.clear()
        _state_repeat_count_cache.clear()
    if n_dropped or n_heartbeat:
        logging.info(f" [{origin_dataset}]: STATE_UPDATE de-dup: suppressed "
                     f"{n_dropped} repeated, kept {n_kept} "
                     f"(new/changed {n_kept - n_heartbeat}, heartbeat {n_heartbeat})")
    return state_df[keep]


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def _is_config_true(value: Any) -> bool:
    """Interpret a config value as a boolean (accepts bool or 'true'/'1'/'yes' str)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return False


def _memory_trim_settings(dataset_config: dict) -> Tuple[bool, float]:
    enabled = _is_config_true(dataset_config.get('memory_trim', _MEMORY_TRIM_DEFAULT))
    try:
        interval = float(dataset_config.get(
            'memory_trim_interval', _MEMORY_TRIM_INTERVAL_DEFAULT))
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError
    except (TypeError, ValueError):
        interval = _MEMORY_TRIM_INTERVAL_DEFAULT
        logging.warning(
            f"[{dataset_config.get('origin_dataset', '')}]: invalid memory_trim_interval; "
            f"defaulting to {interval:g}s")
    return enabled, interval


def _use_cache_sse(config: dict) -> bool:
    """Whether to run the object-dataset SSE cache for a config.

    The SSE fan-out cache (object_cache_sse_listener + object_cache_refresh_listener)
    is the SINGLE object-sync mechanism.  It is always on for active transformer
    configs; old object_sync/object_sync_interval keys are ignored and warned
    about during config validation.
    """
    return True


def event_loop_manager(config_list: List[dict]) -> None:
    """
    Set up asyncio coroutines to read in SSE messages and 
    Process() instances to process each feed.

    Args:
        config_list (List[dict]):
    """

    # Bootstrap fetch of object_df.  retrieve_objects() already retries 3x
    # internally before raising; if it still fails, keep retrying with
    # capped exponential backoff rather than killing the process.  Worker
    # processes (launch_event_processors) will then perform their own full
    # sync on first iteration, so it's also acceptable to fall back to an
    # empty template if Crucible remains unreachable for a long time.
    object_df = None
    bootstrap_attempt = 0
    while object_df is None:
        try:
            object_df = retrieve_objects(config_list[0]['object_dataset'],
                                         limit_to_ID_fields=False)
            # Determine if superseded dataset exists
            if "superseded_object_dataset" in config_list[0].keys():
                superseded_object_df = retrieve_objects(
                    config_list[0]['superseded_object_dataset'],
                    limit_to_ID_fields=False)
                if len(superseded_object_df) > 0:
                    object_df = pd.concat(
                        [object_df, superseded_object_df], ignore_index=True)
        except Exception as e:
            bootstrap_attempt += 1
            backoff = min(2 ** bootstrap_attempt, 60)
            logging.error(f"event_loop_manager: bootstrap retrieve_objects "
                          f"failed (attempt {bootstrap_attempt}): {e}")
            logging.error(f"event_loop_manager: retrying in {backoff}s")
            time.sleep(backoff)
            object_df = None

    # see if there are any objects in the object_definitions_dataset:
    if len(object_df) == 0:
        object_df = create_object_from_template()

    coroutine_list = []
    child_processes = []

    # Shared object-cache SSE fan-out — the ONLY object-sync mechanism
    # (opt out per config with `object_sync: false`).  A single SSE
    # subscription per object dataset broadcasts every object update to a
    # PER-WORKER cache queue.  A single shared Queue can't be used: shared
    # .get() gives competing consumers (one worker per message), whereas each
    # worker must see every update to keep its local object cache consistent.
    cache_sse_enabled = any(_use_cache_sse(c) for c in config_list)
    cache_queues: List[Queue] = []

    # Deduplicate configs by origin_dataset to prevent spawning multiple
    # processes (and SSE listeners) for the same feed.  Duplicate records in
    # the Crucible Stream_Manager_Configuration dataset would otherwise cause
    # independent processes to create objects for the same events in parallel,
    # resulting in duplicate objects.
    seen_origin_datasets = set()

    for dataset_config in config_list:

        if 'disabled' in dataset_config.keys() and \
            dataset_config['disabled']:
            logging.info(f"[{dataset_config['origin_dataset']}]: Transformer for {dataset_config['query']} is disabled")
            continue

        origin = dataset_config.get('origin_dataset', '')
        if origin in seen_origin_datasets:
            logging.warning(f"[{origin}]: Duplicate config for origin_dataset '{origin}' found in "
                            "Stream_Manager_Configuration - skipping to prevent duplicate object creation")
            continue
        seen_origin_datasets.add(origin)

        source_queue_maxsize = max(1, int(dataset_config.get(
            'source_queue_maxsize', _SOURCE_QUEUE_MAXSIZE_DEFAULT)))
        message_queue = Queue(maxsize=source_queue_maxsize)

        # Store script sources and log level so forkserver children can re-import scripts
        # and configure logging (children don't inherit the parent's logging config)
        dataset_config['_script_sources'] = _script_sources.copy()
        dataset_config['_log_level'] = logging.root.level

        if 'object_enrichment_class' in dataset_config:
            object_enricher = eval("object_enrichment." + dataset_config['object_enrichment_class']+ '()')
        else:
            object_enricher = None



        # create consumer processes for each event dataset
        if 'number_of_transformer_processes' in dataset_config.keys() or 'num_transformer_workers' in dataset_config.keys():
            if 'number_of_transformer_processes' in dataset_config:
                number_of_transformer_processes = int(dataset_config['number_of_transformer_processes'])
            elif 'num_consumers' in dataset_config:
                number_of_transformer_processes = int(dataset_config['num_consumers'])
            else:
                number_of_transformer_processes = int(dataset_config['num_transformer_workers'])
        else:
            number_of_transformer_processes = 1
        
        if number_of_transformer_processes > 1 and 'dynamic' in dataset_config['correlation_type'].lower():
            logging.warning(f"[{dataset_config['origin_dataset']}]: dynamic correlation may not work properly with multiple transformer processes")

        logging.info(f"[{dataset_config['origin_dataset']}]: spawning {number_of_transformer_processes} transformer worker process(es)")

        # Optional STICKY per-object routing.  When `transformer_shard_key` (a
        # dotted path into the RAW source record, e.g. 'vehicle.vehicle.label'
        # for RTD or 'icao24' for OpenSky) is set AND more than one worker is
        # used, each record is consistently hashed to a fixed worker so every
        # report for a given object is processed in arrival order by the same
        # worker.  This stops the tracker from dropping cross-worker
        # out-of-order ("too old") measurements.  The key MUST be a raw source
        # field: the destination unique ID (identity.navalPennant / the derived
        # intLabel) is produced by custom functions inside process_events and is
        # not available at dispatch time.  Absent/1-worker => unchanged
        # shared-queue (competing-consumer) behavior.
        shard_key = dataset_config.get('transformer_shard_key')
        use_sticky = bool(shard_key) and number_of_transformer_processes > 1

        worker_queues = []
        for _ in range(number_of_transformer_processes):
            # Each worker gets its OWN cache queue so the object-dataset SSE
            # listener can fan out (broadcast) every update to all workers.
            if cache_sse_enabled:
                cache_q = Queue(maxsize=_OBJECT_CACHE_QUEUE_MAXSIZE)
                cache_queues.append(cache_q)
            else:
                cache_q = None
            # Sticky routing gives each worker its own event queue; otherwise
            # all workers share the single competing-consumer message_queue.
            worker_q = Queue(maxsize=source_queue_maxsize) if use_sticky else message_queue
            worker_queues.append(worker_q)
            proc = Process(
                target=launch_event_processors,
                args=(worker_q, object_df, dataset_config, cache_q),
                daemon=True)
            proc.start()
            child_processes.append(proc)
        
        # create a SSE listener coroutine for each event dataset
        if use_sticky:
            logging.info(f"[{dataset_config['origin_dataset']}]: sticky record routing "
                         f"enabled on '{shard_key}' across {number_of_transformer_processes} worker(s); "
                         f"source_queue_maxsize={source_queue_maxsize}")
            coroutine_list.append(SSE_listener(None, dataset_config,
                                               worker_queues=worker_queues, shard_key=shard_key,
                                               source_queue_maxsize=source_queue_maxsize))
        else:
            logging.info(f"[{dataset_config['origin_dataset']}]: source_queue_maxsize="
                         f"{source_queue_maxsize} shared across "
                         f"{number_of_transformer_processes} worker(s)")
            coroutine_list.append(SSE_listener(
                message_queue, dataset_config,
                source_queue_maxsize=source_queue_maxsize))

    # Create the single object-cache SSE fan-out listener(s) if enabled — one
    # per distinct object dataset, each broadcasting to every worker's queue.
    # Alongside each SSE listener, register one object_cache_refresh_listener
    # per object dataset: it does the periodic FULL pull + UUID reconcile ONCE
    # in the parent and fans the result out to every worker (instead of each
    # worker polling Crucible, which was N× the read load).
    # Unconditional wiring diagnostic so a missing object-cache SSE is explained
    # even when the guard below is False (previously this logged NOTHING).
    logging.info(f"object_sync: wiring object cache — cache_sse_enabled={cache_sse_enabled} "
                 f"worker_cache_queues={len(cache_queues)} "
                 f"configs={len(config_list)} spawned_feeds={len(seen_origin_datasets)}")
    if not (cache_sse_enabled and cache_queues):
        logging.warning("object_sync: object cache NOT wired — "
                        f"cache_sse_enabled={cache_sse_enabled}, "
                        f"worker_cache_queues={len(cache_queues)}. "
                        "No object_cache_sse_listener/object_cache_refresh_listener will run. "
                        "(cache_sse_enabled False => every config has object_sync: false; "
                        "worker_cache_queues 0 => no worker processes were spawned, e.g. all "
                        "configs disabled or duplicate-skipped.)")
    if cache_sse_enabled and cache_queues:
        object_datasets: List[str] = []
        refresh_configs: List[dict] = []
        seen_object_ds: set = set()
        for config in config_list:
            if not _use_cache_sse(config):
                continue
            for _key in ('object_dataset', 'superseded_object_dataset'):
                ds = config.get(_key)
                if ds and ds not in object_datasets:
                    object_datasets.append(ds)
            _obj_ds = config.get('object_dataset')
            if _obj_ds and _obj_ds not in seen_object_ds:
                seen_object_ds.add(_obj_ds)
                refresh_configs.append(config)
        for ds in object_datasets:
            coroutine_list.append(object_cache_sse_listener(cache_queues, ds))
        for cfg in refresh_configs:
            coroutine_list.append(object_cache_refresh_listener(
                cache_queues, cfg, initial_remote_count=len(object_df)))
        logging.info(f"object_sync: object cache enabled: {len(cache_queues)} worker cache "
                     f"queue(s); object-dataset SSE fan-out for {object_datasets}; "
                     f"full-pull+reconcile fan-out for {[c['object_dataset'] for c in refresh_configs]}")

    # create a deduplication process if needed
    for dataset_config in config_list:
        if _is_config_true(dataset_config.get('deduplication')) and not _is_config_true(os.getenv('CRUCIBLE_DISABLE_DEDUPE', 'false')):
            dedupe_proc = Process(target=dedupe_objects, args=(dataset_config,))
            dedupe_proc.start()
            child_processes.append(dedupe_proc)
            break #only one deduplication process is needed
        else:
            continue

    # note: nothing is returned from these coroutines
    try:
        asyncio.run(coroutine_launcher(coroutine_list)) # this is blocking
    finally:
        for proc in child_processes:
            if proc.is_alive():
                logging.info(f"Transformer: Terminating child process {proc.pid}")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.join(timeout=3)


async def connect_with_timeout(url: str, session: Any, headers: dict, timeout: float) -> Any:
    try:
        from aiohttp_sse_client import client as sse_client
        return await asyncio.wait_for( sse_client.EventSource(url, session=session, headers=headers, timeout=None).__aenter__(), timeout)
    except asyncio.TimeoutError:
        logging.error(f"Timeout error occurred: Initial connection to {url} could not be established within {timeout} seconds.")
        return None

def launch_event_processors(message_queue: Queue,
                            object_df: pd.DataFrame,
                            dataset_config: dict,
                            cache_queue: Optional[Queue] = None,
                            ) -> None:
    import sys, os
    print(f"[CHILD PID={os.getpid()}] launch_event_processors ENTERED for {dataset_config.get('origin_dataset','?')}", flush=True, file=sys.stderr)

    # Configure logging in the child (forkserver children don't inherit parent's config)
    if '_log_level' in dataset_config:
        _, _, _, _log_utils, _, _ = _import_cruciblelib_modules()
        _log_utils.get_logger(log_type='transformer', log_level=dataset_config['_log_level'])

    # Reconstruct object_enricher in the child process (avoids pickling module objects)
    # Also re-import custom_functions and unit_conversions since forkserver children
    # don't inherit the parent's dynamically-imported modules.
    object_enricher = None
    if '_script_sources' in dataset_config:
        _sources = dataset_config['_script_sources']
        if 'custom_functions' in _sources:
            mod = _load_module_from_source('custom_functions', _sources['custom_functions'])
            globals()['custom_functions'] = mod
        if 'unit_conversions' in _sources:
            mod = _load_module_from_source('unit_conversions', _sources['unit_conversions'])
            globals()['unit_conversions'] = mod
        if 'object_enrichment' in _sources and 'object_enrichment_class' in dataset_config:
            mod = _load_module_from_source('object_enrichment', _sources['object_enrichment'])
            globals()['object_enrichment'] = mod
            object_enricher = eval("object_enrichment." + dataset_config['object_enrichment_class'] + '()')
    
    # Get lists of identifier column names for the origin and destination datasets
    # (allow for multi-column identifiers )

    if not isinstance(dataset_config['origin_unique_ID_column'], list):
        dataset_config['origin_unique_ID_column'] = [dataset_config['origin_unique_ID_column']]
    if not isinstance(dataset_config['destination_unique_ID_column'], list):
        dataset_config['destination_unique_ID_column'] = [dataset_config['destination_unique_ID_column']]

    # add source UUID column to the origin->destination mapping
    dest_cols = [item['destination_column'] for item in dataset_config['origin_to_destination_mapping']]
    if 'source.uuid' not in dest_cols:
        dataset_config['origin_to_destination_mapping'].append(
                {
                    "destination_column": "source.uuid",
                    "origin_column": "crucibleHeader.uuid"
                }  )

    # If needed, add ID lookup columns to the origin->destination mapping
    for origin, destination in zip(dataset_config['origin_unique_ID_column'], dataset_config['destination_unique_ID_column']):
        dataset_config['origin_to_destination_mapping'].append(
            {"origin_column": origin, "destination_column": destination})


    last_reconcile_time = None # unused in the SSE-cache path; kept for the reconcile fan-out cadence bookkeeping
    print(f"[CHILD PID={os.getpid()}] about to call instantiate_api_controllers()", flush=True, file=sys.stderr)
    global rc, wc, auth
    try:
        auth, rc, wc = instantiate_api_controllers()
        print(f"[CHILD PID={os.getpid()}] instantiate_api_controllers() SUCCEEDED", flush=True, file=sys.stderr)
    except BaseException as e:
        print(f"[CHILD PID={os.getpid()}] instantiate_api_controllers() FAILED: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
        raise

    # Push-based object cache: the object-dataset SSE fan-out is the ONLY
    # object-sync mechanism (opt out entirely with object_sync: false).  Object
    # updates are merged from the shared SSE fan-out queue; the worker never
    # polls Crucible.
    cache_sse_enabled = _use_cache_sse(dataset_config) and cache_queue is not None
    if cache_sse_enabled:
        logging.info(f"[{dataset_config['origin_dataset']}]: object_sync: object cache ENABLED — "
                     f"merging object updates from the shared SSE fan-out (no Crucible polling)")
    memory_trim_enabled, memory_trim_interval = _memory_trim_settings(dataset_config)
    logging.info(
        f"[{dataset_config['origin_dataset']}]: memory trim "
        f"enabled={memory_trim_enabled} interval={memory_trim_interval:g}s")

    while True:

        # refresh the access token if necessary
        rc.token = auth.get_token()
        wc.token = auth.get_token()

        # Drain object-cache updates before waiting for the next source event.
        # Quiet feeds must still consume DROPPED tombstones and periodic repair
        # messages instead of leaving their bounded cache queues permanently full.
        if cache_sse_enabled:
            try:
                object_df = _drain_cache_queue_into_object_df(
                    cache_queue, object_df, dataset_config)
            except Exception as e:
                logging.error(f"[{dataset_config['origin_dataset']}]: "
                              f"object_sync cache-queue merge failed, keeping local object_df: {e}")
                logging.error(traceback.format_exc())

        try:
            # see if there are any events in the queue
            event = message_queue.get(timeout=1)
        except Exception:
            # if not, go back to the top of the loop
            _trim_process_memory_if_due(memory_trim_enabled, memory_trim_interval)
            time.sleep(0.5)
            continue

        object_df = asyncio.run(process_events(event, object_df, dataset_config,object_enricher))
        _trim_process_memory_if_due(memory_trim_enabled, memory_trim_interval)
        # this needs to be asyncio.run() because process_events() is an asyncio coroutine
        # in order to accommodate the to_thread() call inside of it



    #nothing is returned from this function, loop continues


def _prune_missing_remote_uuids(object_df: pd.DataFrame,
                                remote_uuids: Optional[set],
                                reconcile_safety_seconds: Optional[int],
                                dataset_config: dict) -> pd.DataFrame:
    """
    Prune local objects whose UUID is not in the authoritative remote UUID set,
    protecting (a) rows created locally but not yet upserted (new_object) and
    (b) rows written within the safety window (Crucible read-after-write lag).
    Used by the SSE-cache reconcile fan-out (object_cache_refresh_listener).

    Returns object_df unchanged when remote_uuids is None (transient read
    failure), the key column is absent, or the frame is empty.
    """
    if (remote_uuids is None or object_primary_key not in object_df.columns
            or len(object_df) == 0):
        return object_df

    safety = int(reconcile_safety_seconds or 10)
    safety_cutoff = None
    if safety > 0:
        safety_cutoff = dt.now(tz=tz.utc) - timedelta(seconds=safety)

    local_uuids = object_df[object_primary_key].astype(str)
    missing_remote = ~local_uuids.isin(remote_uuids)
    # Always protect rows we created locally that haven't been upserted yet.
    if 'new_object' in object_df.columns:
        missing_remote = missing_remote & ~object_df['new_object'].fillna(False).astype(bool)

    # Protect rows written very recently (within the safety window) to avoid
    # dropping objects we just upserted that haven't become visible to reads
    # yet.  A row counts as recently written if EITHER its platform
    # crucibleHeader.updatedDate OR the local write timestamp is within the
    # window.  NaT in both columns is treated as "old" and eligible for pruning.
    num_protected_recent = 0
    if safety_cutoff is not None:
        recently_written = pd.Series(False, index=object_df.index)
        if 'crucibleHeader.updatedDate' in object_df.columns:
            header_updated = pd.to_datetime(
                object_df['crucibleHeader.updatedDate'], utc=True, errors='coerce')
            recently_written = recently_written | (header_updated >= safety_cutoff).fillna(False)
        if local_write_time_column in object_df.columns:
            local_written = pd.to_datetime(
                object_df[local_write_time_column], utc=True, errors='coerce')
            recently_written = recently_written | (local_written >= safety_cutoff).fillna(False)
        num_protected_recent = int((missing_remote & recently_written).sum())
        missing_remote = missing_remote & ~recently_written

    num_to_prune = int(missing_remote.sum())
    raw_missing = int((~local_uuids.isin(remote_uuids)).sum())
    logging.info(
        f"[{dataset_config.get('origin_dataset','')}]: "
        f"object_sync UUID reconcile: objects_in_cache={len(object_df)} "
        f"objects_in_crucible={len(remote_uuids)} "
        f"absent_from_crucible={raw_missing} "
        f"kept_because_recently_written={num_protected_recent} "
        f"objects_removed={num_to_prune} safety_window_seconds={safety}")
    if num_to_prune > 0:
        object_df = object_df[~missing_remote].reset_index(drop=True)
    return object_df


class _RoutedMessage:
    """Lightweight stand-in for an aiohttp_sse_client MessageEvent carrying a
    sharded subset of a source message's records (sticky per-object routing).
    process_events() only reads ``.type`` and ``.data``."""
    __slots__ = ('type', 'data')

    def __init__(self, data: str):
        self.type = 'message'
        self.data = data


def _source_event_record_count(event: Any) -> int:
    data = getattr(event, 'data', None)
    if not data:
        return 0
    try:
        records = json.loads(data) if isinstance(data, str) else data
    except (ValueError, TypeError):
        return 0
    if isinstance(records, list):
        return len(records)
    return 1 if isinstance(records, dict) else 0


def _enqueue_source_message(target_queue: Queue, message: Any, origin: str,
                            maxsize: int, *, worker_index: Optional[int] = None,
                            record_count: Optional[int] = None) -> bool:
    """Enqueue one raw-source message without allowing backlog memory to grow
    without bound. When full, evict the oldest queued message to favor freshness;
    logs aggregate evicted/rejected messages and records plus queue pressure."""
    queue_label = origin if worker_index is None else f'{origin}:worker-{worker_index}'
    now = time.monotonic()
    stats = _source_queue_stats.setdefault(queue_label, {
        'last_pressure_log': 0.0,
        'last_drop_log': 0.0,
        'evicted_messages': 0.0,
        'evicted_records': 0.0,
        'rejected_messages': 0.0,
        'rejected_records': 0.0,
    })
    try:
        target_queue.put_nowait(message)
    except queue.Full:
        oldest = None
        try:
            oldest = target_queue.get(timeout=0.01)
        except queue.Empty:
            pass
        if oldest is not None:
            stats['evicted_messages'] += 1
            stats['evicted_records'] += _source_event_record_count(oldest)
        enqueued = True
        try:
            target_queue.put_nowait(message)
        except queue.Full:
            enqueued = False
            if record_count is None:
                record_count = _source_event_record_count(message)
            stats['rejected_messages'] += 1
            stats['rejected_records'] += int(record_count or 0)
        if now - stats['last_drop_log'] >= _SOURCE_QUEUE_LOG_INTERVAL_SECONDS:
            depth = _safe_qsize(target_queue)
            logging.error(
                f"[{queue_label}]: raw source queue FULL; "
                f"evicted_oldest_messages={int(stats['evicted_messages'])} "
                f"records={int(stats['evicted_records'])}; "
                f"rejected_newest_messages={int(stats['rejected_messages'])} "
                f"records={int(stats['rejected_records'])} since last log; "
                f"queue_depth={depth} maxsize={maxsize}")
            stats['evicted_messages'] = 0
            stats['evicted_records'] = 0
            stats['rejected_messages'] = 0
            stats['rejected_records'] = 0
            stats['last_drop_log'] = now
        return enqueued

    depth = _safe_qsize(target_queue)
    pressure_threshold = max(1, int(maxsize * _SOURCE_QUEUE_PRESSURE_FRACTION))
    if (depth >= pressure_threshold
            and now - stats['last_pressure_log'] >= _SOURCE_QUEUE_LOG_INTERVAL_SECONDS):
        logging.warning(
            f"[{queue_label}]: raw source queue backpressure: "
            f"queue_depth={depth} maxsize={maxsize} "
            f"utilization={depth / maxsize:.0%}")
        stats['last_pressure_log'] = now
    return True


def _route_event_to_shards(event: Any, worker_queues: List[Queue],
                           key_parts: List[str], origin: str,
                           source_queue_maxsize: int) -> None:
    """Shard one SSE message's records across per-worker queues by a stable hash
    of the raw source field addressed by ``key_parts`` (a dotted path split), so
    every report for a given object is routed to the same worker in arrival
    order.  Runs in the single parent event loop, so Python's per-process string
    hash stays consistent for the whole run.  keep-alive / empty messages are
    dropped (they are no-ops downstream)."""
    if getattr(event, 'type', None) == 'keep-alive':
        return
    data = getattr(event, 'data', None)
    if not data or len(data) <= 5:
        return
    try:
        records = json.loads(data) if isinstance(data, str) else data
    except (ValueError, TypeError):
        return
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        return
    n = len(worker_queues)
    shards: List[list] = [[] for _ in range(n)]
    for rec in records:
        k = rec
        for p in key_parts:
            k = k.get(p) if isinstance(k, dict) else None
        shards[hash(str(k)) % n].append(rec)
    for i, shard in enumerate(shards):
        if shard:
            _enqueue_source_message(
                worker_queues[i], _RoutedMessage(json.dumps(shard)), origin,
                source_queue_maxsize, worker_index=i, record_count=len(shard))


async def SSE_listener(message_queue: Optional[Queue], dataset_config: dict,
                       worker_queues: Optional[List[Queue]] = None,
                       shard_key: Optional[str] = None,
                       source_queue_maxsize: int = _SOURCE_QUEUE_MAXSIZE_DEFAULT) -> None:

    """
    Handles SSE connections and sends incoming
    messages to a multiprocessing Queue. These
    messages are then processed by the
    process_events() function, which are run in separate
    multiprocessing Processes.

    Args:
        message_queue:
        dataset_config (dict):

    """
    # Precompute the sticky-routing key path (None => plain shared-queue mode).
    key_parts = shard_key.split('.') if shard_key else None
    origin = dataset_config['origin_dataset']

    async def _on_event(event) -> None:
        if worker_queues is not None:
            assert key_parts is not None
            _route_event_to_shards(
                event, worker_queues, key_parts, origin, source_queue_maxsize)
        else:
            assert message_queue is not None
            _enqueue_source_message(
                message_queue, event, origin, source_queue_maxsize)

    await run_sse_listener(dataset_config['query'], auth, _on_event, label=origin)


async def object_cache_sse_listener(cache_queues: List[Queue], object_dataset: str) -> None:
    """
    Single SSE subscription to an object dataset that FANS OUT every message to
    every worker's cache queue (one queue per process).

    A single multiprocessing.Queue can't be shared for this: shared .get() gives
    competing consumers (each message goes to exactly one worker), whereas every
    worker must see every object update to keep its local cache consistent.  So
    the listener writes each event to ALL per-worker queues.

    Deletes are NOT delivered over the object-dataset SSE; object_cache_refresh_listener()
    fans out a periodic full-pull snapshot and an authoritative remote UUID set
    so each worker can prune deletions and repair any gap caused by an SSE drop.
    """
    query = 'select * from ' + object_dataset
    label = f"object_sync:object_cache_sse:{object_dataset}"
    logging.info(f"[{label}]: fan-out to {len(cache_queues)} worker queue(s)")

    async def _on_event(event) -> None:
        # Broadcast to every worker queue.  Drop on a full queue (slow worker);
        # the periodic full refresh + UUID reconcile repairs any gap. We do not
        # block the live object SSE reader here because an overloaded worker
        # should not allow unbounded cache-message memory growth in the parent.
        # Age-off marks objects DROPPED immediately before hard deletion. Never
        # discard that tombstone under queue pressure: it is the fast path that
        # prevents workers from treating deleted objects as still present.
        is_delete_tombstone = _event_contains_dropped_object(event)
        dropped = await _broadcast_to_cache_queues(
            cache_queues, event, block=is_delete_tombstone)
        if is_delete_tombstone:
            logging.info(f"[{label}]: reliably fanned out DROPPED object tombstone "
                         f"to {len(cache_queues)} worker queue(s)")
        if dropped:
            logging.warning(f"[{label}]: dropped live object-cache SSE update for {dropped} full worker queue(s); reconcile will repair")

    await run_sse_listener(query, auth, _on_event, label=label)


def _event_contains_dropped_object(event: Any) -> bool:
    """Whether an object-dataset SSE message contains an age-off tombstone."""
    data = getattr(event, 'data', None)
    if not data:
        return False
    try:
        records = json.loads(data) if isinstance(data, str) else data
    except (ValueError, TypeError):
        return False
    if isinstance(records, dict):
        records = [records]
    return isinstance(records, list) and any(
        isinstance(record, dict)
        and str(record.get('entityStatus', '')).upper() == 'DROPPED'
        for record in records)


def _safe_qsize(q: Queue) -> int:
    """Best-effort queue size for diagnostics; multiprocessing qsize may be unsupported."""
    try:
        return int(q.qsize())
    except (NotImplementedError, AttributeError, OSError):
        return -1


async def _broadcast_to_cache_queues(cache_queues: List[Queue], message: Any,
                                     *, block: bool = False) -> int:
    """Put a message on every worker cache queue.

    Live object SSE uses non-blocking/drop-on-full (full refresh + reconcile
    repairs gaps). Periodic full-refresh messages use blocking put to avoid
    accumulating newer full snapshots behind an already-backed-up worker queue.
    Returns the number of queues that dropped the message in non-blocking mode.
    """
    dropped = 0
    for q in cache_queues:
        try:
            if block:
                await asyncio.to_thread(q.put, message)
            else:
                q.put_nowait(message)
        except queue.Full:
            dropped += 1
    return dropped


def _validate_remote_uuid_snapshot(
        remote_uuids: Optional[set], previous_count: Optional[int],
        suspicious_observations: int) -> Tuple[Optional[set], Optional[int], int]:
    """Require confirmation before accepting an empty or sharply smaller snapshot.

    Object reads can occasionally return a successful but incomplete result
    while the read tier is overloaded. Treating that result as authoritative
    immediately can evict every worker's cache and create a delete/recreate
    storm. Bulk deletes already have a reliable DROPPED-tombstone fast path, so
    delaying suspicious reconcile pruning until a second observation is safe.
    """
    if remote_uuids is None:
        return None, previous_count, 0

    current_count = len(remote_uuids)
    sharply_smaller = (
        previous_count is not None
        and previous_count > 0
        and current_count < previous_count * _OBJECT_CACHE_SNAPSHOT_MIN_RETAINED_FRACTION)
    suspicious = (current_count == 0 and previous_count != 0) or sharply_smaller
    if suspicious:
        suspicious_observations += 1
        if suspicious_observations < _OBJECT_CACHE_SNAPSHOT_CONFIRMATIONS:
            logging.warning(
                "object_sync: ignoring unconfirmed suspicious UUID snapshot "
                f"count={current_count} previous_count={previous_count} "
                f"confirmation={suspicious_observations}/"
                f"{_OBJECT_CACHE_SNAPSHOT_CONFIRMATIONS}")
            return None, previous_count, suspicious_observations

    return remote_uuids, current_count, 0


async def object_cache_refresh_listener(cache_queues: List[Queue],
                                        dataset_config: dict,
                                        initial_remote_count: Optional[int] = None) -> None:
    """
        Shared periodic FULL-PULL and UUID-RECONCILE fan-out for one object dataset.

    Runs once in the parent event loop (like object_cache_sse_listener) and
        replaces per-worker Crucible polling.  The object-dataset SSE is the primary
        cache-sync path; these periodic reads are repair paths:
            * refresh_interval: expensive SELECT * full-object repair for missed SSE
                creates/updates.
            * reconcile_interval: cheap UUID-only existence check for hard deletes
                that are not delivered over the object SSE.

        Two fan-out message types may be produced:
      * {'_cache_msg': 'records', 'records': [...]} — a full snapshot of the
        remote objects.  Each worker merges it with only_add_new_rows=True, so a
        stale remote row can never clobber a fresher local object; this re-adds
        objects missed by a dropped SSE create.
      * {'_cache_msg': 'reconcile', 'remote_uuids': [...], 'safety_seconds': N}
        — the authoritative remote UUID set.  Each worker prunes local objects
        whose UUID is absent (respecting the same safety window as the reconcile
        prune) to catch hard deletes not delivered over the object SSE.

        When a full pull and reconcile are due in the same cycle, keep the
        reconcile message even when the full pull is complete: workers must apply
        pruning AFTER merging records. When the full pull hits the row limit, a
        separate ID-only UUID query is still required so we do not prune objects
        that merely fell beyond SELECT * LIMIT.

    Combine optimisation: when the full pull is NOT truncated by the query row
    limit, its own objectId.uuid set IS the authoritative remote set, so the
    prune list is derived from the full pull and NO second UUID query is made.
    Only when the full pull hits the row limit (its UUID set may be incomplete)
    does it fall back to the dedicated ID-only UUID query so we never prune real
    objects that merely fell beyond the full-pull row limit.
    """
    object_dataset = dataset_config['object_dataset']
    superseded = dataset_config.get('superseded_object_dataset')
    # Full pull is the expensive repair path for missed object-cache SSE updates.
    # UUID-only reconcile is cheaper and catches hard deletes, so it can run more
    # often without pulling every object record.
    refresh_interval = _interval_seconds(dataset_config.get(
        'refresh_interval', _OBJECT_CACHE_REFRESH_DEFAULT_SECONDS))
    reconcile_interval = _interval_seconds(dataset_config.get(
        'reconcile_interval', _OBJECT_CACHE_RECONCILE_DEFAULT_SECONDS))
    safety = 10

    logging.info(f"[object_sync:object_cache_refresh:{object_dataset}]: full object pull "
                 f"every {refresh_interval}s; UUID reconcile every {reconcile_interval}s "
                 f"to {len(cache_queues)} worker queue(s)")

    refresh_generation = 0
    next_refresh_time = 0.0
    next_reconcile_time = 0.0
    last_authoritative_uuid_count = initial_remote_count
    suspicious_snapshot_observations = 0

    async def _fetch_remote_uuids() -> Optional[set]:
        remote = await asyncio.to_thread(_retrieve_remote_object_uuids, object_dataset)
        if remote is not None and superseded:
            sup_uuids = await asyncio.to_thread(_retrieve_remote_object_uuids, superseded)
            if sup_uuids is not None:
                remote = remote | sup_uuids
            else:
                # Couldn't reliably read superseded dataset; skip the prune this
                # cycle to avoid false-positive deletions.
                remote = None
        return remote

    while True:
        try:
            now = time.monotonic()
            refresh_due = refresh_interval > 0 and now >= next_refresh_time
            reconcile_due = reconcile_interval > 0 and now >= next_reconcile_time
            if not refresh_due and not reconcile_due:
                future_due_times = [
                    t for t in (next_refresh_time, next_reconcile_time)
                    if t > now]
                if not future_due_times:
                    await asyncio.sleep(60.0)
                    continue
                next_due = min(future_due_times)
                await asyncio.sleep(max(1.0, next_due - now))
                continue

            rc.token = auth.get_token()
            qsize_before = [_safe_qsize(q) for q in cache_queues]
            refresh_generation += 1
            full_records: List[dict] = []
            truncated = False
            n_records = 0
            remote_uuids: Optional[set] = None

            # ---- full field pull (fan out as upserts) ----
            if refresh_due:
                full_records = await asyncio.to_thread(
                    retrieve_object_records, object_dataset)
                if superseded:
                    sup_records = await asyncio.to_thread(
                        retrieve_object_records, superseded)
                    if sup_records:
                        full_records.extend(sup_records)

                truncated = len(full_records) >= _ROW_LIMIT
                n_records = len(full_records)
                if n_records > 0:
                    for i in range(0, n_records, _OBJECT_CACHE_FANOUT_CHUNK):
                        # rc.search() already returns list[dict] natively. Slice
                        # that list directly instead of materializing a DataFrame
                        # only to call to_dict('records') again in the parent.
                        records = full_records[i:i + _OBJECT_CACHE_FANOUT_CHUNK]
                        await _broadcast_to_cache_queues(
                            cache_queues,
                            {'_cache_msg': 'records',
                             '_cache_generation': refresh_generation,
                             'records': records},
                            block=True)
                next_refresh_time = now + refresh_interval

            # ---- reconcile UUID set (fan out as prune) ----
            # Combine: when full refresh is also due, derive the prune set from
            # the full pull unless it was truncated.  Reconcile-only cycles do
            # the cheap ID-only query.
            if refresh_due and full_records and not truncated:
                remote_uuids = set()
                for rec in full_records:
                    uuid_val = rec.get(object_primary_key)
                    obj_id = rec.get('objectId')
                    if uuid_val is None and isinstance(obj_id, dict):
                        uuid_val = obj_id.get('uuid')
                    if uuid_val:
                        remote_uuids.add(str(uuid_val))
            else:
                # An empty full-object pull is not sufficient evidence that a
                # previously populated dataset is empty. Corroborate it with
                # the independent leaf-UUID query, then apply the consecutive
                # suspicious-snapshot guard below.
                remote_uuids = await _fetch_remote_uuids()

            remote_uuids, last_authoritative_uuid_count, suspicious_snapshot_observations = (
                _validate_remote_uuid_snapshot(
                    remote_uuids, last_authoritative_uuid_count,
                    suspicious_snapshot_observations))
            n_uuids = -1
            if remote_uuids is not None:
                n_uuids = len(remote_uuids)
                await _broadcast_to_cache_queues(
                    cache_queues,
                    {'_cache_msg': 'reconcile',
                     '_cache_generation': refresh_generation,
                     'remote_uuids': list(remote_uuids),
                     'safety_seconds': safety},
                    block=True)
            if reconcile_due or refresh_due:
                next_reconcile_time = now + reconcile_interval

            qsize_after = [_safe_qsize(q) for q in cache_queues]
            q_before_max = max(qsize_before) if qsize_before else -1
            q_after_max = max(qsize_after) if qsize_after else -1

            logging.info(f"[object_sync:object_cache_refresh:{object_dataset}]: fanned out "
                         f"generation={refresh_generation} "
                         f"refresh_due={refresh_due} reconcile_due={reconcile_due} "
                         f"full_object_pull_records={n_records} (hit_row_limit={truncated}) "
                         f"uuid_reconcile_count={n_uuids} "
                         f"queue_backlog_before_max={q_before_max} "
                         f"queue_backlog_after_max={q_after_max}")
        except Exception as e:
            logging.error(f"[object_sync:object_cache_refresh:{object_dataset}]: refresh failed: {e}")
            logging.error(traceback.format_exc())

        await asyncio.sleep(1.0)


def _drain_cache_queue_into_object_df(cache_queue: Optional[Queue],
                                      object_df: pd.DataFrame,
                                      dataset_config: dict) -> pd.DataFrame:
    """
    Drain all pending object updates from the shared SSE cache queue and merge
    them into the local object_df (push-based replacement for the per-worker
    poll).  Messages are coalesced (last-write-wins per uuid) so a
    slow drain / bursty stream stays cheap.

    dedupe_objects flags duplicates with entityStatus='DROPPED' BEFORE deleting
    them; that update rides the object SSE, so any uuid arriving as DROPPED is
    pruned from the local cache in near-real-time (external hard-deletes without
    a DROPPED flag are still caught by the periodic UUID reconcile).

    The queue also carries two control messages fanned out by
    object_cache_refresh_listener():
      * {'_cache_msg': 'records', ...}   — a periodic full-pull snapshot,
        merged exactly like SSE upserts (only_add_new_rows).
      * {'_cache_msg': 'reconcile', ...} — the authoritative remote UUID set;
        applied LAST (after all upserts) to prune deleted objects.
    """
    payloads = []
    # Full-refresh records are grouped by generation so a slow worker that
    # drains multiple queued snapshots only merges the newest one. This bounds
    # catch-up work and avoids spending memory/CPU on stale full snapshots.
    refresh_payloads_by_generation: Dict[int, List[dict]] = {}
    reconcile_msg = None   # keep only the newest reconcile (coalesce)
    reconcile_generation = -1
    drained = 0
    backlog_start = -1
    pull_rows = 0          # rows from the periodic FULL OBJECT PULL snapshot
    stale_pull_rows = 0
    sse_rows = 0           # rows from SSE of latest object changes
    if cache_queue is None:
        return object_df
    backlog_start = _safe_qsize(cache_queue)
    while True:
        try:
            event = cache_queue.get_nowait()
        except queue.Empty:
            break
        drained += 1
        # Control messages from the shared full-pull/reconcile fan-out.
        if isinstance(event, dict) and '_cache_msg' in event:
            kind = event['_cache_msg']
            if kind == 'records':
                rows = event.get('records', [])
                generation = event.get('_cache_generation')
                if generation is None:
                    payloads.extend(rows)
                    pull_rows += len(rows)
                else:
                    generation = int(generation)
                    refresh_payloads_by_generation.setdefault(generation, []).extend(rows)
            elif kind == 'reconcile':
                generation = int(event.get('_cache_generation', reconcile_generation + 1))
                if generation >= reconcile_generation:
                    reconcile_msg = event
                    reconcile_generation = generation
            continue
        # SSE MessageEvent objects.
        if getattr(event, 'type', None) == 'keep-alive':
            continue
        data = getattr(event, 'data', None)
        if not data or len(data) <= 5:
            continue
        try:
            records = json.loads(data) if isinstance(data, str) else data
        except (ValueError, TypeError):
            continue
        if isinstance(records, list):
            payloads.extend(records)
            sse_rows += len(records)
        elif isinstance(records, dict):
            payloads.append(records)
            sse_rows += 1

    if refresh_payloads_by_generation:
        latest_generation = max(refresh_payloads_by_generation)
        latest_rows = refresh_payloads_by_generation[latest_generation]
        payloads.extend(latest_rows)
        pull_rows += len(latest_rows)
        stale_pull_rows = sum(
            len(rows) for generation, rows in refresh_payloads_by_generation.items()
            if generation != latest_generation)
        if reconcile_msg is not None and reconcile_generation < latest_generation:
            reconcile_msg = None

    if not payloads and reconcile_msg is None:
        return object_df

    # ---- merge upserts (SSE events + full-pull snapshot) ----
    if payloads:
        fresh = pd.json_normalize(payloads)
        if len(fresh) > 0 and object_primary_key in fresh.columns:
            # Soft-delete tombstones flagged by the dedup process.
            dropped_uuids = set()
            if 'entityStatus' in fresh.columns:
                dropped_uuids = set(
                    fresh.loc[fresh['entityStatus'].astype(str).str.upper() == 'DROPPED',
                              object_primary_key].dropna().astype(str))

            object_df = update_dataframe(
                object_df, fresh,
                unique_id_column_name=object_primary_key,
                add_columns_and_rows_from_new_df_to_old_df=True,
                only_add_new_rows=True,
            )

            if dropped_uuids and object_primary_key in object_df.columns:
                object_df = object_df[
                    ~object_df[object_primary_key].astype(str).isin(dropped_uuids)
                ].reset_index(drop=True)

            logging.info(f"[{dataset_config.get('origin_dataset','')}]: object_sync cache merge: "
                         f"queue_messages_read={drained} "
                         f"object_records_merged={len(fresh)} "
                         f"(from_full_object_pull={pull_rows} stale_full_object_pull_dropped={stale_pull_rows} "
                         f"from_sse_live_changes={sse_rows}) "
                         f"cache_queue_backlog_start={backlog_start} "
                         f"objects_deleted_remotely_removed_locally={len(dropped_uuids)} "
                         f"objects_in_cache_now={len(object_df)}")

    # ---- apply reconcile prune LAST so we never drop a row just merged in ----
    if reconcile_msg is not None:
        object_df = _prune_missing_remote_uuids(
            object_df,
            set(reconcile_msg.get('remote_uuids', [])),
            reconcile_msg.get('safety_seconds'),
            dataset_config)

    return object_df


def dedupe_objects(dataset_config: dict) -> None:
    """
    Deduplicates objects in the object dataset
    """
    if _is_config_true(os.getenv('CRUCIBLE_DISABLE_DEDUPE', 'false')):
        logging.info(f"[{dataset_config.get('origin_dataset','')}]: dedupe_objects deactivated via CRUCIBLE_DISABLE_DEDUPE environment variable")
        return

    if not _is_config_true(dataset_config.get('deduplication')):
        logging.info(f"[{dataset_config.get('origin_dataset','')}]: dedupe_objects deactivated because deduplication is False")
        return

    # Configure logging in the child (forkserver children don't inherit parent's config)
    if '_log_level' in dataset_config:
        _, _, _, _log_utils, _, _ = _import_cruciblelib_modules()
        _log_utils.get_logger(log_type='transformer', log_level=dataset_config['_log_level'])

    global rc, wc, auth

    # Parse the dedup cycle interval once, defensively — a missing or malformed
    # deduplication_interval must not raise (and kill the process) at the bottom
    # of the loop.
    try:
        _dedupe_interval = int(dataset_config['deduplication_interval'])
    except (KeyError, ValueError, TypeError):
        _dedupe_interval = 30
        logging.warning(
            f"[{dataset_config.get('origin_dataset','')}]: invalid/missing "
            f"deduplication_interval; defaulting to {_dedupe_interval}s")

    # One-time setup (controller instantiation + config fetch) with retry.
    # Nothing re-spawns the dedup Process, so a transient failure here must be
    # retried rather than terminating deduplication for the life of the
    # transformer (previously an exception here killed the process silently).
    _setup_attempt = 0
    while True:
        try:
            auth, rc, wc = instantiate_api_controllers()

            config_list = find_and_validate_configs(
                dataset_config['perspective'],
                include_scripts=False,
                rc_instance=rc)

            # Check if any config in the perspective has deduplication enabled
            any_dedupe_enabled = any(
                _is_config_true(c.get('deduplication')) for c in config_list
            ) if config_list else _is_config_true(dataset_config.get('deduplication'))

            if not any_dedupe_enabled:
                logging.info(
                    f"[{dataset_config.get('origin_dataset','')}]: dedupe_objects deactivated "
                    f"because deduplication is disabled in perspective config")
                return

            # Build destination_ID_field_list once (it doesn't change between cycles)
            _dedupe_id_field_list = []
            for config in config_list:
                if isinstance(config['destination_unique_ID_column'], list):
                    _dedupe_id_field_list.extend(config['destination_unique_ID_column'])
                if isinstance(config['destination_unique_ID_column'], str):
                    _dedupe_id_field_list.append(config['destination_unique_ID_column'])
            _dedupe_id_field_list = list(set(_dedupe_id_field_list))
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            _setup_attempt += 1
            backoff = min(2 ** _setup_attempt, 60)
            logging.error(
                f"[{dataset_config.get('origin_dataset','')}]: dedupe_objects "
                f"setup failed (attempt {_setup_attempt}): {e}; retrying in {backoff}s")
            logging.error(traceback.format_exc())
            time.sleep(backoff)

    # Local cache of objects (ID-only columns) — populated on first cycle,
    # then incrementally updated with only recently-changed objects.
    cached_objects = None
    last_dedupe_time = None
    # Extra margin (seconds) added to since_seconds to account for clock
    # skew and Crucible write propagation lag.
    _dedupe_margin = 30

    while True:
        if _is_config_true(os.getenv('CRUCIBLE_DISABLE_DEDUPE', 'false')):
            logging.info(f"[{dataset_config.get('origin_dataset','')}]: dedupe_objects deactivated via CRUCIBLE_DISABLE_DEDUPE environment variable; exiting")
            break

        try:
            wc.token = auth.get_token()
            rc.token = auth.get_token()

            logging.info(f"Deduplicating {dataset_config['object_dataset']}")

            if cached_objects is None:
                # First cycle: full pull
                objects = retrieve_objects(dataset_config['object_dataset'],
                                           limit_to_ID_fields=True)
            elif len(cached_objects) == 0 or object_primary_key not in cached_objects.columns:
                # Cache is unusable (empty / column-less). This happens when a
                # prior cycle's full pull returned zero rows — e.g. the object
                # dataset was empty at startup or Crucible read-lag returned
                # nothing — and rc.search yielded a DataFrame with no columns.
                # That empty frame otherwise gets copied forward on every quiet
                # cycle and later makes the incremental merge crash (or run on a
                # partial view). Re-bootstrap with a full pull so the cache self-heals.
                logging.info(f"[{dataset_config['origin_dataset']}]: dedupe cache empty/uninitialized — re-bootstrapping with a full pull")
                objects = retrieve_objects(dataset_config['object_dataset'],
                                           limit_to_ID_fields=True)
            else:
                # Incremental: only pull objects changed since last cycle
                since = int(time.time() - last_dedupe_time) + _dedupe_margin
                fresh = retrieve_objects(dataset_config['object_dataset'],
                                         limit_to_ID_fields=True,
                                         since_seconds=since)
                if len(fresh) > 0:
                    # Merge incremental results into cache
                    objects = update_dataframe(
                        cached_objects, fresh,
                        unique_id_column_name=object_primary_key,
                        add_columns_and_rows_from_new_df_to_old_df=True,
                    )
                else:
                    objects = cached_objects.copy()

            last_dedupe_time = time.time()

            # Reconcile the local dedup cache against the AUTHORITATIVE set of
            # object UUIDs that currently exist in the live object dataset.
            # The incremental pull only ADDS changed rows, so objects that were
            # superseded (moved to the superseded_object_dataset by the object
            # manager), deleted by a prior dedup cycle, or removed externally
            # linger in the local cache. Deleting them as "duplicates" only
            # yields 404s. A supersede-map check misses any object removed
            # without a (still-in-window) management event, which is why 404s
            # persisted. Dropping every cached object whose UUID is no longer on
            # the server prunes ALL such stale entries (and self-heals the cache
            # via objects -> cached_objects below), so only objects that
            # actually still exist are ever considered for deletion.
            remote_uuids = None
            try:
                remote_uuids = _retrieve_remote_object_uuids(dataset_config['object_dataset'])
                if remote_uuids is not None and object_primary_key in objects.columns:
                    _before = len(objects)
                    objects = objects[
                        objects[object_primary_key].astype(str).isin(remote_uuids)
                    ].reset_index(drop=True)
                    _removed = _before - len(objects)
                    if _removed:
                        logging.info(
                            f"[{dataset_config['origin_dataset']}]: dedupe pruned "
                            f"{_removed} stale object(s) no longer present in "
                            f"{dataset_config['object_dataset']} from delete candidates")
            except Exception as e:
                logging.warning(
                    f"[{dataset_config['origin_dataset']}]: could not reconcile against "
                    f"remote object UUIDs for dedupe ({e}); proceeding with local cache")

            delete_events = []
            if len(objects) > 1:
                objects = objects.sort_values(by='crucibleHeader.updatedDate', ascending=False)
                col_names = objects.columns
                destination_ID_field_list = list(set(col_names).intersection(_dedupe_id_field_list))

                # Normalize integer ID columns so that custom IDs are consistent
                # regardless of how values were stored (e.g. float64 vs int64 vs string).
                int_conversion(objects, int_cols)

                # Use the production-proven row-wise custom-ID implementation.
                objects['custom_ID'] = objects[destination_ID_field_list].apply(
                    create_custom_id, axis=1)
                logging.info("Custom ID column created for deduplication:")
                with pd.option_context('display.max_colwidth', None):
                    logging.info(f"\n{objects[['custom_ID']].head().to_string()}")
                logging.info('-'*50)

                # Filter out objects with empty IDs (all NA values) - these should not be deduplicated
                objects_with_ids = objects[objects['custom_ID'].str.strip() != '']

                dupes = objects_with_ids.groupby('custom_ID').size()
                dupes = dupes[dupes > 1].index.tolist()

                coroutine_list = []
                delete_events = []
#new code starts here
                dropped_records = []
                # logging.info(f"[{dataset_config['origin_dataset']}]: Found {len(dupes)} duplicates in local copy of {dataset_config['object_dataset']}")
                for ID in dupes:
                    # Get all but the first (most recent due to sort) object with this ID
                    dupe_objects = objects_with_ids[objects_with_ids['custom_ID'] == ID]
                    # Skip the first one (keep it), delete the rest
                    for objectUUID in dupe_objects[object_primary_key].values[1:]:
                        logging.info(f"[{dataset_config['origin_dataset']}]: Deleting duplicate {objectUUID} with custom_ID {ID}")

                        # Flag the object as DROPPED in the live object dataset
                        # before deletion, so consumers see a tombstone.
                        dropped_records.append(
                            {'objectId': {'uuid': objectUUID}, 'entityStatus': 'DROPPED'})

                        # Create a DELETE event for the object management event dataset
                        delete_event = {
                            'objectId': objectUUID,
                            'action': 'DELETE',
                            "source": "MANAGER",
                            "edhControlSet": ["CLS:U"],
                        }
                        delete_events.append(delete_event)
                # Mark duplicates as DROPPED before deleting them, so consumers
                # see the deleted state as a tombstone (mirrors object_manager's
                # remove_superseded_objects).
                if dropped_records:
                    try:
                        wc.token = auth.get_token()
                        failed_dropped_records, _ = asyncio.run(write_batch_chunked(
                            dropped_records,
                            dataset_config['object_dataset'],
                            wc.update_entity_record_batch_by_name,
                            int(dataset_config.get('batch_update_chunk_size', len(dropped_records))),
                            label=f" [{dataset_config['origin_dataset']}] DROPPED flag: ",
                            token_refresher=lambda: setattr(wc, 'token', auth.get_token()),
                            max_concurrent_writes=dataset_config.get('batch_write_max_concurrent'),
                            transient_retry_attempts=2,
                        ))
                        failed_dropped_ids = set()
                        for record in failed_dropped_records:
                            object_id = record.get('objectId')
                            if isinstance(object_id, dict):
                                object_id = object_id.get('uuid')
                            if object_id:
                                failed_dropped_ids.add(str(object_id))
                        if failed_dropped_ids:
                            logging.error(
                                f"[{dataset_config['origin_dataset']}]: Leaving "
                                f"{len(failed_dropped_ids)} duplicate(s) in the live dataset "
                                f"because their DROPPED update failed")
                            delete_events = [
                                event for event in delete_events
                                if str(event['objectId']) not in failed_dropped_ids
                            ]
                        logging.info(
                            f"[{dataset_config['origin_dataset']}]: Flagged "
                            f"{len(delete_events)}/{len(dropped_records)} duplicate(s) as "
                            f"DROPPED before deletion")
                        # Allow consumers time to observe the DROPPED tombstone
                        # before deletion (mirrors object_manager's
                        # remove_superseded_objects).
                        if delete_events:
                            time.sleep(1)
                    except Exception as e:
                        logging.error(f"[{dataset_config['origin_dataset']}]: Error flagging duplicates as DROPPED: {e}")
                        logging.error(traceback.format_exc())
                        delete_events = []
                # Delete duplicate objects
                if delete_events:
                    coroutine_list = [
                        asyncio.to_thread(
                            wc.delete_entity_record_by_name,
                            dataset_config['object_dataset'],
                            event['objectId'])
                        for event in delete_events
                    ]
                    wc.token = auth.get_token()
                    asyncio.run(coroutine_launcher(coroutine_list))

                # Write DELETE actions to object management event dataset if configured
                if delete_events and 'object_management_event_dataset' in dataset_config:
                    try:
                        # Refresh token immediately before write operation
                        wc.token = auth.get_token()
                        wc.write_record_batch_by_name(dataset_config['object_management_event_dataset'], delete_events)
                        logging.info(f"[{dataset_config['origin_dataset']}]: Recorded {len(delete_events)} DELETE actions in {dataset_config['object_management_event_dataset']}")
                    except Exception as e:
                        logging.warning(f"[{dataset_config['origin_dataset']}]: Failed to record DELETE actions to Object Management Event Dataset: {e}")
            else:
                logging.info(f"[{dataset_config['origin_dataset']}]: No objects to deduplicate")

            # Update the cache: remove deleted UUIDs, store current state
            if delete_events:
                deleted_uuids = {e['objectId'] for e in delete_events}
                objects = objects[~objects[object_primary_key].isin(deleted_uuids)].reset_index(drop=True)
            # Drop the temporary custom_ID column before caching
            cached_objects = objects.drop(columns=['custom_ID'], errors='ignore')

        except Exception as e:
            # Never let a dedupe-cycle failure kill the dedup process; without
            # this, a transient API timeout silently terminates the dedup
            # subprocess for the lifetime of the transformer.
            logging.error(f"[{dataset_config['origin_dataset']}]: "
                          f"Deduplication cycle failed: {e}")
            logging.error(traceback.format_exc())

        time.sleep(_dedupe_interval)
        # nothing is returned from this function


def _remove_failed_new_objects(
        object_df: pd.DataFrame,
        object_updated_df: pd.DataFrame,
        failed_objects: List[Dict[str, Any]],
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Remove failed object definitions and realign the updated-row mask."""
    failed_uuids = set()
    for rec in failed_objects:
        obj_id = rec.get('objectId', {})
        uuid_val = obj_id.get('uuid') if isinstance(obj_id, dict) else obj_id
        if uuid_val:
            failed_uuids.add(str(uuid_val))

    if failed_uuids:
        object_df = object_df[
            ~object_df['objectId.uuid'].astype(str).isin(failed_uuids)
        ].reset_index(drop=True)
        object_updated_df = object_updated_df[
            ~object_updated_df['objectId.uuid'].astype(str).isin(failed_uuids)
        ].reset_index(drop=True)

    updated_rows_index = (object_df['updated'] == True)
    return object_df, object_updated_df, updated_rows_index


async def process_events(event: Any, 
                         object_df: pd.DataFrame, 
                         dataset_config: dict, 
                         object_enricher: Any,
                         ) -> pd.DataFrame:
    """
    Performs processing of events received from the
    SSE listener. 

    Args:
        event: a server-side event
        object_df (pd.DataFrame):
        dataset_config (dict):
        object_enricher (Any):
    Returns:
        tuple: (object_df: pd.DataFrame)
    """
    _, _, utils, _, _, DetailedHTTPError = _import_cruciblelib_modules()

    try:

        ##########################################
        # Read in the event data
        ##########################################

        if event.type == 'keep-alive':
            logging.info(" "+dataset_config['origin_dataset']+": "+"Received keep-alive, continue...")
        else:
            msg_length = len(event.data)
            if msg_length > 5:
                info_string = 'Message received for ' + dataset_config['query']
                logging.info('-' * 50)
                logging.info(' '+dataset_config['origin_dataset']+' '+'Transformer Event Processing')
                logging.info(info_string)

                ##########################################
                # Pre-processing of event data
                ##########################################

                # convert incoming JSON message to DataFrame
                # [PERF] per-stage wall-clock timing to confirm the per-event
                # bottleneck (merge_and_update CPU vs the Crucible writes I/O).
                # Toggle every [PERF] line via PERF_TIMING at the top of the file.
                _perf = time.perf_counter if PERF_TIMING else _perf_disabled
                _t0 = _perf(); _t = _t0; _timings = {}
                event_df = flatten_crucible_dataset(event.data,)


                
                # make unit conversions if necessary
                if 'unit_conversions' in dataset_config:
                    unit_conversion_mapping=dataset_config["unit_conversions"]
                    for j in range(len(unit_conversion_mapping)):
                        funcstring = unit_conversion_mapping[j]['unit_conversion']
                        origin_col = unit_conversion_mapping[j]['origin_column']
                        if not ('PLACEHOLDER' in funcstring.upper()):
                            if origin_col not in event_df.columns:
                                logging.warning(f"[{dataset_config['origin_dataset']}]: unit_conversion skipped - column '{origin_col}' not found in event data")
                                continue
                            func = eval("unit_conversions."+funcstring)
                            event_df[origin_col] = \
                                event_df[origin_col].map(lambda x: func(x))

                ##########################################
                # run custom function if necessary
                ##########################################
                if 'custom_functions' in dataset_config:
                    for j in range(len(dataset_config['custom_functions'])):
                        custom_func = eval(
                            "custom_functions." + dataset_config['custom_functions'][j]['function_name'])
                        event_df, object_df = custom_func(event_df, object_df,
                                                                    dataset_config)

                event_df = copy.deepcopy(event_df)  # this is necessary to avoid a SettingWithCopyWarning
                                                    # under certain conditions.


                ##########################################
                # Field name mapping.
                # event_df column names will be mapped to 
                # object_df column names using the source_to_destination_mapping.
                # This will allow us to update the object_df with values from event_df.
                ##########################################

        
                literal_mapping = dataset_config['origin_to_destination_mapping']
                literal_mapping = [el for el in literal_mapping if 'literal' in el.keys()]

                for mapping in literal_mapping:
                    event_df[mapping['destination_column']] = mapping['literal']
                    if 'type' in mapping:
                        if mapping['type'].lower() == 'int':
                            event_df[mapping['destination_column']] = event_df[mapping['destination_column']].astype('Int64')
                        elif mapping['type'].lower() == 'float':
                            event_df[mapping['destination_column']] = event_df[mapping['destination_column']].astype(float)
                        elif mapping['type'].lower() == 'str':
                            event_df[mapping['destination_column']] = event_df[mapping['destination_column']].astype(str)
                        # add other types as needed



                event_df = rename_columns(event_df, dataset_config['origin_to_destination_mapping'], dataset_config['origin_dataset'])
                

                ##########################################
                # Normalize integer ID columns early
                # so that merge/update operations see consistent
                # Int64 types regardless of whether the source
                # feed sends strings or ints.
                ##########################################
                int_conversion(event_df, int_cols)
                int_conversion(object_df, int_cols)

                ##########################################
                # Create ID strings for event and object dataframes
                ##########################################
      
       
                
                object_df, event_df = create_ID_column(object_df,event_df,dataset_config)
                object_df, event_df = _drop_blank_ids_and_dedupe(
                    object_df, event_df, dataset_config['destination_unique_ID_column'],
                    dataset_config['origin_dataset'])
                _now = _perf(); _timings['preprocess'] = _now - _t; _t = _now


                
                ##########################################
                # create new objects for dynamic object correlation
                ##########################################


                if "dynamic" in dataset_config['correlation_type'].lower():
                    
                    # see if uuid is in the origin_to_destination_mapping
                    generate_uuid=True
                    for item in dataset_config['origin_to_destination_mapping']:
                        if object_primary_key in item['destination_column']:
                            generate_uuid=False
        
                    object_df, event_df = create_new_objects(event_df,
                                    object_df,
                                    custom_ID_column_name,
                                    generate_uuid)

                ##########################################
                # Update objects with properties
                # from incoming message (event_df)
                ##########################################
        
                _now = _perf(); _timings['create_new_objects'] = _now - _t; _t = _now
                object_df,object_updated_df = merge_and_update(
                            object_df, event_df,
                            dataset_config,)
                _now = _perf(); _timings['merge_and_update'] = _now - _t; _t = _now


                ##########################################
                # enrich object_df with reference data
                ##########################################

                if object_enricher:
                    # enrich the new objects with reference data
                    new_object_rows = object_df['new_object'] # selects all rows where new_object is True
                    if len(object_df[new_object_rows]) > 0:
                        new_object_df = copy.deepcopy(object_df[new_object_rows])
                        enriched_object_df = object_enricher.enrich_object_with_reference_data(new_object_df,event_df)
                        object_df=update_dataframe(object_df, enriched_object_df,unique_id_column_name=custom_ID_column_name)
                        object_updated_df=update_dataframe(object_updated_df, enriched_object_df,unique_id_column_name=custom_ID_column_name)
                        object_updated_df=object_updated_df.copy() # recommended to avoid performance warning
              

                ##########################################
                # set the source dataset name 
                ##########################################

                object_updated_df['source.datasetName'] = get_dataset_name(dataset_config["query"])
                
                ##########################################
                # get subset of rows that have been updated
                ##########################################

                updated_rows_index = (object_df['updated'] == True)

                ##########################################
                # set the descriptive label
                ##########################################
       
                # objectId.descriptiveLabel is a REQUIRED field on Live_POV_Objects, and
                # _fast_df_to_nested_json drops empty strings before the write, so it MUST
                # be non-empty or the upsert 400s ("required property 'descriptiveLabel'
                # not found"). The old semantic label was computed row-by-row via
                # returnDescriptiveLabel(...).apply(axis=1) — a CPU hot spot (the bulk of
                # the 'finalize' bucket). Replaced with a CHEAP VECTORIZED fallback: the
                # first non-blank of a few identity fields, else the environment, else
                # 'UNKNOWN'. No per-row Python.
                _dl = None
                for _dl_col in ('identity.sconum','identity.callsign', 
                                'identity.vesselName', 'identity.dynamicIdentifier', 'identity.airPlatformType',
                                'identity.hullNumber', 'identity.environment.environment'):
                    if _dl_col not in object_df.columns:
                        continue
                    _dl_s = object_df.loc[updated_rows_index, _dl_col]
                    # treat NaN / blank / whitespace as missing so fillna cascades
                    _dl_s = _dl_s.where(_dl_s.notna() & (_dl_s.astype(str).str.strip() != ''))
                    _dl = _dl_s if _dl is None else _dl.fillna(_dl_s)
                if _dl is None:
                    _dl = pd.Series('UNKNOWN', index=object_df.loc[updated_rows_index].index)
                else:
                    _dl = _dl.fillna('UNKNOWN')
                object_df.loc[updated_rows_index, 'objectId.descriptiveLabel'] = _dl
                object_updated_df.loc[updated_rows_index, 'objectId.descriptiveLabel'] = object_df.loc[updated_rows_index, 'objectId.descriptiveLabel']



                ##########################################
                # send Objects to Crucible
                ##########################################


                if (True in updated_rows_index.tolist()):

                    # Convert specific object schema columns to Int64, if present,
                    # to avoid conversion to floats. (And not lower-case int64; Int64 allows for NA values,
                    #  which are common in these identifier columns, and prevents them from being converted to floats with decimal points)

                    int_conversion(object_updated_df, int_cols)
                    int_conversion(object_df, int_cols)

                    # Get timestamp for this update & standardize all timestamp formats
                    object_updated_df = update_timestamps(object_updated_df, updated_rows_index)
                    object_df = update_timestamps(object_df, updated_rows_index)

                    # Stamp a local-only write timestamp on every row we are about
                    # to create/update.  The reconcile safety window in _prune_missing_remote_uuids()
                    # uses this to protect freshly-created objects from being pruned
                    # before Crucible's read lag clears (their crucibleHeader is
                    # stripped before the upsert, so crucibleHeader.updatedDate is
                    # not available locally).
                    object_df.loc[updated_rows_index, local_write_time_column] = get_current_timestamp_string()

                    _timings['finalize'] = _perf() - _t
                    # note that object_df doesn't include kinematics, its seems
                    new_objects_df = object_df[object_df['new_object']]
                    if len(new_objects_df) > 0:

                        # Drop every non-object-definition column group in ONE
                        # slice (was 5 separate full-frame copies).
                        new_objects_df = new_objects_df.loc[:, ~new_objects_df.columns.str.startswith(
                            ('source', 'upstreamSource', 'collectionType',
                             'estimatedKinematics.uncertainty.uncertaintyEllipse', 'crucibleHeader'))]

                        # upsert new object to to object_dataset (e.g., Live_POV_Objects)
                        info_string = " ["+dataset_config['origin_dataset']+"]: "+'# of new objects defined: ' + str(len(new_objects_df))
                        logging.info('')
                        logging.info(info_string)
                        logging.info(" ["+dataset_config['origin_dataset']+"]: "+f'   and sent to {dataset_config["object_dataset"]}')
                        logging.info('')

                        
                        # no longer need to fix_dtypes here
                        # because of int_conversion() earlier
                        # and df_to_formatted_JSON() should preserve dtypes, not convert to strings
                        ##### fix_dtypes(new_objects_df)

                        # Convert DataFrame to properly formatted JSON
                        JSON_object_definition = _fast_df_to_nested_json(
                            new_objects_df.drop(columns=temp_col_list),\
                            )

                        # Guard: never POST an object whose objectId.uuid failed
                        # to resolve (e.g. multi-valued/blank unique ID) — it only
                        # yields a "required property 'uuid' not found" 400 loop.
                        JSON_object_definition = drop_records_missing_object_id(
                            JSON_object_definition,
                            label=f' [{dataset_config["origin_dataset"]}]: ')

                        chunk_size = int(dataset_config['batch_update_chunk_size'])
                        max_concurrent = dataset_config.get('batch_write_max_concurrent')
                        _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
                        _t_wnew = _perf()
                        failed_objects, _ = await write_batch_chunked(
                            JSON_object_definition,
                            dataset_config['object_dataset'],
                            wc.upsert_by_name,
                            chunk_size,
                            label=f' [{dataset_config["origin_dataset"]}]: ',
                            token_refresher=_refresh_token,
                            max_concurrent_writes=max_concurrent,
                            transient_retry_attempts=2,
                        )
                        _timings['write_new_objects'] = _perf() - _t_wnew
                        if failed_objects:
                            # Extract UUIDs of objects that failed to write — these don't exist
                            # in the object dataset yet, so we must remove them from object_df
                            # and object_updated_df to avoid downstream "object not found" errors.
                            logging.warning(
                                f' [{dataset_config["origin_dataset"]}]: {len(failed_objects)} new objects failed to write to '
                                f'{dataset_config["object_dataset"]}; removing from processing to avoid downstream errors'
                            )
                            object_df, object_updated_df, updated_rows_index = _remove_failed_new_objects(
                                object_df, object_updated_df, failed_objects)

                    else:
                        logging.info('')
                        logging.info(" ["+dataset_config['origin_dataset']+"]: "+'No new objects defined')
                        logging.info('')

                    # Get only those objects that have been updated
                    # in prep for insert
                    _t_be = _perf()
                    object_updated_df = object_updated_df[updated_rows_index]
                    logging.info('')
                    logging.info(f'# records from {get_dataset_name(dataset_config["query"])}: {len(event_df)}')

                    
                    # fix_dtypes no longer need to fix dtypes here
                    # because of int_conversion() earlier
                    # and df_to_formatted_JSON() should preserve dtypes, not convert to strings
                    #### fix_dtypes(object_updated_df)

                    # Convert DataFrame to properly formatted JSON
                    object_updated_df.drop(columns=temp_col_list,inplace=True)
                    object_updated_df = object_updated_df.loc[:, ~object_updated_df.columns.str.startswith('crucibleHeader')]

                    # send kinematics and state updates separately

                    # first create state updates (select the non-kinematics
                    # columns directly instead of copying the whole frame first)
                    state_df = object_updated_df.loc[:, ~object_updated_df.columns.str.startswith('estimatedKinematics')].copy()
                    state_df['eventType'] = 'STATE_UPDATE'
                    # Suppress repetitive, low-information STATE_UPDATEs: a position
                    # feed re-sends the same identity/mode/edh every kinematic tick, so
                    # only emit a state update when an object's meaningful state fields
                    # changed since the last one written — but let one duplicate through
                    # per object every state_heartbeat_seconds as a heartbeat so state is
                    # never fully silent. Opt out per feed with dedupe_state_updates: false.
                    if _is_config_true(dataset_config.get('dedupe_state_updates', True)):
                        state_df = _filter_repeated_state_updates(
                            state_df, dataset_config.get('origin_dataset', ''),
                            heartbeat_seconds=dataset_config.get('state_heartbeat_seconds',
                                                                 _STATE_HEARTBEAT_SECONDS_DEFAULT),
                            heartbeat_every_n=dataset_config.get('state_heartbeat_every_n',
                                                                 _STATE_HEARTBEAT_EVERY_N_DEFAULT))
                    state_JSON = _fast_df_to_nested_json(state_df)
                    state_JSON = drop_records_missing_object_id(
                        state_JSON, label=f' [{dataset_config["origin_dataset"]}]: ')
                    logging.info(" ["+dataset_config['origin_dataset']+"]: "+ \
                                 f'Sending {len(state_df)} state updates to {dataset_config["object_event_dataset"]}')
      

                    # then create kinematic updates (build the column subset
                    # directly instead of copying the whole frame first)
                    col_names = object_updated_df.columns
                    kin_cols = []
                    for col in col_names:
                        if 'estimatedKinematics' in col:
                            kin_cols.append(col)
                        if 'objectId' in col:
                            kin_cols.append(col)
                        if 'source' in col:
                            kin_cols.append(col)
                        if 'upstreamSource' in col:
                            kin_cols.append(col)
                        if 'collectionType' in col:
                            kin_cols.append(col)
                        if 'edhControlSet' in col:
                            kin_cols.append(col)
                        if 'mode' in col:
                            kin_cols.append(col)
                        if 'identity' in col:
                            kin_cols.append(col)
                        if 'ecef' in col:
                            kin_cols.append(col)
                    kin_df = object_updated_df.loc[:, kin_cols].copy()
                    kin_df['eventType'] = 'KINEMATIC_UPDATE'
                    
                    if 'skip_ECEF_conversion' in dataset_config.keys() and dataset_config['skip_ECEF_conversion']:
                        pass
                        
                    else:
                        kin_df = add_ECEF_kinematics(kin_df)

                    if kin_df.empty:
                        logging.info(" ["+dataset_config['origin_dataset']+"]: "+'No kinematic updates to send')
                        kin_JSON = None
                    else:
                        kin_JSON = _fast_df_to_nested_json(kin_df)
                        kin_JSON = drop_records_missing_object_id(
                            kin_JSON, label=f' [{dataset_config["origin_dataset"]}]: ')
                        logging.info(" ["+dataset_config['origin_dataset']+"]: "+ \
                                     f'Sending {len(kin_df)} kinematic updates to {dataset_config["object_event_dataset"]}')

                    # write formatted JSON to Crucible object events dataset concurrently
                    chunk_size = int(dataset_config['batch_write_chunk_size'])
                    max_concurrent = dataset_config.get('batch_write_max_concurrent')
                    _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
                    write_tasks = []
                    for event_type, JSON_object in (
                            ('STATE_UPDATE', state_JSON),
                            ('KINEMATIC_UPDATE', kin_JSON)):
                        if JSON_object:
                            write_tasks.append((
                                event_type,
                                write_batch_chunked(
                                    JSON_object,
                                    dataset_config["object_event_dataset"],
                                    wc.write_record_batch_by_name,
                                    chunk_size,
                                    label=f' [{dataset_config["origin_dataset"]}]: ',
                                    token_refresher=_refresh_token,
                                    max_concurrent_writes=max_concurrent,
                                ),
                            ))
                    _timings['build_events'] = _perf() - _t_be
                    _t_we = _perf()
                    if write_tasks:
                        write_results = await asyncio.gather(
                            *(task for _, task in write_tasks))
                        for (event_type, _), (failed_records, _) in zip(
                                write_tasks, write_results):
                            if failed_records:
                                failed_object_ids = []
                                for record in failed_records:
                                    object_id = record.get('objectId')
                                    if isinstance(object_id, dict):
                                        object_id = object_id.get('uuid')
                                    if object_id:
                                        failed_object_ids.append(str(object_id))
                                logging.error(
                                    f' [{dataset_config["origin_dataset"]}]: '
                                    f'{len(failed_records)} {event_type} event(s) '
                                    f'failed to write to {dataset_config["object_event_dataset"]}; '
                                    f'append-only events were not replayed to avoid duplicates. '
                                    f'objectIds={failed_object_ids[:5]}'
                                )
                    _timings['write_events'] = _perf() - _t_we
                    if PERF_TIMING:
                        _timings['total'] = _perf() - _t0
                        _timings['other'] = _timings['total'] - (
                            _timings.get('preprocess', 0.0)
                            + _timings.get('create_new_objects', 0.0)
                            + _timings.get('merge_and_update', 0.0)
                            + _timings.get('finalize', 0.0)
                            + _timings.get('build_events', 0.0)
                            + _timings.get('write_new_objects', 0.0)
                            + _timings.get('write_events', 0.0))
                        logging.info(
                            f" [{dataset_config['origin_dataset']}]: [PERF transformer] "
                            f"total={_timings['total']:.3f}s "
                            f"merge_and_update={_timings.get('merge_and_update', 0.0):.3f}s "
                            f"write_events={_timings.get('write_events', 0.0):.3f}s "
                            f"write_new_objects={_timings.get('write_new_objects', 0.0):.3f}s "
                            f"finalize={_timings.get('finalize', 0.0):.3f}s "
                            f"build_events={_timings.get('build_events', 0.0):.3f}s "
                            f"preprocess={_timings.get('preprocess', 0.0):.3f}s "
                            f"create_new_objects={_timings.get('create_new_objects', 0.0):.3f}s "
                            f"other={_timings['other']:.3f}s "
                            f"| object_df_rows={len(object_df)} event_rows={len(event_df)} "
                            f"updated_rows={int(updated_rows_index.sum())}")

                    # write formatted JSON to Crucible entity dataset in a new thread
                    # this is deprecated, but kept for backward compatibility
                    if 'skip_object_manager' in dataset_config.keys() and \
                        dataset_config['skip_object_manager']:

                        object_diffs = object_updated_df.copy()
                        object_diffs = object_diffs.loc[:, ~object_diffs.columns.str.startswith(
                            ('source', 'upstreamSource'))]

                        JSON_object = _fast_df_to_nested_json(object_diffs )
                        # Update objects directly, bypassing the object manager
                        logging.info(" ["+dataset_config['origin_dataset']+"]: "+ \
                                     f'Sending {len(object_diffs)} object updates to {dataset_config["object_dataset"]}')
                        logging.info('')
                        update_chunk_size = int(dataset_config['batch_update_chunk_size'])
                        _refresh_token = lambda: setattr(wc, 'token', auth.get_token())
                        failed_object_updates, _ = await write_batch_chunked(
                            JSON_object,
                            dataset_config['object_dataset'],
                            wc.update_entity_record_batch_by_name,
                            update_chunk_size,
                            label=f' [{dataset_config["origin_dataset"]}]: ',
                            token_refresher=_refresh_token,
                            max_concurrent_writes=max_concurrent,
                            transient_retry_attempts=2,
                        )
                        if failed_object_updates:
                            logging.error(
                                f' [{dataset_config["origin_dataset"]}]: '
                                f'{len(failed_object_updates)} direct object update(s) '
                                f'failed after transient retries'
                            )
                        

                    logging.info('-' * 50)

                else:
                    logging.info(" ["+dataset_config['origin_dataset']+"]: "+ \
                                 'Insert skipped: no matches found between event and object datasets')


            else:

                status_string = " ["+dataset_config['origin_dataset']+"]: "+\
                        'Empty message received for ' + dataset_config["query"]
                logging.info(status_string)

        return object_df
    
    except DetailedHTTPError as e:
        logging.error(f"[{dataset_config['origin_dataset']}]: An error occurred in process_events: {e}")
        # logging.error(traceback.format_exc())
        return object_df

    except Exception as e:
        logging.error(f"[{dataset_config['origin_dataset']}]: An error occurred in process_events: {e}")
        logging.error(traceback.format_exc())
        return object_df      

            
def create_new_objects(event_df: pd.DataFrame,
                        object_df: pd.DataFrame,
                        custom_ID_column_name: str,
                        generate_uuid: bool = True,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates new objects for dynamic object correlation
    """
    def _custom_id_has_value(custom_id: str) -> bool:
        return any(
            part.split('-', 1)[1].strip()
            for part in str(custom_id).split('_')
            if '-' in part
        )

    event_ids = event_df[custom_ID_column_name].dropna().astype(str)
    defined_ids = set(object_df[custom_ID_column_name].dropna().astype(str))
    # Preserve first-seen order and never create a shared "blank identity" object
    # for rows whose unique-ID fields are all empty/NA.
    new_object_list = [
        oid for oid in pd.unique(event_ids)
        if _custom_id_has_value(oid) and oid not in defined_ids
    ]
    if len(new_object_list) > 0:


        first_empty_row_index = len(object_df) # this will be the index of the first empty row
        object_df = add_blank_rows(object_df, len(new_object_list))
        object_df[custom_ID_column_name]=object_df[custom_ID_column_name].fillna('')
        # get first empty row in object_df

        # add new objects to object_df — vectorised (single update() call
        # instead of one per object, which was O(n²) for large batches)

        template_row = create_object_from_template().iloc[0].to_dict()
        n = len(new_object_list)
        indices = list(range(first_empty_row_index, first_empty_row_index + n))
        new_rows_df = pd.DataFrame([template_row] * n, index=indices)

        if generate_uuid:
            # Deterministic UUID derived from the custom ID so the same
            # object always maps to the same objectId across runs.
            new_rows_df[object_primary_key] = [
                uuid.uuid5(uuid.NAMESPACE_DNS, str(oid)).hex.lower()
                for oid in new_object_list
            ]

        # Single update() fills all blank rows at once
        object_df.update(new_rows_df)
        object_df.loc[indices, custom_ID_column_name] = new_object_list
        object_df.loc[indices, 'updated'] = True
        object_df.loc[indices, 'new_object'] = True

        # add these default columns to the event df:
        # left join event df and object df, but only on the new objects
        event_df_with_pydantic_defaults = event_df.merge(
            object_df[object_df['new_object']],
            how='left', on=custom_ID_column_name, suffixes=('_old', None))
        # drop _old columns
        droplist = []
        for col in event_df_with_pydantic_defaults.columns:
            if '_old' in col:
                droplist.append(col)
        event_df_with_pydantic_defaults = event_df_with_pydantic_defaults.drop(columns=droplist)

        event_df = event_df.merge(event_df_with_pydantic_defaults, how='left', on=custom_ID_column_name, suffixes=(None, '_defaults'))
        # drop _old columns
        droplist = []
        for col in event_df.columns:
            if '_defaults' in col:
                droplist.append(col)
        event_df = event_df.drop(columns=droplist)

    return object_df, event_df


def merge_and_update(object_df: pd.DataFrame,
                 event_df: pd.DataFrame,
                 dataset_config: dict,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Updates object_df with info from event_df. This is done by merging the two dataframes.
    The output object_updated_df contains diffs to object_df. It has None types
    in columns that weren't updated. The output object_df contains all values, including
    None types.


    Args:
        object_df (pd.DataFrame):
        event_df (pd.DataFrame):
        dataset_config (dict):

    Returns:
        object_df: pd.DataFrame
        object_updated_df: pd.DataFrame

    '''

    # make a copy of object_df; this is what will hold the merged event data,
    # while object_df only holds the previous object data.
    object_updated_df = object_df.copy()

    # Strip crucibleHeader columns from event_df so that source-feed
    # metadata (updatedDate, createdDate, tenantId, …) does not
    # overwrite the object's own crucibleHeader.  The object's
    # crucibleHeader.updatedDate must reflect when the *entity* was
    # last written to the platform, not when the source event was
    # ingested — otherwise the reconcile safety window never expires
    # and externally-deleted objects can never be pruned.
    event_df = event_df.loc[:, ~event_df.columns.str.startswith('crucibleHeader')]

    # Add columns from the mapping that weren't in the origin object_df.
    # This is necessary because the mapping may have columns that aren't in
    # object_updated_df.
    # Those missing columns need to be present for update() to work below.
    column_list = [dictionary['destination_column'] for dictionary in dataset_config['origin_to_destination_mapping']]
    # Add any mapping columns missing from object_df.  Copy object_df ONCE
    # (it was previously copied once per missing column, duplicating the whole
    # up-to-81k-row frame N times) so update() can still propagate them.
    missing_columns = [column for column in column_list if column not in object_df.columns]
    if missing_columns:
        object_df = object_df.copy()
        for column in missing_columns:
            object_updated_df[column] = None
            object_df[column] = None

    
    # remove old values from object_updated_df
    # so that they're not propagated to the new objects
    col_names = [col for col in object_updated_df.columns if
                 (col != custom_ID_column_name
                  and col != 'updated'
                  and col != object_primary_key
                  and (col not in dataset_config['destination_unique_ID_column']))]
    # Convert numpy numeric/bool columns to object so they can hold None.
    # Pandas 2.x rejects None for non-nullable int64, float64, and bool dtypes.
    for col in col_names:
        if object_updated_df[col].dtype.kind in ('i', 'u', 'f', 'b'):
            object_updated_df[col] = object_updated_df[col].astype('object')
    object_updated_df.loc[:, col_names] = None

    # merge event_df into object_updated_df
    # using an **inner** join to get only rows that match an event
    merged_df_with_newer_vals = object_updated_df.merge(event_df, how='inner',
                                                        on=custom_ID_column_name,
                                                        suffixes=('_old', None))
    merged_df_with_newer_vals['updated'] = True

    droplist = []
    for col in merged_df_with_newer_vals.columns:
        if '_old' in col:
            droplist.append(col)
    merged_df_with_newer_vals = merged_df_with_newer_vals.drop(columns=droplist)
    
    # **left** join to get the rows from merged_df_with_newer_vals
    # to line up with object_updated_df.
    # This is prep for the update step below, which assumes that all rows in the
    # object_updated_df are present in the merged_df_with_newer_vals.
    # The drop_duplicates() step above is necessary because the merge above
    # can create duplicate rows in merged_df_with_newer_vals.
    merged_df_with_newer_vals = object_updated_df.merge(merged_df_with_newer_vals, how='left',
                                                        on=custom_ID_column_name,
                                                        suffixes=('_old', None))

    droplist = []
    for col in merged_df_with_newer_vals.columns:
        if '_old' in col:
            droplist.append(col)
    merged_df_with_newer_vals = merged_df_with_newer_vals.drop(columns=droplist)

    # Note on the update method: 
    # it updates values in place using
    # columns that are present in both dataframes.
    # This is used to avoid adding new columns from event_df
    # unless they are explicitly mapped in the config file.

    # Pandas 2.x update() rejects values whose dtype doesn't match the
    # target column (e.g. float values into int64).  Convert all rigid
    # numpy numeric/bool columns to object so update() can accept any value.
    # This also covers int_cols (identity.navalPennant, identity.mmsiNumber,
    # trackQuality) — they'll be re-normalized to Int64 by int_conversion()
    # in process_events() before writing to Crucible.
    for df in (object_df, object_updated_df, merged_df_with_newer_vals):
        _rigid_cols = df.select_dtypes(include=['number', 'bool']).columns
        if len(_rigid_cols) > 0:
            df[_rigid_cols] = df[_rigid_cols].astype('object')

    object_updated_df.update(merged_df_with_newer_vals)
    object_df.update(object_updated_df)
    

    return object_df,object_updated_df


def rename_columns(event_df: pd.DataFrame, origin_to_destination_mapping: List[dict], origin_dataset: str) -> pd.DataFrame:
    """
    Renames columns in a pandas DataFrame according to the mappings in
    the list origin_to_destination_mapping.

    Args:
        event_df (pd.DataFrame):
        origin_to_destination_mapping (str):

    Returns:
        pd.DataFrame
    """


    # Filter out literal mappings (they don't need column renaming)
    literal_mappings = [el for el in origin_to_destination_mapping if 'literal' in el.keys()]
    column_mappings = [el for el in origin_to_destination_mapping if 'literal' not in el.keys()]
    
    mapping_list = [[el["origin_column"], el["destination_column"]] for el in column_mappings]
    # Add literal mappings as special case: destination_column -> destination_column
    for el in literal_mappings:
        mapping_list.append([el["destination_column"], el["destination_column"]])
    orig_cols = [el[0] for el in mapping_list]
    dest_cols = [el[1] for el in mapping_list]

    if len(dest_cols) != len(set(dest_cols)):
        # duplicate column in destination
        raise ValueError(
            'Duplicate destination columns in config files are not allowed')
    # Now check for duplicate **origin** columns in mapping of origin->destination
    # this is allowed and is handled below:
    index_dict = {}
    for i, col in enumerate(orig_cols):
        if col not in index_dict:
            index_dict[col] = 1
        else:
            index_dict[col] += 1
            # Give this repeated origin mapping its own temporary source column.
            new_column_name = col + '__' + str(index_dict[col])
            event_df[new_column_name] = event_df[col]
            mapping_list[i][0] = new_column_name

    event_cols = event_df.columns.values.tolist()
    for i,col in enumerate(dest_cols):
        if col in event_cols:
            # A true same-name mapping needs a temporary source before the
            # destination is removed. Otherwise retain the declared origin and
            # discard the stale destination value (for example,
            # crucibleHeader.uuid -> source.uuid).
            if mapping_list[i][0] == col:
                new_column_name = col + '__'+str(i)
                event_df[new_column_name] = event_df[col]
                mapping_list[i][0] = new_column_name
            event_df.drop(columns=[col],inplace=True)
       
    mapping_dict = dict(mapping_list) 

    try:
        event_df = event_df.rename(
            columns=mapping_dict,
            inplace=False,
            errors='raise') # an error is raised if one or more mappings fail

    except Exception:
        logging.warning(' ')
        logging.warning('Warning for origin-to-destination mapping:')
        for key in mapping_dict.keys():
            if key not in event_df.columns.values.tolist():
                logging.warning(f'[{origin_dataset}]:    {key} --> {mapping_dict[key]} mapping skipped ')

        event_df = event_df.rename(
            columns=mapping_dict,
            inplace=False,) # no error is raised if mappings fail

    if 'source.uuid' in event_df.columns:
        event_df['source.uuid'] = event_df['source.uuid'].map(
            lambda value: normalize_uuid(value) if pd.notna(value) else value).to_numpy()

    return event_df

def populate_source_column(object_df: pd.DataFrame, event_df: pd.DataFrame, dataset_config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # create source_column if it doesn't already exist
    if dataset_config['source_column'] not in event_df.columns:
        event_df[dataset_config['source_column']] = pd.NA
    if dataset_config['source_column'] not in object_df.columns:
        object_df[dataset_config['source_column']] = pd.NA
    # make sure it is an 'object' column to allow for lists & dicts in cells
    event_df[dataset_config['source_column']] = event_df[dataset_config['source_column']].astype('object')
    object_df[dataset_config['source_column']] = object_df[dataset_config['source_column']].astype('object')

    # use .apply to place dictionary of source info into cells:
    current_timestamp_string = get_current_timestamp_string()
    event_df[dataset_config['source_column']] = \
        event_df.apply(
            lambda row: create_source(row, dataset_config['source_mapping'],
                                      dataset_config['query'],
                                      current_timestamp_string),
            axis=1)
    
    return object_df, event_df

def create_ID_column(object_df: pd.DataFrame, event_df: pd.DataFrame, dataset_config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This functions adds a custom unique ID column that can be used
    for joining event and object dataframes. It consists of a concatenation
    of the columns in dataset_config['origin_unique_ID_column'] for event_df
    and dataset_config['destination_unique_ID_column'] for object_df.
    It will be dropped before write.

    """

    # create a new column, if it doesn't already exist, to record row updates
    # rows that aren't updated will be dropped before insert/upsert
    object_df['updated'] = False
    object_df['new_object'] = False
    # Ensure the local write-time column exists (used by the reconcile safety
    # window) WITHOUT clobbering values stamped on prior events.
    if local_write_time_column not in object_df.columns:
        object_df[local_write_time_column] = None

    # origin_ID_field_list = dataset_config['origin_unique_ID_column']
    destination_ID_field_list = dataset_config['destination_unique_ID_column']
    
    # Add columns to both dataframes if they don't exist
    for col in destination_ID_field_list:
        if col not in object_df.columns:
            object_df[col] = pd.NA
        if col not in event_df.columns:
            event_df[col] = pd.NA
  

    object_df[custom_ID_column_name] = object_df[destination_ID_field_list].apply(
        create_custom_id, axis=1)
    event_df[custom_ID_column_name] = event_df[destination_ID_field_list].apply(
        create_custom_id, axis=1)
    logging.debug(f"Object custom ID from process_events:\n{object_df[custom_ID_column_name].head().to_string()}")
    logging.debug(f"Event custom ID from process_events:\n{event_df[custom_ID_column_name].head().to_string()}")
    logging.debug('-'*50)

    return object_df, event_df


def _drop_blank_ids_and_dedupe(object_df: pd.DataFrame, event_df: pd.DataFrame,
                               id_cols: List[str], origin: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows with a blank unique-ID value from matching or creating objects,
    then collapse duplicate custom IDs in both frames."""
    event_blank = _has_blank_id(event_df, id_cols)
    if event_blank.any():
        logging.warning(f"[{origin}]: Dropped {int(event_blank.sum())} "
                        f"record(s) with a blank value in unique ID field(s) {id_cols}")
        event_df = event_df.loc[~event_blank]

    # Blank-ID cached objects get no key (unmatchable) and are kept out of the dedupe.
    obj_blank = _has_blank_id(object_df, id_cols)
    object_df.loc[obj_blank, custom_ID_column_name] = None
    duplicate = object_df[custom_ID_column_name].duplicated(keep='last') & ~obj_blank
    if duplicate.any():
        logging.debug(f"[{origin}]: Duplicate rows found in local copy of object_df. Removing duplicates and continuing.")
        object_df = object_df.loc[~duplicate].reset_index(drop=True)

    event_df = event_df.groupby(custom_ID_column_name, as_index=False).agg('last')
    return object_df, event_df


def _has_blank_id(df: pd.DataFrame, id_cols: List[str]) -> pd.Series:
    """True for rows where any unique-ID column is NA or blank."""
    if df.empty:
        return pd.Series(False, index=df.index)
    return df[id_cols].apply(lambda s: s.map(lambda v: safe_str(v).strip() == '')).any(axis=1)


def update_dataframe(old_df: pd.DataFrame, new_df: pd.DataFrame,
                     unique_id_column_name: str = object_primary_key,
                     add_columns_and_rows_from_new_df_to_old_df: bool = True,
                     only_add_new_rows: bool = False ) -> pd.DataFrame:
    """
    Updates data in the 'old' DataFrame with data from the 'new' DataFrame.

    Args:
        old_df (pd.DataFrame):
        new_df (pd.DataFrame):
        unique_id_column_name (str):
        add_columns_and_rows_from_new_df_to_old_df (bool):
        only_add_new_rows (bool): when True, rows whose unique ID already exists
            in old_df are NEVER overwritten by new_df; only brand-new rows (and
            new columns) are taken from new_df.  Used for downloaded/SSE object
            merges so a stale remote row can't clobber a fresher local object.

    Returns:
        pd.DataFrame
    """
    old_df = old_df.copy()
    # If new_df has no key column there is nothing to merge in — keep old_df.
    if unique_id_column_name not in new_df.columns:
        return old_df
    # If old_df has no key column (e.g. the dedupe cache was initialised from an
    # empty object pull, so it has no columns yet), there is nothing to merge
    # INTO — the deduped new_df is the result. Guards against the KeyError in the
    # merge below when the first full pull returned an empty dataset.
    if unique_id_column_name not in old_df.columns:
        return new_df.groupby(unique_id_column_name, as_index=False).agg('last')
    # When only_add_new_rows is True, remember which IDs already exist locally so
    # downloaded values never overwrite an existing local object.
    preexisting_ids = None
    if only_add_new_rows and unique_id_column_name in old_df.columns:
        preexisting_ids = set(old_df[unique_id_column_name].dropna().astype(str))
    # Drop duplicates by aggregating on the unique_id_column_name column.
    # Keep the last (most recent) values for each unique ID.
    new_df = new_df.groupby(unique_id_column_name,as_index=False).agg('last')

    # Add any columns and rows from new_df that are not in old_df.
    if add_columns_and_rows_from_new_df_to_old_df:
        # label the columns from the new_df with suffix 'new'
        old_df = old_df.merge(new_df, how='outer', on=unique_id_column_name, suffixes=(None, '__new'))
        # The purpose of this merge is to add any columns & rows from new_df that are not in old_df.
        # This is prep for the update step below, which will only update common column names.

        # drop the ...'__new' columns in favor of older values from old_df,
        # we'll replace those later in the update step.
        droplist = []
        for col in old_df.columns:
            if '__new' in col:
                droplist.append(col)
        old_df = old_df.drop(columns=droplist)


    # Merge the two dataframes on the unique_id_column_name column.
    # Columns from the old_df will be labeled with 'old' suffix

    merged_df_with_newer_vals = old_df.merge(new_df, how='left', on=unique_id_column_name, suffixes=('_old', None))
    droplist = []
    for col in merged_df_with_newer_vals.columns:
        if 'old' in col:
            droplist.append(col)
    merged_df_with_newer_vals = merged_df_with_newer_vals.drop(columns=droplist)

    # The two dataframes, old_df and merged_df_with_newer_vals,
    # now have the same columns and rows.
    # This allows us to use the .update() method to 
    # update the old_df with values from the new_df.

    if preexisting_ids is not None:
        # Exclude rows that already existed locally so DataFrame.update() only
        # fills newly-added rows. Assigning NaN across the preexisting rows is
        # not safe for strict dtypes such as bool (pandas raises
        # "Invalid value 'nan' for dtype 'bool'").
        existing_mask = merged_df_with_newer_vals[unique_id_column_name].astype(str).isin(preexisting_ids)
        merged_df_with_newer_vals = merged_df_with_newer_vals.loc[
            ~existing_mask.to_numpy(dtype=bool, copy=False)]

    old_df.update(merged_df_with_newer_vals)

    return old_df

def get_current_timestamp_string(offset_seconds: float = 0.0) -> str:
    """
    Gets the current time in UTC with a format like the following:
       2021-03-04T15:00:00.000Z

    Note: put this into utils when possible
    Returns:
         str
    """
    # Get the current time in UTC and format it in the way dictated by the schema
    datetime_format_string = "%Y-%m-%dT%H:%M:%S.%f"
    current_time = dt.now(tz=tz.utc)
    if offset_seconds != 0.0:
        current_time = current_time + timedelta(seconds=offset_seconds)
    current_time_string = current_time.strftime(datetime_format_string)[:-3] + 'Z'
    return current_time_string


def update_timestamps(df: pd.DataFrame, updated_rows_index: List[int]) -> pd.DataFrame:
    """
    Updates timestamp column formats to match the schema (e.g. 2021-03-04T15:00:00.000Z).

    Args:
        df (pd.DataFrame):
        updated_rows_index (List[int]):

    Returns:
        pd.DataFrame
    """
    default_timestamp = '1970-01-01T00:00:00.000Z'
    # this default is replaced by None below, but it needs to
    # be a valid timestamp for the timestamp parsing library to work
    datetime_format_string = "%Y-%m-%dT%H:%M:%S.%f"

    timestamp_columns = [col for col in df.columns if "timestamp" in col.casefold()]

    # Fill missing values with a default timestamp in prep for parsing library
    df.loc[:, timestamp_columns] = df[timestamp_columns].fillna(default_timestamp)

    df_temp = df.loc[updated_rows_index, :].copy()
    # Parse timestamps using pandas' to_datetime() method
    # and format them using the strftime() method
    for col in timestamp_columns:
        df_temp[col] = pd.to_datetime(df_temp[col],
                                      utc=True,
                                      format='ISO8601',
                                      errors='coerce')
        df_temp[col] = df_temp[col].map(
            lambda x: x.strftime(datetime_format_string)[:-3] + 'Z' if pd.notna(x) else '1970-01-01T00:00:00.000Z'
        )

    df.loc[updated_rows_index, timestamp_columns] = df_temp[timestamp_columns]

    # Default value needs to be None type so that these values are overwritten
    # by the .update method.  Only the timestamp columns can ever hold the
    # sentinel (they were fillna'd with it above), so null it in place
    # per-column instead of building a full-frame boolean mask + df.where()
    # copy of the entire (up-to-81k-row) frame on every event.
    for col in timestamp_columns:
        df.loc[df[col] == default_timestamp, col] = None

    return df


def flatten_crucible_dataset(input: Any, ) -> pd.DataFrame:
    '''
    Produces a flattened dataframe.
    Note: this accepts a flattened dataframe
    as well, but will leave it unchanged.

    NOTE: put this into utils when possible
    
    Args:
        input (pd.DataFrame or JSON string):
        

    Returns:
        pd.DataFrame
    '''

    if isinstance(input, str):
        input = json.loads(input)

    # this returns a dataframe from a nested JSON string retrieved 

    df = pd.json_normalize(input)
    return df


# convert an uncertainty ellipse in WGS84 into a covariance matrix in ECEF

def uncertainty_ellipse_to_covariance_matrix(major_axis: float, minor_axis: float, azimuth: float, padding: float = 3) -> np.ndarray:

    """
    Convert uncertainty ellipse parameters to a covariance matrix.
    Parameters:
    major_axis (float): Length of the major axis of the ellipse.
    minor_axis (float): Length of the minor axis of the ellipse.
    azimuth (float): Orientation angle of the ellipse in radians.
    padding (float, optional): Value to pad the covariance matrix with. Default is 3.
    This corresponds to the uncertainty in the z-axis.
    Returns:
    numpy.ndarray: A 3x3 covariance matrix derived from the ellipse parameters.
    """
    # Ensure inputs are numeric (guards against string values from source data)
    major_axis = float(major_axis)
    minor_axis = float(minor_axis)
    azimuth = float(azimuth)

    # first, convert the major and minor axes from 95% confidence ellipse
    # to 1 sigma covariance matrix
    # 95% uncertainty axes are 2.448 times longer than 1 sigma axes
    semi_maj = major_axis/2.448
    semi_min = minor_axis/2.448
    theta = azimuth  # clockwise from North
    c = np.cos(theta)
    s = np.sin(theta)
    # In ENU frame, azimuth CW from North means major axis direction is (sinθ, cosθ) in [E, N]
    M = np.array([[semi_maj**2 * s**2 + semi_min**2 * c**2, (semi_maj**2 - semi_min**2) * s * c],
                  [(semi_maj**2 - semi_min**2) * s * c, semi_maj**2 * c**2 + semi_min**2 * s**2]])
    M = np.pad(M, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    M[-1, -1] = padding

    return M

def pandas_wgs84_vel_to_ecef_by_row(row: pd.Series) -> pd.Series:

    # only return a velocity vector if we have valid east and north speeds
    if ( 'estimatedKinematics.velocity.eastSpeed' in row  \
            and 'estimatedKinematics.velocity.northSpeed' in row  \
            and not pd.isna(row['estimatedKinematics.velocity.eastSpeed']) \
            and not pd.isna(row['estimatedKinematics.velocity.northSpeed']) ):

        # set down speed to 0 if it is NaN or missing
        if 'estimatedKinematics.velocity.downSpeed' not in row:
            row['estimatedKinematics.velocity.downSpeed'] = 0.0
        if pd.isna(row[ 'estimatedKinematics.velocity.downSpeed']):
            row['estimatedKinematics.velocity.downSpeed'] = 0.0

        v = enu_to_ecef_vector(row['estimatedKinematics.position.latitude'],
                      row['estimatedKinematics.position.longitude'],
                      row['estimatedKinematics.velocity.eastSpeed'],
                      row['estimatedKinematics.velocity.northSpeed'],
                      -row['estimatedKinematics.velocity.downSpeed'])
        return pd.Series(v, index=['ecefVelocity.dx', 'ecefVelocity.dy', 'ecefVelocity.dz'])
    else:
        return pd.Series([np.nan, np.nan, np.nan], index=['ecefVelocity.dx', 'ecefVelocity.dy', 'ecefVelocity.dz'])

def add_ECEF_kinematics(df: pd.DataFrame) -> pd.DataFrame:


    # drop rows with no kinematicsTimestamp — a position with no observation
    # time is not a usable kinematic update. Filter the same way as lat/lon and
    # warn with the number dropped. If the column is entirely absent, no row
    # carries an observation time, so drop them all rather than emit timeless
    # kinematic updates.
    if 'estimatedKinematics.kinematicsTimestamp' in df.columns:
        _n_before = len(df)
        df = df.dropna(subset=['estimatedKinematics.kinematicsTimestamp'])
        _n_dropped = _n_before - len(df)
        if _n_dropped:
            logging.warning(f"add_ECEF_kinematics: dropped {_n_dropped} row(s) for missing "
                            f"estimatedKinematics.kinematicsTimestamp")
    elif len(df) > 0:
        logging.warning(f"add_ECEF_kinematics: dropped all {len(df)} row(s) - no "
                        f"estimatedKinematics.kinematicsTimestamp column present")
        df = df.iloc[0:0]

    # drop rows with no lat, lon, alt
    df = df.dropna(subset=['estimatedKinematics.position.latitude',
                            'estimatedKinematics.position.longitude',
                            ])

    # exit if there are no rows left
    if len(df) == 0:
        return df
    
    # fill missing altitudes with 0
    if 'estimatedKinematics.position.altitude' not in df.columns:
        df['estimatedKinematics.position.altitude'] = 0.0
    df['estimatedKinematics.position.altitude']=df['estimatedKinematics.position.altitude'].fillna(0.0,)

    df['ecefPosition.x'], df['ecefPosition.y'], df['ecefPosition.z'] = ecef_transformer.transform( \
        df['estimatedKinematics.position.latitude'],
        df['estimatedKinematics.position.longitude'],
        df['estimatedKinematics.position.altitude'],
        radians=True,inplace=False )

    # --- Vectorized ECEF velocity conversion ---
    _has_east = 'estimatedKinematics.velocity.eastSpeed' in df.columns
    _has_north = 'estimatedKinematics.velocity.northSpeed' in df.columns
    if _has_east and _has_north:
        _lat = pd.to_numeric(df['estimatedKinematics.position.latitude'], errors='coerce').values
        _lon = pd.to_numeric(df['estimatedKinematics.position.longitude'], errors='coerce').values
        _clon = np.cos(_lon); _slon = np.sin(_lon)
        _clat = np.cos(_lat); _slat = np.sin(_lat)

        _v_e = pd.to_numeric(df['estimatedKinematics.velocity.eastSpeed'], errors='coerce').values
        _v_n = pd.to_numeric(df['estimatedKinematics.velocity.northSpeed'], errors='coerce').values
        if 'estimatedKinematics.velocity.downSpeed' in df.columns:
            _v_d = pd.to_numeric(df['estimatedKinematics.velocity.downSpeed'], errors='coerce').fillna(0.0).values
        else:
            _v_d = np.zeros(len(df))
        _v_u = -_v_d

        _valid_vel = np.isfinite(_v_e) & np.isfinite(_v_n)
        _dx = -_slon * _v_e + (-_slat * _clon) * _v_n + (_clat * _clon) * _v_u
        _dy =  _clon * _v_e + (-_slat * _slon) * _v_n + (_clat * _slon) * _v_u
        _dz =  _clat * _v_n + _slat * _v_u
        df['ecefVelocity.dx'] = np.where(_valid_vel, _dx, np.nan)
        df['ecefVelocity.dy'] = np.where(_valid_vel, _dy, np.nan)
        df['ecefVelocity.dz'] = np.where(_valid_vel, _dz, np.nan)
    else:
        df['ecefVelocity.dx'] = np.nan
        df['ecefVelocity.dy'] = np.nan
        df['ecefVelocity.dz'] = np.nan
    

    if 'estimatedKinematics.uncertainty.uncertaintyEllipse.semiMajorAxisLength' in df.columns:
        if 'estimatedKinematics.uncertainty.uncertaintyEllipse.semiMinorAxisLength' not in df.columns:
            df[:,'estimatedKinematics.uncertainty.uncertaintyEllipse.semiMinorAxisLength'] = 1000.0
        if 'estimatedKinematics.uncertainty.uncertaintyEllipse.orientation' not in df.columns:
            df[:,'estimatedKinematics.uncertainty.uncertaintyEllipse.orientation'] = 0.0
        df['estimatedKinematics.uncertainty.uncertaintyEllipse.semiMajorAxisLength']=\
            pd.to_numeric(df['estimatedKinematics.uncertainty.uncertaintyEllipse.semiMajorAxisLength'], errors='coerce').fillna(1000.0)
        df['estimatedKinematics.uncertainty.uncertaintyEllipse.semiMinorAxisLength']=\
            pd.to_numeric(df['estimatedKinematics.uncertainty.uncertaintyEllipse.semiMinorAxisLength'], errors='coerce').fillna(1000.0)
        df['estimatedKinematics.uncertainty.uncertaintyEllipse.orientation']=\
            pd.to_numeric(df['estimatedKinematics.uncertainty.uncertaintyEllipse.orientation'], errors='coerce').fillna(0.0)
        # --- Vectorized covariance matrix computation ---
        # Replaces 3 row-by-row .apply() passes with a single np.einsum call.
        _major = pd.to_numeric(df['estimatedKinematics.uncertainty.uncertaintyEllipse.semiMajorAxisLength'], errors='coerce').values / 2.448
        _minor = pd.to_numeric(df['estimatedKinematics.uncertainty.uncertaintyEllipse.semiMinorAxisLength'], errors='coerce').values / 2.448
        _theta = pd.to_numeric(df['estimatedKinematics.uncertainty.uncertaintyEllipse.orientation'], errors='coerce').values
        _ct = np.cos(_theta); _st = np.sin(_theta)
        _padding = 1.e4  # uncertainty in z-axis of 100m

        _N = len(df)
        # ENU covariance matrices (N, 3, 3)
        _C = np.zeros((_N, 3, 3))
        _C[:, 0, 0] = _major**2 * _st**2 + _minor**2 * _ct**2
        _C[:, 0, 1] = (_major**2 - _minor**2) * _st * _ct
        _C[:, 1, 0] = _C[:, 0, 1]
        _C[:, 1, 1] = _major**2 * _ct**2 + _minor**2 * _st**2
        _C[:, 2, 2] = _padding

        # ENU→ECEF rotation matrices (N, 3, 3) — reuse lat/lon trig if available
        _cov_lat = pd.to_numeric(df['estimatedKinematics.position.latitude'], errors='coerce').values
        _cov_lon = pd.to_numeric(df['estimatedKinematics.position.longitude'], errors='coerce').values
        _cov_clon = np.cos(_cov_lon); _cov_slon = np.sin(_cov_lon)
        _cov_clat = np.cos(_cov_lat); _cov_slat = np.sin(_cov_lat)
        _R = np.zeros((_N, 3, 3))
        _R[:, 0, 0] = -_cov_slon
        _R[:, 0, 1] = -_cov_slat * _cov_clon
        _R[:, 0, 2] =  _cov_clat * _cov_clon
        _R[:, 1, 0] =  _cov_clon
        _R[:, 1, 1] = -_cov_slat * _cov_slon
        _R[:, 1, 2] =  _cov_clat * _cov_slon
        _R[:, 2, 1] =  _cov_clat
        _R[:, 2, 2] =  _cov_slat

        # Rotated covariance: R @ C @ R^T  for all rows at once
        _RCov = np.einsum('nij,njk,nlk->nil', _R, _C, _R)

        df['positionCovariance.xx'] = _RCov[:, 0, 0]
        df['positionCovariance.xy'] = _RCov[:, 0, 1]
        df['positionCovariance.xz'] = _RCov[:, 0, 2]
        df['positionCovariance.yy'] = _RCov[:, 1, 1]
        df['positionCovariance.yz'] = _RCov[:, 1, 2]
        df['positionCovariance.zz'] = _RCov[:, 2, 2]
    
        

    return df
    
# end of ECEF conversion functions

def safe_str(x: Any) -> str:
    """
    Safely convert values to strings, handling Int64 NA values and floats.
    Returns empty string for NA/None values, and converts whole-number floats to int strings.
    Also normalises numeric-looking strings (e.g. "123.0" → "123") so that
    custom IDs stay consistent regardless of how the value was stored.
    
    Args:
        x: Value to convert to string
    
    Returns:
        str: String representation of the value
    """
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

def create_custom_id(row: pd.Series) -> str:
    """
    Create a custom ID by concatenating column names and values.
    
    Args:
        row: A pandas Series (row from a DataFrame)
    
    Returns:
        str: A concatenated string of column names and values
    """
    # Sort by column name to ensure consistent ordering regardless of input column order
    sorted_items = sorted(zip(row.index, row.values))
    return '_'.join(f"{safe_str(k)}-{safe_str(v)}" for k, v in sorted_items)
    
def add_blank_rows(df: pd.DataFrame, number_of_rows: int) -> pd.DataFrame:
    """
    Adds blank rows to the end of a pandas DatFrame.

    Args:
        df (pd.DataFrame):
        number_of_rows (int):

    Returns:
        pd.DataFrame
    """

    df = df.reset_index(drop=True)
    df = df.reindex(
        df.index.values.tolist() +
        list(range(len(df), len(df) +
                   number_of_rows)))
    return df

def add_kinematics_columns(object_df: pd.DataFrame) -> pd.DataFrame:
     # if needed, add kinematics columns to object_df
    col_names=object_df.columns
    reqd_cols = ['estimatedKinematics.velocity.downSpeed',
                 'estimatedKinematics.velocity.eastSpeed',
                 'estimatedKinematics.velocity.northSpeed',
                 'estimatedKinematics.position.latitude',
                 'estimatedKinematics.position.longitude',
                 'estimatedKinematics.position.altitude',]  
    for col in reqd_cols:
        if col not in col_names:
            object_df[col]=np.nan  

    return object_df     


def create_source(row: pd.Series, source_mapping: List[Dict[str, str]],
                  query: str,
                  timestamp: str) -> List[Dict[str, Any]]:
    """
    Create a dictionary of source information.

    Args:
        row (pd.Series):
        source_mapping (List[Dict[str, str]):
        query (str):
        timestamp (str):

    Returns:
         List[Dict[str, str]]
    """
    # create dictionary of source info:
    source_dict = {}

    for j in range(len(source_mapping)):
        try:
            source_dict[source_mapping[j]['destination_column']] = row[source_mapping[j]['origin_column']]
        except:
            # if 'origin_column' doesn't exist in row, treat it as a hard-coded value:
            source_dict[source_mapping[j]['destination_column']] = source_mapping[j]['origin_column']
    # add event dataset name,config file name to source_dict

    source_dict['datasetName'] = get_dataset_name(query)
    source_dict['sourceLastUpdatedTimestamp'] = timestamp

    return [source_dict]


def get_dataset_name(query: str) -> str:
    """
    Retrieves the name of a dataset using regex matching.

    Args:
        query (str):

    Returns:
        str
    """
    regex = r'(?<=from\s)(?:\'|")?([^\s\'"]+)(?:\'|")?'
    match = re.search(regex, query, flags=re.IGNORECASE)

    if match:
        dataset_name = match.group(1)
        return dataset_name
    else:
        return "dataset_name_unknown"

def _retrieve_remote_object_uuids(object_dataset: str) -> Optional[set]:
    """
    Returns the set of object UUIDs that currently exist in the remote
    object dataset.  Used by the reconcile fan-out to detect locally-cached
    objects that have been deleted externally (without a corresponding DELETE
    record in the object management event dataset).

    Returns None on error so the caller can distinguish a transient failure
    from "the dataset is genuinely empty" (which returns an empty set).
    """
    rc.token = auth.get_token()
    # ID-only query keeps this cheap even for large datasets.
    # NOTE: select the leaf scalar field (objectId.uuid), NOT the whole
    # objectId/crucibleHeader structs.  Selecting whole ROW/struct columns
    # makes Crucible's ResultSetToRecords converter throw
    # "Failed to convert ResultSet with relational data type [ROW-->root...]".
    # A leaf-field select is returned flattened (column label 'uuid').
    query = ("select " + object_dataset + ".objectId.uuid from " + object_dataset +
             " limit 200000")
    try:
        df = rc.search(query, format='dataframe',)
    except Exception as err:
        logging.warning(f"_retrieve_remote_object_uuids: {err=}, {type(err)=}")
        return None

    if df is None or len(df) == 0:
        return set()

    # Leaf-field selects come back with the trailing segment as the column
    # label ('uuid'); fall back to the dotted/struct names just in case.
    for col in ('uuid', 'objectId.uuid', 'objectId'):
        if col in df.columns:
            return set(df[col].dropna().astype(str).tolist())
    return set()


def retrieve_objects(dataset_name: str, limit_to_ID_fields: bool = False,
                     since_seconds: Optional[int] = None) -> pd.DataFrame:
    """
    Retrieves objects from a Crucible dataset.

    Args:
        dataset_name (str): dataset to query
        limit_to_ID_fields (bool): if True, only fetch ID columns
        since_seconds (int): if set, only return rows whose
            crucibleHeader.updatedDate is within the last N seconds
            (uses TIMESTAMP_OFFSET in the WHERE clause)

    Returns:
        pd.DataFrame
    """
    rc.token = auth.get_token()

    if limit_to_ID_fields:
        select_clause = 'select crucibleHeader, objectId, `identity` from ' + dataset_name
    else:
        select_clause = 'select * from ' + dataset_name

    if since_seconds is not None and since_seconds > 0:
        where_clause = (" where " + dataset_name +
                        ".crucibleHeader.updatedDate > TIMESTAMP_OFFSET(-" +
                        str(int(since_seconds)) + ", 'seconds')")
    else:
        where_clause = ''

    query = select_clause + where_clause + ' limit ' + str(_ROW_LIMIT)

    for i in range(3):
        try:
            object_df = rc.search(query, format='dataframe',)
            return object_df
        except Exception as err:
            logging.error(f"{err=}, {type(err)=}")
            if i<2: 
                time.sleep(2)
            else:
                raise Exception("Error with attempting to retrieve data from Crucible")

    return object_df


def retrieve_object_records(dataset_name: str) -> List[dict]:
    """Retrieve full object records as the native rc.search list-of-dicts output.

    Used by object-cache refresh fan-out to avoid materializing a DataFrame only
    to convert it back to ``records`` again.
    """
    rc.token = auth.get_token()
    query = 'select * from ' + dataset_name + ' limit ' + str(_ROW_LIMIT)
    for i in range(3):
        try:
            records = rc.search(query) or []
            return records if isinstance(records, list) else []
        except Exception as err:
            logging.error(f"{err=}, {type(err)=}")
            if i < 2:
                time.sleep(2)
            else:
                raise Exception("Error with attempting to retrieve data from Crucible")
    return []



def create_object_from_template( ) -> object:
    """
    Create an object DatFrame from a model template

    note: put this into utils when possible

    Returns:
        pd.DataFrame
    """

    object_dict_template = \
    {
        "edhControlSet": [
            "CLS:U"
        ],
        "identity": {
            "dynamicIdentifier": "",
            "environment": {
                "confidence": 100,
                "environment": "AIR"
            },
            "standard": {
                "allegiance": "",
                "confidence": 100,
                "standardIdentity": "NEUTRAL"
            }
        },
        "mobility": "MOVER",
        "mode": "LIVE",
        "objectId": {
            "descriptiveLabel": "",
            "uuid": ""
        },
        "entityStatus": "UNKNOWN"
    }
    new_object_df = pd.json_normalize([object_dict_template])
    return new_object_df



def fix_dtypes(df: pd.DataFrame) -> None:
    """
    NOW DEPRECATED: this function is no longer used.
    Converts DatFrame columns of types int64, Int64, or bool to type object for compatibility 
    with JSON conversion.

    note: put this into utils when possible

    Args:
        df (pd.DataFrame):
    """
    dtype_list = df.dtypes
    for col in dtype_list.index:
        if dtype_list[col] == 'int64' or dtype_list[col] == 'Int64' or dtype_list[col] == 'bool':
            # df[col]=df[col].astype(float)
            # df[col]=df[col].astype(int)
            df[col] = df[col].astype(object)
    # nothing is returned from this function, which performs operations 
    # on dataframes in place


def returnDescriptiveLabel(row: pd.Series) -> str:
    # if enviroment is SEA_SURFACE or SEA_SUBSURFACE and vesselName is not blank then return vesselName
    if row['identity.environment.environment'] in ['SEA_SURFACE' ,'SEA_SUBSURFACE']:
        # if blue we should have vesselName
        if row['identity.standard.standardIdentity'] in ['FRIEND','ASSUMED_FRIEND'] and \
           'identity.vesselName'in row and row['identity.vesselName'] and \
           row['identity.vesselName']==row['identity.vesselName']:
         
               return row['identity.vesselName']
        # Concate TypeCode and Pennant if exist
        if  'identity.vesselTypeCode' in row and 'identity.navalPennant' in row: 
            if pd.notna(row['identity.vesselTypeCode']) and pd.notna(row['identity.navalPennant']):
                return( row['identity.vesselTypeCode'] + '-' + str(int(row['identity.navalPennant'])) )
        # if vesselClass exists and not blank
        if 'identity.vesselClass' in row and pd.notna(row['identity.vesselClass']) and str(row['identity.vesselClass']).strip():
            return row['identity.vesselClass']
        if 'identity.vesselTypeCode' in row and row['identity.vesselTypeCode'] and row['identity.vesselTypeCode']==row['identity.vesselTypeCode']:
 
            return row['identity.vesselTypeCode']
        if 'identity.vesselType' in row and row['identity.vesselType'] and row['identity.vesselType']==row['identity.vesselType']:
        
            return row['identity.vesselType']
        if 'identity.hullNumber' in row and row['identity.hullNumber'] and row['identity.hullNumber']==row['identity.hullNumber']:

            return row['identity.hullNumber']
        else:
    
            return row['identity.environment.environment']
    elif row['identity.environment.environment'] == 'AIR':
        # if callsign exists and not blank
        if 'identity.callsign' in row and row['identity.callsign'] and row['identity.callsign']==row['identity.callsign']:

            return row['identity.callsign']
        if 'identity.airPlatformType' in row and row['identity.airPlatformType'] and row['identity.airPlatformType']==row['identity.airPlatformType']:
        
            return row['identity.airPlatformType']
        if 'identity.airframeType' in row and row['identity.airframeType'] and row['identity.airframeType']==row['identity.airframeType']:
         
            return row['identity.airframeType']
        else:
  
            return row['identity.environment.environment']
    else:
        if 'identity.description' in row and row['identity.description'] and row['identity.description']==row['identity.description']:
      
            return row['identity.description']
        if 'identity.callsign' in row and row['identity.callsign'] and row['identity.callsign']==row['identity.callsign']:

            return row['identity.callsign']
        if 'identity.hullNumber' in row and row['identity.hullNumber'] and row['identity.hullNumber']==row['identity.hullNumber']:

            return row['identity.hullNumber']
        else:

            return row['identity.environment.environment']


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

    # Configure logging BEFORE instantiating controllers so they pick up the correct level
    _, _, _, log_utils, _, _ = _import_cruciblelib_modules()
    log_utils.get_logger(log_type='transformer', log_level=numeric_level)
    
    # Now instantiate the API controllers (they will use the configured log level)
    global rc, wc, auth
    auth, rc, wc = instantiate_api_controllers()

    # get config dictionaries from Crucible stream manager datasets:

    config_list \
        = find_and_validate_configs(stream_manager_perspective, include_scripts=True, rc_instance=rc, caller_globals=globals())

    # run main program and retry if there is a connection failure
    while True:
        try:
            event_loop_manager(config_list)
            # note: exceptions thrown in event_loop_manager and functions
            # called within are caught here and the program will retry.
            # All tasks are cancelled after exiting asyncio.gather()
            # in event_loop_manager.
        except (KeyboardInterrupt, SystemExit):
            logging.info("Transformer: Received shutdown signal, exiting...")
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

    # Ensure Ctrl-C / kill kills the entire process group (forkserver + all workers)
    def _sigterm_handler(signum: int, frame: Any) -> None:
        logging.info(f"Transformer: Received signal {signum}, killing process group")
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    log_level = args.log
    stream_manager_perspective = args.stream_manager_perspective

    run(stream_manager_perspective, log_level=log_level)


