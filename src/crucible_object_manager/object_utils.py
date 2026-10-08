"""
Shared utility functions for the object_manager package.

This module contains functions that are imported across multiple scripts
in the object_manager directory.

Crucible write-API behavior (VERIFIED against the crucible-prototype server; the
correlators use the v1 WriteController, which is the default):
  * update_entity_record_batch_by_name = PUT. Returns r.json() = a LIST of ONLY
    the records that were successfully updated. A record whose primary key does
    NOT exist is neither created nor errored: the server raises
    ResourceNotFoundException internally (UpdateEntityRecordLoader), buckets the
    record into failedRecordBatch, and OMITS it from the response
    (FluoWriteService returns UPDATED_RECORDS_KEY when includeFailedRecords is
    false). So a returned list SHORTER than what was sent is the signal that
    some records did not exist.
  * write_record_batch_by_name / upsert_by_name = POST. Return an int status
    code (e.g. 204), NOT a list. write_* creates; upsert_* is create-or-update.
  * _extract_failed_records() relies on this: for a LIST response (PUT update)
    it diffs the ids of sent vs returned records and reports the missing ones
    as deleted_ids; for a non-list response (POST) it reports nothing dropped.
    write_batch_chunked() therefore returns
    (failed_records = calls that raised, deleted_ids = records the server
    silently omitted because they don't exist / were deleted).
  * WARNING: this silent-omission behavior is v1-specific. The v2 PUT with the
    default includeFailedRecords=false returns an EMPTY list regardless of
    success, which would make _extract_failed_records flag EVERY record as
    missing. Do NOT switch these correlators to the v2 WriteController without
    revisiting this logic.
"""

import os
import sys
import time
import json
import types
import random
import logging
import asyncio
import traceback
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Tuple, Optional, Set, Callable
from filterpy.stats import covariance_ellipse


# --- Constants ---

int_cols = ['identity.navalPennant', 'identity.mmsiNumber', 'trackQuality']

# Global dict holding script source code so child processes can re-create modules.
_script_sources: Dict[str, str] = {}


def normalize_uuid(value: Any) -> str:
    """Remove UUID separators for Crucible's 32-hex schema format."""
    return str(value).strip().replace('-', '')


def _fast_df_to_nested_json(df: pd.DataFrame) -> List[dict]:
    """Fast replacement for utils.df_to_formatted_JSON.

    Converts a flattened DataFrame (columns like 'a.b.c') back into a list
    of nested dicts.  Mirrors df_to_formatted_JSON semantics: drops
    None / NaN / NaT / pd.NA values and empty strings so they don't bloat
    the JSON payload, and converts numpy scalar types (np.integer,
    np.bool_) to native Python types so the result is JSON-serializable.
    Uses df.to_dict('records') + dict key splitting instead of per-row
    Python iteration over every column, which is much faster for wide
    frames.
    """
    # Drop columns that are entirely None/NaN — saves work downstream
    df = df.dropna(axis=1, how='all')

    records = df.to_dict('records')
    result: List[dict] = []
    for flat in records:
        nested: dict = {}
        for key, val in flat.items():
            # Skip None / NaN / NaT / pd.NA values and empty strings, and
            # normalise numpy scalar types.  Containers (lists/dicts/arrays)
            # are passed through unchanged.
            if val is None:
                continue
            if not isinstance(val, (list, tuple, dict, np.ndarray)):
                try:
                    if pd.isna(val):
                        continue
                except (ValueError, TypeError):
                    pass
                if val == '':
                    continue
                if isinstance(val, np.integer):
                    val = int(val)
                elif isinstance(val, np.bool_):
                    val = bool(val)
            parts = key.split('.')
            d = nested
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val
        result.append(nested)
    return result


# --- Cruciblelib lazy imports ---

def _import_cruciblelib_modules() -> tuple:
    """Lazy-import cruciblelib controller and utility modules."""
    try:
        from cruciblelib import read_controller, write_controller, utils, log_utils, authenticator
        from cruciblelib.exceptions import DetailedHTTPError
    except ImportError:
        from . import read_controller, write_controller, utils, log_utils, authenticator
        from .exceptions import DetailedHTTPError
    return read_controller, write_controller, utils, log_utils, authenticator, DetailedHTTPError


# --- Resilient SSE listener ---

async def connect_with_timeout(url: str, session: Any, headers: dict, timeout: float) -> Any:
    """Open an SSE EventSource with a connect timeout; return None on failure."""
    from aiohttp_sse_client import client as sse_client
    try:
        return await asyncio.wait_for(
            sse_client.EventSource(url, session=session, headers=headers, timeout=None).__aenter__(),
            timeout,
        )
    except asyncio.TimeoutError:
        logging.error(f"Timeout error occurred: Initial connection to {url} could not be established within {timeout} seconds.")
        return None
    except Exception as e:
        logging.error(f"Connection error for {url}: {e}")
        return None


async def put_to_queue(queue: Any, data: Any) -> None:
    """Put data on either an asyncio.Queue or a multiprocessing Queue."""
    if hasattr(queue, 'put_nowait'):
        try:
            if asyncio.iscoroutinefunction(queue.put):
                await queue.put(data)
            else:
                queue.put(data)
        except Exception:
            queue.put(data)
    else:
        queue.put(data)


async def run_sse_listener(sql_query: str, auth: Any, on_event: Callable[[Any], Any],
                           *, label: Optional[str] = None, ssl_verify: Optional[bool] = None) -> None:
    """Resilient SSE read loop shared by every object-side listener.

    Connects to the Crucible SSE endpoint for ``sql_query`` and invokes
    ``await on_event(event)`` for each RAW sse_client event (which has ``.data``);
    callers parse/route the event however they need.  Never returns — it
    reconnects forever with incremental backoff.  Hardened so a large or stalled
    SSE payload can neither kill the listener (aiohttp ``LineTooLong``) nor hang
    it silently forever:
      * the read buffer is large and configurable, so a big snapshot no longer
        raises ``LineTooLong`` and drops the listener, and
      * a read watchdog forces a reconnect if events stop flowing, so a stalled
        read can't freeze the stream with no recovery.

    Environment overrides:
      CRUCIBLE_SSE_READ_BUFSIZE  bytes, default 256 MB (raise if snapshots grow)
      CRUCIBLE_SSE_READ_TIMEOUT  seconds, default 180; 0 disables the watchdog
    """
    import aiohttp
    from aiohttp.http_exceptions import LineTooLong
    _, _, utils, _, _, _ = _import_cruciblelib_modules()

    tag = label if label is not None else sql_query
    MAX_BACKOFF = 60
    PERSISTENT_FAILURE_THRESHOLD = 10
    # Object/event snapshots grow as the number of live objects climbs, so a
    # single SSE payload can be large; a payload bigger than this raises aiohttp
    # LineTooLong and kills the listener, so make it large and configurable.
    read_bufsize = int(os.getenv('CRUCIBLE_SSE_READ_BUFSIZE', str(256 * 1024 * 1024)))
    # Watchdog: if no event (or keepalive) arrives within this many seconds,
    # force a reconnect instead of hanging silently forever on a stalled or
    # oversized read.  0 disables it.
    read_timeout = float(os.getenv('CRUCIBLE_SSE_READ_TIMEOUT', '180'))
    if ssl_verify is None:
        ssl_verify = os.getenv('CRUCIBLE_SSL_VERIFY', 'false').lower() not in ('false', '0', 'no')

    protocol_string = 'https://'
    sse_url = os.getenv('CRUCIBLE_SSE_URL', protocol_string + os.environ['CRUCIBLE_SERVICES_HOST'] + '/api/v1/read/search/sse')
    url = sse_url + '?query=' + sql_query
    headers = {'Content-Type': 'text/plain'}
    logging.info(f"[{tag}]: Connecting to {url}")
    # Track consecutive failures to drive the incremental-backoff strategy.
    attempt = 0
    while True:
        try:
            # Refresh access token before each connection attempt.
            await utils.refresh_access_token(auth, headers)
            # Configurable read buffer (default 256 MB) to support large snapshots.
            async with aiohttp.ClientSession(read_bufsize=read_bufsize, connector=aiohttp.TCPConnector(ssl=ssl_verify)) as session:
                event_source = await connect_with_timeout(url, session, headers, 10)
                if event_source:
                    try:
                        # Iterate manually so we can apply a read watchdog: a
                        # stalled or oversized read must not hang the listener
                        # forever with no recovery.
                        event_iter = event_source.__aiter__()
                        while True:
                            try:
                                if read_timeout > 0:
                                    event = await asyncio.wait_for(event_iter.__anext__(), timeout=read_timeout)
                                else:
                                    event = await event_iter.__anext__()
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                logging.warning(
                                    f"[{tag}]: No SSE event received in {read_timeout:.0f}s; "
                                    f"forcing reconnect (read watchdog)."
                                )
                                attempt += 1
                                break
                            attempt = 0
                            await on_event(event)
                    except LineTooLong as e:
                        logging.error(
                            f"[{tag}]: SSE payload exceeded the {read_bufsize}-byte read buffer: {e}. "
                            f"Raise CRUCIBLE_SSE_READ_BUFSIZE if this persists. Reconnecting with backoff."
                        )
                        attempt += 1
                    except aiohttp.ClientResponseError as e:
                        if getattr(e, 'status', None) == 401:
                            logging.error(f"[{tag}]: Received 401 status despite refreshing token, retrying...")
                        else:
                            logging.error(f"[{tag}]: Response error occurred: {e}")
                            logging.error(traceback.format_exc())
                        attempt += 1
                    except aiohttp.ClientConnectionError as e:
                        logging.error(f"[{tag}]: Connection error occurred: {e}")
                        logging.error(traceback.format_exc())
                        attempt += 1
                    except aiohttp.ClientPayloadError as e:
                        if '400' in str(e):
                            logging.warning(f"[{tag}]: Server returned 400 \u2014 dataset may be empty or column not yet available. Will retry with backoff.")
                        else:
                            logging.error(f"[{tag}]: Payload error occurred: {e}")
                            logging.error(traceback.format_exc())
                        attempt += 1
                    except Exception as e:
                        logging.error(f"[{tag}]: An unexpected error occurred: {e}")
                        logging.error(traceback.format_exc())
                        attempt += 1
                    finally:
                        # Ensure connections are closed properly.
                        try:
                            await event_source.__aexit__(None, None, None)
                        except Exception as cleanup_error:
                            logging.warning(f"[{tag}]: Error during connection cleanup: {cleanup_error}")
                else:
                    attempt += 1

                # Apply backoff after any error.
                if attempt > 0:
                    if attempt >= PERSISTENT_FAILURE_THRESHOLD and attempt % PERSISTENT_FAILURE_THRESHOLD == 0:
                        logging.critical(f"[{tag}]: {attempt} consecutive failed connection attempts. Check service health.")
                    backoff = min(2 ** attempt, MAX_BACKOFF) + random.uniform(0, 1)
                    logging.info(f"[{tag}]: Reconnecting in {backoff:.1f}s (attempt {attempt})")
                    await asyncio.sleep(backoff)
        except Exception as e:
            logging.error(f"[{tag}]: An error occurred while creating the session: {e}")
            logging.error(traceback.format_exc())
            attempt += 1
            if attempt >= PERSISTENT_FAILURE_THRESHOLD and attempt % PERSISTENT_FAILURE_THRESHOLD == 0:
                logging.critical(f"[{tag}]: {attempt} consecutive failed connection attempts. Check service health.")
            backoff = min(2 ** attempt, MAX_BACKOFF) + random.uniform(0, 1)
            logging.info(f"[{tag}]: Reconnecting in {backoff:.1f}s (attempt {attempt})")
            await asyncio.sleep(backoff)


async def SSE_listener(sql_query: str, message_queue: Any, auth: Any) -> None:
    """Resilient drop-in replacement for cruciblelib.utils.SSE_listener.

    Streams Crucible SSE results for ``sql_query`` and puts each parsed event
    (``json.loads(event.data)``) onto ``message_queue`` (skipping empty
    payloads).  Delegates the resilient connection/read/watchdog/backoff loop to
    ``run_sse_listener``.
    """
    async def _on_event(event: Any) -> None:
        try:
            event_data = json.loads(event.data)
        except json.JSONDecodeError:
            logging.warning(f"[{sql_query}]: Skipping malformed event: {str(event.data)[:200]}")
            return
        if len(event_data) > 0:
            await put_to_queue(message_queue, event_data)

    await run_sse_listener(sql_query, auth, _on_event)


# --- Error helpers ---

def is_timeout_error(err: Exception) -> bool:
    """Return True if the exception represents a network timeout rather than a data/validation error."""
    # DetailedHTTPError with status_code=0 means no HTTP response was received (transport-level error)
    status = getattr(err, 'status_code', None)
    if status == 0:
        return True
    # 504 Gateway Time-out is a transient upstream timeout, not a data/validation error
    if status == 504:
        return True
    # Check the string representation for common timeout signatures
    err_str = str(err).lower()
    for pattern in ('timed out', 'timeout', 'read timeout', 'connect timeout', 'time-out'):
        if pattern in err_str:
            return True
    # Check the wrapped original exception type name
    original = getattr(err, 'original_exception', None)
    if original is not None:
        type_name = type(original).__name__.lower()
        if 'timeout' in type_name:
            return True
    return False


# --- API controller instantiation ---

def instantiate_api_controllers() -> tuple:
    """Create and return (auth, rc, wc). Caller is responsible for
    assigning to its own module globals."""
    print(f"[CHILD PID={os.getpid()}] instantiate: importing modules...", flush=True, file=sys.stderr)
    read_controller, write_controller, _, _, authenticator, _ = _import_cruciblelib_modules()
    print(f"[CHILD PID={os.getpid()}] instantiate: modules imported. Creating Authenticator...", flush=True, file=sys.stderr)
    _auth = authenticator.Authenticator()
    print(f"[CHILD PID={os.getpid()}] instantiate: Authenticator created. Creating ReadController...", flush=True, file=sys.stderr)
    _rc = read_controller.ReadController()
    _rc.timeout = (30, 120)
    print(f"[CHILD PID={os.getpid()}] instantiate: ReadController created. Creating WriteController...", flush=True, file=sys.stderr)
    _wc = write_controller.WriteController()
    _wc.set_timeouts(post=(30, 120), put=(30, 120))
    print(f"[CHILD PID={os.getpid()}] instantiate: ALL DONE", flush=True, file=sys.stderr)
    return _auth, _rc, _wc


# --- Asyncio helpers ---

async def coroutine_launcher(coroutine_list: List[Any]) -> None:
    """
    Given a list of asyncio coroutines, launches a list
    of necessary processing tasks

    Args:
        coroutine_list (List[Any]): a list of asyncio coroutines
    """
    task_list = []
    for corout in coroutine_list:
        task_list.append(corout)
    # Capture each coroutine's name so a crash can be attributed by function
    # (e.g. object_cache_sse_listener) rather than an opaque index.
    coro_names = [getattr(c, '__qualname__', getattr(c, '__name__', repr(c)))
                  for c in coroutine_list]
    # note: nothing is returned from these tasks

    results = await asyncio.gather(*task_list, return_exceptions=True)
    # asyncio.gather(return_exceptions=True) SWALLOWS exceptions — a coroutine
    # that dies (e.g. an object_cache SSE listener) otherwise vanishes with no
    # trace.  Surface any exception so it appears in the logs.
    for _idx, _res in enumerate(results):
        if isinstance(_res, BaseException):
            _name = coro_names[_idx] if _idx < len(coro_names) else f"#{_idx}"
            logging.error(f"coroutine_launcher: coroutine '{_name}' (#{_idx}) exited with "
                          f"{type(_res).__name__}: {_res}")
            # HTTP/operational errors carry a status_code and are already logged
            # with their URL + status by the controller, so the abbreviated
            # DetailedHTTPError frames add only noise. Emit the full traceback
            # only for genuinely unexpected crashes; keep it at DEBUG otherwise.
            if getattr(_res, 'status_code', None) is not None:
                logging.debug("".join(traceback.format_exception(
                    type(_res), _res, _res.__traceback__)))
            else:
                logging.error("".join(traceback.format_exception(
                    type(_res), _res, _res.__traceback__)))


def _get_object_id(record: dict) -> Optional[str]:
    """Extract the objectId UUID string from a record dict."""
    obj_id = record.get('objectId')
    if isinstance(obj_id, dict):
        return obj_id.get('uuid')
    return record.get('objectId.uuid', obj_id)


def drop_records_missing_object_id(json_list: List[dict], label: str = '') -> List[dict]:
    """Drop records with no usable objectId.uuid before writing.

    A record whose objectId.uuid is missing/blank — e.g. because its
    unique-ID column arrived multi-valued (a list) or empty and could not be
    resolved to a scalar — is rejected by the server with a confusing
    "required property 'uuid' not found" 400 and then retried on every cycle.
    Skip these locally and log why, so one malformed feed record can't stall
    the whole batch.
    """
    if not json_list:
        return json_list
    kept: List[dict] = []
    dropped = 0
    for rec in json_list:
        uuid_val = _get_object_id(rec)
        if uuid_val is None or (isinstance(uuid_val, str) and uuid_val.strip() == ''):
            dropped += 1
            continue
        kept.append(rec)
    if dropped:
        logging.warning(
            f"{label}Dropped {dropped} record(s) with no objectId.uuid "
            f"(unresolved/multi-valued unique ID) before write"
        )
    return kept


def _extract_failed_records(result: Any, sent_chunk: List[dict], chunk_start: int, label: str, dataset_name: str, _logging: Any) -> Tuple[List[dict], Set[str]]:
    """Detect records that were silently dropped by the server.

    The update API returns only the successfully written records in its response.
    Records that could not be updated (e.g. already deleted by
    remove_superseded_objects) are omitted from the response without error.

    We detect dropped records by comparing objectIds sent vs objectIds returned.

    Args:
        result: The response from update_entity_record_batch_by_name (list of dicts).
        sent_chunk: The original list of records that were sent in this chunk.
        chunk_start: The offset index of this chunk within the full batch.
        label: Log prefix label.
        dataset_name: Target dataset name.
        _logging: Logger module.

    Returns:
        tuple: (dropped_records: list[dict], deleted_ids: set[str])
            - dropped_records: original records the server silently rejected
            - deleted_ids: objectId UUIDs that were dropped (likely already deleted)
    """
    if not result or not isinstance(result, list):
        # Non-list response (e.g. int status code from POST write) — can't compare
        return [], set()

    if len(result) >= len(sent_chunk):
        # All records returned — no drops
        return [], set()

    # Build set of objectIds that came back in the response
    returned_ids = set()
    for r in result:
        if isinstance(r, dict):
            rid = _get_object_id(r)
            if rid:
                returned_ids.add(rid)

    # Identify which sent records were dropped
    dropped_records = []
    deleted_ids = set()
    for rec in sent_chunk:
        obj_id = _get_object_id(rec)
        if obj_id and obj_id not in returned_ids:
            dropped_records.append(rec)
            deleted_ids.add(obj_id)

    if deleted_ids:
        _logging.warning(
            f"{label}Chunk at offset {chunk_start}: {len(deleted_ids)} of {len(sent_chunk)} "
            f"record(s) silently dropped by {dataset_name} (likely already deleted): "
            f"{list(deleted_ids)[:5]}{'...' if len(deleted_ids) > 5 else ''}"
        )

    return dropped_records, deleted_ids


async def write_batch_chunked(
    json_list: List[dict],
    dataset_name: str,
    write_func: Callable,
    chunk_size: int,
    *,
    label: str = '',
    token_refresher: Optional[Callable] = None,
    max_concurrent_writes: int = None,
    transient_retry_attempts: int = 0,
    retry_backoff_seconds: float = 0.25,
) -> tuple:
    """
    Write a list of JSON records to a Crucible dataset in concurrent chunks.

    Splits *json_list* into chunks of *chunk_size*, fires all chunks
    concurrently via asyncio.gather, and falls back to sub-chunk writes
    for any chunk that fails (except timeouts/auth errors, which skip
    the sub-chunk fallback and return the entire chunk as failed).

    Returns:
        tuple: (failed_records, deleted_ids)
            - failed_records: list of records that failed to write (timeout, auth,
              validation, server error). Empty on full success.
            - deleted_ids: set of objectId UUIDs that were already removed from the
              dataset (superseded/deleted race condition). Callers should update
              their supersede_map with these.

    Args:
        json_list: Records to write.
        dataset_name: Target dataset name.
        write_func: Callable (e.g. wc.write_record_batch_by_name or
                    wc.update_entity_record_batch_by_name).
        chunk_size: Max records per chunk.
        label: Optional label for log messages.
        token_refresher: Optional callable that refreshes the auth token
                         before each write (e.g. lambda: setattr(wc, 'token', auth.get_token())).
        max_concurrent_writes: Optional limit on how many chunks can be
                               in-flight simultaneously. None means no limit
                               (all chunks fire at once). Use to reduce server
                               pressure when many shards write concurrently.
        transient_retry_attempts: Additional retries for transient transport or
                     server errors. Enable only for idempotent writes.
        retry_backoff_seconds: Base delay for exponential retry backoff.
    """
    if not json_list:
        return [], set()

    import logging as _logging

    total = len(json_list)
    chunks = [json_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
    _logging.info(f"{label}Writing {total} records to {dataset_name} in {len(chunks)} chunks of {chunk_size}"
                  f"{f' (max {max_concurrent_writes} concurrent)' if max_concurrent_writes else ' (concurrent)'}")

    semaphore = asyncio.Semaphore(max_concurrent_writes) if max_concurrent_writes else None
    all_deleted_ids = set()

    def _is_retryable_error(err: Exception) -> bool:
        if is_timeout_error(err) or isinstance(
                err, (ConnectionError, ConnectionResetError, BrokenPipeError)):
            return True
        status_code = getattr(err, 'status_code', getattr(err, 'status', None))
        return (status_code in (408, 425, 429)
                or isinstance(status_code, int) and 500 <= status_code <= 599)

    async def _write_chunk(
            chunk: List[dict], chunk_start: int) -> Tuple[List[dict], List[dict]]:
        """Return failed records and the subset failed by transient errors."""
        try:
            if token_refresher:
                token_refresher()
            result = await asyncio.to_thread(write_func, dataset_name, chunk)
            failed, deleted = _extract_failed_records(result, chunk, chunk_start, label, dataset_name, _logging)
            all_deleted_ids.update(deleted)
            return failed, []
        except Exception as chunk_err:
            if is_timeout_error(chunk_err):
                # Likely a stale keep-alive connection — retry once with a
                # fresh connection before giving up.
                try:
                    if token_refresher:
                        token_refresher()
                    result = await asyncio.to_thread(write_func, dataset_name, chunk)
                    failed, deleted = _extract_failed_records(result, chunk, chunk_start, label, dataset_name, _logging)
                    all_deleted_ids.update(deleted)
                    return failed, []
                except Exception as retry_err:
                    _logging.warning(
                        f"{label}Chunk {chunk_start}-{chunk_start + len(chunk)} of {total} "
                        f"to {dataset_name} timed out on retry ({retry_err}), skipping"
                    )
                    retryable = list(chunk) if _is_retryable_error(retry_err) else []
                    return list(chunk), retryable
            status_code = getattr(chunk_err, 'status_code', None)
            if status_code in (401, 403):
                _logging.warning(
                    f"{label}Chunk {chunk_start}-{chunk_start + len(chunk)} of {total} "
                    f"to {dataset_name} got {status_code} (auth expired), skipping fallback"
                )
                return list(chunk), []
            response_body = getattr(chunk_err, 'response_text', None)
            # Cap the fallback sub-chunk size at 50 (independent of chunk_size)
            # so a failed oversized chunk is still retried in granular sub-chunks
            # (isolating bad records) rather than one giant all-or-nothing retry.
            fallback_size = min(50, chunk_size // 4)
            _logging.warning(
                f"{label}Chunk {chunk_start}-{chunk_start + len(chunk)} of {total} "
                f"to {dataset_name} failed ({chunk_err}), retrying in sub-chunks of {fallback_size}"
            )
            if response_body:
                _logging.warning(f"{label}Server response: {response_body}")
            failed = []
            retryable = []
            sub_chunks = [chunk[i:i + fallback_size] for i in range(0, len(chunk), fallback_size)]
            for sc_idx, sub_chunk in enumerate(sub_chunks):
                try:
                    if token_refresher:
                        token_refresher()
                    sub_result = await asyncio.to_thread(write_func, dataset_name, sub_chunk)
                    sub_failed, sub_deleted = _extract_failed_records(
                        sub_result, sub_chunk, chunk_start + sc_idx * fallback_size,
                        label, dataset_name, _logging)
                    all_deleted_ids.update(sub_deleted)
                    failed.extend(sub_failed)
                except Exception as sub_err:
                    if is_timeout_error(sub_err):
                        _logging.warning(
                            f"{label}Sub-chunk {sc_idx} at offset {chunk_start + sc_idx * fallback_size} "
                            f"timed out ({sub_err}), skipping remaining sub-chunks"
                        )
                        remaining_sub_chunks = sub_chunks[sc_idx:]
                        for remaining in remaining_sub_chunks:
                            failed.extend(remaining)
                        if _is_retryable_error(sub_err):
                            retryable.extend(
                                record for remaining in remaining_sub_chunks
                                for record in remaining)
                        break
                    sub_response = getattr(sub_err, 'response_text', None)
                    _logging.error(
                        f"{label}Sub-chunk {sc_idx} ({len(sub_chunk)} records at offset {chunk_start + sc_idx * fallback_size}) "
                        f"REJECTED by {dataset_name}: {sub_err}"
                    )
                    if sub_response:
                        _logging.error(f"{label}Server response: {sub_response}")
                    failed.extend(sub_chunk)
                    if _is_retryable_error(sub_err):
                        retryable.extend(sub_chunk)
            return failed, retryable

    async def _write_chunk_throttled(
            chunk: List[dict], chunk_start: int) -> Tuple[List[dict], List[dict]]:
        if semaphore:
            async with semaphore:
                return await _write_chunk(chunk, chunk_start)
        return await _write_chunk(chunk, chunk_start)

    results = await asyncio.gather(*[
        _write_chunk_throttled(chunk, i * chunk_size)
        for i, chunk in enumerate(chunks)
    ])
    failed_by_identity = {
        id(record): record
        for chunk_failed, _ in results
        for record in chunk_failed
    }
    retryable = [record for _, chunk_retryable in results for record in chunk_retryable]

    for retry_index in range(max(0, transient_retry_attempts)):
        if not retryable:
            break
        await asyncio.sleep(max(0.0, retry_backoff_seconds) * (2 ** retry_index))
        retry_chunks = [retryable[i:i + chunk_size]
                        for i in range(0, len(retryable), chunk_size)]
        retry_results = await asyncio.gather(*[
            _write_chunk_throttled(chunk, i * chunk_size)
            for i, chunk in enumerate(retry_chunks)
        ])
        attempted_ids = {id(record) for record in retryable}
        for record_id in attempted_ids:
            failed_by_identity.pop(record_id, None)
        retryable = []
        for retry_failed, retryable_failed in retry_results:
            for record in retry_failed:
                failed_by_identity[id(record)] = record
            retryable.extend(retryable_failed)

    return list(failed_by_identity.values()), all_deleted_ids


# --- DataFrame helpers ---

def int_conversion(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Converts columns in the DataFrame to int64 to prevent ints from converting to floats.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        columns (List[str]): The columns to convert.

    Returns:
        None
    """
    for col in columns:
        if col in df.columns:
            # First convert to numeric (handles strings), then to Int64
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    for col in columns:
        if col in df.columns:
            if not (df[col].dtype == 'Int64'):
                raise TypeError(f"Column {col} could not be converted to Int64")


# --- Coordinate transform utilities ---

def enu_to_ecef_rotation_matrix(lat: float, lon: float) -> np.ndarray:
    """Rotation matrix from local ENU (East, North, Up) frame to ECEF.
    Columns correspond to [East, North, Up] basis vectors in ECEF."""
    clon = np.cos(lon)
    slon = np.sin(lon)
    clat = np.cos(lat)
    slat = np.sin(lat)
    return np.array([[-slon, -slat * clon, clat * clon],
                     [ clon, -slat * slon, clat * slon],
                     [ 0.,    clat,         slat]])

def ecef_to_enu_rotation_matrix(lat: float, lon: float) -> np.ndarray:
    """Rotation matrix from ECEF to local ENU (East, North, Up) frame.
    Rows correspond to [East, North, Up] components."""
    clon = np.cos(lon)
    slon = np.sin(lon)
    clat = np.cos(lat)
    slat = np.sin(lat)
    return np.array([[-slon,         clon,         0.],
                     [-slat * clon, -slat * slon,  clat],
                     [ clat * clon,  clat * slon,  slat]])

def enu_to_ecef_vector(lat: float, lon: float, v_east: float, v_north: float, v_up: float) -> np.ndarray:
    """Rotate a vector from local ENU frame to ECEF coordinates."""
    M = enu_to_ecef_rotation_matrix(lat, lon)
    v = np.dot(M, np.array([v_east, v_north, v_up]))
    return v

def ecef_to_enu_vector(lat: float, lon: float, v_x: float, v_y: float, v_z: float) -> Tuple[float, float, float]:
    """Rotate a vector from ECEF coordinates to local ENU frame.
    Returns (v_east, v_north, v_up)."""
    M = ecef_to_enu_rotation_matrix(lat, lon)
    v = np.dot(M, np.array([v_x, v_y, v_z]))
    v_east = v[0]
    v_north = v[1]
    v_up = v[2]
    return v_east, v_north, v_up

def rotate_covariance_matrix_enu_to_ecef(lat: float, lon: float, cov: np.ndarray) -> np.ndarray:
    """Rotate a covariance matrix in [E, N, U] order to ECEF."""
    M = enu_to_ecef_rotation_matrix(lat, lon)
    return np.dot(M, np.dot(cov, M.T))

def rotate_covariance_matrix_ecef_to_enu(lat: float, lon: float, cov: np.ndarray) -> np.ndarray:
    """Rotate a covariance matrix from ECEF to [E, N, U] order."""
    M = ecef_to_enu_rotation_matrix(lat, lon)
    return np.dot(M, np.dot(cov, M.T))

def covariance_matrix_to_uncertainty_ellipse(cov: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a covariance matrix to uncertainty ellipse parameters.
    Parameters:
    cov (np.ndarray): A 2x2 or 3x3 covariance matrix in [E, N, U] order.
    Returns:
    float: Length of the major axis of the ellipse.
    float: Length of the minor axis of the ellipse.
    float: Azimuth (orientation) of the ellipse in radians, CW from North.
    """
    cov_2d = cov[:2, :2]
    orientation, major_axis, minor_axis = covariance_ellipse(cov_2d, deviations=2.448)
    # filterpy returns angle CCW from first axis (East in ENU);
    # convert to azimuth CW from North
    azimuth = (np.pi/2 - orientation) % (2 * np.pi)
    # Schema max for Angle is 6.283185; 2π ≡ 0 for angles
    if azimuth >= 6.283185:
        azimuth = 0.0
    return major_axis, minor_axis, azimuth

def pandas_ecef_vel_to_wgs84_by_row(row: pd.Series) -> pd.Series:
    v_east, v_north, v_up = ecef_to_enu_vector(
                      row['estimatedKinematics.position.latitude'],
                      row['estimatedKinematics.position.longitude'],
                      row['ecefVelocity.dx'],
                      row['ecefVelocity.dy'],
                      row['ecefVelocity.dz'])
    return pd.Series([v_east, v_north, -v_up],
                     index=['estimatedKinematics.velocity.eastSpeed',
                            'estimatedKinematics.velocity.northSpeed',
                            'estimatedKinematics.velocity.downSpeed'])


# --- Script loading utilities ---

def _load_module_from_source(module_name: str, source_code: str) -> types.ModuleType:
    """Load a Python module from source code string without writing to disk."""
    mod = types.ModuleType(module_name)
    mod.__file__ = f"<in-memory:{module_name}>"
    exec(compile(source_code, mod.__file__, 'exec'), mod.__dict__)
    sys.modules[module_name] = mod
    return mod


def import_script_files(config_list: List[dict], script_configs: List[dict], caller_globals: Optional[dict] = None) -> None:
    '''
    Imports script files dynamically in-memory (no temp directory needed).
    Args: config_list (List[dict]): list of dictionaries containing configuration information
          script_configs (List[dict]): list of dictionaries containing script information
          caller_globals (dict): the caller's globals() dict to inject loaded modules into
    '''
    if caller_globals is None:
        caller_globals = globals()

    perspective_scripts = {}
    for config in config_list:
        for key in config.keys():
            if 'script_name' in key:
                # check to see if the script was already defined with a different key
                if key in perspective_scripts.keys() and \
                    perspective_scripts[key] != config[key]:
                    logging.error(" ")
                    logging.error(" ")
                    logging.error('-' * 50)
                    logging.error(f"Error: more than one {key} has been defined ")
                    logging.error(f"    for perspective {config['perspective']}")
                    logging.error('-' * 50)
                    logging.error(" ")
                    logging.error(" ")
                    # sleep for a few seconds to allow the error message to be read
                    time.sleep(5)
                perspective_scripts[key]=config[key]
                # e.g. perspective_scripts['custom_functions_script_name']='custom_functions'
                

    # dict of scripts and their associated filenames
    script_filenames = \
    {'custom_functions_script_name': 'custom_functions.py',
     'unit_conversions_script_name': 'unit_conversions.py',
     'object_enrichment_script_name': 'object_enrichment.py'}



    # script dict is a dictionary of script names and their bodies
    # from the Stream_Manager_Functions dataset
    script_dict = {}
    {script_dict.update({script['script_name']: script['script_body']}) for script in script_configs}

    # import the scripts using _load_module_from_source (in-memory, no files written)
    for script_name, script_body in script_dict.items():
        if script_name in perspective_scripts.values():
            #get the key for the script name
            script_name_key = list(perspective_scripts.keys())[list(perspective_scripts.values()).index(script_name)]

            module_name = script_filenames[script_name_key].replace('.py', '')
            mod = _load_module_from_source(module_name, script_body)
            caller_globals[module_name] = mod
            _script_sources[module_name] = script_body

            logging.info(f"Imported {module_name} from Stream_Manager_Functions")


# --- Configuration loading ---

def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return False


def _apply_disable_all_other_datasets(config_list: List[dict]) -> None:
    """Honor the single-feed focus flag used by ad-hoc deployments.

    Accept both ``disable_all_other_datasets`` and the legacy misspelling
    ``disable_all_other_datsets``. At most one config may set either key true; when set,
    every other config in the perspective is marked disabled before the rest of
    config validation/routing runs.
    """
    focus_configs = [
        config for config in config_list
        if (
            _config_bool(config.get('disable_all_other_datasets'))
            or _config_bool(config.get('disable_all_other_datsets'))
        )
    ]
    if len(focus_configs) > 1:
        origins = [str(config.get('origin_dataset', '<unknown>')) for config in focus_configs]
        raise ValueError(
            "Only one Stream_Manager_Configuration row may set "
            f"disable_all_other_datasets=true; found {origins}")
    if not focus_configs:
        return

    enabled_config = focus_configs[0]
    enabled_origin = enabled_config.get('origin_dataset', '<unknown>')
    for config in config_list:
        config['disabled'] = config is not enabled_config
    enabled_config['disabled'] = False
    logging.warning(
        f"disable_all_other_datasets=true on {enabled_origin}; disabling all other "
        "configs in this perspective")


def uses_object_event_kinematics(config: dict) -> bool:
    return config.get('bypass_tracker', False) is True


def find_and_validate_configs(stream_manager_perspective: str, include_scripts: bool = True, rc_instance: Optional[Any] = None, caller_globals: Optional[dict] = None) -> List[dict]:
    """
    Args:
        stream_manager_perspective (str): name of the stream manager perspective
        include_scripts (bool): whether to import script files
        rc_instance: ReadController instance (required)
        caller_globals (dict): the caller's globals() dict for dynamic script imports

    Returns:
        tuple: (List[dict], List)
    """
 
    _rc = rc_instance
    if _rc is None:
        raise ValueError("rc_instance must be provided")
    
    dataset_name='Stream_Manager_Configuration'
    try:
        config_list=_rc.search("select * from " + dataset_name +
                          " where perspective = '"
                          + stream_manager_perspective + "'",
                         )
    except Exception as err:
        logging.error(f"{err=}, {type(err)=}")
        raise Exception("Error with attempting to retrieve Stream_Manager_Configuration from Crucible")

    if len(config_list) == 0:
        logging.error('-' * 50)
        logging.error(f" No Stream Manager configuration found for {stream_manager_perspective}")
        logging.error('-' * 50)
        raise ValueError(f"No Stream Manager configuration found for {stream_manager_perspective}")

    perspective_config_origin = f'{stream_manager_perspective}_perspective_config'
    perspective_configs = [
        config for config in config_list
        if config.get('origin_dataset').lower() == perspective_config_origin.lower()
    ]
    if len(perspective_configs) > 1:
        raise ValueError(
            f"Multiple perspective configs found with origin_dataset "
            f"'{perspective_config_origin}'")
    perspective_config = perspective_configs[0] if perspective_configs else None
    config_list = [
        config for config in config_list
        if config.get('origin_dataset').lower() != perspective_config_origin.lower()
    ]
    if perspective_config:
        logging.info(
            "Using perspective_config %s with keys and values: %s",
            perspective_config_origin,
            perspective_config)


    # Crucible will convert lists of strings into concatenated strings; 
    # convert these back to lists (if needed)
    for d in config_list:
        if ',' in d['origin_unique_ID_column']:
            d['origin_unique_ID_column'] = d['origin_unique_ID_column'].split(',')
        if ',' in d['destination_unique_ID_column']:
            d['destination_unique_ID_column'] = d['destination_unique_ID_column'].split(',')

    _apply_disable_all_other_datasets(config_list)
    for config in config_list:
        if uses_object_event_kinematics(config):
            config['skip_tracker'] = True
      
    # Make sure that objectId.descriptiveLabel isn't used 
    # as the destination unique ID column.
    # This field is overwritten programmatically and should not be used.
    for d in config_list:
        if 'disabled' in d.keys() and not d['disabled'] or 'disabled' not in d.keys():
            if d['destination_unique_ID_column'] == 'objectId.descriptiveLabel':
                logging.warning('-' * 50)
                logging.warning(f" For {d['origin_dataset']}: ")
                logging.warning(" descriptiveLabel should not be used as destination_unique_ID_column")
                logging.warning('-' * 50)
                time.sleep(5)
            if isinstance(d['destination_unique_ID_column'],list):
                for col in d['destination_unique_ID_column']:
                    if col == 'objectId.descriptiveLabel':
                        logging.warning('-' * 50)
                        logging.warning(f" For {d['origin_dataset']}: ")
                        logging.warning(" descriptiveLabel should not be used as destination_unique_ID_column")
                        logging.warning('-' * 50)
                        time.sleep(5)

    # Propagate perspective-level settings. Values on individual feed configs
    # are ignored when the optional perspective config defines the same key.
    deprecated_object_sync_keys = [
        'object_sync',
        'object_sync_interval',
        'object_sync_reconcile_interval',
    ]
    for d in config_list:
        if d.get('disabled'):
            continue
        for key in deprecated_object_sync_keys:
            if key in d:
                logging.warning(
                    f"Config key '{key}' is deprecated and ignored. Object sync is SSE-driven; "
                    "set perspective-level 'refresh_interval' for full-pull repair and "
                    "'reconcile_interval' for UUID-only hard-delete checks.")

    perspective_level_keys = [
        'object_event_dataset',
        'object_dataset',
        'object_management_event_dataset',
        'track_event_dataset',
        'superseded_object_dataset',
        'component_track_event_dataset',
        'component_track_head_dataset',
        'principal_track_event_dataset',
        'principal_track_head_dataset',
        'refresh_interval',
        'reconcile_interval',
        'batch_write_chunk_size',
        'batch_update_chunk_size',
        'batch_write_max_concurrent',
        'deduplication',
        'deduplication_interval',
        'object_sync_full_replace',
        'enable_sources_array',
        'num_fusion_workers',
        'use_numpy_kalman',
        'use_numpy_fusion',
        'skip_head_preload',
        'head_update_interval_seconds',
        'num_object_manager_workers',
    ]
    perspective_level_keys.extend(sorted({
        key
        for config in ([perspective_config] if perspective_config else []) + config_list
        for key in config
        if key.endswith('script_name')
    }))
    dataset_keys = {
        key for key in perspective_level_keys
        if key.endswith('_dataset') and key != 'origin_dataset'
    }
    for key in perspective_level_keys:
        defined_values = [
            (config.get('origin_dataset', '<unknown>'), config[key])
            for config in config_list
            if key in config
        ]
        if (
            perspective_config is not None
            and key in dataset_keys
            and key not in perspective_config
            and defined_values
        ):
            raise ValueError(
                f"Dataset key '{key}' must be defined in "
                f"{perspective_config_origin}; dataset config values are ignored")
        if perspective_config is not None and key in perspective_config:
            value = perspective_config[key]
            if defined_values:
                logging.warning(
                    "Perspective-level key '%s' on dataset configs is ignored "
                    "because %s defines %r. Ignored values: %r",
                    key,
                    perspective_config_origin,
                    value,
                    defined_values)
            conflicting_values = []
        else:
            value = defined_values[0][1] if defined_values else None
            conflicting_values = [
                (origin, feed_value)
                for origin, feed_value in defined_values[1:]
                if feed_value != value
            ]
        if conflicting_values:
            raise ValueError(
                f"Perspective-level key '{key}' must agree across all configs. "
                f"Expected {value!r} from {defined_values[0][0]}; conflicts: "
                f"{conflicting_values!r}")
        if value is not None:
            for d in config_list:
                d[key] = value

    # Log the resolved perspective-level key values
    for key in perspective_level_keys:
        # Use the first active config's value as representative
        resolved = None
        for d in config_list:
            if key in d and ('disabled' not in d or not d['disabled']):
                resolved = d[key]
                break
        logging.info(f"Perspective-level key '{key}' = {resolved}")

    # Set default for batch_write_chunk_size if not defined in any config
    for d in config_list:
        if 'batch_write_chunk_size' not in d:
            d['batch_write_chunk_size'] = 250
        if 'batch_update_chunk_size' not in d:
            d['batch_update_chunk_size'] = 50
        # Default batch_write_max_concurrent to a reasonable value if unset,
        # and ensure it is an int (Crucible returns strings). Capping concurrency
        # avoids overwhelming the write endpoint (which caused read timeouts).
        if d.get('batch_write_max_concurrent') is None:
            d['batch_write_max_concurrent'] = 4
        else:
            d['batch_write_max_concurrent'] = int(d['batch_write_max_concurrent'])
        if d.get('object_manager_batch_write_max_concurrent') is not None:
            d['object_manager_batch_write_max_concurrent'] = int(
                d['object_manager_batch_write_max_concurrent'])

    for config in config_list:
        if 'origin_dataset_name' in config.keys():
            logging.warning('-' * 50)
            logging.warning(f" For {config['origin_dataset']}: ")
            logging.warning("origin_dataset_name is deprecated, use origin_dataset instead")
            logging.warning('-' * 50)
            config['origin_dataset'] = config['origin_dataset_name']
            time.sleep(5)

    dataset_name='Stream_Manager_Functions'
    script_configs=_rc.search("select * from " + dataset_name,
                             )
  
    # write script files to config directory and import them
    # note: this is done here for two reasons:
    # 1) scope: these need to be imported either at the top of the file or in this function
    # 2) we don't know which functions will be needed until we read the config
    if include_scripts:
        import_script_files(config_list, script_configs, caller_globals=caller_globals)

    return config_list


# --- Supersede map utilities ---

def build_supersede_map(management_event_df: pd.DataFrame, init_supersede_map: Dict[str, str] = {}) -> Dict[str, str]:
    """
    Given a DataFrame of management events (SUPERSEDE, DELETE, RESTORE actions), build a map of objectIds.
    For SUPERSEDE: maps to their latest supersededBy object, following chains.
    For DELETE: maps to None to indicate the object has been deleted.
    For RESTORE: removes the object from the map (undoes supersession/deletion).
    Also follows chains starting from init_supersede_map.
    
    Args:
        management_event_df (pd.DataFrame): DataFrame containing management events
        init_supersede_map (Dict[str, str]): Initial supersede map to build upon
        
    Returns:
        Dict[str, str]: Mapping of superseded objectId -> superseding objectId (or None if deleted)
    """
    if management_event_df.empty:
        return init_supersede_map
    
    # Ensure the DataFrame has the necessary columns
    required_columns = {'objectId', 'crucibleHeader.updatedDate', 'action'}
    if not required_columns.issubset(management_event_df.columns):
        logging.warning(f"build_supersede_map: Missing columns: {required_columns - set(management_event_df.columns)}")
        return dict(init_supersede_map) if init_supersede_map else {}
    
    # Start with existing mappings, then overlay with new events
    supersedes = dict(init_supersede_map)
    
    # Keep only the LATEST event per objectId, then apply it.  Sorting by
    # updatedDate descending and dropping duplicate objectIds (keep='first')
    # gives the same "latest event wins" selection and tie-breaking as the
    # previous row-by-row loop, but without iterating every row.
    latest = (management_event_df
              .sort_values(by='crucibleHeader.updatedDate', ascending=False)
              .drop_duplicates(subset='objectId', keep='first'))

    action = latest['action']
    # SUPERSEDE -> map to supersededBy (None if the column is absent, matching
    # the previous row.get('supersededBy')).
    if 'supersededBy' in latest.columns:
        _sup = latest.loc[action == 'SUPERSEDE', ['objectId', 'supersededBy']]
        supersedes.update(dict(zip(_sup['objectId'], _sup['supersededBy'])))
    else:
        supersedes.update({oid: None for oid in latest.loc[action == 'SUPERSEDE', 'objectId']})
    # DELETE -> map to None (tombstone).
    supersedes.update({oid: None for oid in latest.loc[action == 'DELETE', 'objectId']})
    # RESTORE -> remove from the map (object is active again).
    for oid in latest.loc[action == 'RESTORE', 'objectId']:
        supersedes.pop(oid, None)
    
    
    # Process all objectIds in supersedes (includes both init_supersede_map and new events)
    result = {}
    for idx, object_id in enumerate(supersedes):
        
        current = object_id
        next_superseded = supersedes.get(current)
        
        # Treat NaN/non-string values as None (can happen if supersededBy column
        # was missing or had NaN values in the DataFrame)
        if next_superseded is not None and not isinstance(next_superseded, str):
            next_superseded = None
        
        # Follow the chain until we reach a supersededBy that is not in the mapping
        # Add cycle detection to prevent infinite loops
        visited = {object_id}
        while next_superseded in supersedes:
            if next_superseded in visited:
                # Circular reference detected — stop here; the cycle entry point
                # is the valid "root" (e.g. A→B→A resolves A to itself)
                break
            visited.add(next_superseded)
            current = next_superseded
            next_superseded = supersedes.get(current)
            # Sanitize non-string values in the chain
            if next_superseded is not None and not isinstance(next_superseded, str):
                next_superseded = None
        
        # Determine final mapping
        if next_superseded is None:
            # Chain ends in deletion (or object itself was deleted) -> None
            result[object_id] = None
        else:
            # Chain ends at a valid object (may be self for circular chains)
            result[object_id] = next_superseded

    
    return result


def _find_management_updated_col(df: pd.DataFrame) -> Optional[str]:
    for col in ('crucibleHeader.updatedDate', 'updatedDate'):
        if col in df.columns:
            return col
    for col in df.columns:
        if col.endswith('.updatedDate'):
            return col
    return None


def _timestamp_offset_seconds(timestamp: Any) -> int:
    ts = pd.to_datetime(timestamp, utc=True, errors='coerce')
    if pd.isna(ts):
        ts = pd.Timestamp.now(tz='UTC') - pd.Timedelta(seconds=5)
    age_seconds = (pd.Timestamp.now(tz='UTC') - ts).total_seconds()
    return max(1, int(np.ceil(age_seconds)))


def _latest_management_timestamp(management_event_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if management_event_df is None or management_event_df.empty:
        return None
    updated_col = _find_management_updated_col(management_event_df)
    if updated_col is None:
        return None
    updated = pd.to_datetime(management_event_df[updated_col], utc=True, errors='coerce').dropna()
    if updated.empty:
        return None
    return updated.max()


def _empty_management_cursor() -> pd.Timestamp:
    # If there were no rows, advance to near-now so future cycles do a tiny
    # delta query instead of repeatedly scanning the full 30-day window.  The
    # small overlap protects events committed while this process was querying.
    return pd.Timestamp.now(tz='UTC') - pd.Timedelta(seconds=5)


def get_management_events(
        perspective_config: Dict[str, Any],
        rc_instance: Optional[Any] = None,
        auth_instance: Optional[Any] = None,
        actions: Tuple[str, ...] = ('SUPERSEDE', 'DELETE', 'RESTORE'),
        since_timestamp: Optional[Any] = None,
        lookback_days: int = 30,
        limit: int = 10000,
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """Read object-management events and return ``(df, latest_updatedDate)``.

    ``since_timestamp`` switches this from the historical 30-day bootstrap read
    to a cheap incremental read.  This keeps long-running side processes from
    rebuilding the full supersede map on every management wakeup/timeout. Reads
    are paginated so active datasets cannot silently truncate supersede state.
    """
    _rc = rc_instance
    _auth = auth_instance
    if _rc is None:
        raise ValueError("rc_instance must be provided")
    if _auth is None:
        raise ValueError("auth_instance must be provided")

    object_management_event_dataset = perspective_config.get('object_management_event_dataset')
    if not object_management_event_dataset:
        return pd.DataFrame(), since_timestamp

    _rc.token = _auth.get_token()
    action_predicate = ' OR '.join(
        f"{object_management_event_dataset}.action = '{action}'"
        for action in actions)
    if since_timestamp is None:
        time_predicate = (
            f"{object_management_event_dataset}.crucibleHeader.updatedDate > "
            f"TIMESTAMP_OFFSET(-{int(lookback_days)},'days')")
    else:
        since_seconds = _timestamp_offset_seconds(since_timestamp)
        time_predicate = (
            f"{object_management_event_dataset}.crucibleHeader.updatedDate > "
            f"TIMESTAMP_OFFSET(-{since_seconds},'seconds')")

    page_size = max(1, int(limit))
    pages = []
    offset = 0
    while True:
        management_sql_query = f"""
            SELECT * FROM {object_management_event_dataset}
            WHERE ({action_predicate})
            AND {time_predicate}
            ORDER BY {object_management_event_dataset}.crucibleHeader.updatedDate ASC,
                     {object_management_event_dataset}.crucibleHeader.uuid ASC
            OFFSET {offset} ROWS FETCH NEXT {page_size} ROWS ONLY
        """

        try:
            page = _rc.search(
                management_sql_query, format='dataframe', auto_backtick=False)
        except Exception as e:
            error_msg = str(e)
            if (
                "Table 'crucibleHeader' not found" in error_msg
                or "Error processing query" in error_msg
                or "Bad Request" in error_msg
            ):
                logging.warning("Empty Object Management Events dataset, returning empty management event frame")
            else:
                logging.error(f"Error occurred while fetching management events: {e}")
            break

        if page is None or page.empty:
            break
        pages.append(page)
        if len(page) < page_size:
            break
        offset += page_size

    management_event_df = (
        pd.concat(pages, ignore_index=True) if pages else pd.DataFrame())

    latest = _latest_management_timestamp(management_event_df)
    if latest is None and since_timestamp is None:
        latest = _empty_management_cursor()
    elif latest is None:
        latest = since_timestamp
    if len(pages) > 1:
        logging.info(
            f"management event query loaded {len(management_event_df)} row(s) "
            f"across {len(pages)} page(s)")
    return management_event_df, latest


def get_supersede_map(perspective_config: Dict[str, Any], rc_instance: Optional[Any] = None,
                      auth_instance: Optional[Any] = None,
                      return_latest_timestamp: bool = False) -> Dict[str, str]:
    """
    Returns a dictionary of superseded objects from
    the object management event dataset.
    
    Args:
        perspective_config (Dict[str, Any]): Configuration containing dataset names
        rc_instance: Optional ReadController instance. If None, raises ValueError.
        auth_instance: Optional Authenticator instance. If None, raises ValueError.
        
    Returns:
        Dict[str, str]: Mapping of superseded objectId -> superseding objectId
    """
    print("Object Manager: get_supersede_map() starting...", flush=True)
    _rc = rc_instance
    _auth = auth_instance
    if _rc is None:
        raise ValueError("rc_instance must be provided")
    if _auth is None:
        raise ValueError("auth_instance must be provided")

    #onetime download of all management events
    supersede_map = {}

    latest_timestamp = None
    if 'object_management_event_dataset' in perspective_config:
        object_management_event_dataset = perspective_config['object_management_event_dataset']
        try:
            print(f"Object Manager: get_supersede_map() querying {object_management_event_dataset}...", flush=True)
            management_event_df, latest_timestamp = get_management_events(
                perspective_config, rc_instance=_rc, auth_instance=_auth)
            print(f"Object Manager: get_supersede_map() query complete, got {len(management_event_df) if hasattr(management_event_df, '__len__') else 'unknown'} rows", flush=True)
        except Exception as e:
            print(f"Object Manager: get_supersede_map() query error: {e}", flush=True)
            management_event_df = pd.DataFrame()
        if not management_event_df.empty:
            supersede_map = build_supersede_map(management_event_df)

    print(f"Object Manager: get_supersede_map() returning {len(supersede_map)} entries", flush=True)
    if return_latest_timestamp:
        return supersede_map, latest_timestamp
    return supersede_map


def get_duplicate_protected_ids(
        perspective_config: Dict[str, Any], rc_instance: Any,
        auth_instance: Any, restore_cooldown_seconds: int = 300,
        supersede_map: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Return object IDs the duplicate detector must temporarily ignore."""
    if supersede_map is None:
        supersede_map = get_supersede_map(
            perspective_config, rc_instance=rc_instance,
            auth_instance=auth_instance)
    protected_ids = set(supersede_map)
    protected_ids.update(
        value for value in supersede_map.values() if value is not None)

    cooldown = int(perspective_config.get(
        'restore_cooldown_seconds', restore_cooldown_seconds))
    if cooldown <= 0:
        return protected_ids

    try:
        since = pd.Timestamp.now(tz='UTC') - pd.Timedelta(seconds=cooldown)
        restore_df, _ = get_management_events(
            perspective_config, rc_instance=rc_instance,
            auth_instance=auth_instance, actions=('RESTORE',),
            since_timestamp=since)
        if not restore_df.empty:
            key_col = ('objectId' if 'objectId' in restore_df.columns
                       else 'objectId.uuid' if 'objectId.uuid' in restore_df.columns
                       else None)
            if key_col:
                restored_ids = set(restore_df[key_col].dropna().astype(str))
                protected_ids.update(restored_ids)
                logging.info(
                    f"Protected {len(restored_ids)} recently restored object(s) "
                    f"for {cooldown}s")
    except Exception as exc:
        logging.warning(f"Could not load recent RESTORE actions: {exc}")
    return protected_ids
