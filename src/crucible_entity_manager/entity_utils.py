"""
Shared utility functions for the entity_manager package.

This module contains functions that are imported across multiple scripts
in the entity_manager directory.

Crucible write-API behavior (v2 WriteController):
    * write_record_batch_by_name / upsert_by_name = POST. Return an int status
        code (normally 204), not records. HTTP/API exceptions are failures.
    * update_entity_record_batch_by_name = PUT. The v2 helper calls it with
        include_failed_records=True. The returned list contains records that failed
        validation or could not be updated; an empty list means no failed records.
    * write_batch_chunked() keeps these contracts explicit through response_mode
        instead of inferring behavior from a callable.
"""

from __future__ import annotations

from filterpy.stats import covariance_ellipse
from datetime import timezone as tz
from datetime import datetime as dt
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import traceback
import logging
import asyncio
import random
import signal
import types
import json
import hashlib
import math
import time
import sys
import os

# --- Constants ---

stream_manager_config_dataset_name = 'Entity_Stream_Manager_Configurations'
stream_manager_functions_dataset_name = 'Entity_Stream_Manager_Functions'
PERSPECTIVE_CONFIG_PREFIX = 'perspective_config'

# Global dict holding script source code so child processes can re-create modules.
_script_sources: Dict[str, str] = {}


def canonical_identity_value(value: Any) -> str:
    """Render one identity value without Pandas dtype coercion."""
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, str):
        if "." in value:
            try:
                numeric = float(value)
                if math.isfinite(numeric) and numeric.is_integer():
                    return str(int(numeric))
            except ValueError:
                pass
        return value
    if isinstance(value, list):
        return json.dumps(
            [canonical_identity_value(item) for item in value],
            separators=(",", ":"),
        )
    return str(value)


def identity_custom_id(
    record: Dict[str, Any], field_paths: Optional[List[str]] = None
) -> str:
    """Build a stable identity key from selected nested record paths.

    ``identity.*`` expands all current identity fields, preserving the original
    schema-flexible behavior. Explicit dotted paths select individual fields.
    """
    selected_paths = field_paths if field_paths is not None else ["identity.*"]
    values: Dict[str, Any] = {}
    for path in selected_paths:
        if path == "identity.*":
            identity = record.get("identity")
            if isinstance(identity, dict):
                for key, value in identity.items():
                    values[f"identity.{key}"] = value
            continue

        current: Any = record
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        values[path] = current

    parts = []
    for path, value in sorted(values.items()):
        canonical = canonical_identity_value(value)
        if canonical:
            label = path.removeprefix("identity.")
            parts.append(f"{label}:{canonical}")
    return "-".join(parts)


# --- Cruciblelib lazy imports ---

def _import_cruciblelib_modules() -> Tuple[Any, Any, Any, Any]:
    """Lazy-import cruciblelib controller and utility modules."""
    try:
        from cruciblelib import read_controller, write_controller, utils, authenticator
    except Exception:
        from . import read_controller, write_controller, utils, authenticator
    return read_controller, write_controller, utils, authenticator

def ensure_api_controllers(
    current_auth: Any,
    current_read_controller: Any,
    current_write_controller: Any,
) -> Tuple[Any, Any, Any]:
    """Return initialized API controllers, preserving an existing set."""
    if (
        current_auth is None
        or current_read_controller is None
        or current_write_controller is None
    ):
        return instantiate_api_controllers()
    return current_auth, current_read_controller, current_write_controller


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
    """Put data on either an asyncio.Queue or a multiprocessing Queue.
    Handles bounded queues gracefully when a consumer process falls behind."""
    try:
        if hasattr(queue, 'put_nowait'):
            if asyncio.iscoroutinefunction(queue.put):
                await queue.put(data)
            else:
                try:
                    queue.put_nowait(data)
                except Exception:
                    try:
                        queue.put(data, timeout=0.1)
                    except Exception as exc:
                        logging.warning(f"Raw source queue full, dropped message: {exc}")
        else:
            queue.put(data)
    except Exception as exc:
        logging.warning(f"Failed to put message on queue: {exc}")


async def SSE_listener(sql_query: str, message_queue: Any, auth: Any) -> None:
    """Resilient drop-in replacement for cruciblelib.utils.SSE_listener.

    Streams Crucible SSE results for ``sql_query`` and puts each parsed event
    (``json.loads(event.data)``) onto ``message_queue``.  Hardened so a large or
    stalled SSE payload can neither kill the listener (aiohttp ``LineTooLong``)
    nor hang it silently forever:
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
    _, _, utils, _ = _import_cruciblelib_modules()

    MAX_BACKOFF = 60
    PERSISTENT_FAILURE_THRESHOLD = 10
    # Entity/event snapshots grow as the number of live tracks climbs, so a
    # single SSE payload can be large; a payload bigger than this raises aiohttp
    # LineTooLong and kills the listener, so make it large and configurable.
    read_bufsize = int(os.getenv('CRUCIBLE_SSE_READ_BUFSIZE', str(256 * 1024 * 1024)))
    # Watchdog: if no event (or keepalive) arrives within this many seconds,
    # force a reconnect instead of hanging silently forever on a stalled or
    # oversized read.  0 disables it.
    read_timeout = float(os.getenv('CRUCIBLE_SSE_READ_TIMEOUT', '180'))

    protocol_string = 'https://'
    sse_url = os.getenv('CRUCIBLE_SSE_URL', protocol_string + os.environ['CRUCIBLE_SERVICES_HOST'] + '/api/v1/read/search/sse')
    url = sse_url + '?query=' + sql_query
    headers = {'Content-Type': 'text/plain'}
    ssl_verify = os.getenv('CRUCIBLE_SSL_VERIFY', 'false').lower() not in ('false', '0', 'no')
    logging.info(f'Connecting to {url}')
    # Track consecutive failures to drive the incremental-backoff strategy.
    attempt = 0
    while True:
        try:
            # Refresh access token before each connection attempt.
            await utils.refresh_access_token(auth, headers)
            # Configurable read buffer (default 256 MB) to support large snapshots.
            async with aiohttp.ClientSession(read_bufsize=read_bufsize, connector=aiohttp.TCPConnector(ssl=ssl_verify)) as session:
                connection_timeout = 10
                event_source = await connect_with_timeout(url, session, headers, connection_timeout)
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
                                    f"[{sql_query}]: No SSE event received in {read_timeout:.0f}s; "
                                    f"forcing reconnect (read watchdog)."
                                )
                                attempt += 1
                                break
                            try:
                                event_data = json.loads(event.data)
                            except json.JSONDecodeError:
                                logging.warning(f"[{sql_query}]: Skipping malformed event: {str(event.data)[:200]}")
                                continue
                            if len(event_data) > 0:
                                attempt = 0
                                await put_to_queue(message_queue, event_data)
                    except LineTooLong as e:
                        logging.error(
                            f"[{sql_query}]: SSE payload exceeded the {read_bufsize}-byte read buffer: {e}. "
                            f"Raise CRUCIBLE_SSE_READ_BUFSIZE if this persists. Reconnecting with backoff."
                        )
                        attempt += 1
                    except aiohttp.ClientResponseError as e:
                        if getattr(e, 'status', None) == 401:
                            logging.error(f"[{sql_query}]: Received 401 status despite refreshing token, retrying...")
                        else:
                            logging.error(f"[{sql_query}]: Response error occurred: {e}")
                            logging.error(traceback.format_exc())
                        attempt += 1
                    except aiohttp.ClientConnectionError as e:
                        logging.error(f"[{sql_query}]: Connection error occurred: {e}")
                        logging.error(traceback.format_exc())
                        attempt += 1
                    except aiohttp.ClientPayloadError as e:
                        error_str = str(e)
                        if '400' in error_str:
                            logging.warning(f"[{sql_query}]: Server returned 400 \u2014 dataset may be empty or column not yet available. Will retry with backoff.")
                        else:
                            logging.error(f"[{sql_query}]: Payload error occurred: {e}")
                            logging.error(traceback.format_exc())
                        attempt += 1
                    except Exception as e:
                        logging.error(f"[{sql_query}]: An unexpected error occurred: {e}")
                        logging.error(traceback.format_exc())
                        attempt += 1
                    finally:
                        # Ensure connections are closed properly.
                        try:
                            await event_source.__aexit__(None, None, None)
                        except Exception as cleanup_error:
                            logging.warning(f"[{sql_query}]: Error during connection cleanup: {cleanup_error}")
                else:
                    attempt += 1

                # Apply backoff after any error.
                if attempt > 0:
                    if attempt >= PERSISTENT_FAILURE_THRESHOLD and attempt % PERSISTENT_FAILURE_THRESHOLD == 0:
                        logging.critical(f"[{sql_query}]: {attempt} consecutive failed connection attempts. Check service health.")
                    backoff = min(2 ** attempt, MAX_BACKOFF) + random.uniform(0, 1)
                    logging.info(f"[{sql_query}]: Reconnecting in {backoff:.1f}s (attempt {attempt})")
                    await asyncio.sleep(backoff)
        except Exception as e:
            logging.error(f"[{sql_query}]: An error occurred while creating the session: {e}")
            logging.error(traceback.format_exc())
            attempt += 1
            if attempt >= PERSISTENT_FAILURE_THRESHOLD and attempt % PERSISTENT_FAILURE_THRESHOLD == 0:
                logging.critical(f"[{sql_query}]: {attempt} consecutive failed connection attempts. Check service health.")
            backoff = min(2 ** attempt, MAX_BACKOFF) + random.uniform(0, 1)
            logging.info(f"[{sql_query}]: Reconnecting in {backoff:.1f}s (attempt {attempt})")
            await asyncio.sleep(backoff)


def set_write_controller_timeouts(wc: Any, connect: float = 15.0, read: float = 120.0, write: float = 30.0, pool: float = 15.0) -> None:
    """Apply standard v2 controller timeouts (in seconds).

    Centralizes the timeout configuration used by all entity scripts so
    the values stay consistent in one place. Works for both read and write
    controllers (they share the same ``.timeout`` interface).
    """
    from cruciblelib.controllers.v2.controller import TimeoutConfig
    wc.timeout = TimeoutConfig(connect=connect, read=read, write=write, pool=pool)


def instantiate_api_controllers() -> Tuple[Any, Any, Any]:
    """Create and return (auth, rc, wc) with v2 controllers and standard timeouts.

    Centralizes controller creation for all entity scripts so every process
    shares the same read/write timeout configuration. This parallels
    object_manager's instantiate_api_controllers(); the read controller gets
    the same generous read timeout as the write controller.
    """
    read_controller, write_controller, _, authenticator = _import_cruciblelib_modules()
    _auth = authenticator.Authenticator()
    _rc = read_controller.ReadController(version='v2')
    set_write_controller_timeouts(_rc)
    _wc = write_controller.WriteController(version='v2')
    set_write_controller_timeouts(_wc)
    return _auth, _rc, _wc


# --- Error helpers ---

def is_timeout_error(err: Exception) -> bool:
    """Return True if the exception represents a network timeout rather than a data/validation error."""
    status = getattr(err, 'status_code', None)
    if status == 0:
        return True
    # 504 Gateway Time-out is a transient upstream timeout, not a data/validation error
    if status == 504:
        return True
    err_str = str(err).lower()
    for pattern in ('timed out', 'timeout', 'read timeout', 'connect timeout', 'time-out'):
        if pattern in err_str:
            return True
    original = getattr(err, 'original_exception', None)
    if original is not None:
        type_name = type(original).__name__.lower()
        if 'timeout' in type_name:
            return True
    return False


# --- Asyncio helpers ---

def _get_record_id(record: dict) -> Optional[str]:
    """Extract the primary-key value from an entity-dataset record.

    Entity-type datasets in Crucible key on different fields depending on the
    dataset (entityId for the Entities dataset, trackId for track-head
    datasets, objectId for object datasets). Try them in priority order; a
    given dataset uses a single key type, so both sent and returned records
    resolve via the same field.
    """
    eid = record.get('entityId')
    if isinstance(eid, str):
        return eid
    tid = record.get('trackId')
    if isinstance(tid, str):
        return tid
    obj = record.get('objectId')
    if isinstance(obj, dict):
        return obj.get('uuid')
    return record.get('objectId.uuid', obj if isinstance(obj, str) else None)


def _extract_dropped_records(result: Any, sent_chunk: List[dict], chunk_start: int,
                             label: str, dataset_name: str,
                             response_mode: str = 'write') -> Tuple[List[dict], set]:
    """Extract failed v2 entity updates using the explicit response mode.

    v2 update responses contain failed records only when
    ``include_failed_records=True``. POST writes return a status code and do
    not provide record-level results.

    Returns:
        tuple: (dropped_records, dropped_ids)
            - dropped_records: original records missing from the response
            - dropped_ids: their primary-key values

    The returned set contains failed primary-key values for the update/create
    fallback path.
    """
    if response_mode == 'write':
        return [], set()
    if response_mode != 'entity_update':
        raise ValueError(f"Unsupported Crucible write response mode: {response_mode}")
    if not result or not isinstance(result, list):
        return [], set()

    failed_ids = set()
    for record in result:
        if isinstance(record, dict):
            record_id = _get_record_id(record)
            if record_id:
                failed_ids.add(record_id)

    failed_records = []
    for record in sent_chunk:
        record_id = _get_record_id(record)
        if record_id and record_id in failed_ids:
            failed_records.append(record)

    if failed_records:
        logging.warning(
            f"{label}Chunk at offset {chunk_start}: {len(failed_records)} of {len(sent_chunk)} "
            f"record(s) failed to update in {dataset_name}: "
            f"{list(failed_ids)[:5]}{'...' if len(failed_ids) > 5 else ''}"
        )

    return failed_records, failed_ids


def _invoke_write(write_func: Callable, dataset_name: str, chunk: List[dict],
                  response_mode: str) -> Any:
    """Invoke a v2 POST write or entity PUT update with its contract."""
    if response_mode == 'entity_update':
        return write_func(dataset_name, chunk, include_failed_records=True)
    if response_mode == 'write':
        return write_func(dataset_name, chunk)
    raise ValueError(f"Unsupported Crucible write response mode: {response_mode}")


async def write_batch_chunked(
    json_list: List[dict],
    dataset_name: str,
    write_func: Callable,
    chunk_size: int,
    *,
    response_mode: str = 'write',
    label: str = '',
    token_refresher: Optional[Callable] = None,
    max_concurrent_writes: Optional[int] = None,
) -> Tuple[list, set]:
    """
    Write a list of JSON records to a Crucible dataset in concurrent chunks.

    ``response_mode='write'`` handles POST/204 writes. Use
    ``response_mode='entity_update'`` for v2 PUT updates; the helper requests
    failed records explicitly and returns them to the caller.

    Splits *json_list* into chunks of *chunk_size*, fires all chunks
    concurrently via asyncio.gather, and falls back to sub-chunk writes
    for any chunk that fails (except timeouts/auth errors, which skip
    the sub-chunk fallback and return the entire chunk as failed).

        Returns:
                tuple: (failed_records, failed_ids)
                        - failed_records: records rejected by the API or failed after
                            retries. For entity updates, these are the records returned by
                            v2 with ``include_failed_records=True``.
                        - failed_ids: primary-key values for those failed entity updates;
                            empty for POST writes.

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
                               in-flight simultaneously. None means no limit.
    """
    import asyncio as _asyncio

    if response_mode not in ('write', 'entity_update'):
        raise ValueError(f"Unsupported Crucible write response mode: {response_mode}")

    if not json_list:
        return [], set()

    total = len(json_list)
    chunks = [json_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
    logging.info(f"{label}Writing {total} records to {dataset_name} in {len(chunks)} chunks of {chunk_size}"
                 f"{f' (max {max_concurrent_writes} concurrent)' if max_concurrent_writes else ' (concurrent)'}")

    semaphore = _asyncio.Semaphore(max_concurrent_writes) if max_concurrent_writes else None
    all_failed_ids = set()

    async def _write_chunk(chunk: List[dict], chunk_start: int) -> list:
        """Returns list of records that failed to write or were dropped."""
        try:
            if token_refresher:
                token_refresher()
            result = await _asyncio.to_thread(
                _invoke_write, write_func, dataset_name, chunk, response_mode)
            failed, failed_ids = _extract_dropped_records(
                result, chunk, chunk_start, label, dataset_name, response_mode)
            all_failed_ids.update(failed_ids)
            return failed
        except Exception as chunk_err:
            if is_timeout_error(chunk_err):
                # Likely a stale keep-alive connection — retry once with a
                # fresh connection before giving up.
                try:
                    if token_refresher:
                        token_refresher()
                    result = await _asyncio.to_thread(
                        _invoke_write, write_func, dataset_name, chunk, response_mode)
                    failed, failed_ids = _extract_dropped_records(
                        result, chunk, chunk_start, label, dataset_name, response_mode)
                    all_failed_ids.update(failed_ids)
                    return failed
                except Exception as retry_err:
                    logging.warning(
                        f"{label}Chunk {chunk_start}-{chunk_start + len(chunk)} of {total} "
                        f"to {dataset_name} timed out on retry ({retry_err}), skipping"
                    )
                    return list(chunk)
            status_code = getattr(chunk_err, 'status_code', None)
            if status_code in (401, 403):
                logging.warning(
                    f"{label}Chunk {chunk_start}-{chunk_start + len(chunk)} of {total} "
                    f"to {dataset_name} got {status_code} (auth expired), skipping fallback"
                )
                return list(chunk)
            response_body = getattr(chunk_err, 'response_text', None)
            # Cap the fallback sub-chunk size at 50 (independent of chunk_size)
            # so a failed oversized chunk is still retried in granular sub-chunks
            # (isolating bad records) rather than one giant all-or-nothing retry.
            fallback_size = max(1, min(50, chunk_size // 4))
            logging.warning(
                f"{label}Chunk {chunk_start}-{chunk_start + len(chunk)} of {total} "
                f"to {dataset_name} failed ({chunk_err}), retrying in sub-chunks of {fallback_size}"
            )
            if response_body:
                logging.warning(f"{label}Server response: {response_body}")
            failed = []
            sub_chunks = [chunk[i:i + fallback_size] for i in range(0, len(chunk), fallback_size)]
            for sc_idx, sub_chunk in enumerate(sub_chunks):
                try:
                    if token_refresher:
                        token_refresher()
                    sub_result = await _asyncio.to_thread(
                        _invoke_write, write_func, dataset_name, sub_chunk, response_mode)
                    sub_failed, sub_failed_ids = _extract_dropped_records(
                        sub_result, sub_chunk,
                        chunk_start + sc_idx * fallback_size,
                        label, dataset_name, response_mode)
                    all_failed_ids.update(sub_failed_ids)
                    failed.extend(sub_failed)
                except Exception as sub_err:
                    if is_timeout_error(sub_err):
                        logging.warning(
                            f"{label}Sub-chunk {sc_idx} at offset {chunk_start + sc_idx * fallback_size} "
                            f"timed out ({sub_err}), skipping remaining sub-chunks"
                        )
                        for remaining in sub_chunks[sc_idx:]:
                            failed.extend(remaining)
                        break
                    sub_response = getattr(sub_err, 'response_text', None)
                    logging.error(
                        f"{label}Sub-chunk {sc_idx} ({len(sub_chunk)} records at offset {chunk_start + sc_idx * fallback_size}) "
                        f"REJECTED by {dataset_name}: {sub_err}"
                    )
                    if sub_response:
                        logging.error(f"{label}Server response: {sub_response}")
                    failed.extend(sub_chunk)
            return failed

    async def _write_chunk_throttled(chunk: List[dict], chunk_start: int) -> list:
        if semaphore:
            async with semaphore:
                return await _write_chunk(chunk, chunk_start)
        return await _write_chunk(chunk, chunk_start)

    results = await _asyncio.gather(*[
        _write_chunk_throttled(chunk, i * chunk_size)
        for i, chunk in enumerate(chunks)
    ])
    failed = [record for chunk_failed in results for record in chunk_failed]
    return failed, all_failed_ids


def _track_id_value(record: dict) -> Any:
    track_id = record.get('trackId')
    if isinstance(track_id, dict):
        return track_id.get('uuid')
    return track_id


async def best_effort_update_track_records(
        records: List[dict], dataset_name: str, batch_size: int,
        write_func: Callable, batch_writer: Callable, *,
        label: str, log_prefix: str = '',
        token_refresher: Optional[Callable] = None,
        max_concurrent_writes: Optional[int] = None) -> List[dict]:
    """Update existing track heads and return failed records for later retry."""
    if not records or not dataset_name:
        return []
    try:
        failed_records, missing_ids = await batch_writer(
            records, dataset_name, write_func, batch_size,
            response_mode='entity_update', label=label,
            token_refresher=token_refresher,
            max_concurrent_writes=max_concurrent_writes)
        retry_ids = {
            str(track_id) for track_id in
            (_track_id_value(record) for record in failed_records)
            if track_id is not None
        }
        retry_ids.update(str(track_id) for track_id in missing_ids)
        retry_records = [
            record for record in records
            if _track_id_value(record) is not None
            and str(_track_id_value(record)) in retry_ids
        ]
        if retry_records:
            logging.warning(
                f"{log_prefix}Deferred {len(retry_records)} track-head update(s) "
                f"for retry ({len(failed_records)} rejected, {len(missing_ids)} missing)")
        return retry_records
    except Exception as err:
        logging.warning(
            f"{log_prefix}Track-head update failed; deferring "
            f"{len(records)} record(s) for retry: {err}")
        return records


def collect_finished_track_update_tasks(
        tasks: List[Any], label: str,
        pending_by_track: Dict[str, Dict[str, Any]],
        log_prefix: str = '') -> List[Any]:
    """Move failed records from completed head-update tasks to a retry buffer."""
    if not tasks:
        return []
    still_pending = []
    completed = 0
    for task in tasks:
        if not task.done():
            still_pending.append(task)
            continue
        completed += 1
        try:
            for record in task.result() or []:
                track_id = _track_id_value(record)
                if track_id is not None:
                    pending_by_track[str(track_id)] = record
        except Exception as exc:
            logging.warning(
                f"{log_prefix}Background {label} task failed; records will be "
                f"retried by the next batch: {exc}")
    if completed:
        logging.info(
            f"{log_prefix}Completed {completed} background {label} task(s); "
            f"still_running={len(still_pending)}")
    return still_pending


async def write_entity_updates_with_create_fallback(
    update_records: List[dict],
    dataset_name: str,
    wc,
    *,
    update_chunk_size: int,
    create_chunk_size: int,
    create_records: Optional[List[dict]] = None,
    key_field: str = 'entityId',
    label: str = '',
    token_refresher: Optional[Callable] = None,
    max_concurrent_writes: Optional[int] = None,
) -> list:
    """Update entity-dataset records, re-creating any the server reports not-found.

    Entity-type datasets reject updates whose primary key does not yet exist
    (the server logs ResourceNotFoundException and silently drops the record
    from the update response). This writes via update_entity_record_batch_by_name,
    detects the dropped primary keys, and re-creates them via
    write_record_batch_by_name so the write is not lost.

    Args:
        update_records: records to send through the entity update path.
        create_records: optional full records (keyed by *key_field*) to use when
            re-creating. Supply this when the update payload omits fields needed
            to create a valid record (e.g. identity.* stripped from updates), so
            re-created entities are complete rather than degraded. Falls back to
            the matching update_records when omitted.
        key_field: primary-key field used to match dropped records (entityId).

    Returns:
        list: records that still failed after the create retry. Empty on success.
    """
    if not update_records:
        return []
    failed, dropped_ids = await write_batch_chunked(
        update_records, dataset_name, wc.update_entity_record_batch_by_name,
        update_chunk_size, response_mode='entity_update', label=label,
        token_refresher=token_refresher,
        max_concurrent_writes=max_concurrent_writes)
    if not dropped_ids:
        return failed
    source = create_records if create_records is not None else update_records
    to_create = [r for r in source if r.get(key_field) in dropped_ids]
    if not to_create:
        return failed
    logging.info(
        f"{label}Re-creating {len(to_create)} not-found record(s) in {dataset_name} via create path"
    )
    created_failed, _ = await write_batch_chunked(
        to_create, dataset_name, wc.write_record_batch_by_name,
        create_chunk_size, label=label, token_refresher=token_refresher,
        max_concurrent_writes=max_concurrent_writes)
    # The silent drops were re-created; keep only genuine create failures.
    failed = [f for f in failed if f.get(key_field) not in dropped_ids]
    failed.extend(created_failed)
    return failed


# --- Signal handling ---

def terminate(sig: int, frame: Any) -> None:
    logging.info(f"\nReceived signal {sig}. Killing process group. \n")
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


# --- Shared record-routing utilities ---

def get_current_timestamp_string(offset_hours: int = 0) -> str:
    """Return the current UTC time, optionally offset, in schema format."""
    datetime_format_string = "%Y-%m-%dT%H:%M:%S.%f"
    timestamp = dt.now(tz=tz.utc) + timedelta(hours=offset_hours)
    return timestamp.strftime(datetime_format_string)[:-3] + 'Z'


def stable_shard(value: Any, shard_count: int) -> int:
    """Map a value to a deterministic shard independent of PYTHONHASHSEED."""
    digest = hashlib.md5(
        str(value).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return int(digest, 16) % shard_count


def last_record_by_track(records: List[dict]) -> List[dict]:
    """Retain the last record for each scalar or UUID-wrapped track ID."""
    latest: Dict[str, dict] = {}
    for record in records:
        track_id = record.get("trackId")
        if isinstance(track_id, dict):
            track_id = track_id.get("uuid")
        if track_id is not None:
            latest[str(track_id)] = record
    return list(latest.values())


def coalesce_existing_head_updates(
    heads: List[dict],
    last_emit_by_track: Dict[str, float],
    interval_seconds: float,
    label: str,
    log_prefix: str = "",
) -> List[dict]:
    """Rate-limit existing track-head updates while keeping the latest batch."""
    if not heads or interval_seconds <= 0:
        return heads
    now = time.monotonic()
    kept = []
    for head in heads:
        track_id = head.get("trackId")
        if isinstance(track_id, dict):
            track_id = track_id.get("uuid")
        if track_id is None:
            continue
        track_id = str(track_id)
        if now - last_emit_by_track.get(track_id, -1e30) >= interval_seconds:
            kept.append(head)
            last_emit_by_track[track_id] = now
    coalesced = len(heads) - len(kept)
    if coalesced:
        logging.info(
            "%sCoalesced existing %s track-head update(s); "
            "coalesced=%d kept=%d interval=%gs",
            log_prefix,
            label,
            coalesced,
            len(kept),
            interval_seconds,
        )
    return kept


def skip_head_preload(config: dict) -> bool:
    """Return whether configuration disables track-head preloading."""
    configured = config.get(
        "skip_head_preload", os.getenv("CRUCIBLE_SKIP_HEAD_PRELOAD", "false")
    )
    return str(configured).strip().lower() in {"1", "true", "yes", "on"}


def load_all_records_for_preload(
    dataset_name: str,
    config: dict,
    read_controller: Any,
    authenticator: Any,
    limit_config_key: str,
) -> List[dict]:
    """Load the newest records used to initialize tracker or fuser shards."""
    try:
        read_controller.token = authenticator.get_token()
        limit = int(
            config.get(
                limit_config_key,
                os.getenv("CRUCIBLE_HEAD_PRELOAD_LIMIT", "100000"),
            )
        )
        records = read_controller.search(
            f"SELECT * FROM {dataset_name} "
            f"ORDER BY {dataset_name}.crucibleHeader.updatedDate DESC LIMIT {limit}"
        ) or []
        if len(records) >= limit:
            logging.warning(
                "Head preload for %s hit the %d-row LIMIT; the oldest heads "
                "were dropped. Raise %s / CRUCIBLE_HEAD_PRELOAD_LIMIT.",
                dataset_name,
                limit,
                limit_config_key,
            )
        return records
    except Exception as error:
        logging.warning(
            "Head preload: search for %s failed (%s); shards will start with "
            "no preloaded heads",
            dataset_name,
            error,
        )
        return []

# --- Coordinate transform utilities ---

def enu_to_ecef_rotation_matrix(lat: float, lon: float) -> np.ndarray:
    """Rotation matrix from local ENU (East, North, Up) frame to ECEF.
    Columns correspond to [East, North, Up] basis vectors in ECEF.
    Parameters:
    lat (float): Latitude in radians.
    lon (float): Longitude in radians.
    Returns:
    np.ndarray: A 3x3 rotation matrix that transforms [E, N, U] to ECEF.
    """
    clon = np.cos(lon)
    slon = np.sin(lon)
    clat = np.cos(lat)
    slat = np.sin(lat)
    return np.array([[-slon, -slat * clon, clat * clon],
                     [ clon, -slat * slon, clat * slon],
                     [ 0.,    clat,         slat]])

def ecef_to_enu_rotation_matrix(lat: float, lon: float) -> np.ndarray:
    """Rotation matrix from ECEF to local ENU (East, North, Up) frame.
    Rows correspond to [East, North, Up] components.
    Parameters:
    lat (float): Latitude in radians.
    lon (float): Longitude in radians.
    Returns:
    np.ndarray: A 3x3 rotation matrix that transforms ECEF to [E, N, U].
    """
    clon = np.cos(lon)
    slon = np.sin(lon)
    clat = np.cos(lat)
    slat = np.sin(lat)
    return np.array([[-slon,         clon,         0.],
                     [-slat * clon, -slat * slon,  clat],
                     [ clat * clon,  clat * slon,  slat]])

def enu_to_ecef_vector(lat: float, lon: float, v_east: float, v_north: float, v_up: float) -> np.ndarray:
    """
    Rotate a vector from local ENU frame to ECEF coordinates.
    Parameters:
    lat (float): Latitude in radians.
    lon (float): Longitude in radians.
    v_east (float): Eastward component of the vector.
    v_north (float): Northward component of the vector.
    v_up (float): Upward component of the vector.
    Returns:
    np.ndarray: A 3-element vector in ECEF coordinates.
    """

    M = enu_to_ecef_rotation_matrix(lat, lon)
    v = np.dot(M, np.array([v_east, v_north, v_up]))
    return v

def ecef_to_enu_vector(lat: float, lon: float, v_x: float, v_y: float, v_z: float) -> Tuple[float, float, float]:
    """
    Rotate a vector from ECEF coordinates to local ENU frame.
    Parameters:
    lat (float): Latitude in radians.
    lon (float): Longitude in radians.
    v_x (float): X component of the vector.
    v_y (float): Y component of the vector.
    v_z (float): Z component of the vector.
    Returns:
    tuple: (v_east, v_north, v_up) in local ENU coordinates.
    """

    M = ecef_to_enu_rotation_matrix(lat, lon)
    v = np.dot(M, np.array([v_x, v_y, v_z]))
    v_east = v[0]
    v_north = v[1]
    v_up = v[2]
    return v_east, v_north, v_up

def rotate_covariance_matrix_enu_to_ecef(lat: float, lon: float, cov: np.ndarray) -> np.ndarray:
    """Rotate a covariance matrix in [E, N, U] order to ECEF.
    Parameters:
    lat (float): Latitude in radians.
    lon (float): Longitude in radians.
    cov (np.ndarray): A 3x3 covariance matrix in local ENU coordinates.
    Returns:
    np.ndarray: A 3x3 covariance matrix in ECEF coordinates.
    """
    M = enu_to_ecef_rotation_matrix(lat, lon)
    return np.dot(M, np.dot(cov, M.T))

def rotate_covariance_matrix_ecef_to_enu(lat: float, lon: float, cov: np.ndarray) -> np.ndarray:
    """Rotate a covariance matrix from ECEF to [E, N, U] order.
    Parameters:
    lat (float): Latitude in radians.
    lon (float): Longitude in radians.
    cov (np.ndarray): A 3x3 covariance matrix in ECEF coordinates.
    Returns:
    np.ndarray: A 3x3 covariance matrix in local ENU coordinates.
    """
    M = ecef_to_enu_rotation_matrix(lat, lon)
    return np.dot(M, np.dot(cov, M.T))

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

def covariance_matrix_to_uncertainty_ellipse(cov: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a covariance matrix to uncertainty ellipse parameters.
    Parameters:
    cov (np.ndarray): A 2x2 or 3x3 covariance matrix.
    Returns:
    float: Length of the major axis of the ellipse.
    float: Length of the minor axis of the ellipse.
    float: Orientation angle of the ellipse in radians.
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
                    perspective_scripts[key] != config.get(key):
                    logging.error(" ")
                    logging.error(" ")
                    logging.error('-' * 50)
                    logging.error(f"Error: more than one {key} has been defined ")
                    logging.error(f"    for perspective {config.get('perspective')}")
                    logging.error('-' * 50)
                    logging.error(" ")
                    logging.error(" ")
                    # sleep for a few seconds to allow the error message to be read
                    time.sleep(5)
                perspective_scripts[key]=config.get(key)
                # e.g. perspective_scripts['custom_functions_script_name']='custom_functions'
                

    # dict of scripts and their associated filenames
    script_filenames = \
    {'custom_functions_script_name': 'custom_functions.py',
     'unit_conversions_script_name': 'unit_conversions.py'}



    # script dict is a dictionary of script names and their bodies
    # from the Stream_Manager_Functions dataset
    script_dict = {}
    {script_dict.update({script.get('script_name'): script.get('script_body')}) for script in script_configs}

    # import the scripts
    for script_name, script_body in script_dict.items():
        if script_name in perspective_scripts.values():
            #get the key for the script name
            script_name_key = list(perspective_scripts.keys())[list(perspective_scripts.values()).index(script_name)]

            module_name =  script_filenames[script_name_key].replace('.py', '')
            mod = _load_module_from_source(module_name, script_body)
            caller_globals[module_name] = mod
            _script_sources[module_name] = script_body

            logging.info(f"Imported {module_name} from Entity_Stream_Manager_Functions")


# --- Configuration loading ---

def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return False


def buffer_latest_records_by_track(pending: Dict[str, dict], records: List[dict]) -> int:
    """Buffer the latest record for each trackId and return replacements made."""
    coalesced = 0
    for record in records:
        track_id = record.get('trackId')
        if isinstance(track_id, dict):
            track_id = track_id.get('uuid')
        if track_id is None:
            continue
        key = str(track_id)
        if key in pending:
            coalesced += 1
        pending[key] = record
    return coalesced


def _apply_disable_all_other_datasets(config_list: List[dict]) -> None:
    """Honor the single-feed focus flag used by ad-hoc deployments.

    Accept both the canonical ``disable_all_other_datasets`` key and the legacy
    misspelling ``disable_all_other_datsets``. At most one datafeed config may
    enable focus mode; when set, every other datafeed config is marked disabled
    before the rest of config validation/routing runs.
    """
    focus_configs = [
        config for config in config_list
        if _config_bool(config.get('disable_all_other_datasets'))
        or _config_bool(config.get('disable_all_other_datsets'))
    ]
    if len(focus_configs) > 1:
        origins = [str(config.get('origin_dataset', '<unknown>')) for config in focus_configs]
        raise ValueError(
            "Only one Entity Stream Manager configuration row may set "
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
        "entity datafeed configs in this perspective")

def find_and_validate_configs(stream_manager_perspective: str, include_scripts: bool = True, rc_instance: Optional[Any] = None, caller_globals: Optional[dict] = None) -> dict:
    """
    Args:
        stream_manager_perspective (str): name of perspective in Stream Manager Configuration
        include_scripts (bool): whether to import script files
        rc_instance: ReadController instance (required)
        caller_globals (dict): the caller's globals() dict for dynamic script imports

    Returns:
        dict: {'datafeed_configs': List[dict], 'perspective_config': dict, 'script_configs': list}
    """
    _rc = rc_instance
    if _rc is None:
        raise ValueError("rc_instance must be provided")
    
    try:
        config_list=_rc.search(f"select * from {stream_manager_config_dataset_name} " 
                            f"where {stream_manager_config_dataset_name}.perspective = '{stream_manager_perspective}'")
    except Exception as e:
        logging.error('-' * 50)
        logging.error(f"Error retrieving Entity Stream Manager configuration for {stream_manager_perspective}: {e}")
        logging.error('-' * 50)
        raise ValueError(f"Error retrieving Entity Stream Manager configuration for {stream_manager_perspective}: {e}")
    
    if len(config_list) == 0:
        logging.error('-' * 50)
        logging.error(f" No Entity Stream Manager configuration found for {stream_manager_perspective}")
        logging.error('-' * 50)
        raise ValueError(f"No Entity Stream Manager configuration found for {stream_manager_perspective}")

    # Separate perspective config from entity transformer configs.
    # The perspective config record has origin_dataset starting with PERSPECTIVE_CONFIG_PREFIX
    # and holds shared fields (dataset names, script names) for the perspective.
    perspective_config = {}
    datafeed_configs = []
    for config in config_list:
        origin_dataset = config.get('origin_dataset', '')
        if origin_dataset.startswith(PERSPECTIVE_CONFIG_PREFIX):
            perspective_config = config
            logging.info(f"Found perspective config: {origin_dataset}")
        else:
            datafeed_configs.append(config)

    _apply_disable_all_other_datasets(datafeed_configs)

    # Merge perspective config fields into each transformer config.
    # Transformer-specific values take precedence over perspective defaults.
    if not perspective_config:
        logging.error(f"No perspective config record found for {stream_manager_perspective}. "
                        "Expecting all config fields in each transformer config.")
        raise ValueError(f"No perspective config record found for {stream_manager_perspective}. "
                         "Expecting all config fields in each transformer config.")
    for config in datafeed_configs:
        if 'origin_to_destination_mapping' not in config.keys():
            logging.warning('-' * 50)
            logging.warning(f" For {config.get('origin_dataset')}: ")
            logging.warning("origin_to_destination_mapping is missing")
            logging.warning('-' * 50)
            time.sleep(5)
        mapping = config.get('origin_to_destination_mapping')
        # Check for duplicate mappings to the same destination column,
        # duplicate destination columns are allowed
        destination_columns = [el.get('destination_column') for el in mapping if 'origin_column' in el]
        if len(destination_columns) != len(set(destination_columns)):
            raise ValueError("Stream Manager Configuration: duplicate destination columns are not allowed in the origin_to_destination_mapping")
        # Ensure "identity..." is in one of the destination columns
        if not any("identity." in col for col in destination_columns):
            raise ValueError('Stream Manager Configuration: no destination column contains "identity." in origin_to_destination_mapping')

        # The tracker reads raw Report_Events and creates deterministic trackIds.
        required_keys = [
            'entity_management_event_dataset',
            'report_event_dataset',
            'principal_track_event_dataset',
            'principal_track_head_dataset',
        ]
        for key in required_keys:
            if key not in perspective_config.keys():
                raise ValueError(f"Stream Manager Configuration: Missing required key '{key}' in '{perspective_config.get('origin_dataset')}' configuration.")

    script_configs=_rc.search("select * from " + stream_manager_functions_dataset_name)
    
    disabled_config_list = [config for config in datafeed_configs if 'disabled' in config.keys() and config.get('disabled') is True]
    for config in disabled_config_list:
        logging.warning('-' * 50)
        logging.warning(f"Configuration for {config.get('origin_dataset')} in perspective {config.get('perspective')} is disabled. Skipping...")
        logging.warning('-' * 50)
    config_list = [config for config in datafeed_configs if 'disabled' not in config.keys() or config.get('disabled') is False]
    if include_scripts:
        import_script_files([perspective_config] + config_list, script_configs, caller_globals=caller_globals)

    # Merge perspective config (dataset names, script names) into each datafeed config
    for config in config_list:
        for key, value in perspective_config.items():
            if key not in config:
                config[key] = value
        # Store script sources so spawned child processes can re-import them
        config['_script_sources'] = _script_sources.copy()

    # Existing component/principal head writes share one perspective-level
    # coalescing interval, matching the object correlators.
    head_update_interval = perspective_config.get('head_update_interval_seconds')
    if head_update_interval is not None:
        for config in config_list:
            config['head_update_interval_seconds'] = head_update_interval

    # Set default for batch_write_chunk_size if not provided by perspective_config
    for d in config_list + [perspective_config]:
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
        # num_fusion_workers is an OPTIONAL perspective-level key (number of
        # parallel fusion worker processes). It is intentionally NOT in
        # required_keys, so it may be omitted: the generic perspective_config
        # merge above copies it onto each datafeed config when present, and the
        # fuser falls back to its NUM_FUSION_WORKERS default when absent.
        # Normalize to int here because Crucible returns config values as strings.
        if d.get('num_fusion_workers') is not None:
            d['num_fusion_workers'] = int(d['num_fusion_workers'])
        # skip_head_preload is an optional perspective-level key. The generic
        # perspective_config merge above copies it onto each datafeed config;
        # tracker/fuser readers accept bools or truthy strings.

    return {'datafeed_configs': config_list, 'perspective_config': perspective_config, 'script_configs': script_configs}


# --- Supersede map utilities ---

def _nested_record_value(record: dict, path: str, default=None):
    current = record
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def build_supersede_map(management_events: List[dict],
                        init_supersede_map: Optional[Dict[str, str]] = None
                        ) -> Dict[str, str]:
    """
    Build a trackId supersede map from SUPERSEDE, DELETE, and RESTORE events.
    For SUPERSEDE: maps to their latest supersededBy, following chains.
    For DELETE: maps to None to indicate the entity has been deleted.
    For RESTORE: removes the entity from the map (undoes supersession/deletion).
    Also follows chains starting from init_supersede_map.
    
    Args:
        management_events (List[dict]): Nested entity management event records.
        init_supersede_map (Dict[str, str]): Initial supersede map to build upon
        
    Returns:
        Dict[str, str]: Mapping of superseded trackId -> superseding trackId (or None if deleted)
    """
    if not management_events:
        return dict(init_supersede_map or {})
    supersedes = dict(init_supersede_map or {})
    latest = {}
    ordered = sorted(
        management_events,
        key=lambda record: str(_nested_record_value(
            record, 'crucibleHeader.updatedDate', '')),
        reverse=True)
    for event in ordered:
        track_id = _nested_record_value(event, 'trackId')
        if track_id is not None:
            latest.setdefault(str(track_id), event)
    for track_id, event in latest.items():
        action = _nested_record_value(event, 'action')
        if action == 'SUPERSEDE':
            superseded_by = _nested_record_value(event, 'supersededBy')
            if isinstance(superseded_by, (float, np.floating)) and np.isnan(superseded_by):
                superseded_by = None
            supersedes[track_id] = (
                str(superseded_by) if superseded_by is not None else None)
        elif action == 'DELETE':
            supersedes[track_id] = None
        elif action == 'RESTORE':
            supersedes.pop(track_id, None)
    
    # Resolve every trackId through the complete supersede chain.
    result = {}
    for entity_id in supersedes:
        current = entity_id
        next_superseded = supersedes.get(current)

        if next_superseded is not None and not isinstance(next_superseded, str):
            next_superseded = None
        
        # Follow the chain until we reach a supersededBy that is not in the mapping
        # Add cycle detection to prevent infinite loops
        visited = {entity_id}
        while next_superseded in supersedes:
            if next_superseded in visited:
                # Circular reference detected - break the loop
                break
            visited.add(next_superseded)
            current = next_superseded
            next_superseded = supersedes.get(current)
            if next_superseded is not None and not isinstance(next_superseded, str):
                next_superseded = None
        
        # Determine final mapping
        if next_superseded is None:
            # Chain ends in deletion (or entity itself was deleted) -> None
            result[entity_id] = None
        else:
            # Chain ends at a valid entity
            result[entity_id] = next_superseded

    return result


def _timestamp_offset_seconds(timestamp: Any) -> int:
    if isinstance(timestamp, dt):
        parsed = timestamp
    else:
        try:
            parsed = dt.fromisoformat(str(timestamp).replace('Z', '+00:00'))
        except (TypeError, ValueError):
            parsed = dt.now(tz.utc) - timedelta(seconds=5)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz.utc)
    age_seconds = (dt.now(tz.utc) - parsed).total_seconds()
    return max(1, int(np.ceil(age_seconds)))


def _latest_management_timestamp(
        management_events: List[dict]) -> Optional[dt]:
    timestamps = []
    for event in management_events or []:
        value = (_nested_record_value(event, 'crucibleHeader.updatedDate')
                 or _nested_record_value(event, 'updatedDate'))
        if value is None:
            continue
        try:
            parsed = dt.fromisoformat(str(value).replace('Z', '+00:00'))
        except (TypeError, ValueError):
            continue
        timestamps.append(parsed if parsed.tzinfo else parsed.replace(tzinfo=tz.utc))
    return max(timestamps) if timestamps else None


def _empty_management_cursor() -> dt:
    return dt.now(tz.utc) - timedelta(seconds=5)


def get_management_events(
        config: Dict[str, Any], rc_instance: Optional[Any] = None,
        auth_instance: Optional[Any] = None,
        actions: Tuple[str, ...] = ('SUPERSEDE', 'DELETE', 'RESTORE'),
        since_timestamp: Optional[Any] = None, lookback_days: int = 30,
        limit: int = 10000) -> Tuple[List[dict], Optional[dt]]:
    """Read all matching entity-management events without truncating state."""
    if rc_instance is None:
        raise ValueError("rc_instance must be provided")
    if auth_instance is None:
        raise ValueError("auth_instance must be provided")

    management_dataset = config.get('entity_management_event_dataset')
    if not management_dataset:
        return [], since_timestamp

    rc_instance.token = auth_instance.get_token()
    action_predicate = ' OR '.join(
        f"{management_dataset}.action = '{action}'" for action in actions)
    if since_timestamp is None:
        time_predicate = (
            f"{management_dataset}.crucibleHeader.updatedDate > "
            f"TIMESTAMP_OFFSET(-{int(lookback_days)},'days')")
    else:
        since_seconds = _timestamp_offset_seconds(since_timestamp)
        time_predicate = (
            f"{management_dataset}.crucibleHeader.updatedDate > "
            f"TIMESTAMP_OFFSET(-{since_seconds},'seconds')")

    page_size = max(1, int(limit))
    pages: List[List[dict]] = []
    offset = 0
    while True:
        management_sql_query = f"""
            SELECT * FROM {management_dataset}
            WHERE ({action_predicate})
            AND {time_predicate}
            ORDER BY {management_dataset}.crucibleHeader.updatedDate ASC,
                     {management_dataset}.crucibleHeader.uuid ASC
            OFFSET {offset} ROWS FETCH NEXT {page_size} ROWS ONLY
        """
        try:
            page = rc_instance.search(
                management_sql_query, auto_backtick=False)
        except Exception as exc:
            error_msg = str(exc)
            if (
                "Table 'crucibleHeader' not found" in error_msg
                or "Error processing query" in error_msg
                or "Bad Request" in error_msg
            ):
                logging.warning(
                    "Empty Entity Management Events dataset, returning no management events")
            else:
                logging.error(
                    f"Error occurred while fetching management events: {exc}")
            break

        if not page:
            break
        pages.append(page)
        if len(page) < page_size:
            break
        offset += page_size

    management_events = [record for page in pages for record in page]
    latest = _latest_management_timestamp(management_events)
    if latest is None and since_timestamp is None:
        latest = _empty_management_cursor()
    elif latest is None:
        latest = since_timestamp
    if len(pages) > 1:
        logging.info(
            f"entity management event query loaded {len(management_events)} row(s) "
            f"across {len(pages)} page(s)")
    return management_events, latest


def get_supersede_map(
        config: Dict[str, Any], rc_instance: Optional[Any] = None,
        auth_instance: Optional[Any] = None,
        return_latest_timestamp: bool = False) -> Dict[str, str]:
    """
    Returns a dictionary of superseded entities from the entity management event dataset.
    Queries for SUPERSEDE, DELETE, and RESTORE actions and builds a chain-resolved map
    via build_supersede_map.
    
    Args:
        config (Dict[str, Any]): Configuration containing dataset names
        rc_instance: ReadController instance (required).
        auth_instance: Authenticator instance (required).
        
    Returns:
        Dict[str, str]: Mapping of superseded trackId -> superseding trackId (or None if deleted)
    """
    _rc = rc_instance
    _auth = auth_instance
    if _rc is None:
        raise ValueError("rc_instance must be provided")
    if _auth is None:
        raise ValueError("auth_instance must be provided")

    supersede_map = {}

    management_dataset = config.get('entity_management_event_dataset')
    if not management_dataset:
        logging.warning("No entity management event dataset configured")
        return supersede_map

    try:
        management_events, latest_timestamp = get_management_events(
            config, rc_instance=_rc, auth_instance=_auth)
    except Exception as exc:
        logging.error(f"Error occurred while fetching management events: {exc}")
        management_events = []
        latest_timestamp = None

    if management_events:
        supersede_map = build_supersede_map(management_events)

    logging.info(f"get_supersede_map returning {len(supersede_map)} entries")
    if return_latest_timestamp:
        return supersede_map, latest_timestamp
    return supersede_map


def get_duplicate_protected_ids(
        config: Dict[str, Any], rc_instance: Any, auth_instance: Any,
        restore_cooldown_seconds: int = 300,
        supersede_map: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Return track IDs the duplicate detector must temporarily ignore."""
    if supersede_map is None:
        supersede_map = get_supersede_map(
            config, rc_instance=rc_instance, auth_instance=auth_instance)
    protected_ids = set(supersede_map)
    protected_ids.update(
        value for value in supersede_map.values() if value is not None)

    cooldown = int(config.get(
        'restore_cooldown_seconds', restore_cooldown_seconds))
    management_dataset = config.get('entity_management_event_dataset')
    if cooldown <= 0 or not management_dataset:
        return protected_ids

    try:
        since = dt.now(tz.utc) - timedelta(seconds=cooldown)
        restore_events, _ = get_management_events(
            config, rc_instance=rc_instance, auth_instance=auth_instance,
            actions=('RESTORE',), since_timestamp=since)
        restored_ids = {
            str(track_id) for event in restore_events
            if (track_id := _nested_record_value(event, 'trackId')) is not None}
        protected_ids.update(restored_ids)
        if restored_ids:
            logging.info(
                f"Protected {len(restored_ids)} recently restored track(s) "
                f"for {cooldown}s")
    except Exception as exc:
        logging.warning(f"Could not load recent RESTORE actions: {exc}")
    return protected_ids
