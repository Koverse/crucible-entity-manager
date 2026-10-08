#!/usr/bin/env python
"""
Object & Track Age-Off for the object pipeline.

Periodically deletes STALE objects and track heads so the live datasets (and the
transformer's per-worker object cache) don't grow without bound.

Objects
    For each environment, delete objects whose ``entityStatus != 'CONFIRMED'`` and
    whose ``crucibleHeader.updatedDate`` is older than a per-environment threshold,
    using the batch ``delete_entity_record_by_query`` API. Unless disabled with
    the launcher/CLI option, objects are first marked ``DROPPED`` to prune caches.
    CONFIRMED objects are NEVER deleted. With DROPPED marking enabled, delete
    responses are verified because the service may commit before returning an
    error, transient pages are retried, and a ``DELETE`` management event is written
    for every object confirmed absent. With marking disabled, direct query deletes
    skip UUID listing, post-delete verification, and management-event reconstruction.

Superseded objects
    The superseded-object dataset (``superseded_object_dataset``, default
    ``<perspective>_SupersededObjects``) is aged off with the SAME per-environment
    thresholds. Those records have already left the live pipeline, so they are
    optionally flagged ``DROPPED`` and deleted, but no management events are written
    for them.

Tracks
    Component + principal track heads whose ``trackUpdatedTimestamp`` (the last
    genuine observation time) is older than the same per-environment threshold are
    deleted too. Track heads carry no management events, so none are written.
    (The pipeline does not persist a ``stale`` field — it is dropped before write —
    so ``trackUpdatedTimestamp`` is the staleness field.)

Thresholds are per-environment (see DEFAULT_AGE_OFF_HOURS) and can be overridden
by the launcher or CLI.

Launch (button / correlator style — runs as a periodic daemon):
    from object_manager.object_age_off import run
    run('Live_POV', log_level='INFO')

CLI:
    python object_age_off.py Live_POV [--once] [--dry-run] [--interval 15m] [--log INFO]
"""
import argparse
import json
import logging
import os
import re
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    from cruciblelib import authenticator, read_controller, write_controller, log_utils
except ImportError:
    from . import authenticator, read_controller, write_controller, log_utils

try:
    from object_utils import find_and_validate_configs
except ImportError:
    from .object_utils import find_and_validate_configs


# Per-environment staleness thresholds in HOURS. An object/track in a given
# environment is aged off once it has not been updated for this long. CONFIRMED
# objects are exempt. Override through run(age_hours=...) or the CLI --age-<env>
# flags (for example --age-air 4 or --age-sea-surface 48).
DEFAULT_AGE_OFF_HOURS: Dict[str, float] = {
    'AIR': 4,
    'GROUND': 24,
    'SEA_SURFACE': 48,       # 2 days
    'SEA_SUBSURFACE': 48,    # 2 days
    'SPACE': 168,            # 7 days
    'UNKNOWN': 24,
}
_DEFAULT_INTERVAL = '15m'
_MGMT_EVENT_CHUNK = 5000     # DELETE events written per batch
_DELETE_TIMEOUT = (30, 120)  # (connect, read) for the bulk delete call
# delete_entity_record_by_query DELETES the query's results and respects LIMIT,
# but a single large delete 504s at the gateway (Fluo deletes run ~27 rows/s and
# the server-side gateway times out ~60s). So we page deletes with a small LIMIT
# (500 ~= 19s, safely under the gateway) and cap deletions per target per cycle so
# a periodic run never overruns its interval. Both are launcher/CLI-overridable.
_DEFAULT_BATCH_SIZE = 500          # rows deleted per delete_entity_record_by_query call
_DEFAULT_MAX_PER_TARGET = 20000    # max rows deleted per (env, dataset) per cycle
_DEFAULT_OBJECT_RETRIES = 3        # retries for transient object select/delete failures

# Controllers (initialized in _init_controllers)
rc = None
wc = None
auth = None


class _AgeOffProgress:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.lock = threading.Lock()
        self.data = {
            'status': 'starting',
            'deleted': 0,
            'managementEvents': 0,
            'markedDropped': 0,
            'objectsConfirmed': 0,
            'objectsUnconfirmed': 0,
            'componentTrackHeadsConfirmed': 0,
            'componentTrackHeadsUnconfirmed': 0,
            'principalTrackHeadsConfirmed': 0,
            'principalTrackHeadsUnconfirmed': 0,
            'updatedAt': time.time(),
        }
        self._write()

    def update(self, **values: Any) -> None:
        if not self.path:
            return
        with self.lock:
            for key, value in values.items():
                if key not in ('status', 'updatedAt', 'error'):
                    self.data[key] += int(value)
                else:
                    self.data[key] = value
            self.data['updatedAt'] = time.time()
            self._write()

    def _write(self) -> None:
        if not self.path:
            return
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, 'w', encoding='utf-8') as handle:
                json.dump(self.data, handle)
            os.replace(tmp, self.path)
        except Exception as exc:
            logging.warning(f"age_off: progress write failed: {exc}")

    def record_delete_outcome(self, target: Optional[str], confirmed: int = 0,
                              unconfirmed: int = 0) -> None:
        values: Dict[str, int] = {}
        if confirmed:
            values['deleted'] = confirmed
        if target:
            if confirmed:
                values[f'{target}Confirmed'] = confirmed
            if unconfirmed:
                values[f'{target}Unconfirmed'] = unconfirmed
        if values:
            self.update(**values)


def _init_controllers() -> None:
    """Instantiate auth + read + v2 write controllers.

    A v2 WriteController is required for delete_entity_record_by_query; it also
    serves the management-event writes (upsert/write_record_batch_by_name).
    """
    global rc, wc, auth
    auth = authenticator.Authenticator(
        client_id=os.environ['CRUCIBLE_CLIENT_ID'],
        client_secret=os.environ['CRUCIBLE_CLIENT_SECRET'],
        grant_type=os.environ['CRUCIBLE_GRANT_TYPE'],
        token_url=os.environ['CRUCIBLE_TOKEN_URL'],
        log_level=logging.INFO,
        tls_cert=os.environ['CRUCIBLE_CERT_PATH'],
    )
    rc = read_controller.ReadController()
    wc = write_controller.WriteController(version='v2')


def _refresh_tokens() -> None:
    tok = auth.get_token()
    rc.token = tok
    wc.token = tok


def _parse_interval(s: str) -> int:
    """'30m' / '45s' / '2h' -> seconds."""
    s = str(s).strip().lower()
    unit, value = s[-1], s[:-1]
    mult = {'s': 1, 'm': 60, 'h': 3600}.get(unit)
    if mult is None or not value:
        raise ValueError(f"age_off: bad interval {s!r} (use e.g. 30m, 45s, 2h)")
    return int(float(value) * mult)


def _is_true(v: Any) -> bool:
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _age_off_hours(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Per-environment thresholds from defaults plus launcher/CLI overrides."""
    hours = dict(DEFAULT_AGE_OFF_HOURS)
    for env, val in (overrides or {}).items():
        if val is None:
            continue
        try:
            hours[str(env).upper()] = float(val)
        except (TypeError, ValueError):
            logging.warning(f"age_off: ignoring non-numeric override {env}={val!r}")
    return hours


def _extract_uuid(rec: dict) -> Optional[str]:
    oid = rec.get('objectId')
    if isinstance(oid, dict):
        oid = oid.get('uuid')
    if oid is None:
        oid = rec.get('objectId.uuid')
    return str(oid) if oid else None


def _extract_edh(rec: dict) -> list:
    edh = rec.get('edhControlSet')
    return edh if isinstance(edh, list) and edh else ['CLS:U']


def _write_delete_events(mgmt_dataset: str, deleted_records: List[dict],
                         progress: Optional[_AgeOffProgress] = None) -> int:
    """Write a DELETE action to the object management event dataset for each
    object Crucible confirmed deleted. Returns the number of events written."""
    events = []
    for rec in deleted_records:
        uuid = _extract_uuid(rec)
        if not uuid:
            continue
        events.append({
            'objectId': uuid,
            'action': 'DELETE',
            'source': 'MANUAL',
            'edhControlSet': _extract_edh(rec),
        })
    written = 0
    for i in range(0, len(events), _MGMT_EVENT_CHUNK):
        chunk = events[i:i + _MGMT_EVENT_CHUNK]
        try:
            wc.token = auth.get_token()
            wc.write_record_batch_by_name(mgmt_dataset, chunk)
            written += len(chunk)
            if progress:
                progress.update(managementEvents=len(chunk))
        except Exception as e:
            logging.error(f"age_off: failed writing {len(chunk)} DELETE events "
                          f"to {mgmt_dataset}: {e}")
    return written


def _count(dataset: str, where: str) -> Optional[int]:
    try:
        rc.token = auth.get_token()
        df = rc.search(f"select count(*) as n from {dataset} where {where}",
                       format='dataframe', auto_backtick=True)
        return int(df.iloc[0, 0]) if df is not None and len(df) else 0
    except Exception as e:
        logging.warning(f"age_off: count failed on {dataset}: {e}")
        return None


def _is_empty_match_400(exc: Exception) -> bool:
    """delete_entity_record_by_query returns HTTP 400 when the query matches NO
    rows; detect that so paged deletion stops cleanly instead of erroring."""
    if getattr(exc, 'status_code', None) == 400:
        return True
    s = str(exc).lower()
    return 'status 400' in s or '400 bad request' in s


def _derive_count_query(base_query: str) -> Optional[str]:
    """Rewrite 'SELECT <cols> FROM ...' -> 'SELECT count(*) as n FROM ...' (best effort)."""
    cq, nsub = re.subn(r'^\s*select\s+.*?\s+from\s+', 'SELECT count(*) as n FROM ',
                       base_query, count=1, flags=re.IGNORECASE | re.DOTALL)
    return cq if nsub else None


def _count_remaining_matches(base_query: str) -> Optional[int]:
    count_query = _derive_count_query(base_query)
    if not count_query:
        return None
    try:
        rc.token = auth.get_token()
        df = rc.search(count_query, format='dataframe', auto_backtick=True)
        return int(df.iloc[0, 0]) if df is not None and len(df) else 0
    except Exception as e:
        logging.warning(f"age_off: delete outcome verification count failed: {e}")
        return None


def _delete_in_batches(base_query: str, cap: int, batch: int,
                       mgmt_dataset: Optional[str] = None,
                       expected_count: Optional[int] = None,
                       progress: Optional[_AgeOffProgress] = None,
                       progress_target: Optional[str] = None) -> tuple:
    """Delete rows matching `base_query` (a full SELECT WITHOUT a LIMIT) in
    LIMIT-bounded pages until the match set is exhausted or `cap` rows have been
    deleted this cycle (keeps each call under the gateway timeout). When
    `mgmt_dataset` is given, a DELETE event is written for every deleted object,
    per page. Ambiguous API errors and short success responses are verified
    against the remaining match count because Crucible may commit a delete before
    the gateway reports an error. Returns (deleted, events)."""
    total = events = 0
    remaining = expected_count
    if remaining is None:
        remaining = _count_remaining_matches(base_query)
    while total < cap:
        lim = min(batch, cap - total)
        expected_this_page = min(lim, remaining) if remaining is not None else lim
        if expected_this_page <= 0:
            break
        qry = f"{base_query} LIMIT {lim}"
        result = {}
        delete_error = None
        try:
            logging.info(f"age_off: sending delete command for up to {lim} record(s)")
            wc.token = auth.get_token()
            with wc.timeout_scope(_DELETE_TIMEOUT):
                result = wc.delete_entity_record_by_query(qry) or {}
        except Exception as e:
            delete_error = e
        deleted = result.get('deletedRecords') or []
        failed = result.get('failedKeys') or []
        n = int(result.get('totalDeleted', len(deleted)))

        # The delete service can commit records and then return a gateway error,
        # or return a short/zero count. Verify only those ambiguous outcomes;
        # normal full-page responses incur no extra read.
        if delete_error is not None or n < expected_this_page:
            after = _count_remaining_matches(base_query)
            if remaining is not None and after is not None:
                inferred = max(0, min(expected_this_page, remaining - after))
                if inferred > n:
                    logging.warning(
                        f"age_off: delete response was ambiguous ({delete_error or f'reported {n}'}); "
                        f"outcome verification confirms {inferred} record(s) deleted")
                    n = inferred
                remaining = after
            elif delete_error is not None:
                if not _is_empty_match_400(delete_error):
                    logging.error(f"age_off: delete batch failed and its outcome could not be verified: "
                                  f"{delete_error}")
                    if progress:
                        progress.record_delete_outcome(
                            progress_target, unconfirmed=expected_this_page)
                break
        elif remaining is not None:
            remaining = max(0, remaining - n)

        total += n
        if progress and n:
            progress.record_delete_outcome(progress_target, confirmed=n)
        logging.info(f"age_off: delete command sent successfully; deleted {n} record(s)")
        if failed:
            logging.warning(f"age_off: {len(failed)} key(s) failed to delete")
        if mgmt_dataset and deleted:
            events += _write_delete_events(mgmt_dataset, deleted, progress)
        if mgmt_dataset and n > len(deleted):
            logging.warning(
                f"age_off: {n - len(deleted)} verified deletion(s) had no returned record; "
                "DELETE management events could not be reconstructed")
        if n == 0 or n < expected_this_page or remaining == 0:
            break  # match set exhausted
    return total, events


def _delete_query_direct(base_query: str, cap: int, batch: int,
                         progress: Optional[_AgeOffProgress] = None,
                         progress_target: Optional[str] = None,
                         expected_count: Optional[int] = None) -> int:
    """Submit direct query deletes without UUID selection or post-delete reads.

    A successful response server-confirms its reported deletion count. A timeout
    is treated as unknown because Crucible may complete the delete server-side.
    Returns only server-confirmed deletions; the actual committed count may be
    higher because timeout batches are not independently verified.
    """
    reported_total = 0
    submitted = 0
    while submitted < cap:
        lim = min(batch, cap - submitted)
        expected_this_page = (min(lim, max(0, expected_count - submitted))
                              if expected_count is not None else lim)
        if expected_this_page <= 0:
            break
        query = f"{base_query} LIMIT {lim}"
        try:
            logging.info(f"age_off: sending direct query delete for up to "
                         f"{lim} record(s)")
            wc.token = auth.get_token()
            with wc.timeout_scope(_DELETE_TIMEOUT):
                result = wc.delete_entity_record_by_query(query) or {}
            reported = int(result.get('totalDeleted', 0))
            reported_total += reported
            if progress and reported:
                progress.record_delete_outcome(progress_target, confirmed=reported)
            logging.info(f"age_off: direct query delete server-confirmed "
                         f"{reported} record(s)")
            if reported < lim:
                break
        except Exception as e:
            if _is_empty_match_400(e):
                break
            logging.warning(f"age_off: direct query delete outcome is unknown: {e}")
            if progress:
                progress.record_delete_outcome(
                    progress_target, unconfirmed=expected_this_page)
        submitted += lim
    return reported_total


def _run_select_uuids(query: str) -> Optional[List[str]]:
    """Run a SELECT of objectId.uuid.

    Returns an empty list when the query matched no rows and None when the read
    failed, so object age-off can retry transient errors instead of treating them
    as a completed target.
    """
    try:
        rc.token = auth.get_token()
        df = rc.search(query, format='dataframe', auto_backtick=True)
    except Exception as e:
        if _is_empty_match_400(e):
            return []
        logging.error(f"age_off: select-uuids failed: {e}")
        return None
    if df is None or len(df) == 0:
        return []
    for col in ('uuid', 'objectId.uuid', 'objectId'):
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()
    return []


def _flag_objects_dropped(dataset: str, uuids: List[str],
                          progress: Optional[_AgeOffProgress] = None) -> int:
    """Partial-update the objects' entityStatus to 'DROPPED' BEFORE deletion. This
    tombstone rides the object-dataset SSE, so the transformer's per-worker object
    caches prune these uuids in near-real-time instead of waiting for the ~150s
    full-pull reconcile."""
    recs = [{'objectId': {'uuid': u}, 'entityStatus': 'DROPPED'} for u in uuids]
    marked = 0
    for i in range(0, len(recs), _MGMT_EVENT_CHUNK):
        try:
            wc.token = auth.get_token()
            with wc.timeout_scope(_DELETE_TIMEOUT):
                wc.update_entity_record_batch_by_name(dataset, recs[i:i + _MGMT_EVENT_CHUNK])
            marked += len(recs[i:i + _MGMT_EVENT_CHUNK])
            if progress:
                progress.update(markedDropped=len(recs[i:i + _MGMT_EVENT_CHUNK]))
        except Exception as e:
            logging.warning(f"age_off: DROPPED flag write failed on {dataset} "
                            f"({len(uuids)} obj(s)): {e}")
    return marked


def _query_existing_object_uuids(dataset: str, uuids: List[str]) -> Optional[set]:
    if not uuids:
        return set()
    in_list = ', '.join("'" + str(u).replace("'", "''") + "'" for u in uuids)
    query = (f"SELECT {dataset}.objectId.uuid FROM {dataset} "
             f"WHERE {dataset}.objectId.uuid IN ({in_list})")
    try:
        rc.token = auth.get_token()
        df = rc.search(query, format='dataframe', auto_backtick=True)
    except Exception as e:
        logging.warning(f"age_off: UUID delete outcome verification failed on {dataset}: {e}")
        return None
    if df is None or len(df) == 0:
        return set()
    for col in ('uuid', 'objectId.uuid', 'objectId'):
        if col in df.columns:
            return set(df[col].dropna().astype(str).tolist())
    logging.warning(f"age_off: UUID delete outcome verification on {dataset} "
                    "returned no UUID column")
    return None


def _delete_objects_by_uuid_confirmed(dataset: str, uuids: List[str]) -> List[str]:
    """Delete explicit UUIDs and return only UUIDs confirmed absent afterward."""
    attempted = list(dict.fromkeys(str(uuid) for uuid in uuids))
    if not attempted:
        return []
    in_list = ', '.join("'" + uuid.replace("'", "''") + "'" for uuid in attempted)
    query = f"SELECT * FROM {dataset} WHERE {dataset}.objectId.uuid IN ({in_list})"
    result = {}
    delete_error = None
    try:
        logging.info(f"age_off: sending delete command for {len(attempted)} object(s) "
                     f"from {dataset}")
        wc.token = auth.get_token()
        with wc.timeout_scope(_DELETE_TIMEOUT):
            result = wc.delete_entity_record_by_query(query) or {}
    except Exception as e:
        delete_error = e

    returned_records = result.get('deletedRecords') or []
    returned_ids = {_extract_uuid(record) for record in returned_records}
    returned_ids.discard(None)
    reported = int(result.get('totalDeleted', len(returned_records)))
    if delete_error is None and reported >= len(attempted):
        confirmed = attempted
    else:
        existing = _query_existing_object_uuids(dataset, attempted)
        if existing is None:
            if delete_error is not None and not _is_empty_match_400(delete_error):
                logging.error(f"age_off: delete-by-uuid failed and its outcome could not be verified "
                              f"on {dataset}: {delete_error}")
            confirmed = [uuid for uuid in attempted if uuid in returned_ids]
        else:
            confirmed = [uuid for uuid in attempted if uuid not in existing]
            logging.warning(
                f"age_off: UUID delete response was ambiguous "
                f"({delete_error or f'reported {reported}'}); outcome verification confirms "
                f"{len(confirmed)} of {len(attempted)} object(s) deleted")

    logging.info(f"age_off: delete command completed on {dataset}; "
                 f"confirmed {len(confirmed)} object(s) deleted")
    return confirmed


def _delete_objects_by_uuid(dataset: str, uuids: List[str]) -> int:
    """Delete objects by UUID and return the verified number deleted."""
    return len(_delete_objects_by_uuid_confirmed(dataset, uuids))


def _select_uuids_query(base_query: str, dataset: str) -> Optional[str]:
    """Rewrite a base SELECT into 'SELECT <dataset>.objectId.uuid FROM ...'."""
    q, n = re.subn(r'^\s*select\s+.*?\s+from\s+', f'SELECT {dataset}.objectId.uuid FROM ',
                   base_query, count=1, flags=re.IGNORECASE | re.DOTALL)
    return q if n else None


def _drop_and_delete_objects(object_dataset: str, select_uuids_query: str,
                             mgmt_dataset: Optional[str], cap: int, batch: int,
                             retries: int = _DEFAULT_OBJECT_RETRIES,
                             progress: Optional[_AgeOffProgress] = None) -> tuple:
    """Object age-off path: per LIMIT-bounded page, SELECT the uuids, flag them
    DROPPED (near-real-time transformer cache prune via the object SSE), delete
    them by uuid, and write DELETE management events. `select_uuids_query` is a
    full `SELECT <ds>.objectId.uuid FROM <ds> WHERE ...` WITHOUT a LIMIT.
    Returns (deleted, events)."""
    total = events = 0
    retries = max(0, int(retries))
    while total < cap:
        lim = min(batch, cap - total)
        uuids = None
        for attempt in range(retries + 1):
            uuids = _run_select_uuids(f"{select_uuids_query} LIMIT {lim}")
            if uuids is not None:
                break
            if attempt < retries:
                logging.warning(f"age_off: object UUID select failed; restarting page "
                                f"({attempt + 1}/{retries})")
                _refresh_tokens()
        if uuids is None:
            logging.error("age_off: object UUID select retries exhausted; stopping target")
            break
        if not uuids:
            break
        _flag_objects_dropped(object_dataset, uuids, progress)
        deleted_uuids = []
        for attempt in range(retries + 1):
            deleted_uuids = _delete_objects_by_uuid_confirmed(object_dataset, uuids)
            if deleted_uuids:
                break
            if attempt < retries:
                logging.warning(f"age_off: object delete was not confirmed; restarting page "
                                f"({attempt + 1}/{retries})")
                _refresh_tokens()
        total += len(deleted_uuids)
        if progress:
            progress.record_delete_outcome(
                'objects', confirmed=len(deleted_uuids),
                unconfirmed=len(uuids) - len(deleted_uuids))
        if mgmt_dataset and deleted_uuids:
            events += _write_delete_events(
                mgmt_dataset, [{'objectId': {'uuid': u}} for u in deleted_uuids], progress)
        if not deleted_uuids:
            logging.error("age_off: no selected objects were confirmed deleted; "
                          "stopping this target to avoid an infinite retry loop")
            break
        if len(uuids) < lim:
            break
    return total, events


def _strip_trailing_limit(q: str) -> str:
    """Remove a trailing 'LIMIT n' so we can page the query ourselves."""
    return re.sub(r'\s+limit\s+\d+\s*$', '', q.strip(), flags=re.IGNORECASE).strip()


def _query_target_dataset(q: str) -> Optional[str]:
    """Best-effort extraction of the dataset name following FROM."""
    m = re.search(r'\bfrom\s+([^\s]+)', q, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _age_off_custom_query(query: str, object_dataset: str, mgmt_dataset: str,
                          dry_run: bool, cap: int, batch: int,
                          mark_dropped: bool = True,
                          object_retries: int = _DEFAULT_OBJECT_RETRIES,
                          progress: Optional[_AgeOffProgress] = None) -> None:
    """Age off using an operator-supplied query INSTEAD of the per-environment
    logic. Paged like the built-in path (any trailing LIMIT is stripped). DELETE
    events are written only when the query targets the configured object dataset
    (so a track-head query doesn't emit object management events)."""
    base = _strip_trailing_limit(query)
    target = _query_target_dataset(base)
    mgmt = mgmt_dataset if (target and object_dataset and target == object_dataset) else None
    if dry_run:
        cnt = None
        cq = _derive_count_query(base)
        if cq:
            try:
                rc.token = auth.get_token()
                df = rc.search(cq, format='dataframe', auto_backtick=True)
                cnt = int(df.iloc[0, 0]) if df is not None and len(df) else 0
            except Exception as e:
                logging.warning(f"age_off: custom-query count failed: {e}")
        logging.info(f"age_off[DRY-RUN] custom query on {target}: would delete "
                     f"{cnt if cnt is not None else 'unknown'} "
                     f"(DELETE events -> {mgmt or 'none'})")
        return
    deleted, events = (0, 0)
    if mgmt:
        if mark_dropped:
            su = _select_uuids_query(base, target)
            if su:
                deleted, events = _drop_and_delete_objects(
                    target, su, mgmt, cap, batch, object_retries, progress)
            else:
                deleted, events = _delete_in_batches(
                    base, cap, batch, mgmt_dataset=mgmt, progress=progress)
        else:
            deleted = _delete_query_direct(
                base, cap, batch, progress, 'objects')
    else:
        if mark_dropped and target == object_dataset:
            su = _select_uuids_query(base, target)
            if su:
                deleted, events = _drop_and_delete_objects(
                    target, su, None, cap, batch, object_retries, progress)
            else:
                deleted, events = _delete_in_batches(base, cap, batch, progress=progress)
        elif target == object_dataset:
            deleted = _delete_query_direct(
                base, cap, batch, progress, 'objects')
        else:
            deleted, events = _delete_in_batches(base, cap, batch, progress=progress)
    logging.info(f"age_off custom query on {target}: deleted {deleted}, "
                 f"DELETE events {events} (events -> {mgmt or 'none'})")


def _age_off_objects(object_dataset: str, mgmt_dataset: Optional[str],
                     hours: Dict[str, float], dry_run: bool,
                     cap: int, batch: int, label: str = 'objects',
                     mark_dropped: bool = True,
                     object_retries: int = _DEFAULT_OBJECT_RETRIES,
                     progress: Optional[_AgeOffProgress] = None) -> tuple:
    """Delete non-CONFIRMED stale objects per environment. Objects are flagged
    DROPPED (when enabled for transformer cache pruning) then hard-deleted. When
    `mgmt_dataset` is given a DELETE event is written per deleted object; pass None
    to skip events (used for superseded objects, which already left the pipeline)."""
    total_deleted = total_events = 0
    for env, thr in hours.items():
        where = (f"{object_dataset}.entityStatus <> 'CONFIRMED' "
                 f"and {object_dataset}.identity.environment.environment = '{env}' "
                 f"and {object_dataset}.crucibleHeader.updatedDate "
                 f"< TIMESTAMP_OFFSET(-{thr}, 'hours')")
        n = _count(object_dataset, where)
        logging.info(f"age_off {label} {env} (>{thr}h, not CONFIRMED): "
                     f"found {n if n is not None else 'unknown'} object(s)")
        if n == 0:
            logging.info(f"age_off {label} {env} (>{thr}h, not CONFIRMED): "
                         "deleting 0 object(s); nothing to age off")
            continue
        if dry_run:
            logging.info(f"age_off[DRY-RUN] {label} {env} (>{thr}h, not CONFIRMED): "
                         f"would delete {n if n is not None else '?'}")
            continue
        to_delete = min(n, cap) if n is not None else cap
        logging.info(f"age_off {label} {env} (>{thr}h): deleting up to "
                     f"{to_delete} object(s) this cycle")
        if mark_dropped:
            deleted, events = _drop_and_delete_objects(
                object_dataset,
                f"SELECT {object_dataset}.objectId.uuid FROM {object_dataset} WHERE {where}",
                mgmt_dataset, cap, batch, object_retries, progress)
        else:
            deleted = _delete_query_direct(
                f"SELECT * FROM {object_dataset} WHERE {where}", cap, batch,
                progress, 'objects', n)
            events = 0
        total_deleted += deleted
        total_events += events
        logging.info(f"age_off {label} {env} (>{thr}h): candidates {n}, "
                     f"deleted {deleted}, DELETE events {events}")
    return total_deleted, total_events


def _age_off_tracks(track_dataset: str, hours: Dict[str, float], dry_run: bool,
                    cap: int, batch: int,
                    progress: Optional[_AgeOffProgress] = None,
                    progress_target: Optional[str] = None) -> int:
    """Delete stale track heads (by trackUpdatedTimestamp) per environment."""
    total = 0
    for env, thr in hours.items():
        where = (f"{track_dataset}.environment = '{env}' "
                 f"and {track_dataset}.trackUpdatedTimestamp "
                 f"< TIMESTAMP_OFFSET(-{thr}, 'hours')")
        n = _count(track_dataset, where)
        logging.info(f"age_off tracks {track_dataset} {env} (>{thr}h): "
                     f"found {n if n is not None else 'unknown'} track head(s)")
        if n == 0:
            logging.info(f"age_off tracks {track_dataset} {env} (>{thr}h): "
                         "deleting 0 track head(s); nothing to age off")
            continue
        if dry_run:
            logging.info(f"age_off[DRY-RUN] tracks {track_dataset} {env} (>{thr}h): "
                         f"would delete {n if n is not None else '?'}")
            continue
        to_delete = min(n, cap) if n is not None else cap
        logging.info(f"age_off tracks {track_dataset} {env} (>{thr}h): deleting up to "
                     f"{to_delete} track head(s) this cycle")
        deleted, _ = _delete_in_batches(
            f"SELECT * FROM {track_dataset} WHERE {where}", cap, batch,
            expected_count=n, progress=progress, progress_target=progress_target)
        total += deleted
        logging.info(f"age_off tracks {track_dataset} {env} (>{thr}h): "
                     f"candidates {n}, deleted {deleted}")
    return total


def _resolve_datasets(config: dict, perspective: str) -> Dict[str, str]:
    return {
        'object': config.get('object_dataset', f'{perspective}_Objects'),
        'superseded': config.get('superseded_object_dataset',
                                 f'{perspective}_SupersededObjects'),
        'mgmt': config.get('object_management_event_dataset',
                           f'{perspective}_ObjectManagementEvents'),
        'component': config.get('component_track_head_dataset',
                                f'{perspective}_ComponentTrackHeads'),
        'principal': config.get('principal_track_head_dataset',
                                f'{perspective}_PrincipalTrackHeads'),
    }


def _age_off_cycle(config: dict, perspective: str, dry_run: bool,
                   hours: Dict[str, float], cap: int, batch: int,
                   custom_query: Optional[str] = None,
                   mark_dropped: bool = True,
                   object_retries: int = _DEFAULT_OBJECT_RETRIES,
                   progress: Optional[_AgeOffProgress] = None) -> None:
    ds = _resolve_datasets(config, perspective)
    tag = '[DRY-RUN] ' if dry_run else ''
    if progress:
        progress.update(status='running')
    if custom_query:
        logging.info(f"age_off: {tag}cycle start; CUSTOM QUERY mode; "
                     f"batch={batch} max_per_target={cap}; query={custom_query!r}")
        _refresh_tokens()
        _age_off_custom_query(custom_query, ds['object'], ds['mgmt'], dry_run, cap, batch,
                              mark_dropped, object_retries, progress)
        if progress:
            progress.update(status='cycle complete')
        logging.info(f"age_off: {tag}cycle done (custom query)")
        return
    logging.info(f"age_off: {tag}cycle start; thresholds(h)={hours}; "
                 f"batch={batch} max_per_target={cap}; datasets={ds}")
    _refresh_tokens()
    def _age_off_object_datasets() -> tuple:
        obj_deleted, events = _age_off_objects(
            ds['object'], ds['mgmt'], hours, dry_run, cap, batch,
            mark_dropped=mark_dropped, object_retries=object_retries, progress=progress)
        superseded_deleted, _ = _age_off_objects(
            ds['superseded'], None, hours, dry_run, cap, batch,
            label='superseded objects', mark_dropped=mark_dropped,
            object_retries=object_retries, progress=progress)
        return obj_deleted, superseded_deleted, events

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix='age-off') as executor:
        object_future = executor.submit(
            _age_off_object_datasets)
        component_future = executor.submit(
            _age_off_tracks, ds['component'], hours, dry_run, cap, batch, progress,
            'componentTrackHeads')
        principal_future = executor.submit(
            _age_off_tracks, ds['principal'], hours, dry_run, cap, batch, progress,
            'principalTrackHeads')

        obj_del, sup_del, evt = object_future.result()
        comp_del = component_future.result()
        prin_del = principal_future.result()
    if progress:
        progress.update(status='cycle complete')
    logging.info(f"age_off: {tag}cycle done; objects_deleted={obj_del} "
                 f"superseded_deleted={sup_del} "
                 f"delete_events={evt} component_tracks_deleted={comp_del} "
                 f"principal_tracks_deleted={prin_del}")


def _load_perspective_config(stream_manager_perspective: str) -> dict:
    """Fetch the perspective config only to resolve perspective dataset names.
    Falls back to an empty config (with <perspective>_* dataset defaults) on error."""
    try:
        config_list = find_and_validate_configs(
            stream_manager_perspective, include_scripts=False, rc_instance=rc)
    except Exception as e:
        logging.warning(f"age_off: could not load configs ({e}); "
                        f"using {stream_manager_perspective}_* dataset defaults")
        config_list = []
    config = {}
    for c in (config_list or []):
        if not c.get('disabled'):
            config = dict(c)
            break
    if not config and config_list:
        config = dict(config_list[0])
    config.setdefault('perspective', stream_manager_perspective)
    return config


def run(stream_manager_perspective: str, log_level: str = 'INFO',
        once: bool = False, dry_run: Optional[bool] = None,
        interval: Optional[str] = None,
        age_hours: Optional[Dict[str, float]] = None,
        max_per_env: Optional[int] = None,
        batch_size: Optional[int] = None,
        query: Optional[str] = None,
        mark_dropped: Optional[bool] = None,
        object_retries: Optional[int] = None,
        progress_path: Optional[str] = None) -> None:
    """Run object/track age-off. Launched by the Object Manager button as a
    periodic daemon; also runnable one-shot from the CLI.

    age_hours: optional {env: hours} overrides (from the CLI --age-<env> flags or
    the Streamlit text boxes) that take precedence over defaults.
    max_per_env: cap on rows deleted per (environment, dataset) per cycle.
    batch_size: rows deleted per delete_entity_record_by_query call (LIMIT).
    mark_dropped: whether objects are partially updated to entityStatus='DROPPED'
        before hard deletion. Defaults to true.
    object_retries: retries for transient object UUID-select or unconfirmed-delete
        failures. Defaults to 3.
    """
    numeric_level = getattr(logging, str(log_level).upper(), logging.INFO)
    log_utils.get_logger(log_type='age_off', log_level=numeric_level)
    logging.getLogger().setLevel(numeric_level)

    def _terminate(signum, _frame):
        logging.info(f"age_off: received signal {signum}, exiting")
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    _init_controllers()
    _refresh_tokens()
    config = _load_perspective_config(stream_manager_perspective)
    perspective = config.get('perspective', stream_manager_perspective)
    hours = _age_off_hours(age_hours)

    if dry_run is None:
        dry_run = _is_true(os.getenv('CRUCIBLE_AGE_OFF_DRY_RUN', 'false'))
    if interval is None:
        interval = _DEFAULT_INTERVAL
    if max_per_env is None:
        max_per_env = _DEFAULT_MAX_PER_TARGET
    if batch_size is None:
        batch_size = _DEFAULT_BATCH_SIZE
    if mark_dropped is None:
        mark_dropped = True
    if object_retries is None:
        object_retries = _DEFAULT_OBJECT_RETRIES
    progress = _AgeOffProgress(progress_path)

    logging.info(f"age_off: starting for perspective '{perspective}' "
                 f"(dry_run={dry_run}, once={once}, interval={interval}, "
                 f"max_per_env={max_per_env}, batch_size={batch_size}, "
                 f"mark_dropped={mark_dropped}, object_retries={object_retries}, "
                 f"{'CUSTOM QUERY mode' if query else 'per-environment mode'}, "
                 f"thresholds(h)={hours})")

    if once:
        try:
            _age_off_cycle(config, perspective, dry_run, hours, max_per_env, batch_size,
                           custom_query=query, mark_dropped=mark_dropped,
                           object_retries=object_retries, progress=progress)
        except Exception as exc:
            progress.update(status='error', error=str(exc))
            raise
        progress.update(status='complete')
        return

    sleep_s = _parse_interval(interval)
    while True:
        try:
            _age_off_cycle(config, perspective, dry_run, hours, max_per_env, batch_size,
                           custom_query=query, mark_dropped=mark_dropped,
                           object_retries=object_retries, progress=progress)
        except SystemExit:
            raise
        except Exception as e:
            progress.update(status='error', error=str(e))
            logging.error(f"age_off: cycle failed: {e}")
            logging.error(traceback.format_exc())
        time.sleep(sleep_s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object & track age-off')
    parser.add_argument('stream_manager_perspective', help='Perspective, e.g. Live_POV')
    parser.add_argument('--once', action='store_true',
                        help='Run a single age-off pass then exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what WOULD be deleted; delete nothing')
    parser.add_argument('--interval', default=None,
                        help='Loop interval, e.g. 15m, 45s, 2h (default 15m)')
    parser.add_argument('--log', default='INFO',
                        help='INFO (default), DEBUG, WARNING, or ERROR')
    parser.add_argument('--max-per-env', type=int, default=None, metavar='N',
                        help=f'Max rows deleted per environment+dataset per cycle '
                             f'(default {_DEFAULT_MAX_PER_TARGET})')
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                        help=f'Rows deleted per delete call / LIMIT (default {_DEFAULT_BATCH_SIZE})')
    parser.add_argument('--query', default=None, metavar='SQL',
                        help='Optional full delete query (SELECT ... FROM <dataset> WHERE ...) that '
                             'REPLACES the per-environment age-off. Paged automatically (a trailing '
                             'LIMIT is ignored); honors --dry-run. DELETE events written only when the '
                             'query targets the object dataset.')
    parser.add_argument('--no-mark-dropped', action='store_true',
                        help="Delete objects without first setting entityStatus='DROPPED'")
    parser.add_argument('--object-retries', type=int, default=None, metavar='N',
                        help=f'Retries after transient object select/delete failures '
                             f'(default {_DEFAULT_OBJECT_RETRIES})')
    # Per-environment age thresholds (hours); generated from DEFAULT_AGE_OFF_HOURS
    # so the flags stay in sync (e.g. --age-air, --age-sea-surface).
    for _env, _default in DEFAULT_AGE_OFF_HOURS.items():
        parser.add_argument(
            f"--age-{_env.lower().replace('_', '-')}", type=float, default=None,
            metavar='HOURS',
            help=f"Hours before a stale {_env} object/track is aged off (default {_default})")
    args = parser.parse_args()

    _cli_ages = {env: getattr(args, f"age_{env.lower()}")
                 for env in DEFAULT_AGE_OFF_HOURS
                 if getattr(args, f"age_{env.lower()}", None) is not None}

    run(args.stream_manager_perspective, log_level=args.log, once=args.once,
        dry_run=(True if args.dry_run else None), interval=args.interval,
        age_hours=(_cli_ages or None),
        max_per_env=args.max_per_env, batch_size=args.batch_size, query=args.query,
        mark_dropped=(False if args.no_mark_dropped else None),
        object_retries=args.object_retries)
