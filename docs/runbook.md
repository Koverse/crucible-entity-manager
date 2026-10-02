# Runbook

How to run, observe and recover crucible-entity-manager on Executor.

## Starting

Install the definitions in [deploy/executions](../deploy/executions/README.md)
and start one Execution per partition. Start the components in pipeline order:
transformer, tracker, fuser, duplicates. Each one reads only what its
upstream writes, so the order matters only for the first data to flow.

## Environment

The profile supplies these; the pipeline reads them:

| Variable | Purpose |
|---|---|
| `CRUCIBLE_SERVICES_HOST` | Crucible host; also builds the SSE endpoint |
| `CRUCIBLE_CLIENT_ID`, `CRUCIBLE_CLIENT_SECRET`, `CRUCIBLE_TOKEN_URL` and the other `CRUCIBLE_*_URL` | Authentication, through cruciblelib |
| `CRUCIBLE_SSE_URL` | Overrides the SSE endpoint |
| `CRUCIBLE_SSE_READ_TIMEOUT` | Seconds without an event or keep-alive before reconnecting (default 180) |
| `CRUCIBLE_SSE_READ_BUFSIZE` | aiohttp read buffer, in bytes (default 256 MiB); lines up to twice this size are accepted |
| `CRUCIBLE_SSL_VERIFY` | TLS verification for SSE |
| `CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS`, `CRUCIBLE_SKIP_HEAD_PRELOAD`, `CRUCIBLE_HEAD_PRELOAD_LIMIT` | Defaults for the matching configuration keys |

## Logs

Every line carries the component and partition, for example
`tracker p1/3`. Lines to look for:

| Message | Meaning |
|---|---|
| `SSE subscribed at …` | A source connected. The first one marks the start of data. |
| `SSE gap …` (WARNING) | A disconnect: the records between the last event and the resubscription were not received and will not be (no replay). Logged on resubscription; a pod restart is a gap with no such line. |
| `Head preload … hit its N-row limit` | Older heads were not loaded. Tracks whose heads were dropped start fresh; raise the preload limit if this matters. |
| `… record(s) to <heads> failed` | A head write failed; that track's events are held until its head is written. |
| `Dropped … withheld event(s)` | Held events lost at the buffer limits or with an idle track. |
| `… record(s) to <dataset> failed` | Any other failed write; the warnings just before it give the server's response. Authoritative writes that fail before a drain are reported here, not in the exit code. |
| `Shutdown left N received record(s) unprocessed` | Records queued at SIGTERM; like a gap, they are not recovered. |
| `Drained with authoritative writes unconfirmed` (ERROR) | The process exits 1; the counts say which datasets. |
| `Shutdown deadline reached …` (stderr) | The watchdog ended the process at `--drain-seconds`. Uncertain writes may or may not have landed. |

## Health

With `--health-port`, `GET /healthz` returns 200 or 503 with a JSON list of
problems: the owner loop stalled for `--stall-seconds` with work in hand, or a
source silent past its watchdog window. It reports healthy during preload.
Executor does not pass probes through yet (DESIGN.md §12); until it does, the
SSE watchdog's reconnect is the mitigation for a stalled stream.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Clean shutdown: no authoritative write was unconfirmed at exit or lost during the drain. Earlier write failures and records queued at shutdown are only logged; check the logs above. |
| 1 | A failure, or a shutdown that left authoritative writes unconfirmed or lost. |
| 2 | Invalid command-line arguments. |

The Deployment restarts the pod after any exit.

## Recovery

- **A pod restarts.** It rebuilds state from heads. Tracks resume from their
  last written head; the error from a stale prior decays with the next
  measurements. Records sent while the pod was down are a gap.
- **Configuration errors** fail at startup with the row and key named. Fix the
  row in `Entity_Stream_Manager_Configurations` and rerun.
- **Hook errors** (a missing script or function) fail at startup. At run time,
  a custom function that raises is skipped for that batch and logged; a unit
  conversion that raises drops the record it failed on (`Dropped a record:
  unit conversion …`), so the record is lost.
- **A wrong SUPERSEDE.** Write a RESTORE management event for the superseded
  track. The duplicate identifier will not supersede it again during the
  restore cooldown, and the fuser rebuilds the survivor's principal track.
- **Changing the partition count.** Stop every partition of the component,
  then start the new set, all with the same `--partitions`.

## Reruns and rollouts

A rerun is a RollingUpdate: old and new pods overlap for a few seconds. During
that window event datasets can receive duplicate rows and heads are
last-writer-wins (DESIGN.md §5.5, D10). Downstream, the tracker and fuser drop
inputs they have already applied, so duplicates are not fused twice within a
pod.
