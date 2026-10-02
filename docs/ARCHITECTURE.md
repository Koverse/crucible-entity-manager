# Architecture

This is the condensed, current description of crucible-entity-manager. The
full design, with the reasoning behind each decision, is in
[DESIGN.md](DESIGN.md).

## The pipeline

```text
origin datasets ──SSE──▶ transformer ──▶ ReportEvents
ReportEvents    ──SSE──▶ tracker     ──▶ ComponentTrackHeads, ComponentTrackEvents
ComponentTrackEvents ─SSE─▶ fuser    ──▶ PrincipalTrackHeads, PrincipalTrackEvents
EntityManagementEvents ─SSE─▶ fuser       (supersede, delete, restore)
ComponentTrackEvents ─poll─▶ duplicates ──▶ EntityManagementEvents (SUPERSEDE)
```

| Component | Input | Output | Partitions |
|---|---|---|---|
| `transformer` | Each feed's origin dataset | Report events | Any number, by the origin record's `crucibleHeader.uuid` |
| `tracker` | Each tracked feed's reports | Component track heads and events | Any number, by component `trackId` |
| `fuser` | All component track events, management events | Principal track heads and events, `associatedPrincipalTrack` stamps | Exactly one per perspective |
| `duplicates` | Recent component track events (polled) | SUPERSEDE management events | Exactly one per perspective |

## One pod, one component, one partition

Each Executor Execution runs `src/run_entity_manager.py <component>
<perspective> [options]` in one asyncio process:

```text
SseSource(s) ──owned records──▶ RecordQueue ──▶ owner loop ──▶ component ──▶ BatchWriter
                                 (bounded)       (one task)    (pure steps)   (bounded concurrency)
```

- **Sources** (`crucible/sse.py`) read Server-Sent Events, keep only the
  records this partition owns, and block when the queue is full, which pushes
  back on the server (D1). A reconnect is a data gap, logged when the source
  resubscribes (D9). Downtime across a pod restart is a gap too, visible only
  as the time between the old pod's last logs and the new pod's subscription.
- **The owner loop** (`runtime/service.py`) is the only task that touches a
  component's state. It handles batches in arrival order and calls `tick`
  every `--tick-seconds` for timed work, even when no input arrives.
- **Writes** (`crucible/writer.py`) are chunked and concurrent, with the
  baseline's failure handling: one retry on a timeout, no fallback on an
  authorization failure, and sub-chunks to isolate rejected records.

## Keyed state

The tracker and the fuser keep state per key (track or principal). The shared
machinery in `components/` bounds and orders it:

- `keyed.py`: `KeyedState` bounds per-key state by idle time and an LRU cap
  (D2). `AppliedInputs` drops an input a key has already applied (D4).
- `heads.py`: `HeadSync` writes a key's head before its events. An unknown
  key's head goes through PUT, an existence check, and a create only if the
  head is confirmed missing, so an existing head is never overwritten (D11).
  Events wait for their head (D8). Updates to known heads are best-effort,
  deferred to the update interval and sent by one background task.

## Numerics

`core/` holds the pure numerics, with no I/O: the six-state Kalman filter and
covariance intersection (`kalman.py`), the RTS smoother (`smoother.py`), and
WGS84/ECEF conversions (`geodesy.py`). The tracker's and fuser's filter
policies (`tracking.py`, `fusion.py`) keep their own noise tables and guards,
as at the baseline (D6). Parity with the baseline crucible-streamlit `1b534df`
is checked by fixtures in `tests/fixtures/parity`, generated from the baseline
code by `tests/parity/generate.py`.

## Configuration

Per-perspective configuration comes from the `Entity_Stream_Manager_Configurations`
dataset, parsed once and validated (`config/perspective.py`). Customer hooks
(unit conversions and custom functions) come from
`Entity_Stream_Manager_Functions` (`hooks/loader.py`). Runtime settings come
from the command line (`cli.py`).

## Startup and shutdown

1. Authenticate, load and validate the configuration, and build the component.
2. With `--health-port`, start `GET /healthz` (healthy during preload).
   Preload state from heads (best effort), then start the sources and the
   owner loop.
3. On SIGTERM: stop the sources, finish the current batch, flush the writes.
   A watchdog thread ends the process at `--drain-seconds` no matter what.
   The exit code is 1 if, at exit, an authoritative write is still unconfirmed
   or was lost during the drain. It does not cover writes that failed earlier
   (those are logged when they fail) or records still queued at shutdown.

## Layering

Enforced by `tests/unit/test_architecture.py`:

- `core` imports nothing else from the package; `config`, `hooks` and
  `crucible` build on it.
- `components` use the client-free parts of `crucible` and receive their
  client, writers and sources through the `Services` protocol.
- Only `runtime` wires in the concrete clients. `cruciblelib` is imported only
  in `crucible/client.py`.
