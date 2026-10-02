## Findings

1. **Major · BLOCKER — §§5.3, 5.5/D8–D11: pending work can stall indefinitely.** Failed head PUTs and existence reads are retried on the “next cycle,” and association stamps are retried “each cycle.” The owner loop is described as taking work from the input queue, with no scheduled wakeup. If input stops, withheld events may never be released and stamps may neither retry nor reach their age-based removal. [Design: owner loop](docs/DESIGN.md:231), [D11 retry](docs/DESIGN.md:273), [stamp rule](docs/DESIGN.md:265). **Recommendation:** specify a periodic retry and expiry wakeup independent of SSE input, and test an idle stream after a transient failure.

2. **Major · BLOCKER — §5.7: the hard deadline can still slip.** The daemon `threading.Timer` fixes the identified case where inline CPU work blocks an event-loop timer. But the design does not say how SIGTERM arms it; an event-loop signal callback could itself be delayed by that CPU work. The timer also logs and flushes every handler *before* `os._exit`; a blocked handler can prevent exit. [Design: inline CPU](docs/DESIGN.md:176), [deadline path](docs/DESIGN.md:477). **Recommendation:** arm an independent watchdog at signal receipt and give diagnostics a bounded path that cannot delay the final exit.

3. **Minor · BLOCKER — §5.7: exit-status rules conflict.** The design calls existing-head updates best-effort, yet defines an unwritten “head create or PUT” as an authoritative failure; those updates use PUT. Implementers could return exit code 1 for writes the next sentence says must not affect it. [Best-effort updates](docs/DESIGN.md:241), [exit status](docs/DESIGN.md:489). **Recommendation:** classify writes by their role, explicitly covering D11 PUTs, existing-head updates, and association stamps.

4. **Minor · RESIDUAL — §5.5/D4: transformer overlap can evade deduplication.** The whole-record fingerprint and float JSON rule are specified, with a compaction round-trip test. The assertion that two transformer pods produce identical content assumes deterministic hooks; the design permits dataset-supplied code. Differing output for one origin record will have differing fingerprints and both copies can be fused during accepted D10 overlap. [Fingerprint claim](docs/DESIGN.md:379), [hook contract](docs/DESIGN.md:550). **Recommendation:** state the determinism assumption and this D10 consequence.

The principal-head rehydration claim checks out: the baseline writes fused `identity.*` to the head, and the design states that ranks and potentially recent best-effort values cannot be recovered. [Baseline write](src/objectApps/correlators/entity_manager/entity_track_fuser.py:939), [design limit](docs/DESIGN.md:260). D11’s check/create race is also stated as accepted D10 overlap; its query now validates IDs, chunks requests, and fails closed.

## Pass-4 status

| # | Status | Reason |
|---|---|---|
| 1 | Resolved | The non-atomic check/create race is explicitly assigned to accepted D10 overlap. |
| 2 | Resolved | Existence reads are chunked, validate IDs, and fail closed. |
| 3 | Partially resolved | Whole-record hashing and float serialization are specified; transformer output determinism remains an unstated assumption. |
| 4 | Resolved | Head values support rehydration; missing ranks and stale values are stated as D2 behavior. |
| 5 | Partially resolved | Uncertain writes and log flushing are addressed, and the timer is off-loop; the timer’s arming and blocking diagnostic path remain open. |
| 6 | Resolved | Stamp count and age limits are stated, and the D2 summary matches the body. |

**NOT READY: idle-stream retry, hard-deadline enforcement, and exit-status classification.**