# Execution definitions

`definitions.yaml` holds one Executor definition per component, in the format
of the `definitions:` map in crucible-prototype's
`crucible/charts/templates/execution-api-configmap.yaml`.

## Installing

Until the repo-file definition sync is wired up (DESIGN.md §12), either:

- add the entries to the configmap's `definitions:` map, or
- `POST` each one to `/api/v2/executions/definitions`.

Replace `LIVE_POV` with the perspective, and pin `source.commit` for
production.

## Partitions

The transformer and the tracker can run as several partitions. Each partition
is its own Execution, started from the same definition with its own name and
arguments:

```text
POST /api/v2/executions/definitions/entity-tracker/start
{"executionName": "entity-tracker-p0", "args": ["tracker", "LIVE_POV", "--partition", "0", "--partitions", "3"]}
{"executionName": "entity-tracker-p1", "args": ["tracker", "LIVE_POV", "--partition", "1", "--partitions", "3"]}
{"executionName": "entity-tracker-p2", "args": ["tracker", "LIVE_POV", "--partition", "2", "--partitions", "3"]}
```

Every partition of a component must use the same `--partitions`. Each pod logs
its partition as `p<index>/<count>`, so a mismatch is visible in the logs.
Changing the partition count means stopping every partition and starting the
new set.

The fuser and the duplicate identifier run as exactly one Execution per
perspective; they refuse `--partitions` greater than 1.

## Rollout

A rerun is a RollingUpdate, so the old and new pods overlap for a few seconds
(DESIGN.md §5.5, D10). That is accepted for the transformer, tracker and fuser.
The duplicate identifier should use a `Recreate` rollout once the platform
allows a per-definition strategy (DESIGN.md §12).
