"""Parity of the tracker with the baseline, using fixtures from generate.py.

The fixture applies the two documented fixes to the baseline output (each
event's ``reportIds`` from its own report; geodetic angles in radians) and
omits the time-dependent ``stale`` field.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest

from crucible_entity_manager.components.heads import HeadSync, HeadTargets
from crucible_entity_manager.components.keyed import KeyedState
from crucible_entity_manager.components.tracker import FeedTracker
from crucible_entity_manager.components.tracking import TrackState
from crucible_entity_manager.config.perspective import WriteSettings, parse_perspective
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.writer import BatchWriter, DrainBudget, WriteLedger
from tests.parity.test_transformer_parity import assert_same
from tests.unit.components.fakes import FakeCrucible
from tests.unit.config.test_perspective import feed_row, perspective_row

FIXTURE: Final = Path(__file__).resolve().parent.parent / "fixtures" / "parity" / "tracker.json"
ALTITUDE_ATOL_M: Final = 1e-4
"""The baseline altitude formula errs by up to ~0.1 mm here; the port is exact."""

with FIXTURE.open() as handle:
    PARITY: Final[dict[str, Any]] = json.load(handle)


def feed_tracker(mode: str) -> FeedTracker:
    row = feed_row(
        "Radar_Feed",
        crucible_tracker=mode,
        component_track_event_dataset="ComponentTrackEvents",
        component_track_head_dataset="ComponentTrackHeads",
    )
    (feed,) = parse_perspective("Blue", [perspective_row(), row], {}).feeds
    crucible = FakeCrucible()
    writer = BatchWriter(
        crucible,
        crucible,
        WriteSettings(),
        ledger=WriteLedger(),
        budget=DrainBudget(request_seconds=1.0),
    )
    heads = HeadSync(
        writer,
        HeadTargets("ComponentTrackHeads", "ComponentTrackEvents"),
        update_interval_seconds=0.0,
        label="",
    )
    return FeedTracker(
        feed,
        "ComponentTrackHeads",
        heads,
        KeyedState(TrackState),
        lambda: datetime(2026, 9, 30, tzinfo=UTC),
    )


def without_stale(heads: list[JSONObject]) -> list[JSONObject]:
    return [{key: value for key, value in head.items() if key != "stale"} for head in heads]


def assert_altitudes_close(
    actual: list[JSONObject], expected: list[JSONObject]
) -> list[JSONObject]:
    """Compare altitudes to `ALTITUDE_ATOL_M`, then drop them from both sides."""
    for got, want in zip(actual, expected, strict=True):
        got_geodetic, want_geodetic = got.get("geodetic"), want.get("geodetic")
        if isinstance(got_geodetic, dict) and isinstance(want_geodetic, dict):
            altitude, wanted = (
                got_geodetic.pop("altitude", None),
                want_geodetic.pop("altitude", None),
            )
            assert (altitude is None) == (wanted is None)
            if isinstance(altitude, float) and isinstance(wanted, float):
                assert abs(altitude - wanted) <= ALTITUDE_ATOL_M
    return actual


@pytest.mark.parametrize("name", sorted(PARITY.keys() - {"baseline_sha", "generator"}))
def test_tracker_matches_the_baseline(name: str) -> None:
    case = PARITY[name]
    tracker = feed_tracker(case["mode"])
    tracker.restore(case["heads"])
    for index, batch in enumerate(case["batches"]):
        heads, events = tracker.outputs(batch["reports"])
        events = assert_altitudes_close(events, batch["events"])
        heads = assert_altitudes_close(without_stale(heads), batch["heads"])
        assert_same(events, batch["events"], f"batch {index} events")
        assert_same(heads, batch["heads"], f"batch {index} heads")
