"""Parity of the fuser with the baseline, using fixtures from generate.py.

The fixture applies three documented fixes to the baseline output: deleted
component tracks are not fused, geodetic coordinates come from the fused
position in radians, and fused identity starts from the preloaded principal
heads. The time-dependent ``stale`` field is omitted.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import pytest

from crucible_entity_manager.components.fuser import (
    MANAGEMENT,
    Fuser,
    FuserDatasets,
    FuserSetup,
)
from crucible_entity_manager.components.fusion import Fusion, FusionParams
from crucible_entity_manager.components.heads import HeadSync, HeadTargets
from crucible_entity_manager.config.perspective import WriteSettings, parse_perspective
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.writer import BatchWriter, DrainBudget, WriteLedger
from tests.parity.test_tracker_parity import assert_altitudes_close, without_stale
from tests.parity.test_transformer_parity import assert_same
from tests.unit.components.fakes import FakeCrucible
from tests.unit.config.test_perspective import feed_row, perspective_row

FIXTURE: Final = Path(__file__).resolve().parent.parent / "fixtures" / "parity" / "fuser.json"
ATOL: Final = 1e-8
"""The baseline fuser computes the covariance-intersection mean before symmetrizing the
covariance; `core.kalman` symmetrizes first, as the baseline tracker does. The two agree
to rounding: the largest absolute difference across the fixture is 1.9e-9, in values of
order 1e6, which matters only for components near zero, where `RTOL` alone is too tight."""

with FIXTURE.open() as handle:
    PARITY: Final[dict[str, Any]] = json.load(handle)


async def fuser_for(case: dict[str, Any]) -> Fuser:
    perspective = parse_perspective(
        "Blue",
        [
            perspective_row(
                component_track_event_dataset="ComponentTrackEvents",
                component_track_head_dataset="ComponentTrackHeads",
            ),
            feed_row("Radar"),
        ],
        {},
    )
    crucible = FakeCrucible()
    principal_heads = cast("list[JSONObject]", case["principal_heads"])
    component_heads = cast("list[JSONObject]", case["component_heads"])

    def responder(sql: str) -> list[JSONObject]:
        if "PrincipalTrackHeads" in sql:
            return principal_heads
        if "ComponentTrackHeads" in sql:
            return component_heads
        return []

    crucible.responder = responder
    writer = BatchWriter(
        crucible,
        crucible,
        WriteSettings(),
        ledger=WriteLedger(),
        budget=DrainBudget(request_seconds=1.0),
    )
    params = (
        None
        if case["passthrough"]
        else FusionParams(
            Fusion.COVARIANCE_INTERSECTION if case["ci"] else Fusion.KALMAN, case["omega"]
        )
    )
    setup = FuserSetup(
        perspective=perspective,
        datasets=FuserDatasets.of(perspective.datasets),
        params=params,
        lookback_days=30,
        clock=lambda: datetime(2026, 9, 30, tzinfo=UTC),
    )
    heads = HeadSync(
        writer,
        HeadTargets("PrincipalTrackHeads", "PrincipalTrackEvents"),
        update_interval_seconds=0.0,
        label="",
    )
    fuser = Fuser(setup, heads, None, {}, crucible)
    await fuser.prepare()
    return fuser


@pytest.mark.parametrize("name", sorted(PARITY.keys() - {"baseline_sha", "generator"}))
async def test_fuser_matches_the_baseline(name: str) -> None:
    case = PARITY[name]
    fuser = await fuser_for(case)
    for index, step in enumerate(case["steps"]):
        if "management" in step:
            await fuser.handle(MANAGEMENT, step["management"])
            continue
        heads, events = fuser.outputs(step["components"])
        events = assert_altitudes_close(events, step["events"])
        heads = assert_altitudes_close(without_stale(heads), step["heads"])
        assert_same(events, step["events"], f"step {index} events", atol=ATOL)
        assert_same(heads, step["heads"], f"step {index} heads", atol=ATOL)
