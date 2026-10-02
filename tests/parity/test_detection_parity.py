"""Parity of duplicate detection with the baseline, using fixtures from generate.py.

The fixture feeds the baseline its events in ascending time order, which is the
documented fix for its time-reversed pairing, and runs it in UTC.
"""

import json
from pathlib import Path
from typing import Any, Final

import pytest

from crucible_entity_manager.components.detection import (
    DetectionParams,
    build_histories,
    find_duplicates,
)
from tests.parity.test_transformer_parity import assert_same

FIXTURE: Final = Path(__file__).resolve().parent.parent / "fixtures" / "parity" / "duplicates.json"

with FIXTURE.open() as handle:
    PARITY: Final[dict[str, Any]] = json.load(handle)


@pytest.mark.parametrize("case", PARITY["cases"])
def test_detection_matches_the_baseline(case: dict[str, Any]) -> None:
    events = list(reversed(case["events"]))
    histories = build_histories(events, set(case["protected"]))
    found = {
        f"{duplicate.track_id_1}:{duplicate.track_id_2}": {
            "mean_mahalanobis": duplicate.mean_mahalanobis,
            "mean_velocity_difference": duplicate.mean_velocity_difference,
            "matching_points": duplicate.matching_points,
            "time_overlap_seconds": duplicate.time_overlap_seconds,
            "confidence": duplicate.confidence,
        }
        for duplicate in find_duplicates(histories, DetectionParams())
    }
    assert_same(found, case["pairs"])
