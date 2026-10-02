"""Parity of the supersede map with the baseline, using fixtures from generate.py."""

import json
from pathlib import Path
from typing import Any, Final

import pytest

from crucible_entity_manager.crucible.management import build_supersede_map

FIXTURE: Final = Path(__file__).resolve().parent.parent / "fixtures" / "parity" / "management.json"
CASES: Final[list[dict[str, Any]]] = json.loads(FIXTURE.read_text())["build_supersede_map"]


@pytest.mark.parametrize("case", CASES)
def test_build_supersede_map(case: dict[str, Any]) -> None:
    assert build_supersede_map(case["events"], case["initial"]) == case["expected"]
