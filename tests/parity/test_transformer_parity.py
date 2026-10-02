"""Parity of the transformer with the baseline, using fixtures from generate.py."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest

from crucible_entity_manager.components.transformer import FeedPipeline, dataset_name, transform
from crucible_entity_manager.config.perspective import parse_perspective
from crucible_entity_manager.hooks.loader import load_hook_modules, resolve_feed_hooks

if TYPE_CHECKING:
    from crucible_entity_manager.core.aliases import JSONObject

FIXTURE: Final = Path(__file__).resolve().parent.parent / "fixtures" / "parity" / "transformer.json"
RTOL: Final = 1e-9

with FIXTURE.open() as handle:
    PARITY: Final[dict[str, Any]] = json.load(handle)


@pytest.fixture(scope="module")
def pipeline() -> FeedPipeline:
    perspective = parse_perspective("Parity", [PARITY["perspective_row"], PARITY["feed_row"]], {})
    rows: list[JSONObject] = [
        {"script_name": name, "script_body": body} for name, body in PARITY["scripts"].items()
    ]
    modules = load_hook_modules(rows, perspective.scripts)
    (feed,) = perspective.feeds
    assert feed.query is not None
    return FeedPipeline(feed, resolve_feed_hooks(feed, modules), dataset_name(feed.query))


def assert_same(actual: object, expected: object, path: str = "", *, atol: float = 0.0) -> None:
    """Compare decoded JSON, floats to `RTOL` (and `atol`), reporting where they differ."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            assert_same(actual[key], value, f"{path}.{key}", atol=atol)
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
            assert_same(got, want, f"{path}[{index}]", atol=atol)
    elif isinstance(expected, float):
        assert isinstance(actual, float), path
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=atol, err_msg=path)
    else:
        assert actual == expected, path


@pytest.mark.parametrize("case", PARITY["cases"])
def test_transform_matches_the_baseline(case: dict[str, Any], pipeline: FeedPipeline) -> None:
    assert_same(list(transform(case["records"], pipeline)), case["expected"])
