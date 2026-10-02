import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from crucible_entity_manager.core.identity import (
    canonical_identity_value,
    component_track_id,
    identity_custom_id,
    principal_track_id,
)

if TYPE_CHECKING:
    from crucible_entity_manager.core.aliases import JSONObject


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (7, "7"),
        (7.0, "7"),
        ("7.0", "7"),
        ("7", "7"),
        (np.int64(7), "7"),
        (np.float64(7.0), "7"),
        (2.5, "2.5"),
        ("2.50", "2.50"),
        ("007", "007"),
        ("abc", "abc"),
        ("v1.2", "v1.2"),
        ("", ""),
        (float("nan"), ""),
        (float("inf"), ""),
        (True, "True"),
        ([1, 2.0, "3.0"], '["1","2","3"]'),
        (366999712000123456, "366999712000123456"),
    ],
)
def test_canonical_identity_value(value: object, expected: str) -> None:
    assert canonical_identity_value(value) == expected


class TestIdentityCustomId:
    def test_wildcard_expands_identity_fields_in_sorted_order(self) -> None:
        record: JSONObject = {"identity": {"name": "ALPHA", "mmsi": 366999712}}
        assert identity_custom_id(record) == "mmsi:366999712-name:ALPHA"

    def test_empty_and_missing_values_are_omitted(self) -> None:
        record: JSONObject = {"identity": {"mmsi": 7, "name": "", "callsign": None}}
        assert identity_custom_id(record) == "mmsi:7"

    def test_equal_identities_in_different_representations_match(self) -> None:
        assert identity_custom_id({"identity": {"mmsi": 7}}) == identity_custom_id(
            {"identity": {"mmsi": "7.0"}}
        )

    def test_explicit_paths_select_only_those_fields(self) -> None:
        record: JSONObject = {"identity": {"mmsi": 7, "name": "ALPHA"}, "hull": "H1"}
        assert identity_custom_id(record, ["identity.mmsi", "hull"]) == "hull:H1-mmsi:7"

    def test_absent_explicit_path_is_omitted(self) -> None:
        assert identity_custom_id({"identity": {"mmsi": 7}}, ["identity.mmsi", "x.y"]) == "mmsi:7"

    def test_record_without_identity_gives_empty_id(self) -> None:
        assert identity_custom_id({"speed": 3}) == ""


class TestTrackIds:
    # Heads are keyed by these IDs. A change here orphans every existing head.
    def test_component_track_id_is_stable(self) -> None:
        assert (
            component_track_id("AIS_Feed", "mmsi:366999712") == "59d47a117aea529daeb17032798b0988"
        )

    def test_principal_track_id_is_stable(self) -> None:
        root = "0123456789abcdef0123456789abcdef"
        assert principal_track_id(root) == "6581a2a2291655b280745350853c0cc3"

    def test_component_track_id_depends_on_origin_dataset(self) -> None:
        assert component_track_id("A", "mmsi:7") != component_track_id("B", "mmsi:7")

    def test_ids_are_32_lowercase_hex_characters(self) -> None:
        for track_id in (component_track_id("A", ""), principal_track_id("x")):
            assert re.fullmatch(r"[0-9a-f]{32}", track_id)
