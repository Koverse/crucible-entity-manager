import json
from datetime import UTC, date, datetime

import numpy as np
import pytest

from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.core.records import (
    MISSING,
    clone_record,
    compact_record,
    get_path,
    parse_records,
    remove_path,
    set_path,
)


class TestGetPath:
    def test_returns_nested_value(self) -> None:
        assert get_path({"a": {"b": {"c": 3}}}, "a.b.c") == 3

    def test_explicit_none_is_not_missing(self) -> None:
        assert get_path({"a": None}, "a") is None

    @pytest.mark.parametrize("path", ["x", "a.x", "a.b.c.d"])
    def test_absent_path_is_missing(self, path: str) -> None:
        assert get_path({"a": {"b": 1}}, path) is MISSING

    def test_scalar_in_the_middle_of_a_path_is_missing(self) -> None:
        assert get_path({"a": 5}, "a.b") is MISSING


class TestSetPath:
    def test_creates_intermediate_objects(self) -> None:
        record: JSONObject = {}
        set_path(record, "a.b.c", 1)
        assert record == {"a": {"b": {"c": 1}}}

    def test_replaces_a_scalar_intermediate(self) -> None:
        record: JSONObject = {"a": 5}
        set_path(record, "a.b", 1)
        assert record == {"a": {"b": 1}}

    def test_keeps_sibling_fields(self) -> None:
        record: JSONObject = {"a": {"keep": True}}
        set_path(record, "a.new", 2)
        assert record == {"a": {"keep": True, "new": 2}}


class TestRemovePath:
    def test_prunes_objects_left_empty(self) -> None:
        record: JSONObject = {"a": {"b": {"c": 1}}, "z": 0}
        remove_path(record, "a.b.c")
        assert record == {"z": 0}

    def test_keeps_objects_that_still_have_fields(self) -> None:
        record: JSONObject = {"a": {"b": {"c": 1}, "d": 2}}
        remove_path(record, "a.b.c")
        assert record == {"a": {"d": 2}}

    @pytest.mark.parametrize("path", ["x", "a.x", "a.b.x"])
    def test_absent_path_is_a_no_op(self, path: str) -> None:
        record: JSONObject = {"a": {"b": 1}}
        remove_path(record, path)
        assert record == {"a": {"b": 1}}


def test_clone_record_shares_no_mutable_values() -> None:
    original: JSONObject = {"a": {"b": [1, {"c": 2}]}}
    clone = clone_record(original)
    set_path(clone, "a.b", [])
    assert original == {"a": {"b": [1, {"c": 2}]}}


class TestCompactRecord:
    @pytest.mark.parametrize(
        ("record", "expected"),
        [
            ({"a": None, "b": "", "c": 0, "d": False}, {"c": 0, "d": False}),
            ({"a": float("nan"), "b": float("inf"), "c": 1.5}, {"c": 1.5}),
            ({"a": {"b": None}, "c": {"d": {}}}, {}),
            ({"a": [None, "", {}, {"b": 1}, "x"]}, {"a": [{"b": 1}, "x"]}),
            ({"a": [None, ""]}, {}),
            # Golden cases from compact_record in entity_transformer_records.py at 1b534df:
            ({"a": [[None, "A"]]}, {"a": [[None, "A"]]}),
            ({"a": [[]]}, {"a": [[]]}),
            (
                {"a": [None, "", {}, {"b": None}, "x", 0, False, [None]]},
                {"a": ["x", 0, False, [None]]},
            ),
            ({"a": [{"b": [None, 1]}, {"c": ""}]}, {"a": [{"b": [1]}]}),
            ({"a": [1.5, float("nan")]}, {"a": [1.5]}),
            ({"a": [[{"b": None}, [None, ""]]]}, {"a": [[{"b": None}, [None, ""]]]}),
            (
                {"a": [[float("inf"), np.float64(2.0), np.float64("nan")]]},
                {"a": [[None, 2.0, None]]},
            ),
            (
                {"a": np.float64(2.5), "b": np.int64(3), "c": np.True_},
                {"a": 2.5, "b": 3, "c": True},
            ),
        ],
    )
    def test_compaction_rules(self, record: dict[str, object], expected: JSONObject) -> None:
        assert compact_record(record) == expected

    def test_numpy_scalars_become_python_scalars(self) -> None:
        compacted = compact_record({"a": np.float64(2.5)})
        assert type(compacted["a"]) is float

    def test_dates_become_iso_strings(self) -> None:
        moment = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
        compacted = compact_record({"at": moment, "on": date(2026, 9, 30)})
        assert compacted == {"at": "2026-09-30T12:00:00+00:00", "on": "2026-09-30"}

    @pytest.mark.parametrize(
        "record",
        [
            {"a": object()},
            {"a": (1, 2)},
            {"a": {1: "x"}},
            {"a": [{2: "y"}]},
            {3: "z"},
            {"a": [[object()]]},
            {"a": [[{4: "w"}]]},
        ],
    )
    def test_rejects_values_that_are_not_json(self, record: dict[object, object]) -> None:
        with pytest.raises(TypeError):
            compact_record(record)  # ty: ignore[invalid-argument-type] - runtime guard under test


class TestParseRecords:
    def test_decodes_a_json_object(self) -> None:
        assert parse_records('{"a": 1}') == [{"a": 1}]

    def test_decodes_a_json_array(self) -> None:
        assert parse_records(b'[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]

    def test_copies_decoded_objects_passed_in(self) -> None:
        source: JSONObject = {"a": {"b": 1}}
        (record,) = parse_records(source)
        set_path(record, "a.b", 2)
        assert source == {"a": {"b": 1}}

    @pytest.mark.parametrize("payload", ["1", '"text"', "[1, 2]", '[{"a": 1}, 3]'])
    def test_rejects_non_object_payloads(self, payload: str) -> None:
        with pytest.raises(TypeError):
            parse_records(payload)

    def test_rejects_non_string_keys_in_decoded_objects(self) -> None:
        with pytest.raises(TypeError, match="not a string"):
            parse_records({1: "x"})  # ty: ignore[invalid-argument-type] - runtime guard under test

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_records("{not json")
