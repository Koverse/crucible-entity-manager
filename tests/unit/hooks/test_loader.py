import sys

import pytest

from crucible_entity_manager.config.perspective import FeedConfig, ScriptNames, parse_perspective
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.hooks.loader import (
    HookError,
    hook_row,
    load_hook_modules,
    resolve_feed_hooks,
    validate_record_batch,
)

UNIT_CONVERSIONS = """
import math

def to_radians(value):
    return math.radians(value)

NOT_A_FUNCTION = 3
"""

CUSTOM_FUNCTIONS = """
def tag(records, row):
    return [{**record, "tagged": row["origin_dataset"]} for record in records]
"""

SCRIPTS = ScriptNames(custom_functions="custom", unit_conversions="units")
ROWS: list[JSONObject] = [
    {"script_name": "units", "script_body": UNIT_CONVERSIONS},
    {"script_name": "custom", "script_body": CUSTOM_FUNCTIONS},
    {"script_name": "unrelated", "script_body": "raise RuntimeError('never loaded')"},
    {"script_name": "malformed", "script_body": None},
]


def feed(**extra: JSONValue) -> FeedConfig:
    perspective: JSONObject = {
        "origin_dataset": "perspective_config_Blue",
        "entity_management_event_dataset": "M",
        "report_event_dataset": "R",
        "principal_track_event_dataset": "PE",
        "principal_track_head_dataset": "PH",
    }
    row: JSONObject = {
        "origin_dataset": "AIS",
        "origin_to_destination_mapping": [
            {"origin_column": "mmsi", "destination_column": "identity.mmsi"}
        ],
    }
    row.update(extra)
    return parse_perspective("Blue", [perspective, row], {}).feeds[0]


class TestLoadHookModules:
    def test_compiles_only_the_named_scripts(self) -> None:
        modules = load_hook_modules(ROWS, SCRIPTS)
        assert modules.unit_conversions is not None
        assert modules.custom_functions is not None
        assert modules.unit_conversions.to_radians(180) == pytest.approx(3.141592653589793)

    def test_no_scripts_named_loads_nothing(self) -> None:
        modules = load_hook_modules(ROWS, ScriptNames(None, None))
        assert modules.custom_functions is None
        assert modules.unit_conversions is None

    def test_row_without_a_body_does_not_count_as_the_script(self) -> None:
        with pytest.raises(HookError, match="'malformed' is not in"):
            load_hook_modules(
                ROWS, ScriptNames(custom_functions="malformed", unit_conversions=None)
            )

    def test_missing_script_row(self) -> None:
        with pytest.raises(HookError, match="'absent' is not in Entity_Stream_Manager_Functions"):
            load_hook_modules(ROWS, ScriptNames(custom_functions="absent", unit_conversions=None))

    def test_script_that_fails_to_load(self) -> None:
        with pytest.raises(HookError, match="'unrelated' failed to load: never loaded"):
            load_hook_modules(
                ROWS, ScriptNames(custom_functions="unrelated", unit_conversions=None)
            )

    def test_custom_functions_can_import_unit_conversions(self) -> None:
        rows: list[JSONObject] = [
            {"script_name": "units", "script_body": UNIT_CONVERSIONS},
            {
                "script_name": "custom",
                "script_body": (
                    "import unit_conversions\nhalf_turn = unit_conversions.to_radians(180)\n"
                ),
            },
        ]
        modules = load_hook_modules(rows, SCRIPTS)
        assert modules.custom_functions is not None
        assert modules.custom_functions.half_turn == pytest.approx(3.141592653589793)
        assert sys.modules["unit_conversions"] is modules.unit_conversions

    def test_failed_script_is_not_left_registered(self) -> None:
        sys.modules.pop("custom_functions", None)
        with pytest.raises(HookError):
            load_hook_modules(
                ROWS, ScriptNames(custom_functions="unrelated", unit_conversions=None)
            )
        assert "custom_functions" not in sys.modules

    def test_each_load_replaces_earlier_registrations(self) -> None:
        load_hook_modules(ROWS, SCRIPTS)
        load_hook_modules(ROWS, ScriptNames(custom_functions="custom", unit_conversions=None))
        assert "unit_conversions" not in sys.modules
        assert "custom_functions" in sys.modules

    def test_failed_reload_leaves_nothing_registered(self) -> None:
        load_hook_modules(ROWS, SCRIPTS)
        with pytest.raises(HookError):
            load_hook_modules(
                ROWS, ScriptNames(custom_functions="unrelated", unit_conversions=None)
            )
        assert "custom_functions" not in sys.modules

    def test_scripts_are_isolated_modules(self) -> None:
        first = load_hook_modules(ROWS, SCRIPTS)
        second = load_hook_modules(ROWS, SCRIPTS)
        assert first.unit_conversions is not second.unit_conversions


class TestResolveFeedHooks:
    def test_resolves_hooks_in_configured_order(self) -> None:
        hooks = resolve_feed_hooks(
            feed(
                unit_conversions=[{"origin_column": "lat", "unit_conversion": "to_radians"}],
                custom_functions=[{"function_name": "tag"}],
            ),
            load_hook_modules(ROWS, SCRIPTS),
        )
        ((conversion, to_radians),) = hooks.value_hooks
        assert conversion.origin == "lat"
        assert to_radians(90) == pytest.approx(1.5707963267948966)
        ((name, tag),) = hooks.record_hooks
        assert name == "tag"
        assert tag([{"a": 1}], {"origin_dataset": "AIS"}) == [{"a": 1, "tagged": "AIS"}]

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (
                {"unit_conversions": [{"origin_column": "lat", "unit_conversion": "absent"}]},
                "'absent' is not a function in the unit conversions script",
            ),
            (
                {
                    "unit_conversions": [
                        {"origin_column": "lat", "unit_conversion": "NOT_A_FUNCTION"}
                    ]
                },
                "'NOT_A_FUNCTION' is not a function",
            ),
            (
                {"custom_functions": [{"function_name": "absent"}]},
                "AIS: hook 'absent' is not a function in the custom functions script",
            ),
        ],
    )
    def test_rejects_missing_functions(self, extra: dict[str, JSONValue], message: str) -> None:
        with pytest.raises(HookError, match=message):
            resolve_feed_hooks(feed(**extra), load_hook_modules(ROWS, SCRIPTS))

    def test_hook_configured_without_its_script(self) -> None:
        with pytest.raises(HookError, match="custom functions script"):
            resolve_feed_hooks(
                feed(custom_functions=[{"function_name": "tag"}]),
                load_hook_modules(ROWS, ScriptNames(None, None)),
            )


def test_hook_row_is_a_fresh_deep_copy() -> None:
    config = feed(
        origin_to_destination_mapping=[
            {"origin_column": "mmsi", "destination_column": "identity.mmsi"}
        ]
    )
    first = hook_row(config)
    mapping = first["origin_to_destination_mapping"]
    assert isinstance(mapping, list)
    mapping.clear()
    first["query"] = "changed"
    second = hook_row(config)
    assert second["origin_to_destination_mapping"] == [
        {"origin_column": "mmsi", "destination_column": "identity.mmsi"}
    ]
    assert "query" not in second


class TestValidateRecordBatch:
    def test_returns_copies_the_hook_cannot_mutate(self) -> None:
        returned = [{"identity": {"callsign": "ALPHA"}}]
        (record,) = validate_record_batch(returned, "tag")
        record["identity"] = {}
        assert returned == [{"identity": {"callsign": "ALPHA"}}]

    @pytest.mark.parametrize(
        ("result", "message"),
        [
            ({"id": 1}, "must return a list, got dict"),
            ([{"id": 1}, "x"], "non-object at index 1"),
            ([{1: "x"}], "non-object at index 0"),
        ],
    )
    def test_rejects_results_that_are_not_record_lists(self, result: object, message: str) -> None:
        with pytest.raises(HookError, match=message):
            validate_record_batch(result, "tag")
