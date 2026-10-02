import pytest

from crucible_entity_manager.config.perspective import (
    ConfigError,
    LiteralMapping,
    OriginMapping,
    TrackerMode,
    TrackerSettings,
    UnitConversion,
    WriteSettings,
    parse_perspective,
)
from crucible_entity_manager.core.aliases import JSONObject, JSONValue


def perspective_row(**extra: JSONValue) -> JSONObject:
    row: JSONObject = {
        "perspective": "Blue",
        "origin_dataset": "perspective_config_Blue",
        "entity_management_event_dataset": "EntityManagementEvents",
        "report_event_dataset": "ReportEvents",
        "principal_track_event_dataset": "PrincipalTrackEvents",
        "principal_track_head_dataset": "PrincipalTrackHeads",
        "component_track_event_dataset": "ComponentTrackEvents",
        "component_track_head_dataset": "ComponentTrackHeads",
    }
    row.update(extra)
    return row


def feed_row(origin: str, **extra: JSONValue) -> JSONObject:
    row: JSONObject = {
        "perspective": "Blue",
        "origin_dataset": origin,
        "query": f"SELECT * FROM {origin}",
        "crucible_tracker": "kalman",
        "origin_to_destination_mapping": [
            {"origin_column": "callsign", "destination_column": "identity.callsign"},
        ],
    }
    row.update(extra)
    return row


def parse(*rows: JSONObject, environ: dict[str, str] | None = None) -> object:
    return parse_perspective("Blue", list(rows), environ or {})


class TestPerspectiveRow:
    def test_reads_datasets_and_defaults(self) -> None:
        config = parse_perspective("Blue", [perspective_row(), feed_row("AIS")], {})
        assert config.datasets.report_events == "ReportEvents"
        assert config.datasets.component_track_heads == "ComponentTrackHeads"
        assert config.writes == WriteSettings(
            chunk_size=250, update_chunk_size=50, max_concurrent=4
        )
        assert config.feeds[0].head_update_interval_seconds == 15.0
        assert config.restore_cooldown_seconds == 300
        assert config.preload.tracker_limit == config.preload.fusion_limit == 100_000
        assert not config.preload.skip

    def test_converts_string_values_once(self) -> None:
        row = perspective_row(
            batch_write_chunk_size="100",
            batch_write_max_concurrent=2.0,
            head_update_interval_seconds="7.5",
            skip_head_preload="TRUE",
        )
        config = parse_perspective("Blue", [row, feed_row("AIS")], {})
        assert config.writes.chunk_size == 100
        assert config.writes.max_concurrent == 2
        assert config.feeds[0].head_update_interval_seconds == 7.5
        assert config.preload.skip

    def test_environment_supplies_defaults_the_row_can_override(self) -> None:
        environ = {
            "CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS": "30",
            "CRUCIBLE_SKIP_HEAD_PRELOAD": "yes",
            "CRUCIBLE_HEAD_PRELOAD_LIMIT": "500",
        }
        config = parse_perspective(
            "Blue", [perspective_row(fusion_head_preload_limit=9), feed_row("AIS")], environ
        )
        assert config.feeds[0].head_update_interval_seconds == 30.0
        assert config.preload.skip
        assert config.preload.tracker_limit == 500
        assert config.preload.fusion_limit == 9

    def test_unused_environment_values_are_not_parsed(self) -> None:
        environ = {
            "CRUCIBLE_HEAD_PRELOAD_LIMIT": "lots",
            "CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS": "x",
        }
        row = perspective_row(
            tracker_head_preload_limit=10,
            fusion_head_preload_limit=20,
            head_update_interval_seconds=5,
        )
        config = parse_perspective("Blue", [row, feed_row("AIS")], environ)
        assert (config.preload.tracker_limit, config.preload.fusion_limit) == (10, 20)
        assert config.feeds[0].head_update_interval_seconds == 5.0

    @pytest.mark.parametrize("count", [0, 2])
    def test_requires_exactly_one_perspective_row(self, count: int) -> None:
        rows = [perspective_row() for _ in range(count)] + [feed_row("AIS")]
        with pytest.raises(ConfigError, match="exactly one row"):
            parse_perspective("Blue", rows, {})

    def test_missing_required_dataset_names_the_row_and_key(self) -> None:
        row = perspective_row()
        del row["report_event_dataset"]
        with pytest.raises(
            ConfigError, match="perspective_config_Blue: report_event_dataset is required"
        ):
            parse_perspective("Blue", [row], {})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("batch_write_chunk_size", "0"),
            ("batch_write_chunk_size", "many"),
            ("batch_write_chunk_size", True),
            ("batch_update_chunk_size", 2.5),
            ("batch_write_chunk_size", [1]),
            ("head_update_interval_seconds", "soon"),
            ("head_update_interval_seconds", [1]),
            ("head_update_interval_seconds", False),
            ("head_update_interval_seconds", "-1"),
            ("skip_head_preload", "maybe"),
            ("report_event_dataset", 5),
        ],
    )
    def test_rejects_invalid_values(self, key: str, value: JSONValue) -> None:
        with pytest.raises(ConfigError, match=key):
            parse_perspective("Blue", [perspective_row(**{key: value}), feed_row("AIS")], {})

    def test_rejects_invalid_environment_values(self) -> None:
        with pytest.raises(ConfigError, match="CRUCIBLE_HEAD_PRELOAD_LIMIT"):
            parse_perspective("Blue", [perspective_row()], {"CRUCIBLE_HEAD_PRELOAD_LIMIT": "lots"})
        with pytest.raises(ConfigError, match="CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS"):
            parse_perspective(
                "Blue",
                [perspective_row(), feed_row("AIS")],
                {"CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS": "soon"},
            )

    def test_reports_ignored_legacy_keys(self) -> None:
        config = parse_perspective(
            "Blue",
            [perspective_row(num_fusion_workers=4), feed_row("AIS", num_tracker_workers=2)],
            {},
        )
        assert any("num_fusion_workers is ignored" in warning for warning in config.warnings)
        assert any("AIS: num_tracker_workers is ignored" in warning for warning in config.warnings)


class TestFeeds:
    def test_feed_inherits_perspective_settings_and_can_override_them(self) -> None:
        config = parse_perspective(
            "Blue",
            [
                perspective_row(batch_write_chunk_size=100),
                feed_row("AIS", batch_write_chunk_size=10),
            ],
            {},
        )
        (feed,) = config.feeds
        assert feed.datasets.report_events == "ReportEvents"
        assert feed.writes.chunk_size == 10

    def test_perspective_head_update_interval_overrides_feeds(self) -> None:
        config = parse_perspective(
            "Blue",
            [
                perspective_row(head_update_interval_seconds=30),
                feed_row("AIS", head_update_interval_seconds=90),
            ],
            {},
        )
        assert config.feeds[0].head_update_interval_seconds == 30.0
        assert config.feeds[0].row["head_update_interval_seconds"] == 30

    def test_feed_head_update_interval_applies_when_the_perspective_sets_none(self) -> None:
        config = parse_perspective(
            "Blue",
            [perspective_row(), feed_row("A", head_update_interval_seconds=90), feed_row("B")],
            {},
        )
        assert [feed.head_update_interval_seconds for feed in config.feeds] == [90.0, 15.0]

    def test_parses_mappings_conversions_and_hooks(self) -> None:
        feed = parse_perspective(
            "Blue",
            [
                perspective_row(),
                feed_row(
                    "AIS",
                    origin_to_destination_mapping=[
                        {"origin_column": "mmsi", "destination_column": "identity.mmsi"},
                        {"literal": "SEA_SURFACE", "destination_column": "identity.environment"},
                    ],
                    unit_conversions=[
                        {"origin_column": "lat", "unit_conversion": "to_radians"},
                        {"origin_column": "lon", "unit_conversion": "PLACEHOLDER_function"},
                    ],
                    custom_functions=[{"function_name": "fix_names"}],
                ),
            ],
            {},
        ).feeds[0]
        assert feed.mappings == (
            OriginMapping("mmsi", "identity.mmsi"),
            LiteralMapping("SEA_SURFACE", "identity.environment"),
        )
        assert feed.unit_conversions == (UnitConversion("lat", "to_radians"),)
        assert feed.custom_functions == ("fix_names",)

    @pytest.mark.parametrize(
        ("mapping", "problem"),
        [
            ([{"origin_column": "a", "destination_column": "speed"}], "identity"),
            (
                [
                    {"origin_column": "a", "destination_column": "identity.x"},
                    {"origin_column": "b", "destination_column": "identity.x"},
                ],
                "more than one origin",
            ),
            ([{"origin_column": "a"}], "destination_column"),
            ([{"destination_column": "identity.x"}], "exactly one"),
            (
                [{"origin_column": "a", "literal": 1, "destination_column": "identity.x"}],
                "exactly one",
            ),
            ("identity.x", "list of objects"),
            ([{"origin_column": 5, "destination_column": "identity.x"}], "must be a path"),
        ],
    )
    def test_rejects_invalid_mappings(self, mapping: JSONValue, problem: str) -> None:
        with pytest.raises(ConfigError, match=problem):
            parse_perspective(
                "Blue",
                [perspective_row(), feed_row("AIS", origin_to_destination_mapping=mapping)],
                {},
            )

    @pytest.mark.parametrize(
        ("key", "entries"),
        [
            ("unit_conversions", [{"unit_conversion": "to_radians"}]),
            ("custom_functions", [{"name": "tag"}]),
        ],
    )
    def test_rejects_incomplete_hook_entries(self, key: str, entries: JSONValue) -> None:
        with pytest.raises(ConfigError, match=f"AIS: {key}"):
            parse_perspective("Blue", [perspective_row(), feed_row("AIS", **{key: entries})], {})

    def test_mapping_is_required(self) -> None:
        row = feed_row("AIS")
        del row["origin_to_destination_mapping"]
        with pytest.raises(ConfigError, match="AIS: origin_to_destination_mapping is required"):
            parse_perspective("Blue", [perspective_row(), row], {})

    def test_literal_destinations_may_repeat(self) -> None:
        mapping: JSONValue = [
            {"origin_column": "a", "destination_column": "identity.x"},
            {"literal": 1, "destination_column": "mode"},
            {"literal": 2, "destination_column": "mode"},
        ]
        config = parse_perspective(
            "Blue", [perspective_row(), feed_row("AIS", origin_to_destination_mapping=mapping)], {}
        )
        assert len(config.feeds[0].mappings) == 3

    def test_track_id_fields_default_and_override(self) -> None:
        config = parse_perspective(
            "Blue",
            [
                perspective_row(),
                feed_row("A"),
                feed_row("B", track_id_fields=[" identity.callsign ", "source.datasetName"]),
            ],
            {},
        )
        assert config.feeds[0].track_id_fields == ("identity.*",)
        assert config.feeds[1].track_id_fields == ("identity.callsign", "source.datasetName")

    @pytest.mark.parametrize("fields", [[], "identity.callsign", ["identity.x", ""], [1]])
    def test_rejects_invalid_track_id_fields(self, fields: JSONValue) -> None:
        with pytest.raises(ConfigError, match="track_id_fields"):
            parse_perspective(
                "Blue", [perspective_row(), feed_row("A", track_id_fields=fields)], {}
            )

    @pytest.mark.parametrize("version", [2, "2", "native", "Records"])
    def test_accepts_the_v2_hook_contract(self, version: JSONValue) -> None:
        parse_perspective("Blue", [perspective_row(), feed_row("A", hook_api_version=version)], {})

    @pytest.mark.parametrize("key", ["custom_function_api_version", "hook_api_version"])
    def test_rejects_the_v1_hook_contract(self, key: str) -> None:
        with pytest.raises(ConfigError, match="v2 record contract"):
            parse_perspective("Blue", [perspective_row(), feed_row("A", **{key: 1})], {})

    def test_rejects_duplicate_origin_datasets(self) -> None:
        with pytest.raises(ConfigError, match="more than one feed"):
            parse_perspective("Blue", [perspective_row(), feed_row("A"), feed_row("A")], {})

    def test_hook_scripts_are_perspective_wide(self) -> None:
        rows = [
            perspective_row(custom_functions_script_name="hooks_v1"),
            feed_row("A", custom_functions_script_name="hooks_v1"),
            feed_row("B", custom_functions_script_name="hooks_v2"),
        ]
        with pytest.raises(ConfigError, match="B: custom_functions_script_name"):
            parse_perspective("Blue", rows, {})


class TestEnablement:
    def test_disabled_feeds_are_skipped_and_reported(self) -> None:
        config = parse_perspective(
            "Blue",
            [
                perspective_row(),
                feed_row("A"),
                feed_row("B", disabled=True),
                feed_row("C", disabled="true"),
            ],
            {},
        )
        assert [feed.origin_dataset for feed in config.feeds] == ["A"]
        assert config.disabled_feeds == ("B", "C")

    def test_disabled_feeds_are_not_validated(self) -> None:
        broken = feed_row("B", disabled=True, origin_to_destination_mapping=[])
        config = parse_perspective("Blue", [perspective_row(), feed_row("A"), broken], {})
        assert [feed.origin_dataset for feed in config.feeds] == ["A"]

    @pytest.mark.parametrize("key", ["disable_all_other_datasets", "disable_all_other_datsets"])
    def test_focus_flag_keeps_only_that_feed(self, key: str) -> None:
        config = parse_perspective(
            "Blue",
            [perspective_row(), feed_row("A"), feed_row("B", **{key: True}), feed_row("C")],
            {},
        )
        assert [feed.origin_dataset for feed in config.feeds] == ["B"]
        assert config.disabled_feeds == ("A", "C")

    def test_focus_overrides_the_disabled_flag(self) -> None:
        rows = [perspective_row(), feed_row("A", disabled=True, disable_all_other_datasets=True)]
        assert [feed.origin_dataset for feed in parse_perspective("Blue", rows, {}).feeds] == ["A"]

    def test_only_one_feed_may_set_focus(self) -> None:
        rows = [
            perspective_row(),
            feed_row("A", disable_all_other_datasets=True),
            feed_row("B", disable_all_other_datsets="true"),
        ]
        with pytest.raises(ConfigError, match="only one feed"):
            parse_perspective("Blue", rows, {})


@pytest.mark.parametrize(
    ("value", "mode"),
    [
        ("", TrackerMode.SKIP),
        ("skip", TrackerMode.SKIP),
        ("3rd party", TrackerMode.SKIP),
        ("third-party", TrackerMode.SKIP),
        ("passthrough", TrackerMode.PASSTHROUGH),
        ("Kalman", TrackerMode.KALMAN),
        ("ECEF q=0.1", TrackerMode.KALMAN),
        ("passthrough kalman", TrackerMode.KALMAN),
        ("ci", TrackerMode.KALMAN_CI),
        ("kalman CI", TrackerMode.KALMAN_CI),
        ("covariance intersection", TrackerMode.KALMAN_CI),
        ("cubic", TrackerMode.UNRECOGNIZED),
    ],
)
def test_tracker_modes(value: str, mode: TrackerMode) -> None:
    assert TrackerSettings.parse(value).mode is mode


@pytest.mark.parametrize(
    ("value", "q"), [("kalman q=2.5", 2.5), ("ECEF q = 3", 3.0), ("kalman", None)]
)
def test_tracker_process_noise_override(value: str, q: float | None) -> None:
    assert TrackerSettings.parse(value).process_noise_q == q
