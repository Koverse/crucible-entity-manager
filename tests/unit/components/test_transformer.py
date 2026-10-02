import logging
import math
from typing import cast

import pytest

from crucible_entity_manager.components.transformer import (
    UNKNOWN_DATASET_NAME,
    FeedPipeline,
    RecordOwner,
    Transformer,
    build,
    dataset_name,
    select_feeds,
    transform,
)
from crucible_entity_manager.config.perspective import (
    ConfigError,
    FeedConfig,
    PerspectiveConfig,
    UnitConversion,
    parse_perspective,
)
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.partition import PartitionSpec
from crucible_entity_manager.core.records import get_path
from crucible_entity_manager.hooks.loader import FUNCTIONS_DATASET, FeedHooks, HookError
from tests.unit.components.fakes import FakeCrucible, FakeServices
from tests.unit.config.test_perspective import feed_row, perspective_row

MAPPINGS: list[JSONValue] = [
    {"origin_column": "source.id", "destination_column": "trackId.uuid"},
    {"origin_column": "source.id", "destination_column": "identity.alias"},
    {"origin_column": "id", "destination_column": "identity.id"},
    {"origin_column": "position.latitude", "destination_column": "geodetic.latitude"},
    {"origin_column": "position.longitude", "destination_column": "geodetic.longitude"},
    {"origin_column": "derived.quality", "destination_column": "trackQuality"},
    {"literal": "AIR", "destination_column": "identity.environment"},
]


def smuggled(value: object) -> JSONValue:
    """Pass off a value JSON can't represent, as a misbehaving hook would."""
    return cast("JSONValue", value)


def perspective(*feeds: JSONObject, **extra: JSONValue) -> PerspectiveConfig:
    return parse_perspective("Blue", [perspective_row(**extra), *feeds], {})


def feed(**extra: JSONValue) -> FeedConfig:
    row = feed_row("AIS", origin_to_destination_mapping=MAPPINGS, **extra)
    (config,) = perspective(row).feeds
    return config


def pipeline(
    config: FeedConfig | None = None,
    *,
    value_hooks: FeedHooks | None = None,
) -> FeedPipeline:
    return FeedPipeline(config or feed(), value_hooks or FeedHooks((), ()), "AIS")


class TestTransform:
    def test_maps_nested_paths_one_to_many_and_drops_mapped_origins(self) -> None:
        records: list[JSONObject] = [
            {
                "id": 1,
                "source": {"id": "abc", "kept": True},
                "position": {"latitude": 0.5, "longitude": -1.0},
                "unmapped": "dropped",
                "crucibleHeader": {"uuid": "origin-1"},
            }
        ]
        assert transform(records, pipeline()) == [
            {
                "trackId": {"uuid": "abc"},
                "identity": {"alias": "abc", "id": 1, "environment": "AIR"},
                "geodetic": {"latitude": 0.5, "longitude": -1.0},
                "source": {"kept": True, "datasetName": "AIS", "uuid": "origin-1"},
            }
        ]

    def test_the_origin_uuid_survives_a_mapping_that_moves_it(self) -> None:
        moved: list[JSONValue] = [
            *MAPPINGS,
            {"origin_column": "crucibleHeader.uuid", "destination_column": "identity.originId"},
        ]
        (config,) = perspective(feed_row("AIS", origin_to_destination_mapping=moved)).feeds
        (report,) = transform([{"id": 1, "crucibleHeader": {"uuid": "u1"}}], pipeline(config))
        assert get_path(report, "identity.originId") == "u1"
        assert report["source"] == {"datasetName": "AIS", "uuid": "u1"}

    def test_records_are_not_modified(self) -> None:
        records: list[JSONObject] = [{"id": 1, "source": {"id": "abc"}}]
        transform(records, pipeline())
        assert records == [{"id": 1, "source": {"id": "abc"}}]

    def test_reports_without_mapped_values_still_carry_their_source(self) -> None:
        assert transform([{"other": 1}], pipeline()) == [
            {"identity": {"environment": "AIR"}, "source": {"datasetName": "AIS"}}
        ]

    def test_unit_conversions_apply_before_mapping(self) -> None:
        def to_radians(value: JSONValue) -> JSONValue:
            assert isinstance(value, float)
            return math.radians(value)

        hooks = FeedHooks(((UnitConversion("position.latitude", "to_radians"), to_radians),), ())
        (report,) = transform(
            [{"id": 1, "position": {"latitude": 45.0}}], pipeline(value_hooks=hooks)
        )
        assert report["geodetic"] == {"latitude": math.pi / 4}

    def test_a_failed_unit_conversion_drops_only_that_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def strict(value: JSONValue) -> JSONValue:
            if not isinstance(value, float):
                raise TypeError("not a number")
            return value * 2

        hooks = FeedHooks(((UnitConversion("position.latitude", "double"), strict),), ())
        records: list[JSONObject] = [
            {"id": 1, "position": {"latitude": 1.0}},
            {"id": 2, "position": {"latitude": "bad"}},
            {"id": 3, "position": {"latitude": None}},
            {"id": 4},
        ]
        reports = transform(records, pipeline(value_hooks=hooks))
        assert [get_path(report, "identity.id") for report in reports] == [1, 3, 4]
        assert reports[0]["geodetic"] == {"latitude": 2.0}
        assert "Dropped a record: unit conversion double failed on position.latitude" in caplog.text

    def test_custom_functions_run_in_order_on_fresh_rows(self) -> None:
        rows: list[JSONObject] = []
        already_mutated: list[bool] = []

        def label(records: list[JSONObject], row: object) -> list[JSONObject]:
            assert isinstance(row, dict)
            rows.append(row)
            already_mutated.append("mutated" in row)
            row["mutated"] = True
            for record in records:
                record["derived"] = {"quality": 0.95}
            return records

        def keep_first(records: list[JSONObject], row: object) -> list[JSONObject]:
            del row
            return records[:1]

        hooks = FeedHooks((), (("label", label), ("keep_first", keep_first)))
        config = feed()
        batch: list[JSONObject] = [{"id": 1}, {"id": 2}]
        (report,) = transform(batch, pipeline(config, value_hooks=hooks))
        assert report["trackQuality"] == 0.95
        transform(batch, pipeline(config, value_hooks=hooks))
        assert rows[0] is not rows[1]
        assert already_mutated == [False, False]
        assert "mutated" not in config.row

    @pytest.mark.parametrize(
        "result",
        [pytest.param({"id": 1}, id="not a list"), pytest.param([1], id="not a dict")],
    )
    def test_a_failing_custom_function_is_skipped_for_the_batch(
        self, result: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(records: list[JSONObject], row: object) -> list[JSONObject]:
            del records, row
            return cast("list[JSONObject]", result)

        def raises(records: list[JSONObject], row: object) -> list[JSONObject]:
            raise RuntimeError("boom")

        hooks = FeedHooks((), (("broken", broken), ("raises", raises)))
        (report,) = transform([{"id": 1}], pipeline(value_hooks=hooks))
        assert report["identity"] == {"id": 1, "environment": "AIR"}
        assert "Custom function broken failed; skipped for this batch" in caplog.text
        assert "Custom function raises failed; skipped for this batch" in caplog.text

    def test_records_a_hook_leaves_unrepresentable_are_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def add_set(records: list[JSONObject], row: object) -> list[JSONObject]:
            del row
            records[0]["derived"] = smuggled({"quality": {1, 2}})
            return records

        hooks = FeedHooks((), (("add_set", add_set),))
        reports = transform([{"id": 1}, {"id": 2}], pipeline(value_hooks=hooks))
        assert [report["identity"] for report in reports] == [{"id": 2, "environment": "AIR"}]
        assert "Dropped 1 record(s) holding values JSON cannot represent" in caplog.text

    def test_adds_ecef_kinematics(self) -> None:
        config = feed_row(
            "AIS",
            origin_to_destination_mapping=[
                *MAPPINGS,
                {
                    "origin_column": "ts",
                    "destination_column": "estimatedKinematics.kinematicsTimestamp",
                },
            ],
        )
        (feed_config,) = perspective(config).feeds
        (report,) = transform(
            [
                {
                    "id": 1,
                    "ts": "2026-09-30T00:00:00Z",
                    "position": {"latitude": 0.0, "longitude": 0.0},
                }
            ],
            pipeline(feed_config),
        )
        assert report["ecefPosition"] == {"x": 6_378_137.0, "y": 0.0, "z": 0.0}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("SELECT * FROM AIS WHERE x = 1", "AIS"),
        ("select a from 'Quoted_Feed'", "Quoted_Feed"),
        ('SELECT * FROM "Double" LIMIT 5', "Double"),
        (f"SELECT * FROM {'L' * 50}", "L" * 40),
        ("SHOW TABLES", UNKNOWN_DATASET_NAME),
    ],
)
def test_dataset_name(query: str, expected: str) -> None:
    assert dataset_name(query) == expected


class TestRecordOwner:
    def test_partitions_records_by_origin_uuid(self) -> None:
        records: list[JSONObject] = [
            {"crucibleHeader": {"uuid": f"{index:032x}"}} for index in range(64)
        ]
        owners = [RecordOwner(PartitionSpec(index, 4)) for index in range(4)]
        counts = [sum(owner(record) for owner in owners) for record in records]
        assert counts == [1] * 64
        assert all(any(owner(record) for record in records) for owner in owners)

    @pytest.mark.parametrize(
        "record", [{}, {"crucibleHeader": {"uuid": ""}}, {"crucibleHeader": {"uuid": 7}}]
    )
    def test_records_without_an_origin_uuid_belong_to_partition_zero(
        self, record: JSONObject
    ) -> None:
        assert RecordOwner(PartitionSpec(0, 3))(record)
        assert not RecordOwner(PartitionSpec(1, 3))(record)


def settings(feed: str | None = None) -> RuntimeSettings:
    return RuntimeSettings(component="transformer", perspective="Blue", feed=feed)


class TestBuild:
    async def test_subscribes_to_every_feed_and_writes_reports(self) -> None:
        services = FakeServices(
            settings(),
            perspective(
                feed_row("AIS", query="SELECT * FROM AIS", source_queue_max_records=10),
                feed_row("ADSB", query="SELECT * FROM ADSB", source_queue_max_records=20),
            ),
        )
        component = await build(services)
        assert isinstance(component, Transformer)
        assert component.queue_max_records == 30
        assert [subscription.name for subscription in component.subscriptions()] == ["AIS", "ADSB"]
        assert services.sources["ADSB"][0] == "SELECT * FROM ADSB"
        assert services.crucible.queries == []
        await component.prepare()
        await component.handle("ADSB", [{"callsign": "ALPHA", "crucibleHeader": {"uuid": "u1"}}])
        await component.handle("AIS", [])
        await component.tick()
        await component.close()
        assert services.crucible.posted == {
            "ReportEvents": [
                {
                    "identity": {"callsign": "ALPHA"},
                    "source": {"datasetName": "ADSB", "uuid": "u1"},
                }
            ]
        }
        assert services.ledger.snapshot.clean

    async def test_runs_only_the_selected_feed(self) -> None:
        services = FakeServices(
            settings(feed="ADSB"),
            perspective(
                feed_row("AIS", query="SELECT * FROM AIS"),
                feed_row("ADSB", query="SELECT * FROM ADSB"),
            ),
        )
        component = await build(services)
        assert [subscription.name for subscription in component.subscriptions()] == ["ADSB"]

    async def test_a_feed_without_a_query_is_a_configuration_error(self) -> None:
        services = FakeServices(settings(), perspective(feed_row("AIS", query=None)))
        with pytest.raises(ConfigError, match="AIS: the transformer needs a query"):
            await build(services)

    async def test_loads_hooks_the_perspective_names(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        script = (
            "def shout(records, row):\n"
            "    return [{**r, 'callsign': r['callsign'].upper()} for r in records]\n"
        )

        def responder(sql: str) -> list[JSONObject]:
            assert sql == f"SELECT * FROM {FUNCTIONS_DATASET}"
            return [{"script_name": "functions", "script_body": script}]

        services = FakeServices(
            settings(),
            perspective(
                feed_row(
                    "AIS", query="SELECT * FROM AIS", custom_functions=[{"function_name": "shout"}]
                ),
                custom_functions_script_name="functions",
            ),
            crucible=FakeCrucible(responder),
        )
        component = await build(services)
        await component.handle("AIS", [{"callsign": "alpha"}])
        assert services.crucible.posted["ReportEvents"][0]["identity"] == {"callsign": "ALPHA"}
        assert "Wrote 1 of 1 report(s)" in caplog.text

    async def test_a_missing_hook_function_fails_the_build(self) -> None:
        services = FakeServices(
            settings(),
            perspective(
                feed_row(
                    "AIS", query="SELECT * FROM AIS", custom_functions=[{"function_name": "absent"}]
                ),
                custom_functions_script_name="functions",
            ),
            crucible=FakeCrucible(lambda sql: [{"script_name": "functions", "script_body": ""}]),
        )
        with pytest.raises(HookError, match="'absent' is not a function"):
            await build(services)


class TestSelectFeeds:
    def test_all_feeds_or_the_named_one(self) -> None:
        feeds = perspective(feed_row("AIS"), feed_row("ADSB")).feeds
        assert [config.origin_dataset for config in select_feeds(feeds, None)] == ["AIS", "ADSB"]
        assert [config.origin_dataset for config in select_feeds(feeds, "AIS")] == ["AIS"]

    def test_an_unknown_feed_is_a_configuration_error(self) -> None:
        feeds = perspective(feed_row("AIS")).feeds
        with pytest.raises(ConfigError, match="no enabled feed is named 'Other'"):
            select_feeds(feeds, "Other")
        with pytest.raises(ConfigError, match="no enabled feeds"):
            select_feeds((), None)
