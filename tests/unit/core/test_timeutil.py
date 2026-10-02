from datetime import UTC, datetime, timedelta, timezone

import pytest

from crucible_entity_manager.core.aliases import JSONValue
from crucible_entity_manager.core.timeutil import format_timestamp, parse_timestamp, utc_now


class TestParseTimestamp:
    @pytest.mark.parametrize(
        "text",
        [
            "2026-09-30T12:34:56.789Z",
            "2026-09-30T12:34:56.789000Z",
            "2026-09-30T12:34:56.789+00:00",
            "2026-09-30T14:34:56.789+02:00",
        ],
    )
    def test_parses_iso_variants_to_the_same_utc_instant(self, text: str) -> None:
        assert parse_timestamp(text) == datetime(2026, 9, 30, 12, 34, 56, 789000, tzinfo=UTC)

    def test_accepts_timestamps_without_fractional_seconds(self) -> None:
        assert parse_timestamp("2026-09-30T12:34:56Z") == datetime(
            2026, 9, 30, 12, 34, 56, tzinfo=UTC
        )

    def test_naive_text_is_taken_as_utc(self) -> None:
        assert parse_timestamp("2026-09-30T12:00:00") == datetime(2026, 9, 30, 12, tzinfo=UTC)

    def test_converts_aware_datetimes_to_utc(self) -> None:
        eastern = datetime(2026, 9, 30, 8, tzinfo=timezone(timedelta(hours=-4)))
        parsed = parse_timestamp(eastern)
        assert parsed == datetime(2026, 9, 30, 12, tzinfo=UTC)
        assert parsed is not None
        assert parsed.tzinfo is UTC

    @pytest.mark.parametrize("value", [None, "", "not a time", 1727690000, 1.5, [], {}])
    def test_unparseable_values_give_none(self, value: JSONValue) -> None:
        assert parse_timestamp(value) is None


class TestFormatTimestamp:
    def test_formats_milliseconds_with_z_suffix(self) -> None:
        moment = datetime(2026, 9, 30, 12, 34, 56, 789999, tzinfo=UTC)
        assert format_timestamp(moment) == "2026-09-30T12:34:56.789Z"

    def test_converts_to_utc_before_formatting(self) -> None:
        moment = datetime(2026, 9, 30, 14, tzinfo=timezone(timedelta(hours=2)))
        assert format_timestamp(moment) == "2026-09-30T12:00:00.000Z"

    def test_rejects_naive_datetimes(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            format_timestamp(datetime(2026, 9, 30))  # noqa: DTZ001 - the naive value is the point

    def test_round_trips_through_parse(self) -> None:
        moment = datetime(2026, 9, 30, 12, 34, 56, 789000, tzinfo=UTC)
        assert parse_timestamp(format_timestamp(moment)) == moment


def test_utc_now_is_aware_utc() -> None:
    assert utc_now().tzinfo is UTC
