from datetime import UTC, datetime, timedelta

import pytest

from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.crucible.management import (
    Action,
    build_supersede_map,
    duplicate_protected_ids,
    fetch_management_events,
    load_supersede_map,
    resolve_root,
)
from crucible_entity_manager.crucible.protocols import RequestError, TransientError

NOW = datetime(2026, 9, 30, 12, tzinfo=UTC)


def event(track_id: str, action: str, minute: int, superseded_by: JSONValue = None) -> JSONObject:
    record: JSONObject = {
        "trackId": track_id,
        "action": action,
        "crucibleHeader": {"updatedDate": f"2026-09-30T11:{minute:02d}:00Z"},
    }
    if action == "SUPERSEDE":
        record["supersededBy"] = superseded_by
    return record


class FakeSearch:
    """Answers each query from `pages`, recording the SQL it was given."""

    def __init__(self, *pages: list[JSONObject] | Exception) -> None:
        self.pages = list(pages)
        self.queries: list[tuple[str, bool]] = []

    async def search(self, sql: str, *, auto_backtick: bool = True) -> list[JSONObject]:
        self.queries.append((sql, auto_backtick))
        page = self.pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page


class TestBuildSupersedeMap:
    def test_chains_resolve_to_the_final_survivor(self) -> None:
        result = build_supersede_map(
            [event("A", "SUPERSEDE", 1, "B"), event("B", "SUPERSEDE", 2, "C")]
        )
        assert result == {"A": "C", "B": "C"}

    def test_chain_ending_in_deletion_maps_to_none(self) -> None:
        result = build_supersede_map(
            [
                event("A", "SUPERSEDE", 1, "B"),
                event("B", "SUPERSEDE", 2, "C"),
                event("C", "DELETE", 3),
            ]
        )
        assert result == {"A": None, "B": None, "C": None}

    def test_latest_event_wins_in_any_row_order(self) -> None:
        events = [event("A", "SUPERSEDE", 2, "C"), event("A", "SUPERSEDE", 1, "B")]
        assert build_supersede_map(events) == {"A": "C"}
        assert build_supersede_map(events[::-1]) == {"A": "C"}

    def test_first_of_equally_dated_events_wins(self) -> None:
        events = [event("A", "SUPERSEDE", 1, "B"), event("A", "DELETE", 1)]
        assert build_supersede_map(events) == {"A": "B"}

    def test_restore_removes_the_track(self) -> None:
        assert (
            build_supersede_map([event("A", "SUPERSEDE", 1, "B"), event("A", "RESTORE", 2)]) == {}
        )

    def test_restore_of_a_target_keeps_tracks_pointing_at_it(self) -> None:
        assert build_supersede_map([event("A", "RESTORE", 1)], {"X": "A"}) == {"X": "A"}

    def test_events_extend_an_initial_map(self) -> None:
        result = build_supersede_map([event("A", "SUPERSEDE", 1, "B")], {"X": "A"})
        assert result == {"X": "B", "A": "B"}

    @pytest.mark.parametrize("target", [None, "", float("nan")])
    def test_missing_target_counts_as_deletion(self, target: JSONValue) -> None:
        assert build_supersede_map([event("A", "SUPERSEDE", 1, target)]) == {"A": None}

    def test_cycles_resolve_to_where_they_close(self) -> None:
        result = build_supersede_map(
            [
                event("A", "SUPERSEDE", 1, "B"),
                event("B", "SUPERSEDE", 2, "C"),
                event("C", "SUPERSEDE", 3, "A"),
            ]
        )
        assert result == {"A": "A", "B": "B", "C": "C"}

    def test_unknown_actions_are_ignored(self) -> None:
        assert build_supersede_map([event("A", "MERGE", 1)], {"X": "Y"}) == {"X": "Y"}

    def test_events_without_track_ids_are_ignored(self) -> None:
        assert (
            build_supersede_map([{"action": "DELETE"}, {"trackId": None, "action": "DELETE"}]) == {}
        )

    def test_flattened_header_key_is_not_a_timestamp(self) -> None:
        events: list[JSONObject] = [
            {
                "trackId": "A",
                "action": "SUPERSEDE",
                "supersededBy": "B",
                "crucibleHeader.updatedDate": "2026-01-01T00:00:00Z",
            },
            {
                "trackId": "A",
                "action": "RESTORE",
                "crucibleHeader.updatedDate": "2026-01-02T00:00:00Z",
            },
        ]
        assert build_supersede_map(events) == {"A": "B"}


@pytest.mark.parametrize(
    ("track_id", "expected"),
    [("A", "C"), ("D", None), ("Z", "Z")],
)
def test_resolve_root(track_id: str, expected: str | None) -> None:
    assert resolve_root(track_id, {"A": "C", "B": "C", "D": None}) == expected


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        ({"A": "B", "B": "C"}, "C"),
        ({"A": "B", "B": None}, None),
        ({"A": "B", "B": "A"}, "A"),
        ({"A": "A"}, "A"),
    ],
)
def test_resolve_root_follows_unresolved_chains(
    links: dict[str, str | None], expected: str | None
) -> None:
    assert resolve_root("A", links) == expected


class TestFetchManagementEvents:
    async def test_pages_until_a_short_page(self) -> None:
        client = FakeSearch(
            [event("a", "SUPERSEDE", 1), event("b", "DELETE", 2)], [event("c", "RESTORE", 3)]
        )
        events = await fetch_management_events(client, "ManagementEvents", now=NOW, page_size=2)
        assert [record["trackId"] for record in events] == ["a", "b", "c"]
        (first, backtick_first), (second, _) = client.queries
        assert "OFFSET 0 ROWS FETCH NEXT 2 ROWS ONLY" in first
        assert "OFFSET 2 ROWS FETCH NEXT 2 ROWS ONLY" in second
        assert "TIMESTAMP_OFFSET(-30,'days')" in first
        assert backtick_first is False

    async def test_filters_actions_and_measures_since_in_seconds(self) -> None:
        client = FakeSearch([])
        await fetch_management_events(
            client,
            "ManagementEvents",
            actions=(Action.RESTORE,),
            since=NOW - timedelta(seconds=299.5),
            now=NOW,
        )
        sql = client.queries[0][0]
        assert "ManagementEvents.action = 'RESTORE'" in sql
        assert "SUPERSEDE" not in sql
        assert "TIMESTAMP_OFFSET(-300,'seconds')" in sql

    @pytest.mark.parametrize("body", ["Table 'crucibleHeader' not found", "anything at all"])
    async def test_a_400_on_the_first_page_means_an_empty_dataset(self, body: str) -> None:
        client = FakeSearch(RequestError("Status 400: Bad Request", status=400, body=body))
        assert await fetch_management_events(client, "ManagementEvents", now=NOW) == []

    async def test_other_statuses_on_the_first_page_propagate(self) -> None:
        client = FakeSearch(RequestError("Status 500", status=500, body="Error processing query"))
        with pytest.raises(RequestError):
            await fetch_management_events(client, "ManagementEvents", now=NOW)

    async def test_errors_after_the_first_page_propagate(self) -> None:
        error = RequestError("Status 400: Bad Request", status=400, body="Error processing query")
        client = FakeSearch([event("a", "DELETE", 1)], error)
        with pytest.raises(RequestError):
            await fetch_management_events(client, "ManagementEvents", now=NOW, page_size=1)

    async def test_other_failures_propagate(self) -> None:
        with pytest.raises(TransientError):
            await fetch_management_events(
                FakeSearch(TransientError("timeout")), "ManagementEvents", now=NOW
            )

    async def test_rejects_unsafe_dataset_names(self) -> None:
        with pytest.raises(ValueError, match="valid dataset name"):
            await fetch_management_events(FakeSearch([]), "Events; DROP", now=NOW)


async def test_load_supersede_map() -> None:
    client = FakeSearch([event("A", "SUPERSEDE", 1, "B")])
    assert await load_supersede_map(client, "ManagementEvents", now=NOW) == {"A": "B"}


class TestDuplicateProtectedIds:
    def client(self, restores: list[JSONObject] | Exception) -> FakeSearch:
        return FakeSearch([event("A", "SUPERSEDE", 1, "B"), event("C", "DELETE", 2)], restores)

    async def test_protects_both_sides_and_recent_restores(self) -> None:
        client = self.client([event("R", "RESTORE", 3)])
        protected = await duplicate_protected_ids(
            client, "ManagementEvents", now=NOW, restore_cooldown_seconds=300
        )
        assert protected.protected == {"A", "B", "C", "R"}
        assert (protected.managed, protected.survivors, protected.restored) == (
            {"A", "C"},
            {"B"},
            {"R"},
        )
        assert protected.may_survive("B")
        assert not protected.may_survive("A")
        assert not protected.may_survive("R")
        assert "action = 'RESTORE'" in client.queries[1][0]

    @pytest.mark.parametrize(
        "failure", [RequestError("boom", status=500, body=""), TransientError("timeout")]
    )
    async def test_restore_query_failure_only_drops_the_cooldown(self, failure: Exception) -> None:
        client = self.client(failure)
        protected = await duplicate_protected_ids(
            client, "ManagementEvents", now=NOW, restore_cooldown_seconds=300
        )
        assert protected.protected == {"A", "B", "C"}
        assert protected.restored == frozenset()

    async def test_no_cooldown_skips_the_restore_query(self) -> None:
        client = self.client([])
        await duplicate_protected_ids(
            client, "ManagementEvents", now=NOW, restore_cooldown_seconds=0
        )
        assert len(client.queries) == 1

    async def test_map_failure_propagates(self) -> None:
        with pytest.raises(TransientError):
            await duplicate_protected_ids(
                FakeSearch(TransientError("down")),
                "ManagementEvents",
                now=NOW,
                restore_cooldown_seconds=0,
            )
