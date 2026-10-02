import asyncio
import dataclasses
import logging

import pytest

from crucible_entity_manager.components import fuser as fuser_module
from crucible_entity_manager.components.fuser import (
    COMPONENTS,
    MANAGEMENT,
    Associations,
    AssociationStamps,
    Fuser,
    FuserDatasets,
    build,
    management_query,
)
from crucible_entity_manager.config.perspective import (
    ConfigError,
    Datasets,
    WriteSettings,
    parse_perspective,
)
from crucible_entity_manager.config.runtime import FuserOptions, RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.identity import principal_track_id
from crucible_entity_manager.core.records import get_path
from crucible_entity_manager.crucible.protocols import TransientError
from crucible_entity_manager.crucible.writer import BatchWriter, DrainBudget, WriteLedger
from tests.unit.components.fakes import FakeCrucible, FakeServices
from tests.unit.config.test_perspective import feed_row, perspective_row

C1, C2, C3 = (f"{index:032x}" for index in range(1, 4))
P1, P2, P3 = (principal_track_id(component) for component in (C1, C2, C3))


def component(track: str, seconds: int, serial: int, **extra: JSONValue) -> JSONObject:
    record: JSONObject = {
        "trackId": track,
        "interceptTimestamp": f"2026-09-30T00:00:{seconds:02d}.000Z",
        "environment": "AIR",
        "reportIds": [f"r{serial}"],
        "identity": {"callsign": f"CS-{track[-1]}"},
        "ecefPosition": {"x": 1_000_000.0 + seconds, "y": 2_000_000.0, "z": 3_000_000.0},
    }
    record.update(extra)
    return record


def management(track: str, action: str, target: str | None = None, minute: int = 0) -> JSONObject:
    event: JSONObject = {
        "trackId": track,
        "action": action,
        "crucibleHeader": {"updatedDate": f"2026-09-30T00:{minute:02d}:00.000Z"},
    }
    if target is not None:
        event["supersededBy"] = target
    return event


def services(
    *,
    options: FuserOptions | None = None,
    responder: dict[str, list[JSONObject]] | None = None,
    **perspective_extra: JSONValue,
) -> FakeServices:
    row = perspective_row(
        component_track_event_dataset="ComponentTrackEvents",
        component_track_head_dataset="ComponentTrackHeads",
        **perspective_extra,
    )
    settings = RuntimeSettings(
        component="fuser", perspective="Blue", fuser=options or FuserOptions(passthrough=True)
    )
    service = FakeServices(settings, parse_perspective("Blue", [row, feed_row("Radar")], {}))
    answers = responder or {}

    def respond(sql: str) -> list[JSONObject]:
        for marker, rows in answers.items():
            if marker in sql:
                return rows
        return []

    service.crucible.responder = respond
    return service


async def started(service: FakeServices) -> Fuser:
    component_ = await build(service)
    assert isinstance(component_, Fuser)
    await component_.prepare()
    return component_


def principal_ids(crucible: FakeCrucible) -> list[str]:
    return [str(event["trackId"]) for event in crucible.posted.get("PrincipalTrackEvents", [])]


class TestConfiguration:
    def test_datasets_need_component_events(self) -> None:
        datasets = Datasets("M", "R", "PE", "PH", None, None)
        with pytest.raises(ConfigError, match="component_track_event_dataset"):
            FuserDatasets.of(datasets)

    def test_management_query(self) -> None:
        assert management_query("Mgmt") == (
            "SELECT * FROM Mgmt WHERE action IN ('SUPERSEDE', 'DELETE', 'RESTORE')"
        )

    async def test_build_needs_a_feed(self) -> None:
        service = services()
        service.perspective = dataclasses.replace(service.perspective, feeds=())
        with pytest.raises(ConfigError, match="needs an enabled feed"):
            await build(service)

    async def test_build_wires_sources_and_options(self) -> None:
        service = services(options=FuserOptions(covariance_intersection=False, ci_omega=0.4))
        fuser = await build(service)
        assert isinstance(fuser, Fuser)
        assert service.sources[COMPONENTS][0] == "SELECT * FROM ComponentTrackEvents"
        assert service.sources[MANAGEMENT][0] == management_query("EntityManagementEvents")
        assert fuser.queue_max_records == 50_000
        assert [subscription.name for subscription in fuser.subscriptions()] == [
            COMPONENTS,
            MANAGEMENT,
        ]

    async def test_build_without_component_heads_has_no_stamps(self) -> None:
        row = perspective_row(component_track_event_dataset="ComponentTrackEvents")
        del row["component_track_head_dataset"]
        settings = RuntimeSettings(component="fuser", perspective="Blue")
        service = FakeServices(settings, parse_perspective("Blue", [row, feed_row("Radar")], {}))
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C1, 1, 1)])
        await fuser.tick()
        await fuser.close()
        assert "ComponentTrackHeads" not in service.crucible.updated


class TestAssociations:
    def test_associate_reuses_and_moves(self) -> None:
        associations = Associations()
        assert associations.associate(C1, C1) == (P1, True)
        assert associations.associate(C1, C1) == (P1, False)
        assert associations.associate(C2, C1) == (P1, True)
        assert associations.members(P1) == {C1, C2}
        associations.link(C2, P2)
        assert associations.members(P1) == {C1}
        assert associations.principal_of(C2) == P2

    def test_a_root_keeps_an_existing_principal(self) -> None:
        associations = Associations()
        associations.link(C1, "e" * 32)
        assert associations.associate(C1, C1) == ("e" * 32, False)
        associations.associate(C2, C2)
        assert associations.principal_for_root(C2) == P2

    def test_discard_principal(self) -> None:
        associations = Associations()
        associations.associate(C1, C1)
        associations.associate(C2, C1)
        associations.discard_principal(P1)
        assert associations.principal_of(C1) is None
        assert associations.principal_for_root(C1) is None
        assert associations.components() == []

    def test_eviction_keeps_members_consistent(self) -> None:
        associations = Associations(max_keys=1)
        associations.associate(C1, C1)
        associations.associate(C2, C2)
        assert associations.principal_of(C1) is None
        assert associations.members(P1) == set()


class TestAssociationStamps:
    def stamps(self, crucible: FakeCrucible, clock: list[float] | None = None) -> AssociationStamps:
        writer = BatchWriter(
            crucible,
            crucible,
            WriteSettings(),
            ledger=WriteLedger(),
            budget=DrainBudget(request_seconds=1.0),
        )
        now = clock if clock is not None else [0.0]
        return AssociationStamps(writer, "Heads", label="", clock=lambda: now[0])

    async def test_writes_partial_updates_and_retries_failures(self) -> None:
        crucible = FakeCrucible()
        crucible.put_fails = {C1}
        stamps = self.stamps(crucible)
        stamps.stamp(C1, P1)
        stamps.stamp(C2, P1)
        stamps.flush()
        await stamps.close()
        assert crucible.updated["Heads"] == [{"trackId": C2, "associatedPrincipalTrack": P1}]
        assert stamps.pending == 1
        crucible.put_fails = set()
        await stamps.close()
        assert {"trackId": C1, "associatedPrincipalTrack": P1} in crucible.updated["Heads"]
        assert stamps.pending == 0

    async def test_restamping_before_a_write_keeps_one_stamp(self) -> None:
        crucible = FakeCrucible()
        stamps = self.stamps(crucible)
        stamps.stamp(C1, P1)
        stamps.stamp(C1, P2)
        assert stamps.pending == 1
        await stamps.close()
        assert crucible.updated["Heads"] == [{"trackId": C1, "associatedPrincipalTrack": P2}]

    async def test_a_newer_stamp_replaces_a_retry(self) -> None:
        crucible = FakeCrucible()
        crucible.put_fails = {C1}
        stamps = self.stamps(crucible)
        stamps.stamp(C1, P1)
        stamps.flush()
        stamps.stamp(C1, P2)
        await stamps.close()
        assert stamps.pending == 1
        crucible.put_fails = set()
        await stamps.close()
        assert crucible.updated["Heads"][-1] == {"trackId": C1, "associatedPrincipalTrack": P2}

    async def test_in_flight_stamps_count_toward_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fuser_module, "PENDING_STAMPS", 1)
        stamps = self.stamps(FakeCrucible())
        stamps.stamp(C1, P1)
        stamps.flush()
        stamps.stamp(C2, P2)
        assert (stamps.pending, stamps.dropped) == (0, 1)
        assert stamps.lost(C2)
        await stamps.close()
        stamps.stamp(C2, P2)
        assert not stamps.lost(C2)
        await stamps.close()

    async def test_pending_stamps_are_capped_and_expire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fuser_module, "PENDING_STAMPS", 1)
        clock = [0.0]
        stamps = self.stamps(FakeCrucible(), clock)
        stamps.stamp(C1, P1)
        stamps.stamp(C2, P2)
        assert (stamps.pending, stamps.dropped) == (1, 1)
        assert stamps.lost(C1)
        clock[0] = 3_601.0
        stamps.flush()
        assert (stamps.pending, stamps.dropped) == (0, 2)
        assert stamps.lost(C2)
        await stamps.close()


class TestLostStamps:
    async def test_a_dropped_stamp_is_restamped_by_the_next_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fuser_module, "PENDING_STAMPS", 1)
        service = services()
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C1, 1, 1), component(C2, 1, 2)])
        await fuser.close()
        written = {
            str(stamp["trackId"]) for stamp in service.crucible.updated["ComponentTrackHeads"]
        }
        assert written == {C2}
        await fuser.handle(COMPONENTS, [component(C1, 2, 3)])
        await fuser.close()
        written = {
            str(stamp["trackId"]) for stamp in service.crucible.updated["ComponentTrackHeads"]
        }
        assert written == {C1, C2}


class TestFuser:
    async def test_fuses_components_into_principals_and_stamps_associations(self) -> None:
        service = services()
        fuser = await started(service)
        await fuser.handle(
            COMPONENTS, [component(C1, 2, 1), component(C2, 1, 2), component(C1, 3, 3)]
        )
        await fuser.close()
        assert principal_ids(service.crucible) == [P2, P1, P1]
        stamps = service.crucible.updated["ComponentTrackHeads"]
        assert sorted(
            (str(stamp["trackId"]), str(stamp["associatedPrincipalTrack"])) for stamp in stamps
        ) == [
            (C1, P1),
            (C2, P2),
        ]
        heads = service.crucible.updated["PrincipalTrackHeads"]
        assert {str(head["trackId"]) for head in heads} == {P1, P2}
        assert all("geodetic" in head and "stale" in head for head in heads)
        assert service.ledger.snapshot.clean

    async def test_superseded_components_fuse_into_the_survivor_and_restore_resets(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        service = services(options=FuserOptions())
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C1, 1, 1), component(C2, 2, 2)])
        await fuser.handle(MANAGEMENT, [management(C2, "SUPERSEDE", C1)])
        assert "moved 1 association(s)" in caplog.text
        await fuser.handle(COMPONENTS, [component(C2, 3, 3, identity={"callsign": "DUP"})])
        await fuser.handle(COMPONENTS, [component(C1, 4, 4, identity={"callsign": "ROOT"})])
        events = service.crucible.posted["PrincipalTrackEvents"]
        assert [str(event["trackId"]) for event in events[-2:]] == [P1, P1]
        assert get_path(events[-1], "identity.callsign") == "DUP"
        await fuser.handle(MANAGEMENT, [management(C2, "RESTORE", minute=1)])
        assert "reset 1 restored principal(s)" in caplog.text
        await fuser.handle(COMPONENTS, [component(C2, 5, 5)])
        await fuser.close()
        assert str(service.crucible.posted["PrincipalTrackEvents"][-1]["trackId"]) == P2

    async def test_deleted_and_unkeyed_components_are_not_fused(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        service = services()
        fuser = await started(service)
        await fuser.handle(MANAGEMENT, [management(C3, "DELETE")])
        unkeyed = component(C1, 1, 1)
        del unkeyed["trackId"]
        await fuser.handle(COMPONENTS, [component(C3, 1, 1), unkeyed, component(C1, 2, 2)])
        await fuser.close()
        assert principal_ids(service.crucible) == [P1]
        assert "Skipped 2 component track event(s)" in caplog.text

    async def test_repeated_and_timeless_components_are_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        service = services()
        fuser = await started(service)
        event = component(C1, 1, 1)
        timeless = component(C1, 2, 2, interceptTimestamp="later")
        untraced = component(C1, 3, 3)
        del untraced["reportIds"]
        await fuser.handle(COMPONENTS, [event, {**event}, timeless, untraced, {**untraced}])
        await fuser.tick()
        await fuser.close()
        assert len(principal_ids(service.crucible)) == 3
        assert "Dropped 1 component track event(s) already fused" in caplog.text
        assert "without a parseable interceptTimestamp" in caplog.text

    async def test_preload_restores_state_associations_and_identity(self) -> None:
        old = "e" * 32
        responder: dict[str, list[JSONObject]] = {
            "FROM PrincipalTrackHeads ORDER": [
                {
                    "trackId": P1,
                    "trackUpdatedTimestamp": "2026-09-30T00:00:00.000Z",
                    "identity": {"registry": "R-1"},
                    "ecefPosition": {"x": 1_000_000.0, "y": 2_000_000.0, "z": 3_000_000.0},
                },
                {"identity": {"orphan": True}},
            ],
            "FROM ComponentTrackHeads ORDER": [
                {"trackId": C2, "associatedPrincipalTrack": old},
                {"trackId": C3},
            ],
        }
        service = services(options=FuserOptions(), responder=responder)
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C1, 1, 1), component(C2, 1, 2)])
        await fuser.close()
        events = service.crucible.posted["PrincipalTrackEvents"]
        assert [str(event["trackId"]) for event in events] == [P1, old]
        assert get_path(events[0], "identity.registry") == "R-1"
        assert ("post", "PrincipalTrackHeads", [P1]) not in service.crucible.log

    async def test_preload_failure_and_limit_are_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        failing = services()

        def unavailable(sql: str) -> list[JSONObject]:
            if "TrackHeads ORDER BY" in sql:
                raise TransientError("down")
            return []

        failing.crucible.responder = unavailable
        await started(failing)
        assert "Head preload from PrincipalTrackHeads failed (down)" in caplog.text
        limited = services(
            fusion_head_preload_limit=1,
            responder={"FROM PrincipalTrackHeads ORDER": [{"trackId": P1}]},
        )
        await started(limited)
        assert "hit its 1-row limit" in caplog.text
        assert "Raise fusion_head_preload_limit" in caplog.text

    async def test_a_principal_head_without_a_time_is_restored_at_now(self) -> None:
        responder: dict[str, list[JSONObject]] = {
            "FROM PrincipalTrackHeads ORDER": [
                {"trackId": P1, "ecefPosition": {"x": 1.0, "y": 2.0, "z": 3.0}}
            ]
        }
        service = services(options=FuserOptions(), responder=responder)
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C1, 1, 1)])
        await fuser.close()
        assert principal_ids(service.crucible) == [P1]

    async def test_a_batch_with_nothing_to_fuse_writes_nothing(self) -> None:
        service = services()
        fuser = await started(service)
        await fuser.handle(MANAGEMENT, [management(C1, "DELETE")])
        await fuser.handle(COMPONENTS, [component(C1, 1, 1)])
        await fuser.close()
        assert principal_ids(service.crucible) == []

    async def test_identity_overlay_without_a_component_identity(self) -> None:
        service = services()
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C1, 1, 1)])
        bare = component(C1, 2, 2)
        del bare["identity"]
        await fuser.handle(COMPONENTS, [bare])
        lone = component(C2, 3, 3)
        del lone["identity"]
        await fuser.handle(COMPONENTS, [lone])
        await fuser.close()
        events = service.crucible.posted["PrincipalTrackEvents"]
        assert get_path(events[1], "identity.callsign") == "CS-1"
        assert "identity" not in events[2]

    async def test_preload_can_be_skipped(self) -> None:
        service = services(skip_head_preload=True)
        await started(service)
        assert all("TrackHeads ORDER BY" not in query for query in service.crucible.queries)

    async def test_identity_is_rehydrated_from_stored_heads(self) -> None:
        responder: dict[str, list[JSONObject]] = {
            "SELECT trackId, identity FROM PrincipalTrackHeads": [
                {"trackId": P1, "identity": {"registry": "R-9"}},
                {"trackId": P3, "identity": {"registry": "unrequested"}},
                {"identity": {"registry": "no id"}},
            ]
        }
        service = services(responder=responder)
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C1, 1, 1), component(C1, 2, 2)])
        await fuser.close()
        queries = [
            query for query in service.crucible.queries if "SELECT trackId, identity" in query
        ]
        assert queries == [
            f"SELECT trackId, identity FROM PrincipalTrackHeads WHERE trackId IN ('{P1}')"
        ]
        assert (
            get_path(service.crucible.posted["PrincipalTrackEvents"][0], "identity.registry")
            == "R-9"
        )

    async def test_rehydration_failures_start_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        service = services()
        fuser = await started(service)

        def unavailable(sql: str) -> list[JSONObject]:
            raise TransientError("down")

        service.crucible.responder = unavailable
        await fuser.handle(COMPONENTS, [component(C1, 1, 1)])
        await fuser.close()
        assert "Identity rehydration failed (down)" in caplog.text
        assert principal_ids(service.crucible) == [P1]

    async def test_the_supersede_map_reloads_after_a_reconnect(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        service = services(responder={"FROM EntityManagementEvents": []})
        fuser = await started(service)
        source = service.sources[MANAGEMENT][1]
        source.connected_since_monotonic = 1.0
        await fuser.tick()
        service.crucible.responder = lambda sql: (
            [management(C2, "SUPERSEDE", C1)] if "EntityManagementEvents" in sql else []
        )
        source.connected_since_monotonic = 2.0
        await fuser.tick()
        assert "Reloaded the supersede map after a reconnect" in caplog.text
        await fuser.handle(COMPONENTS, [component(C1, 1, 1), component(C2, 2, 2)])
        await fuser.close()
        assert principal_ids(service.crucible) == [P1, P1]

    async def test_restore_does_not_rehydrate_the_restored_identity(self) -> None:
        responder: dict[str, list[JSONObject]] = {
            "SELECT trackId, identity FROM PrincipalTrackHeads": [
                {"trackId": P1, "identity": {"callsign": "FROM-DUPLICATE"}}
            ]
        }
        service = services(responder=responder)
        fuser = await started(service)
        await fuser.handle(MANAGEMENT, [management(C2, "SUPERSEDE", C1)])
        await fuser.handle(COMPONENTS, [component(C1, 1, 1), component(C2, 1, 2)])
        await fuser.handle(MANAGEMENT, [management(C2, "RESTORE", minute=1)])
        survivor = component(C1, 2, 2)
        del survivor["identity"]
        await fuser.handle(COMPONENTS, [survivor])
        await fuser.close()
        last = service.crucible.posted["PrincipalTrackEvents"][-1]
        assert str(last["trackId"]) == P1
        assert "identity" not in last

    async def test_a_supersede_into_an_unseen_survivor_keeps_the_group_together(self) -> None:
        service = services()
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C2, 1, 1)])
        await fuser.handle(MANAGEMENT, [management(C2, "SUPERSEDE", C1)])
        await fuser.handle(COMPONENTS, [component(C2, 2, 2), component(C1, 3, 3)])
        await fuser.close()
        assert principal_ids(service.crucible) == [P2, P1, P1]
        stamps = service.crucible.updated["ComponentTrackHeads"]
        assert {"trackId": C2, "associatedPrincipalTrack": P1} in stamps

    async def test_a_preloaded_association_anchors_its_group(self) -> None:
        old = "e" * 32
        responder: dict[str, list[JSONObject]] = {
            "FROM ComponentTrackHeads ORDER": [{"trackId": C1, "associatedPrincipalTrack": old}]
        }
        service = services(responder=responder)
        fuser = await started(service)
        await fuser.handle(MANAGEMENT, [management(C2, "SUPERSEDE", C1)])
        await fuser.handle(COMPONENTS, [component(C2, 1, 1), component(C1, 2, 2)])
        await fuser.close()
        assert principal_ids(service.crucible) == [old, old]

    async def test_startup_moves_preloaded_associations_to_their_root(self) -> None:
        old = "e" * 32
        responder: dict[str, list[JSONObject]] = {
            "FROM EntityManagementEvents": [management(C2, "SUPERSEDE", C1)],
            "FROM ComponentTrackHeads ORDER": [
                {"trackId": C1, "associatedPrincipalTrack": old},
                {"trackId": C2, "associatedPrincipalTrack": P2},
            ],
        }
        service = services(responder=responder)
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C2, 1, 1)])
        await fuser.close()
        assert principal_ids(service.crucible) == [old]
        stamps = service.crucible.updated["ComponentTrackHeads"]
        assert {"trackId": C2, "associatedPrincipalTrack": old} in stamps

    async def test_rehydration_reads_the_head_of_a_preloaded_root_principal(self) -> None:
        old = "e" * 32
        responder: dict[str, list[JSONObject]] = {
            "FROM EntityManagementEvents": [management(C2, "SUPERSEDE", C1)],
            "FROM ComponentTrackHeads ORDER": [{"trackId": C1, "associatedPrincipalTrack": old}],
            "SELECT trackId, identity FROM PrincipalTrackHeads": [
                {"trackId": old, "identity": {"registry": "R-OLD"}}
            ],
        }
        service = services(responder=responder, skip_head_preload=False)
        fuser = await started(service)
        await fuser.handle(COMPONENTS, [component(C2, 1, 1)])
        await fuser.close()
        (event,) = service.crucible.posted["PrincipalTrackEvents"]
        assert str(event["trackId"]) == old
        assert get_path(event, "identity.registry") == "R-OLD"

    async def test_a_failed_reload_is_retried_on_the_same_connection(self) -> None:
        service = services()
        fuser = await started(service)
        source = service.sources[MANAGEMENT][1]
        source.connected_since_monotonic = 1.0
        await fuser.tick()

        def unavailable(sql: str) -> list[JSONObject]:
            raise TransientError("down")

        service.crucible.responder = unavailable
        source.connected_since_monotonic = 2.0
        await fuser.tick()
        service.crucible.responder = lambda sql: (
            [management(C2, "SUPERSEDE", C1)] if "EntityManagementEvents" in sql else []
        )
        await fuser.tick()
        await fuser.handle(COMPONENTS, [component(C1, 1, 1), component(C2, 2, 2)])
        await fuser.close()
        assert principal_ids(service.crucible) == [P1, P1]

    async def test_a_failed_reload_keeps_the_current_map(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = services()
        fuser = await started(service)
        source = service.sources[MANAGEMENT][1]
        source.connected_since_monotonic = 1.0
        await fuser.tick()

        def unavailable(sql: str) -> list[JSONObject]:
            raise TransientError("down")

        service.crucible.responder = unavailable
        source.connected_since_monotonic = 2.0
        await fuser.handle(MANAGEMENT, [])
        assert "Supersede map reload failed (down)" in caplog.text
        await asyncio.sleep(0)


async def test_the_kalman_fuser_carries_fused_state() -> None:
    service = services(options=FuserOptions(covariance_intersection=False))
    fuser = await started(service)
    await fuser.handle(
        COMPONENTS,
        [component(C1, 1, 1, positionCovariance={"xx": 100.0, "yy": 100.0, "zz": 100.0})],
    )
    await fuser.close()
    (event,) = service.crucible.posted["PrincipalTrackEvents"]
    assert get_path(event, "positionCovariance.xx") == pytest.approx(50.0)
