import logging
from typing import ClassVar

import pytest

from crucible_entity_manager.config.perspective import WriteSettings
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.core.aliases import JSONObject
from crucible_entity_manager.crucible.client import ClientSettings
from crucible_entity_manager.crucible.sse import SSE_PATH
from crucible_entity_manager.crucible.writer import DrainBudget, WriteLedger
from crucible_entity_manager.runtime import context
from tests.unit.config.test_perspective import feed_row, perspective_row

ENVIRON = {"CRUCIBLE_SERVICES_HOST": "crucible.example"}


class FakeCrucible:
    instances: ClassVar[list["FakeCrucible"]] = []
    rows: ClassVar[list[JSONObject]] = []

    def __init__(self, settings: ClientSettings) -> None:
        self.settings = settings
        self.queries: list[str] = []
        FakeCrucible.instances.append(self)

    async def search(self, sql: str, *, auto_backtick: bool = True) -> list[JSONObject]:
        del auto_backtick
        self.queries.append(sql)
        return FakeCrucible.rows

    async def access_token(self) -> str:
        return "token"

    def write_batch(self, dataset: str, records: list[JSONObject]) -> None:
        del dataset, records

    def update_batch(self, dataset: str, records: list[JSONObject]) -> list[JSONObject]:
        del dataset, records
        return []


@pytest.fixture
def crucible(monkeypatch: pytest.MonkeyPatch) -> type[FakeCrucible]:
    FakeCrucible.instances = []
    FakeCrucible.rows = [
        perspective_row(),
        feed_row("AIS", disable_all_other_datasets=True),
        feed_row("ADSB"),
    ]
    monkeypatch.setattr(context, "CrucibleClient", FakeCrucible)
    return FakeCrucible


async def test_bootstrap_loads_the_perspective(
    crucible: type[FakeCrucible], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    settings = RuntimeSettings(component="tracker", perspective="Blue", request_timeout_seconds=9.0)
    ledger, budget = WriteLedger(), DrainBudget(request_seconds=9.0)
    built = await context.bootstrap(settings, ENVIRON, ledger, budget)
    (client,) = crucible.instances
    assert client.settings.request_timeout_seconds == 9.0
    assert client.queries == [
        (
            "SELECT * FROM Entity_Stream_Manager_Configurations "
            "WHERE Entity_Stream_Manager_Configurations.perspective = 'Blue'"
        )
    ]
    assert [feed.origin_dataset for feed in built.perspective.feeds] == ["AIS"]
    assert built.sse.url == f"https://crucible.example{SSE_PATH}"
    assert (built.ledger, built.budget) == (ledger, budget)
    assert "Disabled feeds: ADSB" in caplog.text


async def test_bootstrap_logs_configuration_warnings(
    crucible: type[FakeCrucible], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    crucible.rows = [perspective_row(num_tracker_workers=4), feed_row("AIS")]
    await context.bootstrap(
        RuntimeSettings(component="tracker", perspective="Blue"),
        ENVIRON,
        WriteLedger(),
        DrainBudget(request_seconds=1.0),
    )
    assert "Configuration: perspective_config_Blue: num_tracker_workers is ignored" in caplog.text
    assert "Disabled feeds" not in caplog.text


async def test_bootstrap_needs_the_crucible_host(crucible: type[FakeCrucible]) -> None:
    with pytest.raises(KeyError, match="CRUCIBLE_SERVICES_HOST"):
        await context.bootstrap(
            RuntimeSettings(component="tracker", perspective="Blue"),
            {},
            WriteLedger(),
            DrainBudget(request_seconds=1.0),
        )
    assert crucible.instances == []


async def test_writers_and_sources_share_the_process_state(crucible: type[FakeCrucible]) -> None:
    built = await context.bootstrap(
        RuntimeSettings(component="tracker", perspective="Blue"),
        ENVIRON,
        WriteLedger(),
        DrainBudget(request_seconds=1.0),
    )
    writer = built.writer(WriteSettings(chunk_size=10, update_chunk_size=5, max_concurrent=1))
    assert writer._ledger is built.ledger
    assert writer._budget is built.budget
    source = built.source("AIS", "SELECT * FROM AIS")
    assert source.name == "AIS"
    assert source.watchdog_seconds == built.sse.read_timeout_seconds
