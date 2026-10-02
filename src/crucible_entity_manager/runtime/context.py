"""What every component is built from, and how the process obtains it."""

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from crucible_entity_manager.config.perspective import (
    CONFIG_DATASET,
    PerspectiveConfig,
    WriteSettings,
    parse_perspective,
)
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.core.timeutil import utc_now
from crucible_entity_manager.crucible.client import ClientSettings, CrucibleClient
from crucible_entity_manager.crucible.protocols import CrucibleService
from crucible_entity_manager.crucible.sql import identifier, string_literal
from crucible_entity_manager.crucible.sse import SseSettings, SseSource
from crucible_entity_manager.crucible.writer import BatchWriter, DrainBudget, WriteLedger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Context:
    """Settings, configuration and shared I/O for building a component."""

    settings: RuntimeSettings
    perspective: PerspectiveConfig
    crucible: CrucibleService
    sse: SseSettings
    ledger: WriteLedger
    budget: DrainBudget
    clock: Callable[[], datetime] = utc_now

    def writer(self, writes: WriteSettings) -> BatchWriter:
        """A writer with `writes` limits that shares this process's ledger and budget."""
        return BatchWriter(
            self.crucible, self.crucible, writes, ledger=self.ledger, budget=self.budget
        )

    def source(self, name: str, query: str) -> SseSource:
        """An SSE subscription to `query`, authenticated with this process's token."""
        return SseSource(name, query, self.sse, self.crucible.access_token, clock=self.clock)


async def bootstrap(
    settings: RuntimeSettings,
    environ: Mapping[str, str],
    ledger: WriteLedger,
    budget: DrainBudget,
) -> Context:
    """Authenticate, then load and validate the perspective's configuration.

    Raises:
        KeyError: If the environment doesn't say where Crucible is.
        CrucibleError: If Crucible can't be reached or rejects the credentials.
        ConfigError: If the configuration is invalid.
    """
    sse = SseSettings.from_environ(environ)
    crucible = await asyncio.to_thread(
        CrucibleClient, ClientSettings(request_timeout_seconds=settings.request_timeout_seconds)
    )
    table = identifier(CONFIG_DATASET)
    rows = await crucible.search(
        f"SELECT * FROM {table} WHERE {table}.perspective = {string_literal(settings.perspective)}"  # noqa: S608
    )
    perspective = parse_perspective(settings.perspective, rows, environ)
    for warning in perspective.warnings:
        logger.warning("Configuration: %s", warning)
    if perspective.disabled_feeds:
        logger.info("Disabled feeds: %s", ", ".join(perspective.disabled_feeds))
    return Context(
        settings=settings,
        perspective=perspective,
        crucible=crucible,
        sse=sse,
        ledger=ledger,
        budget=budget,
    )
