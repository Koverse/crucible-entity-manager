"""The process around a component: signals, the event loop and the exit code."""

import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from typing import NoReturn

from crucible_entity_manager.components.base import Component
from crucible_entity_manager.config.runtime import RuntimeSettings
from crucible_entity_manager.crucible.writer import DrainBudget, WriteLedger
from crucible_entity_manager.runtime.service import STOPPED, Service, unless_stopped
from crucible_entity_manager.runtime.shutdown import (
    EXIT_FAILURE,
    ShutdownWatchdog,
    block_shutdown_signals,
    exit_code,
)

logger = logging.getLogger(__name__)

type Builder = Callable[[WriteLedger, DrainBudget], Awaitable[Component]]
"""Builds the component, writing through the given ledger and budget."""


def run_process(settings: RuntimeSettings, build: Builder) -> int:
    """Run the component `build` makes until shutdown, and return the exit code.

    Call from the main thread before any other thread exists, then pass the
    result to `exit_process`.
    """
    block_shutdown_signals()
    ledger = WriteLedger()
    budget = DrainBudget(request_seconds=settings.request_timeout_seconds)
    watchdog = ShutdownWatchdog(ledger, settings.drain_seconds)
    watchdog.start()
    logger.info(
        "Starting %s for perspective %s, partition %s",
        settings.component,
        settings.perspective,
        settings.partition,
    )
    # The loop is deliberately never closed: closing it joins worker threads,
    # and one blocked in a write may never return. `exit_process` ends the
    # process without waiting for them.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run(settings, build, ledger, budget, watchdog))
    except Exception:
        logger.exception("%s failed", settings.component)
        return EXIT_FAILURE


def exit_process(code: int) -> NoReturn:
    """Flush logs and end the process now, without joining threads."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


async def _run(
    settings: RuntimeSettings,
    build: Builder,
    ledger: WriteLedger,
    budget: DrainBudget,
    watchdog: ShutdownWatchdog,
) -> int:
    stopping = asyncio.Event()

    def begin_shutdown(deadline: float) -> None:
        budget.start(deadline)
        stopping.set()
        logger.info("Shutdown requested; draining for up to %.0fs", settings.drain_seconds)

    watchdog.attach(asyncio.get_running_loop(), begin_shutdown)
    component = await unless_stopped(stopping, build(ledger, budget))
    if component is STOPPED:
        logger.warning("Shutdown began during startup")
        return exit_code(ledger.snapshot)
    return await Service(settings, ledger, stopping).run(component)
