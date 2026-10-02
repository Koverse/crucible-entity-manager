"""Command line: parse the Execution's ``args`` and run one component (DESIGN.md §7)."""

import argparse
import logging
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from crucible_entity_manager.components import duplicates, fuser, tracker, transformer
from crucible_entity_manager.components.base import Component, Services
from crucible_entity_manager.config.runtime import DuplicateOptions, FuserOptions, RuntimeSettings
from crucible_entity_manager.core.partition import PartitionSpec
from crucible_entity_manager.crucible.writer import DrainBudget, WriteLedger
from crucible_entity_manager.runtime.context import bootstrap
from crucible_entity_manager.runtime.process import Builder, run_process

LOG_LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _no_options(parser: argparse.ArgumentParser) -> None:
    del parser


def _fuser_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("fuser options")
    group.add_argument(
        "--no-ci",
        dest="covariance_intersection",
        action="store_false",
        help="fuse with a standard Kalman update instead of covariance intersection",
    )
    group.add_argument(
        "--ci-omega",
        type=_open_unit_interval,
        help="fixed covariance-intersection weight in (0, 1); searched when omitted",
    )
    group.add_argument(
        "--passthrough",
        action="store_true",
        help="copy component kinematics into principal tracks without filtering",
    )


def _duplicate_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("duplicate identifier options")
    defaults = DuplicateOptions()
    _seconds(group, "--poll-interval", defaults.poll_interval_seconds, "interval between searches")
    group.add_argument(
        "--distance-threshold",
        type=_positive_float,
        default=defaults.candidate_search_radius_m,
        help="candidate search radius in meters (default %(default)g)",
    )
    group.add_argument(
        "--mahalanobis-threshold",
        type=_positive_float,
        default=defaults.mahalanobis_threshold_sigma,
        help="largest mean Mahalanobis distance of a duplicate, in sigma (default %(default)g)",
    )
    group.add_argument(
        "--velocity-threshold",
        type=_positive_float,
        default=defaults.velocity_threshold_mps,
        help="largest mean velocity difference in m/s (default %(default)g)",
    )
    group.add_argument(
        "--time-window-hours",
        type=_positive_float,
        default=defaults.time_window_hours,
        help="how far back to read track events (default %(default)g)",
    )
    group.add_argument(
        "--min-matching-points",
        type=_positive_int,
        default=defaults.min_matching_points,
        help="fewest distinct measurement times per track (default %(default)d)",
    )
    _seconds(
        group,
        "--time-alignment-seconds",
        defaults.time_alignment_seconds,
        "how far a track is extrapolated beyond its ends",
    )
    group.add_argument(
        "--min-confidence",
        type=_open_unit_interval,
        default=defaults.min_confidence,
        help="lowest confidence to supersede (default %(default)g)",
    )


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A component the command line can run."""

    summary: str
    build: Callable[[Services], Awaitable[Component]]
    single_partition: bool = False
    add_options: Callable[[argparse.ArgumentParser], None] = _no_options


COMPONENTS: Final[Mapping[str, ComponentSpec]] = {
    "transformer": ComponentSpec("Origin records → report events", transformer.build),
    "tracker": ComponentSpec("Report events → component tracks", tracker.build),
    "fuser": ComponentSpec(
        "Component tracks → principal tracks",
        fuser.build,
        single_partition=True,
        add_options=_fuser_options,
    ),
    "duplicates": ComponentSpec(
        "Duplicate component tracks → SUPERSEDE events",
        duplicates.build,
        single_partition=True,
        add_options=_duplicate_options,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> RuntimeSettings:
    """Parse command-line arguments into settings, exiting with usage on error."""
    parser = _parser()
    args = parser.parse_args(argv)
    spec = COMPONENTS[args.component]
    if not 0 <= args.partition < args.partitions:
        parser.error(f"--partition must be in [0, {args.partitions}), got {args.partition}")
    if spec.single_partition and args.partitions != 1:
        parser.error(f"{args.component} runs as a single partition; --partitions must be 1")
    return RuntimeSettings(
        component=args.component,
        perspective=args.perspective,
        partition=PartitionSpec(args.partition, args.partitions),
        feed=args.feed,
        log_level=args.log_level,
        health_port=args.health_port,
        drain_seconds=args.drain_seconds,
        request_timeout_seconds=args.request_timeout,
        tick_seconds=args.tick_seconds,
        stall_seconds=args.stall_seconds,
        management_lookback_days=args.management_lookback_days,
        max_batch_records=args.max_batch_records,
        fuser=(
            FuserOptions(
                covariance_intersection=args.covariance_intersection,
                ci_omega=args.ci_omega,
                passthrough=args.passthrough,
            )
            if args.component == "fuser"
            else FuserOptions()
        ),
        duplicates=(
            DuplicateOptions(
                poll_interval_seconds=args.poll_interval,
                candidate_search_radius_m=args.distance_threshold,
                mahalanobis_threshold_sigma=args.mahalanobis_threshold,
                velocity_threshold_mps=args.velocity_threshold,
                time_window_hours=args.time_window_hours,
                min_matching_points=args.min_matching_points,
                time_alignment_seconds=args.time_alignment_seconds,
                min_confidence=args.min_confidence,
            )
            if args.component == "duplicates"
            else DuplicateOptions()
        ),
    )


def configure_logging(settings: RuntimeSettings) -> None:
    """Log to stderr, prefixing each line with the component and partition."""
    prefix = f"{settings.component} {settings.partition}".replace("%", "%%")
    logging.basicConfig(
        level=settings.log_level,
        format=f"%(asctime)s %(levelname)s {prefix} %(name)s: %(message)s",
        force=True,
    )


def builder(settings: RuntimeSettings, environ: Mapping[str, str]) -> Builder:
    """Return the builder `run_process` calls once signals are set up."""
    spec = COMPONENTS[settings.component]

    async def build(ledger: WriteLedger, budget: DrainBudget) -> Component:
        context = await bootstrap(settings, environ, ledger, budget)
        return await spec.build(context)

    return build


def main(argv: Sequence[str] | None = None) -> int:
    """Run the component the arguments name, and return the exit code."""
    settings = parse_args(argv)
    configure_logging(settings)
    return run_process(settings, builder(settings, os.environ))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_entity_manager.py",
        description="Run one entity-pipeline component as one partition.",
    )
    components = parser.add_subparsers(
        dest="component", required=True, metavar="component", help="the component to run"
    )
    common = _common_options()
    for name, spec in COMPONENTS.items():
        subparser = components.add_parser(
            name, parents=[common], help=spec.summary, description=spec.summary
        )
        spec.add_options(subparser)
    return parser


def _common_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("perspective", help="the perspective in the configuration dataset")
    parser.add_argument("--partition", type=int, default=0, help="this partition's index")
    parser.add_argument(
        "--partitions", type=_positive_int, default=1, help="partitions of this component"
    )
    parser.add_argument(
        "--feed", help="run only the feed with this origin dataset (transformer and tracker)"
    )
    parser.add_argument(
        "--log-level", type=str.upper, choices=LOG_LEVELS, default="INFO", help="logging level"
    )
    parser.add_argument("--health-port", type=_port, help="serve GET /healthz on this port")
    defaults = RuntimeSettings(component="", perspective="")
    _seconds(parser, "--drain-seconds", defaults.drain_seconds, "time allowed to drain on SIGTERM")
    _seconds(
        parser,
        "--request-timeout",
        defaults.request_timeout_seconds,
        "read timeout for each Crucible request",
    )
    _seconds(parser, "--tick-seconds", defaults.tick_seconds, "interval between timed work")
    _seconds(
        parser,
        "--stall-seconds",
        defaults.stall_seconds,
        "owner-loop stall that makes /healthz fail",
    )
    parser.add_argument(
        "--management-lookback-days",
        type=_positive_int,
        default=defaults.management_lookback_days,
        help="days of management events to load",
    )
    parser.add_argument(
        "--max-batch-records",
        type=_positive_int,
        default=defaults.max_batch_records,
        help="most records handled at once",
    )
    return parser


def _seconds(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    flag: str,
    default: float,
    help_text: str,
) -> None:
    parser.add_argument(
        flag, type=_positive_float, default=default, help=f"{help_text} (default {default:g})"
    )


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        msg = f"must be at least 1, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not (value > 0 and math.isfinite(value)):
        msg = f"must be positive and finite, got {text}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _open_unit_interval(text: str) -> float:
    value = float(text)
    if not 0 < value < 1:
        msg = f"must be between 0 and 1, exclusive, got {text}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _port(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 65535:  # noqa: PLR2004 - the TCP port range
        msg = f"must be a TCP port, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value
