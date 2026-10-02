import logging
import re
from collections.abc import Mapping

import pytest

from crucible_entity_manager import cli
from crucible_entity_manager.components.base import Component, Services
from crucible_entity_manager.config.runtime import DuplicateOptions, FuserOptions, RuntimeSettings
from crucible_entity_manager.core.partition import PartitionSpec
from crucible_entity_manager.crucible.writer import DrainBudget, WriteLedger
from crucible_entity_manager.runtime.process import Builder
from tests.unit.runtime.fakes import FakeComponent


class TestParseArgs:
    def test_defaults(self) -> None:
        assert cli.parse_args(["transformer", "Blue"]) == RuntimeSettings(
            component="transformer", perspective="Blue"
        )

    def test_every_option(self) -> None:
        settings = cli.parse_args(
            [
                "transformer",
                "Blue",
                "--partition",
                "2",
                "--partitions",
                "3",
                "--feed",
                "AIS",
                "--log-level",
                "debug",
                "--health-port",
                "8080",
                "--drain-seconds",
                "25",
                "--request-timeout",
                "10",
                "--tick-seconds",
                "2.5",
                "--stall-seconds",
                "120",
                "--management-lookback-days",
                "7",
                "--max-batch-records",
                "100",
            ]
        )
        assert settings == RuntimeSettings(
            component="transformer",
            perspective="Blue",
            partition=PartitionSpec(2, 3),
            feed="AIS",
            log_level="DEBUG",
            health_port=8080,
            drain_seconds=25.0,
            request_timeout_seconds=10.0,
            tick_seconds=2.5,
            stall_seconds=120.0,
            management_lookback_days=7,
            max_batch_records=100,
        )

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (["unknown", "Blue"], "invalid choice: 'unknown'"),
            (["transformer", "Blue", "--partition", "3", "--partitions", "3"], r"in \[0, 3\)"),
            (["transformer", "Blue", "--partition", "-1"], r"in \[0, 1\)"),
            (["transformer", "Blue", "--partitions", "0"], "must be at least 1"),
            (["transformer", "Blue", "--drain-seconds", "0"], "must be positive and finite"),
            (["transformer", "Blue", "--tick-seconds", "nan"], "must be positive and finite"),
            (["transformer", "Blue", "--stall-seconds", "inf"], "must be positive and finite"),
            (["transformer", "Blue", "--health-port", "70000"], "must be a TCP port"),
            (["transformer", "Blue", "--log-level", "verbose"], "invalid choice: 'VERBOSE'"),
        ],
    )
    def test_rejects_invalid_arguments(
        self, arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exited:
            cli.parse_args(arguments)
        assert exited.value.code == 2
        assert re.search(message, capsys.readouterr().err)

    def test_fuser_options(self) -> None:
        settings = cli.parse_args(
            ["fuser", "Blue", "--no-ci", "--ci-omega", "0.25", "--passthrough"]
        )
        assert settings.fuser == FuserOptions(
            covariance_intersection=False, ci_omega=0.25, passthrough=True
        )
        assert cli.parse_args(["fuser", "Blue"]).fuser == FuserOptions()
        assert cli.parse_args(["tracker", "Blue"]).fuser == FuserOptions()

    @pytest.mark.parametrize("omega", ["0", "1", "1.5", "nan"])
    def test_fuser_omega_is_in_the_open_unit_interval(
        self, omega: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args(["fuser", "Blue", "--ci-omega", omega])
        assert "must be between 0 and 1, exclusive" in capsys.readouterr().err

    def test_duplicate_options(self) -> None:
        settings = cli.parse_args(
            [
                "duplicates",
                "Blue",
                "--poll-interval",
                "60",
                "--distance-threshold",
                "800",
                "--mahalanobis-threshold",
                "2.5",
                "--velocity-threshold",
                "7",
                "--time-window-hours",
                "0.5",
                "--min-matching-points",
                "8",
                "--time-alignment-seconds",
                "10",
                "--min-confidence",
                "0.7",
            ]
        )
        assert settings.duplicates == DuplicateOptions(
            poll_interval_seconds=60.0,
            candidate_search_radius_m=800.0,
            mahalanobis_threshold_sigma=2.5,
            velocity_threshold_mps=7.0,
            time_window_hours=0.5,
            min_matching_points=8,
            time_alignment_seconds=10.0,
            min_confidence=0.7,
        )
        assert cli.parse_args(["duplicates", "Blue"]).duplicates == DuplicateOptions()
        assert cli.parse_args(["fuser", "Blue"]).duplicates == DuplicateOptions()

    def test_the_duplicate_identifier_runs_as_one_partition(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args(["duplicates", "Blue", "--partitions", "3"])
        assert "duplicates runs as a single partition" in capsys.readouterr().err

    def test_the_fuser_runs_as_one_partition(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args(["fuser", "Blue", "--partitions", "2"])
        assert "fuser runs as a single partition" in capsys.readouterr().err

    def test_single_partition_components_reject_more_partitions(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async def build(services: Services) -> Component:
            raise AssertionError(services)

        monkeypatch.setattr(
            cli, "COMPONENTS", {"solo": cli.ComponentSpec("Solo", build, single_partition=True)}
        )
        with pytest.raises(SystemExit):
            cli.parse_args(["solo", "Blue", "--partition", "1", "--partitions", "2"])
        assert "solo runs as a single partition" in capsys.readouterr().err


def test_logging_prefixes_component_and_partition(capsys: pytest.CaptureFixture[str]) -> None:
    settings = RuntimeSettings(
        component="transformer", perspective="Blue", partition=PartitionSpec(1, 4), log_level="INFO"
    )
    cli.configure_logging(settings)
    logging.getLogger("probe").info("hello %s", "world")
    assert "INFO transformer p1/4 probe: hello world" in capsys.readouterr().err


async def test_builder_bootstraps_then_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    component = FakeComponent([])
    sentinel = object()

    async def bootstrap(
        settings: RuntimeSettings,
        environ: Mapping[str, str],
        ledger: WriteLedger,
        budget: DrainBudget,
    ) -> object:
        calls.append(f"bootstrap {settings.perspective} {environ['HOST']}")
        assert isinstance(ledger, WriteLedger)
        assert isinstance(budget, DrainBudget)
        return sentinel

    async def build(services: Services) -> Component:
        assert services is sentinel
        calls.append("build")
        return component

    monkeypatch.setattr(cli, "bootstrap", bootstrap)
    monkeypatch.setattr(cli, "COMPONENTS", {"fake": cli.ComponentSpec("Fake", build)})
    settings = RuntimeSettings(component="fake", perspective="Blue")
    built = await cli.builder(settings, {"HOST": "crucible"})(
        WriteLedger(), DrainBudget(request_seconds=1.0)
    )
    assert built is component
    assert calls == ["bootstrap Blue crucible", "build"]


def test_main_runs_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[RuntimeSettings] = []

    def run_process(settings: RuntimeSettings, build: Builder) -> int:
        assert callable(build)
        seen.append(settings)
        return 3

    monkeypatch.setattr(cli, "run_process", run_process)
    assert cli.main(["transformer", "Blue", "--log-level", "WARNING"]) == 3
    assert seen == [
        RuntimeSettings(component="transformer", perspective="Blue", log_level="WARNING")
    ]
