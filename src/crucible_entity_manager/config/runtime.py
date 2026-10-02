"""Settings that come from the command line (the Execution's ``args``)."""

from dataclasses import dataclass, field

from crucible_entity_manager.core.partition import PartitionSpec


@dataclass(frozen=True, slots=True)
class FuserOptions:
    """The fuser's command-line options, with the defaults of ``1b534df``."""

    covariance_intersection: bool = True
    ci_omega: float | None = None
    """A fixed covariance-intersection weight in (0, 1); ``None`` searches for one."""
    passthrough: bool = False


@dataclass(frozen=True, slots=True)
class DuplicateOptions:
    """The duplicate identifier's command-line options, with the defaults of ``1b534df``.

    The thresholds are the fallback for environments without their own
    detection parameters.
    """

    poll_interval_seconds: float = 30.0
    candidate_search_radius_m: float = 500.0
    mahalanobis_threshold_sigma: float = 3.0
    velocity_threshold_mps: float = 10.0
    time_window_hours: float = 1.0
    min_matching_points: int = 5
    time_alignment_seconds: float = 5.0
    min_confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """How this process runs. Validated by the command-line parser."""

    component: str
    perspective: str
    partition: PartitionSpec = field(default_factory=PartitionSpec.single)
    feed: str | None = None
    """Run only this origin dataset's feed (transformer and tracker)."""

    log_level: str = "INFO"
    health_port: int | None = None
    drain_seconds: float = 20.0
    request_timeout_seconds: float = 15.0
    tick_seconds: float = 5.0
    stall_seconds: float = 300.0
    management_lookback_days: int = 30
    max_batch_records: int = 5_000
    """Most records the owner loop takes from the queue at once."""

    fuser: FuserOptions = field(default_factory=FuserOptions)
    duplicates: DuplicateOptions = field(default_factory=DuplicateOptions)
