"""Parse and validate a perspective's rows from ``Entity_Stream_Manager_Configurations``.

A perspective has one *perspective row*, whose ``origin_dataset`` starts with
``perspective_config``, and one row per datafeed. A datafeed row inherits every
key it doesn't set from the perspective row. ``head_update_interval_seconds`` is
the exception: a perspective value overrides the feed's, so it applies to every
feed. Crucible
returns many values as strings, so numbers and booleans are converted here,
once. Every error names the row and the key.
"""

import enum
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NoReturn, Self

from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.identity import DEFAULT_TRACK_ID_FIELDS

CONFIG_DATASET: Final = "Entity_Stream_Manager_Configurations"
PERSPECTIVE_ROW_PREFIX: Final = "perspective_config"

DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS: Final = 15.0
DEFAULT_HEAD_PRELOAD_LIMIT: Final = 100_000
DEFAULT_RESTORE_COOLDOWN_SECONDS: Final = 300
DEFAULT_SOURCE_QUEUE_MAX_RECORDS: Final = 50_000

SUPPORTED_HOOK_API_VERSIONS: Final = frozenset({"2", "native", "records"})
"""Accepted ``custom_function_api_version`` values; all mean the v2 record contract."""

IGNORED_KEYS: Final = {
    "number_of_transformer_processes": "use --partitions on the Execution instead",
    "num_tracker_workers": "use --partitions on the Execution instead",
    "num_fusion_workers": "the fuser runs as a single partition",
    "use_numpy_kalman": "the NumPy filter is the only implementation",
    "source_queue_max_batches": "use source_queue_max_records",
    "source_queue_maxsize": "use source_queue_max_records",
}
"""Keys from the baseline that no longer have an effect, with what replaces them."""

_TRUE_TEXT: Final = frozenset({"true", "1", "yes", "on"})
_FALSE_TEXT: Final = frozenset({"false", "0", "no", "off", ""})
_PROCESS_NOISE_PATTERN: Final = re.compile(r"q\s*=\s*(\d+(?:\.\d+)?)")


class ConfigError(ValueError):
    """A configuration row is missing a key or has an invalid value."""


class TrackerMode(enum.Enum):
    """How the tracker treats a datafeed, from its ``crucible_tracker`` value."""

    SKIP = enum.auto()
    """Not tracked here: unset, ``skip``, or a third-party tracker."""

    PASSTHROUGH = enum.auto()
    """Reports are copied to component tracks without filtering."""

    KALMAN = enum.auto()
    """Constant-velocity Kalman filtering."""

    KALMAN_CI = enum.auto()
    """Kalman filtering that fuses with covariance intersection."""

    UNRECOGNIZED = enum.auto()
    """A value that names no mode. The tracker rejects it at startup."""


@dataclass(frozen=True, slots=True)
class TrackerSettings:
    """A feed's tracker mode and its optional process-noise override."""

    mode: TrackerMode
    process_noise_q: float | None
    """Default process noise from a ``q=<number>`` token in the mode value."""
    raw: str

    @classmethod
    def parse(cls, value: str) -> Self:
        """Interpret a ``crucible_tracker`` value; matching is case-insensitive.

        ``ecef`` selects Kalman filtering, as it did before ``c7ab0b5``.
        """
        text = value.strip().lower()
        tokens = text.split()
        third_party = any(
            name in text for name in ("3rd party", "3rd-party", "third party", "third-party")
        )
        if not text or "skip" in text or third_party:
            mode = TrackerMode.SKIP
        elif "covariance intersection" in text or "ci" in tokens:
            mode = TrackerMode.KALMAN_CI
        elif "kalman" in text or "ecef" in text:
            mode = TrackerMode.KALMAN
        elif "passthrough" in text:
            mode = TrackerMode.PASSTHROUGH
        else:
            mode = TrackerMode.UNRECOGNIZED
        match = _PROCESS_NOISE_PATTERN.search(value)
        return cls(mode, float(match.group(1)) if match else None, value)


@dataclass(frozen=True, slots=True)
class OriginMapping:
    """Copy the value at `origin` (a dotted path) to `destination`."""

    origin: str
    destination: str


@dataclass(frozen=True, slots=True)
class LiteralMapping:
    """Set `destination` to a constant."""

    value: JSONValue
    destination: str


type FieldMapping = OriginMapping | LiteralMapping


@dataclass(frozen=True, slots=True)
class UnitConversion:
    """Apply the value hook `function_name` to the value at `origin`."""

    origin: str
    function_name: str


@dataclass(frozen=True, slots=True)
class Datasets:
    """The datasets the pipeline reads and writes."""

    management_events: str
    report_events: str
    principal_track_events: str
    principal_track_heads: str
    component_track_events: str | None
    component_track_heads: str | None


@dataclass(frozen=True, slots=True)
class WriteSettings:
    """Chunking and concurrency for writes to Crucible."""

    chunk_size: int = 250
    update_chunk_size: int = 50
    max_concurrent: int = 4


@dataclass(frozen=True, slots=True)
class PreloadSettings:
    """How components rebuild their state from heads at startup."""

    skip: bool
    tracker_limit: int
    fusion_limit: int


@dataclass(frozen=True, slots=True)
class FeedConfig:
    """One enabled datafeed, with perspective defaults applied."""

    origin_dataset: str
    query: str | None
    datasets: Datasets
    writes: WriteSettings
    preload: PreloadSettings
    mappings: tuple[FieldMapping, ...]
    unit_conversions: tuple[UnitConversion, ...]
    custom_functions: tuple[str, ...]
    tracker: TrackerSettings
    track_id_fields: tuple[str, ...]
    source_queue_max_records: int
    head_update_interval_seconds: float
    row: Mapping[str, JSONValue]
    """The merged row. Customer hooks receive deep copies of it (``hooks.loader.hook_row``)."""


@dataclass(frozen=True, slots=True)
class ScriptNames:
    """Names of the hook scripts in ``Entity_Stream_Manager_Functions``."""

    custom_functions: str | None
    unit_conversions: str | None


@dataclass(frozen=True, slots=True)
class PerspectiveConfig:
    """A validated perspective: shared settings and its enabled feeds."""

    name: str
    datasets: Datasets
    writes: WriteSettings
    preload: PreloadSettings
    restore_cooldown_seconds: int
    scripts: ScriptNames
    feeds: tuple[FeedConfig, ...]
    disabled_feeds: tuple[str, ...]
    warnings: tuple[str, ...]
    """Notes the caller should log, such as ignored legacy keys."""


def parse_perspective(
    name: str, rows: Sequence[JSONObject], environ: Mapping[str, str]
) -> PerspectiveConfig:
    """Validate a perspective's configuration rows.

    Args:
        name: The perspective, used in error messages.
        rows: Every row of `CONFIG_DATASET` for the perspective.
        environ: Process environment, for the defaults that the baseline read
            from ``CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS``,
            ``CRUCIBLE_SKIP_HEAD_PRELOAD`` and ``CRUCIBLE_HEAD_PRELOAD_LIMIT``.

    Raises:
        ConfigError: If any perspective setting or enabled feed is invalid.
    """
    perspective_rows = [row for row in rows if _origin(row).startswith(PERSPECTIVE_ROW_PREFIX)]
    if len(perspective_rows) != 1:
        msg = (
            f"perspective {name!r} needs exactly one row whose origin_dataset starts with "
            f"{PERSPECTIVE_ROW_PREFIX!r}; found {len(perspective_rows)}"
        )
        raise ConfigError(msg)
    perspective_row = perspective_rows[0]
    feed_rows = [row for row in rows if row is not perspective_row]
    warnings = [
        f"{_origin(row)}: {key} is ignored; {reason}"
        for row in rows
        for key, reason in IGNORED_KEYS.items()
        if key in row
    ]

    reader = _RowReader(perspective_row, environ)
    enabled_rows, disabled = _apply_enablement(feed_rows)
    feeds = tuple(_parse_feed(_merge(perspective_row, row), environ) for row in enabled_rows)
    _check_unique_origins(feeds)
    _check_script_names(perspective_row, enabled_rows)
    return PerspectiveConfig(
        name=name,
        datasets=reader.datasets(),
        writes=reader.writes(),
        preload=reader.preload(),
        restore_cooldown_seconds=reader.integer(
            "restore_cooldown_seconds", default=DEFAULT_RESTORE_COOLDOWN_SECONDS, minimum=0
        ),
        scripts=ScriptNames(
            custom_functions=reader.optional_text("custom_functions_script_name"),
            unit_conversions=reader.optional_text("unit_conversions_script_name"),
        ),
        feeds=feeds,
        disabled_feeds=tuple(_origin(row) for row in disabled),
        warnings=tuple(warnings),
    )


def _origin(row: JSONObject) -> str:
    value = row.get("origin_dataset")
    return value if isinstance(value, str) else ""


def _merge(perspective_row: JSONObject, feed_row: JSONObject) -> JSONObject:
    merged = {**perspective_row, **feed_row}
    if "head_update_interval_seconds" in perspective_row:
        merged["head_update_interval_seconds"] = perspective_row["head_update_interval_seconds"]
    return merged


def _apply_enablement(rows: list[JSONObject]) -> tuple[list[JSONObject], list[JSONObject]]:
    """Split feeds into enabled and disabled, honoring the single-feed focus flag.

    Both the ``disable_all_other_datasets`` spelling and the legacy misspelling
    ``disable_all_other_datsets`` set focus. At most one feed may set it.
    """
    focused = [
        row
        for row in rows
        if _RowReader(row).flag("disable_all_other_datasets")
        or _RowReader(row).flag("disable_all_other_datsets")
    ]
    if len(focused) > 1:
        origins = ", ".join(_origin(row) for row in focused)
        msg = f"only one feed may set disable_all_other_datasets; found {origins}"
        raise ConfigError(msg)
    if focused:
        return focused, [row for row in rows if row is not focused[0]]
    enabled: list[JSONObject] = []
    disabled: list[JSONObject] = []
    for row in rows:
        (disabled if _RowReader(row).flag("disabled") else enabled).append(row)
    return enabled, disabled


def _parse_feed(row: JSONObject, environ: Mapping[str, str]) -> FeedConfig:
    reader = _RowReader(row, environ)
    mappings = reader.mappings()
    reader.check_hook_api_version()
    return FeedConfig(
        origin_dataset=reader.text("origin_dataset"),
        query=reader.optional_text("query"),
        datasets=reader.datasets(),
        writes=reader.writes(),
        preload=reader.preload(),
        mappings=mappings,
        unit_conversions=reader.unit_conversions(),
        custom_functions=reader.custom_functions(),
        tracker=TrackerSettings.parse(reader.optional_text("crucible_tracker") or ""),
        track_id_fields=reader.track_id_fields(),
        source_queue_max_records=reader.integer(
            "source_queue_max_records", default=DEFAULT_SOURCE_QUEUE_MAX_RECORDS, minimum=1
        ),
        head_update_interval_seconds=reader.head_update_interval_seconds(),
        row=types.MappingProxyType(dict(row)),
    )


_SCRIPT_NAME_KEYS: Final = ("custom_functions_script_name", "unit_conversions_script_name")


def _check_script_names(perspective_row: JSONObject, feed_rows: list[JSONObject]) -> None:
    """Reject feeds that name a different hook script than the perspective.

    One process loads one script of each kind, so per-feed scripts can't be honored.
    """
    for row in feed_rows:
        for key in _SCRIPT_NAME_KEYS:
            if key in row and row[key] != perspective_row.get(key):
                msg = (
                    f"{_origin(row)}: {key} is {row[key]!r}, but the perspective uses "
                    f"{perspective_row.get(key)!r}; hook scripts are perspective-wide"
                )
                raise ConfigError(msg)


def _check_unique_origins(feeds: tuple[FeedConfig, ...]) -> None:
    seen: set[str] = set()
    for feed in feeds:
        if feed.origin_dataset in seen:
            msg = f"origin_dataset {feed.origin_dataset!r} is configured by more than one feed"
            raise ConfigError(msg)
        seen.add(feed.origin_dataset)


def _env_error(name: str, expected: str, text: str) -> NoReturn:
    msg = f"environment variable {name} must be {expected}, got {text!r}"
    raise ConfigError(msg) from None


def _as_int(value: JSONValue) -> int | None:
    """Read an integer from an int, an integral float, or decimal text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_float(value: JSONValue) -> float | None:
    """Read a finite or infinite number from an int, a float, or numeric text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


class _RowReader:
    """Typed, validated access to one configuration row."""

    def __init__(self, row: JSONObject, environ: Mapping[str, str] | None = None) -> None:
        self._row = row
        self._environ = environ or {}
        self._label = _origin(row) or "<row without origin_dataset>"

    def fail(self, key: str, problem: str) -> NoReturn:
        """Raise a `ConfigError` naming this row and `key`."""
        msg = f"{self._label}: {key} {problem}"
        raise ConfigError(msg)

    def text(self, key: str) -> str:
        value = self.optional_text(key)
        if value is None:
            self.fail(key, "is required")
        return value

    def optional_text(self, key: str) -> str | None:
        value = self._row.get(key)
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            self.fail(key, f"must be a string, got {type(value).__name__}")
        return value

    def flag(self, key: str, *, default: bool = False) -> bool:
        value = self._row.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = value.strip().lower() if isinstance(value, str) else None
        if text not in _TRUE_TEXT | _FALSE_TEXT:
            self.fail(key, f"must be true or false, got {value!r}")
        return text in _TRUE_TEXT

    def integer(self, key: str, *, default: int, minimum: int) -> int:
        number = self.optional_integer(key, minimum=minimum)
        return default if number is None else number

    def optional_integer(self, key: str, *, minimum: int) -> int | None:
        value = self._row.get(key)
        if value is None or value == "":
            return None
        number = _as_int(value)
        if number is None or number < minimum:
            self.fail(key, f"must be an integer of at least {minimum}, got {value!r}")
        return number

    def optional_number(self, key: str, *, minimum: float) -> float | None:
        value = self._row.get(key)
        if value is None or value == "":
            return None
        number = _as_float(value)
        if number is None or not number >= minimum:
            self.fail(key, f"must be a number of at least {minimum}, got {value!r}")
        return number

    def env_number(self, name: str, default: float) -> float:
        text = self._environ.get(name)
        if text is None or not text.strip():
            return default
        try:
            return float(text)
        except ValueError:
            _env_error(name, "a number", text)

    def env_integer(self, name: str, default: int) -> int:
        text = self._environ.get(name)
        if text is None or not text.strip():
            return default
        try:
            return int(text)
        except ValueError:
            _env_error(name, "an integer", text)

    def env_flag(self, name: str) -> bool:
        return self._environ.get(name, "").strip().lower() in _TRUE_TEXT

    def datasets(self) -> Datasets:
        return Datasets(
            management_events=self.text("entity_management_event_dataset"),
            report_events=self.text("report_event_dataset"),
            principal_track_events=self.text("principal_track_event_dataset"),
            principal_track_heads=self.text("principal_track_head_dataset"),
            component_track_events=self.optional_text("component_track_event_dataset"),
            component_track_heads=self.optional_text("component_track_head_dataset"),
        )

    def writes(self) -> WriteSettings:
        defaults = WriteSettings()
        return WriteSettings(
            chunk_size=self.integer(
                "batch_write_chunk_size", default=defaults.chunk_size, minimum=1
            ),
            update_chunk_size=self.integer(
                "batch_update_chunk_size", default=defaults.update_chunk_size, minimum=1
            ),
            max_concurrent=self.integer(
                "batch_write_max_concurrent", default=defaults.max_concurrent, minimum=1
            ),
        )

    def preload(self) -> PreloadSettings:
        tracker_limit = self.optional_integer("tracker_head_preload_limit", minimum=1)
        fusion_limit = self.optional_integer("fusion_head_preload_limit", minimum=1)
        if tracker_limit is None or fusion_limit is None:
            fallback = self.env_integer("CRUCIBLE_HEAD_PRELOAD_LIMIT", DEFAULT_HEAD_PRELOAD_LIMIT)
            tracker_limit = fallback if tracker_limit is None else tracker_limit
            fusion_limit = fallback if fusion_limit is None else fusion_limit
        return PreloadSettings(
            skip=self.flag(
                "skip_head_preload", default=self.env_flag("CRUCIBLE_SKIP_HEAD_PRELOAD")
            ),
            tracker_limit=tracker_limit,
            fusion_limit=fusion_limit,
        )

    def head_update_interval_seconds(self) -> float:
        interval = self.optional_number("head_update_interval_seconds", minimum=0.0)
        if interval is None:
            return self.env_number(
                "CRUCIBLE_HEAD_UPDATE_INTERVAL_SECONDS", DEFAULT_HEAD_UPDATE_INTERVAL_SECONDS
            )
        return interval

    def objects(self, key: str) -> list[JSONObject]:
        value = self._row.get(key)
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            self.fail(key, "must be a list of objects")
        return [item for item in value if isinstance(item, dict)]

    def mappings(self) -> tuple[FieldMapping, ...]:
        if "origin_to_destination_mapping" not in self._row:
            self.fail("origin_to_destination_mapping", "is required")
        mappings: list[FieldMapping] = []
        for entry in self.objects("origin_to_destination_mapping"):
            destination = entry.get("destination_column")
            if not isinstance(destination, str) or not destination:
                self.fail(
                    "origin_to_destination_mapping", f"entry {entry} has no destination_column"
                )
            origin = entry.get("origin_column")
            if ("literal" in entry) == (origin is not None):
                self.fail(
                    "origin_to_destination_mapping",
                    f"entry for {destination!r} needs exactly one of origin_column or literal",
                )
            if origin is None:
                mappings.append(LiteralMapping(entry["literal"], destination))
            elif isinstance(origin, str) and origin:
                mappings.append(OriginMapping(origin, destination))
            else:
                self.fail(
                    "origin_to_destination_mapping",
                    f"origin_column for {destination!r} must be a path",
                )
        destinations = [m.destination for m in mappings if isinstance(m, OriginMapping)]
        duplicates = sorted({d for d in destinations if destinations.count(d) > 1})
        if duplicates:
            self.fail("origin_to_destination_mapping", f"maps more than one origin to {duplicates}")
        if not any(destination.startswith("identity.") for destination in destinations):
            self.fail("origin_to_destination_mapping", "maps nothing to an identity.* field")
        return tuple(mappings)

    def unit_conversions(self) -> tuple[UnitConversion, ...]:
        conversions: list[UnitConversion] = []
        for entry in self.objects("unit_conversions"):
            origin, function_name = entry.get("origin_column"), entry.get("unit_conversion")
            if not isinstance(function_name, str) or "PLACEHOLDER" in function_name.upper():
                continue
            if not isinstance(origin, str) or not origin:
                self.fail("unit_conversions", f"entry for {function_name!r} has no origin_column")
            conversions.append(UnitConversion(origin, function_name))
        return tuple(conversions)

    def custom_functions(self) -> tuple[str, ...]:
        names: list[str] = []
        for entry in self.objects("custom_functions"):
            name = entry.get("function_name")
            if not isinstance(name, str) or not name:
                self.fail("custom_functions", f"entry {entry} has no function_name")
            names.append(name)
        return tuple(names)

    def track_id_fields(self) -> tuple[str, ...]:
        value = self._row.get("track_id_fields")
        if value is None:
            return DEFAULT_TRACK_ID_FIELDS
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(path, str) and path.strip() for path in value)
        ):
            self.fail("track_id_fields", "must be a non-empty list of field paths")
        return tuple(path.strip() for path in value if isinstance(path, str))

    def check_hook_api_version(self) -> None:
        for key in ("custom_function_api_version", "hook_api_version"):
            value = self._row.get(key)
            if value is not None and str(value).strip().lower() not in SUPPORTED_HOOK_API_VERSIONS:
                self.fail(key, f"is {value!r}; only the v2 record contract is supported")
