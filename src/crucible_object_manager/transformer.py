#!/usr/bin/env python
"""Polars object transformer with pandas custom-function compatibility.

Lifecycle:
1. run() loads configuration and starts the parent event loop through _serve().
2. The parent loads object snapshots, spawns workers, and routes source/cache SSE.
3. Each worker creates one PolarsObjectProcessor, which keeps its own object_cache
    across batches. It drains cache messages before processing each source batch.
4. process() converts input records to Polars, runs conversions/custom functions,
    maps fields, groups matching identities, joins cached UUIDs, and enriches creates.
5. The worker updates its projected cache and builds separate create, state, and
    kinematic payloads. ECEF transforms run on NumPy arrays extracted from columns.
6. _write_batch() upserts new objects first, then appends their events. Existing
    database objects are updated downstream by the object manager, not directly here.

Data shapes:
- Crucible input/output uses nested dictionaries, such as record['objectId']['uuid'].
- Polars columns use dotted names, such as 'objectId.uuid'; each row is one record.
- objectId.uuid is the database primary key. custom_ID_column is a temporary
  correlation key derived from configured identity fields, never a database key.
- Pandas is used only at configured DataFrame custom-function/enrichment boundaries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import queue
import re
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from multiprocessing import get_context
from typing import Any

import numpy as np
import polars as pl

try:
    from object_utils import (
        _import_cruciblelib_modules, _load_module_from_source, _script_sources,
        drop_records_missing_object_id, find_and_validate_configs,
        instantiate_api_controllers, normalize_uuid, run_sse_listener,
        write_batch_chunked,
    )
    from object_transformer_records import (
        MISSING, _ellipse_to_ecef_covariance, _enu_velocity_to_ecef,
        _geodetic_to_ecef, compact_record, get_path, set_path,
    )
except ImportError:
    from .object_utils import (
        _import_cruciblelib_modules, _load_module_from_source, _script_sources,
        drop_records_missing_object_id, find_and_validate_configs,
        instantiate_api_controllers, normalize_uuid, run_sse_listener,
        write_batch_chunked,
    )
    from .object_transformer_records import (
        MISSING, _ellipse_to_ecef_covariance, _enu_velocity_to_ecef,
        _geodetic_to_ecef, compact_record, get_path, set_path,
    )

# Database keys and temporary processing fields are kept separate.
PRIMARY_KEY = "objectId.uuid"
CUSTOM_ID = "custom_ID_column"
_INTEGER_PATHS = ("identity.navalPennant", "identity.mmsiNumber", "trackQuality")
_LABEL_PATHS = (
    "identity.sconum", "identity.callsign", "identity.vesselName",
    "identity.dynamicIdentifier", "identity.airPlatformType", "identity.hullNumber",
    "identity.environment.environment",
)
_ROW_LIMIT = 100_000
_REFRESH_CHUNK = int(os.getenv("CRUCIBLE_OBJECT_CACHE_FANOUT_CHUNK", "5000"))
_CACHE_QUEUE_SIZE = int(os.getenv("CRUCIBLE_OBJECT_CACHE_QUEUE_MAXSIZE", "20"))
_SOURCE_QUEUE_DEFAULT = int(os.getenv("CRUCIBLE_SOURCE_QUEUE_MAXSIZE", "100"))
# Controllers are initialized independently in the parent and each spawned worker.
auth: Any = None
rc: Any = None
wc: Any = None


# Input normalization and record/columnar conversion helpers.


def _paths(value):
    """Accept a single configured field path or a list and return a tuple of paths."""
    return tuple(str(item) for item in (value if isinstance(value, (list, tuple)) else [value]) if item is not None)


def _is_config_true(value):
    """Read boolean-like configuration values such as True, 'true', and '1'."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_id(value):
    """Canonicalize a scalar identity value; missing/multi-valued IDs become blank.

    Python integers are stringified directly so values above 2**53 stay exact.
    Integral floats and numeric strings such as '123.0' use the same ID as 123.
    """
    if value is None or value is MISSING or isinstance(value, (list, tuple, dict, np.ndarray, pl.Series)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value)
    if "." in text:
        try:
            number = float(text)
            if math.isfinite(number) and number.is_integer():
                return str(int(number))
        except ValueError:
            pass
    return text


def _numeric_int(value):
    """Produce an exact nullable Int64 value, rejecting fractions and overflow.

    Decimal avoids introducing float rounding while parsing integer ID strings.
    Missing, nonnumeric, and nonfinite values become None.
    """
    if value is None or value is MISSING:
        return None
    try:
        number = Decimal(int(value)) if isinstance(value, bool) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not number.is_finite():
        return None
    if number != number.to_integral_value():
        raise TypeError(f"cannot safely cast non-integral value {value!r} to Int64")
    if not -(2**63) <= number < 2**63:
        raise OverflowError(f"integer {value!r} is outside Int64 range")
    return int(number)


def _normalize_timestamp(value):
    """Format a valid observation time as UTC milliseconds; invalid/sentinel times become null."""
    if value is None or value is MISSING or value == "":
        return None
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    result = parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return None if result == "1970-01-01T00:00:00.000Z" else result


def _flatten(record, prefix=""):
    """Convert nested dictionary leaves to dotted column names, keeping list values intact."""
    result = {}
    for key, value in record.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, field))
        else:
            result[field] = value
    return result


def _records_frame(records):
    """Build a Polars frame from nested records without pandas numeric coercion.

    Known integer fields are explicitly Int64. Heterogeneous values that cannot
    share a Polars/Arrow type use Object columns rather than losing information.
    """
    flat = [_flatten(record) for record in records]
    columns = sorted({field for record in flat for field in record})
    series = []
    for field in columns:
        values = [record.get(field) for record in flat]
        values = [None if isinstance(value, float) and not math.isfinite(value) else value for value in values]
        if field in _INTEGER_PATHS:
            series.append(pl.Series(field, [_numeric_int(value) for value in values], dtype=pl.Int64))
            continue
        try:
            series.append(pl.Series(field, values, strict=True))
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            series.append(pl.Series(field, values, dtype=pl.Object))
    return pl.DataFrame(series)


def _frame_records(frame):
    """Rebuild nested write payloads, removing temporary fields and empty/nonfinite values."""
    result = []
    columns = [field for field in frame.columns if field not in (CUSTOM_ID, "_created")]
    for row in frame.select(columns).iter_rows(named=True):
        record = {}
        for field, value in row.items():
            if value is not None:
                set_path(record, field, value)
        result.append(compact_record(record))
    return result


def _pandas_frame(frame):
    """Adapt Polars columns for legacy hooks, preserving nullable integers exactly."""
    import pandas as pd
    import pyarrow as pa

    def dtype(arrow_type):
        """Choose pandas nullable integer dtypes instead of float-with-NaN columns."""
        if pa.types.is_signed_integer(arrow_type):
            return pd.Int64Dtype()
        if pa.types.is_unsigned_integer(arrow_type):
            return pd.UInt64Dtype()
        return None

    objects = [field for field, kind in frame.schema.items() if kind == pl.Object]
    # Arrow handles normal columns; arbitrary Python objects need a separate path.
    result = frame.drop(objects).to_pandas(types_mapper=dtype)
    for field in objects:
        result[field] = pd.Series(frame[field].to_list(), dtype=object)
    return result


def _from_pandas(frame, schema=None):
    """Return hook output via Arrow; retain known column types when no rows remain.

    An empty pandas object column cannot describe its original List/Object type.
    The optional input schema supplies that information without constraining
    populated hook outputs or newly introduced columns.
    """
    import pandas as pd
    import pyarrow as pa
    try:
        result = pl.from_pandas(frame.reset_index(drop=True), nan_to_null=True)
        if result.is_empty() and schema is not None:
            result = pl.DataFrame(schema={field: schema.get(field, dtype) for field, dtype in result.schema.items()})
        return result.with_columns([pl.col(field).map_elements(_numeric_int, return_dtype=pl.Int64)
                                    for field in _INTEGER_PATHS if field in result.columns])
    except (TypeError, ValueError, pa.ArrowException, pl.exceptions.PolarsError):
        pass
    # Some hooks return mixed objects/lists that Arrow cannot represent uniformly.
    records = []
    for row in frame.to_dict("records"):
        record = {}
        for field, value in row.items():
            if not isinstance(value, (list, tuple, dict, np.ndarray)) and pd.isna(value):
                value = None
            set_path(record, str(field), value.tolist() if isinstance(value, np.ndarray) else value)
        records.append(record)
    return _records_frame(records)


def _keyed(frame, fields):
    """Add the temporary correlation key using sorted field names and canonical values.

    Sorting makes multi-field IDs independent of config order. Every component
    must be nonblank; otherwise the row cannot match or create an object.
    """
    frame = frame.with_columns([pl.lit(None).alias(field) for field in fields if field not in frame.columns])
    parts = [pl.col(field).map_elements(_as_id, return_dtype=pl.String, skip_nulls=False) for field in sorted(fields)]
    valid = pl.all_horizontal([part.str.strip_chars() != "" for part in parts])
    return frame.with_columns(pl.when(valid).then(pl.concat_str([
        pl.lit(field + "-") + part for field, part in zip(sorted(fields), parts, strict=True)
    ], separator="_")).otherwise(None).alias(CUSTOM_ID))


def _object_template():
    """Return default object-definition fields, applied only when creating an object."""
    return {
        "edhControlSet": ["CLS:U"], "identity": {"dynamicIdentifier": "",
            "environment": {"confidence": 100, "environment": "AIR"},
            "standard": {"allegiance": "", "confidence": 100, "standardIdentity": "NEUTRAL"}},
        "mobility": "MOVER", "mode": "LIVE",
        "objectId": {"descriptiveLabel": "", "uuid": ""}, "entityStatus": "UNKNOWN",
    }


def _dataset_name(query):
    """Extract the source dataset name used in emitted event provenance."""
    match = re.search(r"(?<=from\s)(?:['\"])?([^\s'\"]+)", query, re.IGNORECASE)
    return match.group(1) if match else "dataset_name_unknown"


def _add_ecef(frame):
    """Filter unusable observations and add ECEF position, velocity, and covariance.

    Latitude/longitude and ellipse orientation are radians; altitude is metres.
    Numeric work is batched in NumPy, then attached as Polars columns. Missing
    altitude/down-speed defaults to zero; absent velocity/ellipse fields are optional.
    """
    position = "estimatedKinematics.position."
    timestamp = "estimatedKinematics.kinematicsTimestamp"
    if not all(field in frame.columns for field in (timestamp, position + "latitude", position + "longitude")):
        return frame.head(0)
    frame = frame.filter(pl.col(timestamp).is_not_null())

    def column(field, default=np.nan):
        """Extract one numeric column as an array, supplying defaults for null/absent values."""
        return frame[field].cast(pl.Float64, strict=False).fill_null(default).to_numpy() if field in frame.columns else np.full(frame.height, default)

    latitude, longitude = column(position + "latitude"), column(position + "longitude")
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    frame = frame.filter(pl.Series(valid))
    latitude, longitude = latitude[valid], longitude[valid]
    altitude = column(position + "altitude", 0.)
    x, y, z = _geodetic_to_ecef(latitude, longitude, altitude)
    additions = [pl.Series(position + "altitude", altitude)]
    additions.extend(pl.Series("ecefPosition." + axis, values) for axis, values in zip(("x", "y", "z"), (x, y, z), strict=True))
    velocity = "estimatedKinematics.velocity."
    if all(velocity + field in frame.columns for field in ("eastSpeed", "northSpeed")):
        east, north = column(velocity + "eastSpeed"), column(velocity + "northSpeed")
        valid_velocity = np.isfinite(east) & np.isfinite(north)
        values = _enu_velocity_to_ecef(latitude, longitude, east, north, -column(velocity + "downSpeed", 0.))
        additions.extend(pl.Series("ecefVelocity." + axis, np.where(valid_velocity, value, np.nan))
                         for axis, value in zip(("dx", "dy", "dz"), values, strict=True))
    ellipse = "estimatedKinematics.uncertainty.uncertaintyEllipse."
    if ellipse + "semiMajorAxisLength" in frame.columns:
        covariance = _ellipse_to_ecef_covariance(latitude, longitude, column(ellipse + "semiMajorAxisLength", 1000.),
            column(ellipse + "semiMinorAxisLength", 1000.), column(ellipse + "orientation", 0.))
        additions.extend(pl.Series("positionCovariance." + field, covariance[:, row, col]) for field, row, col in
            (("xx", 0, 0), ("xy", 0, 1), ("xz", 0, 2), ("yy", 1, 1), ("yz", 1, 2), ("zz", 2, 2)))
    return frame.with_columns(additions)


# Per-worker processing state. This class does not represent a database object.


def _state_signature_value(value):
    """Ignore numeric dtype drift in state signatures without rounding integer values."""
    if value is None or value is MISSING:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return _as_id(value)
    return repr(value)


class PolarsObjectProcessor:
    """Keep one worker's object cache and turn source batches into write payloads.

    self is the processor instance created in _run_worker(), not a Crucible object.
    object_cache is a Polars table containing many cached objects. process() returns
    payloads without network I/O; _write_batch() performs the writes separately.
    """

    def __init__(self, config, records):
        """Initialize cache schema, identity paths, dedup state, and reconciliation timers."""
        self.config = config
        self.destination_paths = _paths(config["destination_unique_ID_column"])
        self.template = _flatten(_object_template())
        # Remember observed input paths even when their values are omitted from the cache.
        self.accepted_paths = set(self.template) | {field for record in records for field in _flatten(record)}
        frame = _records_frame(records) if records else _records_frame([_object_template()]).head(0)
        self.object_cache = self._project(frame)
        self.state_cache = {}
        self.local_write_times = {}
        self.last_timings = {}

    def _project(self, frame):
        """Retain UUID, all identity fields, core/default fields, update time, and configured IDs.

        Kinematics and unrelated payload fields do not live in the worker cache.
        Rows without UUIDs are excluded; repeated UUIDs retain the last record.
        """
        fields = set(self.template) | set(self.destination_paths) | {PRIMARY_KEY, "objectId.descriptiveLabel", "entityStatus", "crucibleHeader.updatedDate"}
        frame = frame.select([field for field in frame.columns if field in fields or field.startswith("identity.")])
        if PRIMARY_KEY not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(PRIMARY_KEY))
        frame = frame.with_columns(pl.col(PRIMARY_KEY).cast(pl.String)).filter(pl.col(PRIMARY_KEY).is_not_null() & (pl.col(PRIMARY_KEY) != ""))
        return _keyed(frame, self.destination_paths).unique(PRIMARY_KEY, keep="last", maintain_order=True)

    def _hooks(self, frame):
        """Run scalar unit conversions and custom functions before source-field mapping.

        Hooks may derive fields required by mappings. DataFrame hooks receive
        pandas frames; JSON hooks receive nested records. Both see the projected
        object cache, and returned cache changes are projected again.
        """
        for conversion in self.config.get("unit_conversions", []):
            name, field = conversion.get("unit_conversion", ""), conversion.get("origin_column")
            if field not in frame.columns or not name or "PLACEHOLDER" in name.upper():
                continue
            function = getattr(globals()["unit_conversions"], name)
            values = [function(value) if value is not None else None for value in frame[field].to_list()]
            frame = frame.with_columns(pl.Series(field, values, strict=False))
        functions = self.config.get("custom_functions", [])
        if not functions:
            return frame
        kind = self.config.get("custom_function_api_version", "dataframe")
        if kind == "dataframe":
            events, objects = _pandas_frame(frame), _pandas_frame(self.object_cache.drop(CUSTOM_ID))
            for entry in functions:
                events, objects = getattr(globals()["custom_functions"], entry["function_name"])(events, objects, self.config)
            frame, cache = _from_pandas(events, schema=frame.schema), _from_pandas(objects, schema=self.object_cache.schema)
        elif kind == "json":
            events, objects = _frame_records(frame), _frame_records(self.object_cache)
            for entry in functions:
                events, objects = getattr(globals()["custom_functions"], entry["function_name"])(events, objects, self.config)
            frame, cache = _records_frame(events), _records_frame(objects)
        else:
            raise ValueError("custom_function_api_version must be 'dataframe' or 'json'")
        self.accepted_paths.update(cache.columns)
        self.object_cache = self._project(cache)
        return frame

    def _mapping(self, frame):
        """Map source fields/literals, normalize IDs/timestamps, and coalesce duplicate identities."""
        mappings = list(self.config.get("origin_to_destination_mapping", []))
        destinations = [item["destination_column"] for item in mappings]
        if len(destinations) != len(set(destinations)):
            raise ValueError("Duplicate destination columns in config files are not allowed")
        for source, destination in zip(_paths(self.config["origin_unique_ID_column"]), self.destination_paths, strict=True):
            if destination not in destinations:
                mappings.append({"origin_column": source, "destination_column": destination})
        if "source.uuid" not in destinations:
            mappings.append({"origin_column": "crucibleHeader.uuid", "destination_column": "source.uuid"})
        expressions = {}
        accepted = self.accepted_paths | {item["destination_column"] for item in mappings}
        for field in frame.columns:
            if field in accepted and not field.startswith("crucibleHeader"):
                expressions[field] = pl.col(field)
        for item in mappings:
            destination, source = item["destination_column"], item.get("origin_column")
            if "literal" in item:
                value = item["literal"]
                kind = str(item.get("type", "")).lower()
                value = _numeric_int(value) if kind == "int" else float(value) if kind == "float" else str(value) if kind == "str" else value
                expressions[destination] = pl.lit(value)
            else:
                expressions[destination] = pl.col(source) if source in frame.columns else pl.lit(None)
        expressions["source.datasetName"] = pl.lit(_dataset_name(str(self.config.get("query", ""))))
        frame = frame.select([expression.alias(field) for field, expression in expressions.items()])
        if "source.uuid" in frame.columns:
            frame = frame.with_columns(pl.col("source.uuid").cast(pl.String).str.strip_chars().str.replace_all("-", "", literal=True))
        frame = frame.with_columns([pl.col(field).map_elements(_numeric_int, return_dtype=pl.Int64) for field in _INTEGER_PATHS if field in frame.columns])
        frame = frame.with_columns([pl.col(field).map_elements(_normalize_timestamp, return_dtype=pl.String)
                                   for field in frame.columns if "imestamp" in field.casefold()])
        self.accepted_paths.update(frame.columns)
        frame = _keyed(frame, self.destination_paths).filter(pl.col(CUSTOM_ID).is_not_null())
        # Last non-null per field preserves partial updates from repeated source IDs.
        return frame.group_by(CUSTOM_ID, maintain_order=True).agg([
            pl.col(field).drop_nulls().last().alias(field) for field in frame.columns if field != CUSTOM_ID])

    def _match(self, events):
        """Join correlation keys to cached UUIDs and mark unmatched rows as creates.

        Static correlation discards unknown identities. Dynamic correlation uses
        a supplied UUID when mapped, otherwise deterministic UUID5 of the custom ID.
        Existing objects keep their cached UUID and do not receive template defaults.
        """
        cache = self.object_cache.filter(pl.col(CUSTOM_ID).is_not_null()).unique(CUSTOM_ID, keep="last", maintain_order=True)
        matched = events.join(cache.select(CUSTOM_ID, pl.col(PRIMARY_KEY).alias("_cached_uuid")), on=CUSTOM_ID, how="left")
        matched = matched.with_columns(pl.col("_cached_uuid").is_null().alias("_created"))
        if "dynamic" not in str(self.config.get("correlation_type", "")).lower():
            matched = matched.filter(~pl.col("_created"))
        uuid_mapped = any(item["destination_column"] == PRIMARY_KEY for item in self.config.get("origin_to_destination_mapping", []))
        if uuid_mapped:
            if PRIMARY_KEY not in matched.columns:
                return matched.head(0).drop("_cached_uuid")
            matched = matched.filter(~pl.col("_created") | (pl.col(PRIMARY_KEY).is_not_null() & (pl.col(PRIMARY_KEY).cast(pl.String) != "")))
            generated = pl.col(PRIMARY_KEY).cast(pl.String)
        else:
            ids = {key: uuid.uuid5(uuid.NAMESPACE_DNS, key).hex for key in matched.filter(pl.col("_created"))[CUSTOM_ID].to_list()}
            generated = pl.col(CUSTOM_ID).replace_strict(ids, default=None, return_dtype=pl.String)
        matched = matched.with_columns(pl.coalesce(pl.col("_cached_uuid"), generated).alias(PRIMARY_KEY)).drop("_cached_uuid")
        additions = []
        for field, value in self.template.items():
            if field in (PRIMARY_KEY, "objectId.descriptiveLabel"):
                continue
            current = pl.col(field) if field in matched.columns else pl.lit(None)
            additions.append(pl.when(pl.col("_created")).then(pl.coalesce(current, pl.lit(value))).otherwise(current).alias(field))
        return self._labels(matched.with_columns(additions))

    def _labels(self, frame):
        """Choose a required nonblank descriptive label from incoming/cached identity values."""
        label_fields = [field for field in _LABEL_PATHS if field in frame.columns or field in self.object_cache.columns]
        if not label_fields:
            return frame.with_columns(pl.lit("UNKNOWN").alias("objectId.descriptiveLabel"))
        prior = self.object_cache.select(PRIMARY_KEY, *[pl.col(field).alias(field + "__label") for field in label_fields if field in self.object_cache.columns])
        frame = frame.join(prior, on=PRIMARY_KEY, how="left")
        labels = []
        for field in label_fields:
            candidates = [pl.col(name).map_elements(_as_id, return_dtype=pl.String) for name in (field, field + "__label") if name in frame.columns]
            labels.append(pl.coalesce([pl.when(candidate.str.strip_chars() != "").then(candidate) for candidate in candidates]))
        frame = frame.with_columns(pl.coalesce(*labels, pl.lit("UNKNOWN")).alias("objectId.descriptiveLabel"))
        return frame.drop([field for field in frame.columns if field.endswith("__label")])

    def _enrich(self, frame):
        """Call the legacy pandas enrichment API for creates and join its returned fields back."""
        class_name = self.config.get("object_enrichment_class")
        created = frame.filter(pl.col("_created"))
        if not class_name or created.is_empty():
            return frame
        enricher = getattr(globals()["object_enrichment"], class_name)()
        enriched = _from_pandas(enricher.enrich_object_with_reference_data(
            _pandas_frame(created.drop("_created")), _pandas_frame(frame.drop("_created"))))
        key = PRIMARY_KEY if PRIMARY_KEY in enriched.columns else CUSTOM_ID
        if key not in enriched.columns:
            raise ValueError("object enrichment must retain objectId.uuid or custom_ID_column")
        enriched = enriched.unique(key, keep="last", maintain_order=True)
        fields = [field for field in enriched.columns if field not in (key, CUSTOM_ID, "_created")]
        frame = frame.with_columns([pl.lit(None).cast(enriched.schema[field]).alias(field) for field in fields if field not in frame.columns])
        patch = enriched.select(key, *[pl.col(field).alias(field + "__enriched") for field in fields])
        frame = frame.join(patch, on=key, how="left").with_columns([
            pl.coalesce(pl.col(field + "__enriched"), pl.col(field)).alias(field) for field in fields
        ]).drop([field + "__enriched" for field in fields])
        return self._labels(frame)

    def _update_cache(self, updates):
        """Merge projected non-null patches by UUID and append newly created objects."""
        updates = self._project(updates)
        if updates.is_empty():
            return
        cache = self.object_cache
        for field in updates.columns:
            if field not in cache.columns:
                cache = cache.with_columns(pl.lit(None).cast(updates.schema[field]).alias(field))
        fields = [field for field in updates.columns if field != PRIMARY_KEY]
        patch = updates.select(PRIMARY_KEY, *[pl.col(field).alias(field + "__patch") for field in fields])
        # coalesce chooses the patch first; null patches must not erase cached values.
        cache = cache.join(patch, on=PRIMARY_KEY, how="left").with_columns([
            pl.coalesce(pl.col(field + "__patch"), pl.col(field)).alias(field) for field in fields
        ]).drop([field + "__patch" for field in fields])
        additions = updates.join(self.object_cache.select(PRIMARY_KEY), on=PRIMARY_KEY, how="anti")
        self.object_cache = self._project(pl.concat([cache, additions], how="diagonal_relaxed"))

    def _emit_state(self, event):
        """Emit changed state or a periodic/count-based heartbeat; never filter kinematic events.

        Signatures ignore timing/provenance-only differences. Per-object jitter
        spreads time-based heartbeats instead of letting all objects emit together.
        """
        if not _is_config_true(self.config.get("dedupe_state_updates", True)):
            return True
        signature = tuple(sorted((field, _state_signature_value(value)) for field, value in _flatten(event).items()
            if field not in (PRIMARY_KEY, "eventType", "collectionType") and "imestamp" not in field.casefold()
            and not field.startswith(("source", "upstreamSource", "latestSource", "latestUpstreamSource"))))
        key = (get_path(event, PRIMARY_KEY), get_path(event, "source.datasetName"),
               str(event.get("upstreamSource", "")), str(event.get("collectionType", "")))
        now, previous = time.monotonic(), self.state_cache.get(key)
        if previous is None or previous[0] != signature:
            if len(self.state_cache) >= 500_000:
                self.state_cache.clear()
            self.state_cache[key] = (signature, now, 0)
            return True
        repeats = previous[2] + 1
        try:
            seconds = max(0., float(self.config.get("state_heartbeat_seconds", 300)))
        except (ValueError, TypeError):
            seconds = 300.
        count_setting = self.config.get("state_heartbeat_every_n", 100)
        try:
            every_n = 0 if count_setting is None else max(0, int(count_setting))
        except (ValueError, TypeError):
            every_n = 100
        jitter = ((hash(key[0]) % 1000) / 1000. - .5) * .5 * seconds
        emit = (seconds > 0 and now - previous[1] >= seconds + jitter) or (every_n > 0 and repeats >= every_n)
        self.state_cache[key] = (signature, now if emit else previous[1], 0 if emit else repeats)
        return emit

    def process(self, payload):
        """Process one source batch and return new_objects, state_events, and kinematic_events.

        This mutates the local cache but does not write to Crucible. Create payloads
        carry full initial definitions; existing-object event payloads carry patches.
        """
        start = time.perf_counter()
        records = json.loads(payload) if isinstance(payload, str) else payload
        records = [records] if isinstance(records, Mapping) else records
        if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
            raise ValueError("event payload must be an object or an array of objects")
        empty = {"new_objects": [], "state_events": [], "kinematic_events": []}
        if not records:
            return empty
        # Phase 1: hooks see raw source fields; mapping then produces destination fields.
        events = self._mapping(self._hooks(_records_frame(records)))
        after_preprocess = time.perf_counter()
        if events.is_empty():
            return empty
        # Phase 2: resolve database UUIDs, enrich creates, and update local cache state.
        frame = self._enrich(self._match(events))
        if frame.is_empty():
            return empty
        self._update_cache(frame)
        for object_id in frame[PRIMARY_KEY].to_list():
            self.local_write_times[object_id] = time.monotonic()
        after_match = time.perf_counter()
        # Phase 3: separate object definitions from append-only state/kinematic events.
        created = frame.filter(pl.col("_created")).select([field for field in frame.columns
            if not field.startswith(("source", "upstreamSource", "collectionType", "crucibleHeader",
                                     "estimatedKinematics.uncertainty.uncertaintyEllipse"))])
        state = frame.select([field for field in frame.columns if not field.startswith(("estimatedKinematics", "crucibleHeader"))])
        state = state.with_columns(pl.lit("STATE_UPDATE").alias("eventType"))
        kin = frame.select([field for field in frame.columns if field.startswith((
            "estimatedKinematics", "objectId", "source", "upstreamSource", "collectionType", "edhControlSet", "mode", "identity", "ecef", "positionCovariance"))])
        kin = _add_ecef(kin.with_columns(pl.lit("KINEMATIC_UPDATE").alias("eventType")))
        # Convert to nested records only at the output boundary; internal work stays columnar.
        result = {"new_objects": _frame_records(created),
            "state_events": [record for record in _frame_records(state) if self._emit_state(record)],
            "kinematic_events": _frame_records(kin)}
        self.last_timings = {"preprocess": after_preprocess - start, "match/cache": after_match - after_preprocess,
                             "events/ecef": time.perf_counter() - after_match}
        return result

    def remove_uuid(self, object_id):
        """Remove one cached object and its recent-write protection timer."""
        self.object_cache = self.object_cache.filter(pl.col(PRIMARY_KEY) != object_id)
        self.local_write_times.pop(object_id, None)

    def apply_remote_records(self, records):
        """Apply remote tombstones and add unseen objects without overwriting local existing rows.

        This is additions-only for live objects, matching the legacy cache-sync
        policy. A tombstone wins over a snapshot row for the same UUID in this batch.
        """
        if not records:
            return
        self.accepted_paths.update(field for record in records for field in _flatten(record))
        dropped = [str(get_path(record, PRIMARY_KEY)) for record in records if record.get("entityStatus") == "DROPPED"]
        for object_id in dropped:
            self.remove_uuid(object_id)
        additions = self._project(_records_frame([record for record in records if record.get("entityStatus") != "DROPPED"]))
        if dropped:
            additions = additions.filter(~pl.col(PRIMARY_KEY).is_in(dropped))
        additions = additions.join(self.object_cache.select(PRIMARY_KEY), on=PRIMARY_KEY, how="anti")
        self.object_cache = self._project(pl.concat([self.object_cache, additions], how="diagonal_relaxed"))

    def reconcile(self, remote_ids, safety_seconds=10.):
        """Prune UUIDs absent remotely, protecting recently written/read objects against read lag."""
        now_mono, now_epoch = time.monotonic(), time.time()
        fields = [PRIMARY_KEY] + (["crucibleHeader.updatedDate"] if "crucibleHeader.updatedDate" in self.object_cache.columns else [])
        for row in self.object_cache.select(fields).iter_rows(named=True):
            object_id = row[PRIMARY_KEY]
            if object_id in remote_ids or now_mono - self.local_write_times.get(object_id, -math.inf) < safety_seconds:
                continue
            stamp = _normalize_timestamp(row.get("crucibleHeader.updatedDate"))
            if stamp and now_epoch - datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() < safety_seconds:
                continue
            self.remove_uuid(object_id)

    def _remove_failed_new_objects(self, records):
        """Forget failed creates and their dedup signatures so a future batch can retry them."""
        failed_ids = {str(get_path(record, PRIMARY_KEY)) for record in records}
        for object_id in failed_ids:
            self.remove_uuid(object_id)
        self.state_cache = {key: value for key, value in self.state_cache.items() if key[0] not in failed_ids}


# Parent/worker lifecycle, queues, and Crucible I/O.


def _enqueue_source_message(target, message):
    """Enqueue a source batch, evicting the oldest on overflow; return (queued, evicted).

    Queue capacity counts whole SSE messages/batches, not individual records.
    Evicted source data is not replayed by object-cache refresh.
    """
    try:
        target.put_nowait(message)
        return True, False
    except queue.Full:
        pass
    try:
        target.get(timeout=.01)
        evicted = True
    except queue.Empty:
        evicted = False
    try:
        target.put_nowait(message)
        return True, evicted
    except queue.Full:
        return False, evicted


def _run_worker(source_queue, cache_queue, initial_records, config):
    """Initialize one child process, then serially drain cache updates and process/write batches."""
    global auth, rc, wc
    _, _, _, log_utils, _, _ = _import_cruciblelib_modules()
    log_utils.get_logger(log_type="transformer", log_level=config.get("_log_level", logging.INFO))
    auth, rc, wc = instantiate_api_controllers()
    for name, source in config.get("_script_sources", {}).items():
        globals()[name] = _load_module_from_source(name, source)
    processor = PolarsObjectProcessor(config, initial_records)
    # The processor owns the projected columns now; release the full startup snapshot.
    initial_records.clear()
    logging.info("[%s]: Polars transformer started with %d objects", config.get("origin_dataset"), processor.object_cache.height)
    while True:
        # Apply remote creates/deletes before matching the next source batch.
        try:
            _drain_cache_queue(cache_queue, processor)
        except Exception:
            logging.exception("Object cache drain failed")
        try:
            event = source_queue.get(timeout=1)
        except queue.Empty:
            continue
        if getattr(event, "type", None) == "keep-alive":
            continue
        try:
            start = time.perf_counter()
            batch = processor.process(getattr(event, "data", event))
            before_write = time.perf_counter()
            asyncio.run(_write_batch(batch, processor, config))
            if _is_config_true(os.getenv("CRUCIBLE_PERF_TIMING", "true")):
                logging.info("[%s]: [PERF transformer polars] process=%.3fs write=%.3fs stages=%s rows=%d cache=%d",
                    config.get("origin_dataset"), before_write - start, time.perf_counter() - before_write,
                    processor.last_timings, len(batch["kinematic_events"]), processor.object_cache.height)
        except Exception:
            logging.exception("[%s]: Polars transformer batch failed", config.get("origin_dataset"))


async def _write_batch(batch, processor, config):
    """Upsert creates first, then append state and kinematic events concurrently.

    Failed creates are removed locally and their events withheld. Failed append-only
    events are logged, not replayed, to avoid duplicate event writes.
    """
    label = f" [{config.get('origin_dataset', '')}]: "
    for key in batch:
        batch[key] = drop_records_missing_object_id(batch[key], label=label)
    failed = []
    if batch["new_objects"]:
        failed, _ = await write_batch_chunked(batch["new_objects"], config["object_dataset"], wc.upsert_by_name,
            int(config["batch_update_chunk_size"]), label=label, token_refresher=lambda: setattr(wc, "token", auth.get_token()),
            max_concurrent_writes=int(config.get("batch_write_max_concurrent") or 1), transient_retry_attempts=2)
    if failed:
        processor._remove_failed_new_objects(failed)
        failed_ids = {get_path(record, PRIMARY_KEY) for record in failed}
        for key in ("state_events", "kinematic_events"):
            batch[key] = [record for record in batch[key] if get_path(record, PRIMARY_KEY) not in failed_ids]
    tasks = [write_batch_chunked(batch[key], config["object_event_dataset"], wc.write_record_batch_by_name,
        int(config["batch_write_chunk_size"]), label=label, token_refresher=lambda: setattr(wc, "token", auth.get_token()),
        max_concurrent_writes=int(config.get("batch_write_max_concurrent") or 1))
        for key in ("state_events", "kinematic_events") if batch[key]]
    for failed_events, _ in await asyncio.gather(*tasks):
        if failed_events:
            logging.error("%s%d append-only events failed; not replayed", label, len(failed_events))


def _drain_cache_queue(cache_queue, processor):
    """Drain live cache updates and the newest refresh snapshot, then reconcile UUID removals."""
    remote, snapshots, reconcile = [], {}, None
    while True:
        try:
            event = cache_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(event, Mapping) and event.get("_cache_msg") == "records":
            generation = event.get("_cache_generation")
            target = remote if generation is None else snapshots.setdefault(int(generation), [])
            target.extend(event.get("records", []))
        elif isinstance(event, Mapping) and event.get("_cache_msg") == "reconcile":
            if reconcile is None or event.get("_cache_generation", 0) >= reconcile.get("_cache_generation", 0):
                reconcile = event
        elif getattr(event, "type", None) != "keep-alive" and getattr(event, "data", None):
            records = json.loads(event.data)
            remote.extend(records if isinstance(records, list) else [records])
    if snapshots:
        # Do not prune using an older UUID snapshot after applying newer object records.
        latest = max(snapshots)
        remote.extend(snapshots[latest])
        if reconcile and reconcile.get("_cache_generation", 0) < latest:
            reconcile = None
    processor.apply_remote_records(remote)
    if reconcile and isinstance(reconcile.get("remote_uuids"), list):
        processor.reconcile(set(reconcile["remote_uuids"]), float(reconcile.get("safety_seconds", 10)))


async def _source_listener(config, source_queues):
    """Consume source SSE; route whole batches or sticky partitions by the configured shard key."""
    async def on_event(event):
        """Ignore keep-alives and enqueue this SSE payload into its worker queue(s)."""
        if getattr(event, "type", None) == "keep-alive":
            return
        if len(source_queues) == 1:
            messages = [(source_queues[0], event)]
        else:
            data = getattr(event, "data", event)
            records = json.loads(data) if isinstance(data, str) else data
            records = [records] if isinstance(records, dict) else records
            partitions = [[] for _ in source_queues]
            for record in records:
                value = get_path(record, config["transformer_shard_key"])
                partitions[hash(str(value)) % len(partitions)].append(record)
            messages = [(target, partition) for target, partition in zip(source_queues, partitions, strict=True) if partition]
        for target, message in messages:
            queued, evicted = _enqueue_source_message(target, message)
            if evicted:
                logging.warning("[%s]: source queue full; evicted oldest queued event", config.get("origin_dataset"))
            if not queued:
                logging.error("[%s]: source queue remained full; dropping incoming event", config.get("origin_dataset"))
    await run_sse_listener(config["query"], auth, on_event, label=str(config.get("origin_dataset", "")))


async def _cache_listener(dataset, queues):
    """Broadcast object-dataset SSE to every worker holding a private copy of that cache."""
    async def on_event(event):
        """Broadcast cache changes; wait for tombstones rather than dropping deletions on overflow."""
        data = getattr(event, "data", None)
        rows = json.loads(data) if data and getattr(event, "type", None) != "keep-alive" else []
        rows = [rows] if isinstance(rows, dict) else rows
        tombstone = any(record.get("entityStatus") == "DROPPED" for record in rows)
        for target in queues:
            try:
                if tombstone:
                    await asyncio.to_thread(target.put, event)
                else:
                    target.put_nowait(event)
            except queue.Full:
                logging.warning("[%s]: object cache queue full; refresh will repair", dataset)
    await run_sse_listener("select * from " + dataset, auth, on_event, label="object_cache:" + dataset)


def _interval_seconds(value):
    """Parse numeric seconds or duration strings such as '150s', '2m', and '1h'."""
    text = str(value).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    result = float(text[:-1] if text[-1:] in units else text) * units.get(text[-1:], 1)
    if not math.isfinite(result):
        raise ValueError(f"invalid interval: {value!r}")
    return result


def _validate_remote_snapshot(remote_ids, previous_count, observations):
    """Require confirmation for empty/sharply smaller UUID snapshots before allowing pruning.

    None means the read failed or was incomplete; it must never imply all objects
    were deleted. The returned count/observations are carried into the next cycle.
    """
    if remote_ids is None:
        return None, previous_count, 0
    count = len(remote_ids)
    smaller = previous_count is not None and previous_count > 0 and count < previous_count * .75
    if (count == 0 and previous_count != 0) or smaller:
        observations += 1
        if observations < 2:
            return None, previous_count, observations
    return remote_ids, count, 0


def _retrieve_records(dataset):
    """Fetch a bounded full-record snapshot, refreshing authentication before the query."""
    rc.token = auth.get_token()
    records = rc.search(f"select * from {dataset} limit {_ROW_LIMIT}")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"invalid object snapshot from {dataset}")
    return records


def _retrieve_remote_ids(dataset):
    """Query UUID existence for reconciliation; return None on failure or possible truncation."""
    try:
        rc.token = auth.get_token()
        rows = rc.search(f"select {dataset}.objectId.uuid from {dataset} limit 200000")
        if not isinstance(rows, list) or len(rows) >= 200000:
            return None
        result = set()
        for row in rows:
            value = row.get("uuid", row.get(PRIMARY_KEY, get_path(row, PRIMARY_KEY)))
            if value is MISSING:
                return None
            if value is not None:
                result.add(str(value))
        return result
    except Exception:
        logging.exception("[%s]: UUID existence query failed; not pruning", dataset)
        return None


async def _cache_refresh(config, queues):
    """Periodically fan out object snapshots and UUID-existence snapshots to workers.

    Full records repair missed additions; UUID lists enable hard-delete pruning.
    Truncated full snapshots require a separate UUID query. Generation numbers
    allow workers to reject stale refresh/reconcile combinations.
    """
    refresh_interval = _interval_seconds(config.get("refresh_interval", 1800))
    reconcile_interval = _interval_seconds(config.get("reconcile_interval", 300))
    generation, next_refresh, next_reconcile = 0, 0., 0.
    previous_count, suspicious = config.get("_initial_object_count"), 0
    while True:
        try:
            now = time.monotonic()
            refresh_due = refresh_interval > 0 and now >= next_refresh
            reconcile_due = reconcile_interval > 0 and now >= next_reconcile
            if not refresh_due and not reconcile_due:
                await asyncio.sleep(1)
                continue
            datasets = [str(config["object_dataset"])]
            if config.get("superseded_object_dataset"):
                datasets.append(str(config["superseded_object_dataset"]))
            records, remote_ids, truncated = [], set(), False
            if refresh_due:
                for dataset in datasets:
                    batch = await asyncio.to_thread(_retrieve_records, dataset)
                    records.extend(batch)
                    truncated |= len(batch) >= _ROW_LIMIT
                next_refresh = now + refresh_interval
            if refresh_due and records and not truncated:
                remote_ids = {str(get_path(record, PRIMARY_KEY)) for record in records if get_path(record, PRIMARY_KEY) is not MISSING}
            else:
                for dataset in datasets:
                    ids = await asyncio.to_thread(_retrieve_remote_ids, dataset)
                    if ids is None:
                        remote_ids = None
                        break
                    remote_ids.update(ids)
            remote_ids, previous_count, suspicious = _validate_remote_snapshot(remote_ids, previous_count, suspicious)
            generation += 1
            for target in queues:
                for start in range(0, len(records), _REFRESH_CHUNK):
                    await asyncio.to_thread(target.put, {"_cache_msg": "records", "_cache_generation": generation, "records": records[start:start + _REFRESH_CHUNK]})
                if remote_ids is not None:
                    await asyncio.to_thread(target.put, {"_cache_msg": "reconcile", "_cache_generation": generation, "remote_uuids": list(remote_ids), "safety_seconds": 10})
            next_reconcile = now + reconcile_interval
        except Exception:
            logging.exception("[%s]: object cache refresh failed", config.get("origin_dataset"))
        await asyncio.sleep(1)


def _worker_count(config):
    """Resolve configured worker-count aliases, ensuring at least one worker."""
    return max(1, int(config.get("number_of_transformer_processes", config.get("num_transformer_workers", config.get("num_consumers", 1)))))


async def _serve(configs):
    """Load shared startup snapshots, spawn feed workers, and run the parent SSE listeners.

    One source queue allows competing consumers; a shard key creates one queue per
    worker for sticky routing. Cache updates always fan out to each private cache.
    Spawn avoids inheriting a live Polars thread pool. Finally, stop child processes.
    """
    context = get_context("spawn")
    processes, listeners, cache_queues_by_dataset, initial_by_dataset, refresh_configs = [], [], {}, {}, {}
    seen_origins = set()
    try:
        for config in configs:
            if config.get("disabled") or config.get("origin_dataset") in seen_origins:
                continue
            seen_origins.add(config.get("origin_dataset"))
            dataset = str(config["object_dataset"])
            if dataset not in initial_by_dataset:
                initial_by_dataset[dataset] = await asyncio.to_thread(_retrieve_records, dataset)
                if config.get("superseded_object_dataset"):
                    initial_by_dataset[dataset].extend(await asyncio.to_thread(_retrieve_records, str(config["superseded_object_dataset"])))
            count = _worker_count(config)
            capacity = max(1, int(config.get("source_queue_max_batches", config.get("source_queue_maxsize", _SOURCE_QUEUE_DEFAULT))))
            sticky = count > 1 and bool(config.get("transformer_shard_key"))
            source_queues = [context.Queue(maxsize=capacity) for _ in range(count if sticky else 1)]
            for worker_index in range(count):
                cache_queue = context.Queue(maxsize=_CACHE_QUEUE_SIZE)
                cache_queues_by_dataset.setdefault(dataset, []).append(cache_queue)
                if config.get("superseded_object_dataset"):
                    cache_queues_by_dataset.setdefault(str(config["superseded_object_dataset"]), []).append(cache_queue)
                child_config = dict(config, _script_sources=_script_sources.copy(), _log_level=logging.root.level)
                process = context.Process(target=_run_worker, args=(source_queues[worker_index] if sticky else source_queues[0],
                    cache_queue, initial_by_dataset[dataset], child_config), daemon=True)
                process.start()
                processes.append(process)
            listeners.append(_source_listener(config, source_queues))
            refresh_configs.setdefault(dataset, dict(config, _initial_object_count=len(initial_by_dataset[dataset])))
        # Share parent listeners by dataset instead of opening one cache SSE per worker.
        for dataset, queues in cache_queues_by_dataset.items():
            listeners.append(_cache_listener(dataset, queues))
        for dataset, config in refresh_configs.items():
            listeners.append(_cache_refresh(config, cache_queues_by_dataset[dataset]))
        for config in configs:
            if _is_config_true(config.get("deduplication")) and not _is_config_true(os.getenv("CRUCIBLE_DISABLE_DEDUPE", "false")):
                # Duplicate cleanup retains the existing legacy implementation.
                try:
                    from transformer_legacy import dedupe_objects
                except ImportError:
                    from .transformer_legacy import dedupe_objects
                process = context.Process(target=dedupe_objects, args=(config,), daemon=True)
                process.start()
                processes.append(process)
                break
        initial_by_dataset.clear()
        await asyncio.gather(*listeners)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)


def run(stream_manager_perspective, log_level=None):
    """Public launcher: initialize logging/controllers/config, then serve until stopped.

    The Streamlit page and command-line entry point both call this function.
    A parent-level failure restarts serving with the already loaded configuration.
    """
    global auth, rc, wc
    level = getattr(logging, (log_level or "INFO").upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    _, _, _, log_utils, _, _ = _import_cruciblelib_modules()
    log_utils.get_logger(log_type="transformer", log_level=level)
    auth, rc, wc = instantiate_api_controllers()
    configs = find_and_validate_configs(stream_manager_perspective, include_scripts=True, rc_instance=rc, caller_globals=globals())
    while True:
        try:
            asyncio.run(_serve(configs))
            return
        except (KeyboardInterrupt, SystemExit):
            return
        except Exception:
            logging.exception("Transformer failed; retrying in 5s")
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stream_manager_perspective")
    parser.add_argument("--use-fork", action="store_true")
    parser.add_argument("--log")
    args = parser.parse_args()
    if args.use_fork:
        logging.warning("--use-fork is ignored: Polars workers require spawn")
    run(args.stream_manager_perspective, args.log)