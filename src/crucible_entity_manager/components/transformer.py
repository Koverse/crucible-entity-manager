"""The transformer: origin records in, report events out (DESIGN.md §5.2, §5.9).

For each feed, a batch of origin records goes through these steps:

1. Unit conversions (value hooks) on configured fields.
2. Custom functions (record hooks) on the whole batch.
3. Field mappings: copy origin paths to destinations, drop mapped origins that
   are not destinations themselves, then set literals.
4. Stamp ``source.datasetName`` and ``source.uuid`` (the origin record's
   ``crucibleHeader.uuid``, read before mapping), keep only the mapped
   destinations and ``source``, and compact.
5. Add ECEF kinematics.

The reports are then POSTed to the perspective's report event dataset. The
transformer keeps no state between batches.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, override

from crucible_entity_manager.components.base import Component, RecordSource, Services, Subscription
from crucible_entity_manager.config.perspective import (
    ConfigError,
    FeedConfig,
    LiteralMapping,
    OriginMapping,
    UnitConversion,
)
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.geodesy import set_report_ecef_kinematics
from crucible_entity_manager.core.partition import PartitionSpec
from crucible_entity_manager.core.records import (
    MISSING,
    Missing,
    clone_record,
    clone_value,
    compact_record,
    get_path,
    remove_path,
    set_path,
)
from crucible_entity_manager.crucible.sql import identifier
from crucible_entity_manager.crucible.writer import BatchWriter, WriteClass
from crucible_entity_manager.hooks.loader import (
    FUNCTIONS_DATASET,
    FeedHooks,
    HookModules,
    hook_row,
    load_hook_modules,
    resolve_feed_hooks,
    validate_record_batch,
)

logger = logging.getLogger(__name__)

SOURCE_ID_PATH: Final = "crucibleHeader.uuid"
UNKNOWN_DATASET_NAME: Final = "dataset_name_unknown"
_DATASET_NAME_LENGTH: Final = 40
_FROM_TABLE: Final = re.compile(r"(?<=from\s)['\"]?([^\s'\"]+)['\"]?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FeedPipeline:
    """One feed's configuration and hooks, ready to transform batches."""

    feed: FeedConfig
    hooks: FeedHooks
    dataset_name: str
    """Written to ``source.datasetName``: the table the feed's query reads."""


def dataset_name(query: str) -> str:
    """Return the first table named after FROM in `query`, truncated as at ``1b534df``."""
    match = _FROM_TABLE.search(query)
    if match is None:
        return UNKNOWN_DATASET_NAME
    return match.expand(r"\1")[:_DATASET_NAME_LENGTH]


def transform(records: Sequence[JSONObject], pipeline: FeedPipeline) -> list[JSONObject]:
    """Turn origin records into report records. `records` are not modified.

    A record whose unit conversion fails, or that a hook left holding a value
    JSON cannot represent, is dropped and counted in a warning. A custom
    function that fails is skipped for this batch.
    """
    label = pipeline.feed.origin_dataset
    batch = _converted([clone_record(record) for record in records], pipeline, label)
    batch = _hooked(batch, pipeline, label)
    reports: list[JSONObject] = []
    unrepresentable = 0
    for record in batch:
        source_id = get_path(record, SOURCE_ID_PATH)
        mapped = _mapped(record, pipeline.feed)
        _stamp_source(mapped, pipeline.dataset_name, source_id)
        try:
            report = _projected(mapped, pipeline.feed)
        except TypeError:
            unrepresentable += 1
            continue
        reports.append(report)
    if unrepresentable:
        logger.warning(
            "[%s] Dropped %d record(s) holding values JSON cannot represent", label, unrepresentable
        )
    set_report_ecef_kinematics(reports)
    return reports


def _converted(records: list[JSONObject], pipeline: FeedPipeline, label: str) -> list[JSONObject]:
    if not pipeline.hooks.value_hooks:
        return records
    kept: list[JSONObject] = []
    for record in records:
        failure = _convert(record, pipeline.hooks)
        if failure is None:
            kept.append(record)
        else:
            conversion, error = failure
            logger.warning(
                "[%s] Dropped a record: unit conversion %s failed on %s: %s",
                label,
                conversion.function_name,
                conversion.origin,
                error,
            )
    return kept


def _convert(record: JSONObject, hooks: FeedHooks) -> tuple[UnitConversion, Exception] | None:
    """Apply every unit conversion to `record`; report the first that fails.

    A missing or null value is left as it is.
    """
    for conversion, convert in hooks.value_hooks:
        value = get_path(record, conversion.origin)
        if value is MISSING or value is None:
            continue
        try:
            set_path(record, conversion.origin, convert(value))
        except Exception as error:
            return conversion, error
    return None


def _hooked(records: list[JSONObject], pipeline: FeedPipeline, label: str) -> list[JSONObject]:
    for name, hook in pipeline.hooks.record_hooks:
        try:
            records = validate_record_batch(hook(records, hook_row(pipeline.feed)), name)
        except Exception:
            logger.exception("[%s] Custom function %s failed; skipped for this batch", label, name)
    return records


def _mapped(record: JSONObject, feed: FeedConfig) -> JSONObject:
    origins = [mapping for mapping in feed.mappings if isinstance(mapping, OriginMapping)]
    values = {mapping.origin: get_path(record, mapping.origin) for mapping in origins}
    for mapping in origins:
        value = values[mapping.origin]
        if value is not MISSING:
            set_path(record, mapping.destination, clone_value(value))
    destinations = {mapping.destination for mapping in origins}
    for origin in values:
        if origin not in destinations:
            remove_path(record, origin)
    for mapping in feed.mappings:
        if isinstance(mapping, LiteralMapping):
            set_path(record, mapping.destination, clone_value(mapping.value))
    return record


def _stamp_source(record: JSONObject, name: str, source_id: JSONValue | Missing) -> None:
    """Set ``source.datasetName``, and ``source.uuid`` from the origin record's UUID.

    The UUID is read before mapping, so it survives a mapping that moves
    ``crucibleHeader.uuid`` elsewhere; the tracker's repeat check (D4) needs it.
    """
    set_path(record, "source.datasetName", name)
    if source_id is not MISSING:
        set_path(record, "source.uuid", source_id)


def _projected(record: JSONObject, feed: FeedConfig) -> JSONObject:
    """Keep the mapped destinations and ``source``, then compact.

    `record` has been stamped, so ``source`` holds at least ``datasetName`` and
    the result is never empty.

    Raises:
        TypeError: If a kept value is not representable as JSON.
    """
    projected: JSONObject = {}
    for mapping in feed.mappings:
        value = get_path(record, mapping.destination)
        if value is not MISSING:
            set_path(projected, mapping.destination, clone_value(value))
    projected["source"] = clone_value(record["source"])
    return compact_record(projected)


@dataclass(frozen=True, slots=True)
class RecordOwner:
    """Owns an origin record when its ``crucibleHeader.uuid`` shards to this partition.

    A record without one belongs to partition 0, so exactly one pod handles it.
    """

    partition: PartitionSpec

    def __call__(self, record: JSONObject) -> bool:
        """Whether this partition handles `record`."""
        source_id = get_path(record, SOURCE_ID_PATH)
        if isinstance(source_id, str) and source_id:
            return self.partition.owns(source_id)
        return self.partition.index == 0


class Transformer(Component):
    """Transforms every selected feed's origin records into report events."""

    name = "transformer"

    def __init__(
        self,
        pipelines: Sequence[FeedPipeline],
        sources: Mapping[str, RecordSource],
        writers: Mapping[str, BatchWriter],
        partition: PartitionSpec,
    ) -> None:
        self._pipelines = {pipeline.feed.origin_dataset: pipeline for pipeline in pipelines}
        self._sources = sources
        self._writers = writers
        self._owns = RecordOwner(partition)
        self.queue_max_records = sum(
            pipeline.feed.source_queue_max_records for pipeline in pipelines
        )

    @override
    def subscriptions(self) -> Sequence[Subscription]:
        return [Subscription(name, self._sources[name], self._owns) for name in self._pipelines]

    @override
    async def prepare(self) -> None:
        return

    @override
    async def handle(self, subscription: str, records: list[JSONObject]) -> None:
        pipeline = self._pipelines[subscription]
        reports = transform(records, pipeline)
        if not reports:
            return
        outcome = await self._writers[subscription].post(
            pipeline.feed.datasets.report_events,
            reports,
            write_class=WriteClass.AUTHORITATIVE,
            label=f"[{subscription}] ",
        )
        logger.debug(
            "[%s] Wrote %d of %d report(s)",
            subscription,
            len(reports) - len(outcome.failed),
            len(reports),
        )

    @override
    async def tick(self) -> None:
        return

    @override
    async def close(self) -> None:
        return


async def build(services: Services) -> Component:
    """Build the transformer for the feeds this process runs.

    Raises:
        ConfigError: If `--feed` names no enabled feed, or a feed has no query.
        HookError: If a hook script or function is missing or fails to load.
        CrucibleError: If the hook scripts can't be read.
    """
    feeds = select_feeds(services.perspective.feeds, services.settings.feed)
    modules = await _load_hooks(services)
    pipelines: list[FeedPipeline] = []
    sources: dict[str, RecordSource] = {}
    writers: dict[str, BatchWriter] = {}
    for feed in feeds:
        if feed.query is None:
            msg = f"{feed.origin_dataset}: the transformer needs a query"
            raise ConfigError(msg)
        hooks = resolve_feed_hooks(feed, modules)
        pipelines.append(FeedPipeline(feed, hooks, dataset_name(feed.query)))
        sources[feed.origin_dataset] = services.source(feed.origin_dataset, feed.query)
        writers[feed.origin_dataset] = services.writer(feed.writes)
    return Transformer(pipelines, sources, writers, services.settings.partition)


def select_feeds(feeds: Sequence[FeedConfig], only: str | None) -> list[FeedConfig]:
    """Return every feed, or only the one whose origin dataset is `only`.

    Raises:
        ConfigError: If there are no feeds, or `only` names none of them.
    """
    selected = [feed for feed in feeds if only is None or feed.origin_dataset == only]
    if not selected:
        msg = (
            f"no enabled feed is named {only!r}" if only else "the perspective has no enabled feeds"
        )
        raise ConfigError(msg)
    return selected


async def _load_hooks(services: Services) -> HookModules:
    scripts = services.perspective.scripts
    if scripts.custom_functions is None and scripts.unit_conversions is None:
        return HookModules(custom_functions=None, unit_conversions=None)
    rows = await services.crucible.search(f"SELECT * FROM {identifier(FUNCTIONS_DATASET)}")  # noqa: S608
    return load_hook_modules(rows, scripts)
