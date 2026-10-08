#!/usr/bin/env python
"""Record-native transformer for entity report events."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import signal
import time
import traceback
from collections.abc import Callable, Mapping
from multiprocessing import Process, Queue, set_start_method
from typing import Any

try:
    from entity_utils import (
        SSE_listener,
        _import_cruciblelib_modules,
        _load_module_from_source,
        ensure_api_controllers,
        find_and_validate_configs,
        terminate,
        write_batch_chunked,
    )
    from entity_transformer_records import (
        MISSING,
        JSONObject,
        add_report_ecef_kinematics,
        clone_record,
        compact_record,
        get_value,
        parse_records,
        remove_value,
        set_value,
    )
except ImportError:
    from .entity_utils import (
        SSE_listener,
        _import_cruciblelib_modules,
        _load_module_from_source,
        ensure_api_controllers,
        find_and_validate_configs,
        terminate,
        write_batch_chunked,
    )
    from .entity_transformer_records import (
        MISSING,
        JSONObject,
        add_report_ecef_kinematics,
        clone_record,
        compact_record,
        get_value,
        parse_records,
        remove_value,
        set_value,
    )

try:
    from cruciblelib.decorators import async_retry as _async_retry
    async_retry: Any = _async_retry
except Exception:
    def _no_retry(function: Callable[..., Any]) -> Callable[..., Any]:
        return function
    async_retry = _no_retry


rc: Any = None
wc: Any = None
auth: Any = None


def _source_queue_max_batches(dataset_config: Mapping[str, Any]) -> int:
    """Return the number of whole SSE batches allowed to wait in memory.

    ``multiprocessing.Queue.maxsize`` counts queue items, not records or bytes.
    One item here is one complete SSE payload and may contain thousands of
    records, so the default is intentionally small. The old names remain as
    fallbacks for deployed configurations.
    """
    configured = dataset_config.get("source_queue_max_batches")
    if configured is None:
        configured = os.getenv("CRUCIBLE_ENTITY_SOURCE_QUEUE_MAX_BATCHES")
    if configured is None:
        configured = dataset_config.get("source_queue_maxsize")
    if configured is None:
        configured = os.getenv("CRUCIBLE_SOURCE_QUEUE_MAXSIZE", "4")
    return max(1, int(configured))


def _validate_record_batch(value: object) -> list[JSONObject]:
    """Validate and copy the value returned by a native custom function.

    Native custom functions receive ``list[dict]`` and must return
    ``list[dict]``. Each dictionary is one transformed entity record.
    """
    # Reject a single dict and every other container type. Requiring a list
    # gives every native custom function one unambiguous return contract.
    if not isinstance(value, list):
        raise TypeError("native custom function must return list[dict]")

    # Validate each element separately so the error identifies the bad record.
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise TypeError(
                f"native custom function record {index} must be a dict, "
                f"got {type(record).__name__}"
            )

    # clone_record performs a recursive deep copy. The transformer can safely
    # add mappings and ECEF fields without mutating objects owned by the custom
    # function or sharing nested dictionaries between pipeline stages.
    return [clone_record(record) for record in value]


def _validate_custom_function_api_version(dataset_config: Mapping[str, Any]) -> None:
    """Require custom functions to use the record-native v2 interface.

    API v2 functions receive and return ``list[dict]`` batches. API v1 used
    pandas and is intentionally unsupported by this transformer.
    """
    # ``hook_api_version`` is a legacy name retained for deployed configs.
    for key in ("custom_function_api_version", "hook_api_version"):
        configured = dataset_config.get(key)
        if configured is None:
            continue
        normalized = str(configured).strip().lower()
        # These values are aliases for the same v2 list-of-records contract.
        if normalized not in {"2", "native", "records"}:
            raise ValueError(
                f"{key}={configured!r} is unsupported; custom functions require API v2"
            )


async def launch_processes(config_list: list[dict[str, Any]]) -> None:
    _, _, utils, _ = _import_cruciblelib_modules()
    listeners = []
    for dataset_config in config_list:
        source_queue_max_batches = _source_queue_max_batches(dataset_config)
        message_queue = Queue(maxsize=source_queue_max_batches)
        logging.info(
            "[%s] source queue capacity=%d SSE batches (not records)",
            dataset_config.get("origin_dataset", "unknown"),
            source_queue_max_batches,
        )
        process_count = int(dataset_config.get("number_of_transformer_processes", 1))
        for _ in range(process_count):
            Process(
                target=utils.async_function_launcher,
                args=(sse_msg_processor, message_queue, dataset_config),
                daemon=True,
            ).start()
        listeners.append(SSE_listener(dataset_config["query"], message_queue, auth))
    results = await asyncio.gather(*listeners, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logging.error("Transformer listener failed: %s", result)


@async_retry
async def sse_msg_processor(message_queue: Queue, dataset_config: dict[str, Any]) -> None:
    global auth, rc, wc
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)
    sources = dataset_config.get("_script_sources", {})
    if "custom_functions" in sources:
        globals()["custom_functions"] = _load_module_from_source(
            "custom_functions", sources["custom_functions"]
        )
    if "unit_conversions" in sources:
        globals()["unit_conversions"] = _load_module_from_source(
            "unit_conversions", sources["unit_conversions"]
        )

    while True:
        try:
            event = message_queue.get(timeout=1)
        except Exception:
            time.sleep(0.5)
            continue
        try:
            records = preprocess(event, dataset_config)
            mappings = dataset_config["origin_to_destination_mapping"]
            records = apply_mappings(records, mappings)

            dataset_name = get_dataset_name(dataset_config.get("query", ""))
            output: list[JSONObject] = []
            for record in records:
                source_id = get_value(record, "crucibleHeader.uuid")
                set_value(record, "source.datasetName", dataset_name)
                if source_id is not MISSING:
                    set_value(record, "source.uuid", source_id)
                compacted = project_report_record(record, mappings)
                if compacted:
                    output.append(compacted)
            output = add_ECEF_kinematics(output)

            def refresh_token() -> None:
                wc.token = auth.get_token()

            report_event_dataset = dataset_config.get("report_event_dataset")
            if not isinstance(report_event_dataset, str):
                raise ValueError("report_event_dataset must be configured")
            await write_batch_chunked(
                output,
                report_event_dataset,
                wc.write_record_batch_by_name,
                int(dataset_config["batch_write_chunk_size"]),
                label=f"[{dataset_config.get('origin_dataset')}] ",
                token_refresher=refresh_token,
                max_concurrent_writes=dataset_config.get("batch_write_max_concurrent"),
            )
        except Exception as error:
            logging.warning("Transformer batch failed: %s", error)
            logging.warning(traceback.format_exc())


def preprocess(event: object, dataset_config: dict[str, Any]) -> list[JSONObject]:
    _validate_custom_function_api_version(dataset_config)
    records = parse_records(event)
    for mapping in dataset_config.get("unit_conversions", []):
        function_name = mapping.get("unit_conversion", "")
        path = mapping.get("origin_column")
        if not path or "PLACEHOLDER" in function_name.upper():
            continue
        try:
            conversion = getattr(globals()["unit_conversions"], function_name)
            for record in records:
                value = get_value(record, path)
                if value is not MISSING:
                    set_value(record, path, conversion(value))
        except Exception as error:
            logging.warning("Unit conversion %s failed for %s: %s", function_name, path, error)

    for configured in dataset_config.get("custom_functions", []):
        function_name = configured.get("function_name")
        try:
            transform = getattr(globals()["custom_functions"], function_name)
            records = _validate_record_batch(transform(records, dataset_config))
        except Exception as error:
            logging.warning("Record transform %s failed: %s", function_name, error)
            logging.warning(traceback.format_exc())
    return records


def rename_columns(
    records: list[JSONObject],
    origin_to_destination_mapping: list[dict[str, Any]],
) -> list[JSONObject]:
    """Map source paths to destination paths, including one-to-many mappings."""
    destinations = {
        mapping.get("destination_column")
        for mapping in origin_to_destination_mapping
        if mapping.get("destination_column")
    }
    result = [clone_record(record) for record in records]
    for record in result:
        # Dotted config names are nested paths; records remain nested dictionaries.
        source_values = {
            source: get_value(record, source)
            for mapping in origin_to_destination_mapping
            if (source := mapping.get("origin_column"))
        }
        for mapping in origin_to_destination_mapping:
            source = mapping.get("origin_column")
            destination = mapping.get("destination_column")
            value = source_values.get(source, MISSING)
            if source and destination and value is not MISSING:
                set_value(record, destination, value)
        for source in source_values:
            if source not in destinations:
                remove_value(record, source)
    return result


def apply_mappings(
    records: list[JSONObject],
    mappings: list[dict[str, Any]],
) -> list[JSONObject]:
    mapped = rename_columns(
        records, [mapping for mapping in mappings if "origin_column" in mapping]
    )
    for mapping in mappings:
        if "literal" not in mapping:
            continue
        for record in mapped:
            set_value(record, mapping["destination_column"], mapping["literal"])
    return mapped


def project_report_record(
    record: JSONObject,
    mappings: list[dict[str, Any]],
) -> JSONObject:
    """Project one report from configured destinations and source metadata."""
    projected: JSONObject = {}
    for mapping in mappings:
        destination = mapping.get("destination_column")
        if not destination:
            continue
        value = get_value(record, destination)
        if value is not MISSING:
            set_value(projected, destination, clone_record({"value": value})["value"])
    source = record.get("source", MISSING)
    if source is not MISSING:
        projected["source"] = clone_record({"value": source})["value"]
    return compact_record(projected)


def add_ECEF_kinematics(records: list[JSONObject]) -> list[JSONObject]:
    return add_report_ecef_kinematics(records)


def get_dataset_name(query: str) -> str:
    match = re.search(r'(?<=from\s)(?:\'|")?([^\s\'"]+)(?:\'|")?', query, flags=re.IGNORECASE)
    if not match:
        return "dataset_name_unknown"
    return match.group(1)[:40]


def run(stream_manager_perspective: str) -> None:
    global auth, rc, wc
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    auth, rc, wc = ensure_api_controllers(auth, rc, wc)
    result = find_and_validate_configs(
        stream_manager_perspective,
        include_scripts=True,
        rc_instance=rc,
        caller_globals=globals(),
    )
    asyncio.run(launch_processes(result["datafeed_configs"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Stream_Manager_Perspective", default="Live")
    parser.add_argument("--log", default="INFO")
    parser.add_argument("--use-fork", action="store_true")
    args = parser.parse_args()
    if args.use_fork:
        set_start_method("forkserver", force=True)
    level = getattr(logging, args.log.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {args.log}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s: %(levelname)s %(name)s %(module)s Func: %(funcName)s:%(lineno)d-%(message)s",
    )
    run(args.Stream_Manager_Perspective)