"""Customer hooks loaded from ``Entity_Stream_Manager_Functions``.

The perspective names up to two scripts. Each is a row holding the Python source
of a module:

- **unit conversions**: value hooks, ``(value) -> value``, applied to one field;
- **custom functions**: record hooks, ``(records, feed_row) -> records``,
  applied to a whole batch. ``feed_row`` is a fresh copy of the feed's merged
  configuration row (`hook_row`), so changes a hook makes to it do not persist.

The modules are registered in ``sys.modules`` as ``unit_conversions`` and
``custom_functions``, as the baseline did. Unit conversions load first, so a
custom-functions script may ``import unit_conversions``. Each load replaces both
registrations: a name whose script is not configured, or fails to load, is left
unregistered.

Only this v2 record contract is supported.

Contract for hook authors:

- A hook must be deterministic: a pure function of its input and the feed row,
  with no clocks, randomness or external calls. Rollout-overlap deduplication
  (DESIGN.md §5.5) relies on it.
- A record hook must return a ``list`` of ``dict`` records and must not keep
  references to them.

Trust boundary: anyone who can write the functions dataset can run code in the
pipeline process. That is inherent in the feature.
"""

import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from crucible_entity_manager.config.perspective import FeedConfig, ScriptNames, UnitConversion
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.records import clone_record, clone_value

FUNCTIONS_DATASET: Final = "Entity_Stream_Manager_Functions"
_MODULE_NAMES: Final = ("unit_conversions", "custom_functions")

type RecordHook = Callable[[list[JSONObject], Mapping[str, JSONValue]], list[JSONObject]]
"""A record hook. Its result is checked at runtime by `validate_record_batch`."""

type ValueHook = Callable[[JSONValue], JSONValue]


class HookError(ValueError):
    """A hook script or function is missing, fails to load, or misbehaves."""


@dataclass(frozen=True, slots=True)
class HookModules:
    """The compiled hook scripts named by the perspective, if any."""

    custom_functions: types.ModuleType | None
    unit_conversions: types.ModuleType | None


@dataclass(frozen=True, slots=True)
class FeedHooks:
    """One feed's hooks, resolved and in the order they run."""

    value_hooks: tuple[tuple[UnitConversion, ValueHook], ...]
    record_hooks: tuple[tuple[str, RecordHook], ...]


def load_hook_modules(rows: Sequence[JSONObject], scripts: ScriptNames) -> HookModules:
    """Compile the scripts named in `scripts` from the functions dataset rows.

    Raises:
        HookError: If a named script has no row, or its source fails to execute.
    """
    for module_name in _MODULE_NAMES:
        sys.modules.pop(module_name, None)
    bodies: dict[str, str] = {}
    for row in rows:
        name, body = row.get("script_name"), row.get("script_body")
        if isinstance(name, str) and isinstance(body, str):
            bodies[name] = body
    unit_conversions = _compile(bodies, scripts.unit_conversions, "unit_conversions")
    custom_functions = _compile(bodies, scripts.custom_functions, "custom_functions")
    return HookModules(custom_functions=custom_functions, unit_conversions=unit_conversions)


def resolve_feed_hooks(feed: FeedConfig, modules: HookModules) -> FeedHooks:
    """Look up every hook `feed` configures.

    Raises:
        HookError: If a configured function is missing or not callable.
    """
    value_hooks = tuple(
        (
            conversion,
            cast(
                "ValueHook",
                _function(
                    modules.unit_conversions, "unit conversions", conversion.function_name, feed
                ),
            ),
        )
        for conversion in feed.unit_conversions
    )
    record_hooks = tuple(
        (
            name,
            cast("RecordHook", _function(modules.custom_functions, "custom functions", name, feed)),
        )
        for name in feed.custom_functions
    )
    return FeedHooks(value_hooks, record_hooks)


def hook_row(feed: FeedConfig) -> JSONObject:
    """Return a fresh deep copy of `feed`'s merged row to pass to a record hook."""
    return {key: clone_value(value) for key, value in feed.row.items()}


def validate_record_batch(result: object, hook_name: str) -> list[JSONObject]:
    """Check a record hook's result and return copies the pipeline owns.

    Raises:
        HookError: If `result` is not a list of objects with string keys.
    """
    if not isinstance(result, list):
        msg = f"record hook {hook_name!r} must return a list, got {type(result).__name__}"
        raise HookError(msg)
    records: list[JSONObject] = []
    for index, record in enumerate(result):
        if not isinstance(record, dict) or not all(isinstance(key, str) for key in record):
            msg = f"record hook {hook_name!r} returned a non-object at index {index}"
            raise HookError(msg)
        records.append(clone_record(cast("JSONObject", record)))
    return records


def _compile(
    bodies: Mapping[str, str], script: str | None, module_name: str
) -> types.ModuleType | None:
    if script is None:
        return None
    if script not in bodies:
        msg = f"hook script {script!r} is not in {FUNCTIONS_DATASET}"
        raise HookError(msg)
    module = types.ModuleType(module_name)
    module.__file__ = f"<{FUNCTIONS_DATASET}:{script}>"
    sys.modules[module_name] = module
    try:
        code = compile(bodies[script], module.__file__, "exec")
        exec(code, module.__dict__)  # noqa: S102 - running customer hook scripts is the feature
    except Exception as error:
        del sys.modules[module_name]
        msg = f"hook script {script!r} failed to load: {error}"
        raise HookError(msg) from error
    return module


def _function(
    module: types.ModuleType | None, script_kind: str, name: str, feed: FeedConfig
) -> Callable[..., object]:
    function = getattr(module, name, None) if module is not None else None
    if not callable(function):
        msg = f"{feed.origin_dataset}: hook {name!r} is not a function in the {script_kind} script"
        raise HookError(msg)
    return function
