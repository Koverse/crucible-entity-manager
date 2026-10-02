"""Generate parity fixtures by running the baseline entity pipeline.

The baseline is crucible-streamlit at `BASELINE_SHA`. Its files are read with
``git show`` from a local clone, so the fixtures don't depend on whatever that
clone has checked out. Each generator feeds seeded random inputs to a baseline
function and records the inputs and outputs as JSON in ``tests/fixtures/parity``.

The baseline needs filterpy (a dev dependency) and cruciblelib on the path.

Usage::

    python tests/parity/generate.py --streamlit-repo ~/Documents/projects/crucible-streamlit
"""

import argparse
import asyncio
import copy
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import numpy as np

from crucible_entity_manager.core.geodesy import geodetic_to_ecef

BASELINE_SHA: Final = "1b534df"
_CORRELATORS: Final = "objectApps/correlators"
_BASELINE_FILES: Final = (
    "entity_manager/entity_utils.py",
    "entity_manager/entity_transformer_records.py",
    "entity_manager/entity_transformer.py",
    "entity_manager/entity_tracker.py",
    "entity_manager/entity_fusion_filter.py",
    "entity_manager/entity_track_fuser.py",
    "entity_manager/entity_duplicate_identifier.py",
    "track_smoother.py",
)
FIXTURE_DIR: Final = Path(__file__).resolve().parent.parent / "fixtures" / "parity"
_EPOCH: Final = datetime(2026, 9, 30, tzinfo=UTC)

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


def main() -> None:
    """Regenerate every parity fixture file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--streamlit-repo", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as workdir:
        _extract_baseline(args.streamlit_repo, Path(workdir))
        sys.path.insert(0, workdir)
        baseline = {name: importlib.import_module(name) for name in _module_names()}
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        for name, generate in GENERATORS.items():
            cases = generate(baseline, np.random.default_rng(_seed(name)))
            _write(name, cases)


def _extract_baseline(repo: Path, target: Path) -> None:
    for relative in _BASELINE_FILES:
        source = subprocess.run(  # noqa: S603 - fixed git arguments, local repository
            ["git", "-C", str(repo), "show", f"{BASELINE_SHA}:{_CORRELATORS}/{relative}"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (target / Path(relative).name).write_text(source)


def _module_names() -> list[str]:
    return [Path(relative).stem for relative in _BASELINE_FILES]


def _seed(name: str) -> int:
    return sum(name.encode())


def _write(name: str, cases: dict[str, Json]) -> None:
    payload = {"baseline_sha": BASELINE_SHA, "generator": "tests/parity/generate.py", **cases}
    path = FIXTURE_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _spd(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    """Return a random symmetric positive-definite matrix."""
    factor = rng.normal(size=(size, size))
    return scale * (factor @ factor.T) + np.eye(size) * scale * 0.1


def _array(values: np.ndarray) -> Json:
    return cast("Json", np.asarray(values, dtype=np.float64).tolist())


# --- kalman ----------------------------------------------------------------------


def _kalman(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    tracker = baseline["entity_tracker"]
    fusion = baseline["entity_fusion_filter"]
    manager = tracker.NumpyKalmanFilterManager({})
    principal = fusion.EntityPrincipalTrackFilter({})

    predict_cases: list[Json] = []
    for _ in range(40):
        mean = rng.normal(scale=1e5, size=6)
        cov = _spd(rng, 6, 50.0)
        dt = float(rng.choice([-5.0, 0.0, 0.5, 12.0, 600.0]))
        q = float(rng.choice([0.001, 0.5, 3.0, 5.0]))
        manager._q_by_track["t"] = q
        state = tracker.GaussianState(mean, cov, timestamp=_EPOCH)
        prediction = manager._predict("t", state, _EPOCH + timedelta(seconds=dt))
        predict_cases.append(
            {
                "mean": _array(mean),
                "covariance": _array(cov),
                "dt": dt,
                "q": q,
                "expected_mean": _array(prediction.state_vector),
                "expected_covariance": _array(prediction.covar),
            }
        )

    update_cases: list[Json] = []
    for index in range(40):
        observed = (0, 1, 2, 3, 4, 5) if index % 2 else (0, 2, 4)
        mean = rng.normal(scale=1e5, size=6)
        cov = _spd(rng, 6, 100.0)
        measurement = rng.normal(scale=1e5, size=len(observed))
        noise = _spd(rng, len(observed), 30.0)
        prediction = tracker.GaussianState(mean, cov, timestamp=_EPOCH)
        detection = tracker.Detection(
            measurement,
            timestamp=_EPOCH,
            measurement_model=tracker.LinearGaussian(6, observed, noise),
        )
        posterior = manager._standard_update("t", prediction, detection)
        update_cases.append(
            {
                "mean": _array(mean),
                "covariance": _array(cov),
                "measurement": _array(measurement),
                "noise": _array(noise),
                "observed": list(observed),
                "expected_mean": _array(posterior.state_vector),
                "expected_covariance": _array(posterior.covar),
            }
        )
    for _ in range(20):
        mean = rng.normal(scale=1e5, size=6)
        cov = _spd(rng, 6, 100.0)
        measurement = rng.normal(scale=1e5, size=6)
        noise = _spd(rng, 6, 30.0)
        fused_mean, fused_cov = principal._standard_update(mean, cov, measurement, noise)
        update_cases.append(
            {
                "mean": _array(mean),
                "covariance": _array(cov),
                "measurement": _array(measurement),
                "noise": _array(noise),
                "observed": [0, 1, 2, 3, 4, 5],
                "expected_mean": _array(fused_mean),
                "expected_covariance": _array(fused_cov),
            }
        )

    ci_cases: list[Json] = []
    for index in range(60):
        mean_a, mean_b = rng.normal(scale=1e5, size=6), rng.normal(scale=1e5, size=6)
        cov_a, cov_b = _spd(rng, 6, 80.0), _spd(rng, 6, float(rng.choice([1.0, 80.0, 5000.0])))
        if index % 3 == 0:
            objective, omega = "FULL_TRACE", None
            fused = manager._covariance_intersection(mean_a, cov_a, mean_b, cov_b)
        else:
            objective = "POSITION_TRACE"
            omega = None if index % 3 == 1 else float(rng.uniform(0.05, 0.95))
            principal.set_fusion_method("ci", omega=omega)
            fused = principal._covariance_intersection(mean_a, cov_a, mean_b, cov_b)
        ci_cases.append(
            {
                "mean_a": _array(mean_a),
                "cov_a": _array(cov_a),
                "mean_b": _array(mean_b),
                "cov_b": _array(cov_b),
                "objective": objective,
                "omega": omega,
                "expected_mean": _array(fused[0]),
                "expected_covariance": _array(fused[1]),
            }
        )
    return {"predict": predict_cases, "update": update_cases, "covariance_intersection": ci_cases}


# --- smoother --------------------------------------------------------------------


def _smoother(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    smoother = baseline["track_smoother"]
    cases: list[Json] = []
    for index in range(30):
        count = 1 if index == 0 else int(rng.integers(2, 40))
        steps = rng.uniform(1.0, 30.0, size=count)
        if count > 3 and index % 4 == 0:
            steps[count // 2] = 1_200.0
        # Datetimes hold microseconds, so round first: the baseline sees exactly these times.
        times = np.round(1.79e9 + np.cumsum(steps), 6)
        velocity = rng.normal(scale=60.0, size=3)
        start = rng.normal(scale=3e6, size=3)
        positions = (
            start + (times - times[0])[:, None] * velocity + rng.normal(scale=40.0, size=(count, 3))
        )
        with_velocity = index % 3 != 0
        velocities = (
            np.tile(velocity, (count, 1)) + rng.normal(scale=2.0, size=(count, 3))
            if with_velocity
            else None
        )
        covariances = (
            np.stack([_spd(rng, 3, 900.0) for _ in range(count)]) if index % 2 == 0 else None
        )
        shuffled = rng.permutation(count)
        stamps = [datetime.fromtimestamp(float(times[i]), UTC) for i in shuffled]
        result = smoother.smooth_track(
            stamps,
            [tuple(positions[i]) for i in shuffled],
            [tuple(velocities[i]) for i in shuffled] if velocities is not None else None,
            return_covariances=True,
            measurement_covariances=covariances[shuffled] if covariances is not None else None,
        )
        cases.append(
            {
                "times": _array(times[shuffled]),
                "positions": _array(positions[shuffled]),
                "velocities": _array(velocities[shuffled]) if velocities is not None else None,
                "position_covariances": (
                    _array(covariances[shuffled]) if covariances is not None else None
                ),
                "expected_times": _array(times),
                "expected_positions": _array(result[0]),
                "expected_velocities": _array(result[1]),
                "expected_position_covariances": _array(result[2]),
            }
        )
    return {"tracks": cases}


# --- records, identity, geodesy --------------------------------------------------


_IDENTITY_VALUES: Final[list[Json]] = [
    None, 0, 7, 7.0, -3.5, True, "7", "7.0", "007", "1.50", "abc", "", "1e3", [1, 2.0, "3.0"],
]  # fmt: skip


def _core(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    utils = baseline["entity_utils"]
    records = baseline["entity_transformer_records"]
    tracker = baseline["entity_tracker"]
    fusion = baseline["entity_fusion_filter"]

    def pick() -> Json:
        return _IDENTITY_VALUES[int(rng.integers(len(_IDENTITY_VALUES)))]

    identity_cases: list[Json] = []
    path_choices: list[list[Json] | None] = [
        None, ["identity.*"], ["identity.k0", "mmsi"], ["identity.*", "mmsi", "nope.x"],
    ]  # fmt: skip
    for _ in range(300):
        record: dict[str, Json] = {
            "identity": {f"k{i}": pick() for i in range(int(rng.integers(0, 5)))},
            "mmsi": pick(),
        }
        paths = path_choices[int(rng.integers(len(path_choices)))]
        custom_id = utils.identity_custom_id(record, paths)
        principal = fusion.EntityPrincipalTrackFilter._deterministic_principal_trackid(custom_id)
        identity_cases.append(
            {
                "record": record,
                "paths": paths,
                "expected_custom_id": custom_id,
                "expected_component_track_id": tracker._track_id_from_custom_id("FEED", custom_id),
                "expected_principal_track_id": principal,
            }
        )

    shard_cases: list[Json] = []
    for _ in range(300):
        key = f"{rng.random():.17f}"
        count = int(rng.integers(1, 10))
        shard_cases.append({"key": key, "count": count, "expected": utils.stable_shard(key, count)})

    reports = [_random_report(rng) for _ in range(300)]
    enriched = records.add_report_ecef_kinematics(copy.deepcopy(reports))

    tracks = [_random_track(rng) for _ in range(300)]
    with_geodetic = records.add_track_wgs84_kinematics(copy.deepcopy(tracks))
    for track in with_geodetic:
        geodetic = track.get("geodetic")
        if isinstance(geodetic, dict):
            # The baseline writes degrees (a regression, DESIGN.md §8); the port writes radians.
            geodetic["latitude"] = math.radians(geodetic["latitude"])
            geodetic["longitude"] = math.radians(geodetic["longitude"])

    return {
        "identity": identity_cases,
        "stable_shard": shard_cases,
        "report_ecef": [
            {"input": r, "expected": e} for r, e in zip(reports, enriched, strict=True)
        ],
        "track_geodetic": [
            {"input": t, "expected": e} for t, e in zip(tracks, with_geodetic, strict=True)
        ],
    }


def _random_report(rng: np.random.Generator) -> dict[str, Json]:
    report: dict[str, Json] = {}
    if rng.random() < 0.9:
        report["estimatedKinematics"] = {"kinematicsTimestamp": "2026-09-30T00:00:00.000Z"}
    if rng.random() < 0.95:
        geodetic: dict[str, Json] = {
            "latitude": float(rng.uniform(-1.5, 1.5)),
            "longitude": float(rng.uniform(-3.1, 3.1)),
        }
        if rng.random() < 0.7:
            geodetic["altitude"] = float(rng.uniform(-100.0, 12_000.0))
        report["geodetic"] = geodetic
    if rng.random() < 0.7:
        report["speed"] = float(rng.uniform(0.0, 300.0))
    if rng.random() < 0.7:
        report["heading"] = float(rng.uniform(0.0, 6.28))
    if rng.random() < 0.7:
        ellipse: dict[str, Json] = {"semiMajorAxisLength": float(rng.uniform(1.0, 5_000.0))}
        if rng.random() < 0.7:
            ellipse["semiMinorAxisLength"] = float(rng.uniform(1.0, 1_000.0))
        if rng.random() < 0.7:
            ellipse["orientation"] = float(rng.uniform(0.0, 6.28))
        report["uncertainty"] = {"uncertaintyEllipse": ellipse}
    return report


def _random_track(rng: np.random.Generator) -> dict[str, Json]:
    if rng.random() < 0.1:
        return {"ecefPosition": {"x": 1.0, "y": 2.0}}
    # Realistic altitudes: the baseline's altitude formula degrades far from the surface.
    x, y, z = geodetic_to_ecef(
        np.array([rng.uniform(-1.55, 1.55)]),
        np.array([rng.uniform(-math.pi, math.pi)]),
        np.array([rng.uniform(-1_000.0, 50_000.0)]),
    )
    return {"ecefPosition": {"x": float(x[0]), "y": float(y[0]), "z": float(z[0])}}


# --- management events ------------------------------------------------------------


def _management(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    utils = baseline["entity_utils"]
    tracks = [f"T{index}" for index in range(8)]

    def random_events(day: int) -> list[Json]:
        events: list[Json] = []
        for _ in range(int(rng.integers(0, 12))):
            action = str(rng.choice(["SUPERSEDE", "SUPERSEDE", "DELETE", "RESTORE"]))
            event: dict[str, Json] = {
                "trackId": str(rng.choice(tracks)),
                "action": action,
                # Few distinct minutes, so ties between events are common.
                "crucibleHeader": {
                    "updatedDate": f"2026-09-{day:02d}T12:{int(rng.integers(0, 6)):02d}:00Z"
                },
            }
            if action == "SUPERSEDE":
                event["supersededBy"] = None if rng.random() < 0.1 else str(rng.choice(tracks))
            events.append(event)
        return events

    cases: list[Json] = []
    for _ in range(200):
        # As in the pipeline, an initial map is the result of earlier events.
        initial = utils.build_supersede_map(random_events(day=29)) if rng.random() < 0.5 else None
        events = random_events(day=30)
        expected = utils.build_supersede_map(copy.deepcopy(events), copy.deepcopy(initial))
        cases.append({"events": events, "initial": initial, "expected": expected})
    return {"build_supersede_map": cases}


# --- transformer -------------------------------------------------------------------

TRANSFORMER_SCRIPTS: Final = {
    "parity_units": (
        "import math\n\n\n"
        "def to_radians(value):\n    return math.radians(float(value))\n\n\n"
        "def feet_to_meters(value):\n    return float(value) * 0.3048\n"
    ),
    "parity_functions": (
        "def tag_domain(records, row):\n"
        "    for record in records:\n"
        "        altitude = record.get('alt_ft')\n"
        "        record['domain_hint'] = 'AIR' if altitude else row.get('default_domain')\n"
        "    return records\n\n\n"
        "def drop_flagged(records, row):\n"
        "    return [record for record in records if not record.get('flagged')]\n"
    ),
}
TRANSFORMER_PERSPECTIVE_ROW: Final[dict[str, Json]] = {
    "perspective": "Parity",
    "origin_dataset": "perspective_config_Parity",
    "entity_management_event_dataset": "EntityManagementEvents",
    "report_event_dataset": "ReportEvents",
    "principal_track_event_dataset": "PrincipalTrackEvents",
    "principal_track_head_dataset": "PrincipalTrackHeads",
    "custom_functions_script_name": "parity_functions",
    "unit_conversions_script_name": "parity_units",
}
TRANSFORMER_FEED_ROW: Final[dict[str, Json]] = {
    "perspective": "Parity",
    "origin_dataset": "AIS_Feed",
    "query": "SELECT * FROM 'AIS_Feed' WHERE valid = true",
    "default_domain": "SURFACE",
    "origin_to_destination_mapping": [
        {"origin_column": "mmsi", "destination_column": "identity.mmsi"},
        {"origin_column": "mmsi", "destination_column": "identity.alias"},
        {"origin_column": "name", "destination_column": "name"},
        {"origin_column": "lat", "destination_column": "geodetic.latitude"},
        {"origin_column": "lon", "destination_column": "geodetic.longitude"},
        {"origin_column": "alt_ft", "destination_column": "geodetic.altitude"},
        {"origin_column": "sog", "destination_column": "speed"},
        {"origin_column": "cog", "destination_column": "heading"},
        {"origin_column": "ts", "destination_column": "estimatedKinematics.kinematicsTimestamp"},
        {
            "origin_column": "pos.err.major",
            "destination_column": "uncertainty.uncertaintyEllipse.semiMajorAxisLength",
        },
        {
            "origin_column": "pos.err.minor",
            "destination_column": "uncertainty.uncertaintyEllipse.semiMinorAxisLength",
        },
        {
            "origin_column": "pos.err.orient",
            "destination_column": "uncertainty.uncertaintyEllipse.orientation",
        },
        {"origin_column": "domain_hint", "destination_column": "environment.domain"},
        {"origin_column": "source.feed", "destination_column": "identity.feed"},
        {"literal": "AIS", "destination_column": "source.system"},
        {"literal": {"code": 7, "tags": ["a", "b"]}, "destination_column": "environment.meta"},
    ],
    "unit_conversions": [
        {"origin_column": "lat", "unit_conversion": "to_radians"},
        {"origin_column": "lon", "unit_conversion": "to_radians"},
        {"origin_column": "cog", "unit_conversion": "to_radians"},
        {"origin_column": "pos.err.orient", "unit_conversion": "to_radians"},
        {"origin_column": "alt_ft", "unit_conversion": "feet_to_meters"},
        {"origin_column": "lat", "unit_conversion": "PLACEHOLDER"},
    ],
    "custom_functions": [{"function_name": "tag_domain"}, {"function_name": "drop_flagged"}],
}


def _random_origin(rng: np.random.Generator, index: int) -> dict[str, Json]:
    record: dict[str, Json] = {"mmsi": int(rng.integers(100_000_000, 999_999_999))}
    if rng.random() < 0.9:
        record["crucibleHeader"] = {"uuid": f"{index:032x}", "createdDate": "2026-09-30"}
    optional: dict[str, Callable[[], Json]] = {
        "name": lambda: str(rng.choice(["ALPHA", "BRAVO", ""])),
        "lat": lambda: float(rng.uniform(-80.0, 80.0)),
        "lon": lambda: float(rng.uniform(-179.0, 179.0)),
        "alt_ft": lambda: float(rng.uniform(0.0, 40_000.0)),
        "sog": lambda: float(rng.uniform(0.0, 30.0)),
        "cog": lambda: float(rng.uniform(0.0, 359.0)),
        "ts": lambda: "2026-09-30T00:00:00.000Z",
        "unmapped": lambda: "dropped",
        "flagged": lambda: bool(rng.random() < 0.3),
        "source": lambda: {"feed": "aisstream", "other": int(rng.integers(0, 9))},
    }
    for key, value in optional.items():
        if rng.random() < 0.85:
            record[key] = value()
    if rng.random() < 0.7:
        error: dict[str, Json] = {"major": float(rng.uniform(1.0, 500.0))}
        if rng.random() < 0.7:
            error["minor"] = float(rng.uniform(1.0, 100.0))
        if rng.random() < 0.7:
            error["orient"] = float(rng.uniform(0.0, 180.0))
        record["pos"] = {"err": error}
    return record


def _transformer(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    module = baseline["entity_transformer"]
    for name, script in (
        ("unit_conversions", "parity_units"),
        ("custom_functions", "parity_functions"),
    ):
        hooks = ModuleType(name)
        exec(TRANSFORMER_SCRIPTS[script], hooks.__dict__)  # noqa: S102 - fixed test scripts
        setattr(module, name, hooks)
    config = {**TRANSFORMER_PERSPECTIVE_ROW, **TRANSFORMER_FEED_ROW}
    mappings = config["origin_to_destination_mapping"]
    query = str(config["query"])
    cases: list[Json] = []
    for batch_index in range(60):
        size = int(rng.integers(1, 12))
        records: list[Json] = [
            _random_origin(rng, batch_index * 100 + index) for index in range(size)
        ]
        # The pipeline at 1b534df, as in entity_transformer.sse_msg_processor.
        mapped = module.apply_mappings(
            module.preprocess(json.dumps(records), copy.deepcopy(config)), mappings
        )
        output: list[Json] = []
        for record in mapped:
            source_id = module.get_value(record, "crucibleHeader.uuid")
            module.set_value(record, "source.datasetName", module.get_dataset_name(query))
            if source_id is not module.MISSING:
                module.set_value(record, "source.uuid", source_id)
            compacted = module.project_report_record(record, mappings)
            if compacted:
                output.append(compacted)
        cases.append({"records": records, "expected": module.add_ECEF_kinematics(output)})
    scripts: dict[str, Json] = {}
    scripts.update(TRANSFORMER_SCRIPTS)
    return {
        "scripts": scripts,
        "perspective_row": TRANSFORMER_PERSPECTIVE_ROW,
        "feed_row": TRANSFORMER_FEED_ROW,
        "cases": cases,
    }


# --- tracker -------------------------------------------------------------------------

TRACKER_MODES: Final = {
    "kalman": "kalman q=2.5",
    "ci": "Kalman CI",
    "passthrough": "passthrough",
}
TRACKER_ORIGIN: Final = "Radar_Feed"
_TRACKER_IDENTITIES: Final[tuple[Mapping[str, str], ...]] = (
    {"callsign": "ALPHA", "environment": "AIR", "standard": "FRIEND"},
    {"callsign": "BRAVO", "environment": "SEA_SURFACE"},
    {"callsign": "CHARLIE", "environment": "GROUND", "standard": "HOSTILE"},
    {"callsign": "DELTA"},
    {"callsign": "ECHO", "environment": "SPACE"},
)


_TRACKER_ORIGIN_ECEF: Final = tuple(
    float(axis[0])
    for axis in geodetic_to_ecef(np.array([0.6]), np.array([-1.2]), np.array([1_000.0]))
)
"""Reports scatter around this surface point, where the baseline altitude formula is accurate."""


def _tracker_time(seconds: float) -> str:
    moment = _EPOCH + timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _random_tracker_report(
    rng: np.random.Generator, identity: Mapping[str, str], seconds: float, serial: int
) -> dict[str, Any]:
    base = np.array(_TRACKER_ORIGIN_ECEF) + rng.normal(0, 5_000.0, 3)
    report: dict[str, Any] = {
        "identity": dict(identity),
        "estimatedKinematics": {"kinematicsTimestamp": _tracker_time(seconds)},
        "source": {"uuid": f"report-{serial}", "datasetName": TRACKER_ORIGIN},
        "crucibleHeader": {"createdDate": _tracker_time(seconds - 1.0)},
        "trackQuality": float(rng.uniform(0.0, 1.0)),
        "mode": "LIVE",
    }
    position = {"x": float(base[0]), "y": float(base[1]), "z": float(base[2])}
    if rng.random() < 0.05:
        del position["z"]  # an incomplete position: the report is skipped
    report["ecefPosition"] = position
    if rng.random() < 0.6:
        report["positionCovariance"] = {
            "xx": float(rng.uniform(10.0, 1_000.0)),
            "yy": float(rng.uniform(10.0, 1_000.0)),
            "zz": float(rng.uniform(10.0, 1_000.0)),
            "xy": float(rng.uniform(-5.0, 5.0)),
        }
    if rng.random() < 0.5:
        report["ecefVelocity"] = {
            "x": float(rng.normal(0, 50.0)),
            "y": float(rng.normal(0, 50.0)),
            "z": float(rng.normal(0, 5.0)),
        }
        if rng.random() < 0.5:
            report["velocityCovariance"] = {
                "dxdx": float(rng.uniform(1.0, 100.0)),
                "dydy": float(rng.uniform(1.0, 100.0)),
                "dzdz": float(rng.choice([0.0, 5.0, 2e10])),
            }
    if rng.random() < 0.3:
        report["edhControlSet"] = {"classification": "U", "releasability": ["USA"]}
    if rng.random() < 0.2:
        report["speed"] = float(rng.uniform(0.0, 300.0))
        report["heading"] = float(rng.uniform(0.0, 6.28))
        report["geodetic"] = {"latitude": 0.5, "longitude": -1.2}
    return report


def _tracker_heads(tracker: ModuleType, rng: np.random.Generator) -> list[dict[str, Any]]:
    heads: list[dict[str, Any]] = []
    for identity in _TRACKER_IDENTITIES[:3]:
        head: dict[str, Any] = {
            "trackId": tracker.assign_track_id({"identity": dict(identity)}, TRACKER_ORIGIN),
            "identity": dict(identity),
            "trackUpdatedTimestamp": _tracker_time(-30.0),
            "ecefPosition": dict(zip("xyz", _TRACKER_ORIGIN_ECEF, strict=True)),
            "positionCovariance": {"xx": 500.0, "yy": 500.0, "zz": 500.0, "xy": 1.0},
            "velocityCovariance": {"dxdx": 50.0, "dydy": 50.0, "dzdz": 50.0},
            "positionVelocityCovariance": {"xdx": 2.0, "ydy": 2.0},
        }
        if rng.random() < 0.5:
            head["ecefVelocity"] = {"x": 10.0, "y": -5.0}
        heads.append(head)
    heads.append({"trackId": "not-this-feed", "ecefPosition": {"x": 1.0, "y": 2.0, "z": 3.0}})
    return heads


@dataclass(frozen=True, slots=True)
class _BaselineTracker:
    """The baseline modules and one feed's filter manager."""

    tracker: ModuleType
    records: ModuleType
    utils: ModuleType
    manager: Any
    config: dict[str, Any]

    _applied_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Record which batch rows the baseline filter turns into track rows."""
        process = self.manager.process_measurements_batch

        def recording(records: list[dict[str, Any]], config: dict[str, Any]) -> Any:  # noqa: ANN401 - the baseline's return
            posteriors = process(records, config)
            self._applied_indices[:] = list(posteriors)
            return posteriors

        self.manager.process_measurements_batch = recording

    @property
    def passthrough(self) -> bool:
        return bool(self.manager.passthrough_tracker)

    def outputs(self, reports: list[dict[str, Any]]) -> dict[str, Json]:
        """The baseline's track events and heads for one batch, as `tracker()` builds them.

        Two deliberate differences are applied to the baseline output: each
        event's ``reportIds`` comes from the report that produced it (the
        baseline pairs them by position, which shifts after a skipped report),
        and geodetic angles are converted to radians (the baseline writes
        degrees since 40aae4e). The time-dependent ``stale`` field is omitted.
        """
        events = copy.deepcopy(reports)
        for event in events:
            event["trackId"] = self.tracker.assign_track_id(event, TRACKER_ORIGIN)
        events.sort(key=lambda record: str(record["estimatedKinematics"]["kinematicsTimestamp"]))
        if self.passthrough:
            rows, heads = self.tracker.copy_kinematics_to_track(events, self.config, self.manager)
            producers = events
        else:
            rows = self.tracker.process_with_kalman(events, self.config, self.manager)
            producers = [events[index] for index in self._applied_indices]
            heads = []
            for row in rows:
                head = self.records.clone_record(row)
                head["trackUpdatedTimestamp"] = head.pop("interceptTimestamp", None)
                head.pop("trackQuality", None)
                heads.append(head)
            rows = _radians(self.records.add_track_wgs84_kinematics(rows))
            heads = _radians(self.records.add_track_wgs84_kinematics(heads))
        for row, producer in zip(rows, producers, strict=True):
            row["reportIds"] = [producer["source"]["uuid"]]
        heads = self.utils.last_record_by_track(heads)
        for head in heads:
            head.pop("stale", None)
            if self.passthrough:
                head.setdefault("speed", 0.0)
                head.setdefault("heading", 0.0)
        return {
            "events": [self.records.compact_record(row) for row in rows],
            "heads": [self.records.compact_record(head) for head in heads],
        }


def _json(value: object) -> Json:
    """`value`, which the baseline built from JSON-compatible parts, as `Json`."""
    return cast("Json", value)


def _radians(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the geodetic angles the baseline derived (in degrees) from a full ECEF position."""
    for record in records:
        geodetic = record.get("geodetic")
        position = record.get("ecefPosition")
        derived = isinstance(position, dict) and all(axis in position for axis in "xyz")
        if derived and isinstance(geodetic, dict):
            geodetic["latitude"] = math.radians(geodetic["latitude"])
            geodetic["longitude"] = math.radians(geodetic["longitude"])
    return records


def _tracker(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    cases: dict[str, Json] = {}
    for name, mode in TRACKER_MODES.items():
        config: dict[str, Any] = {"origin_dataset": TRACKER_ORIGIN, "crucible_tracker": mode}
        manager = baseline["entity_tracker"].NumpyKalmanFilterManager(config)
        manager.passthrough_tracker = mode == "passthrough"
        if "CI" in mode:
            manager.fusion_method = "ci"
        heads = _tracker_heads(baseline["entity_tracker"], rng)
        asyncio.run(manager.initialize_from_heads(mode, preloaded_heads=copy.deepcopy(heads)))
        run = _BaselineTracker(
            baseline["entity_tracker"],
            baseline["entity_transformer_records"],
            baseline["entity_utils"],
            manager,
            config,
        )
        clock = {identity["callsign"]: 0.0 for identity in _TRACKER_IDENTITIES}
        batches: list[Json] = []
        serial = 0
        for batch_index in range(25):
            reports: list[dict[str, Any]] = []
            for _ in range(int(rng.integers(1, 9))):
                identity = _TRACKER_IDENTITIES[int(rng.integers(0, len(_TRACKER_IDENTITIES)))]
                callsign = identity["callsign"]
                jump = 20 * 60.0 if batch_index == 12 and callsign == "ALPHA" else 0.0
                clock[callsign] += float(rng.uniform(0.5, 30.0)) + jump
                serial += 1
                reports.append(_random_tracker_report(rng, identity, clock[callsign], serial))
            batches.append({"reports": _json(reports), **run.outputs(reports)})
        cases[name] = {
            "mode": mode,
            "heads": _json(heads),
            "batches": batches,
        }
    return cases


# --- fuser ---------------------------------------------------------------------------

FUSER_MODES: Final = {
    "ci": (True, None, False),
    "ci_fixed": (True, 0.3, False),
    "kalman": (False, None, False),
    "passthrough": (False, None, True),
}
"""Each mode's (covariance intersection, fixed omega, passthrough)."""

_FUSER_COMPONENTS: Final = tuple(f"{index:032x}" for index in range(1, 8))
_OLD_PRINCIPAL: Final = "e" * 32
"""A principal from an earlier pipeline, kept by a preloaded association."""


def _component_event(
    rng: np.random.Generator, component: str, seconds: float, serial: int
) -> dict[str, Any]:
    base = np.array(_TRACKER_ORIGIN_ECEF) + rng.normal(0, 200.0, 3)
    callsigns = ("ALPHA", "BRAVO", "", None)
    event: dict[str, Any] = {
        "trackId": component,
        "interceptTimestamp": _tracker_time(seconds),
        "trackOriginatedTimestamp": _tracker_time(0.0),
        "standardIdentity": "FRIEND",
        "environment": str(rng.choice(["AIR", "SEA_SURFACE", "GROUND"])),
        "trackQuality": float(rng.uniform(0.0, 1.0)),
        "mode": "LIVE",
        "reportIds": [f"report-{serial}"],
        "identity": {
            "callsign": callsigns[int(rng.integers(0, len(callsigns)))],
            f"tag{int(rng.integers(0, 3))}": str(rng.integers(0, 100)),
        },
        "ecefPosition": {"x": float(base[0]), "y": float(base[1]), "z": float(base[2])},
        "geodetic": {"latitude": 0.6, "longitude": -1.2, "altitude": 1_000.0},
        "positionCovariance": {
            "xx": float(rng.uniform(10.0, 500.0)),
            "yy": float(rng.uniform(10.0, 500.0)),
            "zz": float(rng.uniform(10.0, 500.0)),
            "xy": float(rng.uniform(-2.0, 2.0)),
        },
        "positionVelocityCovariance": {"xdx": 0.5},
    }
    if rng.random() < 0.7:
        event["ecefVelocity"] = {
            "x": float(rng.normal(0, 20.0)),
            "y": float(rng.normal(0, 20.0)),
            "z": float(rng.normal(0, 2.0)),
        }
        event["velocityCovariance"] = {"dxdx": 4.0, "dydy": 4.0, "dzdz": 1.0}
    if rng.random() < 0.05:
        del event["ecefPosition"]["z"]
    return event


def _management_event(
    track: str, action: str, day_seconds: float, target: str | None = None
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "trackId": track,
        "action": action,
        "crucibleHeader": {"updatedDate": _tracker_time(day_seconds)},
    }
    if target is not None:
        event["supersededBy"] = target
    return event


_FUSER_MANAGEMENT: Final[dict[int, list[tuple[str, str, str | None]]]] = {
    4: [(_FUSER_COMPONENTS[1], "SUPERSEDE", _FUSER_COMPONENTS[0])],
    8: [(_FUSER_COMPONENTS[5], "DELETE", None)],
    10: [
        (_FUSER_COMPONENTS[3], "SUPERSEDE", _FUSER_COMPONENTS[2]),
        (_FUSER_COMPONENTS[2], "SUPERSEDE", _FUSER_COMPONENTS[4]),
    ],
    14: [(_FUSER_COMPONENTS[1], "RESTORE", None)],
}
"""Management events applied before the batch with the given index."""


@dataclass(frozen=True, slots=True)
class _BaselineFuser:
    """The baseline fuser worker's per-batch logic, without its writes."""

    fuser: ModuleType
    records: ModuleType
    utils: ModuleType
    kf: Any
    identities: dict[str, Any]
    passthrough: bool

    def apply_management(self, events: list[dict[str, Any]]) -> None:
        updated = self.fuser._apply_management_events_to_supersede_map(
            self.kf.supersede_map, events
        )
        self.kf.update_supersede_map(updated)
        for principal in self.kf.recently_restored_principal_trackids:
            self.identities.pop(principal, None)

    def outputs(self, components: list[dict[str, Any]]) -> dict[str, Json]:
        """The baseline's principal events and heads for one batch.

        Three documented fixes are applied: a deleted component track is not
        fused (the baseline fuses it into a principal derived from "None", or
        into its old principal), geodetic coordinates come from the fused
        position in radians (the baseline discards that result), and the
        fused identity starts from the preloaded principal heads.
        """
        ordered = sorted(
            (event for event in copy.deepcopy(components) if event.get("interceptTimestamp")),
            key=lambda event: str(event["interceptTimestamp"]),
        )
        events, heads = [], []
        for component in ordered:
            component_id = str(component["trackId"])
            if self.fuser._find_supersede_root(component_id, self.kf.supersede_map) is None:
                continue
            environment = component.get("environment")
            principal, principal_id, _, _ = self.kf.get_or_create_principal_track(
                component_id, component_id, environment=environment
            )
            self.kf.set_track_environment(principal, environment)
            self.kf.add_component_track(principal, component_id)
            event = self.fuser.create_principal_track_event(component, principal)
            head = self.fuser.create_principal_track_head(component, principal)
            if not self.passthrough:
                posterior = self.kf.update_record(principal, component)
                if posterior is not None:
                    state = self.kf.state_record(posterior, principal, principal_id)
                    event.update(state)
                    head.update(state)
            superseded = self.kf.supersede_map.get(component_id) is not None
            fused = self.fuser.fuse_track_identity(
                self.identities, principal, component, is_superseded=superseded
            )
            for path, value in fused.items():
                self.records.set_value(event, path, value)
                self.records.set_value(head, path, value)
            events.append(event)
            heads.append(head)
        heads = self.utils.last_record_by_track(heads)
        events = _radians(self.records.add_track_wgs84_kinematics(events))
        heads = _radians(self.records.add_track_wgs84_kinematics(heads))
        for head in heads:
            head.pop("stale", None)
        return {
            "events": [self.records.compact_record(event) for event in events],
            "heads": [self.records.compact_record(head) for head in heads],
        }


def _fuser_heads() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    survivor = _FUSER_COMPONENTS[0]
    principal = uuid.uuid5(uuid.NAMESPACE_DNS, f"entity_principal_track:{survivor}").hex
    principal_heads = [
        {
            "trackId": principal,
            "trackUpdatedTimestamp": _tracker_time(-60.0),
            "identity": {"callsign": "PRELOADED", "registry": "R-1"},
            "ecefPosition": dict(zip("xyz", _TRACKER_ORIGIN_ECEF, strict=True)),
            "positionCovariance": {"xx": 50.0, "yy": 50.0, "zz": 50.0},
        },
        {
            "trackId": _OLD_PRINCIPAL,
            "trackUpdatedTimestamp": _tracker_time(-60.0),
            "ecefPosition": dict(zip("xyz", _TRACKER_ORIGIN_ECEF, strict=True)),
        },
    ]
    component_heads = [
        {"trackId": _FUSER_COMPONENTS[6], "associatedPrincipalTrack": _OLD_PRINCIPAL},
        {"trackId": "f" * 32},
    ]
    return principal_heads, component_heads


def _fuser(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    module = baseline["entity_track_fuser"]
    cases: dict[str, Json] = {}
    for name, (use_ci, omega, passthrough) in FUSER_MODES.items():
        kf = baseline["entity_fusion_filter"].EntityPrincipalTrackFilter({})
        principal_heads, component_heads = _fuser_heads()
        kf.initialize({}, copy.deepcopy(principal_heads), copy.deepcopy(component_heads))
        if use_ci:
            kf.set_fusion_method("ci", omega=omega)
        identities: dict[str, Any] = {}
        for head in principal_heads:
            module.fuse_track_identity(identities, head["trackId"], head)
        run = _BaselineFuser(
            module,
            baseline["entity_transformer_records"],
            baseline["entity_utils"],
            kf,
            identities,
            passthrough,
        )
        clock = dict.fromkeys(_FUSER_COMPONENTS, 0.0)
        steps: list[Json] = []
        serial = 0
        for batch_index in range(18):
            if batch_index in _FUSER_MANAGEMENT:
                management = [
                    _management_event(track, action, batch_index * 10.0 + offset, target)
                    for offset, (track, action, target) in enumerate(_FUSER_MANAGEMENT[batch_index])
                ]
                run.apply_management(management)
                steps.append({"management": _json(management)})
            components: list[dict[str, Any]] = []
            for _ in range(int(rng.integers(2, 8))):
                component = _FUSER_COMPONENTS[int(rng.integers(0, len(_FUSER_COMPONENTS)))]
                stale = rng.random() < 0.08 and clock[component] > 5.0
                clock[component] += -3.0 if stale else float(rng.uniform(0.5, 20.0))
                serial += 1
                components.append(_component_event(rng, component, clock[component], serial))
            steps.append({"components": _json(components), **run.outputs(components)})
        cases[name] = {
            "ci": use_ci,
            "omega": omega,
            "passthrough": passthrough,
            "principal_heads": _json(principal_heads),
            "component_heads": _json(component_heads),
            "steps": steps,
        }
    return cases


# --- duplicate detection -------------------------------------------------------------

_DUPLICATE_TRACKS: Final = {
    # name: (environment, offset east m, velocity east m/s, velocity north m/s, noise m, start s)
    "air-a": ("AIR", 0.0, 120.0, 40.0, 15.0, 0.0),
    "air-b": ("AIR", 5.0, 120.0, 40.0, 15.0, 2.5),
    "air-formation": ("AIR", 400.0, 120.0, 40.0, 15.0, 1.0),
    "sea-a": ("SEA_SURFACE", 0.0, 5.0, 2.0, 60.0, 0.0),
    "sea-b": ("SEA_SURFACE", 40.0, 5.0, 2.0, 60.0, 7.0),
    "ground-a": ("GROUND", 0.0, 10.0, 0.0, 5.0, 0.0),
    "ground-b": ("GROUND", 8.0, -10.0, 0.0, 5.0, 1.5),
    "unknown-a": (None, 0.0, 30.0, 30.0, 20.0, 0.0),
    "unknown-b": (None, 10.0, 30.0, 30.0, 20.0, 3.0),
    "protected": ("AIR", 2.0, 120.0, 40.0, 15.0, 0.5),
}


def _duplicate_events(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Component track events for the scenario tracks, in ascending time order."""
    origin = np.array(_TRACKER_ORIGIN_ECEF)
    up = origin / np.linalg.norm(origin)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    events: list[dict[str, Any]] = []
    for index, (name, (environment, offset, east_mps, north_mps, noise, start)) in enumerate(
        _DUPLICATE_TRACKS.items()
    ):
        track = f"{index + 1:032x}"
        velocity = east * east_mps + north * north_mps
        for step in range(int(rng.integers(12, 20))):
            seconds = start + step * float(rng.uniform(4.0, 8.0)) + step * 0.001
            truth = origin + east * offset + velocity * seconds
            observed = truth + rng.normal(0.0, noise, 3)
            event: dict[str, Any] = {
                "trackId": track,
                "interceptTimestamp": _tracker_time(seconds),
                "ecefPosition": dict(zip("xyz", map(float, observed), strict=True)),
                "ecefVelocity": dict(
                    zip("xyz", map(float, velocity + rng.normal(0, 1.0, 3)), strict=True)
                ),
                "trackOriginatedTimestamp": _tracker_time(start),
                "standardIdentity": "UNKNOWN",
                "name": name,
            }
            if environment is not None:
                event["environment"] = environment
            if rng.random() < 0.9:
                variance = noise**2
                event["positionCovariance"] = {
                    "xx": variance,
                    "xy": 0.0,
                    "xz": 0.0,
                    "yy": variance,
                    "yz": 0.0,
                    "zz": variance,
                }
            events.append(event)
    events.append(
        {
            "trackId": "f" * 32,
            "interceptTimestamp": _tracker_time(1.0),
            "ecefPosition": {"x": 0, "y": 0, "z": 0},
        }
    )
    events.sort(key=lambda event: event["interceptTimestamp"])
    return events


def _duplicates(baseline: dict[str, ModuleType], rng: np.random.Generator) -> dict[str, Json]:
    """The baseline's duplicate pairs, with events fed in ascending time order.

    Ascending input is the documented fix: the baseline reads events newest
    first and pairs those times with the smoother's ascending states. The
    baseline runs in UTC so that its local-time epochs equal the port's.
    """
    module = baseline["entity_duplicate_identifier"]
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        cases: list[Json] = []
        for _ in range(6):
            events = _duplicate_events(rng)
            protected = {f"{len(_DUPLICATE_TRACKS):032x}"}
            setattr(module, "processed_supersedes", set(protected))  # noqa: B010 - an untyped global
            found = module.find_duplicate_tracks(copy.deepcopy(events), [], {})
            pairs = {
                f"{candidate.track_id_1}:{candidate.track_id_2}": {
                    "mean_mahalanobis": float(candidate.avg_position_distance),
                    "mean_velocity_difference": float(candidate.avg_velocity_difference),
                    "matching_points": int(candidate.num_matching_points),
                    "time_overlap_seconds": float(candidate.time_overlap_seconds),
                    "confidence": float(candidate.confidence_score),
                }
                for candidate in found
            }
            cases.append(
                {
                    "events": _json(events),
                    "protected": _json(sorted(protected)),
                    "pairs": _json(pairs),
                }
            )
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()
    return {"cases": cases}


GENERATORS: Final[
    dict[str, Callable[[dict[str, ModuleType], np.random.Generator], dict[str, Json]]]
] = {
    "kalman": _kalman,
    "smoother": _smoother,
    "core": _core,
    "management": _management,
    "transformer": _transformer,
    "tracker": _tracker,
    "fuser": _fuser,
    "duplicates": _duplicates,
}


if __name__ == "__main__":
    main()
