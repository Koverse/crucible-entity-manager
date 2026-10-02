"""Finding duplicate component tracks by their smoothed kinematics (DESIGN.md §5.11).

This is the baseline duplicate identifier's algorithm, unchanged except where
noted:

1. Build each component track's history from recent track events and smooth it
   with an RTS smoother. The smoother returns the times of its states, so a
   history never pairs a time with another time's state (the baseline did, §8).
2. Find candidate pairs: tracks in the same environment with enough positions
   within the candidate search radius of each other, and overlapping in time.
3. Evaluate each pair on a common time grid, extrapolating each track at
   constant velocity: horizontal Mahalanobis distance, median separation, mean
   velocity difference and a confidence score, against per-environment
   thresholds.

Times are UTC epoch seconds.
"""

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

import numpy as np
from scipy.spatial import KDTree

from crucible_entity_manager.core.aliases import FloatArray, JSONObject, JSONValue
from crucible_entity_manager.core.records import MISSING, get_path
from crucible_entity_manager.core.smoother import smooth_track
from crucible_entity_manager.core.timeutil import parse_timestamp

logger = logging.getLogger(__name__)

DEFAULT_POSITION_VARIANCE: Final = 100.0**2
"""Position variance assumed without a covariance, as at ``1b534df``."""
EXTRAPOLATION_PROCESS_NOISE: Final = 1.0
MAX_GRID_POINTS: Final = 500
MOVING_SPEED_MPS: Final = 0.1
TIMESTAMP_PATHS: Final = (
    "interceptTimestamp",
    "trackOriginatedTimestamp",
    "crucibleHeader.createdDate",
)
_POSITION_PATHS: Final = ("ecefPosition.x", "ecefPosition.y", "ecefPosition.z")
_VELOCITY_PATHS: Final = ("ecefVelocity.x", "ecefVelocity.y", "ecefVelocity.z")
_COVARIANCE_PATHS: Final = tuple(
    f"positionCovariance.{term}" for term in ("xx", "xy", "xz", "yy", "yz", "zz")
)


@dataclass(frozen=True, slots=True)
class DetectionParams:
    """Thresholds for one environment."""

    candidate_search_radius_m: float = 500.0
    """Radius of the coarse k-d tree search; it does not itself classify a duplicate."""
    distance_threshold_sigma: float = 3.0
    """Largest mean horizontal Mahalanobis distance of a duplicate."""
    velocity_threshold_mps: float = 10.0
    time_alignment_seconds: float = 5.0
    """How far a track is extrapolated beyond its ends."""
    eval_interval_seconds: float = 10.0
    min_matching_points: int = 5
    """Fewest distinct measurement times each track of a pair needs."""
    min_confidence: float = 0.5
    max_separation_m: float | None = None
    measurement_noise_floor_m: float | None = None


PARAMS_BY_ENVIRONMENT: Final[Mapping[str, DetectionParams]] = {
    # Airborne near-duplicates are often distinct aircraft (in trail, formation
    # or holding), so the gates are tight.
    "AIR": DetectionParams(
        candidate_search_radius_m=300.0,
        distance_threshold_sigma=2.0,
        velocity_threshold_mps=20.0,
        time_alignment_seconds=5.0,
        eval_interval_seconds=5.0,
        min_matching_points=5,
        min_confidence=0.6,
        max_separation_m=250.0,
        measurement_noise_floor_m=30.0,
    ),
    "SEA_SURFACE": DetectionParams(
        candidate_search_radius_m=4000.0,
        velocity_threshold_mps=5.0,
        time_alignment_seconds=300.0,
        eval_interval_seconds=30.0,
        max_separation_m=500.0,
        measurement_noise_floor_m=150.0,
    ),
    "SEA_SUBSURFACE": DetectionParams(
        candidate_search_radius_m=4000.0,
        velocity_threshold_mps=5.0,
        time_alignment_seconds=300.0,
        eval_interval_seconds=60.0,
        max_separation_m=500.0,
        measurement_noise_floor_m=150.0,
    ),
    "GROUND": DetectionParams(
        candidate_search_radius_m=150.0,
        distance_threshold_sigma=2.0,
        velocity_threshold_mps=5.0,
        time_alignment_seconds=5.0,
        eval_interval_seconds=10.0,
        min_matching_points=5,
        min_confidence=0.6,
        max_separation_m=30.0,
        measurement_noise_floor_m=15.0,
    ),
}


@dataclass(frozen=True, slots=True)
class TrackHistory:
    """A component track's smoothed states, in time order."""

    track_id: str
    environment: str
    """Upper-cased; empty when unknown."""
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    covariances: FloatArray
    measurement_times: int
    """Distinct measurement times, which is how much independent evidence there is."""


@dataclass(frozen=True, slots=True)
class Duplicate:
    """Two component tracks judged to be the same object, with the evidence."""

    track_id_1: str
    track_id_2: str
    mean_mahalanobis: float
    mean_velocity_difference: float
    matching_points: int
    time_overlap_seconds: float
    confidence: float


@dataclass(slots=True)
class _Points:
    environment: str
    times: list[float] = field(default_factory=list)
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    velocities: list[tuple[float, float, float]] = field(default_factory=list)
    covariances: list[FloatArray | None] = field(default_factory=list)


def build_histories(events: Sequence[JSONObject], excluded: set[str]) -> dict[str, TrackHistory]:
    """Smoothed histories of the tracks in `events`, leaving out `excluded` tracks.

    Velocities and covariances are used only if every component appears in
    some event, and the timestamp is the first of `TIMESTAMP_PATHS` any event
    has, as at ``1b534df``. Events at the origin or with a non-finite position
    are skipped.
    """
    if not events or not all(_present(events, path) for path in _POSITION_PATHS):
        return {}
    timestamp_path = next((path for path in TIMESTAMP_PATHS if _present(events, path)), None)
    if timestamp_path is None:
        return {}
    has_velocity = all(_present(events, path) for path in _VELOCITY_PATHS)
    has_covariance = all(_present(events, path) for path in _COVARIANCE_PATHS)
    points: dict[str, _Points] = {}
    for event in events:
        track_id = get_path(event, "trackId")
        value = get_path(event, timestamp_path)
        when = None if value is MISSING else parse_timestamp(value)
        if not track_id or not isinstance(track_id, str) or when is None:
            continue
        position = _vector(event, _POSITION_PATHS)
        if (
            position is None
            or not all(math.isfinite(axis) for axis in position)
            or position == (0.0, 0.0, 0.0)
        ):
            continue
        velocity = (_vector(event, _VELOCITY_PATHS) if has_velocity else None) or (0.0, 0.0, 0.0)
        environment = get_path(event, "environment")
        track = points.setdefault(
            track_id, _Points(str(environment).upper() if isinstance(environment, str) else "")
        )
        track.times.append(when.timestamp())
        track.positions.append(position)
        track.velocities.append(velocity)
        track.covariances.append(_covariance(event) if has_covariance else None)
    return {
        track_id: _smoothed(track_id, track, has_velocity=has_velocity)
        for track_id, track in points.items()
        if track_id not in excluded
    }


def find_duplicates(
    histories: Mapping[str, TrackHistory], fallback: DetectionParams
) -> list[Duplicate]:
    """Duplicate pairs, most confident first.

    `fallback` applies to environments without their own parameters. The
    candidate search uses the loosest bounds across all parameters.
    """
    every = [*PARAMS_BY_ENVIRONMENT.values(), fallback]
    candidates = candidate_pairs(
        histories,
        radius_m=max(params.candidate_search_radius_m for params in every),
        time_alignment_seconds=max(params.time_alignment_seconds for params in every),
        min_points=min(params.min_matching_points for params in every),
    )
    found: list[Duplicate] = []
    for first, second in sorted(candidates):
        params = PARAMS_BY_ENVIRONMENT.get(histories[first].environment or "UNKNOWN", fallback)
        duplicate = evaluate_pair(histories[first], histories[second], params)
        if duplicate is not None and duplicate.confidence >= params.min_confidence:
            found.append(duplicate)
    found.sort(key=lambda duplicate: duplicate.confidence, reverse=True)
    return found


def candidate_pairs(
    histories: Mapping[str, TrackHistory],
    *,
    radius_m: float,
    time_alignment_seconds: float,
    min_points: int,
) -> set[tuple[str, str]]:
    """Pairs with `min_points` nearby positions each, in one environment, overlapping in time."""
    owners: list[str] = []
    times: list[float] = []
    blocks: list[FloatArray] = []
    for track_id, history in histories.items():
        finite = np.isfinite(history.positions).all(axis=1)
        blocks.append(history.positions[finite])
        times.extend(float(time) for time in history.times[finite])
        owners.extend([track_id] * int(finite.sum()))
    if len(owners) < 2:  # noqa: PLR2004 - a pair needs two points
        return set()
    positions = np.concatenate(blocks)
    near: defaultdict[tuple[str, str], tuple[set[float], set[float]]] = defaultdict(
        lambda: (set(), set())
    )
    for left, right in KDTree(positions).query_pairs(r=radius_m, output_type="ndarray"):
        first, second = owners[left], owners[right]
        if first == second or histories[first].environment != histories[second].environment:
            continue
        pair = (first, second) if first < second else (second, first)
        first_times, second_times = near[pair]
        first_times.add(times[left] if first == pair[0] else times[right])
        second_times.add(times[right] if first == pair[0] else times[left])
    pairs: set[tuple[str, str]] = set()
    for pair, (first_times, second_times) in near.items():
        if min(len(first_times), len(second_times)) < min_points:
            continue
        first, second = histories[pair[0]], histories[pair[1]]
        if first.times[-1] + time_alignment_seconds < second.times[0] - time_alignment_seconds:
            continue
        if second.times[-1] + time_alignment_seconds < first.times[0] - time_alignment_seconds:
            continue
        pairs.add(pair)
    return pairs


@dataclass(frozen=True, slots=True)
class _Comparison:
    """A pair's per-point comparison over the evaluation grid."""

    distances: list[float]
    separations: list[float]
    velocity_deltas: list[FloatArray]


def evaluate_pair(
    first: TrackHistory, second: TrackHistory, params: DetectionParams
) -> Duplicate | None:
    """Whether two histories are the same object, with the evidence, or ``None``."""
    if first.track_id == second.track_id or first.environment != second.environment:
        return None
    independent = min(first.measurement_times, second.measurement_times)
    if independent < params.min_matching_points:
        return None
    comparison = _compare(first, second, params)
    if comparison is None:
        return None
    mean_distance = float(np.mean(comparison.distances))
    mean_velocity_difference = float(
        np.linalg.norm(np.mean(np.stack(comparison.velocity_deltas), axis=0))
    )
    speeds = np.linalg.norm(np.concatenate([first.velocities, second.velocities]), axis=1)
    moving = bool((speeds > MOVING_SPEED_MPS).any())
    if mean_distance > params.distance_threshold_sigma or (
        moving and mean_velocity_difference > params.velocity_threshold_mps
    ):
        return None
    return Duplicate(
        track_id_1=first.track_id,
        track_id_2=second.track_id,
        mean_mahalanobis=mean_distance,
        mean_velocity_difference=mean_velocity_difference,
        matching_points=independent,
        time_overlap_seconds=max(
            0.0, min(first.times[-1], second.times[-1]) - max(first.times[0], second.times[0])
        ),
        confidence=_confidence(
            independent,
            comparison.distances,
            [float(np.linalg.norm(delta)) for delta in comparison.velocity_deltas],
            params,
            moving=moving,
        ),
    )


def _compare(
    first: TrackHistory, second: TrackHistory, params: DetectionParams
) -> _Comparison | None:
    """Compare the tracks where both have a state on the grid.

    Returns ``None`` if too few grid points compare, or the tracks are too far apart.
    """
    grid = evaluation_grid(
        first, second, params.eval_interval_seconds, params.time_alignment_seconds
    )
    if grid is None:
        return None
    floor = float(params.measurement_noise_floor_m or 0.0)
    comparison = _Comparison([], [], [])
    # The grid lies within both tracks' extrapolation windows, so both have a
    # state at every point; the filter only guards against that changing.
    states = [
        (one, other)
        for one, other in zip(
            interpolate(first, grid, params.time_alignment_seconds),
            interpolate(second, grid, params.time_alignment_seconds),
            strict=True,
        )
        if one is not None and other is not None
    ]
    for one, other in states:
        comparison.distances.append(
            horizontal_mahalanobis(one[0], other[0], one[2], other[2], floor)
        )
        comparison.separations.append(float(np.linalg.norm(one[0] - other[0])))
        comparison.velocity_deltas.append(one[1] - other[1])
    if len(comparison.distances) < params.min_matching_points:
        return None
    if (
        params.max_separation_m is not None
        and float(np.median(comparison.separations)) > params.max_separation_m
    ):
        return None
    return comparison


def evaluation_grid(
    first: TrackHistory,
    second: TrackHistory,
    interval_seconds: float,
    extrapolate_seconds: float,
) -> FloatArray | None:
    """Evenly spaced times over the tracks' overlap, each extended by `extrapolate_seconds`."""
    start = max(first.times[0], second.times[0]) - extrapolate_seconds
    end = min(first.times[-1], second.times[-1]) + extrapolate_seconds
    if end <= start:
        return None
    count = min(max(2, int((end - start) / interval_seconds) + 1), MAX_GRID_POINTS)
    return np.linspace(start, end, count)


def interpolate(
    history: TrackHistory, times: FloatArray, extrapolate_seconds: float
) -> list[tuple[FloatArray, FloatArray, FloatArray] | None]:
    """The track's state at each time, predicted at constant velocity from the nearest state.

    Position variance grows by ``q·dt²`` per axis. A time further than
    `extrapolate_seconds` beyond either end gives ``None``.
    """
    lower = history.times[0] - extrapolate_seconds
    upper = history.times[-1] + extrapolate_seconds
    states: list[tuple[FloatArray, FloatArray, FloatArray] | None] = []
    last = len(history.times) - 1
    for time in times:
        if time < lower or time > upper:
            states.append(None)
            continue
        index = int(np.clip(np.searchsorted(history.times, time, side="right") - 1, 0, last))
        if index < last and history.times[index + 1] - time < time - history.times[index]:
            index += 1
        step = float(time - history.times[index])
        position = history.positions[index] + history.velocities[index] * step
        covariance = history.covariances[index] + EXTRAPOLATION_PROCESS_NOISE * step**2 * np.eye(3)
        states.append((position, history.velocities[index].copy(), covariance))
    return states


def horizontal_mahalanobis(
    first: FloatArray,
    second: FloatArray,
    first_covariance: FloatArray,
    second_covariance: FloatArray,
    noise_floor_m: float = 0.0,
) -> float:
    """Mahalanobis distance between two ECEF positions in the local horizontal plane.

    Altitude is usually unknown in reports, so only the east-north components
    count. A noise floor adds ``floor²`` to each position's horizontal
    variance. A singular projection falls back to horizontal distance over
    ``√2·floor`` (or 100 m without a floor).
    """
    basis = _horizontal_basis(first, second)
    delta = basis.T @ (first - second)
    floor_variance = noise_floor_m**2
    scale = math.sqrt(2.0) * noise_floor_m if floor_variance > 0.0 else 100.0
    covariance = basis.T @ (first_covariance + second_covariance) @ basis
    if floor_variance > 0.0:
        covariance = covariance + 2.0 * floor_variance * np.eye(2)
    try:
        return float(np.sqrt(delta @ np.linalg.solve(covariance, delta)))
    except np.linalg.LinAlgError:
        return float(np.sqrt(delta @ delta)) / scale


def choose_survivor(
    first: str, second: str, created: Callable[[str], datetime | None]
) -> tuple[str, str]:
    """``(superseded, survivor)``: the earlier-created track survives, else the lesser ID."""
    first_created, second_created = created(first), created(second)
    if first_created is not None and second_created is not None and first_created != second_created:
        return (second, first) if first_created < second_created else (first, second)
    return (second, first) if first < second else (first, second)


def creation_time(head: JSONObject) -> datetime | None:
    """A head's ``trackOriginatedTimestamp``, else its ``crucibleHeader.createdDate``."""
    for path in ("trackOriginatedTimestamp", "crucibleHeader.createdDate"):
        value = get_path(head, path)
        when = None if value is MISSING else parse_timestamp(value)
        if when is not None:
            return when
    return None


def _horizontal_basis(first: FloatArray, second: FloatArray) -> FloatArray:
    middle = (first + second) / 2.0
    radius = float(np.linalg.norm(middle))
    if radius < 1.0:
        return np.eye(3)[:, :2]
    up = middle / radius
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(up @ seed)) > 0.9:  # noqa: PLR2004 - near a pole, seed from another axis
        seed = np.array([1.0, 0.0, 0.0])
    east = np.cross(up, seed)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    north /= np.linalg.norm(north)
    return np.column_stack([east, north])


def _confidence(
    independent: int,
    distances: list[float],
    velocity_differences: list[float],
    params: DetectionParams,
    *,
    moving: bool,
) -> float:
    points = min(1.0, independent / 10.0)
    distance_spread = float(np.std(distances)) if len(distances) > 1 else 0.0
    position = 1.0 / (1.0 + distance_spread / params.distance_threshold_sigma)
    velocity = 1.0
    if moving:
        velocity_spread = (
            float(np.std(velocity_differences)) if len(velocity_differences) > 1 else 0.0
        )
        velocity = 1.0 / (1.0 + velocity_spread / max(params.velocity_threshold_mps, 1.0))
    return points * 0.4 + position * 0.4 + velocity * 0.2


def _smoothed(track_id: str, points: _Points, *, has_velocity: bool) -> TrackHistory:
    covariances = (
        np.stack([covariance for covariance in points.covariances if covariance is not None])
        if all(covariance is not None for covariance in points.covariances)
        else None
    )
    try:
        smoothed = smooth_track(
            np.array(points.times),
            np.array(points.positions),
            np.array(points.velocities) if has_velocity else None,
            covariances,
        )
    except (np.linalg.LinAlgError, ValueError) as error:
        logger.warning("Smoothing track %s failed (%s); using its raw states", track_id, error)
        return _raw_history(track_id, points, has_velocity=has_velocity)
    return TrackHistory(
        track_id=track_id,
        environment=points.environment,
        times=smoothed.times,
        positions=smoothed.positions,
        velocities=smoothed.velocities,
        covariances=smoothed.position_covariances,
        measurement_times=len(set(points.times)),
    )


def _raw_history(track_id: str, points: _Points, *, has_velocity: bool) -> TrackHistory:
    """The unsmoothed states in time order with the default variance, the baseline's fallback."""
    order = np.argsort(np.array(points.times), kind="stable")
    count = len(points.times)
    velocities = np.array(points.velocities) if has_velocity else np.zeros((count, 3))
    return TrackHistory(
        track_id=track_id,
        environment=points.environment,
        times=np.array(points.times)[order],
        positions=np.array(points.positions)[order],
        velocities=velocities[order],
        covariances=np.stack([np.eye(3) * DEFAULT_POSITION_VARIANCE] * count),
        measurement_times=len(set(points.times)),
    )


def _present(events: Sequence[JSONObject], path: str) -> bool:
    return any(get_path(event, path) not in (MISSING, None) for event in events)


def _vector(event: JSONObject, paths: tuple[str, str, str]) -> tuple[float, float, float] | None:
    """The three components as floats, or ``None`` if one is not a number.

    A missing or empty component counts as 0, as the baseline's
    ``float(value or 0)`` does.
    """
    values: list[float] = []
    for path in paths:
        value = get_path(event, path)
        if value is MISSING or not value:
            values.append(0.0)
        elif isinstance(value, int | float | str):
            try:
                values.append(float(value))
            except ValueError:
                return None
        else:
            return None
    return values[0], values[1], values[2]


def _covariance(event: JSONObject) -> FloatArray | None:
    values: list[float] = []
    for path in _COVARIANCE_PATHS:
        value: JSONValue | object = get_path(event, path)
        if not isinstance(value, int | float | str):
            return None
        try:
            values.append(float(value))
        except ValueError:
            return None
    xx, xy, xz, yy, yz, zz = values
    covariance = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return covariance if np.isfinite(covariance).all() else None
