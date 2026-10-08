"""
Shared Kalman filter manager for principal track fusion.

Provides a parameterized PrincipalTrackKalmanFilter class used by both
entity_track_fuser.py and object_track_fusion.py to fuse component tracks
into principal tracks using StoneSoup's Kalman filter.

State vector layout: [x, vx, y, vy, z, vz] (interleaved, constant-velocity model)
"""

import uuid
import logging
import traceback
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime as dt
from datetime import timezone as tz


def _isna(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return False
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _notna(value: Any) -> bool:
    return not _isna(value)


def _record_has(record: Any, key: str) -> bool:
    if record is None:
        return False
    if isinstance(record, dict):
        return key in record
    if hasattr(record, 'keys'):
        return key in record.keys()
    return False


def _record_get(record: Any, key: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(key, default)
    if hasattr(record, 'get'):
        return record.get(key, default)
    if hasattr(record, '__getitem__'):
        try:
            return record[key]
        except Exception:
            return default
    return default


def _iter_named_rows(rows: Any) -> List[dict]:
    if rows is None:
        return []
    if hasattr(rows, 'to_dict'):
        try:
            rows = rows.to_dict('records')
        except TypeError:
            pass
    if hasattr(rows, 'iterrows'):
        rows = [dict(r) for _, r in rows.iterrows()]
    if isinstance(rows, dict):
        return [dict(rows)]
    return [dict(r) for r in list(rows)]

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

logger = logging.getLogger(__name__)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState


class PrincipalTrackKalmanFilter:
    """
    Manages Kalman filters for principal tracks, parameterized for both
    entity (v2) and object (v1) pipelines.

    Args:
        config: Dataset configuration dict
        id_field: Primary identifier column name ('objectId' or 'entityId')
        vel_read_cols: Tuple of 3 column names for reading velocity (x, y, z)
        vel_write_cols: Tuple of 3 column names for writing velocity output
        vel_read_fallback: Optional fallback velocity column names
    """

    def __init__(self, config: Dict[str, Any], *,
                 id_field: str = 'objectId',
                 vel_read_cols: Tuple[str, str, str] = (
                     'ecefVelocity.dx', 'ecefVelocity.dy', 'ecefVelocity.dz'),
                 vel_write_cols: Tuple[str, str, str] = (
                     'ecefVelocity.dx', 'ecefVelocity.dy', 'ecefVelocity.dz'),
                 vel_read_fallback: Tuple[str, str, str] = None):
        self.config = config
        self.id_field = id_field
        self.vel_read_cols = vel_read_cols
        self.vel_write_cols = vel_write_cols
        self.vel_read_fallback = vel_read_fallback

        # Environment-based process noise (continuous-time acceleration PSD).
        # q ≈ σ_a² where σ_a is the typical acceleration for that domain.
        # Over interval dt:  σ_Δv = √(q·dt),  σ_Δp = √(q·dt³/3).
        #
        # AIR:            σ_a ≈ 3 m/s²   → q ≈ 5    (commercial turns / descents)
        # GROUND:         σ_a ≈ 2 m/s²   → q ≈ 3    (vehicles braking / turning)
        # SEA_SURFACE:    σ_a ≈ 0.7 m/s² → q ≈ 0.5  (ships maneuvering / accelerating)
        # SEA_SUBSURFACE: σ_a ≈ 0.1 m/s²  → q ≈ 0.01 (submarines)
        # SPACE:          nearly ballistic → q ≈ 0.001
        # UNKNOWN:        moderate default
        self.q_by_environment: Dict[str, float] = {
            'AIR': 5.0,
            'GROUND': 3.0,
            'SEA_SURFACE': 0.5,
            'SEA_SUBSURFACE': 0.5,
            'SPACE': 0.001,
            'UNKNOWN': 5.0,
        }
        self.q_default = 0.5

        # Velocity variance fallback when raw velocity is provided without
        # velocity covariance. Trust the reported value to within a few m/s.
        self.vel_var_by_environment: Dict[str, float] = {
            'AIR': 25.0,             # sigma 5 m/s
            'GROUND': 9.0,           # sigma 3 m/s
            'SEA_SURFACE': 4.0,      # sigma 2 m/s
            'SEA_SUBSURFACE': 4.0,   # sigma 2 m/s
            'SPACE': 100.0,          # sigma 10 m/s
            'UNKNOWN': 25.0,         # sigma 5 m/s
        }
        self.vel_var_default = 25.0  # sigma 5 m/s

        # KF components per principal track
        self.predictor: Dict[str, KalmanPredictor] = {}
        self.updater: Dict[str, KalmanUpdater] = {}
        self.measurement_model: Dict[str, LinearGaussian] = {}
        self.priors: Dict[str, GaussianState] = {}
        self.environment_by_track: Dict[str, str] = {}  # principal_track_id -> environment

        # Track management
        self.id_to_principal_trackid: Dict[str, str] = {}
        self.principal_trackid_to_id: Dict[str, str] = {}
        self.component_trackids_by_principal: Dict[str, Set[str]] = {}
        self.component_trackid_to_principal_trackid: Dict[str, str] = {}
        self.new_principal_trackids: Set[str] = set()
        self.known_principal_trackids: Set[str] = set()
        self.recently_restored_principal_trackids: Set[str] = set()
        self.supersede_map: Dict[str, str] = {}

        # Fusion method: 'kalman' (standard, assumes independence) or
        # 'ci' (Covariance Intersection, robust to unknown correlation)
        self.fusion_method = 'kalman'
        self.ci_omega = None  # None = auto-optimize per update

        self._ci_log_counter = 0

        self.stats = {
            'successful_updates': 0,
            'nan_resets': 0,
            'time_jump_resets': 0,
            'stale_measurements_skipped': 0,
            'covariance_explosion_resets': 0,
            'prediction_nan_resets': 0,
            'exception_resets': 0,
            'measurements_skipped_nan': 0,
        }

        self.initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, supersede_map: Dict[str, str],
                   principal_heads_df: Any = None,
                   component_heads_df: Any = None):
        """
        Initialize from pre-loaded data.

        Args:
            supersede_map: superseded_id -> superseding_id
            principal_heads_df: Iterable/dict-like principal track heads
            component_heads_df: Iterable/dict-like component track associations
        """
        if self.initialized:
            return

        self.supersede_map = supersede_map or {}

        principal_heads = _iter_named_rows(principal_heads_df)
        if principal_heads:
            logger.info(f"Processing {len(principal_heads)} existing principal track heads")
            for head in principal_heads:
                item_id = _record_get(head, self.id_field) or _record_get(head, f'{self.id_field}.uuid')
                principal_track_id = _record_get(head, 'trackId') or _record_get(head, 'trackId.uuid')
                if item_id and principal_track_id and _notna(item_id) and _notna(principal_track_id):
                    self.id_to_principal_trackid[str(item_id)] = principal_track_id
                    self.principal_trackid_to_id[principal_track_id] = str(item_id)
                    self.known_principal_trackids.add(principal_track_id)
                    if all(_record_has(head, col) for col in
                           ['ecefPosition.x', 'ecefPosition.y', 'ecefPosition.z']):
                        self._initialize_filter_from_head(head, principal_track_id)

        component_heads = _iter_named_rows(component_heads_df)
        if component_heads:
            logger.info(f"Processing {len(component_heads)} component track associations")
            for head in component_heads:
                comp_track_id = _record_get(head, 'trackId') or _record_get(head, 'trackId.uuid')
                assoc_principal = _record_get(head, 'associatedPrincipalTrack') or _record_get(head, 'associatedPrincipalTrack.uuid')
                if comp_track_id and assoc_principal and _notna(comp_track_id) and _notna(assoc_principal):
                    self.component_trackid_to_principal_trackid[comp_track_id] = assoc_principal
                    self.component_trackids_by_principal.setdefault(
                        assoc_principal, set()).add(comp_track_id)

        self.initialized = True
        logger.info(
            f"PrincipalTrackKalmanFilter initialized: "
            f"{len(self.supersede_map)} supersede mappings, "
            f"{len(self.id_to_principal_trackid)} principal tracks, "
            f"{len(self.component_trackid_to_principal_trackid)} component associations")

    def _get_q(self, environment: Optional[str] = None) -> float:
        """Return process noise for the given environment."""
        if environment:
            return self.q_by_environment.get(
                environment.upper(), self.q_default)
        return self.q_default

    def get_vel_var_fallback(self, environment: Optional[str] = None) -> float:
        """Return velocity variance fallback for the given environment."""
        if environment:
            return self.vel_var_by_environment.get(
                environment.upper(), self.vel_var_default)
        return self.vel_var_default

    def create_measurement(self, row: Any,
                           principal_track_id: str) -> Optional[Detection]:
        """Create a Detection using environment-appropriate velocity fallback."""
        env = self.environment_by_track.get(principal_track_id)
        vel_var = self.get_vel_var_fallback(env)
        return create_measurement_from_row(
            row, self.vel_read_cols, self.vel_read_fallback,
            vel_var_fallback=vel_var)

    def _initialize_filter_from_head(self, head: Any,
                                     principal_track_id: str):
        """Create KF components from an existing principal track head."""
        try:
            env = _record_get(head, 'environment')
            if env:
                self.environment_by_track[principal_track_id] = str(env)
            q = self._get_q(env)
            transition_model = CombinedLinearGaussianTransitionModel([
                ConstantVelocity(q),
                ConstantVelocity(q),
                ConstantVelocity(q),
            ])
            measurement_model = LinearGaussian(
                ndim_state=6, mapping=(0, 1, 2, 3, 4, 5),
                noise_covar=np.diag([1e7, 1e7, 1e7, 1e7, 1e7, 1e7]),
            )
            predictor = KalmanPredictor(transition_model)
            updater = KalmanUpdater(measurement_model)

            vx = self._read_velocity_component(head, 0)
            vy = self._read_velocity_component(head, 1)
            vz = self._read_velocity_component(head, 2)

            state_vector = np.array([
                _record_get(head, 'ecefPosition.x', 0), vx,
                _record_get(head, 'ecefPosition.y', 0), vy,
                _record_get(head, 'ecefPosition.z', 0), vz,
            ])

            if _record_has(head, 'positionCovariance.xx') and _notna(_record_get(head, 'positionCovariance.xx')):
                covar = _build_covariance_from_row(head)
            else:
                covar = np.diag([1e7, 1e7, 1e7, 1e7, 1e7, 1e7])

            ts = _record_get(head, 'trackUpdatedTimestamp') or _record_get(head, 'interceptTimestamp')
            try:
                timestamp = dt.strptime(
                    ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=tz.utc)
            except Exception:
                timestamp = dt.now(tz=tz.utc)

            self.priors[principal_track_id] = GaussianState(
                state_vector, covar, timestamp=timestamp)
            self.predictor[principal_track_id] = predictor
            self.updater[principal_track_id] = updater
            self.measurement_model[principal_track_id] = measurement_model
        except Exception as e:
            logger.warning(
                f"Could not initialize filter from head for track "
                f"{principal_track_id}: {e}")

    def _read_velocity_component(self, row: Any, axis: int) -> float:
        """Read one velocity component, trying primary then fallback column."""
        val = _record_get(row, self.vel_read_cols[axis], None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            if self.vel_read_fallback:
                val = _record_get(row, self.vel_read_fallback[axis], 0)
        return float(val or 0)

    # ------------------------------------------------------------------
    # Supersede / track management
    # ------------------------------------------------------------------

    def _discard_principal_group(self, principal_track_id: str) -> None:
        """Forget one fused principal group so incoming events rebuild it.

        RESTORE cannot subtract one component's contribution from a fused
        Kalman state. Dropping the small in-memory group is simpler and safer:
        deterministic principal IDs and subsequent component events recreate
        the correct split tracks and associations.
        """
        component_ids = self.component_trackids_by_principal.pop(
            principal_track_id, set())
        for component_id in component_ids:
            self.component_trackid_to_principal_trackid.pop(component_id, None)

        for item_id, mapped_principal in list(self.id_to_principal_trackid.items()):
            if mapped_principal == principal_track_id:
                self.id_to_principal_trackid.pop(item_id, None)

        self.principal_trackid_to_id.pop(principal_track_id, None)
        self.priors.pop(principal_track_id, None)
        self.predictor.pop(principal_track_id, None)
        self.updater.pop(principal_track_id, None)
        self.measurement_model.pop(principal_track_id, None)
        self.environment_by_track.pop(principal_track_id, None)
        self.new_principal_trackids.discard(principal_track_id)
        q_by_track = getattr(self, '_q_by_track', None)
        if q_by_track is not None:
            q_by_track.pop(principal_track_id, None)

    def update_supersede_map(self, new_supersede_map: Dict[str, str]) -> List[Dict[str, str]]:
        """Replace the supersede-map snapshot and update track associations.

        IDs present in the old snapshot but absent from the new snapshot were
        RESTOREd. Their affected fused principal groups are discarded and will
        be rebuilt from subsequent component events.

        Returns a list of ``{'trackId': ..., 'associatedPrincipalTrack': ...}``
        dicts for component tracks that were re-associated to a new principal
        track so the caller can write them back to component track heads.
        """
        restored_ids = set(self.supersede_map) - set(new_supersede_map)
        restored_principals = {
            self.id_to_principal_trackid[restored_id]
            for restored_id in restored_ids
            if restored_id in self.id_to_principal_trackid
        }
        restored_principals.update(
            self._deterministic_principal_trackid(restored_id)
            for restored_id in restored_ids
        )
        self.recently_restored_principal_trackids = set(restored_principals)
        for principal_track_id in restored_principals:
            self._discard_principal_group(principal_track_id)

        self.supersede_map = dict(new_supersede_map)
        re_associations: List[Dict[str, str]] = []

        for superseded_id, superseding_id in new_supersede_map.items():
            # Skip DELETE entries (superseding_id is None) — no target to merge into
            if not superseding_id:
                continue
            principal_id = self.get_principal_id(superseding_id)

            # Ensure the superseding principal track exists
            if principal_id not in self.id_to_principal_trackid:
                continue
            target_ptid = self.id_to_principal_trackid[principal_id]

            # Find the superseded object's old principal track (if any)
            old_ptid = self.id_to_principal_trackid.get(superseded_id)
            if old_ptid and old_ptid != target_ptid:
                # Block cross-environment merges
                old_env = self.environment_by_track.get(old_ptid)
                target_env = self.environment_by_track.get(target_ptid)
                if (old_env and target_env
                        and old_env.upper() != target_env.upper()):
                    logger.info(
                        f"Env guard: supersede {superseded_id} -> {principal_id} "
                        f"blocked ({old_env} vs {target_env})")
                    continue
                # Merge component tracks from old -> target
                old_components = self.component_trackids_by_principal.pop(old_ptid, set())
                self.component_trackids_by_principal.setdefault(target_ptid, set()).update(old_components)
                for ctid in old_components:
                    self.component_trackid_to_principal_trackid[ctid] = target_ptid
                    re_associations.append({
                        'trackId': ctid,
                        'associatedPrincipalTrack': target_ptid,
                    })
                # Point superseded id at the target principal track
                self.id_to_principal_trackid[superseded_id] = target_ptid
                logger.info(
                    f"Supersede {superseded_id} -> {principal_id}: "
                    f"merged {len(old_components)} component tracks into {target_ptid}")
            elif not old_ptid:
                # No existing principal track for superseded id; just update mapping
                self.id_to_principal_trackid[superseded_id] = target_ptid

        return re_associations

    def get_principal_id(self, item_id: str) -> str:
        """Follow supersede chain to find the final (non-superseded) ID."""
        current = item_id
        visited: Set[str] = set()
        while current in self.supersede_map:
            if current in visited:
                logger.warning(f"Circular supersede reference for {item_id}")
                break
            visited.add(current)
            current = self.supersede_map[current]
        return current

    def _env_mismatch(self, ptid: str, environment: Optional[str]) -> bool:
        """Return True if *environment* conflicts with the existing track env."""
        if not environment:
            return False
        existing = self.environment_by_track.get(ptid)
        if not existing:
            return False
        return existing.upper() != str(environment).upper()

    def _deterministic_principal_trackid(self, track_key: str) -> str:
        """Derive a stable principal trackId from the principal key.

        Deterministic (uuid5) so the same principal always maps to the same
        trackId across process restarts and across shards that failed to
        preload existing heads — preventing the unbounded orphan-head growth
        caused by minting a fresh uuid4 on every (re)creation. The id_field is
        included in the seed so object (objectId) and entity (trackId, itself a
        uuid5 of the identity customID) pipelines occupy distinct namespaces.
        """
        return uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"principal_track:{self.id_field}:{track_key}").hex

    def get_or_create_principal_track(
            self, item_id: str,
            component_track_id: str = None,
            environment: str = None
    ) -> Tuple[str, str, bool, bool]:
        """Get or create a principal track for the given item (object/entity).

        Args:
            item_id: An **objectId** (or entityId) -- NOT a trackId.
                     Used to look up the supersede chain and the principal track.
            component_track_id: A **trackId** from the component track event.
                                Used to maintain component-to-principal associations.
            environment: Environment of the incoming component (e.g. 'AIR',
                         'SEA_SURFACE').  When set, cross-environment supersede
                         merges are blocked — a new principal track is created
                         instead.

        Returns:
            (principal_track_id, principal_item_id,
             is_new_principal, is_new_association)
        """
        principal_item_id = self.get_principal_id(item_id)

        # Check if component track already associated
        if (component_track_id
                and component_track_id in self.component_trackid_to_principal_trackid):
            ptid = self.component_trackid_to_principal_trackid[component_track_id]
            # Verify the association is still valid after supersede resolution
            expected_ptid = self.id_to_principal_trackid.get(principal_item_id)
            if expected_ptid and expected_ptid != ptid:
                # Block cross-environment re-wire
                if self._env_mismatch(expected_ptid, environment):
                    logger.info(
                        f"Env guard: refusing re-wire of {component_track_id} "
                        f"({environment}) into {expected_ptid} "
                        f"({self.environment_by_track.get(expected_ptid)})")
                    pid = self.principal_trackid_to_id.get(ptid, principal_item_id)
                    return ptid, pid, False, False
                # Stale association — re-wire to the correct principal track
                old_set = self.component_trackids_by_principal.get(ptid)
                if old_set:
                    old_set.discard(component_track_id)
                self.component_trackids_by_principal.setdefault(
                    expected_ptid, set()).add(component_track_id)
                self.component_trackid_to_principal_trackid[component_track_id] = expected_ptid
                pid = self.principal_trackid_to_id.get(expected_ptid, principal_item_id)
                return expected_ptid, pid, False, True  # is_new_association=True for writeback
            pid = self.principal_trackid_to_id.get(ptid, principal_item_id)
            return ptid, pid, False, False

        # Check if principal already has a track
        if principal_item_id in self.id_to_principal_trackid:
            ptid = self.id_to_principal_trackid[principal_item_id]
            # Block cross-environment merge via supersede
            if item_id != principal_item_id and self._env_mismatch(ptid, environment):
                logger.info(
                    f"Env guard: {item_id} ({environment}) superseded to "
                    f"{principal_item_id} ({self.environment_by_track.get(ptid)}) "
                    f"— creating separate principal track")
                # Fall through to create a new principal track for item_id's own env
            else:
                pid = self.principal_trackid_to_id.get(ptid, principal_item_id)
                if component_track_id:
                    self.component_trackid_to_principal_trackid[
                        component_track_id] = ptid
                if item_id != principal_item_id:
                    self.id_to_principal_trackid[item_id] = ptid
                return ptid, pid, False, component_track_id is not None

        # Create new principal track
        # Use item_id (not principal_item_id) as key when env guard triggered
        track_key = item_id if (
            principal_item_id in self.id_to_principal_trackid
        ) else principal_item_id
        # DETERMINISTIC principal trackId (uuid5 of the principal key) so that a
        # restart — or a shard that failed to preload existing heads — reuses
        # the SAME trackId for the same principal instead of minting a fresh
        # uuid4 orphan head every time. The env-guard case above already keys a
        # separate track on item_id (distinct from principal_item_id), so the
        # two environment-split tracks get distinct deterministic ids without
        # needing the environment in the seed.
        ptid = self._deterministic_principal_trackid(track_key)
        is_known_principal = ptid in self.known_principal_trackids
        self.id_to_principal_trackid[track_key] = ptid
        self.principal_trackid_to_id[ptid] = track_key
        if not is_known_principal:
            self.new_principal_trackids.add(ptid)
            self.known_principal_trackids.add(ptid)
        self.component_trackids_by_principal[ptid] = set()
        if item_id != track_key:
            self.id_to_principal_trackid[item_id] = ptid
        if component_track_id:
            self.component_trackid_to_principal_trackid[
                component_track_id] = ptid
        if environment:
            self.environment_by_track[ptid] = str(environment)
        logger.info(
            f"Created new principal track {ptid} for "
            f"{self.id_field} {track_key}")
        return ptid, track_key, not is_known_principal, component_track_id is not None

    def add_component_track(self, principal_track_id: str,
                            component_track_id: str):
        self.component_trackids_by_principal.setdefault(
            principal_track_id, set()).add(component_track_id)
        self.component_trackid_to_principal_trackid[
            component_track_id] = principal_track_id

    # ------------------------------------------------------------------
    # Kalman filter operations
    # ------------------------------------------------------------------

    def _covariance_intersection(self, pred_mean, pred_covar,
                                  meas_mean, meas_covar):
        """
        Covariance Intersection: fuse two Gaussian estimates without
        assuming statistical independence.

        Finds omega in [0,1] minimizing trace(P_fused), where:
            P_fused^{-1} = omega * P_pred^{-1} + (1-omega) * P_meas^{-1}
            x_fused = P_fused * (omega * P_pred^{-1} * x_pred
                                 + (1-omega) * P_meas^{-1} * x_meas)

        Omega interpretation:
            ~0   → measurement-dominated (prediction covariance >> measurement)
            ~0.5 → balanced (comparable covariances, real fusion)
            ~1   → prediction-dominated (measurement covariance >> prediction)

        Returns (x_fused, P_fused) or (None, None) on failure.
        """
        try:
            P1_inv = np.linalg.inv(pred_covar)
            P2_inv = np.linalg.inv(meas_covar)
        except np.linalg.LinAlgError:
            logger.warning(
                "Singular covariance in CI - falling back to standard update")
            return None, None

        if self.ci_omega is not None:
            best_w = self.ci_omega
        else:
            # Vectorized omega grid search over the POSITION-block trace (state is
            # interleaved [x,vx,y,vy,z,vz]; position indices [0,2,4]). Build all 50
            # candidate fused information matrices at once and invert them with a
            # single batched np.linalg.inv, then argmin the position trace. Batched
            # inv is element-wise identical to the per-omega loop (numpy loops
            # internally over the same LAPACK routine) and argmin returns the first
            # minimum — matching the old strict `t < best_trace` loop — so the
            # result is unchanged, just ~50x fewer Python-level inversions. Positive
            # combinations of the (PD) inverse covariances are non-singular; the
            # rare singular case falls back to the scalar loop.
            _pos = [0, 2, 4]
            ws = np.linspace(0.01, 0.99, 50)
            try:
                M = ws[:, None, None] * P1_inv + (1.0 - ws)[:, None, None] * P2_inv
                P_all = np.linalg.inv(M)
                traces = P_all[:, 0, 0] + P_all[:, 2, 2] + P_all[:, 4, 4]
                k = int(np.argmin(traces))
                best_w = ws[k]
                best_trace = traces[k]
            except np.linalg.LinAlgError:
                best_w = 0.5
                best_trace = np.inf
                for w in ws:
                    try:
                        P_fused = np.linalg.inv(w * P1_inv + (1 - w) * P2_inv)
                        t = sum(P_fused[i, i] for i in _pos)
                        if t < best_trace:
                            best_trace = t
                            best_w = w
                    except np.linalg.LinAlgError:
                        continue
            self._ci_log_counter += 1
            if self._ci_log_counter % 100 == 1:
                logger.info(f"CI omega={best_w:.3f}, pos_trace={best_trace:.1f} (sampled 1/100)")

        try:
            P_fused_inv = best_w * P1_inv + (1 - best_w) * P2_inv
            P_fused = np.linalg.inv(P_fused_inv)
            P_fused = (P_fused + P_fused.T) / 2  # enforce symmetry

            x1 = np.asarray(pred_mean).flatten()
            x2 = np.asarray(meas_mean).flatten()
            x_fused = P_fused @ (best_w * P1_inv @ x1
                                  + (1 - best_w) * P2_inv @ x2)
            return x_fused, P_fused
        except np.linalg.LinAlgError:
            logger.warning(
                "CI fusion failed - falling back to standard update")
            return None, None

    def _reset_filter(self, principal_track_id: str, measurement: Detection):
        for d in (self.priors, self.predictor, self.updater,
                  self.measurement_model):
            d.pop(principal_track_id, None)
        success = self.get_or_create_filter(principal_track_id, measurement)
        logger.warning(
            f"Reset Kalman filter for track {principal_track_id}")
        return success

    def set_fusion_method(self, method: str = 'kalman',
                          omega: Optional[float] = None):
        """Set the fusion method.

        Args:
            method: 'kalman' (standard) or 'ci' (Covariance Intersection).
            omega: Fixed CI weight in (0, 1). None = auto-optimize per update.
                   ~0 = measurement-dominated, ~0.5 = balanced, ~1 = prediction-dominated.
        """
        method = method.lower()
        if method not in ('kalman', 'ci'):
            logger.warning(f"Unknown fusion method '{method}'; defaulting to 'kalman'")
            method = 'kalman'
        self.fusion_method = method
        if omega is not None:
            self.ci_omega = float(np.clip(omega, 0.01, 0.99))
        else:
            self.ci_omega = None
        logger.info(
            f"Fusion method set to '{self.fusion_method}'"
            f"{f' (omega={self.ci_omega})' if self.ci_omega is not None else ' (omega=auto)' if method == 'ci' else ''}")

    def set_track_environment(self, principal_track_id: str,
                               environment: Optional[str]):
        """Associate an environment with a principal track."""
        if environment:
            self.environment_by_track[principal_track_id] = str(environment)

    def get_or_create_filter(self, principal_track_id: str,
                             measurement: Detection) -> bool:
        if principal_track_id not in self.priors:
            if np.isnan(measurement.state_vector).any():
                logger.warning(
                    f"Cannot create filter for {principal_track_id} "
                    f"- NaN in measurement")
                return False

            env = self.environment_by_track.get(principal_track_id)
            q = self._get_q(env)
            transition_model = CombinedLinearGaussianTransitionModel([
                ConstantVelocity(q),
                ConstantVelocity(q),
                ConstantVelocity(q),
            ])

            noise_covar = measurement.measurement_model.noise_covar.copy()
            noise_covar = np.clip(noise_covar, -1e10, 1e10)
            for i in range(noise_covar.shape[0]):
                if noise_covar[i, i] <= 0 or noise_covar[i, i] > 1e10:
                    noise_covar[i, i] = 1e6

            measurement_model = LinearGaussian(
                ndim_state=6, mapping=(0, 1, 2, 3, 4, 5),
                noise_covar=noise_covar)
            predictor = KalmanPredictor(transition_model)
            updater = KalmanUpdater(measurement_model)

            # Use measurement covariance as initial prior so the velocity
            # block reflects the environment-appropriate fallback rather
            # than a blanket 1e6 that would inflate the first prediction.
            init_covar = measurement.measurement_model.noise_covar.copy()
            prior = GaussianState(
                measurement.state_vector.copy(),
                init_covar,
                timestamp=measurement.timestamp)

            self.priors[principal_track_id] = prior
            self.predictor[principal_track_id] = predictor
            self.updater[principal_track_id] = updater
            self.measurement_model[principal_track_id] = measurement_model
            logger.debug(
                f"Created new Kalman filter for principal track "
                f"{principal_track_id}")

        return True

    def _predict(self, principal_track_id: str, current_state: GaussianState,
                 timestamp) -> GaussianState:
        """Kalman time-update. StoneSoup implementation; overridden by
        NumpyPrincipalTrackKalmanFilter with an equivalent numpy kernel."""
        return self.predictor[principal_track_id].predict(
            current_state, timestamp=timestamp)

    def _standard_update(self, principal_track_id: str, prediction: GaussianState,
                         measurement: Detection) -> GaussianState:
        """Kalman measurement-update. StoneSoup implementation; overridden by
        NumpyPrincipalTrackKalmanFilter with an equivalent numpy kernel."""
        hypothesis = SingleHypothesis(prediction, measurement)
        return self.updater[principal_track_id].update(hypothesis)

    def update_filter(self, principal_track_id: str,
                      measurement: Detection) -> Optional[GaussianState]:
        """Predict to measurement time and update with measurement."""
        if not self.get_or_create_filter(principal_track_id, measurement):
            return None

        current_state = self.priors[principal_track_id]
        time_diff = (
            measurement.timestamp - current_state.timestamp).total_seconds()

        if time_diff < 0:
            self.stats['stale_measurements_skipped'] += 1
            logger.warning(
                f"Stale measurement ({abs(time_diff):.1f}s old) for "
                f"{principal_track_id} - skipping")
            return None

        if time_diff > 15 * 60:
            self.stats['time_jump_resets'] += 1
            logger.warning(
                f"Time jump ({time_diff:.1f}s) for "
                f"{principal_track_id} - resetting")
            if self._reset_filter(principal_track_id, measurement):
                return self.priors[principal_track_id]
            return None

        if (np.isnan(current_state.state_vector).any()
                or np.isnan(current_state.covar).any()):
            self.stats['nan_resets'] += 1
            if self._reset_filter(principal_track_id, measurement):
                return self.priors[principal_track_id]
            return None

        try:
            prediction = self._predict(
                principal_track_id, current_state, measurement.timestamp)

            if (np.isnan(prediction.state_vector).any()
                    or np.isnan(prediction.covar).any()):
                self.stats['prediction_nan_resets'] += 1
                if self._reset_filter(principal_track_id, measurement):
                    return self.priors[principal_track_id]
                return None

            if np.max(np.abs(prediction.covar)) > 1e15:
                self.stats['covariance_explosion_resets'] += 1
                if self._reset_filter(principal_track_id, measurement):
                    return self.priors[principal_track_id]
                return None

            # --- Fusion step ---
            pred_pos_trace = sum(prediction.covar[i, i] for i in [0, 2, 4])
            meas_pos_trace = sum(
                measurement.measurement_model.noise_covar[i, i]
                for i in [0, 2, 4])

            if self.fusion_method == 'ci':
                x_fused, P_fused = self._covariance_intersection(
                    np.asarray(prediction.state_vector).flatten(),
                    np.asarray(prediction.covar),
                    np.asarray(measurement.state_vector).flatten(),
                    np.asarray(
                        measurement.measurement_model.noise_covar))
                if x_fused is not None:
                    post = GaussianState(
                        x_fused, P_fused,
                        timestamp=measurement.timestamp)
                else:
                    # CI failed, fall back to standard Kalman
                    post = self._standard_update(
                        principal_track_id, prediction, measurement)
            else:
                post = self._standard_update(
                    principal_track_id, prediction, measurement)

            fused_pos_trace = sum(post.covar[i, i] for i in [0, 2, 4])
            env = self.environment_by_track.get(principal_track_id, '?')
            logger.debug(
                f"KF {str(principal_track_id)[:8]} [{env}] dt={time_diff:.0f}s "
                f"pred_pos_tr={pred_pos_trace:.0f} "
                f"meas_pos_tr={meas_pos_trace:.0f} "
                f"fused_pos_tr={fused_pos_trace:.0f}")

            if (np.isnan(post.state_vector).any()
                    or np.isnan(post.covar).any()):
                self.stats['nan_resets'] += 1
                logger.warning(
                    f"Kalman update produced NaN for {principal_track_id}")
                if self._reset_filter(principal_track_id, measurement):
                    return self.priors[principal_track_id]
                return None

            self.priors[principal_track_id] = post
            self.stats['successful_updates'] += 1
            return post

        except Exception as e:
            self.stats['exception_resets'] += 1
            logger.error(
                f"Exception in Kalman update for "
                f"{principal_track_id}: {e}")
            if self._reset_filter(principal_track_id, measurement):
                return self.priors[principal_track_id]
            return None

    def log_stats(self):
        """Log Kalman filter diagnostic stats."""
        s = self.stats
        total = (s['successful_updates'] + s['nan_resets']
                 + s['time_jump_resets'] + s['covariance_explosion_resets']
                 + s['prediction_nan_resets'] + s['exception_resets'])
        if total > 0:
            rate = s['successful_updates'] / total * 100
            logger.info(
                f"Kalman stats: {s['successful_updates']}/{total} "
                f"successful ({rate:.1f}%) | "
                f"NaN: {s['nan_resets']}, pred NaN: {s['prediction_nan_resets']}, "
                f"covar explosion: {s['covariance_explosion_resets']}, "
                f"time jumps: {s['time_jump_resets']}, "
                f"stale skipped: {s['stale_measurements_skipped']}, "
                f"exceptions: {s['exception_resets']}")


class NumpyPrincipalTrackKalmanFilter(PrincipalTrackKalmanFilter):
    """Drop-in numpy replacement for the StoneSoup Kalman kernel in
    :class:`PrincipalTrackKalmanFilter`.

    Principal-track fusion is a small fixed-size 6-state constant-velocity
    Kalman filter; StoneSoup's per-event object plumbing dominates fuser cost
    (~83% of a fusion batch in production). This subclass overrides ONLY the
    numeric kernel (get_or_create_filter / _predict / _standard_update) with
    plain numpy. All track management, supersede handling, covariance
    intersection, guards, stats and control flow are inherited unchanged, so
    behaviour matches StoneSoup by construction (verified in
    tests/unit/test_fuser_kalman_equivalence.py).

    State is stored as StoneSoup ``GaussianState`` in ``self.priors`` exactly as
    the base class does, so the fuser, preload and writeback paths are
    unaffected. NOTE: unlike the object tracker, the fuser does NOT skip
    out-of-order measurements — it predicts with a SIGNED dt (F uses dt, Q uses
    abs(dt)), matching StoneSoup.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Process-noise PSD fixed per principal track at filter-creation time,
        # mirroring StoneSoup building the transition model once in
        # get_or_create_filter.
        self._q_by_track: Dict[str, float] = {}

    @staticmethod
    def _cv_transition(dt: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
        """Constant-velocity transition F (signed dt) and process noise Q
        (abs dt) for interleaved state [x, vx, y, vy, z, vz] — the block-diagonal
        form of StoneSoup's CombinedLinearGaussianTransitionModel([CV(q)] * 3).
        """
        F = np.eye(6)
        F[0, 1] = F[2, 3] = F[4, 5] = dt
        d = abs(dt)  # StoneSoup ConstantVelocity.covar uses abs(dt)
        qb = q * np.array([[d ** 3 / 3.0, d ** 2 / 2.0],
                           [d ** 2 / 2.0, d]])
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = qb
        Q[2:4, 2:4] = qb
        Q[4:6, 4:6] = qb
        return F, Q

    @staticmethod
    def _measurement_matrix(mapping) -> np.ndarray:
        """Selection matrix H mapping the 6-state vector onto the measured dims."""
        H = np.zeros((len(mapping), 6))
        H[np.arange(len(mapping)), list(mapping)] = 1.0
        return H

    def get_or_create_filter(self, principal_track_id: str,
                             measurement: Detection) -> bool:
        """Create numpy filter state for a new principal track. Mirrors the base
        StoneSoup version: NaN measurement -> no filter (False); initial prior =
        measurement state + measurement covariance."""
        if principal_track_id not in self.priors:
            if np.isnan(measurement.state_vector).any():
                logger.warning(
                    f"Cannot create filter for {principal_track_id} "
                    f"- NaN in measurement")
                return False
            env = self.environment_by_track.get(principal_track_id)
            self._q_by_track[principal_track_id] = self._get_q(env)
            init_covar = np.asarray(
                measurement.measurement_model.noise_covar, dtype=float).copy()
            sv = np.asarray(measurement.state_vector, dtype=float).reshape(-1, 1)
            self.priors[principal_track_id] = GaussianState(
                sv, init_covar, timestamp=measurement.timestamp)
        return True

    def _predict(self, principal_track_id: str, current_state: GaussianState,
                 timestamp) -> GaussianState:
        mean = np.asarray(current_state.state_vector, dtype=float).reshape(-1)
        cov = np.asarray(current_state.covar, dtype=float)
        # SIGNED dt: the fuser does not skip out-of-order measurements, so dt may
        # be negative (backward prediction), matching StoneSoup's predictor.
        dt = (timestamp - current_state.timestamp).total_seconds()
        q = self._q_by_track.get(principal_track_id)
        if q is None:  # preloaded track — derive lazily from its environment
            q = self._q_by_track[principal_track_id] = self._get_q(
                self.environment_by_track.get(principal_track_id))
        F, Q = self._cv_transition(dt, q)
        pred_mean = F @ mean
        pred_cov = F @ cov @ F.T + Q
        return GaussianState(pred_mean.reshape(-1, 1), pred_cov, timestamp=timestamp)

    def _standard_update(self, principal_track_id: str, prediction: GaussianState,
                         measurement: Detection) -> GaussianState:
        pred_mean = np.asarray(prediction.state_vector, dtype=float).reshape(-1)
        pred_cov = np.asarray(prediction.covar, dtype=float)
        z = np.asarray(measurement.state_vector, dtype=float).reshape(-1)
        H = self._measurement_matrix(measurement.measurement_model.mapping)
        R = np.asarray(measurement.measurement_model.noise_covar, dtype=float)
        # Non-Joseph Kalman update, matching stonesoup KalmanUpdater exactly:
        #   S = H P Hᵀ + R;  K = P Hᵀ S⁻¹;  P⁺ = P − K S Kᵀ
        S = H @ pred_cov @ H.T + R
        K = pred_cov @ H.T @ np.linalg.inv(S)
        post_mean = pred_mean + K @ (z - H @ pred_mean)
        post_cov = pred_cov - K @ S @ K.T
        return GaussianState(post_mean.reshape(-1, 1), post_cov,
                             timestamp=measurement.timestamp)

    # ------------------------------------------------------------------
    # Array-native kernel (Stage-2 batched fusion)
    #
    # The methods above still consume a StoneSoup ``Detection`` per event, so
    # the per-event ``create_measurement`` (LinearGaussian + Detection object
    # construction) and pandas ``row.get`` calls dominate fuser cost — exactly
    # what kept the object tracker slow before its batched ``process_measure
    # ments_batch`` path landed (~6.8x). The methods below reproduce the base
    # ``update_filter`` body operating on plain numpy arrays, so a batched
    # fuser loop can bulk-extract columns once and drive the filter without any
    # per-event Detection/LinearGaussian allocation. ``update_filter`` is
    # overridden to route through the same kernel, so the per-row path and the
    # batched path share one code path (validated bit-identical by
    # tests/unit/test_fuser_kalman_equivalence.py and the batch-equivalence
    # tests).
    # ------------------------------------------------------------------
    def _fresh_prior_np(self, principal_track_id: str, state_vec: np.ndarray,
                        noise_covar: np.ndarray, timestamp) -> GaussianState:
        """Array-native prior creation/reset: prior = measurement state +
        measurement covariance (mirrors ``get_or_create_filter`` and
        ``_reset_filter`` for the numpy kernel, which don't clip the covariance
        and keep no StoneSoup predictor/updater state)."""
        sv = np.asarray(state_vec, dtype=float).reshape(-1, 1)
        cov = np.asarray(noise_covar, dtype=float)
        env = self.environment_by_track.get(principal_track_id)
        self._q_by_track[principal_track_id] = self._get_q(env)
        prior = GaussianState(sv, cov, timestamp=timestamp)
        self.priors[principal_track_id] = prior
        return prior

    def _update_core(self, principal_track_id: str, state_vec: np.ndarray,
                     noise_covar: np.ndarray, timestamp) -> Optional[GaussianState]:
        """Array-native equivalent of the base ``update_filter`` body — same
        guard order, thresholds and stats — operating on plain arrays.

        The fuser measurement is ALWAYS the full 6-state (H = I). Stale
        measurements are skipped; large forward time jumps reset the filter.
        Other resets (NaN / covariance explosion / post NaN / exception)
        rebuild the prior from the current measurement and return it.
        """
        m = np.asarray(state_vec, dtype=float).reshape(-1)
        R = np.asarray(noise_covar, dtype=float)

        # get_or_create_filter: a NaN measurement can't seed a filter (the
        # batched caller already skips NaN measurements, but keep the guard so
        # the per-row update_filter path matches the base class).
        if principal_track_id not in self.priors:
            if np.isnan(m).any():
                logger.warning(
                    f"Cannot create filter for {principal_track_id} - NaN in measurement")
                return None
            self._fresh_prior_np(principal_track_id, m, R, timestamp)

        current = self.priors[principal_track_id]
        cur_mean = np.asarray(current.state_vector, dtype=float).reshape(-1)
        cur_cov = np.asarray(current.covar, dtype=float)
        time_diff = (timestamp - current.timestamp).total_seconds()

        if time_diff < 0:
            self.stats['stale_measurements_skipped'] += 1
            logger.warning(
                f"Stale measurement ({abs(time_diff):.1f}s old) for "
                f"{principal_track_id} - skipping")
            return None

        if time_diff > 15 * 60:
            self.stats['time_jump_resets'] += 1
            logger.warning(
                f"Time jump ({time_diff:.1f}s) for {principal_track_id} - resetting")
            return self._fresh_prior_np(principal_track_id, m, R, timestamp)

        if np.isnan(cur_mean).any() or np.isnan(cur_cov).any():
            self.stats['nan_resets'] += 1
            return self._fresh_prior_np(principal_track_id, m, R, timestamp)

        try:
            q = self._q_by_track.get(principal_track_id)
            if q is None:  # preloaded track — derive lazily from its environment
                q = self._q_by_track[principal_track_id] = self._get_q(
                    self.environment_by_track.get(principal_track_id))
            dt_s = time_diff
            F, Q = self._cv_transition(dt_s, q)
            pred_mean = F @ cur_mean
            pred_cov = F @ cur_cov @ F.T + Q

            if np.isnan(pred_mean).any() or np.isnan(pred_cov).any():
                self.stats['prediction_nan_resets'] += 1
                return self._fresh_prior_np(principal_track_id, m, R, timestamp)
            if np.max(np.abs(pred_cov)) > 1e15:
                self.stats['covariance_explosion_resets'] += 1
                return self._fresh_prior_np(principal_track_id, m, R, timestamp)

            if self.fusion_method == 'ci':
                x_fused, P_fused = self._covariance_intersection(
                    pred_mean, pred_cov, m, R)
                if x_fused is not None:
                    post_mean = np.asarray(x_fused, dtype=float).reshape(-1)
                    post_cov = np.asarray(P_fused, dtype=float)
                else:  # CI failed -> standard Kalman fallback
                    H = self._measurement_matrix((0, 1, 2, 3, 4, 5))
                    S = H @ pred_cov @ H.T + R
                    K = pred_cov @ H.T @ np.linalg.inv(S)
                    post_mean = pred_mean + K @ (m - H @ pred_mean)
                    post_cov = pred_cov - K @ S @ K.T
            else:
                H = self._measurement_matrix((0, 1, 2, 3, 4, 5))
                S = H @ pred_cov @ H.T + R
                K = pred_cov @ H.T @ np.linalg.inv(S)
                post_mean = pred_mean + K @ (m - H @ pred_mean)
                post_cov = pred_cov - K @ S @ K.T

            if np.isnan(post_mean).any() or np.isnan(post_cov).any():
                self.stats['nan_resets'] += 1
                logger.warning(
                    f"Kalman update produced NaN for {principal_track_id}")
                return self._fresh_prior_np(principal_track_id, m, R, timestamp)

            post = GaussianState(post_mean.reshape(-1, 1), post_cov, timestamp=timestamp)
            self.priors[principal_track_id] = post
            self.stats['successful_updates'] += 1
            return post
        except Exception as e:
            self.stats['exception_resets'] += 1
            logger.error(
                f"Exception in Kalman update for {principal_track_id}: {e}")
            return self._fresh_prior_np(principal_track_id, m, R, timestamp)

    def update_filter(self, principal_track_id: str,
                      measurement: Detection) -> Optional[GaussianState]:
        """Route the per-row path through the same array-native kernel used by
        the batched fuser loop (extracts arrays from the StoneSoup Detection),
        so both paths are one code path."""
        return self._update_core(
            principal_track_id,
            np.asarray(measurement.state_vector, dtype=float).reshape(-1),
            np.asarray(measurement.measurement_model.noise_covar, dtype=float),
            measurement.timestamp,
        )

    def extract_measurement_columns(self, event_df: Any) -> Dict[str, Any]:
        """Bulk-extract every column needed to build measurements as numpy
        arrays once without taking a pandas dependency.

        The records may be a dataframe-like object or plain list-of-dicts; we
        normalize to a list of dicts and then pull columns by key.
        """
        rows = _iter_named_rows(event_df)
        n = len(rows)
        cols = set()
        for row in rows:
            cols.update(row.keys())

        def col(name, default):
            out = []
            for row in rows:
                val = _record_get(row, name, default)
                out.append(val)
            return np.asarray(out, dtype=object)

        vrc = self.vel_read_cols
        vrf = self.vel_read_fallback
        prepared: Dict[str, Any] = {
            'n': n,
            'px': col('ecefPosition.x', 0.0),
            'py': col('ecefPosition.y', 0.0),
            'pz': col('ecefPosition.z', 0.0),
            'v': [col(vrc[a], None) for a in range(3)],
            'has_v': [vrc[a] in cols for a in range(3)],
            'vf': ([col(vrf[a], 0.0) for a in range(3)] if vrf else None),
            'has_vf': ([vrf[a] in cols for a in range(3)] if vrf else None),
            'pos_cov_present': 'positionCovariance.xx' in cols,
            'pcxx': col('positionCovariance.xx', np.nan),
            'pcxy': col('positionCovariance.xy', 0.0),
            'pcxz': col('positionCovariance.xz', 0.0),
            'pcyy': col('positionCovariance.yy', 1e7),
            'pcyz': col('positionCovariance.yz', 0.0),
            'pczz': col('positionCovariance.zz', 1e7),
            'vel_cov_present': 'velocityCovariance.dxdx' in cols,
            'vcxx': col('velocityCovariance.dxdx', np.nan),
            'vcxy': col('velocityCovariance.dxdy', 0.0),
            'vcxz': col('velocityCovariance.dxdz', 0.0),
            'vcyy': col('velocityCovariance.dydy', 1e7),
            'vcyz': col('velocityCovariance.dydz', 0.0),
            'vczz': col('velocityCovariance.dzdz', 1e7),
        }
        prepared['ts'] = self._parse_measurement_timestamps(rows)
        return prepared

    @staticmethod
    def _parse_measurement_timestamps(rows: List[dict]) -> List[Any]:
        """Bulk timestamp resolution using record dicts only."""
        raw = []
        for row in rows:
            v = _record_get(row, 'interceptTimestamp')
            if v is None or (isinstance(v, str) and not v.strip()):
                v = _record_get(row, 'trackUpdatedTimestamp')
            if v is None or (isinstance(v, str) and not v.strip()):
                v = _record_get(row, 'crucibleHeader.createdDate')
            raw.append(v)
        parsed: List[Any] = []
        for value in raw:
            if value is None or (isinstance(value, str) and not value.strip()):
                parsed.append(dt.now(tz=tz.utc))
                continue
            try:
                dt_value = dt.strptime(str(value), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=tz.utc)
            except Exception:
                try:
                    dt_value = dt.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=tz.utc)
                except Exception:
                    dt_value = dt.now(tz=tz.utc)
            parsed.append(dt_value)
        return parsed

    def measurement_at(self, prepared: Dict[str, Any], i: int,
                       vel_var_fallback: float):
        """Reproduce ``create_measurement_from_row`` for row ``i`` from the
        bulk-extracted arrays: returns ``(state_vec (6,), noise_covar (6,6),
        timestamp)`` or ``(None, None, None)`` when the state contains NaN (the
        per-row path returns ``None`` from ``create_measurement`` and skips)."""
        # --- state vector [x, vx, y, vy, z, vz] with velocity fallback ---
        def _vel(axis):
            v = prepared['v'][axis][i] if prepared['has_v'][axis] else None
            if v is None or (isinstance(v, float) and v != v):
                if prepared['vf'] is not None:
                    v = prepared['vf'][axis][i] if prepared['has_vf'][axis] else 0
            return float(v or 0)

        sv = np.array([
            float(prepared['px'][i]), _vel(0),
            float(prepared['py'][i]), _vel(1),
            float(prepared['pz'][i]), _vel(2),
        ], dtype=float)
        if np.isnan(sv).any():
            return None, None, None

        # --- interleaved 6x6 covariance (mirror _build_covariance_from_row) ---
        pcxx = prepared['pcxx'][i]
        has_pos = prepared['pos_cov_present'] and not (
            pcxx is None or (isinstance(pcxx, float) and pcxx != pcxx))
        if has_pos:
            p00 = float(pcxx); p01 = float(prepared['pcxy'][i]); p02 = float(prepared['pcxz'][i])
            p11 = float(prepared['pcyy'][i]); p12 = float(prepared['pcyz'][i]); p22 = float(prepared['pczz'][i])
        else:
            p00 = p11 = p22 = 1e7
            p01 = p02 = p12 = 0.0

        vcxx = prepared['vcxx'][i]
        has_vel = prepared['vel_cov_present'] and not (
            vcxx is None or (isinstance(vcxx, float) and vcxx != vcxx))
        if has_vel:
            v00 = float(vcxx); v01 = float(prepared['vcxy'][i]); v02 = float(prepared['vcxz'][i])
            v11 = float(prepared['vcyy'][i]); v12 = float(prepared['vcyz'][i]); v22 = float(prepared['vczz'][i])
        else:
            v00 = v11 = v22 = float(vel_var_fallback)
            v01 = v02 = v12 = 0.0

        cov = np.zeros((6, 6))
        cov[0, 0] = p00; cov[2, 2] = p11; cov[4, 4] = p22
        cov[0, 2] = cov[2, 0] = p01; cov[0, 4] = cov[4, 0] = p02; cov[2, 4] = cov[4, 2] = p12
        cov[1, 1] = v00; cov[3, 3] = v11; cov[5, 5] = v22
        cov[1, 3] = cov[3, 1] = v01; cov[1, 5] = cov[5, 1] = v02; cov[3, 5] = cov[5, 3] = v12
        return sv, cov, prepared['ts'][i]


# ------------------------------------------------------------------
# Standalone helpers
# ------------------------------------------------------------------

def _build_covariance_from_row(row: Any,
                               vel_var_fallback: float = 1e4,
                               ) -> np.ndarray:
    """Build interleaved [x,vx,y,vy,z,vz] 6x6 covariance from row data."""
    has_vel_cov = _record_has(row, 'velocityCovariance.dxdx') and _notna(_record_get(row, 'velocityCovariance.dxdx'))

    if _record_has(row, 'positionCovariance.xx') and _notna(_record_get(row, 'positionCovariance.xx')):
        noise_pos = np.array([
            [_record_get(row, 'positionCovariance.xx', 1e7),
             _record_get(row, 'positionCovariance.xy', 0),
             _record_get(row, 'positionCovariance.xz', 0)],
            [_record_get(row, 'positionCovariance.xy', 0),
             _record_get(row, 'positionCovariance.yy', 1e7),
             _record_get(row, 'positionCovariance.yz', 0)],
            [_record_get(row, 'positionCovariance.xz', 0),
             _record_get(row, 'positionCovariance.yz', 0),
             _record_get(row, 'positionCovariance.zz', 1e7)],
        ])
    else:
        noise_pos = np.diag([1e7, 1e7, 1e7])

    if has_vel_cov:
        noise_vel = np.array([
            [_record_get(row, 'velocityCovariance.dxdx', 1e7),
             _record_get(row, 'velocityCovariance.dxdy', 0),
             _record_get(row, 'velocityCovariance.dxdz', 0)],
            [_record_get(row, 'velocityCovariance.dxdy', 0),
             _record_get(row, 'velocityCovariance.dydy', 1e7),
             _record_get(row, 'velocityCovariance.dydz', 0)],
            [_record_get(row, 'velocityCovariance.dxdz', 0),
             _record_get(row, 'velocityCovariance.dydz', 0),
             _record_get(row, 'velocityCovariance.dzdz', 1e7)],
        ])
    else:
        noise_vel = np.diag([vel_var_fallback, vel_var_fallback, vel_var_fallback])

    covar = np.zeros((6, 6))
    covar[0, 0] = noise_pos[0, 0]
    covar[2, 2] = noise_pos[1, 1]
    covar[4, 4] = noise_pos[2, 2]
    covar[0, 2] = covar[2, 0] = noise_pos[0, 1]
    covar[0, 4] = covar[4, 0] = noise_pos[0, 2]
    covar[2, 4] = covar[4, 2] = noise_pos[1, 2]
    covar[1, 1] = noise_vel[0, 0]
    covar[3, 3] = noise_vel[1, 1]
    covar[5, 5] = noise_vel[2, 2]
    covar[1, 3] = covar[3, 1] = noise_vel[0, 1]
    covar[1, 5] = covar[5, 1] = noise_vel[0, 2]
    covar[3, 5] = covar[5, 3] = noise_vel[1, 2]
    return covar


def create_measurement_from_row(
        row: Any,
        vel_read_cols: Tuple[str, str, str],
        vel_read_fallback: Tuple[str, str, str] = None,
        vel_var_fallback: float = 1e4,
) -> Optional[Detection]:
    """Create a StoneSoup Detection from a track event row."""
    try:
        noise_covar = _build_covariance_from_row(row, vel_var_fallback=vel_var_fallback)
        measurement_model = LinearGaussian(
            ndim_state=6, mapping=(0, 1, 2, 3, 4, 5),
            noise_covar=noise_covar)

        ts = (_record_get(row, 'interceptTimestamp')
              or _record_get(row, 'trackUpdatedTimestamp')
              or _record_get(row, 'crucibleHeader.createdDate'))
        try:
            timestamp = dt.strptime(
                str(ts), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=tz.utc)
        except Exception:
            timestamp = dt.now(tz=tz.utc)

        def _vel(axis):
            v = _record_get(row, vel_read_cols[axis], None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                if vel_read_fallback:
                    v = _record_get(row, vel_read_fallback[axis], 0)
            return float(v or 0)

        state_vector = np.array([
            _record_get(row, 'ecefPosition.x', 0), _vel(0),
            _record_get(row, 'ecefPosition.y', 0), _vel(1),
            _record_get(row, 'ecefPosition.z', 0), _vel(2),
        ])

        if np.isnan(state_vector).any():
            logger.warning("Measurement contains NaN - skipping")
            return None

        return Detection(
            state_vector, timestamp=timestamp,
            measurement_model=measurement_model)
    except Exception as e:
        logger.error(f"Error creating measurement: {e}")
        return None


def build_principal_state_dict(
        post: GaussianState,
        principal_track_id: str,
        principal_id: str,
        id_field: str,
        vel_write_cols: Tuple[str, str, str],
) -> Dict[str, Any]:
    """Build a dict of kinematic fields from Kalman filter state."""
    return {
        id_field: principal_id,
        'trackId': principal_track_id,

        'ecefPosition.x': float(post.state_vector[0]),
        'ecefPosition.y': float(post.state_vector[2]),
        'ecefPosition.z': float(post.state_vector[4]),
        vel_write_cols[0]: float(post.state_vector[1]),
        vel_write_cols[1]: float(post.state_vector[3]),
        vel_write_cols[2]: float(post.state_vector[5]),

        'positionCovariance.xx': float(post.covar[0, 0]),
        'positionCovariance.xy': float(post.covar[0, 2]),
        'positionCovariance.xz': float(post.covar[0, 4]),
        'positionCovariance.yy': float(post.covar[2, 2]),
        'positionCovariance.yz': float(post.covar[2, 4]),
        'positionCovariance.zz': float(post.covar[4, 4]),
        'velocityCovariance.dxdx': float(post.covar[1, 1]),
        'velocityCovariance.dxdy': float(post.covar[1, 3]),
        'velocityCovariance.dxdz': float(post.covar[1, 5]),
        'velocityCovariance.dydy': float(post.covar[3, 3]),
        'velocityCovariance.dydz': float(post.covar[3, 5]),
        'velocityCovariance.dzdz': float(post.covar[5, 5]),
    }
