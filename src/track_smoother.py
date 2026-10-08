"""
Lightweight RTS (Rauch-Tung-Striebel) smoother for track histories.

Uses filterpy's KalmanFilter to forward-filter then backward-smooth
a 6-state constant-velocity model [x, y, z, vx, vy, vz] — the standard
filterpy state layout with positions grouped, then velocities.

Usage:
    from cruciblelib.track_smoother import smooth_track

    smoothed_pos, smoothed_vel = smooth_track(timestamps, positions)
    # or with velocity observations:
    smoothed_pos, smoothed_vel = smooth_track(timestamps, positions, velocities)
"""

import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime as dt

from filterpy.kalman import KalmanFilter


def _epoch(t) -> float:
    """Convert a datetime to seconds since epoch."""
    if hasattr(t, 'timestamp'):
        return t.timestamp()
    if t.tzinfo is not None:
        t = t.replace(tzinfo=None)
    return (t - dt(1970, 1, 1)).total_seconds()


def smooth_track(
    timestamps: List[dt],
    positions: List[Tuple[float, float, float]],
    velocities: Optional[List[Tuple[float, float, float]]] = None,
    process_noise_q: float = 1.0,
    measurement_noise_pos: float = 100.0,
    measurement_noise_vel: float = 10.0,
    max_gap_seconds: float = 900.0,
    return_covariances: bool = False,
    measurement_covariances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Kalman-smooth a track history and return smoothed positions & velocities.

    Args:
        timestamps: List of datetime timestamps (must be sortable).
        positions: List of (x, y, z) ECEF positions in meters.
        velocities: Optional list of (vx, vy, vz) ECEF velocities in m/s.
                    If None or all zeros, position-only observations are used.
        process_noise_q: Process noise spectral density (m^2/s^3).
        measurement_noise_pos: Position measurement noise std-dev (meters).
                               Used as a fallback when *measurement_covariances*
                               is not provided.
        measurement_noise_vel: Velocity measurement noise std-dev (m/s).
        max_gap_seconds: Maximum time gap before resetting the filter.
        return_covariances: If True, also return smoothed 3x3 position
                            covariance matrices (N, 3, 3).
        measurement_covariances: Optional (N, 3, 3) array of per-measurement
                                 position covariance matrices (ECEF).  When
                                 provided, each R matrix is built from the
                                 corresponding entry instead of the scalar
                                 *measurement_noise_pos*.

    Returns:
        (smoothed_positions, smoothed_velocities) — each shape (N, 3).
        If *return_covariances* is True, returns
        (smoothed_positions, smoothed_velocities, position_covariances)
        where position_covariances has shape (N, 3, 3).
        On failure or < 2 points, returns the raw inputs as arrays.
    """
    n = len(timestamps)
    pos_arr = np.array(positions, dtype=float)
    if n < 2:
        vel_arr = np.array(velocities, dtype=float) if velocities else np.zeros_like(pos_arr)
        if return_covariances:
            default_cov = np.stack([np.eye(3) * measurement_noise_pos**2] * n)
            return pos_arr, vel_arr, default_cov
        return pos_arr, vel_arr

    # Sort by time
    order = np.argsort([_epoch(t) for t in timestamps])
    timestamps = [timestamps[i] for i in order]
    pos_arr = pos_arr[order]
    has_vel = (velocities is not None and
               any(np.linalg.norm(v) > 0.1 for v in velocities))
    vel_arr = np.array(velocities, dtype=float)[order] if has_vel else None

    # Reorder per-measurement covariances if provided
    meas_covs = None
    if measurement_covariances is not None:
        mc = np.asarray(measurement_covariances, dtype=float)
        if mc.shape == (n, 3, 3):
            meas_covs = mc[order]

    dim_z = 6 if has_vel else 3

    # Find segment boundaries (split on large gaps)
    seg_starts = [0]
    for i in range(1, n):
        gap = abs(_epoch(timestamps[i]) - _epoch(timestamps[i - 1]))
        if gap > max_gap_seconds:
            seg_starts.append(i)
    seg_starts.append(n)

    # Output arrays
    smooth_pos = np.empty((n, 3), dtype=float)
    smooth_vel = np.empty((n, 3), dtype=float)
    smooth_cov = np.empty((n, 3, 3), dtype=float) if return_covariances else None

    for seg_idx in range(len(seg_starts) - 1):
        s = seg_starts[seg_idx]
        e = seg_starts[seg_idx + 1]
        seg_len = e - s
        if seg_len < 2:
            smooth_pos[s:e] = pos_arr[s:e]
            smooth_vel[s:e] = vel_arr[s:e] if has_vel else 0.0
            if smooth_cov is not None:
                for k in range(s, e):
                    smooth_cov[k] = np.eye(3) * measurement_noise_pos**2
            continue

        _smooth_segment(
            timestamps[s:e], pos_arr[s:e],
            vel_arr[s:e] if has_vel else None,
            has_vel, dim_z,
            process_noise_q, measurement_noise_pos, measurement_noise_vel,
            smooth_pos[s:e], smooth_vel[s:e],
            out_pos_cov=smooth_cov[s:e] if smooth_cov is not None else None,
            meas_pos_covs=meas_covs[s:e] if meas_covs is not None else None,
        )

    if return_covariances:
        return smooth_pos, smooth_vel, smooth_cov
    return smooth_pos, smooth_vel


def smooth_track_states(
    timestamps: List[dt],
    positions: List[Tuple[float, float, float]],
    velocities: Optional[List[Tuple[float, float, float]]] = None,
    process_noise_q: float = 1.0,
    measurement_noise_pos: float = 100.0,
    measurement_noise_vel: float = 10.0,
    max_gap_seconds: float = 900.0,
    measurement_covariances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RTS-smooth a track and return the full 6-state smoothed estimates.

    Unlike :func:`smooth_track` (which returns only the position/velocity
    means and the 3x3 position covariance), this returns the full smoothed
    state and its 6x6 covariance at each measurement time.  These can be used
    to evaluate the smoothed track at arbitrary query times via forward/backward
    Kalman prediction — giving smoother-quality interpolation/extrapolation
    instead of a one-sided constant-velocity extrapolation.

    Args:
        Same as :func:`smooth_track`.

    Returns:
        (epochs, means, covs) where
          epochs : (N,) float array of measurement times (epoch seconds, sorted)
          means  : (N, 6) smoothed state [x, y, z, vx, vy, vz]
          covs   : (N, 6, 6) smoothed state covariance
    """
    n = len(timestamps)
    pos_arr = np.array(positions, dtype=float)
    epochs = np.array([_epoch(t) for t in timestamps], dtype=float)

    order = np.argsort(epochs)
    epochs = epochs[order]
    pos_arr = pos_arr[order]
    ts_sorted = [timestamps[i] for i in order]

    default_state_cov = np.diag(
        [measurement_noise_pos**2] * 3 + [measurement_noise_vel**2] * 3)

    if n < 2:
        means = np.zeros((n, 6))
        if n == 1:
            means[0, 0:3] = pos_arr[0]
            if velocities is not None and len(velocities) == 1:
                means[0, 3:6] = np.array(velocities, dtype=float)[0]
        covs = (np.stack([default_state_cov] * n) if n > 0
                else np.empty((0, 6, 6)))
        return epochs, means, covs

    has_vel = (velocities is not None and
               any(np.linalg.norm(v) > 0.1 for v in velocities))
    vel_arr = np.array(velocities, dtype=float)[order] if has_vel else None

    meas_covs = None
    if measurement_covariances is not None:
        mc = np.asarray(measurement_covariances, dtype=float)
        if mc.shape == (n, 3, 3):
            meas_covs = mc[order]

    dim_z = 6 if has_vel else 3

    # Split on large gaps (same policy as smooth_track)
    seg_starts = [0]
    for i in range(1, n):
        if abs(epochs[i] - epochs[i - 1]) > max_gap_seconds:
            seg_starts.append(i)
    seg_starts.append(n)

    means = np.empty((n, 6))
    covs = np.empty((n, 6, 6))

    for seg_idx in range(len(seg_starts) - 1):
        s = seg_starts[seg_idx]
        e = seg_starts[seg_idx + 1]
        if e - s < 2:
            means[s:e, 0:3] = pos_arr[s:e]
            means[s:e, 3:6] = vel_arr[s:e] if has_vel else 0.0
            for k in range(s, e):
                covs[k] = default_state_cov
            continue

        _smooth_segment(
            ts_sorted[s:e], pos_arr[s:e],
            vel_arr[s:e] if has_vel else None,
            has_vel, dim_z,
            process_noise_q, measurement_noise_pos, measurement_noise_vel,
            np.empty((e - s, 3)), np.empty((e - s, 3)),
            meas_pos_covs=meas_covs[s:e] if meas_covs is not None else None,
            out_state=means[s:e], out_state_cov=covs[s:e],
        )

    return epochs, means, covs


def _smooth_segment(
    timestamps, pos_arr, vel_arr, has_vel, dim_z,
    process_noise_q, measurement_noise_pos, measurement_noise_vel,
    out_pos, out_vel, *, out_pos_cov=None, meas_pos_covs=None,
    out_state=None, out_state_cov=None,
):
    """Forward-filter then RTS-smooth a single contiguous segment.

    If *meas_pos_covs* is provided (shape (seg_len, 3, 3)), each
    measurement's R matrix uses the corresponding 3×3 position covariance
    instead of the scalar *measurement_noise_pos*.

    If *out_state* (seg_len, 6) and/or *out_state_cov* (seg_len, 6, 6) are
    provided, the full smoothed 6-state means and covariances are written to
    them (used for smoother-based evaluation at arbitrary query times).
    """
    seg_len = len(timestamps)

    # Build filter
    kf = KalmanFilter(dim_x=6, dim_z=dim_z)
    # State: [x, y, z, vx, vy, vz]
    if has_vel:
        kf.H = np.eye(6)
    else:
        kf.H = np.zeros((3, 6))
        kf.H[0, 0] = 1.0  # x
        kf.H[1, 1] = 1.0  # y
        kf.H[2, 2] = 1.0  # z

    # Helper to build R from a per-point position covariance (or fallback)
    def _build_R(idx):
        if meas_pos_covs is not None:
            R_pos = meas_pos_covs[idx]
        else:
            R_pos = np.eye(3) * measurement_noise_pos**2
        if has_vel:
            R = np.zeros((6, 6))
            R[0:3, 0:3] = R_pos
            R[3:6, 3:6] = np.eye(3) * measurement_noise_vel**2
            return R
        return R_pos

    # Initial state
    p0 = pos_arr[0]
    if has_vel:
        v0 = vel_arr[0]
    else:
        dt0 = max(_epoch(timestamps[1]) - _epoch(timestamps[0]), 0.01)
        v0 = (pos_arr[1] - pos_arr[0]) / dt0
    kf.x = np.array([p0[0], p0[1], p0[2], v0[0], v0[1], v0[2]], dtype=float)

    R0 = _build_R(0)
    R0_pos_var = R0[0, 0] if has_vel else R0[0, 0]
    kf.P = np.diag([R0_pos_var] * 3 + [measurement_noise_vel**2] * 3)

    # Measurement noise — set initial R (will be overwritten per-step)
    kf.R = R0

    # Forward pass — collect means and covariances for RTS
    means = np.empty((seg_len, 6))
    covs = np.empty((seg_len, 6, 6))
    means[0] = kf.x.ravel()
    covs[0] = kf.P.copy()

    for i in range(1, seg_len):
        dt_sec = max(_epoch(timestamps[i]) - _epoch(timestamps[i - 1]), 0.001)
        kf.F = np.eye(6)
        kf.F[0, 3] = dt_sec
        kf.F[1, 4] = dt_sec
        kf.F[2, 5] = dt_sec
        kf.Q = _process_noise_matrix(dt_sec, process_noise_q)

        kf.predict()
        kf.R = _build_R(i)
        if has_vel:
            p, v = pos_arr[i], vel_arr[i]
            z = np.array([p[0], p[1], p[2], v[0], v[1], v[2]])
        else:
            z = pos_arr[i]
        kf.update(z)
        means[i] = kf.x.ravel()
        covs[i] = kf.P.copy()

    # RTS backward pass
    smooth_means = means.copy()
    smooth_covs = covs.copy()
    for i in range(seg_len - 2, -1, -1):
        dt_sec = max(_epoch(timestamps[i + 1]) - _epoch(timestamps[i]), 0.001)
        F = np.eye(6)
        F[0, 3] = dt_sec
        F[1, 4] = dt_sec
        F[2, 5] = dt_sec
        Q = _process_noise_matrix(dt_sec, process_noise_q)

        P_pred = F @ covs[i] @ F.T + Q
        K = covs[i] @ F.T @ np.linalg.inv(P_pred)
        smooth_means[i] = means[i] + K @ (smooth_means[i + 1] - F @ means[i])
        smooth_covs[i] = covs[i] + K @ (smooth_covs[i + 1] - P_pred) @ K.T

    out_pos[:] = smooth_means[:, 0:3]
    out_vel[:] = smooth_means[:, 3:6]
    if out_pos_cov is not None:
        # Extract the 3x3 position block from the 6x6 smoothed covariance
        out_pos_cov[:] = smooth_covs[:, 0:3, 0:3]
    if out_state is not None:
        out_state[:] = smooth_means
    if out_state_cov is not None:
        out_state_cov[:] = smooth_covs


def _process_noise_matrix(dt_sec: float, q: float) -> np.ndarray:
    """Build the 6x6 process noise matrix for constant-velocity model."""
    # Grouped state [x,y,z, vx,vy,vz] — block form:
    # [[dt^3/3 * I,  dt^2/2 * I],
    #  [dt^2/2 * I,  dt     * I]] * q
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = np.eye(3) * (dt_sec**3 / 3.0) * q
    Q[0:3, 3:6] = np.eye(3) * (dt_sec**2 / 2.0) * q
    Q[3:6, 0:3] = np.eye(3) * (dt_sec**2 / 2.0) * q
    Q[3:6, 3:6] = np.eye(3) * dt_sec * q
    return Q
