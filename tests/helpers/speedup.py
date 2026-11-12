# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
# SPDX-License-Identifier: GPL-3.0-only
"""Seeded fixtures and comparison helpers for the cam-IMU speed-up oracle tests."""
import numpy as np
import gtsam

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.calibration.factors.imu_segment import ImuSegment
from mvis.imu import CALYX, KALIBR, RPNG, create_imu_model

from tests._reference import (ref_camera_imu_factor, ref_camera_interp_factor,
                              ref_imu_segment)


def _heights_kw(pids, n_bp):
    """Factor kwargs for the board-heights key: {} when the case has none."""
    return {} if pids is None else dict(board_pids=pids, n_board_points=n_bp)

_G = np.array([0.0, 0.0, -9.81])
_BG = np.array([0.003, -0.002, 0.001])
_BA = np.array([-0.13, 0.11, -0.3])
_HZ = 200.0
_W, _H = 640, 480
_CAM_MODELS = {
    "equidistant": [400., 405., 320., 240., 0.01, -0.002, 0.0005, 0.0],
    "radtan":      [400., 405., 320., 240., -0.05, 0.01, 0.001, -0.001],
    "kb4":         [350., 350., 320., 240., 0.02, -0.01, 3e-3, -5e-4],
}

_TRS = (0.0, 1e-13, 0.02)
_N_BP = 200                                   # board points (heights key length)


def _eq(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


# pinhole models whose jac=False projection gained the z <= 0 guard after the oracles froze
GUARDED = ("equidistant", "radtan")


def oracle_error(cam_model, ref_error, ref_jac_error):
    """What a live factor's `error()` must equal: the oracle's, except pinhole corners with z <= 0.

    The oracles are frozen from when the equidistant and radtan error paths had no z check and
    mirrored such a corner to a finite residual, while their Jacobian paths zeroed it. Since then
    both paths zero it (equidistant #25, radtan cleanup F1), so those rows, the ones the oracle's
    own Jacobian path zeroes, are zero here too. Everything else stays the frozen value."""
    e = np.array(ref_error, float)
    if cam_model in GUARDED:
        behind = np.all(np.asarray(ref_jac_error, float).reshape(-1, 2) == 0.0, axis=1)
        e.reshape(-1, 2)[behind] = 0.0
    return e


def oracle_norms(cam_model, ref_norms, ref_jac_error):
    """As `oracle_error`, for `corner_norms`: those corners are NaN, the guard having skipped them."""
    n = np.array(ref_norms, float)
    if cam_model in GUARDED:
        n[np.all(np.asarray(ref_jac_error, float).reshape(-1, 2) == 0.0, axis=1)] = np.nan
    return n


def assert_close(new, ref, exact=False, rtol=1e-12, atol_err=1e-12, name=""):
    """Compare a kernel output with its oracle.

    `new` / `ref` are either arrays, or `(err, [H_0, H_1, ...])` pairs as the
    factors' `error_and_jacobians` return them. `exact=True` -> `np.array_equal`
    (NaN-aware). Otherwise: residual |Δ| <= atol_err (px); each H block
    |Δ| <= rtol * max|ref block| (an all-zero reference block must stay all
    zero); every exactly-zero ROW of a reference block (invalid corner) must be
    exactly zero in `new`; NaN patterns must match."""
    if isinstance(ref, tuple):
        e_new, H_new = new
        e_ref, H_ref = ref
        assert len(H_new) == len(H_ref), f"{name}: {len(H_new)} vs {len(H_ref)} blocks"
        assert_close(e_new, e_ref, exact, rtol, atol_err, f"{name} err")
        for k, (A, B) in enumerate(zip(H_new, H_ref)):
            A = np.asarray(A, float); B = np.asarray(B, float)
            assert A.shape == B.shape, f"{name} H[{k}] shape {A.shape} vs {B.shape}"
            if exact:
                assert _eq(A, B), f"{name} H[{k}] not bitwise, max |Δ| {np.abs(A - B).max()}"
                continue
            zero_rows = ~np.any(B != 0.0, axis=1)
            assert not np.any(A[zero_rows] != 0.0), f"{name} H[{k}] zero rows not exact"
            scale = np.abs(B).max()
            if scale == 0.0:
                assert not np.any(A != 0.0), f"{name} H[{k}] zero block not exact"
            else:
                d = np.abs(A - B).max()
                assert d <= rtol * scale, f"{name} H[{k}] |Δ| {d} > {rtol} * {scale}"
        return
    A = np.asarray(new, float); B = np.asarray(ref, float)
    assert A.shape == B.shape, f"{name} shape {A.shape} vs {B.shape}"
    if exact:
        assert _eq(A, B), f"{name} not bitwise, max |Δ| {np.nanmax(np.abs(A - B)) if A.size else 0}"
        return
    assert np.array_equal(np.isnan(A), np.isnan(B)), f"{name} NaN pattern differs"
    m = ~np.isnan(B)
    if m.any():
        d = np.abs(A[m] - B[m]).max()
        assert d <= atol_err, f"{name} |Δ| {d} > {atol_err}"


def imu_signal(T=0.6, hz=_HZ, b_g=_BG):
    """Smooth body rate + specific force at realistic stamps, with a near-still
    stretch (|ω_corrected| = 1e-3 rad/s, samples 60-80) and an exactly-still one
    (w_m == b_g, samples 90-100) so the identity-intrinsics propagation hits
    `integrate_RAJ`'s small_w and |ω| < 1e-8 branches."""
    t = np.arange(0.0, T, 1.0 / hz) + 1000.0
    s = t - t[0]
    # |ω| up to ~3.3 rad/s so that |ω| dt crosses _SMALL (0.5 deg) at 200 Hz:
    # both the small_th and the full branch of integrate_RAJ get hit.
    w = np.column_stack([2.6 * np.sin(7 * s), -1.8 * np.cos(5 * s) + 0.3,
                         1.1 * np.sin(3 * s + 1)])
    a = np.column_stack([6.0 * np.sin(4 * s), 9.81 + 2 * np.cos(6 * s),
                         -3.0 * np.sin(5 * s + 0.4)])
    w[60:80] = b_g + 1e-3 * np.array([0.6, -0.8, 0.0])
    w[90:100] = b_g
    return t, w, a


def imu_intr(model, rng=None):
    """Non-identity intrinsics per model (own identity layout; a non-identity
    R for the models that carry one)."""
    rng = np.random.default_rng(11) if rng is None else rng
    m = create_imu_model(model)
    d_dw = np.array([0.02, 0.01, -0.02, -0.01, -0.03, 0.015, 0.02, -0.01, 0.01])
    d_da = np.array([0.01, 0.02, -0.01, -0.01, 0.015, 0.02])
    x_tg = np.array([0.002, -0.001, 0.0005, 0.001, 0.003, -0.002, 0.0, 0.001, -0.001])
    R = gtsam.Rot3.Rodrigues(0.01, -0.02, 0.015).matrix() if m.has_R else None
    return create_imu_model(model, x_dw=m.x_dw + d_dw[:len(m.x_dw)],
                            x_da=m.x_da + d_da, x_tg=x_tg, R=R)


def propagation_targets(t, t_anchor, rng):
    """150 signed Δ (s) spanning >= 8 raw samples per direction, with the
    special cases the plan lists. Returns (deltas, n_expected_on_checkpoint)."""
    d = rng.uniform(-0.05, 0.16, 150)
    special = [0.0,
               t[75] - t_anchor,                 # exactly on a raw sample, forward
               t[66] - t_anchor,                 # exactly on a raw sample, backward
               t[76] - t_anchor + 1e-13,         # within 1e-13 after a checkpoint
               t[77] - t_anchor - 1e-13,         # within 1e-13 before a checkpoint
               1e-13, -1e-13]                    # just off the anchor, both signs
    d[:len(special)] = special
    return d


def board_corners(n, rng, nan_corner=True):
    """n board corners (m) on a 0.6 m board; for n >= 12 corner 5 sits 3 m
    along the board normal (behind the camera for the fixture's board pose)
    and corner 7 is NaN (an out-of-range corner for every model)."""
    p_B = np.column_stack([rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n),
                           np.zeros(n)])
    if n >= 12:
        p_B[5] = [0.0, 0.0, 3.0]
        if nan_corner:
            p_B[7] = [np.nan, np.nan, 0.0]
    return p_B

_T_BOARD = gtsam.Pose3(gtsam.Rot3.Rodrigues(3.1, 0.1, 0.0), gtsam.Point3(0.0, 0.0, 1.5))
_T_CTOI = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.02, -0.01, 0.03), gtsam.Point3(0.05, -0.02, 0.01))


def cam_base(cam_model, tr):
    return CameraCalib(_T_CTOI, 0.004, tr, _CAM_MODELS[cam_model], model=cam_model)

# (gyro, dw_column, frame_dt, heights, rolling_mode, dt_anchor, imu model)
_CAMIMU_VARIANTS = [
    ("calibrated", True, True, True, 0, 0.012, CALYX),
    ("raw", True, False, False, 1, -0.009, CALYX),
    ("calibrated", False, True, False, 1, 0.033, KALIBR),
    ("calibrated", True, False, True, 0, -0.021, RPNG),
]


def camimu_case(cam_model, tr, n, variant, rng):
    """Build (live factor, oracle factor, args) for one fixture case."""
    gyro, dw_column, frame_dt, heights, rmode, dt_anchor, model = variant
    t, w, a = imu_signal()
    t_anchor = t[70] + 0.0021
    base = cam_base(cam_model, tr)
    p_B = board_corners(n, rng)
    uv = np.column_stack([rng.uniform(0, _W, n), rng.uniform(0, _H, n)])
    pids = rng.permutation(_N_BP)[:n] if heights else None
    h = rng.normal(0.0, 0.004, _N_BP) if heights else None
    kw = dict(rolling_mode=rmode, gyro=gyro, dw_column=dw_column, frame_dt=frame_dt,
              **_heights_kw(pids, _N_BP))
    live = CameraCalibImuFactor(uv, p_B, dt_anchor, ImuSegment(t, w, a, model),
                                t_anchor, _G, base, (_W, _H), **kw)
    ref = ref_camera_imu_factor.CameraCalibImuFactor(
        uv, p_B, dt_anchor, ref_imu_segment.ImuSegment(t, w, a, model),
        t_anchor, _G, base, (_W, _H), **kw)
    pose = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.2, 0.3), gtsam.Point3(0.1, 0.2, -0.5))
    vb = np.concatenate([rng.normal(size=3) * 0.3, _BG, _BA])
    xi = rng.normal(size=base.DIM) * 1e-3
    xi_imu = (rng.normal(size=24) * 1e-2) if gyro == "calibrated" else None
    dt_f = np.array([1.3e-3]) if frame_dt else None
    args = (pose, vb, _T_BOARD, xi, xi_imu, dt_f) + ((h,) if heights else ())
    return live, ref, args


def interp_poses(phi_case, rng):
    R_k = gtsam.Rot3.Rodrigues(0.1, 0.2, 0.3)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    mag = {"zero": 0.0, "tiny": 1e-11, "near_pi": np.pi - 1e-6, "normal": 0.05}[phi_case]
    R_k1 = R_k.compose(gtsam.Rot3.Expmap(mag * axis))
    pose_k = gtsam.Pose3(R_k, gtsam.Point3(0.1, 0.2, -0.5))
    pose_k1 = gtsam.Pose3(R_k1, gtsam.Point3(0.12, 0.19, -0.52))
    return pose_k, pose_k1


def interp_case(cam_model, tr, phi_case, lam_case, n, heights, rmode, rng):
    base = cam_base(cam_model, tr)
    p_B = board_corners(n, rng)
    uv = np.column_stack([rng.uniform(0, _W, n), rng.uniform(0, _H, n)])
    pids = rng.permutation(_N_BP)[:n] if heights else None
    h = rng.normal(0.0, 0.004, _N_BP) if heights else None
    dt = 0.05
    dt_frame = {"inside": 0.013, "below": -0.012, "above": 0.061}[lam_case]
    kw = dict(rolling_mode=rmode, **_heights_kw(pids, _N_BP))
    live = CameraCalibInterpFactor(uv, p_B, dt_frame, dt, base, (_W, _H), **kw)
    ref = ref_camera_interp_factor.CameraCalibInterpFactor(uv, p_B, dt_frame, dt, base,
                                                           (_W, _H), **kw)
    pose_k, pose_k1 = interp_poses(phi_case, rng)
    xi = rng.normal(size=base.DIM) * 1e-3
    args = (pose_k, pose_k1, _T_BOARD, xi) + ((h,) if heights else ())
    return live, ref, args


# G10: the forward ACI step takes the shared sign-(+1) H_b, ((Mw @ Tg) @ Ma) vs the frozen
# oracle's (((Rw @ Dw) @ Tg) @ Ra) @ Da: with Tg != 0 the accel-bias Jacobians and P_meas move
# at rounding level (<= 2e-16 relative); every other preintegrator key stays bitwise.
G10_MOVED = {"J_R_ba", "J_p_ba", "J_v_ba", "P_meas"}


def same_as_oracle(k, x, y):
    """Preintegrator key `k` of the live code vs the frozen oracle (see G10_MOVED)."""
    if k in G10_MOVED:
        return np.allclose(x, y, rtol=1e-12, atol=1e-20)
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and np.array_equal(x, y)
