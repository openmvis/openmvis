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
"""CameraCalibImuFactor (elp-stereo-imu-investigation E8): the IMU-propagated
vision factor has no chord model error at truth, analytic Jacobians that match
finite differences for every key (both anchor directions, both gyro modes,
Calyx and Kalibr IMU models), and the pipeline builds it behind
solve.vision_pose = "imu"."""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.calibration.factors.imu_segment import ImuSegment
from mvis.calibration.factors.lie_groups import exp_so3
from mvis.imu import CALYX, KALIBR, create_imu_model
from tests.helpers.imu_fixtures import intr_layout as _intr
from tests.helpers.shaking_rig import (_truth, _board, _rs_measure, _TR, _CV, _W, _H,
                                       _G, _imu, _vb, _BG, _BA)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt_anchor", [0.012, 0.033, -0.009])
def test_no_model_error_at_truth(dt_anchor):
    """At truth the only residual is the integration error of 200 Hz samples:
    far below the chord error of the interpolated factor on the same frame."""
    calib = CameraCalib(gtsam.Pose3(), 0.0, _TR, _CV, model="radtan")
    T_BinW, p_B = _board()
    t_k = 0.082
    t_cap = t_k + dt_anchor
    uv = _rs_measure(t_cap, T_BinW, p_B, calib)
    t, w_m, a_m = _imu()
    seg = ImuSegment(t, w_m, a_m, CALYX)
    f = CameraCalibImuFactor(uv, p_B, dt_anchor, seg, t_k, _G, calib, (_W, _H))
    e = f.error(_truth(t_k), _vb(t_k), T_BinW, np.zeros(calib.DIM), np.zeros(24))
    err_imu = np.linalg.norm(e.reshape(-1, 2), axis=1).max()
    assert err_imu < 0.02, err_imu
    if dt_anchor > 0:
        g = CameraCalibInterpFactor(uv, p_B, dt_anchor, 0.041, calib, (_W, _H))
        e2 = g.error(_truth(t_k), _truth(t_k + 0.041), T_BinW, np.zeros(calib.DIM))
        err_chord = np.linalg.norm(e2.reshape(-1, 2), axis=1).max()
        assert err_imu < 0.05 * err_chord


def test_raw_gyro_variant_sees_dw_error():
    """gyro='raw' ignores the IMU intrinsics: with a 4 % gyro scale in the
    data the residual at truth is pixel-level (the model error of variant A)."""
    calib = CameraCalib(gtsam.Pose3(), 0.0, _TR, _CV, model="radtan")
    T_BinW, p_B = _board()
    intr = create_imu_model(CALYX, x_dw=np.array([0.96, 0, 0, 0, 0.96, 0, 0, 0, 1.0]))
    t_k, dt_anchor = 0.082, 0.035
    uv = _rs_measure(t_k + dt_anchor, T_BinW, p_B, calib)
    t, w_m, a_m = _imu(intr)
    seg = ImuSegment(t, w_m, a_m, CALYX)
    xi_I = create_imu_model(CALYX).local_coordinates(intr)
    f_cal = CameraCalibImuFactor(uv, p_B, dt_anchor, seg, t_k, _G, calib, (_W, _H))
    f_raw = CameraCalibImuFactor(uv, p_B, dt_anchor, seg, t_k, _G, calib, (_W, _H),
                                 gyro="raw")
    e_cal = f_cal.error(_truth(t_k), _vb(t_k), T_BinW, np.zeros(calib.DIM), xi_I)
    e_raw = f_raw.error(_truth(t_k), _vb(t_k), T_BinW, np.zeros(calib.DIM))
    assert np.abs(e_cal).max() < 0.03          # 200 Hz integration floor on the shaking twin
    assert np.abs(e_raw).max() > 0.3


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [CALYX, KALIBR])
@pytest.mark.parametrize("gyro", ["calibrated", "raw"])
@pytest.mark.parametrize("dt_anchor", [0.021, -0.0135])
@pytest.mark.parametrize("cam_model", ["radtan", "kb4"])
def test_jacobians_match_fd(model, gyro, dt_anchor, cam_model):
    """Analytic vs central-difference Jacobians, all keys, away from the
    linearisation point (biases, non-identity intrinsics, td/tr perturbed,
    rolling_mode 1, mixed-sign row offsets through dt_anchor + rho*tr)."""
    cv = {"radtan": _CV,
          "kb4": np.array([430., 428, 320, 240, 0.02, -0.01, 3e-3, -5e-4])}[cam_model]
    T_C = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.02, -0.01, 0.03), gtsam.Point3(0.05, -0.02, 0.01))
    base = CameraCalib(T_C, 0.004, _TR, cv, model=cam_model)
    T_BinW, p_B = _board()
    intr = _intr(model)
    t, w_m, a_m = _imu(intr, _BG, _BA)
    seg = ImuSegment(t, w_m, a_m, model)
    t_k = 0.082
    f0 = CameraCalibImuFactor(np.zeros((len(p_B), 2)), p_B, dt_anchor, seg, t_k, _G,
                              base, (_W, _H), rolling_mode=1, gyro=gyro)
    pose, vb = _truth(t_k), _vb(t_k, _BG, _BA)
    xi_I = (create_imu_model(model).local_coordinates(intr)
            if gyro == "calibrated" else None)
    uv = f0.error(pose, vb, T_BinW, np.zeros(base.DIM), xi_I).reshape(-1, 2)
    f = CameraCalibImuFactor(uv, p_B, dt_anchor, seg, t_k, _G, base, (_W, _H),
                             rolling_mode=1, gyro=gyro)
    xi = np.zeros(base.DIM)
    xi[6] = 2.3e-3                                  # td off the seed
    xi[7] = -1.1e-3                                 # tr off the seed
    xi[0:3] = [0.001, -0.002, 0.0015]
    e, J = f.error_and_jacobians(pose, vb, T_BinW, xi, xi_I)
    Jn = f._numerical_jacobians(pose, vb, T_BinW, xi, xi_I)
    assert len(J) == len(Jn) == (5 if gyro == "calibrated" else 4)
    assert np.isfinite(e).all() and np.abs(e).max() > 1.0   # residual is live
    names = ["pose", "vb", "board", "calib", "imu"]
    for k in range(len(J)):
        A, N = J[k], Jn[k]
        if k == 3:
            # td / tr columns: the position part of the time derivative uses
            # the partial step's constant acceleration -> ~1e-3 relative
            for col in (6, 7):
                scale = max(np.abs(N[:, col]).max(), 1.0)
                assert np.abs(A[:, col] - N[:, col]).max() < 2e-3 * scale, f"calib col {col}"
            rest = [c for c in range(A.shape[1]) if c not in (6, 7)]
            assert np.abs(A[:, rest] - N[:, rest]).max() < 1e-4, "calib pose/intr"
        else:
            assert np.abs(A - N).max() < 1e-4, f"block {names[k]}"


def test_dw_column_off_zeroes_imu_block_only():
    base = CameraCalib(gtsam.Pose3(), 0.0, _TR, _CV, model="radtan")
    T_BinW, p_B = _board()
    intr = _intr(CALYX)
    t, w_m, a_m = _imu(intr, _BG, _BA)
    seg = ImuSegment(t, w_m, a_m, CALYX)
    uv = _rs_measure(0.1, T_BinW, p_B, base)
    kw = dict(rolling_mode=0)
    fa = CameraCalibImuFactor(uv, p_B, 0.018, seg, 0.082, _G, base, (_W, _H), **kw)
    fb = CameraCalibImuFactor(uv, p_B, 0.018, seg, 0.082, _G, base, (_W, _H),
                              dw_column=False, **kw)
    args = (_truth(0.082), _vb(0.082, _BG, _BA), T_BinW, np.zeros(base.DIM),
            create_imu_model(CALYX).local_coordinates(intr))
    ea, Ja = fa.error_and_jacobians(*args)
    eb, Jb = fb.error_and_jacobians(*args)
    assert np.array_equal(ea, eb)
    for k in range(4):
        assert np.array_equal(Ja[k], Jb[k])
    assert np.abs(Ja[4]).max() > 0 and np.abs(Jb[4]).max() == 0


def test_make_factor_keys_and_corner_norms_values():
    base = CameraCalib(gtsam.Pose3(), 0.0, _TR, _CV, model="radtan")
    T_BinW, p_B = _board()
    t, w_m, a_m = _imu()
    seg = ImuSegment(t, w_m, a_m, CALYX)
    uv = _rs_measure(0.1, T_BinW, p_B, base)
    f = CameraCalibImuFactor(uv, p_B, 0.018, seg, 0.082, _G, base, (_W, _H))
    X, V, B, C, I = (gtsam.symbol(ch, 0) for ch in "xvbci")
    gf = f.make_factor(X, V, B, C, k_imu=I)
    assert f.keys == [X, V, B, C, I]
    vals = gtsam.Values()
    vals.insert(X, _truth(0.082)); vals.insert(V, _vb(0.082))
    vals.insert(B, T_BinW); vals.insert(C, np.zeros(base.DIM)); vals.insert(I, np.zeros(24))
    assert gf.error(vals) < 1e-3
    assert np.nanmax(f.corner_norms_values(vals)) < 0.02
    f_raw = CameraCalibImuFactor(uv, p_B, 0.018, seg, 0.082, _G, base, (_W, _H), gyro="raw")
    gf2 = f_raw.make_factor(X, V, B, C)
    assert f_raw.keys == [X, V, B, C] and gf2.error(vals) < 1e-3


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gyro", ["calibrated", "raw"])
def test_frame_dt_jacobians_match_fd(gyro):
    """Per-frame capture-time offset: the extra 2N x 1 block matches FD, all
    other blocks are unchanged by its presence, and it equals the td column."""
    base = CameraCalib(gtsam.Pose3(gtsam.Rot3.Rodrigues(0.02, -0.01, 0.03),
                                   gtsam.Point3(0.05, -0.02, 0.01)), 0.004, _TR, _CV,
                       model="radtan")
    T_BinW, p_B = _board()
    intr = _intr(CALYX)
    t, w_m, a_m = _imu(intr, _BG, _BA)
    seg = ImuSegment(t, w_m, a_m, CALYX)
    pose, vb = _truth(0.082), _vb(0.082, _BG, _BA)
    xi_I = create_imu_model(CALYX).local_coordinates(intr) if gyro == "calibrated" else None
    uv = _rs_measure(0.1, T_BinW, p_B, base)
    f = CameraCalibImuFactor(uv, p_B, 0.021, seg, 0.082, _G, base, (_W, _H),
                             rolling_mode=1, gyro=gyro, frame_dt=True)
    g = CameraCalibImuFactor(uv, p_B, 0.021, seg, 0.082, _G, base, (_W, _H),
                             rolling_mode=1, gyro=gyro)
    xi = np.zeros(base.DIM); xi[6] = 1.3e-3
    d_f = np.array([2.7e-3])
    e, J = f.error_and_jacobians(pose, vb, T_BinW, xi, xi_I, d_f)
    Jn = f._numerical_jacobians(pose, vb, T_BinW, xi, xi_I, d_f)
    assert len(J) == len(Jn) == (6 if gyro == "calibrated" else 5)
    scale = max(np.abs(Jn[-1]).max(), 1.0)
    assert np.abs(J[-1] - Jn[-1]).max() < 2e-3 * scale
    assert np.array_equal(J[-1][:, 0], J[3][:, 6])          # δ_f column == td column
    # δ_f is exactly a td shift for this frame
    xi2 = xi.copy(); xi2[6] += d_f[0]
    e2, J2 = g.error_and_jacobians(pose, vb, T_BinW, xi2, xi_I)
    assert np.allclose(e, e2, atol=1e-9)
    for k in range(len(J2)):
        assert np.allclose(J[k], J2[k], atol=1e-9)


def test_frame_dt_keys_and_values():
    base = CameraCalib(gtsam.Pose3(), 0.0, _TR, _CV, model="radtan")
    T_BinW, p_B = _board()
    t, w_m, a_m = _imu()
    seg = ImuSegment(t, w_m, a_m, CALYX)
    uv = _rs_measure(0.1, T_BinW, p_B, base)
    f = CameraCalibImuFactor(uv, p_B, 0.018, seg, 0.082, _G, base, (_W, _H), frame_dt=True)
    X, V, B, C, I, F = (gtsam.symbol(ch, 0) for ch in "xvbcif")
    gf = f.make_factor(X, V, B, C, k_imu=I, k_dt=F)
    assert f.keys == [X, V, B, C, I, F]
    vals = gtsam.Values()
    vals.insert(X, _truth(0.082)); vals.insert(V, _vb(0.082)); vals.insert(B, T_BinW)
    vals.insert(C, np.zeros(base.DIM)); vals.insert(I, np.zeros(24)); vals.insert(F, np.zeros(1))
    assert gf.error(vals) < 1e-3
    vals.update(F, np.array([0.004]))
    assert gf.error(vals) > 1e-2                              # 4 ms is visible
