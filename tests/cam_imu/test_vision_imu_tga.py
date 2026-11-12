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
"""t_ga in the IMU-propagated vision factor (vision-imu-addition-verification,
2026-11-01): the factor's accel takes ACI's first-order gyro<->accel offset at the
live t_ga, so the IMU and vision factors read the same samples the same way.

Covers: the t_ga column matches finite differences (the other blocks still do);
under Tg = 0 the live shift equals D_p(0) + t_ga J_p_tga for either time direction;
tga off is the previous factor; t_ga = 0 changes no residual; and the shift has the
convention of tests/cam_imu/test_aci_tga.py (a sample stamped t was measured at t - t_ga)."""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor
from mvis.calibration.factors.imu_segment import ImuSegment
from mvis.imu import CALYX, KALIBR, create_imu_model
from tests.helpers.shaking_rig import _truth, _board, _CV, _TR, _W, _H
from tests.helpers.shaking_rig import _imu, _vb, _G, _BG, _BA
from tests.helpers.imu_fixtures import intr_layout as _intr

_T_GA = 5e-3


def _calib():
    T_C = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.02, -0.01, 0.03), gtsam.Point3(0.05, -0.02, 0.01))
    return CameraCalib(T_C, 0.004, _TR, _CV, model="radtan")


def _setup(model=CALYX, **kw):
    base = _calib()
    T_BinW, p_B = _board()
    intr = _intr(model)
    t, w_m, a_m = _imu(intr, _BG, _BA)
    seg = ImuSegment(t, w_m, a_m, model)
    t_k = 0.082
    pose, vb = _truth(t_k), _vb(t_k, _BG, _BA)
    xi_I = create_imu_model(model).local_coordinates(intr)
    f0 = CameraCalibImuFactor(np.zeros((len(p_B), 2)), p_B, 0.021, seg, t_k, _G,
                              base, (_W, _H), rolling_mode=1)
    uv = f0.error(pose, vb, T_BinW, np.zeros(base.DIM), xi_I).reshape(-1, 2)
    mk = lambda **k: CameraCalibImuFactor(uv + 0.5, p_B, 0.021, seg, t_k, _G, base, (_W, _H),
                                          rolling_mode=1, **{**kw, **k})
    xi = np.zeros(base.DIM)
    xi[0:3] = [0.001, -0.002, 0.0015]
    return mk, pose, vb, T_BinW, xi, xi_I, seg, intr


@pytest.mark.parametrize("model", [CALYX, KALIBR])
def test_tga_column_and_others_match_fd(model):
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup(model)
    f = mk(tga=True)
    t_ga = np.array([_T_GA])
    _e, J = f.error_and_jacobians(pose, vb, T_BinW, xi, xi_I, None, None, t_ga)
    Jn = f._numerical_jacobians(pose, vb, T_BinW, xi, xi_I, None, None, t_ga)
    assert len(J) == len(Jn) == 6 and J[-1].shape == (2 * f.n, 1)
    assert np.abs(J[-1]).max() > 0.0                   # the column is live
    assert np.abs(J[-1] - Jn[-1]).max() < 1e-4 * max(1.0, np.abs(Jn[-1]).max())
    for k in (0, 1, 2, 4):
        assert np.abs(J[k] - Jn[k]).max() < 1e-3 * max(1.0, np.abs(Jn[k]).max()), f"block {k}"


@pytest.mark.parametrize("delta", [0.018, -0.018, 0.0013, -0.0013])
def test_live_shift_equals_first_order_under_zero_tg(delta):
    _m, _p, _v, _T, _x, _xi, seg, _i = _setup()
    # intr=None: identity intrinsics, Tg = 0
    t_k = 0.082
    P0 = seg.propagate(t_k, np.array([delta]), _BG, _BA, None, jac=True, t_ga=0.0)[0]
    P = seg.propagate(t_k, np.array([delta]), _BG, _BA, None, jac=True, t_ga=_T_GA)[0]
    assert np.allclose(P.D_R, P0.D_R, atol=1e-14)      # no rotation effect
    assert np.abs(P0.J_R_tga).max() == 0.0
    assert np.abs(P0.J_p_tga).max() > 0.0
    assert np.allclose(P.D_p, P0.D_p + _T_GA * P0.J_p_tga, atol=1e-12)
    # the error() path (jac=False) gives the same pose
    Q = seg.propagate(t_k, np.array([delta]), _BG, _BA, None, jac=False, t_ga=_T_GA)[0]
    assert np.allclose(Q.D_p, P.D_p, atol=1e-14)


def test_target_on_an_imu_sample_keeps_the_column():
    """A capture time exactly on a raw sample takes the checkpoint branch."""
    _m, _p, _v, _T, _x, _xi, seg, _i = _setup()
    t_k = 0.082
    i = int(np.searchsorted(seg.t, t_k)) + 3
    d = np.array([seg.t[i] - t_k])
    eps = 1e-6
    P = seg.propagate(t_k, d, _BG, _BA, None, jac=True, t_ga=_T_GA)[0]
    Pp = seg.propagate(t_k, d, _BG, _BA, None, jac=False, t_ga=_T_GA + eps)[0]
    Pm = seg.propagate(t_k, d, _BG, _BA, None, jac=False, t_ga=_T_GA - eps)[0]
    assert np.abs(P.J_p_tga).max() > 0.0
    assert np.abs(P.J_p_tga - (Pp.D_p - Pm.D_p) / (2 * eps)).max() < 1e-7
    # zero offset: nothing integrated, nothing shifted
    Z = seg.propagate(t_k, np.zeros(1), _BG, _BA, None, jac=True, t_ga=_T_GA)[0]
    assert np.array_equal(Z.D_p, np.zeros(3)) and np.array_equal(Z.J_p_tga, np.zeros(3))


def test_off_is_the_previous_factor():
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup()
    old, off = mk(), mk(tga=False)
    assert not off.tga
    e0, J0 = old.error_and_jacobians(pose, vb, T_BinW, xi, xi_I)
    e1, J1 = off.error_and_jacobians(pose, vb, T_BinW, xi, xi_I)
    assert np.array_equal(e0, e1) and len(J0) == len(J1) == 5
    for a, b in zip(J0, J1):
        assert np.array_equal(a, b)


def test_zero_t_ga_changes_no_residual():
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup()
    e0, J0 = mk().error_and_jacobians(pose, vb, T_BinW, xi, xi_I)
    e1, J1 = mk(tga=True).error_and_jacobians(pose, vb, T_BinW, xi, xi_I, None, None, np.zeros(1))
    assert np.array_equal(e0, e1)
    for a, b in zip(J0, J1[:5]):
        assert np.array_equal(a, b)


def test_sign_matches_a_delayed_accel():
    """Accel delayed by t_ga: applying +t_ga recovers the undelayed propagation."""
    _m, _p, _v, _T, _x, _xi, seg, _i = _setup()
    t_k, d = 0.082, np.array([0.02])
    ref = seg.propagate(t_k, d, _BG, _BA, None, jac=False)[0]
    a_late = np.column_stack([np.interp(seg.t - _T_GA, seg.t, seg.a[:, i]) for i in range(3)])
    late = ImuSegment(seg.t, seg.w, a_late, seg.model)
    raw = late.propagate(t_k, d, _BG, _BA, None, jac=False)[0]
    fix = late.propagate(t_k, d, _BG, _BA, None, jac=False, t_ga=_T_GA)[0]
    assert np.linalg.norm(fix.D_p - ref.D_p) < 0.5 * np.linalg.norm(raw.D_p - ref.D_p)


def test_residual_at_truth_falls_with_the_true_t_ga():
    """Corners from the undelayed motion, IMU with delayed accel: the true t_ga
    brings the residual back to the undelayed one. Sign and size of the term."""
    base = _calib()
    T_BinW, p_B = _board()
    intr = _intr(CALYX)
    t, w_m, a_m = _imu(intr, _BG, _BA)
    t_k = 0.082
    pose, vb = _truth(t_k), _vb(t_k, _BG, _BA)
    xi_I = create_imu_model(CALYX).local_coordinates(intr)
    seg = ImuSegment(t, w_m, a_m, CALYX)
    f0 = CameraCalibImuFactor(np.zeros((len(p_B), 2)), p_B, 0.021, seg, t_k, _G, base, (_W, _H),
                              rolling_mode=1)
    uv = f0.error(pose, vb, T_BinW, np.zeros(base.DIM), xi_I).reshape(-1, 2)
    for _ in range(3):                                # rows set rho: settle the corners
        fr = CameraCalibImuFactor(uv, p_B, 0.021, seg, t_k, _G, base, (_W, _H), rolling_mode=1)
        uv = uv + fr.error(pose, vb, T_BinW, np.zeros(base.DIM), xi_I).reshape(-1, 2)
    late = np.column_stack([np.interp(t - _T_GA, t, a_m[:, i]) for i in range(3)])
    f = CameraCalibImuFactor(uv, p_B, 0.021, ImuSegment(t, w_m, late, CALYX), t_k, _G, base,
                             (_W, _H), rolling_mode=1, tga=True)
    xi = np.zeros(base.DIM)
    ref = CameraCalibImuFactor(uv, p_B, 0.021, seg, t_k, _G, base, (_W, _H),
                               rolling_mode=1).error(pose, vb, T_BinW, xi, xi_I)
    e_off = np.linalg.norm(f.error(pose, vb, T_BinW, xi, xi_I, None, None, np.zeros(1)) - ref)
    e_on = np.linalg.norm(f.error(pose, vb, T_BinW, xi, xi_I, None, None, np.array([_T_GA])) - ref)
    assert e_off > 0.0 and e_on < 0.5 * e_off


def test_keys_and_args_read_back():
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup()
    f = mk(tga=True)
    ks = [gtsam.symbol(c, 0) for c in "xvbcIo"]
    f.make_factor(ks[0], ks[1], ks[2], ks[3], k_imu=ks[4], k_tga=ks[5])
    assert f.keys == ks
    vals = gtsam.Values()
    vals.insert(ks[0], pose); vals.insert(ks[1], vb); vals.insert(ks[2], T_BinW)
    vals.insert(ks[3], xi); vals.insert(ks[4], xi_I); vals.insert(ks[5], np.array([_T_GA]))
    assert float(f.args_from_values(vals)[-1][0]) == _T_GA
    with pytest.raises(AssertionError):
        mk(tga=True, gyro="raw")


@pytest.mark.parametrize("frame_dt", [False, True])
def test_tga_with_frame_dt_and_heights_key_order_and_fd(frame_dt):
    """t_ga is the LAST block and key, after (dt) and heights; all still match FD."""
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup()
    n_bp = 40
    probe = mk()
    rng = np.random.default_rng(3)
    pids = rng.permutation(n_bp)[:probe.n]
    h = rng.normal(0.0, 0.004, n_bp)
    f = mk(tga=True, frame_dt=frame_dt, board_pids=pids, n_board_points=n_bp)
    dt_f = np.array([1.5e-3]) if frame_dt else None
    t_ga = np.array([_T_GA])
    _e, J = f.error_and_jacobians(pose, vb, T_BinW, xi, xi_I, dt_f, h, t_ga)
    Jn = f._numerical_jacobians(pose, vb, T_BinW, xi, xi_I, dt_f, h, t_ga)
    assert len(J) == len(Jn) == 7 + int(frame_dt)
    assert J[-2].shape == (2 * f.n, n_bp) and J[-1].shape == (2 * f.n, 1)
    assert np.abs(J[-1] - Jn[-1]).max() < 1e-4 * max(1.0, np.abs(Jn[-1]).max())
    ks = [gtsam.symbol(c, 0) for c in "xvbcIfHo"]
    f.make_factor(ks[0], ks[1], ks[2], ks[3], k_imu=ks[4], k_dt=(ks[5] if frame_dt else None),
                  k_heights=ks[6], k_tga=ks[7])
    assert f.keys[-1] == ks[7] and f.keys[-2] == ks[6]
    vals = gtsam.Values()
    for k, v in ((ks[0], pose), (ks[2], T_BinW)):
        vals.insert(k, v)
    for k, v in ((ks[1], vb), (ks[3], xi), (ks[4], xi_I), (ks[6], h), (ks[7], t_ga)):
        vals.insert(k, v)
    if frame_dt:
        vals.insert(ks[5], dt_f)
    args = f.args_from_values(vals)
    assert float(args[-1][0]) == _T_GA and np.array_equal(args[-2], h)
    # every path sees the same live t_ga
    assert np.allclose(f.error(*args), f.error_and_jacobians(*args)[0], atol=1e-9)
    assert np.allclose(np.nan_to_num(f.corner_norms(*args)),
                       np.linalg.norm(f.error(*args).reshape(-1, 2), axis=1), atol=1e-12)


def test_tga_factor_without_value_reads_zero():
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup()
    f = mk(tga=True)
    assert np.array_equal(f.error(pose, vb, T_BinW, xi, xi_I),
                          f.error(pose, vb, T_BinW, xi, xi_I, None, None, np.zeros(1)))
    assert np.abs(f.error(pose, vb, T_BinW, xi, xi_I, None, None, np.array([_T_GA]))
                  - f.error(pose, vb, T_BinW, xi, xi_I)).max() > 0.0


def test_make_factor_needs_the_key_and_gtsam_factor_matches():
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup()
    f = mk(tga=True)
    ks = [gtsam.symbol(c, 0) for c in "xvbcIo"]
    with pytest.raises(AssertionError):
        f.make_factor(ks[0], ks[1], ks[2], ks[3], k_imu=ks[4])
    gf = f.make_factor(ks[0], ks[1], ks[2], ks[3], k_imu=ks[4], k_tga=ks[5])
    assert gf.keys()[-1] == ks[5] if hasattr(gf.keys(), "__getitem__") else True
    vals = gtsam.Values()
    vals.insert(ks[0], pose); vals.insert(ks[1], vb); vals.insert(ks[2], T_BinW)
    vals.insert(ks[3], xi); vals.insert(ks[4], xi_I); vals.insert(ks[5], np.array([_T_GA]))
    want = f.error(pose, vb, T_BinW, xi, xi_I, None, None, np.array([_T_GA]))
    assert np.isclose(gf.error(vals), 0.5 * float(want @ want))


def test_nonzero_tg_reaches_the_gyro():
    """With Tg != 0 the shift moves the body rate too, in both time directions,
    and the column still matches a finite difference of the propagation."""
    _m, _p, _v, _T, _x, _xi, seg, _i = _setup()
    model = create_imu_model(CALYX)
    xi = np.zeros(24)
    xi[model.tg_offset if hasattr(model, "tg_offset") else 15:24] = 2e-3 * np.arange(1, 10)
    intr = model.retract(xi)
    assert np.abs(intr.Tg()).max() > 0
    for delta in (0.018, -0.018):
        d = np.array([delta])
        P = seg.propagate(0.082, d, _BG, _BA, intr, jac=True, t_ga=_T_GA)[0]
        eps = 1e-6
        Pp = seg.propagate(0.082, d, _BG, _BA, intr, jac=False, t_ga=_T_GA + eps)[0]
        Pm = seg.propagate(0.082, d, _BG, _BA, intr, jac=False, t_ga=_T_GA - eps)[0]
        assert np.abs(P.J_R_tga).max() > 0
        assert np.abs(P.J_p_tga - (Pp.D_p - Pm.D_p) / (2 * eps)).max() < 1e-6
        dR = P.D_R.T @ (Pp.D_R - Pm.D_R) / (2 * eps)
        fd = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])
        assert np.abs(P.J_R_tga - fd).max() < 1e-6


# --- vision_imu_linearized: integrate once, correct linearly --------------------------------


def _both(tga=True):
    mk, pose, vb, T_BinW, xi, xi_I, seg, _i = _setup()
    args = (pose, vb, T_BinW, xi, xi_I, None, None, np.array([_T_GA]))
    return mk(tga=tga), mk(tga=tga, linearized=True), args, seg


def test_linearized_equals_the_full_factor_at_its_point():
    full, lin, args, _seg = _both()
    np.testing.assert_array_equal(full.error(*args), lin.error(*args))
    _e0, J0 = full.error_and_jacobians(*args)
    _e1, J1 = lin.error_and_jacobians(*args)
    for a, b in zip(J0, J1):
        np.testing.assert_array_equal(a, b)


def test_linearized_jacobians_match_fd_at_its_point():
    _full, lin, args, _seg = _both()
    _e, J = lin.error_and_jacobians(*args)
    Jn = lin._numerical_jacobians(*args)
    assert np.abs(J[-1] - Jn[-1]).max() < 1e-3 * max(1.0, np.abs(Jn[-1]).max())
    for k in (0, 1, 2, 3, 4):
        assert np.abs(J[k] - Jn[k]).max() < 2e-3 * max(1.0, np.abs(Jn[k]).max()), f"block {k}"


def test_linearized_stays_close_to_the_full_factor_nearby():
    full, lin, args, _seg = _both()
    pose, vb, T_BinW, xi, xi_I, a, b, tg = args
    lin.error_and_jacobians(*args)      # store the point
    vb2 = vb.copy()
    vb2[3:9] += [1e-3, -1e-3, 5e-4, 2e-3, -2e-3, 1e-3]
    xi2 = xi.copy()
    xi2[6] += 5e-4                      # td, half a ms, inside TD_REFRESH_S
    xiI2 = xi_I + 1e-4
    near = (pose, vb2, T_BinW, xi2, xiI2, a, b, tg + 5e-4)
    moved = np.abs(full.error(*near) - full.error(*args)).max()
    gap = np.abs(lin.error(*near) - full.error(*near)).max()
    assert moved > 0.05
    assert gap < 0.02 * moved


def test_linearized_with_frame_dt_matches_fd_and_the_full_factor():
    """Per-frame offsets go through the same first-order correction: the dt block is right, and a
    small offset inside TD_REFRESH_S stays close to re-integrating."""
    mk, pose, vb, T_BinW, xi, xi_I, _seg, _i = _setup()
    full, lin = mk(tga=True, frame_dt=True), mk(tga=True, frame_dt=True, linearized=True)
    t_ga = np.array([_T_GA])
    args = (pose, vb, T_BinW, xi, xi_I, np.array([1.5e-3]), None, t_ga)
    np.testing.assert_array_equal(full.error(*args), lin.error(*args))
    _e, J = lin.error_and_jacobians(*args)
    Jn = lin._numerical_jacobians(*args)
    assert len(J) == len(Jn)
    for k, (a, b) in enumerate(zip(J, Jn)):
        assert np.abs(a - b).max() < 2e-3 * max(1.0, np.abs(b).max()), f"block {k}"
    near = (pose, vb, T_BinW, xi, xi_I, np.array([1.5e-3 + 5e-4]), None, t_ga)
    moved = np.abs(full.error(*near) - full.error(*args)).max()
    gap = np.abs(lin.error(*near) - full.error(*near)).max()
    assert moved > 0.05
    assert gap < 0.02 * moved


def test_linearized_integrates_only_when_it_has_to():
    _full, lin, args, seg = _both()
    calls, orig = [], seg.propagate
    seg.propagate = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    pose, vb, T_BinW, xi, xi_I, a, b, tg = args
    lin.error_and_jacobians(*args)
    assert len(calls) == 1
    lin.error_and_jacobians(pose, vb, T_BinW, xi, xi_I, a, b, tg + 1e-4)
    lin.error(*args)
    assert len(calls) == 1              # small moves reuse the point
    far = xi.copy()
    far[6] += 2e-3
    lin.error(pose, vb, T_BinW, far, xi_I, a, b, tg)
    assert len(calls) == 1              # an error-only trial never moves it
    lin.error_and_jacobians(pose, vb, T_BinW, far, xi_I, a, b, tg)
    assert len(calls) == 2              # a linearization past TD_REFRESH_S does
    seg.relinearize_tga(seg.a, 0.0)
    lin.error(pose, vb, T_BinW, far, xi_I, a, b, tg)
    assert len(calls) == 3              # a moved accel always does


def test_td_refresh_can_be_held_off():
    _full, lin, args, seg = _both()
    calls, orig = [], seg.propagate
    seg.propagate = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    pose, vb, T_BinW, xi, xi_I, a, b, tg = args
    lin.error_and_jacobians(*args)
    seg.allow_td_refresh = False
    far = xi.copy()
    far[6] += 2e-3
    lin.error_and_jacobians(pose, vb, T_BinW, far, xi_I, a, b, tg)
    assert len(calls) == 1
