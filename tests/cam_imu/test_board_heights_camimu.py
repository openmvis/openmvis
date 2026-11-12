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
"""Estimated board shape in the cam-IMU solve (imu-scale-followup, 2026-10-31):
one height per board point along the board normal, shared by every vision
factor through the key H0, with a weak prior and a plane / tilt gauge.

Covers: the heights block of BOTH vision factors matches finite differences
(and the other blocks still do with heights on); no ids = unchanged factor; the
gauge removes plane + tilt only; and the pipeline recovers a simulated twist
from a flat start with either vision factor."""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.calibration.factors.imu_segment import ImuSegment
from mvis.calibration.pipeline_cam_imu import (CamImuCalibConfig, H0,
                                               _board_height_factors)
from mvis.imu import CALYX, create_imu_model
from tests.helpers.shaking_rig import _truth, _board, _CV, _TR, _W, _H
from tests.helpers.shaking_rig import _imu, _vb, _G, _BG, _BA
from tests.helpers.imu_fixtures import intr_layout as _intr

_NBP = 40                                  # board has more points than the frame sees


def _ids_and_h(p_B, seed=3):
    rng = np.random.default_rng(seed)
    pids = rng.permutation(_NBP)[:len(p_B)]          # sparse, unordered ids
    h = rng.normal(0.0, 0.004, _NBP)                 # 4 mm heights
    return pids, h


def _calib():
    T_C = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.02, -0.01, 0.03), gtsam.Point3(0.05, -0.02, 0.01))
    return CameraCalib(T_C, 0.004, _TR, _CV, model="radtan")


def _xi(base):
    xi = np.zeros(base.DIM)
    xi[6], xi[7] = 2.3e-3, -1.1e-3
    xi[0:3] = [0.001, -0.002, 0.0015]
    return xi


def test_interp_factor_heights_jacobians_match_fd():
    base = _calib()
    T_BinW, p_B = _board()
    pids, h = _ids_and_h(p_B)
    pk, pk1 = _truth(0.06), _truth(0.10)
    f0 = CameraCalibInterpFactor(np.zeros((len(p_B), 2)), p_B, 0.013, 0.04, base,
                                 (_W, _H), rolling_mode=1)
    uv = f0.error(pk, pk1, T_BinW, np.zeros(base.DIM)).reshape(-1, 2)
    f = CameraCalibInterpFactor(uv, p_B, 0.013, 0.04, base, (_W, _H), rolling_mode=1,
                                board_pids=pids, n_board_points=_NBP)
    xi = _xi(base)
    e, J = f.error_and_jacobians(pk, pk1, T_BinW, xi, h)
    Jn = f._numerical_jacobians(pk, pk1, T_BinW, xi, h)
    assert len(J) == len(Jn) == 5 and J[4].shape == (2 * len(p_B), _NBP)
    assert np.abs(e).max() > 1.0                       # residual is live
    for k in range(5):
        tol = 1e-4 * max(1.0, np.abs(Jn[k]).max())
        assert np.abs(J[k] - Jn[k]).max() < tol, f"block {k}"
    # columns of board points this frame does not see are exactly zero
    unseen = np.setdiff1d(np.arange(_NBP), pids)
    assert np.abs(J[4][:, unseen]).max() == 0.0
    # heights = raising the board points: same residual as baking them in
    p_h = p_B.copy(); p_h[:, 2] += h[pids]
    g = CameraCalibInterpFactor(uv, p_h, 0.013, 0.04, base, (_W, _H), rolling_mode=1)
    assert np.allclose(e, g.error(pk, pk1, T_BinW, xi), atol=1e-9)


def test_interp_factor_without_ids_is_unchanged():
    base = _calib()
    T_BinW, p_B = _board()
    pk, pk1 = _truth(0.06), _truth(0.10)
    uv = np.full((len(p_B), 2), 300.0)
    f = CameraCalibInterpFactor(uv, p_B, 0.013, 0.04, base, (_W, _H), rolling_mode=1)
    assert not f.uses_heights_key
    _e, J = f.error_and_jacobians(pk, pk1, T_BinW, _xi(base))
    assert len(J) == 4
    ks = [gtsam.symbol(c, 0) for c in "xyba"]
    f.make_factor(*ks)
    assert f.keys == ks


@pytest.mark.parametrize("frame_dt", [False, True])
def test_imu_factor_heights_jacobians_match_fd(frame_dt):
    base = _calib()
    T_BinW, p_B = _board()
    pids, h = _ids_and_h(p_B)
    intr = _intr(CALYX)
    t, w_m, a_m = _imu(intr, _BG, _BA)
    seg = ImuSegment(t, w_m, a_m, CALYX)
    t_k = 0.082
    pose, vb = _truth(t_k), _vb(t_k, _BG, _BA)
    xi_I = create_imu_model(CALYX).local_coordinates(intr)
    f0 = CameraCalibImuFactor(np.zeros((len(p_B), 2)), p_B, 0.021, seg, t_k, _G,
                              base, (_W, _H), rolling_mode=1)
    uv = f0.error(pose, vb, T_BinW, np.zeros(base.DIM), xi_I).reshape(-1, 2)
    f = CameraCalibImuFactor(uv, p_B, 0.021, seg, t_k, _G, base, (_W, _H),
                             rolling_mode=1, frame_dt=frame_dt,
                             board_pids=pids, n_board_points=_NBP)
    dt_f = np.array([1.5e-3]) if frame_dt else None
    xi = _xi(base)
    e, J = f.error_and_jacobians(pose, vb, T_BinW, xi, xi_I, dt_f, h)
    Jn = f._numerical_jacobians(pose, vb, T_BinW, xi, xi_I, dt_f, h)
    n_blocks = 6 + int(frame_dt)                       # pose vb board calib imu (dt) h
    assert len(J) == len(Jn) == n_blocks and J[-1].shape == (2 * len(p_B), _NBP)
    for k in range(n_blocks):
        A, N = J[k], Jn[k]
        if k == 3:                                     # td / tr: constant-accel step
            for col in (6, 7):
                assert np.abs(A[:, col] - N[:, col]).max() < 2e-3 * max(np.abs(N[:, col]).max(), 1.0)
            rest = [c for c in range(A.shape[1]) if c not in (6, 7)]
            A, N = A[:, rest], N[:, rest]
        elif frame_dt and k == 5:                      # frame dt column = td column
            assert np.abs(A - N).max() < 2e-3 * max(np.abs(N).max(), 1.0)
            continue
        assert np.abs(A - N).max() < 1e-4 * max(1.0, np.abs(N).max()), f"block {k}"
    # key order: ..., (imu), (dt), heights — args_from_values reads it back
    ks = [gtsam.symbol(c, 0) for c in "xvbcIfH"]
    f.make_factor(ks[0], ks[1], ks[2], ks[3], k_imu=ks[4],
                  k_dt=(ks[5] if frame_dt else None), k_heights=ks[6])
    assert f.keys[-1] == ks[6] and len(f.keys) == 6 + int(frame_dt)
    vals = gtsam.Values()
    vals.insert(ks[0], pose); vals.insert(ks[1], vb); vals.insert(ks[2], T_BinW)
    vals.insert(ks[3], xi); vals.insert(ks[4], xi_I); vals.insert(ks[6], h)
    if frame_dt:
        vals.insert(ks[5], dt_f)
    assert np.allclose(np.linalg.norm(e.reshape(-1, 2), axis=1),
                       f.corner_norms_values(vals), atol=1e-9)


def test_height_gauge_removes_plane_and_tilt_only():
    g = np.linspace(-0.25, 0.25, 6)
    pts = np.array([[x, y, 0.0] for x in g for y in g])
    gauge, prior = _board_height_factors(pts, 0.003)
    def err(h):
        v = gtsam.Values(); v.insert(H0, h)
        return gauge.unwhitenedError(v)
    plane = 0.002 + 0.01 * pts[:, 0] - 0.004 * pts[:, 1]
    twist = 0.03 * pts[:, 0] * pts[:, 1]
    assert np.abs(err(plane)).max() > 1e-3            # plane / tilt are gauged
    assert np.abs(err(twist)).max() < 1e-12           # a twist passes untouched
    assert np.allclose(err(plane + twist), err(plane))


@pytest.mark.parametrize("vision_pose", ["interp", "imu"])
def test_pipeline_recovers_board_twist(vision_pose):
    """Simulated board with a 6 mm peak-to-peak twist, solved from the FLAT
    board: fixed heights leave the twist in the residuals, estimated heights
    recover it and bring the fit back to the pixel-noise floor."""
    from mvis.simulation.synth import build_synth, recover
    from tests.helpers.sim_scenarios import _small_gt
    gt = _small_gt()
    flat = np.array([[0.05 * j, 0.05 * i, 0.0] for i in range(6) for j in range(6)])
    x, y = flat[:, 0] - flat[:, 0].mean(), flat[:, 1] - flat[:, 1].mean()
    twist = 0.003 * (x * y) / np.abs(x * y).max()           # ±3 mm saddle, plane-free
    gt.board_points_local = flat + np.column_stack([0 * x, 0 * x, twist])
    ts = np.asarray(gt.t_states)
    imu_t = np.arange(ts[2], ts[-2], 1.0 / 400.0)
    cams, t_imu, w_m, a_m, aux = build_synth(
        gt, [ts[5:-5:3]], [imu_t], sigma_px=0.3,
        imu_sigmas=dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002),
        seed=7)
    fixed = recover(cams, flat, t_imu, w_m, a_m, aux, CamImuCalibConfig(vision_pose=vision_pose))
    # this 25 cm sim board constrains each height to ~1.3 mm only, so the weak
    # default prior (3 mm) would visibly shrink the twist; a wider prior shows
    # the estimate itself is unbiased (slope ~1)
    est = recover(cams, flat, t_imu, w_m, a_m, aux,
                  CamImuCalibConfig(vision_pose=vision_pose, estimate_board_heights=True,
                                    board_height_sigma=0.03))
    assert est.ok and "board_heights" not in fixed.diagnostics
    bh = est.diagnostics["board_heights"]
    seen = bh["n_obs"] > 0
    assert seen.sum() >= 30 and bh["sigma_m"] is not None
    h = bh["heights_m"][seen]
    slope = np.polyfit(twist[seen], h, 1)[0]
    corr = np.corrcoef(twist[seen], h)[0, 1]
    assert 0.85 < slope < 1.25 and corr > 0.9, f"slope {slope:.3f} corr {corr:.3f}"
    assert est.diagnostics["reproj_rms_px"] < fixed.diagnostics["reproj_rms_px"]
    assert est.diagnostics["reproj_rms_px"] < 0.5          # ~ the 0.3 px noise floor
    assert abs(np.mean(bh["offset_m"])) < 1e-6              # gauge holds the mean


def test_per_camera_vision_pose_builds_mixed_factors():
    """camera_input.cameras[i].vision_pose: one camera interpolated, one
    IMU-propagated in the same graph; the result reports the mix."""
    from mvis.simulation.synth import build_synth, recover
    from tests.helpers.sim_scenarios import _small_gt
    import mvis.calibration.pipeline_cam_imu as P
    gt = _small_gt()
    gt.board_points_local = np.array([[0.05 * j, 0.05 * i, 0.0] for i in range(6) for j in range(6)])
    ts = np.asarray(gt.t_states)
    imu_t = np.arange(ts[2], ts[-2], 1.0 / 400.0)
    cams, t_imu, w_m, a_m, aux = build_synth(
        gt, [ts[5:-5:3]], [imu_t], sigma_px=0.3,
        imu_sigmas=dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002), seed=7)
    cams = [cams[0], dict(cams[0])]        # a second (identical) camera on the same rig
    cams[0]["vision_pose"] = "interp"
    cams[1]["vision_pose"] = "imu"
    seen = {}
    orig = P.build_joint_graph
    def spy(*a, **k):
        out = orig(*a, **k)
        seen["types"] = {(c, type(f).__name__) for c, _j, f in out[2]}
        return out
    P.build_joint_graph = spy
    try:
        r = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux,
                    P.CamImuCalibConfig(vision_pose="interp", estimate_board_heights=True,
                                        max_iterations=3))
    finally:
        P.build_joint_graph = orig
    assert (0, "CameraCalibInterpFactor") in seen["types"] and (1, "CameraCalibImuFactor") in seen["types"]
    assert not any(c == 0 and t == "CameraCalibImuFactor" for c, t in seen["types"])
    assert r.diagnostics["vision_pose"][:2] == ["interp", "imu"]
    assert r.diagnostics["vision_imu"] is not None and "board_heights" in r.diagnostics
    with pytest.raises(ValueError):
        cams[1]["vision_pose"] = "bogus"
        recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux,
                P.CamImuCalibConfig(max_iterations=1))
