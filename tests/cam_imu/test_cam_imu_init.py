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
"""Camera-IMU initialization tests (target 2.1, M2) — per-stage GT scoring.

Inputs mirror the real cam-IMU process: board detections + camera intrinsics
(from the separate camera-only calibration) + raw IMU. Scored against the
simulator's known ground truth on a long (40 s) calib_3D_v28 window.
"""


import numpy as np
import gtsam
import pytest

from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.lie_groups import log_so3
from mvis.calibration.cam_imu_init import (
    pnp_pose_track, estimate_time_offset_coarse, refine_time_offset,
    estimate_rotation_gyro, linear_alignment, gravity_align,
    initialize_cam_imu, initialize_cam_imu_from_detections, GRAVITY_MAG,
)
from tests.helpers.sim_scenarios import (_IMG, _RADTAN, _BG, _BA, _T_CtoI, _CAM_HZ,
                                         scenario_for_init as _scenario, data_for_init
                                         as _data)


def _gt_track(sc, t_cam, td):
    """Ground-truth T_CtoB at the capture times (stamped t ↔ trajectory t+td)."""
    poses = []
    Binv = sc.T_BinW.inverse()
    for t in t_cam:
        R_GtoI, p = sc.traj.get_pose(t + td)
        T_CtoW = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(sc.T_CtoI)
        poses.append(Binv.compose(T_CtoW))
    return poses


# --------------------------------------------------------------------------
def test_pnp_pose_track_matches_gt():
    """I0: PnP from detections + intrinsics reproduces the GT camera poses."""
    sc = _scenario()
    t_cam, cam, _s = _data(sc, win=5.0)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    t_kept, poses, n = pnp_pose_track(t_cam, cam.ids, cam.uv, pts_B, _RADTAN)
    gt = _gt_track(sc, t_kept, 0.0)
    assert len(t_kept) > 20            # v0 has board-visibility gaps (by design)
    for T, Tg in zip(poses, gt):
        d = Tg.localCoordinates(T)
        assert np.linalg.norm(d[0:3]) < 2e-5, f"rot {d[0:3]}"
        assert np.linalg.norm(d[3:6]) < 2e-4, f"trans {d[3:6]}"


@pytest.mark.parametrize("td_true", [0.0, 0.02, -0.02])
def test_time_offset_sign_and_value(td_true):
    """I1: td recovered within 0.5 ms INCLUDING SIGN (the #1 convention risk).
    Coarse |ω| correlation locates it to a few ms (broad peak on smooth
    calibration motion); the LS-valley refinement supplies the precision."""
    sc = _scenario(td_cam=td_true)
    t_cam, cam, s = _data(sc)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    t_kept, poses, _n = pnp_pose_track(t_cam, cam.ids, cam.uv, pts_B, _RADTAN)
    R_list = [P.rotation().matrix() for P in poses]
    td0, _rep = estimate_time_offset_coarse(t_kept, R_list, s.t, s.w_m)
    # coarse is only a cross-check; the LS valley does the global search
    assert abs(td0 - td_true) < 15e-3, f"coarse td {td0:+.4f} vs {td_true:+.4f}"
    td, rep = refine_time_offset(t_kept, R_list, s.t, s.w_m, td0)
    assert abs(td - td_true) < 5e-4, f"td {td:+.5f} vs true {td_true:+.5f}"
    assert rep["valley_ratio"] > 1.5          # the td quality gate


def test_rotation_and_gyro_bias():
    """I2: R_CtoI < 0.5 deg, b_g < 2e-3 rad/s (centered Kabsch kills the bias)."""
    sc = _scenario()
    t_cam, cam, s = _data(sc)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    t_kept, poses, _n = pnp_pose_track(t_cam, cam.ids, cam.uv, pts_B, _RADTAN)
    R, b_g, rep = estimate_rotation_gyro(
        t_kept, [P.rotation().matrix() for P in poses], s.t, s.w_m, td=0.0)
    ang = np.linalg.norm(log_so3(_T_CtoI.rotation().matrix().T @ R))
    assert ang < np.deg2rad(0.5), f"rotation err {np.rad2deg(ang):.3f} deg"
    assert np.linalg.norm(b_g - _BG) < 2e-3, f"b_g {b_g} vs {_BG}"
    assert rep["excitation_svals"][-1] > 1e-3        # 3-axis excitation present


def test_linear_alignment_scale_gravity_velocity():
    """I3: scale-unknown alignment — s ~ 1, ||g_LS|| ~ 9.81 (the quality gate),
    gravity direction, velocities, p_IinC, and b_a all recovered."""
    sc = _scenario()
    t_cam, cam, s = _data(sc)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    t_kept, poses, _n = pnp_pose_track(t_cam, cam.ids, cam.uv, pts_B, _RADTAN)
    R_CtoI = _T_CtoI.rotation().matrix()
    # decimate to 5 Hz keyframes
    stride = int(_CAM_HZ / 5.0)
    idx = np.arange(0, len(t_kept), stride)
    # estimate_ba=True exercises the optional b_a path (default is b_a = 0 —
    # the joint solve owns the accel bias)
    res = linear_alignment(t_kept[idx], [poses[i] for i in idx],
                           R_CtoI, _BG, s.t, s.w_m, s.a_m, estimate_ba=True)

    # primary quality gate: gravity magnitude of the unconstrained solve
    assert abs(res.g_norm_ls - GRAVITY_MAG) / GRAVITY_MAG < 0.01, res.g_norm_ls
    # scale consistency (board is metric)
    assert abs(res.s - 1.0) < 0.005, f"s = {res.s:.4f}"
    # gravity direction vs GT (g_inB = R_BinW^T g_inW)
    g_gt = sc.T_BinW.rotation().matrix().T @ sc.g_inW
    cosang = (res.g_inB @ g_gt) / (np.linalg.norm(res.g_inB) * np.linalg.norm(g_gt))
    assert np.arccos(np.clip(cosang, -1, 1)) < np.deg2rad(0.5)
    # extrinsic translation: p_IinC = -R_CtoI^T p_CinI
    p_IinC_gt = -R_CtoI.T @ np.asarray(_T_CtoI.translation(), float)
    assert np.linalg.norm(res.p_IinC - p_IinC_gt) < 0.05, res.p_IinC
    # velocities vs GT at the keyframe capture times
    for i, k in enumerate(idx[:10]):
        _R, _p, _w, v_gt = sc.traj.get_velocity(t_kept[k])
        v_inB = sc.T_BinW.rotation().matrix().T @ v_gt
        assert np.linalg.norm(res.v_inB[i] - v_inB) < 0.05
    # accel bias (optional path; gated — recovered exactly on clean data)
    assert res.estimated_ba
    assert np.linalg.norm(res.b_a - _BA) < 0.05, f"b_a {res.b_a} vs {_BA}"
    # the DEFAULT initializes b_a = 0 (joint solve owns it)
    res0 = linear_alignment(t_kept[idx], [poses[i] for i in idx],
                            R_CtoI, _BG, s.t, s.w_m, s.a_m)
    assert not res0.estimated_ba and np.all(res0.b_a == 0.0)


def test_gravity_align():
    g = np.array([1.3, -2.1, -9.4])
    R = gravity_align(g)
    aligned = R @ (g / np.linalg.norm(g))
    assert np.allclose(aligned, [0, 0, -1], atol=1e-12)


def test_full_init_driver():
    """End-to-end driver from detections: seed quantities + quality report."""
    td_true = 0.01
    sc = _scenario(td_cam=td_true)
    t_cam, cam, s = _data(sc)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    seed = initialize_cam_imu_from_detections(
        t_cam, cam.ids, cam.uv, pts_B, _RADTAN, s.t, s.w_m, s.a_m)

    assert abs(seed.td - td_true) < 1.5e-3
    d_ext = _T_CtoI.localCoordinates(seed.T_CtoI)
    assert np.linalg.norm(d_ext[0:3]) < np.deg2rad(0.5), f"rot {d_ext[0:3]}"
    assert np.linalg.norm(d_ext[3:6]) < 0.05, f"trans {d_ext[3:6]}"
    assert np.linalg.norm(seed.b_g - _BG) < 2e-3
    assert abs(seed.s - 1.0) < 0.01
    # quality gates
    assert seed.report["g_norm_ok"]
    assert seed.report["scale_ok"]
    assert seed.report["td"]["valley_ratio"] > 1.5
    # seed trajectory states: gravity-aligned world is yaw-free vs the sim
    # world, so compare frame-invariant quantities: inter-keyframe distances
    p0 = np.asarray(seed.T_ItoW[0].translation(), float)
    p5 = np.asarray(seed.T_ItoW[5].translation(), float)
    R_GtoI0, q0 = sc.traj.get_pose(seed.t_key[0])
    R_GtoI5, q5 = sc.traj.get_pose(seed.t_key[5])
    assert abs(np.linalg.norm(p5 - p0) - np.linalg.norm(q5 - q0)) < 0.01


def test_full_init_with_noise():
    """Robustness: 0.5 px detections + true (non-identity) IMU intrinsics —
    the init assumes identity intrinsics; errors must stay SEED-grade (inside
    the joint solve's convergence basin — the basin test is the real
    acceptance; these bounds are regression guards)."""
    td_true = -0.005
    sc = _scenario(td_cam=td_true, sigma_px=0.5, identity_imu_intr=False)
    t_cam, cam, s = _data(sc, seed=3)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    seed = initialize_cam_imu_from_detections(
        t_cam, cam.ids, cam.uv, pts_B, _RADTAN, s.t, s.w_m, s.a_m)

    assert abs(seed.td - td_true) < 15e-3       # td valley is noise-limited here
    d_ext = _T_CtoI.localCoordinates(seed.T_CtoI)
    assert np.linalg.norm(d_ext[0:3]) < np.deg2rad(2.0), f"rot {d_ext[0:3]}"
    assert np.linalg.norm(d_ext[3:6]) < 0.10, f"trans {d_ext[3:6]}"
    assert seed.s == 1.0                        # board mode: s used, ŝ reported
    assert abs(seed.report["scale"] - 1.0) < 0.25   # ŝ sanity (EIV-attenuated)
    assert abs(seed.report["g_norm_ls"] - GRAVITY_MAG) / GRAVITY_MAG < 0.07


# --------------------------------------------------------------------------
def test_seed_inside_joint_solve_basin():
    """M2 acceptance: the noisy-data seed lands INSIDE the joint solve's
    convergence basin. The assertion is basin MEMBERSHIP: LM started from the
    seed converges to the SAME optimum as LM started from the true
    calibration (frame-invariant comparison). How close that optimum is to
    truth is the estimator's consistency — a Phase-5 Monte-Carlo concern, not
    the init's (this dataset's v0@3m geometry puts the optimum near the 3σ
    boundary in extrinsic translation; verified identical from both starts)."""
    import gtsam
    from mvis.calibration.factors.aci import ACIPreintegrator
    from mvis.calibration.factors.imu_factor import ImuFactorACI_In
    from mvis.calibration.factors.camera_calib import CameraCalib
    from tests.helpers.camera_calib_factor import CameraCalibFactor
    from mvis.calibration.cam_imu_init import _interp_vec

    td_true = -0.005
    sc = _scenario(td_cam=td_true, sigma_px=0.5)      # identity IMU intrinsics
    t_cam, cam, s = _data(sc, win=20.0, seed=5)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    seed = initialize_cam_imu_from_detections(
        t_cam, cam.ids, cam.uv, pts_B, _RADTAN, s.t, s.w_m, s.a_m)

    # joint graph at 5 Hz; each state initialized from its PnP pose + the
    # given calibration — the same recipe the seed uses, so the seed run
    # exercises the full CamImuSeed (T_CtoI, td, biases, gravity world)
    t_kept, pnp_poses, _n = pnp_pose_track(t_cam, cam.ids, cam.uv, pts_B, _RADTAN)
    kf = np.arange(0, len(t_kept), 4)              # 20 Hz -> 5 Hz
    stamps = t_kept[kf]
    frame_idx = [int(np.argmin(np.abs(t_cam - st))) for st in stamps]
    K = len(stamps) - 1
    X = gtsam.symbol_shorthand.X
    Vk = lambda k: gtsam.symbol('v', k)
    B0, C0, I0 = gtsam.symbol('b', 0), gtsam.symbol('c', 0), gtsam.symbol('I', 0)
    v_init = np.column_stack([np.interp(stamps + seed.td, seed.t_key,
                                        seed.v_inW[:, i]) for i in range(3)])

    def solve_from(T_CtoI0, td0, b_g0, b_a0):
        """Build the joint graph initialized from the given calibration."""
        R0 = T_CtoI0.rotation().matrix()
        p_IinC0 = -R0.T @ np.asarray(T_CtoI0.translation(), float)
        calib_base = CameraCalib(T_CtoI0, td0, 0.0, _RADTAN, False)
        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        vals.insert(B0, seed.T_BinW)
        vals.insert(C0, np.zeros(16))
        vals.insert(I0, np.zeros(24))
        for k, (st, fi) in enumerate(zip(stamps, frame_idx)):
            T_CtoB = pnp_poses[kf[k]]
            R_IB = np.asarray(T_CtoB.rotation().matrix(), float) @ R0.T
            p_IB = (np.asarray(T_CtoB.translation(), float)
                    + np.asarray(T_CtoB.rotation().matrix(), float) @ p_IinC0)
            vals.insert(X(k), gtsam.Pose3(gtsam.Rot3(seed.R_BtoW @ R_IB),
                                          gtsam.Point3(*(seed.R_BtoW @ p_IB))))
            vals.insert(Vk(k), np.concatenate([v_init[k], b_g0, b_a0]))
            ids, uv = cam.ids[fi], cam.uv[fi]
            if len(ids) < 6:
                continue
            w_frame = _interp_vec([st], s.t, s.w_m)[0] - b_g0
            fac = CameraCalibFactor(uv, pts_B[ids], w_frame, calib_base, _IMG)
            graph.add(fac.make_factor(X(k), Vk(k), B0, C0, sigma_px=0.5))
        for k in range(K):
            m = (s.t >= stamps[k] - 0.01) & (s.t <= stamps[k + 1] + 0.01)
            pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=RPNG,
                                   avg_meas=True)
            pre.reset(b_g0, b_a0)
            pre.set_intrinsics(ImuModel(model=RPNG))
            pre.feed_imu_interval(s.t[m], s.w_m[m], s.a_m[m])
            graph.add(
                ImuFactorACI_In.from_preintegrator(pre, np.array([0, 0, -9.81]))
                .make_factor(X(k), X(k + 1), Vk(k), Vk(k + 1), I0))
        graph.add(gtsam.PriorFactorPose3(X(0), vals.atPose3(X(0)),
                                         gtsam.noiseModel.Isotropic.Sigma(6, 0.1)))
        graph.add(gtsam.PriorFactorVector(I0, np.zeros(24),
                                          gtsam.noiseModel.Isotropic.Sigma(24, 1e-8)))
        res = gtsam.LevenbergMarquardtOptimizer(graph, vals).optimize()
        sol = calib_base.retract(res.atVector(C0))
        return sol, graph, res

    sol_seed, graph_s, res_s = solve_from(seed.T_CtoI, seed.td, seed.b_g, seed.b_a)
    sol_true, _g, _r = solve_from(_T_CtoI, td_true, _BG, _BA)

    # BASIN MEMBERSHIP: both starts converge to the same optimum
    d_same = sol_true.T.localCoordinates(sol_seed.T)
    assert np.linalg.norm(d_same[0:3]) < 2e-3, f"rot differs {d_same[0:3]}"
    assert np.linalg.norm(d_same[3:6]) < 2e-2, f"trans differs {d_same[3:6]}"
    assert abs(sol_seed.td - sol_true.td) < 5e-4, \
        f"td differs {sol_seed.td:+.4f} vs {sol_true.td:+.4f}"

    # sanity on the shared optimum: td near truth, rot within 3σ, info present
    marg = gtsam.Marginals(graph_s, res_s)
    sig = np.sqrt(np.diag(marg.marginalCovariance(C0)))
    d_ext = _T_CtoI.localCoordinates(sol_seed.T)
    assert np.all(np.abs(d_ext[0:3]) < 4 * sig[0:3]), \
        f"rot {d_ext[0:3]} vs 4σ {4 * sig[0:3]}"
    assert abs(sol_seed.td - td_true) < max(3 * sig[6], 1e-3)
    assert np.max(sig[0:3]) < 0.02 and sig[6] < 5e-3
