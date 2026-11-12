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
"""Cam-IMU reprojection factors: analytic-vs-numeric Jacobian parity for ALL
distortion models (cam-imu-distortion-models D1).

radtan / equidistant go through gtsam Cal3DS2/Cal3Fisheye; fov / double_sphere /
eucm / omni_radtan / kb4 go through custom_projections, chained to the pose/point
Jacobians via gtsam transformTo. This pins every model's analytic factor Jacobian
to finite differences (and so guards the variable-dim manifold from D0).
"""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.calibration.factors.camera_feature_factor import CameraFeatureInterpFactor
from tests.helpers.camera_calib_factor import CameraCalibFactor
from mvis.calibration.factors.projection import make_projector
from mvis.calibration.cam_imu_init import pnp_pose_track

# model -> a valid cal_vec (n_intr = len); wide-FOV params keep corners in range
MODELS = {
    "radtan":        [350., 350, 512, 512, 0.0, 0.0, 0.0, 0.0],
    "equidistant":   [350., 350, 512, 512, 0.0, 0.0, 0.0, 0.0],
    "fov":           [350., 350, 512, 512, 0.9],
    "double_sphere": [350., 350, 512, 512, 0.3, 0.6],
    "eucm":          [350., 350, 512, 512, 0.6, 1.1],
    "ucm":           [350., 350, 512, 512, 0.62],
    # omni: [.., xi, k1, k2, p1, p2] with non-zero k/p so every distortion
    # column (p2 included) is FD-checked away from the degenerate zero point
    "omni_radtan":   [350., 350, 512, 512, 1.2, -0.05, 0.01, 1e-3, -1e-3],
    # kb4: [.., k1..k4], all non-zero so every k column is FD-checked
    "kb4":           [350., 350, 512, 512, 0.02, -0.01, 3e-3, -5e-4],
}

_PK = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.05), gtsam.Point3(0, 0, 0))
_PK1 = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.09), gtsam.Point3(0.01, 0, 0))
_TB = gtsam.Pose3(gtsam.Rot3.Rodrigues(1.3, 0.1, 0), gtsam.Point3(0, 0, 1.5))


def _board_corners(n=12, seed=1):
    rng = np.random.default_rng(seed)
    p_B = np.zeros((n, 3))
    p_B[:, 0] = rng.uniform(-0.15, 0.15, n)
    p_B[:, 1] = rng.uniform(-0.15, 0.15, n)
    return p_B


@pytest.mark.parametrize("model", list(MODELS))
def test_interp_factor_jacobians(model):
    """Board-mode (interpolated-pose) factor — the 'cam calib mode' deliverable."""
    p_B = _board_corners()
    cv = np.array(MODELS[model])
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, cv, model=model)
    # synthesize in-range uv from the forward model at xi=0
    f0 = CameraCalibInterpFactor(np.zeros((len(p_B), 2)), p_B, 0.01, 0.05,
                                 base, (1024, 1024))
    uv = f0.error(_PK, _PK1, _TB, np.zeros(base.DIM)).reshape(-1, 2)
    f = CameraCalibInterpFactor(uv, p_B, 0.01, 0.05, base, (1024, 1024))
    xi = np.zeros(base.DIM)
    xi[6] = 1e-4                                   # exercise td/tr columns
    _, J = f.error_and_jacobians(_PK, _PK1, _TB, xi)
    Jn = f._numerical_jacobians(_PK, _PK1, _TB, xi)
    assert len(J[3].T) == base.DIM                 # H_calib width == manifold dim
    for k in range(4):
        assert np.abs(J[k] - Jn[k]).max() < 1e-4, f"{model} block {k}"


@pytest.mark.parametrize("model", list(MODELS))
def test_single_state_factor_jacobians(model):
    """Single-state board factor (Exp(w·td) propagation)."""
    p_B = _board_corners()
    cv = np.array(MODELS[model])
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, cv, model=model)
    w_m = np.array([0.1, -0.05, 0.2])
    pose = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.05), gtsam.Point3(0, 0, 0))
    vb = np.array([0.2, -0.1, 0.05, 0, 0, 0, 0, 0, 0])
    f0 = CameraCalibFactor(np.zeros((len(p_B), 2)), p_B, w_m, base, (1024, 1024))
    uv = f0.error(pose, vb, _TB, np.zeros(base.DIM)).reshape(-1, 2)
    f = CameraCalibFactor(uv, p_B, w_m, base, (1024, 1024))
    xi = np.zeros(base.DIM)
    xi[6] = 1e-4
    _, J = f.error_and_jacobians(pose, vb, _TB, xi)
    Jn = f._numerical_jacobians(pose, vb, _TB, xi)
    for k in range(4):
        assert np.abs(J[k] - Jn[k]).max() < 1e-4, f"{model} block {k}"


@pytest.mark.parametrize("model", ["radtan", "equidistant", "double_sphere", "eucm", "ucm",
                                   "omni_radtan", "kb4"])
def test_feature_factor_jacobians(model):
    """Feature-mode (geometry-free world landmarks) factor."""
    p_B = _board_corners(n=10, seed=2)
    P_world = np.array([_TB.transformFrom(gtsam.Point3(*p)) for p in p_B])
    cv = np.array(MODELS[model])
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, cv, model=model)
    ids = list(range(len(p_B)))
    f0 = CameraFeatureInterpFactor(np.zeros((len(p_B), 2)), ids, 0.01, 0.05,
                                   base, (1024, 1024), z_min=0.0)
    uv = f0.error(_PK, _PK1, np.zeros(base.DIM), P_world).reshape(-1, 2)
    f = CameraFeatureInterpFactor(uv, ids, 0.01, 0.05, base, (1024, 1024), z_min=0.0)
    xi = np.zeros(base.DIM)
    xi[6] = 1e-4
    _, J = f.error_and_jacobians(_PK, _PK1, xi, P_world)
    # analytic blocks: [H_pk, H_pk1, H_calib, H_L_0..]. Pose blocks share the
    # projector with the interp factor (FD-checked there); here FD-check the
    # feature-specific blocks — H_calib (vs xi) and the landmark Jacobians — with
    # a self-contained vector FD (the numdiff helper can't perturb the 2-D P_world).
    def _fd(fn, x, eps=1e-6):
        f0 = fn(x)
        Jm = np.zeros((len(f0), len(x)))
        for j in range(len(x)):
            xp = x.copy(); xp[j] += eps
            Jm[:, j] = (fn(xp) - f0) / eps
        return Jm

    J_calib_fd = _fd(lambda v: f.error(_PK, _PK1, v, P_world), xi.copy())
    assert np.abs(J[2] - J_calib_fd).max() < 1e-4, f"{model} H_calib"
    J_L_fd = _fd(lambda v: f.error(_PK, _PK1, xi, v.reshape(-1, 3)),
                 P_world.reshape(-1).copy())
    assert np.abs(np.hstack(J[3:]) - J_L_fd).max() < 1e-4, f"{model} landmarks"


@pytest.mark.parametrize("model,outliers", [(m, 0) for m in MODELS]
                         + [(m, 3) for m in MODELS if m != "radtan"])
def test_pnp_pose_track_recovers_poses(model, outliers):
    """D2: model-aware init PnP recovers the camera-to-board poses for every model
    (radtan via OpenCV, the rest via the camera-only init's `board_pose_pnp` on
    bearings). G9: that solve has RANSAC, so 3 corners off by 60 px per frame
    do not move the bearing models' poses."""
    rng = np.random.default_rng(7)
    cv = np.array(MODELS[model])
    gx, gy = np.meshgrid(np.linspace(-0.25, 0.25, 6), np.linspace(-0.25, 0.25, 6))
    p_B = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    ids = np.arange(len(p_B))
    proj = make_projector(model, cv)

    t_cam, ids_list, uv_list, gt = [], [], [], []
    for k in range(16):
        # board ~1.2 m in FRONT of the camera (board points have +z in cam frame)
        T_BtoC = gtsam.Pose3(
            gtsam.Rot3.Rodrigues(0.12 * np.sin(k * 0.3), 0.10 * np.cos(k * 0.2),
                                 0.08 * np.sin(k * 0.4)),
            gtsam.Point3(0.06 * np.sin(k * 0.5), 0.04 * np.cos(k * 0.3), 1.2))
        uv = np.array([proj.project(gtsam.Pose3(),
                                    T_BtoC.transformFrom(gtsam.Point3(*p)),
                                    jac=False)[0] for p in p_B])
        uv[rng.choice(len(uv), outliers, replace=False)] += rng.choice([-60.0, 60.0], (outliers, 2))
        t_cam.append(float(k)); ids_list.append(ids); uv_list.append(uv)
        gt.append(T_BtoC.inverse())            # pnp_pose_track returns T_CtoB

    t_kept, poses, _ = pnp_pose_track(t_cam, ids_list, uv_list, p_B, cv, model=model)
    assert len(poses) >= 12, f"{model}: only {len(poses)} PnP frames"
    gt_by_t = {float(k): gt[k] for k in range(16)}
    for t, P in zip(t_kept, poses):
        G = gt_by_t[t]
        dp = np.linalg.norm(np.asarray(P.translation()) - np.asarray(G.translation()))
        dR = np.linalg.norm(gtsam.Rot3.Logmap(P.rotation().between(G.rotation())))
        assert dp < 5e-3, f"{model} t={t}: trans err {dp:.4f} m"
        assert dR < 5e-3, f"{model} t={t}: rot err {dR:.4f} rad"


def test_omni_radtan_9_board_mode_pipeline_e2e():
    """Board-mode cam-IMU end-to-end on omni-radtan data with p2 != 0: the
    9-param bundle (n_intr 9, DIM 17) flows through init PnP (bearings) and the
    joint LM. The observations come from an independent inline Mei + radtan
    formula (not the project_omni_radtan kernel the solver uses), and the
    intrinsics are pinned at the true 9-vector (p2 large, 2e-2), so a p2
    dropped anywhere on the solver side (kernel, CameraCalib or the factors)
    leaves model error the extrinsics only partly absorb: the rotation biases
    by ~0.02 rad and the noise-free RMS rises from ~6e-4 to ~3e-2 px, failing
    both floors. p2 *estimation* is covered by the camera-only e2e in
    test_omni_p2.py."""
    from mvis.detection.chessboard import Chessboard
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, run_cam_imu_calibration_multi)
    from mvis.imu import ImuModel, RPNG
    from mvis.simulation.trajectory_io import load_trajectory
    from mvis.simulation.imu_measurements import imu_times
    from mvis.simulation.scenario import make_scenario
    import os

    class _MeiRadtanCal:      # independent Mei + radtan -> gtsam-Cal3 `uncalibrate`
        def __init__(self, cal_vec):
            self.c = np.asarray(cal_vec, float)

        def uncalibrate(self, xy):
            fx, fy, cx, cy, xi, k1, k2, p1, p2 = self.c
            P = np.array([float(xy[0]), float(xy[1]), 1.0])
            Xs = P / np.linalg.norm(P)                 # onto the unit sphere
            mx, my = Xs[0] / (Xs[2] + xi), Xs[1] / (Xs[2] + xi)
            r2 = mx * mx + my * my
            rad = 1.0 + k1 * r2 + k2 * r2 * r2
            xd = mx * rad + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
            yd = my * rad + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
            return np.array([fx * xd + cx, fy * yd + cy])

    omni = np.array([330.0, 331.0, 321.0, 239.0, 1.1, -0.05, 0.01, 1e-3, 2e-2])
    T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                         gtsam.Point3(0.05, -0.02, 0.01))
    td, img, board = 0.01, (640, 480), Chessboard(7, 7, 0.10)
    traj = load_trajectory(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                                        "sim_trajectories", "calib_3D_v0.txt"), dt=0.05)
    t0 = traj.start_time + 1.0
    sc = make_scenario(traj, t0=t0, depth=3.0, board=board,
                       cal=_MeiRadtanCal(omni), T_CtoI=T_CtoI,
                       imu_intr=ImuModel(model=RPNG), b_g=np.array([0.01, -0.02, 0.005]),
                       b_a=np.array([0.05, -0.03, 0.02]), image_size=img,
                       td_cam=td, sigma_px=0.0)
    win = 15.0
    t_cam = np.arange(t0, t0 + win, 1.0 / 20.0)
    cam = sc.camera(t_cam, seed=7)
    s = sc.base_imu(imu_times(t0 - 0.3, t0 + win + 0.3, 400.0), seed=7)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=omni, model="omni_radtan", image_size=img, pin_intrinsics=True)],
        np.asarray(board.all_points_3d(), float), s.t, s.w_m, s.a_m,
        config=CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0))
    assert res.ok
    assert res.calib.model == "omni_radtan"
    assert res.calib.n_intr == 9 and res.calib.DIM == 17
    np.testing.assert_allclose(res.calib.intr, omni, atol=1e-6)     # pinned, p2 kept
    d0 = T_CtoI.localCoordinates(res.calib.T)
    assert np.linalg.norm(d0[0:3]) < 1.0e-2, f"rot {d0[0:3]}"
    assert abs(res.calib.td - td) < 3e-3, f"td {res.calib.td:+.5f}"
    assert res.diagnostics["reproj_rms_px"] < 0.01


# ── mixed-model and feature-mode rigs through the custom kernels ──

_KB4 = np.array([300.0, 301.0, 321.0, 239.0, 0.02, -0.01, 3e-3, -5e-4])
_OMNI9 = np.array([330.0, 331.0, 321.0, 239.0, 1.1, -0.05, 0.01, 1e-3, 2e-3])


class _ModelCal:
    """CameraModel -> the simulator's gtsam-Cal3 `uncalibrate(xy)` API."""
    def __init__(self, model, cal_vec):
        from mvis.camera.camera_model import build_camera_model
        self._m = build_camera_model(model, np.asarray(cal_vec, float))

    def uncalibrate(self, xy):
        uv = self._m.project(np.array([float(xy[0]), float(xy[1]), 1.0]))
        if not np.isfinite(uv).all():
            raise RuntimeError("point out of model range")
        return uv


def _custom_model_scenario(cal, model, T_CtoI, td):
    import os
    from mvis.detection.chessboard import Chessboard
    from mvis.imu import ImuModel, RPNG
    from mvis.simulation.trajectory_io import load_trajectory
    from mvis.simulation.scenario import make_scenario
    board = Chessboard(7, 7, 0.10)
    traj = load_trajectory(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                                        "sim_trajectories", "calib_3D_v0.txt"), dt=0.05)
    t0 = traj.start_time + 1.0
    sc = make_scenario(traj, t0=t0, depth=3.0, board=board, cal=_ModelCal(model, cal),
                       T_CtoI=T_CtoI, imu_intr=ImuModel(model=RPNG),
                       b_g=np.array([0.01, -0.02, 0.005]),
                       b_a=np.array([0.05, -0.03, 0.02]), image_size=(640, 480),
                       td_cam=td, sigma_px=0.0)
    return sc, board, t0


def test_mixed_omni9_kb4_rig_e2e():
    """omni_radtan cam0 (9 params, DIM 17) + kb4 cam1 (DIM 16): per-cam dims stay
    distinct and the rig extrinsic is recovered through the mixed-dim
    joint-marginal block offset (cam0's DIM, not 16)."""
    import dataclasses
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, run_cam_imu_calibration_multi)
    from mvis.simulation.imu_measurements import imu_times
    T0 = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                     gtsam.Point3(0.05, -0.02, 0.01))
    T1 = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.03, 0.17, -0.02])),
                     gtsam.Point3(-0.12, 0.03, 0.02))
    sc, board, t0 = _custom_model_scenario(_OMNI9, "omni_radtan", T0, 0.01)
    sc1 = dataclasses.replace(sc, cal=_ModelCal("kb4", _KB4),
                              T_CtoI=T1, td_cam=-0.005)
    win = 20.0
    t_c0 = np.arange(t0, t0 + win, 1.0 / 20.0)
    t_c1 = t_c0 + 0.025
    c0, c1 = sc.camera(t_c0, seed=11), sc1.camera(t_c1, seed=12)
    s = sc.base_imu(imu_times(t0 - 0.3, t0 + win + 0.3, 400.0), seed=11)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_c0, ids_list=list(c0.ids), uv_list=list(c0.uv), cal_vec=_OMNI9,
              model="omni_radtan", image_size=(640, 480), pin_intrinsics=True),
         dict(t_cam=t_c1, ids_list=list(c1.ids), uv_list=list(c1.uv), cal_vec=_KB4,
              model="kb4", image_size=(640, 480), pin_intrinsics=True)],
        np.asarray(board.all_points_3d(), float), s.t, s.w_m, s.a_m,
        config=CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0))
    assert res.ok and len(res.cameras) == 2
    k0, k1 = res.cameras[0]["calib"], res.cameras[1]["calib"]
    assert (k0.model, k0.n_intr, k0.DIM) == ("omni_radtan", 9, 17)
    assert (k1.model, k1.n_intr, k1.DIM) == ("kb4", 8, 16)
    assert len(res.cameras[0]["calib_sigma"]) == 17
    assert np.linalg.norm(T0.localCoordinates(k0.T)[0:3]) < 1.0e-2
    assert np.linalg.norm(T1.localCoordinates(k1.T)[0:3]) < 1.0e-2
    d_rel = T0.inverse().compose(T1).localCoordinates(res.cameras[1]["T_CtoC0"])
    assert np.linalg.norm(d_rel[0:3]) < 1.0e-2, f"rig rot {d_rel[0:3]}"
    assert np.linalg.norm(d_rel[3:6]) < 3e-2, f"rig trans {d_rel[3:6]}"
    d_td_rel = (k0.td - k1.td) - (0.01 + 0.005)
    assert abs(d_td_rel) < 1.5e-3, f"relative td {d_td_rel:+.5f}"


# ── feature mode (anonymous tracks): the cam's model reaches SfM + CameraCalib ──

@pytest.mark.parametrize("model,cal", [("omni_radtan", _OMNI9), ("kb4", _KB4)])
def test_feature_multi_respects_model(model, cal):
    """Feature-mode multi-cam e2e (pinned intrinsics): the SfM seed undistorts
    through the cam's own model (normalize_detections(..., model=)) and the
    CameraCalib bundle carries it (n_intr 9 / 8). Before the fix the model
    was dropped: an omni9 / kb4 cal_vec went to cv2 as radtan coefficients
    and into CameraCalib as model='radtan'."""
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, run_cam_imu_calibration_feature_multi)
    from mvis.simulation.imu_measurements import imu_times
    T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                         gtsam.Point3(0.05, -0.02, 0.01))
    sc, board, t0 = _custom_model_scenario(cal, model, T_CtoI, 0.01)
    win = 20.0
    t_cam = np.arange(t0, t0 + win, 1.0 / 20.0)
    cam = sc.camera(t_cam, seed=3)
    s = sc.base_imu(imu_times(t0 - 0.3, t0 + win + 0.3, 400.0), seed=3)
    res = run_cam_imu_calibration_feature_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=cal, model=model, is_fisheye=True, image_size=(640, 480))],
        s.t, s.w_m, s.a_m, sfm_mode="planar",
        config=CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0))
    k = res.cameras[0]["calib"]
    assert k.model == model and k.n_intr == len(cal)
    np.testing.assert_allclose(k.intr, cal, atol=1e-6)            # pinned
    # feature mode on one small planar target is weak on td / translation even
    # for radtan here (td +-5 ms, p_CinI cm-level), so score the rotation and
    # the reprojection fit, which a wrong (radtan) undistortion would break
    assert np.linalg.norm(T_CtoI.localCoordinates(k.T)[0:3]) < 1.0e-2
    assert res.cameras[0]["reproj_rms_px"] < 0.5, res.cameras[0]["reproj_rms_px"]
