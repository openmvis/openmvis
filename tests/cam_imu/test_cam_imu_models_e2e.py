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
"""D4 — end-to-end cam-IMU pipeline for non-gtsam distortion models.

The unit tests (`test_cam_imu_distortion_models`) prove per-factor Jacobian parity;
these drive the WHOLE production path (raw simulated detections + raw IMU -> init
-> joint LM) for a model gtsam can't represent, so init (custom unproject PnP) and
the joint graph (model-agnostic projection) are exercised together.

Two cases:
  * single double_sphere camera recovers T_CtoI / td at the noise floor;
  * a MIXED-model two-camera rig (double_sphere cam0 + radtan cam1, DIFFERENT
    per-cam manifold dim) recovers both bundles AND the rig extrinsic — the
    sharpest mixed-model risk (the joint-marginal block-offset path, which must
    use cam0's DIM as the offset, not a hard-coded 16).
"""
import os

import numpy as np
import gtsam

from mvis.detection.chessboard import Chessboard
from mvis.camera.camera_model import build_camera_model
from mvis.calibration.pipeline_cam_imu import (
    CamImuCalibConfig, run_cam_imu_calibration_multi, format_report)
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                     "sim_trajectories", "calib_3D_v0.txt")
_IMG = (640, 480)
_DS = np.array([350.0, 350.0, 320.0, 240.0, -0.15, 0.55])   # fx,fy,cx,cy,xi,alpha
_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])
_T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                      gtsam.Point3(0.05, -0.02, 0.01))
_TD = 0.01
_CAM_HZ = 20.0
_IMU_HZ = 400.0


class _ModelCal:
    """Adapter giving a `camera.CameraModel` the gtsam-Cal3 `uncalibrate(xy)` API
    the simulator's projector calls: a normalized point (x, y) is the camera-frame
    ray (x, y, 1), so `uncalibrate` == `model.project([x, y, 1])`."""
    def __init__(self, model: str, cal_vec):
        self._m = build_camera_model(model, np.asarray(cal_vec, float))

    def uncalibrate(self, xy):
        uv = self._m.project(np.array([float(xy[0]), float(xy[1]), 1.0]))
        if not np.isfinite(uv).all():
            raise RuntimeError("point out of model range")
        return uv


def _imu_intr():
    from mvis.imu import ImuModel, RPNG
    return ImuModel(model=RPNG)


def _board():
    return Chessboard(7, 7, 0.10)


def test_double_sphere_pipeline_recovers_calibration():
    """Single double_sphere camera: the full pipeline recovers T_CtoI / td and
    drives reprojection to the (noise-free) floor."""
    traj = load_trajectory(_TRAJ, dt=0.05)
    t0 = traj.start_time + 1.0
    sc = make_scenario(traj, t0=t0, depth=3.0, board=_board(),
                       cal=_ModelCal("double_sphere", _DS), T_CtoI=_T_CtoI,
                       imu_intr=_imu_intr(), b_g=_BG, b_a=_BA, image_size=_IMG,
                       td_cam=_TD, sigma_px=0.0)
    win = 25.0
    t_cam = np.arange(t0, t0 + win, 1.0 / _CAM_HZ)
    cam = sc.camera(t_cam, seed=7)
    t_imu = imu_times(t0 - 0.3, t0 + win + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=7)
    board_pts = np.asarray(_board().all_points_3d(), float)

    cfg = CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=_DS, model="double_sphere", image_size=_IMG,
              pin_intrinsics=True)],
        board_pts, s.t, s.w_m, s.a_m, config=cfg)
    print("\n" + format_report(res))
    assert res.ok
    assert res.calib.model == "double_sphere"
    assert res.calib.n_intr == 6
    d0 = _T_CtoI.localCoordinates(res.calib.T)
    assert np.linalg.norm(d0[0:3]) < 1.0e-2, f"rot {d0[0:3]}"
    assert abs(res.calib.td - _TD) < 3e-3, f"td {res.calib.td:+.5f}"
    assert res.diagnostics["reproj_rms_px"] < 0.2


def test_mixed_model_two_camera_rig():
    """double_sphere cam0 (DIM=14) + radtan cam1 (DIM=16) on the same board+IMU.
    Recovers both bundles, the per-cam dims stay distinct, and the rig extrinsic
    T_C1toC0 is recovered through the mixed-dim joint-marginal block offset."""
    import dataclasses

    T_CtoI_2 = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.03, 0.17, -0.02])),
                           gtsam.Point3(-0.12, 0.03, 0.02))
    td2 = -0.005
    traj = load_trajectory(_TRAJ, dt=0.05)
    t0 = traj.start_time + 1.0
    sc = make_scenario(traj, t0=t0, depth=3.0, board=_board(),
                       cal=_ModelCal("double_sphere", _DS), T_CtoI=_T_CtoI,
                       imu_intr=_imu_intr(), b_g=_BG, b_a=_BA, image_size=_IMG,
                       td_cam=_TD, sigma_px=0.0)
    sc2 = dataclasses.replace(sc, cal=_ModelCal("radtan", _RADTAN),
                              T_CtoI=T_CtoI_2, td_cam=td2)
    assert np.allclose(sc2.T_BinW.matrix(), sc.T_BinW.matrix())   # shared board

    win = 25.0
    t_cam1 = np.arange(t0, t0 + win, 1.0 / _CAM_HZ)
    t_cam2 = t_cam1 + 0.5 / _CAM_HZ
    cam1 = sc.camera(t_cam1, seed=11)
    cam2 = sc2.camera(t_cam2, seed=12)
    t_imu = imu_times(t0 - 0.3, t0 + win + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=11)
    board_pts = np.asarray(_board().all_points_3d(), float)

    cfg = CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam1, ids_list=list(cam1.ids), uv_list=list(cam1.uv),
              cal_vec=_DS, model="double_sphere", image_size=_IMG,
              pin_intrinsics=True),
         dict(t_cam=t_cam2, ids_list=list(cam2.ids), uv_list=list(cam2.uv),
              cal_vec=_RADTAN, model="radtan", image_size=_IMG,
              pin_intrinsics=True)],
        board_pts, s.t, s.w_m, s.a_m, config=cfg)
    print("\n" + format_report(res))
    assert res.ok
    assert len(res.cameras) == 2
    # the mixed-dim invariant: the two cameras keep distinct manifold dims
    assert res.cameras[0]["calib"].model == "double_sphere"
    assert res.cameras[0]["calib"].n_intr == 6 and res.cameras[0]["calib"].DIM == 14
    assert res.cameras[1]["calib"].model == "radtan"
    assert res.cameras[1]["calib"].n_intr == 8 and res.cameras[1]["calib"].DIM == 16

    d0 = _T_CtoI.localCoordinates(res.calib.T)
    assert np.linalg.norm(d0[0:3]) < 1.0e-2, f"base rot {d0[0:3]}"
    c1 = res.cameras[1]
    d1 = T_CtoI_2.localCoordinates(c1["calib"].T)
    assert np.linalg.norm(d1[0:3]) < 1.0e-2, f"extra rot {d1[0:3]}"
    # absolute td carries the weakly-observable COMMON-MODE slide pushed by the
    # interpolation chord error on this aggressive trajectory (see
    # test_two_camera_rig); the RELATIVE td is vision-determined and tight.
    assert abs(res.calib.td - _TD) < 8e-3, f"base td {res.calib.td:+.5f}"
    assert abs(c1["calib"].td - td2) < 8e-3, f"extra td {c1['calib'].td:+.5f}"
    d_td_rel = (res.calib.td - c1["calib"].td) - (_TD - td2)
    assert abs(d_td_rel) < 1.5e-3, f"relative td {d_td_rel:+.5f}"

    # rig extrinsic through the mixed-dim joint-marginal block offset
    T_rel_gt = _T_CtoI.inverse().compose(T_CtoI_2)
    d_rel = T_rel_gt.localCoordinates(c1["T_CtoC0"])
    assert np.linalg.norm(d_rel[0:3]) < 1.0e-2, f"rig rot {d_rel[0:3]}"
    assert np.linalg.norm(d_rel[3:6]) < 3e-2, f"rig trans {d_rel[3:6]}"
