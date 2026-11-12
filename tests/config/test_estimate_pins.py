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
"""U3 — the unified estimate: hard pins actually HOLD their quantity (cam-IMU).

A `False` in the estimate: block adds a _TINY-sigma prior on that quantity's calib
component, so its posterior marginal sigma collapses to ~0 while the components left
free keep a normal (cm / ms / unit) marginal. We assert exactly that — one solve per
test, with an internal control (a free component in the same solve stays non-tiny).

Covers: base cam-IMU extrinsic (sig_C[0:6]) + base td (sig_C[6]); aux-IMU extrinsic
(_As[0:6]) + inter-IMU td (_As[6]). Free-baseline magnitudes are pinned by
test_pipeline_cam_imu (extrinsic ~cm, td ~ms).
"""

import os

import numpy as np
import gtsam

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.calibration.pipeline_cam_imu import (
    CamImuCalibConfig, run_cam_imu_calibration, run_cam_imu_calibration_multi)
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                     "sim_trajectories", "calib_3D_v0.txt")
_IMG = (640, 480)
_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])
_T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                      gtsam.Point3(0.05, -0.02, 0.01))
_TD = 0.01
_TINY_TOL = 1e-4     # a _TINY-prior-pinned marginal collapses well below this


def _scenario(sigma_px=0.0):
    traj = load_trajectory(_TRAJ, dt=0.05)
    return make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=sigma_px)


def _base_data(sc, win=25.0, seed=3):
    t0 = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + win, 1.0 / 20.0)
    cam = sc.camera(t_cam, seed=seed)
    t_imu = imu_times(t0 - 0.3, t0 + win + 0.3, 400.0)
    s = sc.base_imu(t_imu, seed=seed)
    return t_cam, cam, s


def test_pin_base_extrinsic_and_td_holds():
    """estimate.camera.extrinsics=false / td=false -> the cam-IMU extrinsic
    (calib_sigma[0:6]) and td (calib_sigma[6]) marginals collapse to ~0, while
    the freely-estimated camera intrinsics (calib_sigma[8:16]) stay non-tiny."""
    sc = _scenario(sigma_px=0.0)
    t_cam, cam, s = _base_data(sc)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)
    cfg = CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0,
                            imu_intrinsics="pinned",
                            estimate_extrinsics=False, estimate_td=False)
    res = run_cam_imu_calibration(t_cam, list(cam.ids), list(cam.uv), board_pts,
                                  _RADTAN, s.t, s.w_m, s.a_m, _IMG, config=cfg)
    assert res.ok
    sig = np.asarray(res.calib_sigma, float)
    # pinned: extrinsic + td marginals collapsed
    assert np.all(sig[0:6] < _TINY_TOL), f"extrinsic not pinned: {sig[0:6]}"
    assert sig[6] < _TINY_TOL, f"td not pinned: {sig[6]}"
    # control: the freely-estimated intrinsics keep a real marginal
    assert np.any(sig[8:16] > _TINY_TOL), f"intrinsics unexpectedly tiny: {sig[8:16]}"


def test_pin_aux_extrinsic_and_td_holds():
    """estimate.imu.extrinsics=false / td=false -> the aux-IMU extrinsic
    (imu calib_sigma[0:6]) and inter-IMU td ([6]) marginals collapse, while the
    base camera extrinsic (left free) keeps a real marginal in the same solve."""
    from mvis.simulation.scenario import AuxImu

    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    td_s, bg_s = 0.004, np.array([-0.008, 0.015, -0.01])
    ba_s = np.array([0.02, -0.04, 0.03])
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=0.5,
        aux=AuxImu(T_IstoI=T_IstoI, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_s))
    t_cam, cam, s = _base_data(sc, win=30.0, seed=21)
    t0 = sc.traj.start_time + 1.0
    t_aux = imu_times(t0 - 0.3, t0 + 30.3, 250.0)
    sa = sc.aux_imu(t_aux, seed=22)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # pin the AUX extrinsic+td; leave the base camera free (the control).
    cfg = CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=2.0,
                            imu_intrinsics="pinned",
                            estimate_aux_extrinsics=False, estimate_aux_td=False)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=_RADTAN, image_size=_IMG)],
        board_pts, s.t, s.w_m, s.a_m,
        aux_imus=[dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux')],
        config=cfg)
    assert res.ok and len(res.imus) == 1
    asig = np.asarray(res.imus[0]['calib_sigma'], float)
    assert np.all(asig[0:6] < _TINY_TOL), f"aux extrinsic not pinned: {asig[0:6]}"
    assert asig[6] < _TINY_TOL, f"aux td not pinned: {asig[6]}"
    # control: base camera extrinsic was left free -> real marginal
    bsig = np.asarray(res.calib_sigma, float)
    assert np.any(bsig[0:6] > _TINY_TOL), f"base extrinsic unexpectedly tiny: {bsig[0:6]}"
