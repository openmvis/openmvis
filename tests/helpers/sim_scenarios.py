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
"""Simulated scenarios on calib_3D_v0 for the cam-IMU init and pipeline tests, and
the small closed-loop ground truth for the recovery guards."""
import os
import re

import numpy as np
import gtsam

from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.simulation.from_result import GroundTruth
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario
from mvis.simulation.synth import build_synth
from mvis.simulation.trajectory_io import load_trajectory

# calib_3D_v0 @ 3 m depth keeps the board ~75% visible across a 40 s window
# (v28 ranges too far for a fixed board); the alignment's max_interval gate
# drops the blind-gap intervals.
_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                     "sim_trajectories", "calib_3D_v0.txt")
_IMG = (640, 480)
_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])
_T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                      gtsam.Point3(0.05, -0.02, 0.01))
_WIN = 40.0          # init window (s) — long by design (offline calibration)
_CAM_HZ = 20.0
_IMU_HZ = 400.0
_MARGIN = 0.3        # IMU stream margin past the camera window


def scenario_for_init(td_cam=0.0, sigma_px=0.0, identity_imu_intr=True):
    traj = load_trajectory(_TRAJ, dt=0.05)
    board = Chessboard(7, 7, 0.10)
    intr = (ImuModel(model=RPNG) if identity_imu_intr else ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.zeros(9),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(), model=RPNG))
    return make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=board,
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=intr, b_g=_BG, b_a=_BA, image_size=_IMG,
        td_cam=td_cam, sigma_px=sigma_px)


def data_for_init(sc, win=_WIN, cam_hz=_CAM_HZ, seed=0):
    """Camera detections + raw IMU stream over the init window."""
    t0 = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + win, 1.0 / cam_hz)
    cam = sc.camera(t_cam, seed=seed)
    t_imu = imu_times(t0 - _MARGIN, t0 + win + _MARGIN, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=seed)
    return t_cam, cam, s

_TD = 0.01


def scenario_for_pipeline(sigma_px=0.0):
    traj = load_trajectory(_TRAJ, dt=0.05)
    return make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG),     # identity: phase A pins honest
        b_g=_BG, b_a=_BA, image_size=_IMG, td_cam=_TD, sigma_px=sigma_px)


def data_for_pipeline(sc, win=_WIN, seed=0):
    t0 = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + win, 1.0 / _CAM_HZ)
    cam = sc.camera(t_cam, seed=seed)
    t_imu = imu_times(t0 - 0.3, t0 + win + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=seed)
    return t_cam, cam, s

_AUX_WIN = 5.0   # short window for speed


def _make_aux_scenario():
    """Return scenario kwargs for a 1-cam + base-IMU + 1-aux-IMU setup."""
    from mvis.detection.chessboard import Chessboard
    from mvis.imu import ImuModel, RPNG
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
    from mvis.simulation.trajectory_io import load_trajectory
    from mvis.simulation.imu_measurements import imu_times
    from mvis.simulation.scenario import make_scenario, AuxImu

    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    bg_s = np.array([-0.008, 0.015, -0.01])
    ba_s = np.array([0.02, -0.04, 0.03])
    td_s = 0.004

    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=0.0,
        aux=AuxImu(T_IstoI=T_IstoI, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_s))

    t0 = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + _AUX_WIN, 1.0 / _CAM_HZ)
    cam = sc.camera(t_cam, seed=21)
    t_imu = imu_times(t0 - 0.3, t0 + _AUX_WIN + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=21)
    t_aux = imu_times(t0 - 0.3, t0 + _AUX_WIN + 0.3, 250.0)
    sa = sc.aux_imu(t_aux, seed=22)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    return dict(
        t_cam=t_cam, cam=cam, s=s, sa=sa,
        board_pts=board_pts,
    )

# robustly-observable camera-side params — the estimator-correctness signal
_STRICT = re.compile(r"^cam\d+/(fx|fy|cx|cy|k1|k2|p1|p2|rot\d|pos0|pos1)$")


def _small_gt():
    dt = 1.0 / 30.0
    ts = 1.0 + np.arange(240) * dt              # 8 s
    T_states = []
    for t in ts:
        M = np.eye(4)
        M[:3, :3] = (gtsam.Rot3.Rz(0.5 * np.sin(1.3 * t))
                     .compose(gtsam.Rot3.Rx(0.3 * np.sin(0.7 * t)))).matrix()
        M[:3, 3] = [0.3 * np.sin(t), 0.3 * np.cos(0.8 * t), 0.1 * np.sin(0.5 * t)]
        T_states.append(M.tolist())
    Tb = np.eye(4); Tb[:3, 3] = [0.0, 0.0, 1.3]
    return GroundTruth(
        cams=[dict(intr8=[400., 405., 320., 240., -0.05, 0.01, 0.0, 0.0],
                   T_CtoI=np.eye(4).tolist(), td=0.003, is_fisheye=False,
                   image_size=[640, 480])],
        imus=[dict(name="base", is_base=True, xi24=[0.0] * 24,
                   T_IstoI=None, td=0.0)],
        b_g=[0.001, 0.0, -0.001], b_a=[0.01, -0.01, 0.0],
        T_BinW=Tb.tolist(), t_states=ts.tolist(), T_states=T_states,
        imu_model="calyx", g_inW=[0.0, 0.0, -9.81])


def _polled_scenario(T_s=1.0 / 11.0, P_pub=1.0 / 9.0, latency=0.004, seed=11):
    """Camera free-running on a T_s sensor clock, published + stamped by a
    P_pub polling loop that takes the latest finished frame (the ELP/usb_cam
    pattern): images at the sensor capture times, stamps on the polling grid."""
    gt = _small_gt()
    gt.board_points_local = np.array(
        [[0.05 * j, 0.05 * i, 0.0] for i in range(6) for j in range(6)])
    ts = np.asarray(gt.t_states)
    sensor = ts[6] + 0.013 + np.arange(0, int((ts[-6] - ts[6]) / T_s)) * T_s
    stamps = np.arange(sensor[0] + 0.05, sensor[-1], P_pub)
    cap = sensor[np.searchsorted(sensor, stamps - latency) - 1]
    imu_t = np.arange(ts[2], ts[-2], 1.0 / 400.0)
    cams, t_imu, w_m, a_m, aux = build_synth(
        gt, [cap], [imu_t], sigma_px=0.5,
        imu_sigmas=dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002),
        seed=seed)
    order = {float(c): s for c, s in zip(cap, stamps)}
    cams[0]["t_cam"] = np.asarray([order[float(c)] for c in cams[0]["t_cam"]])
    return gt, cams, t_imu, w_m, a_m, aux, T_s
