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
"""Seeded multi-board cam-IMU recovery under pixel noise (acceptance gate).

Full-motion fisheye rig + 4 known 1 m boards; camera intrinsics PINNED (given).
The hypothesis: full motion + multi-board observation recovers the weak cam-IMU
lever. Single seed; tolerances are loose enough for sigma_px=0.5 px corner noise.
"""
import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG, create_imu_model
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.trajectories import load_sine3d
from mvis.simulation.multi_board import (
    place_boards_ring, MultiBoardScenario, forward_T_CtoI)
from mvis.calibration.multi_board_pipeline import run_multi_board_calibration

T265_FISHEYE = np.array([288.74401, 288.91833, 416.14622, 405.97177,
                         -0.00777028, 0.04423187, -0.04124983, 0.00730763])


@pytest.mark.slow
def test_multi_board_recovers_lever_within_tol():
    traj = load_sine3d(target_span=1.0)
    board = Chessboard(11, 11, 0.10)
    poses = place_boards_ring(board, radius=1.5, directions=4)
    T_CtoI = forward_T_CtoI()
    # NON-identity IMU intrinsics so Dw/Da estimation is actually exercised
    intr = ImuModel(x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
                    x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
                    x_tg=np.zeros(9), R=np.eye(3), model=RPNG)
    sc = MultiBoardScenario(
        traj=traj, boards=[board] * 4, T_BinW_list=poses,
        cal=_cal_from_vector(T265_FISHEYE, True), is_fisheye=True, T_CtoI=T_CtoI,
        image_size=(848, 800), imu_intr=intr, b_g=np.zeros(3), b_a=np.zeros(3),
        sigma_px=0.5)
    t = imu_times(traj.start_time, traj.start_time + 68.0, 10.0)
    cam_stream = sc.camera(t, seed=1)
    imu = sc.base_imu(imu_times(traj.start_time, traj.start_time + 68.0, 200.0), seed=1)
    cam = dict(t_cam=cam_stream.t, ids_list=cam_stream.ids, uv_list=cam_stream.uv,
               cal_vec=T265_FISHEYE, is_fisheye=True, image_size=(848, 800))
    bpts = {b: sc.board_points(b) for b in range(4)}
    res = run_multi_board_calibration(cam, bpts, imu.t, imu.w_m, imu.a_m)
    assert res.ok
    lever_mm = 1e3 * np.linalg.norm(
        np.asarray(res.calib.T.translation()) - np.asarray(T_CtoI.translation()))
    assert lever_mm < 20.0, f"lever error {lever_mm:.1f} mm under full motion"
    # td recovered (GT 0)
    assert abs(res.calib.td) < 2e-3, f"td {res.calib.td*1e3:.2f} ms"
    # IMU gyro intrinsics (Dw) ON and recovered: gyro scale is a key weak param
    est = create_imu_model(RPNG).retract(res.imu_intr_xi)
    dw_err = np.max(np.abs(np.diag(est.Dw()) - np.diag(intr.Dw())))
    assert dw_err < 0.03, f"gyro Dw diag error {dw_err:.3f}"
    # all 4 board poses recovered (relative to the anchor)
    bp = res.diagnostics['board_poses']
    for b in (1, 2, 3):
        rel_gt = poses[0].between(poses[b])
        rel_est = gtsam.Pose3(np.array(bp[0])).between(gtsam.Pose3(np.array(bp[b])))
        dt = 1e3 * np.linalg.norm(np.asarray(rel_gt.translation())
                                  - np.asarray(rel_est.translation()))
        assert dt < 30.0, f"board {b} off by {dt:.1f} mm"
