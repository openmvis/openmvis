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
"""Hybrid multi-board init: direct anchor-PnP when boards are co-visible (fisheye)
and IMU-bridge fallback when they are not (pinhole). Both must localize the 3
non-anchor boards to ground truth."""

import numpy as np
import gtsam

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.trajectories import load_sine3d
from mvis.simulation.multi_board import (
    place_boards_ring, MultiBoardScenario, forward_T_CtoI)
from mvis.calibration.multi_board_init import init_multi_board

T265_FISHEYE = np.array([288.74401, 288.91833, 416.14622, 405.97177,
                         -0.00777028, 0.04423187, -0.04124983, 0.00730763])
PINHOLE = np.array([400., 400., 320., 240., -0.28, 0.07, 0.0005, -0.0004])


def _scn(cal_vec, is_fisheye, board, image_size):
    traj = load_sine3d(target_span=1.0)
    poses = place_boards_ring(board, radius=1.5, directions=4)
    intr = ImuModel(x_dw=np.array([1.0, 0, 1.0, 0, 0, 1.0]),
                    x_da=np.array([1.0, 0, 1.0, 0, 0, 1.0]), x_tg=np.zeros(9),
                    R=np.eye(3), model=RPNG)
    return poses, MultiBoardScenario(
        traj=traj, boards=[board] * 4, T_BinW_list=poses,
        cal=_cal_from_vector(cal_vec, is_fisheye), is_fisheye=is_fisheye,
        T_CtoI=forward_T_CtoI(), image_size=image_size, imu_intr=intr,
        b_g=np.zeros(3), b_a=np.zeros(3), sigma_px=0.0)


def _run(cal_vec, is_fisheye, board, image_size):
    poses_gt, sc = _scn(cal_vec, is_fisheye, board, image_size)
    t = imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 10.0)
    cam = sc.camera(t)
    imu = sc.base_imu(imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 200.0))
    bpts = {b: sc.board_points(b) for b in range(4)}
    model = "equidistant" if is_fisheye else "radtan"
    seed, board_poses, report = init_multi_board(
        cam.t, cam.ids, cam.uv, bpts, cal_vec, imu.t, imu.w_m, imu.a_m,
        anchor_board=0, model=model)
    return poses_gt, board_poses, report


def _assert_boards(poses_gt, board_poses, dR_tol, dt_tol_mm):
    for b in (1, 2, 3):
        T_rel_gt = poses_gt[0].between(poses_gt[b])
        T_rel_est = board_poses[0].between(board_poses[b])
        dR = np.rad2deg(np.linalg.norm(gtsam.Rot3.Logmap(
            T_rel_gt.rotation().inverse().compose(T_rel_est.rotation()))))
        dt_mm = 1e3 * np.linalg.norm(
            np.asarray(T_rel_gt.translation()) - np.asarray(T_rel_est.translation()))
        assert dR < dR_tol, f"board {b} rot init off {dR:.2f} deg"
        assert dt_mm < dt_tol_mm, f"board {b} trans init off {dt_mm:.0f} mm"


def test_fisheye_direct_co_visible_init():
    # adjacent boards (1,3) are co-visible with the anchor -> DIRECT; the opposite
    # board (2) is 180 deg away, never co-visible even at 163 deg -> BRIDGE. So the
    # fisheye case exercises BOTH modes; what matters is all 3 land on GT.
    poses_gt, board_poses, report = _run(
        T265_FISHEYE, True, Chessboard(11, 11, 0.10), (848, 800))
    modes = {b: v['mode'] for b, v in report['per_board'].items()}
    assert sum(m == 'direct' for m in modes.values()) >= 2, modes  # the adjacent pair
    _assert_boards(poses_gt, board_poses, dR_tol=1.5, dt_tol_mm=30.0)


def test_pinhole_imu_bridge_init():
    # never co-visible -> IMU-bridge fallback localizes the other 3 boards
    poses_gt, board_poses, report = _run(
        PINHOLE, False, Chessboard(7, 7, 0.04), (640, 480))
    assert all(v['mode'] == 'bridge' for v in report['per_board'].values())
    _assert_boards(poses_gt, board_poses, dR_tol=2.0, dt_tol_mm=60.0)
