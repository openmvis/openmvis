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
"""Multi-board scenario: ring placement, board-tagged IDs, and the two camera
regimes — t265 fisheye (all boards co-visible) vs pinhole (sparse, non-co-visible)."""

import numpy as np

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
import gtsam
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.trajectories import load_sine3d
from mvis.simulation.multi_board import (
    place_boards_ring, MultiBoardScenario, visibility_audit, forward_T_CtoI,
    BOARD_ID_STRIDE, board_of, local_of)

T265_FISHEYE = np.array([288.74401, 288.91833, 416.14622, 405.97177,
                         -0.00777028, 0.04423187, -0.04124983, 0.00730763])
PINHOLE = np.array([400., 400., 320., 240., -0.28, 0.07, 0.0005, -0.0004])


def _imu():
    return ImuModel(x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
                    x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
                    x_tg=np.zeros(9),
                    R=gtsam.Rot3.Expmap([0.02, -0.01, 0.03]).matrix(), model=RPNG)


def fisheye_scn():
    """t265 fisheye + 1 m boards -> all 4 boards co-visible every frame."""
    traj = load_sine3d(target_span=1.0)
    board = Chessboard(11, 11, 0.10)
    poses = place_boards_ring(board, radius=1.5, directions=4)
    return MultiBoardScenario(
        traj=traj, boards=[board] * 4, T_BinW_list=poses,
        cal=_cal_from_vector(T265_FISHEYE, True), is_fisheye=True,
        T_CtoI=forward_T_CtoI(), image_size=(848, 800), imu_intr=_imu(),
        b_g=np.array([0.01, -0.02, 0.005]), b_a=np.array([0.05, -0.03, 0.02]),
        sigma_px=0.0)


def pinhole_scn():
    """narrow pinhole + 0.28 m boards -> boards sparse, never co-visible (bridge case)."""
    traj = load_sine3d(target_span=1.0)
    board = Chessboard(7, 7, 0.04)
    poses = place_boards_ring(board, radius=1.5, directions=4)
    return MultiBoardScenario(
        traj=traj, boards=[board] * 4, T_BinW_list=poses,
        cal=_cal_from_vector(PINHOLE, False), is_fisheye=False,
        T_CtoI=forward_T_CtoI(), image_size=(640, 480), imu_intr=_imu(),
        b_g=np.array([0.01, -0.02, 0.005]), b_a=np.array([0.05, -0.03, 0.02]),
        sigma_px=0.0)


def test_id_helpers_roundtrip():
    gid = 3 * BOARD_ID_STRIDE + 17
    assert board_of(gid) == 3 and local_of(gid) == 17


def test_fisheye_all_boards_co_visible():
    sc = fisheye_scn()
    times = imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 10.0)
    audit = visibility_audit(sc.camera(times), n_boards=4)
    assert all(n > 100 for n in audit['per_board_frames']), audit['per_board_frames']
    assert audit['co_visible_frames'] > 400, \
        f"expected heavy co-visibility, got {audit['co_visible_frames']}"


def test_pinhole_boards_sparse_non_co_visible():
    sc = pinhole_scn()
    times = imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 10.0)
    audit = visibility_audit(sc.camera(times), n_boards=4)
    assert audit['co_visible_frames'] == 0, \
        f"pinhole should never co-visible, got {audit['co_visible_frames']}"
    assert sum(audit['per_board_frames']) > 0


def test_tagged_ids_decode_to_owning_board():
    sc = fisheye_scn()
    times = imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 10.0)
    seen = np.concatenate([ids for ids in sc.camera(times).ids if len(ids)])
    assert set(np.unique(board_of(seen))).issubset({0, 1, 2, 3})
