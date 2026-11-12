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
"""Rolling-shutter camera projection (simulator Phase 3b).

Validates: readout=0 reduces to global shutter; the per-feature capture time is a
self-consistent fixed point (reprojecting at that time reproduces the uv); RS
differs from global under motion; all three rolling modes converge."""

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.camera_measurements import (
    project_points, project_points_rolling, simulate_camera,
    board_points_in_global, place_board_facing_camera, _rs_row_time,
)

_CALVEC = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_IMG = (640, 480)
_TCI = gtsam.Pose3()           # camera coincident with IMU


def _cal():
    return _cal_from_vector(_CALVEC, is_fisheye=False)


def _traj_rs(dt=0.05, n=12):
    """Look-down camera that yaws and translates (so rows shift during readout)."""
    ctrl = []
    for k in range(n):
        R = np.diag([1.0, -1.0, -1.0]) @ gtsam.Rot3.Rz(0.5 * k * dt).matrix()
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [0.12 * k * dt, 0.06 * k * dt, 2.0]
        ctrl.append(T)
    return BsplineSE3(0.0, dt, ctrl)


def _scene():
    traj = _traj_rs()
    t0 = traj.start_time + 0.2
    board = Chessboard(6, 6, 0.04)
    T_BinG = place_board_facing_camera(traj, t0, 1.2, board)
    pts = board_points_in_global(board, T_BinG)
    return traj, t0, pts


# --------------------------------------------------------------------------
def test_rolling_reduces_to_global_when_readout_zero():
    cal, (traj, t0, pts) = _cal(), _scene()
    ids_rs, uv_rs, _ = project_points_rolling(traj, t0, cal, _TCI, pts, _IMG,
                                              readout=0.0, rolling_mode=0)
    R_GtoI, p = traj.get_pose(t0)
    T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p))
    ids_g, uv_g = project_points(T_CtoG, cal, pts, _IMG)
    assert np.array_equal(ids_rs, ids_g)
    assert np.allclose(uv_rs, uv_g, atol=1e-12)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_rolling_capture_time_is_fixed_point(mode):
    """Reprojecting each feature at its converged capture time reproduces its uv,
    and that time equals the row-time of that row."""
    cal, (traj, t0, pts) = _cal(), _scene()
    readout = 0.030
    ids, uv, ts = project_points_rolling(traj, t0, cal, _TCI, pts, _IMG,
                                         readout=readout, rolling_mode=mode)
    assert len(ids) > 0
    h = _IMG[1]
    for k, fid in enumerate(ids):
        R_GtoI, p = traj.get_pose(ts[k])
        T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p))
        uv_rp = gtsam.PinholeCameraCal3DS2(T_CtoG, cal).project(gtsam.Point3(*pts[fid]))
        assert np.allclose(uv_rp, uv[k], atol=0.2)                    # consistent uv
        assert abs(ts[k] - _rs_row_time(t0, uv_rp[1], h, readout, mode)) < 5e-5


def test_rolling_differs_from_global_under_motion():
    cal, (traj, t0, pts) = _cal(), _scene()
    R_GtoI, p = traj.get_pose(t0)
    T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p))
    ids_g, uv_g = project_points(T_CtoG, cal, pts, _IMG)
    ids_rs, uv_rs, _ = project_points_rolling(traj, t0, cal, _TCI, pts, _IMG,
                                              readout=0.030, rolling_mode=0)
    common = np.intersect1d(ids_g, ids_rs)
    assert len(common) > 0
    g = {i: uv_g[j] for j, i in enumerate(ids_g)}
    r = {i: uv_rs[j] for j, i in enumerate(ids_rs)}
    max_shift = max(np.linalg.norm(g[i] - r[i]) for i in common)
    assert max_shift > 0.5                                            # RS actually moved rows


def test_simulate_camera_readout_zero_matches_global():
    cal, (traj, t0, pts) = _cal(), _scene()
    times = [t0, t0 + 0.1]
    glob = simulate_camera(traj, times, cal, _TCI, pts, _IMG, readout=0.0)
    rs0 = simulate_camera(traj, times, cal, _TCI, pts, _IMG, readout=0.0, rolling_mode=1)
    assert np.array_equal(glob.ids[0], rs0.ids[0])
    assert np.allclose(glob.uv[0], rs0.uv[0], atol=1e-12)
