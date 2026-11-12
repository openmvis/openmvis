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
"""Per-board pose variables in the joint graph: symbol indexing + extraction.
Backward compatibility (single-board path unchanged) is guarded by the existing
tests/cam_imu/test_pipeline_cam_imu.py suite, run alongside this file."""

import numpy as np
import gtsam
import pytest

from mvis.calibration.pipeline_cam_imu import _B, B0, board_poses_from_values


def test_B_symbol_indexes_boards():
    assert _B(0) == gtsam.symbol('b', 0) == B0
    assert _B(2) == gtsam.symbol('b', 2)
    assert _B(2) != _B(0)


def test_board_poses_from_values_roundtrip():
    vals = gtsam.Values()
    p1 = gtsam.Pose3(gtsam.Rot3.Expmap([0.1, 0.0, 0.2]), gtsam.Point3(1, 2, 3))
    vals.insert(_B(0), gtsam.Pose3())
    vals.insert(_B(1), p1)
    got = board_poses_from_values(vals, [0, 1])
    assert got[1].equals(p1, 1e-9)
    assert got[0].equals(gtsam.Pose3(), 1e-9)


# --- N=4 end-to-end driver recovery (noise-free, fisheye) ---
from mvis.detection.chessboard import Chessboard          # noqa: E402
from mvis.imu import ImuModel, RPNG                   # noqa: E402
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector  # noqa: E402
from mvis.simulation.imu_measurements import imu_times  # noqa: E402
from mvis.simulation.trajectories import load_sine3d  # noqa: E402
from mvis.simulation.multi_board import (             # noqa: E402
    place_boards_ring, MultiBoardScenario, forward_T_CtoI)
from mvis.calibration.multi_board_pipeline import run_multi_board_calibration  # noqa: E402

_T265 = np.array([288.74401, 288.91833, 416.14622, 405.97177,
                  -0.00777028, 0.04423187, -0.04124983, 0.00730763])


def _fisheye_scn():
    traj = load_sine3d(target_span=1.0)
    board = Chessboard(11, 11, 0.10)
    poses = place_boards_ring(board, radius=1.5, directions=4)
    intr = ImuModel(x_dw=np.array([1., 0, 1., 0, 0, 1.]),
                    x_da=np.array([1., 0, 1., 0, 0, 1.]), x_tg=np.zeros(9),
                    R=np.eye(3), model=RPNG)
    return poses, MultiBoardScenario(
        traj=traj, boards=[board] * 4, T_BinW_list=poses,
        cal=_cal_from_vector(_T265, True), is_fisheye=True, T_CtoI=forward_T_CtoI(),
        image_size=(848, 800), imu_intr=intr, b_g=np.zeros(3), b_a=np.zeros(3),
        sigma_px=0.0)


def test_n4_noisefree_recovers_board_poses_and_extrinsic():
    poses_gt, sc = _fisheye_scn()
    t = imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 10.0)
    cam_stream = sc.camera(t)
    imu = sc.base_imu(imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 200.0))
    cam = dict(t_cam=cam_stream.t, ids_list=cam_stream.ids, uv_list=cam_stream.uv,
               cal_vec=_T265, is_fisheye=True, image_size=(848, 800))
    bpts = {b: sc.board_points(b) for b in range(4)}
    res = run_multi_board_calibration(cam, bpts, imu.t, imu.w_m, imu.a_m)
    assert res.ok
    dt_mm = 1e3 * np.linalg.norm(
        np.asarray(res.calib.T.translation()) - np.asarray(sc.T_CtoI.translation()))
    assert dt_mm < 5.0, f"lever off by {dt_mm:.1f} mm on noise-free data"
    bp = res.diagnostics['board_poses']
    for b in (1, 2, 3):
        T_est = gtsam.Pose3(np.array(bp[b]))
        rel_gt = poses_gt[0].between(poses_gt[b])
        rel_est = gtsam.Pose3(np.array(bp[0])).between(T_est)
        dt = 1e3 * np.linalg.norm(np.asarray(rel_gt.translation())
                                  - np.asarray(rel_est.translation()))
        assert dt < 10.0, f"board {b} off by {dt:.1f} mm"


@pytest.mark.slow
def test_multi_board_lm_uses_the_joint_solve_params(monkeypatch):
    """G5: the multi-board LM takes `_lm_params(cfg)`, so its damping is capped like the other
    joint solves (1e2, not GTSAM's 1e5)."""
    import mvis.calibration.multi_board_pipeline as mbp
    from mvis.calibration.lm_params import LM_LAMBDA_UPPER_DEFAULT

    class _Stop(Exception):
        pass
    seen, real = [], mbp._lm_params

    def spy(cfg, lambda_initial=None):
        seen.append(real(cfg, lambda_initial))
        raise _Stop
    monkeypatch.setattr(mbp, "_lm_params", spy)
    _poses, sc = _fisheye_scn()
    t = imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 10.0)
    cs = sc.camera(t)
    imu = sc.base_imu(imu_times(sc.traj.start_time, sc.traj.start_time + 68.0, 200.0))
    with pytest.raises(_Stop):
        run_multi_board_calibration(
            dict(t_cam=cs.t, ids_list=cs.ids, uv_list=cs.uv, cal_vec=_T265, is_fisheye=True,
                 image_size=(848, 800)),
            {b: sc.board_points(b) for b in range(4)}, imu.t, imu.w_m, imu.a_m)
    assert len(seen) == 1 and seen[0].getlambdaUpperBound() == LM_LAMBDA_UPPER_DEFAULT
