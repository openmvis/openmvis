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
"""Phase 2 (sim-from-result): simulate_cameras — multi-camera digital-twin sensor model."""
import os

import numpy as np
import gtsam

from mvis.simulation.from_result import GroundTruth
from mvis.simulation.multicam import simulate_cameras


def _gt_and_board():
    """A GT where two cameras at ~origin look +z at a planar board 2 m away."""
    g = np.linspace(-0.2, 0.2, 5)
    board = np.array([[x, y, 0.0] for x in g for y in g])          # 25 corners
    T_BinW = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 2))
    ts = np.linspace(0.0, 1.0, 30)
    T_states = [gtsam.Pose3(gtsam.Rot3.Identity(),
                            gtsam.Point3(0.05 * np.sin(2 * t), 0.0, 0.0)).matrix().tolist()
                for t in ts]
    intr = [400., 400, 320, 240, 0, 0, 0, 0]
    cams = [dict(intr8=intr, T_CtoI=gtsam.Pose3().matrix().tolist(), td=0.0,
                 readout=0.0, is_fisheye=False, image_size=[640, 480]),
            dict(intr8=intr, T_CtoI=gtsam.Pose3(gtsam.Rot3.Identity(),
                 gtsam.Point3(0.05, 0, 0)).matrix().tolist(), td=0.0,
                 readout=0.01, is_fisheye=False, image_size=[640, 480])]  # 2nd is rolling
    gt = GroundTruth(cams=cams, imus=[], b_g=[0, 0, 0], b_a=[0, 0, 0],
                     T_BinW=T_BinW.matrix().tolist(), t_states=ts.tolist(),
                     T_states=T_states, imu_model="calyx", g_inW=[0, 0, -9.81])
    return gt, board


def test_simulate_cameras_sees_board():
    gt, board = _gt_and_board()
    sims = simulate_cameras(gt, board, rates=[30.0, 30.0], sigma_px=0.0)
    assert len(sims) == 2
    for s in sims:
        assert s['t_cam'].size > 0
        assert sum(1 for ids in s['ids_list'] if len(ids)) > 0     # frames see the board
        assert s['image_size'] == (640, 480)
    # ids are valid board-point indices (0..24)
    k = next(i for i, ids in enumerate(sims[0]['ids_list']) if len(ids))
    assert sims[0]['ids_list'][k].max() < len(board)


def test_twin_consistent_with_td_and_rolling_shutter():
    """Regression: a camera with a large td AND rolling shutter must simulate and
    reproject consistently. Catches (1) simulate_cameras dropping the td, (2) a
    non-RS-aware reprojection. Before the fixes the ELP twin panels were ~48 px off."""
    from mvis.simulation.from_result import GroundTruth
    from mvis.simulation.multicam import simulate_cameras, traj_from_gt
    from mvis.simulation.video import CalibScene, _reproject_frame
    g = np.linspace(-0.15, 0.15, 5)
    board = np.array([[x, y, 0.0] for x in g for y in g])
    T_BinW = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 2))
    ts = np.linspace(0.0, 2.0, 60)                              # fast enough to expose a td mismatch
    T_states = [gtsam.Pose3(gtsam.Rot3.Identity(),
                gtsam.Point3(0.1 * np.sin(4 * t), 0.1 * np.cos(4 * t), 0.0)).matrix().tolist()
                for t in ts]
    cam = dict(intr8=[400., 400, 320, 240, 0, 0, 0, 0], T_CtoI=gtsam.Pose3().matrix().tolist(),
               td=-0.12, readout=0.017, is_fisheye=False, image_size=[640, 480])  # ELP-like
    gt = GroundTruth(cams=[cam], imus=[], b_g=[0, 0, 0], b_a=[0, 0, 0],
                     T_BinW=T_BinW.matrix().tolist(), t_states=ts.tolist(),
                     T_states=T_states, imu_model="calyx", g_inW=[0, 0, -9.81])
    s = simulate_cameras(gt, board, rates=[25.0], sigma_px=0.0)[0]
    scene = CalibScene(traj=traj_from_gt(gt), board_points={0: board},
                       board_poses={0: gtsam.Pose3(np.asarray(gt.T_BinW, float))},
                       cal_vec=np.array(cam['intr8'], float), is_fisheye=False, image_size=(640, 480),
                       T_CtoI=gtsam.Pose3(np.asarray(cam['T_CtoI'], float)),
                       td=cam['td'], readout=cam['readout'])
    errs = []
    for k in range(len(s['t_cam'])):
        ids = s['ids_list'][k]
        if not len(ids):
            continue
        det = np.asarray(s['uv_list'][k]).reshape(-1, 2)
        rep = _reproject_frame(scene, float(s['t_cam'][k]), ids, rows=det[:, 1])
        errs.append(np.nanmax(np.linalg.norm(det - rep, axis=1)))
    assert np.median(errs) < 1.0                               # consistent; was ~48 px pre-fix


def test_simulate_imus():
    from mvis.simulation.multicam import simulate_imus
    from mvis.simulation.imu_measurements import imu_times
    gt, _ = _gt_and_board()
    gt.imus = [dict(name="imu0", is_base=True, xi24=[0.0] * 24, T_IstoI=None, td=0.0),
               dict(name="aux1", is_base=False, xi24=[0.0] * 24, td=0.001,
                    T_IstoI=gtsam.Pose3(gtsam.Rot3.Identity(),
                                        gtsam.Point3(0.1, -0.05, 0)).matrix().tolist())]
    t = imu_times(gt.t_states[0], gt.t_states[-1], 200.0)
    base, aux = simulate_imus(gt, t, noise=None)
    assert base is not None and base.w_m.shape[0] == len(t)
    assert len(aux) == 1 and aux[0]['name'] == "aux1" and aux[0]['a_m'].shape[0] == len(t)


def test_run_digital_twin_renders(tmp_path):
    from mvis.simulation.run import run_digital_twin
    gt, board = _gt_and_board()
    out, sims = run_digital_twin(gt, board, str(tmp_path / "twin.mp4"),
                                 rates=[30.0, 30.0], fps=4, stride=4)
    assert os.path.exists(out) and os.path.getsize(out) > 0 and len(sims) == 2
