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
"""Board shape is off by default: `estimate.board_points` unset and explicitly false solve the same
problem on both paths. The cam-IMU graph then has no heights key H0. The camera-only graph always has
per-corner landmarks (the heights are their z), so there "off" means they are pinned, no gauge factor and no heights out."""
import cv2
import gtsam
import numpy as np

from mvis.calibration import optimizer as O
from mvis.calibration import pipeline_cam_imu as P
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.core.config import CamImuRunConfig, CamRunConfig
from mvis.core.types import FrameObservations, Observation
from mvis.detection.chessboard import Chessboard
from mvis.simulation.synth import build_synth, recover
from tests.helpers.sim_scenarios import _small_gt

BOARD = "board: {board_type: chessboard, rows: 5, cols: 4, square_size: 0.03}\n"
IMG = (640, 480)
K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
DIST = np.array([-0.1, 0.01, 0.0, 0.0, 0.0])


def _estimate(value):
    return "" if value is None else f"estimate: {{board_points: {value}}}\n"


def _cam_frames(board, n=8):
    rng = np.random.RandomState(3)
    pts = board.all_points_3d().astype(np.float64)
    frames = []
    while len(frames) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(0.3, 0.5)])
        uv = cv2.projectPoints(pts, rvec, tvec, K, DIST)[0].reshape(-1, 2)
        if np.all((uv > 15) & (uv < np.array(IMG) - 15)):
            uv = uv + rng.randn(*uv.shape) * 0.3
            frames.append(FrameObservations(
                camera_id=0, timestamp=float(len(frames)), image_size=IMG,
                observations=[Observation(point_id=i, pixel=uv[i], row=int(uv[i, 1])) for i in range(len(uv))]))
    return frames


def _keys(graph):
    return {k for i in range(graph.size()) if graph.at(i) is not None for k in graph.at(i).keys()}


def test_camera_only_unset_equals_false_and_has_no_heights(tmp_path, monkeypatch):
    board = Chessboard(rows=5, cols=4, square_size=0.03)
    frames = _cam_frames(board)
    init = Initializer(board)
    intr, dist = init.init_intrinsics_opencv(frames, IMG, "radtan")
    poses = init.init_board_poses(frames, intr, dist, "radtan")
    built = []
    real_lm = gtsam.LevenbergMarquardtOptimizer

    def spy(graph, values, params):
        built.append((graph.size(), values.size(), _keys(graph)))
        return real_lm(graph, values, params)
    monkeypatch.setattr(O.gtsam, "LevenbergMarquardtOptimizer", spy)

    def solve(value):
        p = tmp_path / "c.yaml"
        p.write_text("input: {cameras: []}\n" + BOARD + "solve: {}\n" + _estimate(value))
        on = CamRunConfig.from_yaml(str(p)).optimizer.optimize_board_points
        opts = OptimizerOptions(max_iterations=8, pixel_sigma=1.0, board_heights=on)
        return Optimizer(board, opts).solve_single_camera(frames, intr, dist, poses, distortion_model="radtan",
                                                          image_size=IMG)

    unset, false = solve(None), solve("false")
    assert unset.board_heights is None and false.board_heights is None
    assert np.array_equal(unset.cameras[0].intrinsics, false.cameras[0].intrinsics)
    assert np.array_equal(unset.cameras[0].distortion, false.cameras[0].distortion)
    assert unset.overall_rms == false.overall_rms
    assert built[0] == built[1]
    assert solve("true").board_heights is not None         # the check can tell a heights graph apart
    assert built[2][0] == built[0][0] + 1                  # the gauge factor


def _check_cam_imu(tmp_path, monkeypatch, vision_pose):
    gt = _small_gt()
    gt.board_points_local = np.array([[0.05 * j, 0.05 * i, 0.0] for i in range(6) for j in range(6)])
    ts = np.asarray(gt.t_states)
    cams, t_imu, w_m, a_m, aux = build_synth(
        gt, [ts[5:-5:6]], [np.arange(ts[2], ts[-2], 1.0 / 400.0)], sigma_px=1.0,
        imu_sigmas=dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002), seed=7)
    has_heights = []
    real_build = P.build_joint_graph

    def spy(*a, **k):
        out = real_build(*a, **k)
        has_heights.append(out[1].exists(P.H0) or P.H0 in _keys(out[0]))
        return out
    monkeypatch.setattr(P, "build_joint_graph", spy)

    def solve(value):
        p = tmp_path / "c.yaml"
        p.write_text("input: {format: euroc, path: x}\n" + BOARD + "camera_input: {path: y}\n"
                     f"solve: {{max_iterations: 2, vision_pose: {vision_pose}}}\noutput: {{path: z.json}}\n"
                     + _estimate(value))
        return recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, CamImuRunConfig.from_yaml(str(p)).solve)

    unset, false = solve(None), solve("false")
    assert has_heights == [False, False]
    assert "board_heights" not in unset.diagnostics and "board_heights" not in false.diagnostics
    assert unset.diagnostics["iterations"] == false.diagnostics["iterations"] > 0
    assert unset.diagnostics["err_final"] == false.diagnostics["err_final"]
    np.testing.assert_array_equal(unset.calib.intr, false.calib.intr)
    np.testing.assert_array_equal(unset.calib.T.matrix(), false.calib.T.matrix())
    np.testing.assert_array_equal(unset.imu_intr_xi, false.imu_intr_xi)
    assert "board_heights" in solve("true").diagnostics     # the check can tell a heights graph apart
    assert has_heights[-1] is True


def test_cam_imu_interp_unset_equals_false_and_has_no_heights(tmp_path, monkeypatch):
    _check_cam_imu(tmp_path, monkeypatch, "interp")


def test_cam_imu_vision_imu_unset_equals_false_and_has_no_heights(tmp_path, monkeypatch):
    _check_cam_imu(tmp_path, monkeypatch, "imu")
