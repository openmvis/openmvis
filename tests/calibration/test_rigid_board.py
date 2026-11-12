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
"""`estimate.board_points` false keeps the board at its given shape; true moves corners only along z.

The observations come from a bent board, so a board that is free to move would follow the bend."""
import cv2
import numpy as np
import pytest

from mvis.calibration import optimizer as O
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.core.types import FrameObservations, Observation
from mvis.detection.chessboard import Chessboard

IMG = (640, 480)
K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
DIST = np.array([-0.1, 0.01, 0.0, 0.0, 0.0])
BEND_M = 0.002


def _frames(board, n=12):
    rng = np.random.RandomState(5)
    nominal = board.all_points_3d().astype(np.float64)
    centre = nominal[:, :2] - nominal[:, :2].mean(0)
    bent = nominal + np.c_[np.zeros((len(nominal), 2)), BEND_M * (centre ** 2).sum(1) / (centre ** 2).sum(1).max()]
    frames = []
    while len(frames) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(0.3, 0.5)])
        uv = cv2.projectPoints(bent, rvec, tvec, K, DIST)[0].reshape(-1, 2)
        if np.all((uv > 15) & (uv < np.array(IMG) - 15)):
            frames.append(FrameObservations(
                camera_id=0, timestamp=float(len(frames)), image_size=IMG,
                observations=[Observation(point_id=i, pixel=uv[i], row=int(uv[i, 1])) for i in range(len(uv))]))
    return frames


def _solved_corners(monkeypatch, board_heights):
    """Offsets of the solved corners from the board's own positions."""
    board = Chessboard(rows=5, cols=4, square_size=0.03)
    frames = _frames(board)
    init = Initializer(board)
    intr, dist = init.init_intrinsics_opencv(frames, IMG, "radtan")
    poses = init.init_board_poses(frames, intr, dist, "radtan")
    solved = {}
    real = O.gtsam.LevenbergMarquardtOptimizer

    class Spy:
        def __init__(self, graph, values, params):
            self._lm = real(graph, values, params)

        def __getattr__(self, name):
            return getattr(self._lm, name)

        def optimize(self):
            solved["values"] = self._lm.optimize()
            return solved["values"]

    monkeypatch.setattr(O.gtsam, "LevenbergMarquardtOptimizer", Spy)
    opts = OptimizerOptions(max_iterations=15, pixel_sigma=1.0, board_heights=board_heights)
    Optimizer(board, opts).solve_single_camera(frames, intr, dist, poses, distortion_model="radtan", image_size=IMG)
    nominal = board.all_points_3d().astype(np.float64)
    return np.array([np.asarray(solved["values"].atPoint3(O._sym("L", i)), float) for i in range(len(nominal))]) - nominal


def test_board_points_false_keeps_every_corner_at_its_position(monkeypatch):
    assert np.abs(_solved_corners(monkeypatch, False)).max() < 1e-5


def test_board_points_true_moves_corners_only_along_z(monkeypatch):
    offset = _solved_corners(monkeypatch, True)
    assert np.abs(offset[:, :2]).max() < 1e-5
    assert np.abs(offset[:, 2]).max() > 1e-4
