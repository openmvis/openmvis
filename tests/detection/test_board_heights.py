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
"""Board pose + per-corner heights (imu-scale-followup M3).

A chessboard with a known out-of-plane twist is imaged from oblique views; the
optimiser, seeded with the flat board, must recover the heights (plane removed)
and must leave a flat board flat.
"""
import cv2
import numpy as np
import pytest

from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.core.types import FrameObservations, Observation
from mvis.detection.chessboard import Chessboard

IMG = (640, 480)
K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
DIST = np.array([-0.1, 0.01, 0.0, 0.0, 0.0])


def _remove_plane(pts, h):
    X = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 1]])
    c, *_ = np.linalg.lstsq(X, h, rcond=None)
    return h - X @ c


def _twist(pts, amp=0.005):
    x = (pts[:, 0] - pts[:, 0].mean()) / np.ptp(pts[:, 0])
    y = (pts[:, 1] - pts[:, 1].mean()) / np.ptp(pts[:, 1])
    return amp * (4 * x * y + 0.5 * np.cos(2 * np.pi * x))      # twist + bow, p-p ~ amp


def _frames(board, heights, num_frames=40, noise=0.3, seed=3):
    rng = np.random.RandomState(seed)
    pts = board.all_points_3d().astype(np.float64)
    pts[:, 2] += heights
    frames = []
    while len(frames) < num_frames:
        rvec = rng.uniform(-0.6, 0.6, 3)                        # oblique views: heights need parallax
        tvec = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.35, 0.6)])
        uv, _ = cv2.projectPoints(pts, rvec, tvec, K, DIST)
        uv = uv.reshape(-1, 2)
        if not np.all((uv[:, 0] > 15) & (uv[:, 0] < IMG[0] - 15) & (uv[:, 1] > 15) & (uv[:, 1] < IMG[1] - 15)):
            continue
        uv += rng.randn(*uv.shape) * noise
        obs = [Observation(point_id=i, pixel=uv[i], row=int(uv[i, 1])) for i in range(len(uv))]
        frames.append(FrameObservations(camera_id=0, timestamp=float(len(frames)),
                                        observations=obs, image_size=IMG))
    return frames


def _solve(board, frames, heights_on):
    init = Initializer(board)
    intr, dist = init.init_intrinsics_opencv(frames, IMG, "radtan")
    poses = init.init_board_poses(frames, intr, dist, "radtan")
    opts = OptimizerOptions(max_iterations=60, pixel_sigma=1.0, board_heights=heights_on)
    return Optimizer(board, opts).solve_single_camera(frames, intr, dist, poses,
                                                      distortion_model="radtan", image_size=IMG)


@pytest.fixture
def board():
    return Chessboard(rows=7, cols=5, square_size=0.03)


def test_recovers_twisted_board(board):
    pts = board.all_points_3d()
    truth = _remove_plane(pts, _twist(pts))
    res = _solve(board, _frames(board, truth), heights_on=True)
    assert res.board_heights is not None and len(res.board_heights) == board.num_points()
    est = _remove_plane(pts, res.board_heights)
    err = est - truth
    assert np.std(truth) > 1.5e-3                                # the twist is there to find
    assert np.sqrt(np.mean(err ** 2)) < 0.3e-3, f"height rms error {np.sqrt(np.mean(err**2))*1e3:.2f} mm"
    assert abs(np.mean(res.board_heights)) < 1e-5               # gauge: no mean plane in the heights
    assert res.board_point_bias is not None and res.board_point_bias.shape == (board.num_points(), 3)
    assert np.median(np.hypot(res.board_point_bias[:, 0], res.board_point_bias[:, 1])) < 0.1


def test_flat_board_stays_flat(board):
    res = _solve(board, _frames(board, np.zeros(board.num_points()), seed=7), heights_on=True)
    assert np.sqrt(np.mean(res.board_heights ** 2)) < 0.15e-3


def test_twist_shows_as_bias_when_off(board):
    pts = board.all_points_3d()
    truth = _remove_plane(pts, _twist(pts))
    res = _solve(board, _frames(board, truth), heights_on=False)
    assert res.board_heights is None                             # legacy: nothing estimated
    b = res.board_point_bias
    assert np.max(np.hypot(b[:, 0], b[:, 1])) > 0.3               # ... but the diagnostic sees the twist
