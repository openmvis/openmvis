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
"""T0c end-to-end proof: a REAL solver run through the bridge produces results
identical to the direct path — same init, same GTSAM solve, same numbers.
"""
from __future__ import annotations

import cv2
import numpy as np

from mvis.detection.chessboard import Chessboard
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.core.types import FrameObservations, Observation
from mvis.mvisdb.bridge import db_from_frame_observations, frame_observations_from_db


def _synthetic_frames(board, K, dist, num_frames=12, image_size=(640, 480), seed=7):
    rng = np.random.RandomState(seed)
    pts3 = board.all_points_3d().astype(np.float32)
    frames = []
    attempts = 0
    while len(frames) < num_frames and attempts < num_frames * 8:
        attempts += 1
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1),
                         rng.uniform(0.5, 0.9)])
        img, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
        img = img.reshape(-1, 2)
        w, h = image_size
        if not np.all((img[:, 0] > 20) & (img[:, 0] < w - 20)
                      & (img[:, 1] > 20) & (img[:, 1] < h - 20)):
            continue
        img = img + rng.randn(*img.shape) * 0.5
        obs = [Observation(point_id=i, pixel=img[i], row=int(img[i, 1]))
               for i in range(len(img))]
        frames.append(FrameObservations(camera_id=0, timestamp=float(len(frames)),
                                        observations=obs, image_size=image_size))
    assert len(frames) == num_frames
    return frames


def _solve(board, frames, image_size=(640, 480)):
    init = Initializer(board)
    intrinsics, dist = init.init_intrinsics_opencv(frames, image_size, "radtan")
    poses = init.init_board_poses(frames, intrinsics, dist, "radtan")
    opt = Optimizer(board, OptimizerOptions(max_iterations=100, pixel_sigma=1.0))
    return opt.solve_single_camera(frames, intrinsics, dist, poses,
                                   distortion_model="radtan", image_size=image_size)


def test_frames_round_trip_field_identical():
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    frames = _synthetic_frames(board, K, dist)
    db = db_from_frame_observations({0: frames}, board,
                                    image_sizes={0: (640, 480)})
    back = frame_observations_from_db(db, 0)
    assert len(back) == len(frames)
    for a, b in zip(frames, back):
        assert a.timestamp == b.timestamp
        assert a.camera_id == b.camera_id
        assert a.image_size == b.image_size
        assert len(a.observations) == len(b.observations)
        for oa, ob in zip(a.observations, b.observations):
            assert oa.point_id == ob.point_id
            assert oa.pixel.tobytes() == ob.pixel.tobytes()   # raw IEEE754 bits
            assert oa.row == ob.row


def test_solver_identical_through_bridge():
    """init + GTSAM solve on originals vs on bridge-round-tripped frames:
    the results must be numerically identical — the T0c gate."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    frames = _synthetic_frames(board, K, dist)

    direct = _solve(board, frames)
    db = db_from_frame_observations({0: frames}, board,
                                    image_sizes={0: (640, 480)})
    bridged = _solve(board, frame_observations_from_db(db, 0))

    cd, cb = direct.cameras[0], bridged.cameras[0]
    np.testing.assert_array_equal(np.asarray(cd.intrinsics), np.asarray(cb.intrinsics))
    np.testing.assert_array_equal(np.asarray(cd.distortion), np.asarray(cb.distortion))
    assert direct.overall_rms == bridged.overall_rms
    assert len(direct.board_poses) == len(bridged.board_poses)
    for pa, pb in zip(direct.board_poses, bridged.board_poses):
        ma = pa.matrix() if hasattr(pa, "matrix") else np.asarray(pa)
        mb = pb.matrix() if hasattr(pb, "matrix") else np.asarray(pb)
        np.testing.assert_array_equal(ma, mb)
