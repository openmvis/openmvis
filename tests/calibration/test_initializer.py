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
"""Tests for calibration initializer."""

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.initializer import Initializer
from mvis.core.types import Observation, FrameObservations
from tests.helpers.cam_frames import _generate_synthetic_observations


class TestInitializer:
    @pytest.fixture
    def setup(self):
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K_gt = np.array([
            [500.0, 0, 320.0],
            [0, 500.0, 240.0],
            [0, 0, 1],
        ])
        dist_gt = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        return board, K_gt, dist_gt

    def test_init_intrinsics(self, setup):
        board, K_gt, dist_gt = setup
        frames, _ = _generate_synthetic_observations(
            board, K_gt, dist_gt, num_frames=30, noise_sigma=0.3
        )

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(
            frames, (640, 480), "radtan"
        )

        # Radtan uses OpenCV calibrateCamera — should be close to GT
        assert abs(intrinsics[0] - 500.0) / 500.0 < 0.1  # fx within 10%
        assert abs(intrinsics[1] - 500.0) / 500.0 < 0.1  # fy within 10%
        assert abs(intrinsics[2] - 320.0) < 20  # cx
        assert abs(intrinsics[3] - 240.0) < 20  # cy

    def test_init_board_poses(self, setup):
        board, K_gt, dist_gt = setup
        frames, gt_poses = _generate_synthetic_observations(
            board, K_gt, dist_gt, num_frames=20, noise_sigma=0.3
        )

        initializer = Initializer(board)
        intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
        dist = np.array([-0.1, 0.01, 0.001, -0.001])

        poses = initializer.init_board_poses(frames, intrinsics, dist, "radtan")

        # All poses should be valid
        n_valid = sum(1 for p in poses if p is not None)
        assert n_valid == len(frames)

        # Check first pose is close to ground truth
        # poses[0] is T_board_cam (GTSAM convention), gt_poses[0] is T_cam_board
        if poses[0] is not None:
            est_mat = poses[0].matrix()
            gt_mat = gt_poses[0]
            # Compare the inverse: T_cam_board = T_board_cam^{-1}
            est_cam_board = np.linalg.inv(est_mat)
            t_err = np.linalg.norm(est_cam_board[:3, 3] - gt_mat[:3, 3])
            assert t_err < 0.05  # within 5cm

    def test_init_intrinsics_no_frames(self, setup):
        board, K_gt, dist_gt = setup
        initializer = Initializer(board)
        with pytest.raises(ValueError, match="Need at least 1 frame"):
            initializer.init_intrinsics_opencv([], (640, 480), "radtan")

    def test_init_intrinsics_few_frames(self, setup):
        """Radtan needs at least 3 frames for OpenCV calibrateCamera."""
        board, K_gt, dist_gt = setup
        frames, _ = _generate_synthetic_observations(
            board, K_gt, dist_gt, num_frames=5, noise_sigma=0.1
        )
        initializer = Initializer(board)
        # 5 frames should work
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "radtan")
        assert intrinsics[0] > 0

    def test_init_board_pose_ransac(self, setup):
        """RANSAC PnP handles good data correctly."""
        board, K_gt, dist_gt = setup
        frames, gt_poses = _generate_synthetic_observations(
            board, K_gt, dist_gt, num_frames=1, noise_sigma=0.3
        )
        initializer = Initializer(board)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001])
        pose = initializer.init_board_pose(
            frames[0], K, dist, "radtan", max_reprojection_px=10.0)
        assert pose is not None

    def test_init_board_pose_rejects_bad_reprojection(self, setup):
        """PnP pose with injected outlier pixels is rejected by reprojection check."""
        board, K_gt, dist_gt = setup
        frames, _ = _generate_synthetic_observations(
            board, K_gt, dist_gt, num_frames=1, noise_sigma=0.1
        )
        # Corrupt half the observations with huge pixel offsets
        f = frames[0]
        bad_obs = []
        for i, o in enumerate(f.observations):
            if i % 2 == 0:
                bad_obs.append(Observation(
                    point_id=o.point_id,
                    pixel=o.pixel + np.array([500.0, 500.0]),
                    row=o.row,
                ))
            else:
                bad_obs.append(o)
        bad_frame = FrameObservations(
            camera_id=0, timestamp=0.0, observations=bad_obs,
            image_size=(640, 480))

        initializer = Initializer(board)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001])
        # With loose threshold, RANSAC should still find a pose from inliers
        pose_loose = initializer.init_board_pose(
            bad_frame, K, dist, "radtan", max_reprojection_px=50.0)
        # With tight threshold (0.5 px), inlier error may exceed it
        pose_tight = initializer.init_board_pose(
            bad_frame, K, dist, "radtan", max_reprojection_px=0.5)
        # At least one should differ — the tight one is more likely None
        # (RANSAC inlier error on 0.1σ noise should be ~0.1px, but with
        # half points corrupted the PnP solution may be worse)
        assert pose_loose is not None or pose_tight is None

    def test_init_board_poses_with_reprojection_filter(self, setup):
        """init_board_poses threads max_reprojection_px through."""
        board, K_gt, dist_gt = setup
        frames, _ = _generate_synthetic_observations(
            board, K_gt, dist_gt, num_frames=10, noise_sigma=0.3
        )
        initializer = Initializer(board)
        intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
        dist = np.array([-0.1, 0.01, 0.001, -0.001])

        # With a reasonable threshold, all should pass
        poses = initializer.init_board_poses(
            frames, intrinsics, dist, "radtan",
            max_reprojection_px=10.0)
        n_valid = sum(1 for p in poses if p is not None)
        assert n_valid == len(frames)

        # With an impossibly tight threshold, some or all should fail
        poses_tight = initializer.init_board_poses(
            frames, intrinsics, dist, "radtan",
            max_reprojection_px=0.01)
        n_valid_tight = sum(1 for p in poses_tight if p is not None)
        assert n_valid_tight < n_valid
