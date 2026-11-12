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
"""Tests for diverged board pose detection after GTSAM optimization."""

import numpy as np
import gtsam
import pytest

from mvis.calibration.pipeline_cam import _detect_diverged_poses_from_state


def _pose_at_z(z: float) -> gtsam.Pose3:
    """Create a board pose where the board is at distance z in camera frame.

    T_board_cam (GTSAM convention) is what the optimizer stores.
    T_cam_board = T_board_cam.inverse() gives the board in camera frame.
    We want T_cam_board.translation()[2] = z.
    So T_cam_board = [I, [0,0,z]], meaning T_board_cam = [I, [0,0,-z]].
    """
    return gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, -z))


class TestDetectDivergedPoses:

    def test_normal_poses_not_flagged(self):
        """Poses at reasonable distances (0.5-2m) should not be flagged."""
        poses = [_pose_at_z(0.5), _pose_at_z(1.0), _pose_at_z(1.5), _pose_at_z(2.0)]
        diverged = _detect_diverged_poses_from_state(poses)
        assert len(diverged) == 0

    def test_identity_pose_flagged(self):
        """Identity pose (z=0) should be flagged — board at camera center."""
        poses = [_pose_at_z(1.0), gtsam.Pose3(), _pose_at_z(1.0)]
        diverged = _detect_diverged_poses_from_state(poses)
        assert diverged == {1}

    def test_behind_camera_flagged(self):
        """Negative z (board behind camera) should be flagged."""
        poses = [_pose_at_z(1.0), _pose_at_z(-0.5), _pose_at_z(1.0)]
        diverged = _detect_diverged_poses_from_state(poses)
        assert diverged == {1}

    def test_too_far_flagged(self):
        """z > 20m should be flagged."""
        poses = [_pose_at_z(1.0), _pose_at_z(50.0), _pose_at_z(1.0)]
        diverged = _detect_diverged_poses_from_state(poses)
        assert diverged == {1}

    def test_infinity_flagged(self):
        """Extremely large z (optimizer divergence) should be flagged."""
        poses = [_pose_at_z(1.0), _pose_at_z(1e10), _pose_at_z(1.0)]
        diverged = _detect_diverged_poses_from_state(poses)
        assert diverged == {1}

    def test_none_poses_skipped(self):
        """None poses (no PnP result) should be skipped, not flagged."""
        poses = [_pose_at_z(1.0), None, _pose_at_z(1.0)]
        diverged = _detect_diverged_poses_from_state(poses)
        assert len(diverged) == 0

    def test_boundary_min_z(self):
        """Pose exactly at min_z boundary should pass."""
        poses = [_pose_at_z(0.05)]
        diverged = _detect_diverged_poses_from_state(poses, min_z=0.05)
        assert len(diverged) == 0

        poses = [_pose_at_z(0.04)]
        diverged = _detect_diverged_poses_from_state(poses, min_z=0.05)
        assert diverged == {0}

    def test_boundary_max_z(self):
        """Pose exactly at max_z boundary should pass."""
        poses = [_pose_at_z(20.0)]
        diverged = _detect_diverged_poses_from_state(poses, max_z=20.0)
        assert len(diverged) == 0

        poses = [_pose_at_z(20.1)]
        diverged = _detect_diverged_poses_from_state(poses, max_z=20.0)
        assert diverged == {0}

    def test_multiple_diverged(self):
        """Multiple diverged poses detected simultaneously."""
        poses = [
            _pose_at_z(1.0),    # ok
            gtsam.Pose3(),      # identity → z=0
            _pose_at_z(1.0),    # ok
            _pose_at_z(100.0),  # too far
            _pose_at_z(-1.0),   # behind camera
            _pose_at_z(0.8),    # ok
        ]
        diverged = _detect_diverged_poses_from_state(poses)
        assert diverged == {1, 3, 4}

    def test_custom_thresholds(self):
        """Custom min_z and max_z thresholds work."""
        poses = [_pose_at_z(0.3), _pose_at_z(5.0)]
        # Default: both ok
        assert len(_detect_diverged_poses_from_state(poses)) == 0
        # Tighter: 0.3m too close
        assert _detect_diverged_poses_from_state(poses, min_z=0.5) == {0}
        # Tighter: 5.0m too far
        assert _detect_diverged_poses_from_state(poses, max_z=3.0) == {1}

    def test_empty_list(self):
        """Empty pose list returns empty set."""
        assert _detect_diverged_poses_from_state([]) == set()

    def test_all_none(self):
        """All-None list returns empty set."""
        assert _detect_diverged_poses_from_state([None, None, None]) == set()

    def test_with_rotation(self):
        """Rotated poses still have correct z-depth check."""
        # Board at 1m, rotated 30 degrees
        R = gtsam.Rot3.Expmap(np.array([0.3, 0.2, 0.1]))
        T_cam_board = gtsam.Pose3(R, gtsam.Point3(0.1, -0.1, 1.0))
        T_board_cam = T_cam_board.inverse()
        poses = [T_board_cam]
        diverged = _detect_diverged_poses_from_state(poses)
        assert len(diverged) == 0  # z≈1.0m is fine
