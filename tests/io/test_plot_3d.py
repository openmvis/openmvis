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
"""Tests for the 3D camera-pose plot (display transform)."""

import numpy as np

from mvis.analysis.plot_3d import (
    plot_camera_poses,
    _to_display_points,
    _to_display_pose,
)
from mvis.detection.chessboard import Chessboard


def test_display_transform_permutes_axes():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    disp = _to_display_points(pts)
    # [x,y,z] -> [z,x,y]
    np.testing.assert_allclose(disp, [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])


def test_display_transform_puts_board_on_yz():
    """Board points at z=0 should map to x=0 (YZ plane) in the plot."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    # Put the camera 0.5m in front of the board, identity rotation
    T_board_cam = np.eye(4)
    T_board_cam[2, 3] = -0.5
    fig = plot_camera_poses([T_board_cam], board, title="test")

    # First trace is the board-points scatter; assert all x ≈ 0
    pts_trace = fig.data[0]
    xs = np.array(pts_trace.x)
    np.testing.assert_allclose(xs, 0.0, atol=1e-9)
    # y and z carry the original board-frame x and y
    ys = np.array(pts_trace.y)
    zs = np.array(pts_trace.z)
    assert not np.allclose(ys, 0.0)
    assert not np.allclose(zs, 0.0)


def test_display_pose_is_4x4():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    Td = _to_display_pose(T)
    assert Td.shape == (4, 4)
    # Position in display frame = [z, x, y] = [3, 1, 2]
    np.testing.assert_allclose(Td[:3, 3], [3.0, 1.0, 2.0])


def test_camera_position_in_display_frame():
    """A camera at z=-0.5 in board frame should land at x=-0.5 in display."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    T_board_cam = np.eye(4)
    T_board_cam[2, 3] = -0.5  # camera behind board (looking at +z from -z)
    fig = plot_camera_poses([T_board_cam], board, title="test")

    # Find the camera-position scatter (by name prefix "Camera 0")
    for trace in fig.data:
        if getattr(trace, "name", None) == "Camera 0 (ref)":
            # Position in display frame: x = orig_z = -0.5
            assert float(trace.x[0]) == -0.5
            break
    else:
        raise AssertionError("Camera 0 scatter trace not found")
