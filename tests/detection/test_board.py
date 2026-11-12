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
"""Tests for calibration board classes."""

import numpy as np
import cv2
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.detection.board_factory import create_board
from mvis.core.config import BoardConfig


class TestChessboard:
    def test_creation(self):
        board = Chessboard(rows=9, cols=6, square_size=0.025)
        assert board.rows == 9
        assert board.cols == 6
        assert board.square_size == 0.025
        assert board.num_points() == 54
        assert board.min_points_for_pose() == 4

    def test_point_3d(self):
        board = Chessboard(rows=3, cols=4, square_size=0.1)
        # First point at origin
        p0 = board.point_3d(0)
        np.testing.assert_allclose(p0, [0, 0, 0])

        # Point (row=0, col=1) -> [0.1, 0, 0]
        p1 = board.point_3d(1)
        np.testing.assert_allclose(p1, [0.1, 0, 0])

        # Point (row=1, col=0) -> [0, 0.1, 0]
        p4 = board.point_3d(4)
        np.testing.assert_allclose(p4, [0, 0.1, 0])

    def test_all_points_3d(self):
        board = Chessboard(rows=3, cols=4, square_size=0.05)
        pts = board.all_points_3d()
        assert pts.shape == (12, 3)
        # All z=0 for planar board on XY plane
        np.testing.assert_allclose(pts[:, 2], 0.0)

    def test_detect_on_synthetic_image(self):
        """Test detection on a synthetically rendered chessboard image."""
        rows, cols = 7, 5
        square_px = 40
        board = Chessboard(rows=rows, cols=cols, square_size=0.025)

        # Create a synthetic chessboard image
        img_h = (rows + 2) * square_px
        img_w = (cols + 2) * square_px
        img = np.ones((img_h, img_w), dtype=np.uint8) * 255

        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y0 = (r + 1) * square_px
                    x0 = (c + 1) * square_px
                    img[y0 : y0 + square_px, x0 : x0 + square_px] = 0

        observations = board.detect(img)
        assert len(observations) == rows * cols
        # Check that point IDs are sequential
        ids = [obs.point_id for obs in observations]
        assert sorted(ids) == list(range(rows * cols))

    def test_detect_no_board(self):
        """Test that detection returns empty on blank image."""
        board = Chessboard(rows=9, cols=6, square_size=0.025)
        blank = np.ones((480, 640), dtype=np.uint8) * 128
        observations = board.detect(blank)
        assert len(observations) == 0


class TestBoardFactory:
    def test_create_chessboard(self):
        config = BoardConfig(board_type="chessboard", rows=9, cols=6, square_size=0.025)
        board = create_board(config)
        assert isinstance(board, Chessboard)
        assert board.num_points() == 54

    def test_create_unknown_raises(self):
        config = BoardConfig(board_type="unknown_board")
        with pytest.raises(ValueError):
            create_board(config)
