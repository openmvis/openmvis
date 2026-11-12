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
"""Tests for analysis and reporting."""

import numpy as np
import pytest

from mvis.core.types import (
    CalibrationResult,
    CameraParams,
    ReprojectionErrorStats,
)
from mvis.analysis.report import print_report


class TestReport:
    def test_print_report_single_camera(self):
        result = CalibrationResult(
            cameras=[
                CameraParams(
                    camera_id=0,
                    intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
                    distortion=np.array([-0.1, 0.01, 0.001, -0.001]),
                    distortion_model="radtan",
                    image_size=(640, 480),
                ),
            ],
            reprojection_errors=[
                ReprojectionErrorStats(
                    camera_id=0,
                    rms=0.25,
                    mean=0.20,
                    median=0.18,
                    max=1.1,
                    num_observations=5000,
                ),
            ],
            overall_rms=0.25,
        )

        report = print_report(result)
        assert "MVIS" in report
        assert "Camera 0" in report
        assert "fx = 500.0000" in report
        assert "Median:" in report
        assert "P90:" in report
        assert "P95:" in report
        # RMS has been dropped from the text report in favour of robust stats.
        assert "RMS" not in report

    def test_print_report_empty(self):
        result = CalibrationResult(cameras=[])
        report = print_report(result)
        assert "MVIS" in report
        assert "RMS" not in report

    def test_stats_include_percentiles(self):
        """ReprojectionErrorStats should carry p90/p95 populated by optimizer."""
        from mvis.calibration.optimizer import Optimizer, OptimizerOptions
        from mvis.detection.chessboard import Chessboard

        # Tiny construction just to get an instance
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        opts = OptimizerOptions()
        opt = Optimizer(board, opts)
        errors = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0]
        stats = opt._errors_to_stats(errors, cam_id=7)
        assert stats.camera_id == 7
        # Monotonic: median <= p90 <= p95 <= max
        assert stats.median <= stats.p90 <= stats.p95 <= stats.max
        # Sanity: p95 of a small positive list is near the large end
        assert stats.p95 > stats.median


class TestHtmlReport:
    def test_summary_table_has_percentile_columns(self):
        """HTML report section 7 should include P90 and P95 columns."""
        from mvis.analysis.html_report import generate_html_report
        from mvis.detection.chessboard import Chessboard
        import tempfile
        import os

        result = CalibrationResult(
            cameras=[CameraParams(
                camera_id=0,
                intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
                distortion=np.array([-0.1, 0.01, 0.001, -0.001]),
                distortion_model="radtan",
                image_size=(640, 480),
            )],
            reprojection_errors=[ReprojectionErrorStats(
                camera_id=0, rms=0.25, mean=0.2, median=0.18,
                p90=0.7, p95=0.9, max=1.1, num_observations=5000)],
            overall_rms=0.25,
            board_poses=[],
            board_pose_times=[],
        )

        board = Chessboard(rows=7, cols=5, square_size=0.03)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.html")
            generate_html_report(
                result=result,
                all_frames={0: []},
                board=board,
                detection_images={},
                error_data={},
                plot_3d_fig=None,
                title="test",
                output_path=out,
            )
            with open(out) as f:
                html = f.read()

        assert "P90 (px)" in html
        assert "P95 (px)" in html
        # Values present in the table
        assert "0.7000" in html  # p90 formatted
        assert "0.9000" in html  # p95 formatted
