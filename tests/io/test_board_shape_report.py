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
"""Board Shape section of the camera-only and cam-IMU reports: present with a map when the run
estimated the board shape, absent otherwise."""
import os
import tempfile

import numpy as np
import pytest

from mvis.analysis.cam_imu_report import cam_imu_html_report
from mvis.analysis.plot_errors import plot_board_shape
from mvis.core.config import BoardConfig
from mvis.detection.board_factory import create_board
from tests.helpers.cam_imu_report_fakes import _fake_result

BOARD = create_board(BoardConfig(board_type="apriltag", rows=3, cols=4, tag_size=0.05,
                                 tag_spacing=0.3, tag_family="t36h11"))


def _bowl(board):
    p = np.asarray(board.all_points_3d(), float)
    r2 = (p[:, 0] - p[:, 0].mean()) ** 2 + (p[:, 1] - p[:, 1].mean()) ** 2
    h = 0.004 * r2 / r2.max()
    return h - h.mean()


def _section(html, sec_id):
    i = html.index(f'id="{sec_id}"')
    return html[i:html.index('<div class="section', i + 1)]


def test_plot_marks_unseen_corners_and_reports_seen_stats():
    h = _bowl(BOARD)
    n_obs = np.ones(len(h), int)
    n_obs[:4] = 0
    ax = plot_board_shape(h, BOARD, n_obs).axes[0]
    assert "corners seen" in ax.get_title() and f"{len(h) - 4} corners" in ax.get_title()
    assert ax.get_legend() is not None                          # the hollow "not seen" markers


def test_plot_refuses_heights_for_another_board():
    with pytest.raises(ValueError, match="board heights"):
        plot_board_shape(np.zeros(5), BOARD)


def test_cam_imu_report_shows_board_shape_only_when_estimated():
    plain = cam_imu_html_report(_fake_result(), board=BOARD)
    assert 'id="sec-board-shape"' not in plain

    r = _fake_result()
    h = _bowl(BOARD)
    r.diagnostics["board_heights"] = dict(heights_m=h, sigma_m=np.full(len(h), 2e-4), n_obs=np.ones(len(h), int),
                                         rms_mm=float(np.std(h) * 1e3), p2p_mm=float(np.ptp(h) * 1e3),
                                         n_seen=len(h))
    body = _section(cam_imu_html_report(r, board=BOARD), "sec-board-shape")
    assert f"{np.ptp(h) * 1e3:.2f}" in body and "0.20" in body             # peak to peak, median sigma
    assert "data:image/png;base64," in body                                # the map

    no_board = _section(cam_imu_html_report(r), "sec-board-shape")          # table only
    assert "data:image/png" not in no_board and "Peak to peak" in no_board


def test_camera_only_report_shows_board_shape_only_when_estimated():
    from mvis.analysis.generate_report import generate_calibration_report
    from mvis.core.types import CalibrationResult, CameraParams, ReprojectionErrorStats

    def html_for(heights):
        result = CalibrationResult(
            cameras=[CameraParams(camera_id=0, intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
                                  distortion=np.zeros(4), distortion_model="radtan", image_size=(640, 480))],
            reprojection_errors=[ReprojectionErrorStats(camera_id=0, rms=0.25, mean=0.2, median=0.18, p90=0.7,
                                                        p95=0.9, max=1.1, num_observations=100)],
            overall_rms=0.25, reference_camera=0, board_heights=heights,
            board_height_n_obs=None if heights is None else np.r_[np.zeros(4, int), np.ones(len(heights) - 4, int)])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.html")
            generate_calibration_report(result=result, all_frames={0: []}, board=BOARD, calibrations={},
                                        image_sizes={0: (640, 480)}, title="t", output_path=out)
            return open(out).read()

    assert 'id="sec-board-shape"' not in html_for(None)
    h = _bowl(BOARD)
    body = _section(html_for(h), "sec-board-shape")
    assert f"{np.ptp(h[4:]) * 1e3:.2f} mm" in body and "data:image/png;base64," in body   # stats over seen corners
    assert f"{len(h) - 4} of {len(h)}" in body


def test_camera_only_solve_counts_the_corners_it_used():
    """Corners left out by corner subsampling get no factor, so their height is unconstrained: the
    solve says so with board_height_n_obs, and the written board_shape keeps its stats to the rest."""
    from mvis.io.output_writer import OutputWriter
    from mvis.core.types import CalibrationResult, CameraParams, ReprojectionErrorStats
    h = np.array([0.0, 0.002, 0.0, -0.002])
    r = CalibrationResult(
        cameras=[CameraParams(camera_id=0, intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
                              distortion=np.zeros(4), distortion_model="radtan", image_size=(640, 480))],
        reprojection_errors=[ReprojectionErrorStats(0, 0.3, 0.3, 0.3)], overall_rms=0.3,
        board_heights=h, board_height_n_obs=np.array([0, 5, 0, 5]))
    d = OutputWriter.to_calibration_dict(r)["board_shape"]
    assert d["n_obs"] == [0, 5, 0, 5] and abs(d["rms_mm"] - 2.0) < 1e-9 and abs(d["peak_to_peak_mm"] - 4.0) < 1e-9
