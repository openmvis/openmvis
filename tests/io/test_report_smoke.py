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
"""End-to-end smoke test for the camera-only report path.

Guards the class of bug where the *calibration* succeeds but the *report*
generation crashes — e.g. the `generate_report` -> `motion_analysis._log_so3`
ImportError after the lie-groups rename, which silently turned the entire
camera-only regression red (calibrations were correct; the process exited 1 at
report time). The existing test_analysis test calls `generate_html_report`
directly and so missed it; this one drives `generate_calibration_report`, which
runs the motion-analysis path (the import + `log_so3` usage) that broke.

No dataset needed — a minimal synthetic CalibrationResult with a short board-pose
trajectory exercises the motion-metrics branch.
"""
import os
import tempfile

import numpy as np

from mvis.core.types import (
    CalibrationResult, CameraParams, ReprojectionErrorStats)
from mvis.detection.chessboard import Chessboard
from mvis.analysis.generate_report import generate_calibration_report


def _pose(rotvec, t):
    """4x4 SE(3) from a small rotation vector (board-pose trajectory sample)."""
    import gtsam
    return gtsam.Pose3(gtsam.Rot3.Expmap(np.asarray(rotvec, float)),
                       gtsam.Point3(0.0, 0.0, 1.5)).matrix()


def test_generate_calibration_report_runs_motion_path():
    result = CalibrationResult(
        cameras=[CameraParams(
            camera_id=0,
            intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
            distortion=np.array([-0.1, 0.01, 0.001, -0.001]),
            distortion_model="radtan",
            image_size=(640, 480))],
        reprojection_errors=[ReprojectionErrorStats(
            camera_id=0, rms=0.25, mean=0.2, median=0.18,
            p90=0.7, p95=0.9, max=1.1, num_observations=5000)],
        overall_rms=0.25,
        # a short board-pose trajectory -> exercises the motion / log_so3 branch
        board_poses=[_pose([0.0, 0.0, 0.0], 0.0),
                     _pose([0.02, -0.03, 0.05], 0.1),
                     _pose([0.05, -0.01, 0.10], 0.2)],
        board_pose_times=[0.0, 0.1, 0.2],
        reference_camera=0,
    )
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.html")
        # must not raise (this is the path the _log_so3 ImportError broke)
        generate_calibration_report(
            result=result,
            all_frames={0: []},
            board=board,
            calibrations={},
            image_sizes={0: (640, 480)},
            title="smoke",
            output_path=out,
        )
        assert os.path.exists(out)
        assert os.path.getsize(out) > 2000   # a real, non-trivial report
        with open(out) as f:
            assert "<!DOCTYPE html>" in f.read()


def test_data_quality_section_folded_by_default():
    """Calibration Data Quality renders inside a closed <details>."""
    result = CalibrationResult(
        cameras=[CameraParams(
            camera_id=0, intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
            distortion=np.array([-0.1, 0.01, 0.001, -0.001]),
            distortion_model="radtan", image_size=(640, 480))],
        reprojection_errors=[ReprojectionErrorStats(
            camera_id=0, rms=0.25, mean=0.2, median=0.18,
            p90=0.7, p95=0.9, max=1.1, num_observations=5000)],
        overall_rms=0.25,
        board_poses=[_pose([0.0, 0.0, 0.0], 0.0),
                     _pose([0.02, -0.03, 0.05], 0.1),
                     _pose([0.05, -0.01, 0.10], 0.2)],
        board_pose_times=[0.0, 0.1, 0.2], reference_camera=0)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.html")
        generate_calibration_report(
            result=result, all_frames={0: []},
            board=Chessboard(rows=7, cols=5, square_size=0.03),
            calibrations={}, image_sizes={0: (640, 480)},
            title="fold", output_path=out)
        html = open(out).read()
    i = html.index("Calibration Data Quality</h2>")
    nxt = html.index("<h2>", i)
    body = html[i:nxt]
    assert '<details class="appendix"><summary>' in body
    assert "<details class=\"appendix\" open" not in body


def test_result_files_and_config_folded_like_cam_imu():
    """Cam-only: verbatim config fold inside Configuration + a Result Files
    section (same format as the cam-IMU report), both near the end."""
    result = CalibrationResult(
        cameras=[CameraParams(
            camera_id=0, intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
            distortion=np.array([-0.1, 0.01, 0.001, -0.001]),
            distortion_model="radtan", image_size=(640, 480))],
        reprojection_errors=[ReprojectionErrorStats(
            camera_id=0, rms=0.25, mean=0.2, median=0.18,
            p90=0.7, p95=0.9, max=1.1, num_observations=5000)],
        overall_rms=0.25,
        board_poses=[_pose([0.0, 0.0, 0.0], 0.0), _pose([0.02, -0.03, 0.05], 0.1)],
        board_pose_times=[0.0, 0.1], reference_camera=0)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.yaml")
        open(cfg, "w").write("solve:\n  max_iterations: 15\n")
        calib = os.path.join(tmp, "x_calib.yaml")
        open(calib, "w").write("cam0:\n  fx: 500.0\n")
        out = os.path.join(tmp, "report.html")
        generate_calibration_report(
            result=result, all_frames={0: []},
            board=Chessboard(rows=7, cols=5, square_size=0.03),
            calibrations={}, image_sizes={0: (640, 480)}, title="t",
            output_path=out,
            calibration_info={"config_path": cfg, "result_files": [calib]})
        html = open(out).read()
    assert "OpenMVIS" in html
    i_res = html.index('id="sec-results"')
    i_cfg = html.index('id="sec-config"')
    assert html.index('id="sec-intrinsics"') < i_res < i_cfg       # moved to the end
    assert "x_calib.yaml</code> &middot; YAML" in html[i_res:i_cfg]
    assert "fx: 500.0" in html[i_res:i_cfg]
    cfg_html = html[i_cfg:html.index("</div>", html.index("</details>", i_cfg))]
    assert "<summary>Run configuration &middot; YAML, 2 lines" in cfg_html
    assert "max_iterations: 15" in cfg_html and 'class="copy-btn"' in cfg_html


def _appendix_report(tmp, **kw):
    """Report with 4 frames + their thumbnails; returns the HTML text."""
    from mvis.core.types import FrameObservations, Observation
    result = CalibrationResult(
        cameras=[CameraParams(
            camera_id=0, intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
            distortion=np.array([-0.1, 0.01, 0.001, -0.001]),
            distortion_model="radtan", image_size=(640, 480))],
        reprojection_errors=[ReprojectionErrorStats(
            camera_id=0, rms=0.25, mean=0.2, median=0.18,
            p90=0.7, p95=0.9, max=1.1, num_observations=10)],
        overall_rms=0.25, reference_camera=0)
    frames = [FrameObservations(0, 0.1 * i, [Observation(j, [10.0 * j, 5.0]) for j in range(i + 1)], (640, 480))
              for i in range(4)]
    thumbs = [np.full((24, 32, 3), 40 * i, np.uint8) for i in range(4)]
    out = os.path.join(tmp, "report.html")
    generate_calibration_report(
        result=result, all_frames={0: frames}, board=Chessboard(rows=7, cols=5, square_size=0.03),
        calibrations={}, image_sizes={0: (640, 480)}, title="appendix", output_path=out,
        all_detection_images={0: thumbs}, **kw)
    return open(out).read()


def test_lowest_detection_appendix_off_by_default():
    """Appendix A (PNG frames, ~11 MB on a real run) is left out unless asked for."""
    with tempfile.TemporaryDirectory() as tmp:
        html = _appendix_report(tmp)
    assert "sec-appendix-a" not in html
    assert 'class="appendix-item"' not in html


def test_lowest_detection_appendix_capped_by_appendix_max_frames():
    with tempfile.TemporaryDirectory() as tmp:
        html = _appendix_report(tmp, appendix_max_frames=2)
    assert 'id="sec-appendix-a"' in html
    assert "Up to 2 frames per camera" in html
    assert html.count('class="appendix-item"') == 2
