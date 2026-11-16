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
"""The Summary's "Calibration parameters by sensor" folds (param_folds) in the camera-IMU and the
camera-only report: one closed fold per sensor, key numbers on the closed line, every parameter inside
with ± σ, the sensor's YAML behind a Copy button."""
import re

import numpy as np

from mvis.analysis.param_folds import folds_block, num, sensor_fold, val, vec


def test_formats_one_precision_per_unit_and_sigma_only_when_nonzero():
    assert num(449.42381, "px2") == "449.42" and num(0.14213, "px") == "0.142" and num(0.0153, "ms") == "+0.015"
    assert val(53.77, "mm", 0.47) == "53.77 <span class='pm'>± 0.47</span> mm"
    assert val(53.77, "mm", 0.0) == "53.77 mm" and val(None, "mm") == "—"
    assert val(449.42, "px2", 1e-9) == "449.42 px"          # a pinned parameter's ~0 σ is not printed
    assert vec([1.0, -2.0, 3.0], "mm", [0.1, 0.0, None]).count("± 0.10") == 1


def test_fold_is_closed_and_leaves_out_empty_groups_and_rows():
    html = sensor_fold("cam0", "kb4 · fx 449.42", [("A", [("x", "1"), ("y", None)]), ("Empty", [])], "id: 0\n")
    assert html.startswith("<details class='appendix param-fold'>") and " open" not in html.split(">")[0]
    assert "<th>x</th>" in html and "<th>y</th>" not in html and "Empty" not in html
    assert "class='copy-btn'" in html and "id: 0" in html
    assert "Expand all" in folds_block([html]) and folds_block([]) == ""


def test_cam_imu_summary_has_a_fold_per_camera_and_imu():
    from mvis.analysis.cam_imu_report import cam_imu_html_report
    from tests.io.test_cam_imu_report import _result_with_geometry
    r = _result_with_geometry()
    for e in r.cameras:
        e.update(reproj_median_px=0.142, reproj_p95_px=0.312, reproj_mean_px=0.155, image_size=(1280, 1024))
    html = cam_imu_html_report(r, config="solve:\n  sigma_g: 6.0e-4\n")
    summary = html[html.index('id="sec-summary"'):html.index('id="sec-3d"') if 'id="sec-3d"' in html else None]
    heads = re.findall(r"<summary><b>([^<]+)</b>", summary)
    assert heads[:1] == ["cam0"] and "imu0 (base)" in heads
    assert "reproj median / P95 0.142 / 0.312 px" in summary and "sigma_g" in summary
    assert "lever x, y, z (mm)" in summary and "field of view H × V" in summary


def test_cam_only_summary_has_a_fold_per_camera_with_its_pose_and_yaml():
    from mvis.analysis.html_report import _cam_only_param_folds
    from mvis.core.types import CalibrationResult, CameraParams, ExtrinsicTransform, ReprojectionErrorStats
    cams = [CameraParams(camera_id=i, intrinsics=[450.0, 450.0, 640.0, 512.0], distortion=[0.23, -0.01, -0.03, 0.01],
                         distortion_model="kb4", image_size=(1280, 1024)) for i in range(2)]
    T = np.eye(4); T[:3, 3] = [0.0664, 0.0, 0.0]
    res = CalibrationResult(cameras=cams, extrinsics=[ExtrinsicTransform(1, 0, T, np.zeros(3), T[:3, 3])],
                            reprojection_errors=[ReprojectionErrorStats(i, 0.2, 0.15, 0.14, 1.5, 1000, 0.27, 0.31, 0.09)
                                                 for i in range(2)])
    html = _cam_only_param_folds(res)
    assert re.findall(r"<summary><b>([^<]+)</b>", html) == ["cam0", "cam1"]
    assert "baseline 66.40 mm to cam0" in html and "Pose in cam0" in html and "k1" in html
    assert html.count("class='copy-btn'") == 2 and "extrinsics:" in html
