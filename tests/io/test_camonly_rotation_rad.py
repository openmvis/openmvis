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
"""Camera-only outputs report extrinsic rotations in rad (decision D10, F13)."""
import numpy as np

from mvis.analysis import html_report
from mvis.calibration.factors.lie_groups import exp_so3
from mvis.core.types import (CalibrationResult, CameraParams, ExtrinsicTransform,
                             ReprojectionErrorStats)
from mvis.detection.chessboard import Chessboard
from mvis.io.output_writer import OutputWriter

SO3 = np.array([0.01, -0.02, 0.03])
T_XYZ = np.array([-0.1, 0.002, 0.003])


def _result():
    cams = [CameraParams(camera_id=i, intrinsics=np.array([500., 500., 320., 240.]), distortion=np.zeros(4),
                         distortion_model="radtan", image_size=(640, 480)) for i in (0, 1)]
    T = np.eye(4)
    T[:3, :3] = np.asarray(exp_so3(SO3), float)
    T[:3, 3] = T_XYZ
    ext = ExtrinsicTransform(from_camera=1, to_camera=0, T_to_from=T, so3=SO3, translation=T_XYZ)
    return CalibrationResult(cameras=cams, extrinsics=[ext], reference_camera=0, overall_rms=0.3,
                             reprojection_errors=[ReprojectionErrorStats(i, 0.3, 0.3, 0.3) for i in (0, 1)])


def test_html_report_extrinsics_rotation_column_in_rad(tmp_path):
    out = tmp_path / "r.html"
    html_report.generate_html_report(_result(), {0: [], 1: []}, Chessboard(rows=7, cols=5, square_size=0.03),
                                     {}, {}, None, output_path=str(out))
    html = out.read_text()
    assert "<th>|Rotation| (rad)</th>" in html and "Rotation (deg)" not in html
    assert f'<td class="num">{float(np.linalg.norm(SO3)):.6f}</td>' in html


def test_cam_stats_extrinsics_list_carries_rot_rad(tmp_path, monkeypatch):
    from mvis.regression import cam_stats
    from mvis.regression.manifest import RunSpec
    monkeypatch.setattr(cam_stats, "RESULTS_ROOT", tmp_path)
    d = tmp_path / "ds" / "c"
    d.mkdir(parents=True)
    OutputWriter.write_calibration_yaml(_result(), str(d / "ds_c.yaml"))
    s = cam_stats.collect_stats(RunSpec(mode="cam", dataset="ds", collection="c", config=""))
    e = s["extrinsics_list"][0]
    assert abs(e["rot_rad"] - float(np.linalg.norm(SO3))) < 1e-12
    assert "rot_deg" not in e
