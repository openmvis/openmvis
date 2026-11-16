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
"""A calyx IMU's Dw shown as R · U in the cam-IMU report: the same QR split as cam_imu.json's
Dw_scale / Dw_shear / gyro_misalign_rotvec, R a rotation, U upper-triangular with a positive diagonal."""
import numpy as np
import pytest

from mvis.analysis.cam_imu_report import _dw_decomp, _dw_decomp_html, _text_dw_decomp
from mvis.calibration.factors.lie_groups import exp_so3
from mvis.io.result_schema import calyx_dw_decomposed, calyx_dw_qr

R0 = exp_so3(np.array([0.002, -0.001, 0.0015]))
U0 = np.array([[0.995, -0.002, 0.0008], [0.0, 0.998, 0.0015], [0.0, 0.0, 0.9997]])
DW = R0 @ U0


def test_qr_split_recovers_rotation_and_upper_triangular_factor():
    R, U = calyx_dw_qr(DW)
    np.testing.assert_allclose(R @ U, DW, atol=1e-12)
    np.testing.assert_allclose(R, R0, atol=1e-12)
    np.testing.assert_allclose(U, U0, atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_report_split_matches_the_summary_fields():
    dd = _dw_decomp(DW)
    s = calyx_dw_decomposed(DW, 9)
    assert np.allclose(dd["scale"], s["Dw_scale"]) and np.allclose(dd["shear"], s["Dw_shear"])
    assert np.allclose(dd["rotvec_rad"], s["gyro_misalign_rotvec"], atol=1e-6)
    assert dd["angle_rad"] == pytest.approx(np.linalg.norm([0.002, -0.001, 0.0015]))


def test_html_and_text_show_scale_shear_and_misalignment():
    dd = _dw_decomp(DW)
    html = _dw_decomp_html(dd)
    assert "Dw = R · U (calyx)" in html and "R (misalignment)" in html and "U (scale + shear)" in html
    assert "+0.995000" in html and "misalignment log(R) (rad)" in html
    text = "\n".join(_text_dw_decomp(dd))
    assert "scale diag(U) 0.99500 0.99800 0.99970" in text and "|0.002693|" in text
