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
"""P4c (code-cleanup): one per-camera intrinsics/distortion block shared by the cam-only
(output_writer) and cam-IMU (cam_imu_output) serializers, instead of two hand-rolled copies."""
from mvis.io.result_schema import camera_intrinsics_block


def test_camera_intrinsics_block_shape():
    b = camera_intrinsics_block(0, "radtan", [400, 401, 320, 240],
                                [0.1, -0.2, 0.001, 0.002], (640, 480), 0.0)
    assert b["id"] == 0
    assert b["model"] == "pinhole_radtan"
    assert b["intrinsics"] == {"fx": 400.0, "fy": 401.0, "cx": 320.0, "cy": 240.0}
    assert b["distortion"] == {"model": "radtan", "coeffs": [0.1, -0.2, 0.001, 0.002]}
    assert b["image_size"] == [640, 480]
    assert b["rolling_shutter_readout"] == 0.0


def test_floats_are_plain_python_floats():
    import numpy as np
    b = camera_intrinsics_block(1, "equidistant", np.array([1, 2, 3, 4]),
                                np.array([0.1, 0.2]), (1280, 800), 0.012)
    assert all(isinstance(v, float) for v in b["intrinsics"].values())
    assert all(isinstance(c, float) for c in b["distortion"]["coeffs"])
    assert b["rolling_shutter_readout"] == 0.012


def test_imu_diag_extracts_rounded_scales():
    import numpy as np
    from mvis.imu import ImuModel
    from mvis.io.result_schema import imu_diag
    ii = ImuModel.from_tangent(np.zeros(24), "calyx")     # identity model
    dw, da = imu_diag(ii)
    assert dw == [1.0, 1.0, 1.0] and da == [1.0, 1.0, 1.0]
    # ndigits is honored (summary uses 5, writer/report use 6) so callers keep exact output
    dw5, _ = imu_diag(ii, ndigits=5)
    assert dw5 == [1.0, 1.0, 1.0]
    assert all(isinstance(x, float) for x in dw)
