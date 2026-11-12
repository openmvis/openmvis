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

def test_calyx_dw_decomposed_roundtrip():
    import numpy as np
    from mvis.io.result_schema import calyx_dw_decomposed
    th = 0.05
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    U = np.array([[1.01, 0.002, -0.003], [0, 0.99, 0.004], [0, 0, 1.005]])
    d = calyx_dw_decomposed(R @ U)
    assert abs(d["Dw_scale"][0] - 1.01) < 1e-4
    assert abs(d["Dw_scale"][1] - 0.99) < 1e-4
    assert abs(d["gyro_misalign_rotvec"][2] - th) < 1e-5
    ident = calyx_dw_decomposed(np.eye(3))
    assert ident["Dw_scale"] == [1.0, 1.0, 1.0]
    assert ident["gyro_misalign_rotvec"] == [0.0, 0.0, 0.0]
