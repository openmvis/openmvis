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
"""Board pose block of the cam-IMU summary: only the attitude to gravity is gauge-free."""
import json

import gtsam
import numpy as np

from mvis.cli.main import board_pose_summary


def test_flat_board_has_gravity_along_minus_normal():
    s = board_pose_summary(gtsam.Pose3())
    assert np.allclose(s["g_inB"], [0, 0, -1]) and s["normal_tilt_deg"] == 0.0


def test_vertical_board_and_yaw_invariance():
    R = gtsam.Rot3(np.column_stack([[0, 1, 0], [0, 0, -1], [1, 0, 0]]))   # board y points down
    s = board_pose_summary(gtsam.Pose3(R, gtsam.Point3(1, 2, 3)))
    assert np.allclose(s["g_inB"], [0, 1, 0]) and abs(s["normal_tilt_deg"] - 90) < 1e-9
    assert abs(s["g_inplane_deg"] - 90) < 1e-9
    # a world yaw (the gauge) changes T_BinW but not the gravity-relative attitude
    Y = gtsam.Pose3(gtsam.Rot3.Yaw(0.7), gtsam.Point3(5, 0, 0))
    s2 = board_pose_summary(Y.compose(gtsam.Pose3(R, gtsam.Point3(1, 2, 3))))
    assert np.allclose(s2["g_inB"], s["g_inB"], atol=1e-9) and s2["T_BinW"] != s["T_BinW"]
    json.dumps(s2)
