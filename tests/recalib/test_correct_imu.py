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
# tests/recalib/test_correct_imu.py
import numpy as np
from types import SimpleNamespace
from mvis.imu import ImuModel
from mvis.recalib.calib_spec import ImuCalib
from mvis.recalib.correct_imu import correct_imu_msg

def _msg(w, a):
    return SimpleNamespace(
        angular_velocity=SimpleNamespace(x=w[0], y=w[1], z=w[2]),
        linear_acceleration=SimpleNamespace(x=a[0], y=a[1], z=a[2]))

def test_correct_recovers_truth():
    rng = np.random.default_rng(2)
    xi = rng.normal(scale=1e-3, size=24)
    intr = ImuModel.from_tangent(xi, "rpng")
    b_g, b_a = np.array([0.01, -0.02, 0.03]), np.array([0.1, -0.2, 0.3])
    w_true, a_true = rng.normal(size=3), rng.normal(size=3)
    w_m, a_m = intr.measure(w_true, a_true, b_g, b_a)        # forward (raw)
    imu = ImuCalib("imu0", "rpng", intr, b_g, b_a, 0.0, "base")
    out = correct_imu_msg(_msg(w_m, a_m), imu, remove_bias=True)
    np.testing.assert_allclose(
        [out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z],
        w_true, atol=1e-9)
    np.testing.assert_allclose(
        [out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z],
        a_true, atol=1e-9)

def test_keep_bias_leaves_bias_in():
    rng = np.random.default_rng(3)
    intr = ImuModel.from_tangent(np.zeros(24), "rpng")       # identity intrinsics
    b_g, b_a = np.array([0.05, 0, 0]), np.array([0, 0.05, 0])
    imu = ImuCalib("imu0", "rpng", intr, b_g, b_a, 0.0, "base")
    out = correct_imu_msg(_msg([0.05, 0, 0], [0, 0.05, 0]), imu, remove_bias=False)
    # identity intrinsics + keep bias => output unchanged from raw
    assert abs(out.angular_velocity.x - 0.05) < 1e-12
    assert abs(out.linear_acceleration.y - 0.05) < 1e-12
