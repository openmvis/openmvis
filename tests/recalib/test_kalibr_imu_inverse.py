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
"""mvis.io.kalibr.invert_scaled_misaligned_imu undoes Kalibr's scale-misalignment IMU model."""
import numpy as np

from mvis.io.kalibr import invert_scaled_misaligned_imu


def _rot(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()


def test_inverse_recovers_the_true_measurement():
    rng = np.random.default_rng(3)
    M_g = np.tril(np.eye(3) + 0.01 * rng.standard_normal((3, 3)))
    M_a = np.tril(np.eye(3) + 0.01 * rng.standard_normal((3, 3)))
    C = _rot([0.01, -0.02, 0.005])
    A = 1e-3 * rng.standard_normal((3, 3))
    b_g, b_a = 1e-3 * rng.standard_normal(3), 0.05 * rng.standard_normal(3)
    for _ in range(20):
        w, a = rng.standard_normal(3), 9.8 * rng.standard_normal(3)
        a_m = M_a @ a + b_a
        w_m = M_g @ (C @ w) + A @ (C @ a) + b_g
        w2, a2 = invert_scaled_misaligned_imu(w_m, a_m, b_g, b_a, M_g, M_a, C, A)
        assert np.allclose(w2, w, atol=1e-12) and np.allclose(a2, a, atol=1e-12)


def test_identity_model_is_a_bitwise_no_op():
    rng = np.random.default_rng(0)
    w_m, a_m = rng.standard_normal(3), rng.standard_normal(3)
    I, Z = np.eye(3), np.zeros((3, 3))
    w, a = invert_scaled_misaligned_imu(w_m, a_m, np.zeros(3), np.zeros(3), I, I, I, Z)
    assert w.tobytes() == w_m.tobytes() and a.tobytes() == a_m.tobytes()
