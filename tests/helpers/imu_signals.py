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
"""Raw IMU signals and intrinsics for the preintegration tests."""
import numpy as np

from mvis.imu import RPNG, create_imu_model


def _make_sequence(n=40, dt=0.003, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n + 1) * dt
    w = np.array([[0.3 * np.sin(2 * tt), 0.2 * np.cos(3 * tt), 0.1 * tt] for tt in t]) \
        + rng.randn(n + 1, 3) * 0.01
    a = np.array([[0.5 * np.cos(tt), 9.81 + 0.3 * np.sin(tt), 0.2 * tt] for tt in t]) \
        + rng.randn(n + 1, 3) * 0.01
    return t, w, a


def _w(t):
    return np.array([0.8 * np.sin(2 * np.pi * 0.5 * t),
                     0.5 * np.sin(2 * np.pi * 0.7 * t + 1.0),
                     0.3 * np.cos(2 * np.pi * 0.3 * t)])


def _a(t):
    return np.array([1.5 * np.sin(2 * np.pi * 0.6 * t),
                     9.81 + 0.8 * np.sin(2 * np.pi * 0.4 * t + 0.5),
                     0.9 * np.cos(2 * np.pi * 0.8 * t)])


def _intr_with_tg():
    """RPNG intrinsics with non-trivial Dw/Da/Tg so the g-sensitivity path of
    J_R_tga is exercised (identity intrinsics zero the gyro row)."""
    xi = np.zeros(24)
    xi[0:6] = [0.02, -0.01, 0.015, 0.003, -0.002, 0.004]      # Dw
    xi[6:12] = [-0.015, 0.01, 0.02, -0.004, 0.002, 0.003]     # Da
    xi[12:21] = 0.01 * np.array([1, -2, 3, 2, -1, 1, -3, 2, -1])  # Tg
    return create_imu_model(RPNG).retract(xi)
