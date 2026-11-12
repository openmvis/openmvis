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
"""SE(3) wiggle trajectory and non-identity IMU intrinsics shared by the IMU tests."""
import numpy as np
import gtsam

from mvis.imu import ImuModel, create_imu_model


def pose_mat(R, p):
    T = np.eye(4); T[0:3, 0:3] = R; T[0:3, 3] = p
    return T


def wiggle(t):
    """A smooth, fully-coupled SE(3) trajectory (rotation + translation)."""
    R = (gtsam.Rot3.Rz(0.5 * np.sin(0.9 * t))
         * gtsam.Rot3.Ry(0.3 * t)
         * gtsam.Rot3.Rx(0.2 * np.cos(0.7 * t))).matrix()
    p = np.array([2.0 * np.sin(0.7 * t), 1.5 * np.cos(0.5 * t), 0.2 * t])
    return pose_mat(R, p)


def intr_full(model):
    """Fixed non-identity Dw/Da/Tg/R (6-parameter Dw layout)."""
    return ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.array([0.01, -0.02, 0.0, 0.005, 0.0, -0.01, 0.0, 0.008, 0.0]),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(), model=model)


def intr_layout(model):
    """Non-identity intrinsics built from each model's own identity layout
    (the Dw/Da parameter vectors are laid out differently per model)."""
    m = create_imu_model(model)
    d_dw = np.array([0.02, 0.01, -0.02, -0.01, -0.03, 0.015, 0.02, -0.01, 0.01])
    d_da = np.array([0.01, 0.02, -0.01, -0.01, 0.015, 0.02])
    x_tg = np.array([0.002, -0.001, 0.0005, 0.001, 0.003, -0.002, 0.0, 0.001, -0.001])
    return create_imu_model(model, x_dw=m.x_dw + d_dw[:len(m.x_dw)],
                            x_da=m.x_da + d_da, x_tg=x_tg)
