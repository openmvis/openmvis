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
"""Analytic pixel-vs-intrinsics Jacobians for radtan (Cal3DS2) and fisheye (Cal3Fisheye).

Moved verbatim from `mvis.calibration.factors.camera_intrinsics` (2026-11-07,
post-merge cleanup PR B): no production code called them, the factors use
GTSAM's `Dcal`. `tests/camera/test_analytical_jacobians.py` checks them against
finite differences and GTSAM. Imports nothing from mvis.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-10


def intrinsic_jacobian_radtan(p_cam: np.ndarray, cal_vec: np.ndarray) -> np.ndarray:
    """Analytical Jacobian of pixel w.r.t. intrinsics for radtan (Cal3DS2).

    K = [fx, fy, cx, cy, k1, k2, p1, p2]. Returns 2×8.
    """
    X, Y, Z = p_cam[0], p_cam[1], p_cam[2]
    fx, fy = cal_vec[0], cal_vec[1]
    k1, k2, p1, p2 = cal_vec[4], cal_vec[5], cal_vec[6], cal_vec[7]

    x_n = X / Z
    y_n = Y / Z
    r2 = x_n ** 2 + y_n ** 2
    r4 = r2 ** 2
    rad = 1.0 + k1 * r2 + k2 * r4
    x_d = rad * x_n + 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n ** 2)
    y_d = rad * y_n + p1 * (r2 + 2.0 * y_n ** 2) + 2.0 * p2 * x_n * y_n

    J = np.zeros((2, 8))
    J[0, 0] = x_d; J[0, 2] = 1.0
    J[1, 1] = y_d; J[1, 3] = 1.0
    J[0, 4] = fx * x_n * r2
    J[0, 5] = fx * x_n * r4
    J[0, 6] = fx * 2.0 * x_n * y_n
    J[0, 7] = fx * (r2 + 2.0 * x_n ** 2)
    J[1, 4] = fy * y_n * r2
    J[1, 5] = fy * y_n * r4
    J[1, 6] = fy * (r2 + 2.0 * y_n ** 2)
    J[1, 7] = fy * 2.0 * x_n * y_n
    return J


def intrinsic_jacobian_fisheye(p_cam: np.ndarray, cal_vec: np.ndarray) -> np.ndarray:
    """Analytical Jacobian of pixel w.r.t. intrinsics for fisheye (Cal3Fisheye).

    K = [fx, fy, cx, cy, k1, k2, k3, k4]. Returns 2×8.
    """
    X, Y, Z = p_cam[0], p_cam[1], p_cam[2]
    fx, fy = cal_vec[0], cal_vec[1]
    k1, k2, k3, k4 = cal_vec[4], cal_vec[5], cal_vec[6], cal_vec[7]

    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(rho, Z)
    if rho < _EPS:
        s = 1.0
    else:
        r_d = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9
        s = r_d / rho

    J = np.zeros((2, 8))
    J[0, 0] = s * X if rho >= _EPS else X
    J[1, 1] = s * Y if rho >= _EPS else Y
    J[0, 2] = 1.0
    J[1, 3] = 1.0
    if rho >= _EPS:
        for i, power in enumerate((3, 5, 7, 9)):
            J[0, 4 + i] = fx * X * theta ** power / rho
            J[1, 4 + i] = fy * Y * theta ** power / rho
    return J
