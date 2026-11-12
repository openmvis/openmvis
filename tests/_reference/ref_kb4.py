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
"""Frozen, separately written kb4 formula, the G3 truth (plan kb4-kb8). Never edited, and it
imports nothing from mvis, so a wrong kernel cannot pass by generating its own data.

`tests/_reference/ref_projection.py` does not qualify: it imports `project_kb` live.

    theta = atan2(rho, Z), theta_d = theta (1 + k1 t2 + k2 t2^2 + k3 t2^3 + k4 t2^4), t2 = theta^2
    u = fx theta_d X / rho + cx,  v = fy theta_d Y / rho + cy

`basalt_project` is Basalt's `kannala_brandt_camera4.hpp:157-171`, in its Horner form.
"""
import math


def project(P, cal):
    X, Y, Z = float(P[0]), float(P[1]), float(P[2])
    fx, fy, cx, cy, k1, k2, k3, k4 = [float(v) for v in cal]
    rho = math.sqrt(X * X + Y * Y)
    th = math.atan2(rho, Z)
    t2 = th * th
    thd = th + k1 * th * t2 + k2 * th * t2 ** 2 + k3 * th * t2 ** 3 + k4 * th * t2 ** 4
    if rho < 1e-300:
        return (cx, cy)
    return (fx * thd * X / rho + cx, fy * thd * Y / rho + cy)


def basalt_project(P, cal):
    x, y, z = float(P[0]), float(P[1]), float(P[2])
    fx, fy, cx, cy, k1, k2, k3, k4 = [float(v) for v in cal]
    r = math.sqrt(x * x + y * y)
    if r > 1e-8:
        th = math.atan2(r, z)
        t2 = th * th
        r_th = k4 * t2
        r_th = (r_th + k3) * t2
        r_th = (r_th + k2) * t2
        r_th = (r_th + k1) * t2
        r_th = (r_th + 1.0) * th
        return (fx * x * r_th / r + cx, fy * y * r_th / r + cy)
    return (fx * x / z + cx, fy * y / z + cy)


def theta_d_px(cal, th):
    """f * theta_d(theta), the radial mapping a fit must reproduce."""
    fx, _fy, _cx, _cy, k1, k2, k3, k4 = [float(v) for v in cal]
    t2 = th * th
    return fx * th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
