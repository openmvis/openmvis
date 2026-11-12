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
"""The shaking camera+IMU twin of the dense-knots and vision-IMU factor tests:
closed-form truth trajectory, board, rolling-shutter pixels, exact raw IMU."""
import numpy as np
import gtsam

from mvis.core.shutter import row_offset
from mvis.calibration.factors.lie_groups import exp_so3, so3_right_jacobian
from mvis.calibration.factors.projection import make_projector
from mvis.imu import CALYX, create_imu_model

_CV = np.array([430., 428, 320, 240, 0.0, 0.0, 0.0, 0.0])
_W, _H = 640, 480
_TR = 0.017

# Shaking motion: angular acceleration ~ 40 rad/s^2, linear 6 m/s^2.
_W0 = np.array([1.5, -1.0, 0.4])
_ALPHA = np.array([40.0, 25.0, -15.0])
_V0 = np.array([0.2, 0.1, -0.05])
_ACC = np.array([6.0, -4.0, 2.0])
_R0 = exp_so3(np.array([0.1, -0.2, 0.05]))


def _truth(t):
    """IMU pose in world at time t (rotation vector quadratic in t)."""
    R = _R0 @ exp_so3(_W0 * t + 0.5 * _ALPHA * t * t)
    p = np.array([0.0, 0.0, -1.0]) + _V0 * t + 0.5 * _ACC * t * t
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*p))


def _board():
    """Board 1 m in front of the camera at t=0 (camera = IMU, T_CtoI = I)."""
    T0 = _truth(0.0)
    T_BinW = T0.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 1.0)))
    g = np.linspace(-0.25, 0.25, 6)
    p_B = np.array([[x, y, 0.0] for x in g for y in g])
    return T_BinW, p_B


def _rs_measure(t_cap, T_BinW, p_B, calib):
    """Rolling-shutter pixels from the TRUE trajectory: row v is captured at
    t_cap + rho(v) * tr (fixed point on v)."""
    proj = make_projector(calib.model, calib.intr)
    uv = np.zeros((len(p_B), 2))
    for i, pb in enumerate(p_B):
        p_W = T_BinW.transformFrom(gtsam.Point3(*pb))
        v = _H / 2
        for _ in range(8):
            t = t_cap + row_offset(v, _H, 0) * _TR
            px, *_ = proj.project(_truth(t).compose(calib.T), p_W, jac=False)
            v = px[1]
        uv[i] = px
    return uv

_G = np.array([0.0, 0.0, -9.81])
_HZ = 200.0


def _true_body_rates(t):
    """Body rate and specific force of the test_dense_knots trajectory."""
    phi = _W0 * t + 0.5 * _ALPHA * t * t
    w = so3_right_jacobian(phi) @ (_W0 + _ALPHA * t)
    R = _truth(t).rotation().matrix()
    a = R.T @ (_ACC - _G)
    return w, a


def _true_velocity(t):
    return _V0 + _ACC * t


def _imu(intr=None, b_g=None, b_a=None, T=0.5, hz=_HZ):
    """Raw IMU record of the truth trajectory (exact, through intr.measure)."""
    intr = create_imu_model(CALYX) if intr is None else intr
    b_g = np.zeros(3) if b_g is None else b_g
    b_a = np.zeros(3) if b_a is None else b_a
    t = np.arange(-0.1, T, 1.0 / hz)
    w_m = np.zeros((len(t), 3)); a_m = np.zeros((len(t), 3))
    for i, ti in enumerate(t):
        w, a = _true_body_rates(ti)
        w_m[i], a_m[i] = intr.measure(w, a, b_g, b_a)
    return t, w_m, a_m


def _vb(t, b_g=None, b_a=None):
    return np.concatenate([_true_velocity(t),
                           np.zeros(3) if b_g is None else b_g,
                           np.zeros(3) if b_a is None else b_a])

_BG = np.array([0.003, -0.002, 0.001])
_BA = np.array([-0.13, 0.11, -0.3])
