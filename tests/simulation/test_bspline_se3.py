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
"""BsplineSE3 trajectory generator: degenerate cases + finite-difference checks
that the closed-form velocity/acceleration match differentiating the spline's
own pose. Self-consistent (FD is taken on the spline, not on the sampled fn)."""

import numpy as np
import gtsam
import pytest

from mvis.simulation.bspline_se3 import BsplineSE3
from tests.helpers.imu_fixtures import (wiggle as _wiggle, pose_mat as _pose)
from mvis.calibration.factors.lie_groups import vee_so3 as _vee


def _Rz(a):
    return gtsam.Rot3.Rz(a).matrix()


def _spline(dt=0.1):
    return BsplineSE3.from_pose_fn(_wiggle, t0=0.0, t1=6.0, dt=dt)


# --------------------------------------------------------------------------
def test_static_trajectory():
    """Identical control poses -> constant pose, zero velocity/acceleration."""
    R = _Rz(0.4)
    p = np.array([1.0, -2.0, 0.5])
    spl = BsplineSE3(0.0, 0.1, [_pose(R, p)] * 6)
    for t in (0.15, 0.3, 0.45):
        R_GtoI, p_q, w, v, alpha, a = spl.get_acceleration(t)
        assert np.allclose(R_GtoI, R.T, atol=1e-9)
        assert np.allclose(p_q, p, atol=1e-9)
        assert np.allclose(w, 0, atol=1e-9)
        assert np.allclose(v, 0, atol=1e-9)
        assert np.allclose(alpha, 0, atol=1e-9)
        assert np.allclose(a, 0, atol=1e-9)


def test_constant_linear_velocity():
    """Pure translation at constant rate -> v = const, w = 0, a = 0."""
    vel = np.array([0.3, -0.1, 0.2])
    dt = 0.1
    ctrl = [_pose(np.eye(3), i * dt * vel) for i in range(8)]
    spl = BsplineSE3(0.0, dt, ctrl)
    for t in (0.25, 0.4, 0.55):
        _, _, w, v, alpha, a = spl.get_acceleration(t)
        assert np.allclose(v, vel, atol=1e-9)
        assert np.allclose(w, 0, atol=1e-9)
        assert np.allclose(a, 0, atol=1e-9)
        assert np.allclose(alpha, 0, atol=1e-9)


def test_velocity_matches_finite_difference():
    spl = _spline()
    h = 1e-5
    for t in (1.0, 2.3, 3.7, 5.0):
        R_GtoI, p, w, v = spl.get_velocity(t)
        R0, p0 = spl.get_pose(t - h)
        R1, p1 = spl.get_pose(t + h)
        v_fd = (p1 - p0) / (2 * h)
        # w_IinI from R_ItoG and its derivative
        Rd = (R1.T - R0.T) / (2 * h)            # d R_ItoG / dt
        w_fd = _vee(R_GtoI @ Rd)                # R_ItoG^T = R_GtoI
        assert np.allclose(v, v_fd, atol=1e-5), f"v @ t={t}"
        assert np.allclose(w, w_fd, atol=1e-5), f"w @ t={t}"
        # pose self-consistency between the two queries
        assert np.allclose(R_GtoI, spl.get_pose(t)[0], atol=1e-12)


def test_acceleration_matches_finite_difference():
    spl = _spline()
    h = 1e-5
    for t in (1.0, 2.3, 3.7, 5.0):
        _, _, w, v, alpha, a = spl.get_acceleration(t)
        _, _, w0, v0 = spl.get_velocity(t - h)
        _, _, w1, v1 = spl.get_velocity(t + h)
        a_fd = (v1 - v0) / (2 * h)
        alpha_fd = (w1 - w0) / (2 * h)          # d w_IinI/dt == alpha_IinI
        assert np.allclose(a, a_fd, atol=1e-4), f"a @ t={t}"
        assert np.allclose(alpha, alpha_fd, atol=1e-4), f"alpha @ t={t}"


def test_queryable_range():
    spl = _spline(dt=0.5)
    assert spl.start_time == pytest.approx(spl.t0 + 0.5)
    # interior queries succeed
    R_GtoI, p = spl.get_pose(0.5 * (spl.start_time + spl.end_time))
    assert R_GtoI.shape == (3, 3) and p.shape == (3,)


def test_constant_axis_rotation_closed_form():
    """Equally-spaced rotations about a fixed axis -> constant body angular rate,
    zero angular acceleration, zero linear velocity (closed-form, exact)."""
    dt, dth = 0.05, 0.1
    ctrl = [_pose(gtsam.Rot3.Rz(k * dth).matrix(), np.zeros(3)) for k in range(8)]
    spl = BsplineSE3(0.0, dt, ctrl)
    omega = dth / dt
    for t in (0.2, 0.3, 0.4):
        _, _, w, v, alpha, a = spl.get_acceleration(t)
        assert np.allclose(w, [0, 0, omega], atol=1e-6)
        assert np.allclose(alpha, 0, atol=1e-6)
        assert np.allclose(v, 0, atol=1e-9)
        assert np.allclose(a, 0, atol=1e-6)


def test_uniform_circular_centripetal():
    """Uniform circular translation (fixed orientation) -> centripetal accel
    a ~= -omega^2 * r toward the center, zero angular velocity."""
    r, om, dt, n = 2.0, 1.5, 0.02, 40
    ctrl = [_pose(np.eye(3), np.array([r * np.cos(om * k * dt),
                                       r * np.sin(om * k * dt), 0.0]))
            for k in range(n)]
    spl = BsplineSE3(0.0, dt, ctrl)
    for t in (0.3, 0.4, 0.5):
        _, p, w, _v, _alpha, a = spl.get_acceleration(t)
        assert np.allclose(w, 0, atol=1e-3)
        a_exp = -om ** 2 * np.array([p[0], p[1], 0.0])
        assert np.allclose(a, a_exp, rtol=0.03, atol=0.03)


def test_right_translation_invariance():
    """spline(P_i · G)(t) == spline(P_i)(t) · G for constant G (cumulative-spline
    property; underpins the multi-IMU dual-derivation test)."""
    base = _spline()
    G = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.1, -0.2, 0.05])),
                    gtsam.Point3(0.3, -0.1, 0.2)).matrix()
    aux = BsplineSE3(base.t0, base.dt, [P @ G for P in base.ctrl])
    for t in (1.0, 2.5, 4.0):
        Tb, _, _ = base._eval(t, 0)
        Ta, _, _ = aux._eval(t, 0)
        assert np.allclose(Ta, Tb @ G, atol=1e-10)
