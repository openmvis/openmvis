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
"""Tests for analytical Lie group Jacobians.

Verifies each Jacobian against numerical finite differences.
"""

import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.lie_groups import (
    skew,
    so3_right_jacobian,
    so3_left_jacobian,
    so3_left_jacobian_inv,
    so3_right_jacobian_inv,
    se3_right_jacobian,
    se3_left_jacobian_inv,
    se3_adjoint,
)
from tests._reference.ref_intrinsic_jacobians import (
    intrinsic_jacobian_radtan, intrinsic_jacobian_fisheye,
)
from mvis.calibration.factors.pose_interpolation import (
    se3_interpolate_jacobians,
)


# =====================================================================
# Helpers
# =====================================================================

def _random_pose(rng):
    """Generate a random Pose3."""
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + 1e-10
    angle = rng.uniform(0.1, 1.0)
    R = gtsam.Rot3.Expmap(axis * angle)
    t = gtsam.Point3(*(rng.randn(3) * 0.5))
    return gtsam.Pose3(R, t)


def _numerical_jacobian_pose(func, pose, eps=1e-7):
    """Numerical Jacobian of func(Pose3) w.r.t. pose tangent space."""
    f0 = func(pose)
    n = len(f0) if isinstance(f0, np.ndarray) else 1
    J = np.zeros((n, 6))
    for j in range(6):
        delta = np.zeros(6)
        delta[j] = eps
        fp = func(pose.retract(delta))
        if isinstance(f0, np.ndarray):
            J[:, j] = (fp - f0) / eps
        else:
            J[0, j] = (fp - f0) / eps
    return J


# =====================================================================
# SO(3) tests
# =====================================================================

class TestSO3Jacobians:

    def test_skew(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        assert np.allclose(skew(v) @ w, np.cross(v, w))

    def test_right_jacobian_identity(self):
        """J_r(0) = I."""
        J = so3_right_jacobian(np.zeros(3))
        assert np.allclose(J, np.eye(3), atol=1e-10)

    def test_right_jacobian_vs_numerical(self):
        """J_r(φ) matches numerical differentiation of Exp."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            phi = rng.randn(3) * 0.5
            Jr = so3_right_jacobian(phi)

            # Numerical: d/dδ Exp(φ + δ) at δ=0, in body frame
            # Exp(φ+δ) ≈ Exp(φ) · Exp(J_r(φ)·δ)
            # So: Log(Exp(φ)⁻¹ · Exp(φ+δ)) / ε ≈ J_r(φ) · e_j
            eps = 1e-7
            Jr_num = np.zeros((3, 3))
            R0 = gtsam.Rot3.Expmap(phi)
            for j in range(3):
                delta = np.zeros(3)
                delta[j] = eps
                R_pert = gtsam.Rot3.Expmap(phi + delta)
                log_diff = gtsam.Rot3.Logmap(R0.inverse().compose(R_pert))
                Jr_num[:, j] = log_diff / eps

            assert np.allclose(Jr, Jr_num, atol=1e-5), \
                f"Jr mismatch for phi={phi}: max err={np.max(np.abs(Jr - Jr_num))}"

    def test_left_jacobian_inv_vs_numerical(self):
        """J_l⁻¹ · J_l = I (verified numerically)."""
        rng = np.random.RandomState(123)
        for _ in range(10):
            phi = rng.randn(3) * 0.5
            Jl = so3_left_jacobian(phi)
            Jl_inv = so3_left_jacobian_inv(phi)
            assert np.allclose(Jl_inv @ Jl, np.eye(3), atol=1e-6)

    def test_left_right_relation(self):
        """J_l(φ) = J_r(-φ)."""
        rng = np.random.RandomState(99)
        phi = rng.randn(3) * 0.8
        assert np.allclose(so3_left_jacobian(phi),
                           so3_right_jacobian(-phi), atol=1e-10)


# =====================================================================
# SE(3) tests
# =====================================================================

class TestSE3Jacobians:

    def test_adjoint_identity(self):
        """Adj(I) = I."""
        Ad = se3_adjoint(gtsam.Pose3())
        assert np.allclose(Ad, np.eye(6), atol=1e-10)

    def test_adjoint_composition(self):
        """Adj(T1·T2) = Adj(T1)·Adj(T2)."""
        rng = np.random.RandomState(42)
        T1 = _random_pose(rng)
        T2 = _random_pose(rng)
        Ad12 = se3_adjoint(T1.compose(T2))
        Ad1_Ad2 = se3_adjoint(T1) @ se3_adjoint(T2)
        assert np.allclose(Ad12, Ad1_Ad2, atol=1e-8)

    def test_se3_right_jacobian_identity(self):
        """J_r(0) = I₆."""
        J = se3_right_jacobian(np.zeros(6))
        assert np.allclose(J, np.eye(6), atol=1e-10)


# =====================================================================
# SLERP Jacobian tests
# =====================================================================

class TestSLERPJacobians:

    def test_slerp_at_zero(self):
        """λ=0 → T_board = B₀."""
        rng = np.random.RandomState(42)
        B0 = _random_pose(rng)
        B1 = _random_pose(rng)
        T, J_B0, J_B1, v_lam = se3_interpolate_jacobians(B0, B1, 0.0)
        assert np.allclose(T.matrix(), B0.matrix(), atol=1e-10)

    def test_slerp_at_one(self):
        """λ=1 → T_board = B₁."""
        rng = np.random.RandomState(42)
        B0 = _random_pose(rng)
        B1 = _random_pose(rng)
        T, J_B0, J_B1, v_lam = se3_interpolate_jacobians(B0, B1, 1.0)
        assert np.allclose(T.matrix(), B1.matrix(), atol=1e-8)

    def test_J_B0_vs_numerical(self):
        """dT_board/dB₀ matches finite differences."""
        rng = np.random.RandomState(42)
        B0 = _random_pose(rng)
        B1 = _random_pose(rng)
        lam = 0.4

        T0, J_B0, _, _ = se3_interpolate_jacobians(B0, B1, lam)

        eps = 1e-7
        J_num = np.zeros((6, 6))
        for j in range(6):
            delta = np.zeros(6)
            delta[j] = eps
            B0_pert = B0.retract(delta)
            T_pert, _, _, _ = se3_interpolate_jacobians(B0_pert, B1, lam)
            # Compute tangent-space difference
            log_diff = gtsam.Pose3.Logmap(T0.between(T_pert))
            J_num[:, j] = log_diff / eps

        assert np.allclose(J_B0, J_num, atol=1e-4), \
            f"J_B0 max err: {np.max(np.abs(J_B0 - J_num)):.6f}"

    def test_J_B1_vs_numerical(self):
        """dT_board/dB₁ matches finite differences."""
        rng = np.random.RandomState(42)
        B0 = _random_pose(rng)
        B1 = _random_pose(rng)
        lam = 0.6

        T0, _, J_B1, _ = se3_interpolate_jacobians(B0, B1, lam)

        eps = 1e-7
        J_num = np.zeros((6, 6))
        for j in range(6):
            delta = np.zeros(6)
            delta[j] = eps
            B1_pert = B1.retract(delta)
            T_pert, _, _, _ = se3_interpolate_jacobians(B0, B1_pert, lam)
            log_diff = gtsam.Pose3.Logmap(T0.between(T_pert))
            J_num[:, j] = log_diff / eps

        assert np.allclose(J_B1, J_num, atol=1e-4), \
            f"J_B1 max err: {np.max(np.abs(J_B1 - J_num)):.6f}"

    def test_v_lam_vs_numerical(self):
        """dT_board/dλ matches finite differences."""
        rng = np.random.RandomState(42)
        B0 = _random_pose(rng)
        B1 = _random_pose(rng)
        lam = 0.5

        T0, _, _, v_lam = se3_interpolate_jacobians(B0, B1, lam)

        eps = 1e-7
        T_pert, _, _, _ = se3_interpolate_jacobians(B0, B1, lam + eps)
        v_num = gtsam.Pose3.Logmap(T0.between(T_pert)) / eps

        assert np.allclose(v_lam, v_num, atol=1e-4), \
            f"v_lam max err: {np.max(np.abs(v_lam - v_num)):.6f}"

    def test_slerp_extrapolation(self):
        """λ > 1 extrapolates beyond B₁ (no clamp)."""
        rng = np.random.RandomState(42)
        B0 = _random_pose(rng)
        B1 = _random_pose(rng)
        T_extrap, _, _, _ = se3_interpolate_jacobians(B0, B1, 1.5)
        # Should NOT equal B1
        assert not np.allclose(T_extrap.matrix(), B1.matrix(), atol=0.01)

    def test_multiple_lambda_values(self):
        """Jacobians are correct for various λ including negative and >1."""
        rng = np.random.RandomState(42)
        B0 = _random_pose(rng)
        B1 = _random_pose(rng)

        for lam in [-0.2, 0.0, 0.3, 0.5, 0.7, 1.0, 1.3]:
            T0, J_B0, J_B1, v_lam = se3_interpolate_jacobians(B0, B1, lam)

            # Quick numerical check on v_lam
            eps = 1e-7
            T_pert, _, _, _ = se3_interpolate_jacobians(B0, B1, lam + eps)
            v_num = gtsam.Pose3.Logmap(T0.between(T_pert)) / eps
            assert np.allclose(v_lam, v_num, atol=1e-3), \
                f"v_lam mismatch at λ={lam}: max err={np.max(np.abs(v_lam - v_num)):.6f}"


# =====================================================================
# Intrinsic Jacobian tests
# =====================================================================

class TestIntrinsicJacobians:

    def test_radtan_vs_numerical(self):
        """Radtan intrinsic Jacobian matches finite differences."""
        p_cam = np.array([0.3, -0.2, 1.5])
        K = np.array([500.0, 500.0, 320.0, 240.0, -0.1, 0.01, 0.001, -0.001])

        J = intrinsic_jacobian_radtan(p_cam, K)

        # Numerical
        eps = 1e-7

        def _project_radtan(K_vec):
            X, Y, Z = p_cam
            fx, fy, cx, cy = K_vec[0], K_vec[1], K_vec[2], K_vec[3]
            k1, k2, p1, p2 = K_vec[4], K_vec[5], K_vec[6], K_vec[7]
            x_n, y_n = X / Z, Y / Z
            r2 = x_n ** 2 + y_n ** 2
            r4 = r2 ** 2
            rad = 1.0 + k1 * r2 + k2 * r4
            x_d = rad * x_n + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n ** 2)
            y_d = rad * y_n + p1 * (r2 + 2 * y_n ** 2) + 2 * p2 * x_n * y_n
            return np.array([fx * x_d + cx, fy * y_d + cy])

        J_num = np.zeros((2, 8))
        p0 = _project_radtan(K)
        for j in range(8):
            delta = np.zeros(8)
            delta[j] = eps
            J_num[:, j] = (_project_radtan(K + delta) - p0) / eps

        assert np.allclose(J, J_num, atol=1e-4), \
            f"Radtan J max err: {np.max(np.abs(J - J_num)):.6f}"

    def test_fisheye_vs_numerical(self):
        """Fisheye intrinsic Jacobian matches finite differences."""
        p_cam = np.array([0.5, -0.3, 1.0])
        K = np.array([300.0, 300.0, 424.0, 400.0, -0.01, 0.05, -0.03, 0.001])

        J = intrinsic_jacobian_fisheye(p_cam, K)

        eps = 1e-7

        def _project_fisheye(K_vec):
            X, Y, Z = p_cam
            fx, fy, cx, cy = K_vec[0], K_vec[1], K_vec[2], K_vec[3]
            k1, k2, k3, k4 = K_vec[4], K_vec[5], K_vec[6], K_vec[7]
            rho = np.sqrt(X ** 2 + Y ** 2)
            theta = np.arctan2(rho, Z)
            r_d = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9
            if rho > 1e-10:
                s = r_d / rho
            else:
                s = 1.0
            return np.array([fx * s * X + cx, fy * s * Y + cy])

        J_num = np.zeros((2, 8))
        p0 = _project_fisheye(K)
        for j in range(8):
            delta = np.zeros(8)
            delta[j] = eps
            J_num[:, j] = (_project_fisheye(K + delta) - p0) / eps

        assert np.allclose(J, J_num, atol=1e-4), \
            f"Fisheye J max err: {np.max(np.abs(J - J_num)):.6f}"

    def test_radtan_on_axis(self):
        """Point on optical axis (x_n=y_n=0)."""
        p_cam = np.array([0.0, 0.0, 1.0])
        K = np.array([500.0, 500.0, 320.0, 240.0, -0.1, 0.01, 0.0, 0.0])
        J = intrinsic_jacobian_radtan(p_cam, K)
        # u = fx*0 + cx = cx, v = fy*0 + cy = cy
        # du/dfx = x_d = 0, du/dcx = 1
        assert J[0, 0] == 0.0  # du/dfx
        assert J[0, 2] == 1.0  # du/dcx
        assert J[1, 1] == 0.0  # dv/dfy
        assert J[1, 3] == 1.0  # dv/dcy
