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
"""Tests for the unified Lie-group library (calibration/factors/lie_groups.py).

Covers the SE(3) helpers promoted out of bspline_se3.py (W1) and the
log_so3 / cv2.Rodrigues parity that justified deduping motion_analysis (W2).
"""
import numpy as np
import pytest

from mvis.calibration.factors.lie_groups import (
    skew, vee_so3,
    exp_so3, log_so3,
    hat_se3, exp_se3, log_se3, inv_se3,
)

# Tangent 6-vectors in gtsam ordering [omega(3), rho(3)].
_XIS = [
    np.array([0.0, 0.0, 0.0, 1.0, -2.0, 3.0]),          # pure translation
    np.array([0.3, -0.1, 0.2, 0.0, 0.0, 0.0]),          # pure rotation
    np.array([0.4, 0.2, -0.3, 0.5, -0.7, 1.1]),         # mixed
    np.array([1.2, -0.9, 0.6, -2.0, 0.3, 0.8]),         # larger rotation
    np.array([1e-9, -2e-9, 5e-10, 0.1, 0.2, -0.3]),     # near-zero rotation
]


@pytest.mark.parametrize("xi", _XIS)
def test_se3_exp_log_roundtrip(xi):
    """log_se3(exp_se3(xi)) recovers xi."""
    T = exp_se3(xi)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
    xi_rt = log_se3(T)
    np.testing.assert_allclose(xi_rt, xi, atol=1e-9)


@pytest.mark.parametrize("xi", _XIS)
def test_inv_se3_is_matrix_inverse(xi):
    """inv_se3 matches the true matrix inverse and is a valid SE(3) inverse."""
    T = exp_se3(xi)
    Tinv = inv_se3(T)
    np.testing.assert_allclose(Tinv, np.linalg.inv(T), atol=1e-12)
    np.testing.assert_allclose(T @ Tinv, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("xi", _XIS)
def test_hat_se3_structure(xi):
    """hat_se3 places skew(omega) in the upper-left and rho in the last column."""
    M = hat_se3(xi)
    assert M.shape == (4, 4)
    np.testing.assert_allclose(M[0:3, 0:3], skew(xi[0:3]), atol=1e-15)
    np.testing.assert_allclose(M[0:3, 3], xi[3:6], atol=1e-15)
    np.testing.assert_allclose(M[3], 0.0, atol=1e-15)


def test_vee_inverts_skew():
    """vee_so3 is the inverse of skew."""
    for v in ([1.0, 2.0, 3.0], [-0.5, 0.0, 4.2], [0.0, 0.0, 0.0]):
        v = np.array(v)
        np.testing.assert_allclose(vee_so3(skew(v)), v, atol=1e-15)


_PHIS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.3, -0.1, 0.2]),
    np.array([1.5, -0.9, 0.6]),
    np.array([np.pi - 1e-3, 0.0, 0.0]),
    np.array([1e-10, -2e-10, 0.0]),
]


@pytest.mark.parametrize("phi", _PHIS)
def test_log_so3_matches_cv2_rodrigues(phi):
    """log_so3 (gtsam) agrees with cv2.Rodrigues — justifies the W2 dedup."""
    cv2 = pytest.importorskip("cv2")
    R = exp_so3(phi)
    rvec_gtsam = log_so3(R)
    rvec_cv2, _ = cv2.Rodrigues(R)
    np.testing.assert_allclose(rvec_gtsam, rvec_cv2.flatten(), atol=1e-9)
