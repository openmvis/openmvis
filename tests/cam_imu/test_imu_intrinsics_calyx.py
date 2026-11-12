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
"""K0 tests for the Calyx IMU-intrinsic model (calibration/factors/imu_intrinsics).

Covers the new full-3x3 gyro correction, its 3x9 Jacobian, the RQ/QR
decomposition, and the model-aware 24-dim manifold. Kalibr/RPNG regression lives
in the existing test_imu_manifold.py / test_imu_factor.py (the no-op gate).
"""
import numpy as np
import pytest

from mvis.imu import (
    KALIBR, RPNG, CALYX, n_dw_for,
    dw_matrix, da_matrix, dw_jacobian, da_jacobian, decompose_dw,
    ImuModel,
)

_RNG = np.random.default_rng(0)


def test_n_dw_per_model():
    assert n_dw_for(KALIBR) == 6
    assert n_dw_for(RPNG) == 6
    assert n_dw_for(CALYX) == 9


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_default_intrinsics_are_identity(model):
    """Regression: the default ImuModel must build IDENTITY Dw/Da for EVERY
    model. The original Kalibr bug used the upper-tri 6-vec for lower-tri Kalibr,
    yielding a rank-deficient seed [[1,0,0],[0,0,0],[1,0,1]]."""
    ii = ImuModel(model=model)
    np.testing.assert_allclose(ii.Dw(), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ii.Da(), np.eye(3), atol=1e-12)


def test_calyx_dw_is_full_rowmajor():
    x = np.arange(1.0, 10.0)
    np.testing.assert_array_equal(dw_matrix(x, CALYX), x.reshape(3, 3))


def test_calyx_default_dw_is_identity():
    ii = ImuModel(model=CALYX)
    assert ii.x_dw.shape == (9,)
    np.testing.assert_array_equal(ii.Dw(), np.eye(3))


def test_calyx_da_is_upper_triangular():
    """Calyx reuses the upper-triangular Da (same as RPNG)."""
    x = np.array([1.1, 0.02, 0.9, -0.01, 0.03, 1.05])
    np.testing.assert_array_equal(da_matrix(x, CALYX), da_matrix(x, RPNG))
    Da = da_matrix(x, CALYX)
    assert Da[1, 0] == 0.0 and Da[2, 0] == 0.0 and Da[2, 1] == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_calyx_dw_jacobian_matches_fd(seed):
    rng = np.random.default_rng(seed)
    x_dw = np.eye(3).flatten() + 0.05 * rng.standard_normal(9)
    w = rng.standard_normal(3)
    H = dw_jacobian(w, CALYX)
    assert H.shape == (3, 9)
    eps = 1e-6
    fd = np.zeros((3, 9))
    for j in range(9):
        dx = np.zeros(9); dx[j] = eps
        fd[:, j] = (dw_matrix(x_dw + dx, CALYX) @ w
                    - dw_matrix(x_dw - dx, CALYX) @ w) / (2 * eps)
    np.testing.assert_allclose(H, fd, atol=1e-7)


@pytest.mark.parametrize("seed", range(8))
def test_decompose_dw_roundtrip(seed):
    rng = np.random.default_rng(seed)
    # full near-identity gyro correction with det > 0 (physical)
    Dw = np.eye(3) + 0.08 * rng.standard_normal((3, 3))
    R, U = decompose_dw(Dw)
    # exact reconstruction
    np.testing.assert_allclose(R @ U, Dw, atol=1e-12)
    # U upper-triangular, positive diagonal
    np.testing.assert_allclose(np.tril(U, -1), 0.0, atol=1e-12)
    assert np.all(np.diag(U) > 0)
    # R a proper rotation
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_imuintrinsics_decompose_method():
    Dw = np.array([[1.02, 0.03, -0.01],
                   [0.0,  0.99, 0.02],
                   [0.01, 0.0,  1.01]])
    ii = ImuModel(x_dw=Dw.flatten(), model=CALYX)
    R, U = ii.decompose()
    np.testing.assert_allclose(R @ U, Dw, atol=1e-12)


def test_calyx_manifold_dim_and_layout():
    ii = ImuModel(model=CALYX)
    assert ii.DIM == 24
    assert ii.n_dw == 9 and ii.has_R is False
    assert (ii.da_off, ii.tg_off, ii.r_off) == (9, 15, 24)


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_manifold_axioms(model):
    """retract / local_coordinates are inverse, for all three models."""
    base = ImuModel(model=model)
    xi = 0.01 * _RNG.standard_normal(24)
    moved = base.retract(xi)
    back = base.local_coordinates(moved)
    np.testing.assert_allclose(back, xi, atol=1e-9)
    # round-trip the other way
    np.testing.assert_allclose(
        base.local_coordinates(base), np.zeros(24), atol=1e-12)


def test_calyx_retract_has_no_rotation_dof():
    """The last 3 tangent dims feed x_tg (not a rotation) under Calyx."""
    base = ImuModel(model=CALYX)
    xi = np.zeros(24); xi[23] = 0.5          # last dim
    moved = base.retract(xi)
    np.testing.assert_array_equal(moved.R, np.eye(3))      # R untouched
    assert moved.x_tg[8] == pytest.approx(0.5)             # it went to Tg
