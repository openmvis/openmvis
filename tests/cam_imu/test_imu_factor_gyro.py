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
"""ImuFactorACI_Multi_In_SO3 (gyro-only): preintegration + factor validation."""

import numpy as np
import gtsam
import pytest

from mvis.imu import ImuModel, KALIBR, RPNG
from mvis.calibration.factors.aci import ACIGyroPreintegrator, whitener_rot_corrected
from mvis.calibration.factors.lie_groups import so3_left_jacobian
from tests.helpers.imu_factor_cases import (_seq, _intr, _build, _consistent)
from mvis.calibration.factors.lie_groups import log_so3 as _log


# --------------------- preintegration FD ---------------------
@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_gyro_preint_jacobians_fd(model):
    t, w = _seq(seed=1)
    bg = np.array([0.01, -0.02, 0.005])

    def run(bg_, intr_):
        p = ACIGyroPreintegrator(0.01, 0.001, model=model, avg_meas=True)
        p.reset(bg_); p.set_intrinsics(intr_); p.feed_imu_interval(t, w)
        return p

    base = run(bg, _intr(model))
    eps = 1e-6
    # bias Jacobian J_R_bg
    for j in range(3):
        d = np.zeros(3); d[j] = eps
        pp, pm = run(bg + d, _intr(model)), run(bg - d, _intr(model))
        fd = _log(base.D_R.T @ pp.D_R) - _log(base.D_R.T @ pm.D_R)
        assert np.allclose(fd / (2 * eps), base.J_R_bg[:, j], rtol=3e-2, atol=2e-4)
    # intrinsic (Dw) Jacobian J_R_in
    x_dw = _intr(model).x_dw
    for j in range(6):
        d = np.zeros(6); d[j] = eps
        ip = ImuModel(x_dw=x_dw + d, x_da=np.ones(6), x_tg=np.zeros(9), R=np.eye(3), model=model)
        im = ImuModel(x_dw=x_dw - d, x_da=np.ones(6), x_tg=np.zeros(9), R=np.eye(3), model=model)
        pp, pm = run(bg, ip), run(bg, im)
        fd = _log(base.D_R.T @ pp.D_R) - _log(base.D_R.T @ pm.D_R)
        assert np.allclose(fd / (2 * eps), base.J_R_in[:, j], rtol=3e-2, atol=2e-4)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_gyro_factor_residual_and_jacobians(model):
    f = _build(model)
    pose0, pose1, bg0, bg1, xi_calib, d_th = _consistent(f, model)
    e0, H = f.error_and_jacobians(pose0, pose1, bg0, bg1, xi_calib)
    H_p0, H_p1, H_bg0, H_bg1, H_calib = H
    assert e0.shape == (6,) and np.max(np.abs(e0)) < 1e-6

    W = whitener_rot_corrected(f.P_meas, d_th)
    Jl = so3_left_jacobian(d_th)
    # NON-CIRCULAR: analytic bias + Dw blocks (valid at residual≈0)
    H_bg0_an = np.zeros((6, 3)); H_bg0_an[0:3, :] = -Jl @ f.J_R_bg; H_bg0_an[3:6, :] = -np.eye(3)
    H_bg1_an = np.zeros((6, 3)); H_bg1_an[3:6, :] = np.eye(3)
    H_dw_an = np.zeros((6, 6)); H_dw_an[0:3, :] = -Jl @ f.J_R_in
    assert np.allclose(H_bg0, W @ H_bg0_an, atol=1e-4)
    assert np.allclose(H_bg1, W @ H_bg1_an, atol=1e-4)
    assert np.allclose(H_calib[:, 7:13], W @ H_dw_an, atol=1e-4)

    # Taylor check of the full Jacobian
    rng = np.random.RandomState(7)
    dp0, dp1 = rng.randn(6) * 1e-6, rng.randn(6) * 1e-6
    dbg0, dbg1, dcal = rng.randn(3) * 1e-6, rng.randn(3) * 1e-6, rng.randn(31) * 1e-6
    pred = e0 + H_p0 @ dp0 + H_p1 @ dp1 + H_bg0 @ dbg0 + H_bg1 @ dbg1 + H_calib @ dcal
    actual = f.whitened_error(pose0.retract(dp0), pose1.retract(dp1), bg0 + dbg0, bg1 + dbg1, xi_calib + dcal)
    assert np.allclose(pred, actual, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_gyro_analytic_matches_numerical(model):
    f = _build(model)
    for seed in (4, 12, 20, 28):
        pose0, pose1, bg0, bg1, xi_calib, _ = _consistent(f, model, seed=seed)
        _, Ha = f.error_and_jacobians(pose0, pose1, bg0, bg1, xi_calib)
        _, Hn = f._numerical_jacobians(pose0, pose1, bg0, bg1, xi_calib)
        for i, (a, n) in enumerate(zip(Ha, Hn)):
            den = max(1.0, np.max(np.abs(a)), np.max(np.abs(n)))
            assert np.max(np.abs(a - n)) / den < 1e-4, f"seed {seed} block {i}"


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_gyro_factor_in_graph(model):
    f = _build(model)
    pose0, pose1, bg0, bg1, xi_calib, _ = _consistent(f, model)
    X = gtsam.symbol_shorthand.X
    keys = [X(0), X(1), gtsam.symbol('b', 0), gtsam.symbol('b', 1), gtsam.symbol('Y', 0)]
    factor = f.make_factor(*keys)
    vals = gtsam.Values()
    vals.insert(keys[0], pose0); vals.insert(keys[1], pose1)
    vals.insert(keys[2], bg0); vals.insert(keys[3], bg1)
    vals.insert(keys[4], xi_calib)
    assert factor.unwhitenedError(vals).shape == (6,)
    assert factor.linearize(vals) is not None
