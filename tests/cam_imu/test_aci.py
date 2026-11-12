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
"""ACI preintegration: internal-consistency tests.

Validates the analytic bias + intrinsic Jacobians against finite-difference
re-integration. This catches sign/indexing/structural errors (which produce
O(1) discrepancies).

NOTE on tolerance: the ACI analytic bias/intrinsic Jacobians are *first-order*
(O(dt)) — exact in the continuous limit. Indirect couplings (e.g. Tg → gyro →
rotation → velocity) therefore carry an O(dt) discrepancy vs a true finite
difference (verified: error scales linearly with dt). The C++ ACI_In uses the
identical analytic formula, so it has the identical O(dt) error — exact
machine-precision agreement is validated separately against the C++ oracle.
Here we use a finer dt and an O(dt)-aware tolerance.
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import ImuModel, KALIBR, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from tests.helpers.imu_signals import _make_sequence
from mvis.calibration.factors.lie_groups import log_so3 as _log_so3

# O(dt)-aware tolerance: first-order Jacobians + finite dt. Real structural bugs
# produce O(1) errors and still fail loudly.
_RTOL, _ATOL = 3e-2, 2e-4


def _run(t, w, a, bg, ba, intr, model):
    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
    pre.reset(bg, ba)
    pre.set_intrinsics(intr)
    pre.feed_imu_interval(t, w, a)
    return pre


def _central_diff(base, run_plus, run_minus, eps):
    """Central-difference (Δp, Δv, Δθ) of preint outputs for one param direction.

    run_plus/run_minus: callables returning an integrator at +eps / -eps.
    Rotation derivative is taken in the right-tangent at the base D_R.
    """
    p, m = run_plus(), run_minus()
    d_p = (p.D_p - m.D_p) / (2 * eps)
    d_v = (p.D_v - m.D_v) / (2 * eps)
    th_p = _log_so3(base.D_R.T @ p.D_R)
    th_m = _log_so3(base.D_R.T @ m.D_R)
    d_th = (th_p - th_m) / (2 * eps)
    return d_p, d_v, d_th


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_static_zero_input(model):
    """No rotation, zero gravity in integrator: D stays identity/zero."""
    t = np.arange(6) * 0.01
    w = np.zeros((6, 3))
    a = np.zeros((6, 3))
    pre = _run(t, w, a, np.zeros(3), np.zeros(3), ImuModel(model=model), model)
    assert np.allclose(pre.D_R, np.eye(3), atol=1e-12)
    assert np.allclose(pre.D_v, 0, atol=1e-12)
    assert np.allclose(pre.D_p, 0, atol=1e-12)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_bias_jacobians_fd(model):
    """Analytic J_*_bg / J_*_ba match finite-difference re-integration."""
    t, w, a = _make_sequence(seed=1)
    bg = np.array([0.01, -0.02, 0.005])
    ba = np.array([0.05, -0.03, 0.02])
    intr = ImuModel(model=model)
    base = _run(t, w, a, bg, ba, intr, model)

    eps = 1e-6
    for j in range(3):
        d = np.zeros(3); d[j] = eps
        # gyro bias
        d_p, d_v, d_th = _central_diff(
            base, lambda: _run(t, w, a, bg + d, ba, intr, model),
            lambda: _run(t, w, a, bg - d, ba, intr, model), eps)
        assert np.allclose(d_p, base.J_p_bg[:, j], rtol=_RTOL, atol=_ATOL)
        assert np.allclose(d_v, base.J_v_bg[:, j], rtol=_RTOL, atol=_ATOL)
        assert np.allclose(d_th, base.J_R_bg[:, j], rtol=_RTOL, atol=_ATOL)
        # accel bias
        d_p, d_v, d_th = _central_diff(
            base, lambda: _run(t, w, a, bg, ba + d, intr, model),
            lambda: _run(t, w, a, bg, ba - d, intr, model), eps)
        assert np.allclose(d_p, base.J_p_ba[:, j], rtol=_RTOL, atol=_ATOL)
        assert np.allclose(d_v, base.J_v_ba[:, j], rtol=_RTOL, atol=_ATOL)
        assert np.allclose(d_th, base.J_R_ba[:, j], rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_intrinsic_jacobians_fd(model):
    """Analytic J_*_in (24 cols) match FD: x_Dw,x_Da,x_Tg additive; R right-retract."""
    t, w, a = _make_sequence(seed=2)
    bg = np.array([0.01, -0.02, 0.005])
    ba = np.array([0.05, -0.03, 0.02])
    # non-trivial intrinsics
    x_dw = np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02])
    x_da = np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99])
    x_tg = np.array([0.01, -0.02, 0.0, 0.005, 0.0, -0.01, 0.0, 0.008, 0.0])
    R0 = gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix()
    intr = ImuModel(x_dw=x_dw, x_da=x_da, x_tg=x_tg, R=R0, model=model)
    base = _run(t, w, a, bg, ba, intr, model)

    eps = 1e-6

    def intr_with(name, vec):
        kwargs = dict(x_dw=x_dw, x_da=x_da, x_tg=x_tg, R=R0, model=model)
        kwargs[name] = vec
        return _run(t, w, a, bg, ba, ImuModel(**kwargs), model)

    # columns 0..20: x_Dw(6), x_Da(6), x_Tg(9) additive
    for name, base_vec, col0 in [("x_dw", x_dw, 0), ("x_da", x_da, 6), ("x_tg", x_tg, 12)]:
        for j in range(len(base_vec)):
            d = np.zeros(len(base_vec)); d[j] = eps
            c = col0 + j
            d_p, d_v, d_th = _central_diff(
                base, lambda: intr_with(name, base_vec + d),
                lambda: intr_with(name, base_vec - d), eps)
            assert np.allclose(d_p, base.J_p_in[:, c], rtol=_RTOL, atol=_ATOL), f"{name}[{j}] p"
            assert np.allclose(d_v, base.J_v_in[:, c], rtol=_RTOL, atol=_ATOL), f"{name}[{j}] v"
            assert np.allclose(d_th, base.J_R_in[:, c], rtol=_RTOL, atol=_ATOL), f"{name}[{j}] R"

    # columns 21..23: rotation R via right retract R -> R*Exp(delta)
    def intr_R(delta):
        R_pert = R0 @ gtsam.Rot3.Expmap(delta).matrix()
        return _run(t, w, a, bg, ba,
                    ImuModel(x_dw=x_dw, x_da=x_da, x_tg=x_tg, R=R_pert, model=model), model)

    for j in range(3):
        d = np.zeros(3); d[j] = eps
        c = 21 + j
        d_p, d_v, d_th = _central_diff(base, lambda: intr_R(d), lambda: intr_R(-d), eps)
        assert np.allclose(d_p, base.J_p_in[:, c], rtol=_RTOL, atol=_ATOL), f"R[{j}] p"
        assert np.allclose(d_v, base.J_v_in[:, c], rtol=_RTOL, atol=_ATOL), f"R[{j}] v"
        assert np.allclose(d_th, base.J_R_in[:, c], rtol=_RTOL, atol=_ATOL), f"R[{j}] R"
