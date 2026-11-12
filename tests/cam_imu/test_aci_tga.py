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
"""Gyro<->accel time offset t_ga inside the ACI factor (mvis/aci-gyro-accel-td).

t_ga model: the accel applying at gyro time t is a_m(t + t_ga). The preintegrator
accumulates J_R/p/v_tga (one extra intrinsic-pattern column, da/dt_ga = a-dot);
the factor corrects the deltas first-order through an optional 6th scalar key.
"""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.lie_groups import log_so3
from mvis.calibration.factors.numdiff import numerical_jacobian
from mvis.imu import ImuModel, RPNG, create_imu_model
from tests.helpers.imu_signals import (_w, _a, _intr_with_tg)

GRAV = np.array([0.0, 0.0, -9.81])


def _preintegrate(t0, t1, hz, tga, intr, avg_meas=True):
    """Preintegrate the analytic streams with the ACCEL evaluated at t + tga."""
    pre = ACIPreintegrator(1e-3, 1e-5, 1e-2, 1e-4, model=RPNG, avg_meas=avg_meas)
    pre.reset(np.array([0.004, -0.003, 0.002]), np.array([0.03, -0.02, 0.01]))
    pre.set_intrinsics(intr)
    ts = np.arange(t0, t1 + 1e-12, 1.0 / hz)
    pre.feed_imu_interval(ts, [_w(t) for t in ts], [_a(t + tga) for t in ts])
    return pre


def _fd_rel_errs(hz, avg_meas):
    """Per-block relative error of J_*_tga vs central-difference preintegration
    of a shifted accel stream."""
    intr = _intr_with_tg()
    d = 5e-6
    p0 = _preintegrate(0.0, 1.0, hz, 0.0, intr, avg_meas)
    pp = _preintegrate(0.0, 1.0, hz, +d, intr, avg_meas)
    pm = _preintegrate(0.0, 1.0, hz, -d, intr, avg_meas)
    fd_R = log_so3(pm.D_R.T @ pp.D_R) / (2 * d)      # D_R(tga) ~ D_R(0)Exp(J tga)
    fd_p = (pp.D_p - pm.D_p) / (2 * d)
    fd_v = (pp.D_v - pm.D_v) / (2 * d)
    errs = [np.linalg.norm(fd - J) / max(np.linalg.norm(fd), 1e-9)
            for fd, J in ((fd_R, p0.J_R_tga), (fd_p, p0.J_p_tga), (fd_v, p0.J_v_tga))]
    return errs, p0


@pytest.mark.parametrize("avg_meas,rtol", [(True, 1.5e-2), (False, 8e-2)])
def test_preintegrator_J_tga_matches_fd(avg_meas, rtol):
    """J_R/p/v_tga vs FD of preintegrating a shifted accel stream. Agreement is
    bounded by the integrator's own O(dt) discretization (the in-step
    perturbation a-dot(t)*d is treated as constant, like the measurements
    themselves): ~1% at 200 Hz, halving with rate — asserted below."""
    errs200, p0 = _fd_rel_errs(200, avg_meas)
    assert max(errs200) < rtol, errs200
    # the deltas must actually be t_ga-sensitive for the test to mean anything
    assert np.linalg.norm(p0.J_v_tga) > 1e-3
    assert np.linalg.norm(p0.J_R_tga) > 1e-6        # via the Tg coupling
    # first-order convergence in dt: error at 800 Hz well under half of 200 Hz
    errs800, _ = _fd_rel_errs(800, avg_meas)
    assert max(errs800) < 0.6 * max(errs200), (errs200, errs800)


def _consistent_state_tga(f, t_ga, seed=3):
    """Residual≈0 state including the t_ga correction (mirrors the
    test_imu_factor._consistent_state pattern, + J_*_tga terms) — the analytic
    first-order Jacobians are exact here."""
    rng = np.random.RandomState(seed)
    R0 = gtsam.Rot3.Expmap(rng.randn(3) * 0.1).matrix()
    p0 = rng.randn(3) * 0.1
    v0 = rng.randn(3) * 0.1
    bg0 = f.bg_init + rng.randn(3) * 0.002
    ba0 = f.ba_init + rng.randn(3) * 0.01
    xi_in = rng.randn(24) * 0.01

    d_bg, d_ba, d_in = bg0 - f.bg_init, ba0 - f.ba_init, xi_in
    d_th = (f.J_R_bg @ d_bg + f.J_R_ba @ d_ba + f.J_R_in @ d_in
            + f.J_R_tga * t_ga)
    R_bcorr = gtsam.Rot3.Expmap(d_th).matrix()
    Dt, g = f.Dt, f.g_inG
    R_I1toG = R0 @ f.D_R @ R_bcorr
    v1 = v0 + g * Dt + R0 @ (f.D_v + f.J_v_bg @ d_bg + f.J_v_ba @ d_ba
                             + f.J_v_in @ d_in + f.J_v_tga * t_ga)
    p1 = p0 + v0 * Dt + 0.5 * g * Dt ** 2 + R0 @ (
        f.D_p + f.J_p_bg @ d_bg + f.J_p_ba @ d_ba
        + f.J_p_in @ d_in + f.J_p_tga * t_ga)
    pose0 = gtsam.Pose3(gtsam.Rot3(R0), gtsam.Point3(*p0))
    pose1 = gtsam.Pose3(gtsam.Rot3(R_I1toG), gtsam.Point3(*p1))
    vb0 = np.concatenate([v0, bg0, ba0])
    vb1 = np.concatenate([v1, bg0, ba0])
    return pose0, pose1, vb0, vb1, xi_in


def test_factor_residual_zero_at_tga_consistent_state():
    pre = _preintegrate(0.0, 0.5, 200, 0.0, _intr_with_tg())
    fac = ImuFactorACI_In.from_preintegrator(pre, GRAV)
    t_ga = 1.5e-3
    s = _consistent_state_tga(fac, t_ga)
    assert np.max(np.abs(fac.whitened_error(*s, t_ga))) < 1e-6


def test_factor_jacobians_6key_match_numerical():
    pre = _preintegrate(0.0, 0.5, 200, 0.0, _intr_with_tg())
    fac = ImuFactorACI_In.from_preintegrator(pre, GRAV)
    t_ga = np.array([1.5e-3])
    pose0, pose1, vb0, vb1, xi = _consistent_state_tga(fac, float(t_ga[0]))

    e, H = fac.error_and_jacobians(pose0, pose1, vb0, vb1, xi, float(t_ga[0]))
    _e0, Hn = numerical_jacobian(
        lambda P0, P1, V0, V1, XI, TG:
            fac.whitened_error(P0, P1, V0, V1, XI, float(TG[0])),
        [pose0, pose1, vb0, vb1, xi, t_ga])
    assert len(H) == 6
    # the t_ga column must be non-trivial for the check to mean anything
    assert np.linalg.norm(Hn[5]) > 1.0
    for i, (Ha, Hb) in enumerate(zip(H, Hn)):
        assert np.allclose(Hb, Ha, rtol=1e-4, atol=1e-5), f"block {i}"


def test_factor_5key_path_unchanged():
    """Without a t_ga key the factor stays the 5-key version with 5 Jacobians."""
    pre = _preintegrate(0.0, 0.5, 200, 0.0, create_imu_model(RPNG))
    fac = ImuFactorACI_In.from_preintegrator(pre, GRAV)
    e, H = fac.error_and_jacobians(
        gtsam.Pose3(), gtsam.Pose3(), np.zeros(9), np.zeros(9), np.zeros(24))
    assert len(H) == 5 and e.shape == (15,)
    f5 = fac.make_factor(gtsam.symbol('x', 0), gtsam.symbol('x', 1),
                         gtsam.symbol('v', 0), gtsam.symbol('v', 1),
                         gtsam.symbol('I', 0))
    assert f5.size() == 5


def test_closed_loop_recovers_injected_tga():
    """States generated from the TRUE (shifted-accel) stream; factors fed the
    UNSHIFTED stream with t_ga free -> the solve must recover t_ga (+ sign)."""
    intr = create_imu_model(RPNG)               # identity intrinsics, biases zero
    tga_true = 2.0e-3
    K, dt_s, hz = 10, 0.2, 400.0

    X = gtsam.symbol_shorthand.X
    V = lambda k: gtsam.symbol('v', k)
    I0 = gtsam.symbol('I', 0)
    O0 = gtsam.symbol('o', 0)

    # ground-truth states by composing TRUE preintegrated deltas
    poses = [gtsam.Pose3()]
    vels = [np.zeros(3)]
    facs = []
    for k in range(K):
        t0, t1 = k * dt_s, (k + 1) * dt_s
        # true motion: accel stream shifted by tga_true
        pre_true = _rebuild_zero_bias(t0, t1, hz, tga_true, intr)
        R0 = poses[-1].rotation().matrix()
        p0 = np.asarray(poses[-1].translation(), float)
        v0 = vels[-1]
        Dt = pre_true.Dt
        p1 = p0 + v0 * Dt + 0.5 * GRAV * Dt ** 2 + R0 @ pre_true.D_p
        v1 = v0 + GRAV * Dt + R0 @ pre_true.D_v
        R1 = R0 @ pre_true.D_R
        poses.append(gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(*p1)))
        vels.append(v1)
        # measurement-side preintegration: UNSHIFTED accel (t_ga = 0)
        pre_meas = _rebuild_zero_bias(t0, t1, hz, 0.0, intr)
        facs.append(ImuFactorACI_In.from_preintegrator(pre_meas, GRAV))

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    tight = gtsam.noiseModel.Isotropic.Sigma(6, 1e-9)
    tight9 = gtsam.noiseModel.Isotropic.Sigma(9, 1e-9)
    for k in range(K + 1):
        values.insert(X(k), poses[k])
        values.insert(V(k), np.concatenate([vels[k], np.zeros(6)]))
        graph.add(gtsam.PriorFactorPose3(X(k), poses[k], tight))
        graph.add(gtsam.PriorFactorVector(V(k), np.concatenate([vels[k], np.zeros(6)]), tight9))
    values.insert(I0, np.zeros(24))
    values.insert(O0, np.zeros(1))
    graph.add(gtsam.PriorFactorVector(I0, np.zeros(24),
                                      gtsam.noiseModel.Isotropic.Sigma(24, 1e-9)))
    graph.add(gtsam.PriorFactorVector(O0, np.zeros(1),
                                      gtsam.noiseModel.Isotropic.Sigma(1, 0.05)))
    for k, fac in enumerate(facs):
        graph.add(fac.make_factor(X(k), X(k + 1), V(k), V(k + 1), I0, k_tga=O0))

    res = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    tga_est = float(res.atVector(O0)[0])
    assert abs(tga_est - tga_true) < 0.15e-3, tga_est   # within 0.15 ms of 2 ms


def _rebuild_zero_bias(t0, t1, hz, tga, intr):
    """Preintegrator over [t0,t1] with zero init biases (closed-loop helper)."""
    pre = ACIPreintegrator(1e-3, 1e-5, 1e-2, 1e-4, model=RPNG, avg_meas=True)
    pre.reset(np.zeros(3), np.zeros(3))
    pre.set_intrinsics(intr)
    ts = np.arange(t0, t1 + 1e-12, 1.0 / hz)
    pre.feed_imu_interval(ts, [_w(t) for t in ts], [_a(t + tga) for t in ts])
    return pre


# ------------------------------------------------------------------ aux t_ga
def _aux_closed_loop_tga(tga_true):
    """Aux analog of the closed-loop test: base states generated from the TRUE
    (shifted-accel) stream; the AUX Multi_In_V factor (rigid identity mount,
    td=0, calib pinned) is fed the UNSHIFTED stream with its own t_ga free."""
    from mvis.calibration.factors.imu_factor_multi import ImuFactorACI_Multi_In_V
    intr = create_imu_model(RPNG)
    K, dt_s, hz = 10, 0.2, 400.0

    X = gtsam.symbol_shorthand.X
    V = lambda k: gtsam.symbol('v', k)
    S = lambda k: gtsam.symbol('s', k)
    A0 = gtsam.symbol('A', 0)
    O1 = gtsam.symbol('o', 1)

    poses = [gtsam.Pose3()]
    vels = [np.zeros(3)]
    facs = []
    for k in range(K):
        t0, t1 = k * dt_s, (k + 1) * dt_s
        pre_true = _rebuild_zero_bias(t0, t1, hz, tga_true, intr)
        R0 = poses[-1].rotation().matrix()
        p0 = np.asarray(poses[-1].translation(), float)
        v0 = vels[-1]
        Dt = pre_true.Dt
        p1 = p0 + v0 * Dt + 0.5 * GRAV * Dt ** 2 + R0 @ pre_true.D_p
        v1 = v0 + GRAV * Dt + R0 @ pre_true.D_v
        R1 = R0 @ pre_true.D_R
        poses.append(gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(*p1)))
        vels.append(v1)
        pre_meas = _rebuild_zero_bias(t0, t1, hz, 0.0, intr)
        facs.append(ImuFactorACI_Multi_In_V.from_preintegrator(
            pre_meas, GRAV, gtsam.Pose3(), 0.0))

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    tight = gtsam.noiseModel.Isotropic.Sigma(6, 1e-9)
    tight9 = gtsam.noiseModel.Isotropic.Sigma(9, 1e-9)
    for k in range(K + 1):
        vb = np.concatenate([vels[k], np.zeros(6)])       # base [v, bg, ba]
        vbs = np.concatenate([np.zeros(6), vels[k]])      # aux [bg, ba, v]
        values.insert(X(k), poses[k])
        values.insert(V(k), vb)
        values.insert(S(k), vbs)
        graph.add(gtsam.PriorFactorPose3(X(k), poses[k], tight))
        graph.add(gtsam.PriorFactorVector(V(k), vb, tight9))
        graph.add(gtsam.PriorFactorVector(S(k), vbs, tight9))
    values.insert(A0, np.zeros(31))
    values.insert(O1, np.zeros(1))
    graph.add(gtsam.PriorFactorVector(A0, np.zeros(31),
                                      gtsam.noiseModel.Isotropic.Sigma(31, 1e-9)))
    graph.add(gtsam.PriorFactorVector(O1, np.zeros(1),
                                      gtsam.noiseModel.Isotropic.Sigma(1, 0.05)))
    for k, fac in enumerate(facs):
        graph.add(fac.make_factor(X(k), X(k + 1), V(k), V(k + 1),
                                  S(k), S(k + 1), A0, k_tga=O1))
    res = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    return float(res.atVector(O1)[0])


def test_aux_closed_loop_recovers_injected_tga():
    tga_est = _aux_closed_loop_tga(2.0e-3)
    assert abs(tga_est - 2.0e-3) < 0.15e-3, tga_est


def test_aux_closed_loop_zero_case_unbiased():
    tga_est = _aux_closed_loop_tga(0.0)
    assert abs(tga_est) < 0.05e-3, tga_est
