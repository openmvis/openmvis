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
"""ImuFactorACI_Multi_In_V (aux-IMU / IMU-IMU): residual + coupling validation.

Strategy: build a residual≈0 state in the AUX frame (reusing the single-IMU
consistent-state construction), then INVERT the kinematic coupling to recover the
base-IMU states + calibration. The multi factor evaluated on the base states must
(a) reproduce the aux single-IMU residual and (b) be ≈ 0.
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import ImuModel, KALIBR, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.imu_factor_multi import ImuFactorACI_Multi_In_V
from mvis.calibration.factors.lie_groups import exp_so3 as _exp


def _build(model, seed=0, window_td=0.0):
    rng = np.random.RandomState(seed)
    n, dt = 15, 0.005
    t = np.arange(n + 1) * dt
    w = np.array([[0.3 * np.sin(2 * x), 0.2 * np.cos(3 * x), 0.1 * x] for x in t]) + rng.randn(n + 1, 3) * 0.01
    a = np.array([[0.5 * np.cos(x), 9.81 + 0.3 * np.sin(x), 0.2 * x] for x in t]) + rng.randn(n + 1, 3) * 0.01
    bg = np.array([0.01, -0.02, 0.005])
    ba = np.array([0.05, -0.03, 0.02])
    intr = ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.array([0.01, -0.02, 0.0, 0.005, 0.0, -0.01, 0.0, 0.008, 0.0]),
        R=_exp(np.array([0.02, -0.01, 0.03])), model=model)
    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
    pre.reset(bg, ba)
    pre.set_intrinsics(intr)
    pre.feed_imu_interval(t, w, a)
    g_inG = np.array([0.0, 0.0, -9.81])
    # base extrinsic / td estimates (the calib linearization point)
    T_ext_init = gtsam.Pose3(gtsam.Rot3(_exp(np.array([0.05, -0.03, 0.08]))),
                             gtsam.Point3(0.1, -0.05, 0.02))
    # the pipeline sets td_init = window_td = sd.td, so the propagated RESIDUAL
    # (calib.td - window_td) stays small; mirror that here (0.004 base offset
    # so xi can still perturb td around the linearization point)
    td_init = 0.004 + window_td
    multi = ImuFactorACI_Multi_In_V.from_preintegrator(pre, g_inG, T_ext_init,
                                                       td_init, window_td=window_td)
    return multi, multi.base


def _consistent_aux(f, seed=3):
    """Residual≈0 state for the single-IMU base factor (aux frame). Returns the
    intrinsics tangent xi_in (=Δ_in) rather than (R_intr, x21)."""
    rng = np.random.RandomState(seed)
    R0 = _exp(rng.randn(3) * 0.1)
    p0 = rng.randn(3) * 0.1
    v0 = rng.randn(3) * 0.1
    bg0 = f.bg_init + rng.randn(3) * 0.002
    ba0 = f.ba_init + rng.randn(3) * 0.01
    xi_in = rng.randn(24) * 0.01
    d_bg, d_ba, d_in = bg0 - f.bg_init, ba0 - f.ba_init, xi_in
    d_th = f.J_R_bg @ d_bg + f.J_R_ba @ d_ba + f.J_R_in @ d_in
    R_bcorr = _exp(d_th)
    Dt, g = f.Dt, f.g_inG
    R1 = R0 @ f.D_R @ R_bcorr
    v1 = v0 + g * Dt + R0 @ (f.D_v + f.J_v_bg @ d_bg + f.J_v_ba @ d_ba + f.J_v_in @ d_in)
    p1 = p0 + v0 * Dt + 0.5 * g * Dt ** 2 + R0 @ (f.D_p + f.J_p_bg @ d_bg + f.J_p_ba @ d_ba + f.J_p_in @ d_in)
    pose_Is0 = gtsam.Pose3(gtsam.Rot3(R0), gtsam.Point3(*p0))
    pose_Is1 = gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(*p1))
    vb_Is0 = np.concatenate([v0, bg0, ba0])   # [v, bg, ba]
    vb_Is1 = np.concatenate([v1, bg0, ba0])
    return pose_Is0, pose_Is1, vb_Is0, vb_Is1, xi_in


def _invert_coupling(multi, pose_Is, vb_Is, w_inIs, R_IstoI, p_IsinI, td, v_base):
    """Recover base (pose, vb_base, vbs) from an aux (pose_Is, vb_Is=[v,bg,ba])."""
    R_IstoG = pose_Is.rotation().matrix()
    p_IsinG = np.asarray(pose_Is.translation(), float)
    R_ItoG = R_IstoG @ R_IstoI.T
    w_inI = R_IstoI @ w_inIs
    R_pose = R_ItoG @ _exp(w_inI * td).T
    p_IinG = p_IsinG - R_ItoG @ p_IsinI
    p_pose = p_IinG - v_base * td
    pose = gtsam.Pose3(gtsam.Rot3(R_pose), gtsam.Point3(*p_pose))
    vb_base = np.concatenate([v_base, np.zeros(6)])      # base biases unused
    v_Is, bg_s, ba_s = vb_Is[0:3], vb_Is[3:6], vb_Is[6:9]
    vbs = np.concatenate([bg_s, ba_s, v_Is])             # aux layout [bg, ba, v]
    return pose, vb_base, vbs


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_residual_consistent_via_coupling(model):
    multi, base = _build(model)
    pose_Is0, pose_Is1, vb_Is0, vb_Is1, xi_in = _consistent_aux(base)
    # base residual at this aux state is ~0
    e_aux = base.whitened_error(pose_Is0, pose_Is1, vb_Is0, vb_Is1, xi_in)
    assert np.max(np.abs(e_aux)) < 1e-6

    # calib tangent = [d_pose(6), d_td(1), xi_in(24)]; evaluate the actual calib
    rng = np.random.RandomState(7)
    xi_calib = np.concatenate([rng.randn(6) * 0.02, [0.001], xi_in])
    calib = multi.base_calib.retract(xi_calib)
    R_IstoI, p_IsinI, td = calib.R_IstoI, calib.p_IsinI, calib.td

    # free base velocities; invert the coupling to base states
    v_base0, v_base1 = rng.randn(3) * 0.1, rng.randn(3) * 0.1
    pose0, vb0, vbs0 = _invert_coupling(multi, pose_Is0, vb_Is0, multi.w_inIs0, R_IstoI, p_IsinI, td, v_base0)
    pose1, vb1, vbs1 = _invert_coupling(multi, pose_Is1, vb_Is1, multi.w_inIs1, R_IstoI, p_IsinI, td, v_base1)

    # coupling round-trips: aux states recovered from base states match
    a0, a1, vba0, vba1 = multi._aux_states(pose0, pose1, vb0, vb1, vbs0, vbs1, calib)
    assert np.allclose(a0.rotation().matrix(), pose_Is0.rotation().matrix(), atol=1e-9)
    assert np.allclose(np.asarray(a0.translation()), np.asarray(pose_Is0.translation()), atol=1e-9)
    assert np.allclose(vba0, vb_Is0, atol=1e-9)

    # multi residual on base states reproduces the aux residual (≈ 0)
    e_multi = multi.whitened_error(pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib)
    assert np.allclose(e_multi, e_aux, atol=1e-9)
    assert np.max(np.abs(e_multi)) < 1e-6


def _consistent_full(model, seed=5, window_td=0.0):
    """Build a residual≈0 state and return both base-state args and the derived
    aux states (for cross-checking against the base factor)."""
    multi, base = _build(model, window_td=window_td)
    pose_Is0, pose_Is1, vb_Is0, vb_Is1, xi_in = _consistent_aux(base, seed=seed)
    rng = np.random.RandomState(seed + 6)
    xi_calib = np.concatenate([rng.randn(6) * 0.02, [0.001], xi_in])
    calib = multi.base_calib.retract(xi_calib)
    R_IstoI, p_IsinI = calib.R_IstoI, calib.p_IsinI
    td = calib.td - window_td      # the factor propagates by this RESIDUAL
    pose0, vb0, vbs0 = _invert_coupling(multi, pose_Is0, vb_Is0, multi.w_inIs0, R_IstoI, p_IsinI, td, rng.randn(3) * 0.1)
    pose1, vb1, vbs1 = _invert_coupling(multi, pose_Is1, vb_Is1, multi.w_inIs1, R_IstoI, p_IsinI, td, rng.randn(3) * 0.1)
    base_args = (pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib)
    aux_args = (pose_Is0, pose_Is1, vb_Is0, vb_Is1, xi_in)
    return multi, base, base_args, aux_args


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_jacobians(model):
    multi, base, base_args, aux_args = _consistent_full(model)
    e0, H = multi.error_and_jacobians(*base_args)
    H_p0, H_p1, H_vb0, H_vb1, H_vbs0, H_vbs1, H_calib = H
    assert e0.shape == (15,) and np.max(np.abs(e0)) < 1e-6

    # (1) NON-CIRCULAR: aux-bias + calib-intrinsic blocks equal the base factor's
    # ANALYTIC Jacobians (valid because we're at residual≈0).
    eb, Hb = base.error_and_jacobians(*aux_args)
    Hb_vb_Is0, Hb_vb_Is1, Hb_intr = Hb[2], Hb[3], Hb[4]
    # vbs layout [bg,ba,v] -> vb_Is layout [v,bg,ba]: reorder columns
    reorder0 = np.column_stack([Hb_vb_Is0[:, 3:6], Hb_vb_Is0[:, 6:9], Hb_vb_Is0[:, 0:3]])
    reorder1 = np.column_stack([Hb_vb_Is1[:, 3:6], Hb_vb_Is1[:, 6:9], Hb_vb_Is1[:, 0:3]])
    assert np.allclose(H_vbs0, reorder0, atol=1e-4)
    assert np.allclose(H_vbs1, reorder1, atol=1e-4)
    assert np.allclose(H_calib[:, 7:31], Hb_intr, atol=1e-4)

    # (2) first-order Taylor check of the FULL Jacobian (validates coupling blocks)
    pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib = base_args
    rng = np.random.RandomState(99)
    dp0, dp1 = rng.randn(6) * 1e-6, rng.randn(6) * 1e-6
    dvb0, dvb1 = rng.randn(9) * 1e-6, rng.randn(9) * 1e-6
    dvbs0, dvbs1 = rng.randn(9) * 1e-6, rng.randn(9) * 1e-6
    dcal = rng.randn(31) * 1e-6
    pred = (e0 + H_p0 @ dp0 + H_p1 @ dp1 + H_vb0 @ dvb0 + H_vb1 @ dvb1
            + H_vbs0 @ dvbs0 + H_vbs1 @ dvbs1 + H_calib @ dcal)
    actual = multi.whitened_error(pose0.retract(dp0), pose1.retract(dp1),
                                  vb0 + dvb0, vb1 + dvb1, vbs0 + dvbs0, vbs1 + dvbs1,
                                  xi_calib + dcal)
    assert np.allclose(pred, actual, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_analytic_matches_numerical(model):
    """The analytic Jacobian equals the central-difference reference (per block),
    across several random consistent states (rules out seed-specific luck).
    window_td != 0 exercises the large-td window-shift path — the td (calib
    col 6) and pose columns are evaluated at the RESIDUAL (calib.td -
    window_td), so this confirms the td Jacobian after the robustness fix."""
    for window_td in (0.0, 0.05):
        for seed in (5, 13, 21, 29):
            multi, _, base_args, _ = _consistent_full(model, seed=seed,
                                                       window_td=window_td)
            _, Ha = multi.error_and_jacobians(*base_args)
            _, Hn = multi._numerical_jacobians(*base_args)
            for i, (a, n) in enumerate(zip(Ha, Hn)):
                den = max(1.0, np.max(np.abs(a)), np.max(np.abs(n)))
                assert np.max(np.abs(a - n)) / den < 1e-4, \
                    f"window_td {window_td} seed {seed} block {i}"


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_custom_factor_in_graph(model):
    multi, base, base_args, _ = _consistent_full(model)
    pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib = base_args
    X, V = gtsam.symbol_shorthand.X, gtsam.symbol_shorthand.V
    keys = [X(0), X(1), V(0), V(1), gtsam.symbol('s', 0), gtsam.symbol('s', 1), gtsam.symbol('Y', 0)]
    factor = multi.make_factor(*keys)
    vals = gtsam.Values()
    vals.insert(keys[0], pose0); vals.insert(keys[1], pose1)
    vals.insert(keys[2], vb0); vals.insert(keys[3], vb1)
    vals.insert(keys[4], vbs0); vals.insert(keys[5], vbs1)
    vals.insert(keys[6], xi_calib)
    err = factor.unwhitenedError(vals)
    assert err.shape == (15,)
    assert np.allclose(err, multi.whitened_error(*base_args))
    assert factor.linearize(vals) is not None


# ------------------------------------------------------------------ aux t_ga
@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_tga_column_analytic_matches_numerical(model):
    """The appended aux-t_ga column equals a central difference of the error
    over t_ga (several consistent states; t_ga=0 and a nonzero point)."""
    for seed in (5, 13, 21):
        multi, _, base_args, _ = _consistent_full(model, seed=seed)
        for tga0 in (0.0, 0.002):
            _, Ha = multi.error_and_jacobians(*base_args, t_ga=tga0)
            assert len(Ha) == 8
            h = 1e-6
            ep = multi.whitened_error(*base_args, t_ga=tga0 + h)
            em = multi.whitened_error(*base_args, t_ga=tga0 - h)
            col = ((ep - em) / (2 * h)).reshape(15, 1)
            den = max(1.0, np.max(np.abs(col)))
            assert np.max(np.abs(Ha[7] - col)) / den < 1e-4, f"seed {seed} tga {tga0}"


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_custom_factor_with_tga_key(model):
    """8-key factor: t_ga key wired, error matches, linearizes; and the 7-key
    path stays byte-identical (t_ga absent -> same error as before)."""
    multi, _, base_args, _ = _consistent_full(model)
    pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib = base_args
    X, V = gtsam.symbol_shorthand.X, gtsam.symbol_shorthand.V
    keys = [X(0), X(1), V(0), V(1), gtsam.symbol('s', 0), gtsam.symbol('s', 1),
            gtsam.symbol('Y', 0)]
    k_tga = gtsam.symbol('o', 1)
    factor = multi.make_factor(*keys, k_tga=k_tga)
    vals = gtsam.Values()
    vals.insert(keys[0], pose0); vals.insert(keys[1], pose1)
    vals.insert(keys[2], vb0); vals.insert(keys[3], vb1)
    vals.insert(keys[4], vbs0); vals.insert(keys[5], vbs1)
    vals.insert(keys[6], xi_calib)
    vals.insert(k_tga, np.array([0.0015]))
    err = factor.unwhitenedError(vals)
    assert np.allclose(err, multi.whitened_error(*base_args, t_ga=0.0015))
    assert factor.linearize(vals) is not None
    # 7-key back-compat
    f7 = multi.make_factor(*keys)
    v7 = gtsam.Values()
    for k, v in zip(keys, [pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib]):
        try:
            v7.insert(k, v)
        except Exception:
            v7.insert(k, np.asarray(v, float))
    assert np.allclose(f7.unwhitenedError(v7),
                       multi.whitened_error(*base_args))
