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
"""Speed-up step K1 (bitwise scalar cleanup in the IMU path) — gates beyond
the frozen-oracle fixture of `test_speedup_equivalence.py`:

  * `integrate_R` == `integrate_RAJ[:4]` == the frozen `ref_aci.integrate_RAJ`
    bitwise on 10k random inputs that include the branch edges (|ω| = 0,
    just below / at / above 1e-8, |ω| and |ω|·dt at `_SMALL`, dt = 0);
  * jac=False and jac=True give the same `D_*` / `dtheta_dDelta` / `w_inI`
    for every model and for raw gyro (jac=False now takes `integrate_R`);
  * `_IntrPack` holds exactly the left-prefix products the pre-K1 sites
    evaluated inline (checked against the inline expressions, `array_equal`);
  * `ACIPreintegrator` is `array_equal` to `ref_aci.ACIPreintegrator` on the
    `test_aci` / `test_aci_tga` fixtures, avg_meas on / off, and on the direct
    `feed_imu_single` path without an endpoint (a_dot = 0);
  * `_Corrected` / `_step` accept an `ImuModel` as well as a pack (bench /
    external callers) and give the same numbers;
  * `imu_model`: the class registry is cached, the identity sides of
    `R_wtoI` / `R_atoI` are the shared read-only `_I3`, the model's own `R` is
    still returned by reference, and `_model_class` keeps `parse_imu_model`'s
    error for unknown ids.
"""
import numpy as np
import pytest

from mvis.calibration.factors import aci as A
from mvis.calibration.factors.aci import (integrate_R, integrate_RAJ, _IntrPack,
                                          ACIPreintegrator, _SMALL)
from mvis.calibration.factors.imu_segment import (ImuSegment, _Corrected, _State,
                                                  _step)
from mvis.calibration.factors.lie_groups import skew
from mvis.imu import (CALYX, KALIBR, RPNG, create_imu_model, dw_jacobian,
                      da_jacobian, tg_jacobian)
from mvis.imu import imu_model as M

from tests._reference import ref_aci
from tests.helpers.speedup import (imu_signal, imu_intr, propagation_targets,
                                            _BG, _BA, _eq)
from tests.helpers.imu_signals import _make_sequence
from tests.helpers.imu_signals import _w as _tga_w, _a as _tga_a, _intr_with_tg
from tests.helpers.speedup import same_as_oracle


def _edge_inputs(rng, n=10_000):
    """Random (d_t, w, a) plus every branch edge of `integrate_RAJ`."""
    out = []
    for _ in range(n):
        d_t = rng.choice([1e-4, 2.5e-3, 5e-3, 0.02, 0.05]) * rng.uniform(0.5, 1.5)
        scale = rng.choice([0.0, 5e-9, 1e-8, 1e-4, 5e-3, _SMALL, 0.05, 1.0, 4.0])
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * scale
        out.append((float(d_t), w, rng.normal(size=3) * 5.0))
    ax = np.array([0.6, -0.8, 0.0])
    a = np.array([0.1, 9.7, 0.3])
    for wn in (0.0, 1e-8 - 1e-24, 1e-8, 1e-8 + 1e-24, _SMALL - 1e-18, _SMALL,
               _SMALL + 1e-18):
        for d_t in (0.0, 1e-3, 5e-3, 0.1):
            out.append((d_t, ax * wn, a))
    for d_t in (1e-3, 5e-3):                  # |ω| dt exactly at / around _SMALL
        for wn in (_SMALL / d_t, np.nextafter(_SMALL / d_t, 0.0),
                   np.nextafter(_SMALL / d_t, 1.0)):
            out.append((d_t, ax * wn, a))
    out.append((5e-3, np.array([0.3, -0.2, 0.5]), np.zeros(3)))
    return out


def test_integrate_r_matches_oracle_10k():
    rng = np.random.default_rng(21)
    hits = {"tiny": 0, "small_w": 0, "small_th": 0, "full": 0}
    inputs = _edge_inputs(rng)
    assert len(inputs) >= 10_000
    for d_t, w, a in inputs:
        ref = ref_aci.integrate_RAJ(d_t, w, a)
        r4 = integrate_R(d_t, w)
        raj = integrate_RAJ(d_t, w, a)
        assert len(r4) == 4 and len(raj) == 6
        for x, y, z in zip(r4, raj[:4], ref[:4]):
            assert _eq(x, y) and _eq(x, z)
        for x, z in zip(raj[4:], ref[4:]):
            assert _eq(x, z)
        wn = float(np.linalg.norm(w))
        key = ("tiny" if wn < 1e-8 else "small_w" if wn < _SMALL
               else "small_th" if wn * d_t < _SMALL else "full")
        hits[key] += 1
    assert all(v > 0 for v in hits.values()), hits


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
@pytest.mark.parametrize("gyro", ["calibrated", "raw"])
def test_jac_false_equals_jac_true_states(model, gyro):
    rng = np.random.default_rng(22)
    t, w, a = imu_signal()
    t0 = t[70] + 0.0021
    deltas = propagation_targets(t, t0, rng)
    intr = None if gyro == "raw" else imu_intr(model)
    seg = ImuSegment(t, w, a, model)
    P = seg.propagate(t0, deltas, _BG, _BA, intr, jac=True)
    Q = seg.propagate(t0, deltas, _BG, _BA, intr, jac=False)
    assert len(P) == len(Q) == 150
    for p, q in zip(P, Q):
        for k in ("D_R", "D_v", "D_p", "dtheta_dDelta", "w_inI"):
            assert _eq(getattr(p, k), getattr(q, k)), k
        # the Jacobian blocks are the zero state blocks when jac=False
        assert q.J_R_b.shape == (3, 6) and not np.any(q.J_R_b) and not np.any(q.J_p_in)


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
def test_intr_pack_holds_the_inline_prefix_products(model):
    intr = imu_intr(model)
    p = _IntrPack(intr)
    R_atoI, R_wtoI = intr.R_atoI, intr.R_wtoI
    Dw, Da, Tg = intr.Dw(), intr.Da(), intr.Tg()
    # imu_segment._Corrected / feed_imu_single corrections
    assert _eq(p.Ma, R_atoI @ Da) and _eq(p.Mw, R_wtoI @ Dw) and _eq(p.Tg, Tg)
    # feed_imu_single's t_ga row (its H_b / H_in use the sign +1 entries below since G10)
    assert _eq(p.nMwTg, -R_wtoI @ Dw @ Tg)
    # imu_segment._step prefixes per sign
    RwDw, RaDa = R_wtoI @ Dw, R_atoI @ Da
    for sign in (1, -1):
        H_b = np.zeros((6, 6))
        H_b[0:3, 0:3] = -sign * RwDw
        H_b[0:3, 3:6] = sign * RwDw @ Tg @ RaDa
        H_b[3:6, 3:6] = -RaDa
        assert _eq(p.Hb[sign], H_b)
        assert _eq(p.sRw[sign], sign * R_wtoI)
        assert _eq(p.nsMw[sign], -sign * RwDw)
        assert _eq(p.nsMwTgRa[sign], -sign * RwDw @ Tg @ R_atoI)
        assert _eq(p.sMwTg[sign], sign * RwDw @ Tg)
    # bound parameter Jacobians == the registry dispatch
    v = np.array([0.3, -1.2, 2.5])
    assert _eq(p.dwj(v), dw_jacobian(v, model))
    assert _eq(p.daj(v), da_jacobian(v, model))
    assert _eq(p.tgj(v), tg_jacobian(v))
    assert (p.n_dw, p.da_off, p.tg_off, p.has_R) == (intr.n_dw, intr.da_off,
                                                    intr.tg_off, intr.has_R)
    assert p.r_off == (intr.r_off if intr.has_R else None)
    assert p.model == model


def test_pack_captures_values_at_build_time():
    intr = imu_intr(CALYX)
    p = _IntrPack(intr)
    Ma0 = p.Ma.copy()
    intr.x_da = intr.x_da + 0.1
    assert _eq(p.Ma, Ma0) and not _eq(_IntrPack(intr).Ma, Ma0)


def _preint_pair(model, avg_meas, t, w, a, intr=None, bg=None, ba=None):
    outs = []
    for cls in (ACIPreintegrator, ref_aci.ACIPreintegrator):
        pre = cls(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=avg_meas)
        pre.reset(np.zeros(3) if bg is None else bg, np.zeros(3) if ba is None else ba)
        if intr is not None:
            pre.set_intrinsics(intr)
        pre.feed_imu_interval(t, w, a)
        outs.append(pre)
    return outs


_PRE_KEYS = ("D_R", "D_v", "D_p", "Dt", "J_R_bg", "J_R_ba", "J_p_bg", "J_p_ba", "J_v_bg",
             "J_v_ba", "J_R_in", "J_p_in", "J_v_in", "J_R_tga", "J_p_tga", "J_v_tga",
             "P_meas", "a_m_s", "a_m_e", "w_m_s", "w_m_e")


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
@pytest.mark.parametrize("avg_meas", [True, False])
def test_preintegrator_matches_oracle_on_test_aci_fixture(model, avg_meas):
    t, w, a = _make_sequence(seed=1)
    bg = np.array([0.01, -0.02, 0.015]); ba = np.array([0.1, -0.05, 0.2])
    for intr in (None, imu_intr(model)):
        live, ref = _preint_pair(model, avg_meas, t, w, a, intr, bg, ba)
        for k in _PRE_KEYS:
            assert same_as_oracle(k, getattr(live, k), getattr(ref, k)), (k, intr is None)


@pytest.mark.parametrize("avg_meas", [True, False])
def test_preintegrator_matches_oracle_on_test_aci_tga_fixture(avg_meas):
    hz, tga = 400.0, 0.004
    ts = np.arange(0.0, 1.0, 1.0 / hz)
    ws = [_tga_w(t) for t in ts]
    as_ = [_tga_a(t + tga) for t in ts]
    outs = []
    for cls in (ACIPreintegrator, ref_aci.ACIPreintegrator):
        pre = cls(1e-3, 1e-5, 1e-2, 1e-4, model=RPNG, avg_meas=avg_meas)
        pre.reset(np.zeros(3), np.zeros(3))
        pre.set_intrinsics(_intr_with_tg())
        pre.feed_imu_interval(ts, ws, as_)
        outs.append(pre)
    for k in _PRE_KEYS:
        assert same_as_oracle(k, getattr(outs[0], k), getattr(outs[1], k)), k


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_feed_imu_single_without_endpoint_matches_oracle(model):
    """Direct `feed_imu_single` calls without the endpoint sample: a_dot = 0
    (a1_given False), w_m_1 / a_m_1 default to zero; no pack passed."""
    t, w, a = _make_sequence(n=12, seed=3)
    outs = []
    for cls in (ACIPreintegrator, ref_aci.ACIPreintegrator):
        pre = cls(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=False)
        pre.reset(np.array([0.01, -0.02, 0.015]), np.array([0.1, -0.05, 0.2]))
        pre.set_intrinsics(imu_intr(model))
        for i in range(len(t) - 1):
            pre.feed_imu_single(t[i], t[i + 1], w[i], a[i])
        pre.feed_imu_single(t[-1], t[-1], w[-1], a[-1])          # d_t <= 0: skipped
        outs.append(pre)
    for k in _PRE_KEYS[:-4]:
        assert same_as_oracle(k, getattr(outs[0], k), getattr(outs[1], k)), k


def test_corrected_and_step_accept_model_or_pack():
    intr = imu_intr(RPNG)
    pack = _IntrPack(intr)
    wv = np.array([0.3, -0.2, 0.5]); av = np.array([0.1, 9.7, 0.3])
    c_m = _Corrected(intr, _BG, _BA, wv, av)
    c_p = _Corrected(pack, _BG, _BA, wv, av)
    for k in ("a_ina", "a_inI", "w_inw", "w_inI"):
        assert _eq(getattr(c_m, k), getattr(c_p, k))
    for jac in (True, False):
        for sign in (1, -1):
            s_m = _step(_State(), 0.005, c_m, c_p, intr, sign, jac)
            s_p = _step(_State(), 0.005, c_m, c_p, pack, sign, jac)
            for k in ("D_R", "D_v", "D_p", "J_b", "J_in", "Dt", "Jr_last"):
                assert _eq(getattr(s_m, k), getattr(s_p, k)), (k, jac, sign)


def test_model_registry_cached_and_identity_rotations_frozen():
    assert M._model_classes() is M._model_classes()
    for name in ("kalibr", "rpng", "calyx", np.int64(2), 0):
        assert M._model_class(name) is M._model_classes()[M.parse_imu_model(name)]
    with pytest.raises(ValueError):
        M._model_class("nope")
    with pytest.raises(ValueError):
        M._model_class(7)
    c, k, r = create_imu_model(CALYX), create_imu_model(KALIBR), create_imu_model(RPNG)
    for arr in (c.R_wtoI, c.R_atoI, k.R_atoI, r.R_wtoI):
        assert _eq(arr, np.eye(3)) and not arr.flags.writeable
        assert arr is M._I3
    assert k.R_wtoI is k.R and k.R.flags.writeable
    assert r.R_atoI is r.R and r.R.flags.writeable
    with pytest.raises(ValueError):
        c.R_wtoI[0, 0] = 2.0                      # read-only: a slip cannot corrupt it
    assert _eq(c.R_wtoI, np.eye(3))


@pytest.mark.parametrize("n", [1, 7, 8, 9, 81])
def test_propagate_matches_oracle_at_several_query_counts(n):
    """The scalar math.* swap must hold whatever the query count (critic: SIMD
    tail handling); the live propagate equals the frozen oracle bitwise for
    N ∈ {1, 7, 8, 9, 81} mixed-sign targets, jac on and off."""
    from tests._reference import ref_imu_segment
    rng = np.random.default_rng(23 + n)
    t, w, a = imu_signal()
    t0 = t[70] + 0.0021
    deltas = rng.uniform(-0.05, 0.16, n)
    for model in (CALYX, KALIBR, RPNG):
        for jac in (True, False):
            live = ImuSegment(t, w, a, model).propagate(t0, deltas, _BG, _BA,
                                                        imu_intr(model), jac=jac)
            ref = ref_imu_segment.ImuSegment(t, w, a, model).propagate(
                t0, deltas, _BG, _BA, imu_intr(model), jac=jac)
            for p, q in zip(live, ref):
                for k in ("D_R", "D_v", "D_p", "J_R_b", "J_p_b", "J_R_in", "J_p_in",
                          "dtheta_dDelta", "w_inI"):
                    assert _eq(getattr(p, k), getattr(q, k)), (n, model, jac, k)


def test_integrate_r_skew_identity_branch():
    """|ω| < 1e-8 -> k_hat = 0: R = I, J = I, xi_1 = I dt, xi_2 = I dt²/2 exactly."""
    d_t = 0.005
    R, x1, x2, J = integrate_R(d_t, np.zeros(3))
    assert _eq(R, np.eye(3)) and _eq(J, np.eye(3))
    assert _eq(x1, np.eye(3) * d_t) and _eq(x2, 0.5 * d_t * (np.eye(3) * d_t))
    assert _eq(skew(np.zeros(3)), np.zeros((3, 3)))
    assert A._I3.shape == (3, 3)
