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
"""Frozen copy of `mvis.calibration.factors.imu_segment` (S0, never edited).
Only the imports change; `propagate` freezes the live ImuModel through
`wrap_intr` (S0 adapter) so `intr.Dw()` etc. resolve to the copied builders.
"""

from __future__ import annotations

import numpy as np

from mvis.imu import create_imu_model                 # value object only
from .ref_imu_model import (KALIBR, dw_jacobian, da_jacobian, tg_jacobian,
                            wrap_intr)
from .ref_aci import integrate_RAJ
from .ref_lie_groups import skew

_I3 = np.eye(3)


class _Corrected:
    """Corrected measurements at one time (see ImuModel.correct)."""
    __slots__ = ("a_ina", "a_inI", "w_inw", "w_inI")

    def __init__(self, intr, b_g, b_a, w_m, a_m):
        self.a_ina = a_m - b_a
        self.a_inI = intr.R_atoI @ intr.Da() @ self.a_ina
        self.w_inw = w_m - b_g - intr.Tg() @ self.a_inI
        self.w_inI = intr.R_wtoI @ intr.Dw() @ self.w_inw


class _State:
    """Preintegrated deltas + Jacobians (rows [θ, p, v]; cols [b_g, b_a] / ξ24)."""
    __slots__ = ("D_R", "D_v", "D_p", "J_b", "J_in", "Dt", "Jr_last")

    def __init__(self):
        self.D_R = np.eye(3)
        self.D_v = np.zeros(3)
        self.D_p = np.zeros(3)
        self.J_b = np.zeros((9, 6))
        self.J_in = np.zeros((9, 24))
        self.Dt = 0.0
        self.Jr_last = np.eye(3)


class Propagated:
    """Result of one target: deltas in the ANCHOR frame + Jacobian blocks.

    D_v already carries the direction sign (v = v_k + gΔ + R_k D_v for either
    sign of Δ). `dtheta_dDelta` is d(right-tangent θ of R)/dΔ at the target —
    the end-point body rate through Jr of the partial step; the position part
    of the time derivative is v(t_k+Δ) (assembled by the factor)."""
    __slots__ = ("D_R", "D_v", "D_p", "J_R_b", "J_p_b", "J_R_in", "J_p_in",
                 "dtheta_dDelta", "w_inI")

    def __init__(self, st: _State, sign: int, w_inI_end: np.ndarray):
        self.D_R = st.D_R
        self.D_v = sign * st.D_v
        self.D_p = st.D_p
        self.J_R_b, self.J_p_b = st.J_b[0:3], st.J_b[3:6]
        self.J_R_in, self.J_p_in = st.J_in[0:3], st.J_in[3:6]
        self.w_inI = w_inI_end
        # forward: δθ/dΔ = Jr(w̄_eff δ) w_eff, w_eff = w_I; backward: w_eff = −w_I
        # and dδ/dΔ = −1 — the same expression either way.
        self.dtheta_dDelta = st.Jr_last @ w_inI_end


def _step(st: _State, d_t: float, c0: _Corrected, c1: _Corrected,
          intr, sign: int, jac: bool) -> _State:
    """One avg_meas ACI step over d_t from corrected endpoint measurements,
    with the body rate multiplied by `sign` (−1 = reversed time). Returns a
    new state; `st` is untouched. Mirrors `ACIPreintegrator.feed_imu_single`
    minus covariance / t_ga."""
    a_ina = 0.5 * (c0.a_ina + c1.a_ina)
    a_hat = 0.5 * (c0.a_inI + c1.a_inI)
    w_inw = 0.5 * (c0.w_inw + c1.w_inw)
    w_hat = sign * 0.5 * (c0.w_inI + c1.w_inI)

    dR, int_R, iint_R, Jr, int_RAJ, iint_RAJ = integrate_RAJ(d_t, w_hat, a_hat)

    out = _State()
    R_prev = st.D_R
    out.D_p = st.D_p + st.D_v * d_t + R_prev @ iint_R @ a_hat
    out.D_v = st.D_v + R_prev @ int_R @ a_hat
    out.D_R = R_prev @ dR
    out.Dt = st.Dt + d_t
    out.Jr_last = Jr
    if not jac:
        return out

    R_wtoI, R_atoI = intr.R_wtoI, intr.R_atoI
    Dw, Da, Tg = intr.Dw(), intr.Da(), intr.Tg()
    model = intr.model
    RwDw = R_wtoI @ Dw
    RaDa = R_atoI @ Da

    H_b = np.zeros((6, 6))
    H_b[0:3, 0:3] = -sign * RwDw
    H_b[0:3, 3:6] = sign * RwDw @ Tg @ RaDa
    H_b[3:6, 3:6] = -RaDa

    n = intr.n_dw
    da_off, tg_off = intr.da_off, intr.tg_off
    H_in = np.zeros((6, 24))
    H_in[0:3, 0:n] = sign * R_wtoI @ dw_jacobian(w_inw, model)
    H_in[0:3, da_off:da_off + 6] = -sign * RwDw @ Tg @ R_atoI @ da_jacobian(a_ina, model)
    H_in[3:6, da_off:da_off + 6] = R_atoI @ da_jacobian(a_ina, model)
    H_in[0:3, tg_off:tg_off + 9] = -sign * RwDw @ tg_jacobian(a_hat)
    if intr.has_R:
        r_off = intr.r_off
        if model == KALIBR:
            H_in[0:3, r_off:r_off + 3] = -skew(w_hat) @ R_wtoI
        else:
            H_in[0:3, r_off:r_off + 3] = sign * RwDw @ Tg @ skew(a_hat) @ R_atoI
            H_in[3:6, r_off:r_off + 3] = -skew(a_hat) @ R_atoI

    phi_nn = np.eye(9)
    phi_nn[0:3, 0:3] = dR.T
    phi_nn[3:6, 0:3] = -R_prev @ skew(iint_R @ a_hat)
    phi_nn[3:6, 6:9] = _I3 * d_t
    phi_nn[6:9, 0:3] = -R_prev @ skew(int_R @ a_hat)

    H_wa = np.zeros((9, 6))
    H_wa[0:3, 0:3] = Jr * d_t
    H_wa[3:6, 0:3] = -R_prev @ iint_RAJ
    H_wa[6:9, 0:3] = -R_prev @ int_RAJ
    H_wa[3:6, 3:6] = R_prev @ iint_R
    H_wa[6:9, 3:6] = R_prev @ int_R

    out.J_b = phi_nn @ st.J_b + H_wa @ H_b
    out.J_in = phi_nn @ st.J_in + H_wa @ H_in
    return out


class ImuSegment:
    """Raw base-IMU record + propagation from an anchor time to signed offsets."""

    def __init__(self, t_imu, w_m, a_m, model: int):
        self.t = np.asarray(t_imu, float)
        self.w = np.asarray(w_m, float).reshape(-1, 3)
        self.a = np.asarray(a_m, float).reshape(-1, 3)
        assert len(self.t) == len(self.w) == len(self.a) and len(self.t) >= 2
        assert np.all(np.diff(self.t) > 0), "IMU stamps must be strictly increasing"
        self.model = int(model)
        self.identity = wrap_intr(create_imu_model(self.model))   # S0 adapter

    # ------------------------------------------------------------------
    def covers(self, t0: float, t1: float) -> bool:
        """True when [t0, t1] lies inside the IMU record (no clamping needed)."""
        return bool(self.t[0] <= min(t0, t1) and max(t0, t1) <= self.t[-1])

    def _interp(self, tq: float):
        """Linear interpolation of (w_m, a_m) at tq (clamped at the record ends)."""
        t = self.t
        i = int(np.searchsorted(t, tq, side="right")) - 1
        i = min(max(i, 0), len(t) - 2)
        lam = (tq - t[i]) / (t[i + 1] - t[i])
        lam = min(max(lam, 0.0), 1.0)
        return ((1.0 - lam) * self.w[i] + lam * self.w[i + 1],
                (1.0 - lam) * self.a[i] + lam * self.a[i + 1])

    # ------------------------------------------------------------------
    def propagate(self, t_anchor: float, deltas, b_g, b_a,
                  intr=None, jac: bool = True):
        """Propagate from t_anchor to t_anchor + deltas[i] for every i.

        Returns a list of `Propagated` (one per delta, same order). `intr`
        None = identity intrinsics (raw gyro / accel, biases only)."""
        deltas = np.asarray(deltas, float)
        b_g = np.asarray(b_g, float)
        b_a = np.asarray(b_a, float)
        intr = self.identity if intr is None else wrap_intr(intr)   # S0 adapter
        out = [None] * len(deltas)
        for sign in (1, -1):
            idx = np.nonzero(deltas >= 0.0 if sign > 0 else deltas < 0.0)[0]
            if idx.size:
                self._propagate_dir(t_anchor, deltas, idx, sign, b_g, b_a,
                                    intr, jac, out)
        return out

    def _propagate_dir(self, t_anchor, deltas, idx, sign, b_g, b_a, intr,
                       jac, out):
        tau = sign * deltas[idx]                     # ≥ 0, "elapsed" time
        order = np.argsort(tau)
        tau_max = float(tau[order[-1]])
        t = self.t
        if sign > 0:
            inside = np.nonzero((t > t_anchor) & (t < t_anchor + tau_max))[0]
        else:
            inside = np.nonzero((t > t_anchor - tau_max) & (t < t_anchor))[0][::-1]
        # checkpoints: anchor (interpolated) then the raw samples walked in
        # the integration direction
        tau_cp = np.concatenate([[0.0], sign * (t[inside] - t_anchor)])
        w0, a0 = self._interp(t_anchor)
        meas = [_Corrected(intr, b_g, b_a, w0, a0)]
        meas += [_Corrected(intr, b_g, b_a, self.w[i], self.a[i]) for i in inside]
        states = [_State()]
        for m in range(1, len(tau_cp)):
            states.append(_step(states[m - 1], tau_cp[m] - tau_cp[m - 1],
                                meas[m - 1], meas[m], intr, sign, jac))
        for q in order:
            tq = float(tau[q])
            m = int(np.searchsorted(tau_cp, tq, side="right")) - 1
            delta = tq - tau_cp[m]
            if delta < 1e-12:
                # exactly on a checkpoint: the partial step has δ -> 0, so its
                # Jr -> I (the time derivative is continuous across checkpoints)
                st = states[m]
                st_end = _State()
                st_end.D_R, st_end.D_v, st_end.D_p = st.D_R, st.D_v, st.D_p
                st_end.J_b, st_end.J_in, st_end.Dt = st.J_b, st.J_in, st.Dt
                out[idx[q]] = Propagated(st_end, sign, meas[m].w_inI)
                continue
            wq, aq = self._interp(t_anchor + sign * tq)
            cq = _Corrected(intr, b_g, b_a, wq, aq)
            st = _step(states[m], delta, meas[m], cq, intr, sign, jac)
            out[idx[q]] = Propagated(st, sign, cq.w_inI)
