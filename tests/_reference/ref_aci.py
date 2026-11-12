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
"""Frozen copy of `mvis.calibration.factors.aci` (S0, never edited):
`integrate_RAJ` and `ACIPreintegrator` (the gyro-only preintegrator is not
copied — no step touches it). Only the imports change; `set_intrinsics`
freezes the live ImuModel through `wrap_intr` (S0 adapter).
"""

from __future__ import annotations

import numpy as np

from .ref_imu_model import (
    KALIBR, RPNG, dw_jacobian, da_jacobian, tg_jacobian, wrap_intr,
)
from .ref_lie_groups import skew

_I3 = np.eye(3)
_Z3 = np.zeros((3, 3))
_SMALL = (1.0 / 180.0) * np.pi / 2.0  # 0.5 deg threshold (matches C++)


def integrate_RAJ(d_t: float, w_hat: np.ndarray, a_hat: np.ndarray):
    """Closed-form rotation + RAJ integrals over one step.

    Returns (R, xi_1, xi_2, J, xi_3, xi_4):
      R     = Exp(w_hat·d_t)            (incremental rotation dR)
      xi_1  = ∫ R dt        (int_R)
      xi_2  = ∫∫ R dt       (iint_R)
      J     = right Jacobian Jr(w_hat·d_t)
      xi_3  = ∫ R [a]× dt   (int_RAJ)
      xi_4  = ∫∫ R [a]× dt  (iint_RAJ)
    """
    w_norm = float(np.linalg.norm(w_hat))
    d_th = w_norm * d_t
    k_hat = np.zeros(3) if w_norm < 1e-8 else w_hat / w_norm

    d_t2, d_t3 = d_t ** 2, d_t ** 3
    w2, w3 = w_norm ** 2, w_norm ** 3
    cos_dth, sin_dth = np.cos(d_th), np.sin(d_th)
    d_th2, d_th3 = d_th ** 2, d_th ** 3
    sK = skew(k_hat)
    sK2 = sK @ sK
    sA = skew(a_hat)
    kdota = float(k_hat @ a_hat)

    small_w = w_norm < _SMALL
    small_th = d_th < _SMALL

    R = _I3 + sin_dth * sK + (1.0 - cos_dth) * sK2
    if not small_th:
        J = _I3 - (1.0 - cos_dth) / d_th * sK + (d_th - sin_dth) / d_th * sK2
    else:
        # Taylor series of the same Jr(φ): (1-cos θ)/θ ≈ θ/2 - θ³/24 and
        # (θ-sin θ)/θ ≈ θ²/6 - θ⁴/120. (The C++ port's `I - sin θ [k]× + ...`
        # is I - [φ]× ..., twice the true first-order term — a 0.5 % Jacobian
        # error at 200 Hz that only the live-Jacobian factors notice.)
        J = (_I3 - (0.5 * d_th - d_th3 / 24.0) * sK
             + (d_th2 / 6.0 - d_th2 * d_th2 / 120.0) * sK2)

    if not small_w:
        xi_1 = _I3 * d_t + (1.0 - cos_dth) / w_norm * sK + (d_t - sin_dth / w_norm) * sK2
        xi_2 = (0.5 * d_t2 * _I3
                + (d_th - sin_dth) / w2 * sK
                + (0.5 * d_t2 - (1.0 - cos_dth) / w2) * sK2)
        xi_3 = (0.5 * d_t2 * sA
                + (sin_dth - d_th) / w2 * sA @ sK
                + (sin_dth - d_th * cos_dth) / w2 * sK @ sA
                + (0.5 * d_t2 - (1.0 - cos_dth) / w2) * sA @ sK2
                + (0.5 * d_t2 + (1.0 - cos_dth - d_th * sin_dth) / w2) * (sK2 @ sA + kdota * sK)
                - (3 * sin_dth - 2 * d_th - d_th * cos_dth) / w2 * kdota * sK2)
        xi_4 = (1.0 / 6 * d_t3 * sA
                + (2 * (1.0 - cos_dth) - d_th2) / (2 * w3) * sA @ sK
                + (2 * (1.0 - cos_dth) - d_th * sin_dth) / w3 * sK @ sA
                + ((sin_dth - d_th) / w3 + d_t3 / 6) * sA @ sK2
                + ((d_th - 2 * sin_dth + 1.0 / 6 * d_th3 + d_th * cos_dth) / w3) * (sK2 @ sA + kdota * sK)
                + (4 * cos_dth - 4 + d_th2 + d_th * sin_dth) / w3 * kdota * sK2)
    else:
        xi_1 = d_t * (_I3 + sin_dth * sK + (1.0 - cos_dth) * sK2)
        xi_2 = 0.5 * d_t * xi_1
        xi_3 = 0.5 * d_t2 * (sA
                             + sin_dth * (-sA @ sK + sK @ sA + kdota * sK2)
                             + (1.0 - cos_dth) * (sA @ sK2 + sK2 @ sA + kdota * sK))
        xi_4 = 1.0 / 3 * d_t * xi_3

    return R, xi_1, xi_2, J, xi_3, xi_4


class ACIPreintegrator:
    """Analytic combined IMU preintegrator with intrinsics (ports ACI_In)."""

    def __init__(self, sigma_g, sigma_bg, sigma_a, sigma_ba,
                 model: int = RPNG, avg_meas: bool = False):
        self.model = model
        self.avg_meas = avg_meas
        self.sigma_g, self.sigma_bg = sigma_g, sigma_bg
        self.sigma_a, self.sigma_ba = sigma_a, sigma_ba

        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = sigma_g ** 2 * _I3
        Qc[3:6, 3:6] = sigma_a ** 2 * _I3
        Qc[6:9, 6:9] = sigma_bg ** 2 * _I3
        Qc[9:12, 9:12] = sigma_ba ** 2 * _I3
        self.Qc = Qc

        self.grav = np.zeros(3)
        from mvis.imu import create_imu_model          # value object only
        self.intr = wrap_intr(create_imu_model(model))  # S0 adapter
        self.b_g_init = np.zeros(3)
        self.b_a_init = np.zeros(3)
        self.reset(np.zeros(3), np.zeros(3))

    def reset(self, b_g_init: np.ndarray, b_a_init: np.ndarray):
        """Reset the preintegrator to identity at the given init biases (clears D_R/D_v/D_p, all bias + intrinsic Jacobians, and the covariance)."""
        self.b_g_init = np.asarray(b_g_init, float).copy()
        self.b_a_init = np.asarray(b_a_init, float).copy()
        self.Dt = 0.0
        self.D_R = np.eye(3)
        self.D_v = np.zeros(3)
        self.D_p = np.zeros(3)
        self.R_ki2k = np.eye(3)
        # bias Jacobians
        self.J_R_bg = np.zeros((3, 3)); self.J_R_ba = np.zeros((3, 3))
        self.J_v_bg = np.zeros((3, 3)); self.J_v_ba = np.zeros((3, 3))
        self.J_p_bg = np.zeros((3, 3)); self.J_p_ba = np.zeros((3, 3))
        # intrinsic Jacobians (3x24 each)
        self.J_R_in = np.zeros((3, 24))
        self.J_v_in = np.zeros((3, 24))
        self.J_p_in = np.zeros((3, 24))
        # gyro<->accel time-offset Jacobians (3,) — d(delta)/d t_ga at t_ga=0,
        # where the accel applying at gyro time t is a_m(t + t_ga)
        self.J_R_tga = np.zeros(3)
        self.J_v_tga = np.zeros(3)
        self.J_p_tga = np.zeros(3)
        # covariance (small prior, matches C++ 1e-15·I)
        self.P_meas = np.eye(15) * 1e-15

    def set_intrinsics(self, intr):
        """Set the IMU intrinsic model used to correct raw measurements during preintegration."""
        self.intr = wrap_intr(intr)                     # S0 adapter

    def feed_imu_interval(self, t_vec, w_vec, a_vec):
        """Preintegrate a block of IMU samples pairwise; also caches the corrected w/a at the interval endpoints for the multi-IMU time-offset coupling."""
        t_vec = np.asarray(t_vec, float)
        w_vec = [np.asarray(w, float) for w in w_vec]
        a_vec = [np.asarray(a, float) for a in a_vec]
        for i in range(len(t_vec) - 1):
            self.feed_imu_single(t_vec[i], t_vec[i + 1],
                                 w_vec[i], a_vec[i], w_vec[i + 1], a_vec[i + 1])
        # corrected angular/linear measurements at interval endpoints (in IMU
        # frame) — used by the multi-IMU factor's kinematic time-offset coupling.
        R_atoI, Da = self.intr.R_atoI, self.intr.Da()
        R_wtoI, Dw, Tg = self.intr.R_wtoI, self.intr.Dw(), self.intr.Tg()
        self.a_m_s = R_atoI @ Da @ (a_vec[0] - self.b_a_init)
        self.a_m_e = R_atoI @ Da @ (a_vec[-1] - self.b_a_init)
        self.w_m_s = R_wtoI @ Dw @ (w_vec[0] - self.b_g_init - Tg @ self.a_m_s)
        self.w_m_e = R_wtoI @ Dw @ (w_vec[-1] - self.b_g_init - Tg @ self.a_m_e)

    def feed_imu_single(self, t_0, t_1, w_m_0, a_m_0, w_m_1=None, a_m_1=None):
        """Preintegrate one IMU step over [t_0, t_1] (forward/endpoint measurements), updating D_R/D_v/D_p and their bias + intrinsic Jacobians."""
        d_t = t_1 - t_0
        if d_t <= 0:
            return
        a1_given = a_m_1 is not None      # finite-diff a-dot needs a real endpoint
        if w_m_1 is None:
            w_m_1 = np.zeros(3)
        if a_m_1 is None:
            a_m_1 = np.zeros(3)

        R_atoI = self.intr.R_atoI
        R_wtoI = self.intr.R_wtoI
        Dw = self.intr.Dw()
        Da = self.intr.Da()
        Tg = self.intr.Tg()
        model = self.model

        # corrected measurements (intrinsics + bias)
        a0_ina = a_m_0 - self.b_a_init
        a1_ina = a_m_1 - self.b_a_init
        a0_inI = R_atoI @ Da @ a0_ina
        a1_inI = R_atoI @ Da @ a1_ina
        w0_inw = w_m_0 - self.b_g_init - Tg @ a0_inI
        w0_inI = R_wtoI @ Dw @ w0_inw
        w1_inw = w_m_1 - self.b_g_init - Tg @ a1_inI
        w1_inI = R_wtoI @ Dw @ w1_inw

        a_ina, a_hat = a0_ina, a0_inI
        w_inw, w_hat = w0_inw, w0_inI
        if self.avg_meas:
            a_ina = (a0_ina + a1_ina) / 2.0
            a_hat = (a0_inI + a1_inI) / 2.0
            w_inw = (w0_inw + w1_inw) / 2.0
            w_hat = (w0_inI + w1_inI) / 2.0

        dR, int_R, iint_R, Jr, int_RAJ, iint_RAJ = integrate_RAJ(d_t, w_hat, a_hat)

        self.R_ki2k = self.D_R.copy()
        # state recurrence
        self.D_p = self.D_p + self.D_v * d_t + self.R_ki2k @ iint_R @ a_hat
        self.D_v = self.D_v + self.R_ki2k @ int_R @ a_hat
        self.D_R = self.D_R @ dR

        # --- Jacobian blocks ---
        H_b = np.zeros((6, 6))
        H_b[0:3, 0:3] = -R_wtoI @ Dw
        H_b[0:3, 3:6] = R_wtoI @ Dw @ Tg @ R_atoI @ Da
        H_b[3:6, 3:6] = -R_atoI @ Da
        H_n = H_b

        # Intrinsic Jacobian, laid out per the model's tangent (see ImuModel):
        #   Kalibr/RPNG: [Dw(6), Da(6), Tg(9), R(3)]   Calyx: [Dw(9), Da(6), Tg(9)]
        # All 24-dim; offsets come from intr so this stays model-agnostic. For
        # Calyx R_wtoI = R_atoI = I and there is no rotation block (has_R False).
        n = self.intr.n_dw
        da_off, tg_off = self.intr.da_off, self.intr.tg_off
        H_in = np.zeros((6, 24))
        H_in[0:3, 0:n] = R_wtoI @ dw_jacobian(w_inw, model)
        H_in[0:3, da_off:da_off + 6] = -R_wtoI @ Dw @ Tg @ R_atoI @ da_jacobian(a_ina, model)
        H_in[3:6, da_off:da_off + 6] = R_atoI @ da_jacobian(a_ina, model)
        H_in[0:3, tg_off:tg_off + 9] = -R_wtoI @ Dw @ tg_jacobian(a_hat)
        if self.intr.has_R:
            r_off = self.intr.r_off
            if model == KALIBR:
                H_in[0:3, r_off:r_off + 3] = -skew(w_hat) @ R_wtoI
            else:
                H_in[0:3, r_off:r_off + 3] = R_wtoI @ Dw @ Tg @ skew(a_hat) @ R_atoI
                H_in[3:6, r_off:r_off + 3] = -skew(a_hat) @ R_atoI

        phi_nn = np.eye(9)
        phi_nn[0:3, 0:3] = dR.T
        phi_nn[3:6, 0:3] = -self.R_ki2k @ skew(iint_R @ a_hat)
        phi_nn[3:6, 6:9] = _I3 * d_t
        phi_nn[6:9, 0:3] = -self.R_ki2k @ skew(int_R @ a_hat)

        H_wa = np.zeros((9, 6))
        H_wa[0:3, 0:3] = Jr * d_t
        H_wa[3:6, 0:3] = -self.R_ki2k @ iint_RAJ
        H_wa[6:9, 0:3] = -self.R_ki2k @ int_RAJ
        H_wa[3:6, 3:6] = self.R_ki2k @ iint_R
        H_wa[6:9, 3:6] = self.R_ki2k @ int_R

        # accumulate bias Jacobians (9x6 stacked: theta,p,v over bg,ba)
        J_b_k = np.zeros((9, 6))
        J_b_k[0:3, 0:3] = self.J_R_bg; J_b_k[0:3, 3:6] = self.J_R_ba
        J_b_k[3:6, 0:3] = self.J_p_bg; J_b_k[3:6, 3:6] = self.J_p_ba
        J_b_k[6:9, 0:3] = self.J_v_bg; J_b_k[6:9, 3:6] = self.J_v_ba
        J_b_ki = phi_nn @ J_b_k + H_wa @ H_b
        self.J_R_bg, self.J_R_ba = J_b_ki[0:3, 0:3], J_b_ki[0:3, 3:6]
        self.J_p_bg, self.J_p_ba = J_b_ki[3:6, 0:3], J_b_ki[3:6, 3:6]
        self.J_v_bg, self.J_v_ba = J_b_ki[6:9, 0:3], J_b_ki[6:9, 3:6]

        # accumulate intrinsic Jacobians (9x24 stacked)
        J_in_k = np.zeros((9, 24))
        J_in_k[0:3, :] = self.J_R_in
        J_in_k[3:6, :] = self.J_p_in
        J_in_k[6:9, :] = self.J_v_in
        J_in_ki = phi_nn @ J_in_k + H_wa @ H_in
        self.J_R_in = J_in_ki[0:3, :]
        self.J_p_in = J_in_ki[3:6, :]
        self.J_v_in = J_in_ki[6:9, :]

        # accumulate gyro<->accel time-offset Jacobian (one extra column of the
        # intrinsic pattern): da/dt_ga = a-dot (raw-measurement finite difference
        # over the step), entering the accel row directly and the gyro row
        # through the g-sensitivity Tg.
        a_dot = (a_m_1 - a_m_0) / d_t if a1_given else np.zeros(3)
        da_inI_tga = R_atoI @ Da @ a_dot
        H_tga = np.concatenate([-R_wtoI @ Dw @ Tg @ da_inI_tga, da_inI_tga])
        J_tga_k = np.concatenate([self.J_R_tga, self.J_p_tga, self.J_v_tga])
        J_tga_ki = phi_nn @ J_tga_k + H_wa @ H_tga
        self.J_R_tga = J_tga_ki[0:3]
        self.J_p_tga = J_tga_ki[3:6]
        self.J_v_tga = J_tga_ki[6:9]

        # covariance propagation
        Phi = np.eye(15)
        Phi[0:9, 0:9] = phi_nn
        Phi[0:9, 9:15] = H_wa @ H_b
        G = np.zeros((15, 12))
        G[0:9, 0:6] = H_wa @ H_n
        G[9:12, 6:9] = _I3 * d_t
        G[12:15, 9:12] = _I3 * d_t
        Qd = self.Qc / d_t
        self.P_meas = Phi @ self.P_meas @ Phi.T + G @ Qd @ G.T

        self.R_ki2k = self.D_R.copy()
        self.Dt += d_t
