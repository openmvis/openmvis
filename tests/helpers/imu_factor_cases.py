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
"""Built ACI factors with residual-zero states: 5-key SE3 and gyro-only SO3."""
import numpy as np
import gtsam

from mvis.calibration.factors.aci import ACIPreintegrator, ACIGyroPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.imu_factor_gyro import ImuFactorACI_Multi_In_SO3
from mvis.calibration.factors.lie_groups import exp_so3 as _exp
from mvis.imu import ImuModel, CALYX


def _build_factor(model, seed=0):
    rng = np.random.RandomState(seed)
    n, dt = 15, 0.005
    t = np.arange(n + 1) * dt
    w = np.array([[0.3 * np.sin(2 * x), 0.2 * np.cos(3 * x), 0.1 * x] for x in t]) + rng.randn(n + 1, 3) * 0.01
    a = np.array([[0.5 * np.cos(x), 9.81 + 0.3 * np.sin(x), 0.2 * x] for x in t]) + rng.randn(n + 1, 3) * 0.01
    bg = np.array([0.01, -0.02, 0.005])
    ba = np.array([0.05, -0.03, 0.02])
    if model == CALYX:
        # full 3x3 near-identity gyro correction (row-major 9-vector)
        x_dw = (np.eye(3) + np.array([[0.01, 0.02, -0.01],
                                      [0.005, -0.01, 0.015],
                                      [0.0, 0.008, 0.02]])).flatten()
    else:
        x_dw = np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02])
    x_da = np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99])
    x_tg = np.array([0.01, -0.02, 0.0, 0.005, 0.0, -0.01, 0.0, 0.008, 0.0])
    R0 = gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix()
    intr = ImuModel(x_dw=x_dw, x_da=x_da, x_tg=x_tg, R=R0, model=model)

    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
    pre.reset(bg, ba)
    pre.set_intrinsics(intr)
    pre.feed_imu_interval(t, w, a)
    g_inG = np.array([0.0, 0.0, -9.81])
    return ImuFactorACI_In.from_preintegrator(pre, g_inG), pre


def _consistent_state(f, seed=3):
    """Residual≈0 state — biases/intrinsics OFF init (xi_in != 0), pose1/v1/p1
    derived to zero the residual. Here the first-order analytic Jacobian is exact."""
    rng = np.random.RandomState(seed)
    R0 = gtsam.Rot3.Expmap(rng.randn(3) * 0.1).matrix()
    p0 = rng.randn(3) * 0.1
    v0 = rng.randn(3) * 0.1
    bg0 = f.bg_init + rng.randn(3) * 0.002
    ba0 = f.ba_init + rng.randn(3) * 0.01
    xi_in = rng.randn(24) * 0.01          # ImuModel tangent (= Δ_in)

    d_bg = bg0 - f.bg_init
    d_ba = ba0 - f.ba_init
    d_in = xi_in
    d_th = f.J_R_bg @ d_bg + f.J_R_ba @ d_ba + f.J_R_in @ d_in
    R_bcorr = gtsam.Rot3.Expmap(d_th).matrix()

    Dt, g = f.Dt, f.g_inG
    R_I1toG = R0 @ f.D_R @ R_bcorr
    v1 = v0 + g * Dt + R0 @ (f.D_v + f.J_v_bg @ d_bg + f.J_v_ba @ d_ba + f.J_v_in @ d_in)
    p1 = p0 + v0 * Dt + 0.5 * g * Dt ** 2 + R0 @ (f.D_p + f.J_p_bg @ d_bg + f.J_p_ba @ d_ba + f.J_p_in @ d_in)

    pose0 = gtsam.Pose3(gtsam.Rot3(R0), gtsam.Point3(*p0))
    pose1 = gtsam.Pose3(gtsam.Rot3(R_I1toG), gtsam.Point3(*p1))
    vb0 = np.concatenate([v0, bg0, ba0])
    vb1 = np.concatenate([v1, bg0, ba0])   # bias1 = bias0 -> err_b = 0
    return pose0, pose1, vb0, vb1, xi_in


def _seq(n=20, dt=0.003, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n + 1) * dt
    w = np.array([[0.3 * np.sin(2 * x), 0.2 * np.cos(3 * x), 0.1 * x] for x in t]) + rng.randn(n + 1, 3) * 0.01
    return t, w


def _intr(model):
    return ImuModel(x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
                         x_da=np.ones(6), x_tg=np.zeros(9),
                         R=np.eye(3), model=model)


# --------------------- factor ---------------------
def _build(model):
    t, w = _seq(seed=2)
    bg = np.array([0.01, -0.02, 0.005])
    pre = ACIGyroPreintegrator(0.01, 0.001, model=model, avg_meas=True)
    pre.reset(bg); pre.set_intrinsics(_intr(model)); pre.feed_imu_interval(t, w)
    T_ext = gtsam.Pose3(gtsam.Rot3(_exp(np.array([0.05, -0.03, 0.08]))), gtsam.Point3(0.1, -0.05, 0.02))
    return ImuFactorACI_Multi_In_SO3.from_preintegrator(pre, T_ext, 0.004)


def _consistent(f, model, seed=4):
    """Residual≈0: pick base pose0, calib, gyro biases; derive pose1 so err_th=0."""
    rng = np.random.RandomState(seed)
    R_p0 = _exp(rng.randn(3) * 0.1)
    bg0 = f.bg_init + rng.randn(3) * 0.002
    xi_calib = np.concatenate([rng.randn(6) * 0.02, [0.001],
                               np.concatenate([rng.randn(6) * 0.01, np.zeros(18)])])
    calib = f.base_calib.retract(xi_calib)
    d_bg = bg0 - f.bg_init
    d_in = calib.imu_in.x_dw - f.intr_init.x_dw
    d_th = f.J_R_bg @ d_bg + f.J_R_in @ d_in
    R_bcorr = _exp(d_th)
    # want D_R_hat = D_R @ R_bcorr, with D_R_hat = R_Is0toG.T R_Is1toG
    R_IstoI, td = calib.R_IstoI, calib.td
    R_Is0toG = (R_p0 @ _exp(R_IstoI @ f.w_inIs0 * td)) @ R_IstoI
    R_Is1toG = R_Is0toG @ f.D_R @ R_bcorr
    # invert R_Is1toG = (R_p1 @ Exp(R_IstoI w_inIs1 td)) R_IstoI  ->  R_p1
    R_p1 = R_Is1toG @ R_IstoI.T @ _exp(R_IstoI @ f.w_inIs1 * td).T
    pose0 = gtsam.Pose3(gtsam.Rot3(R_p0), gtsam.Point3(0.0, 0.0, 0.0))
    pose1 = gtsam.Pose3(gtsam.Rot3(R_p1), gtsam.Point3(0.0, 0.0, 0.0))
    bg1 = bg0  # err_b = 0
    return pose0, pose1, bg0, bg1, xi_calib, d_th
