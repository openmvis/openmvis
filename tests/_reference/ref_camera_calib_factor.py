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
# K2 oracle (2026-11-01): verbatim copy of the pre-K2 live module (pre-release history),
# only the imports below changed (lie_groups / projection -> the frozen S0 copies;
# value classes and untouched leaves stay live). NEVER EDITED afterwards.
"""CameraCalibFactor — camera reprojection factor over the CameraCalib bundle.

Python port of the C++ MVIS `graph_Sensor/VFCalibTimeIn_SE3.h`, restructured so all
per-camera calibration lives in ONE `CameraCalib` variable (T_CtoI + td + tr +
intrinsics, (8 + n_intr)-dim product manifold) instead of the C++'s four separate keys —
and the scene is a calibration **board pose** key (known corner geometry
`p_FinB`) instead of free 3-D feature points. This is the factor the joint
camera-IMU calibration (`pipeline_cam_imu`) hangs the camera measurements on.

    keys = [pose, vb, board, calib]
      pose  : Pose3    IMU pose T_ItoW at the frame's stamped time
      vb    : Vector9  [v_inW(3), bias_gyro(3), bias_accel(3)]
      board : Pose3    board pose T_BinW
      calib : Vector16 ξ — tangent of CameraCalib around a fixed base

Forward model per corner i (matches the C++ `evaluateErrorOnly`):
    calib   = base.retract(ξ)
    t_i     = calib.td + ρ_i·calib.tr        ρ_i = rolling-shutter row offset of
                                             the MEASURED row (uv_meas[i,1])
    T_td    = ( Exp(w_m·t_i),  v_inI·t_i )   v_inI = R_WtoI·v   (time-offset
                                             propagation, measured ang. velocity)
    T_CtoW  = pose · T_td · calib.T
    r_i     = π_cal( T_CtoW⁻¹ · T_BinW · p_FinB_i ) − uv_meas_i

One factor per FRAME: the residual stacks all N visible corners (2N-dim); each
corner keeps its own rolling-shutter time t_i. A corner whose projection throws
(cheirality / out of model range) contributes zero residual + zero Jacobian rows.

Noise: plain isotropic pixel noise via the gtsam noise model (no state-dependent
covariance here, unlike the IMU factors — no internal whitening).

Jacobians are ANALYTIC (chain rule; the C++ factor is fully numerical, so FD is
the ground truth — `_numerical_jacobians` is kept as the built-in cross-check):
    H_board : J_point · [−R_B·[p_B]×, R_B]                  (transformFrom)
    H_pose  : J_pose · Ad(M⁻¹)  +  velocity coupling        M = T_td·calib.T
              (v_inI = R_Xᵀv ⇒ δv_inI = [v_inI]×·δθ_X, enters T_td's
               translation scaled by t_i — vanishes at t_i = 0)
    H_vb    : velocity cols J_pose · Ad(T_C⁻¹) · [0; Aᵀ·R_WtoI·t_i]; bias cols 0
    H_calib : ξ_T cols J_pose · Jr_se3(ξ_T)   (retract chain around base.T)
              td col   J_pose · Ad(T_C⁻¹) · [w_m; Aᵀ·v_inI]
              tr col   ρ_i × td col
              intr     model J_intr (2 × n_intr; gtsam 2×9 minus skew → 2×8)
"""

from __future__ import annotations

import numpy as np
import gtsam

from .ref_lie_groups import skew, exp_so3, se3_right_jacobian
from mvis.calibration.factors.camera_calib import CameraCalib
from .ref_projection import make_projector
from mvis.calibration.factors.numdiff import numerical_jacobian


def _rs_row_offset(row: float, h: int, mode: int) -> float:
    """Dimensionless rolling-shutter row factor ρ: capture time = td + ρ·tr.

    Modes match MVIS / `simulation.camera_measurements._rs_row_time`:
      0: frame stamped at top    -> ρ = row/H
      1: frame stamped at middle -> ρ = row/H − 0.5
      2: frame stamped at bottom -> ρ = (row − H)/H
    """
    if mode == 0:
        return row / h
    if mode == 1:
        return row / h - 0.5
    return (row - h) / h


def _time_shift_pose(w_m: np.ndarray, v_inI: np.ndarray, t: float) -> gtsam.Pose3:
    """T_td = (Exp(w_m·t), v_inI·t) — the MVIS time-offset propagation."""
    return gtsam.Pose3(gtsam.Rot3(exp_so3(w_m * t)), gtsam.Point3(*(v_inI * t)))


class CameraCalibFactor:
    """Per-frame stacked reprojection factor over [pose, vb, board, calib(ξ16)]."""

    def __init__(self, uv_meas, p_FinB, w_m, calib_base: CameraCalib,
                 image_size, rolling_mode: int = 0):
        self.uv = np.asarray(uv_meas, float).reshape(-1, 2)
        self.p_B = np.asarray(p_FinB, float).reshape(-1, 3)
        assert len(self.uv) == len(self.p_B), "one board corner per measurement"
        self.w_m = np.asarray(w_m, float)
        self.base = calib_base                  # base for the CameraCalib tangent ξ
        self.w_px, self.h_px = int(image_size[0]), int(image_size[1])
        self.rolling_mode = int(rolling_mode)
        self.n = len(self.uv)
        # row factors ρ_i from the MEASURED rows (C++ uses uvm_(1))
        self.rho = np.array([_rs_row_offset(v, self.h_px, self.rolling_mode)
                             for v in self.uv[:, 1]])

    # ------------------------------------------------------------------
    def error(self, pose: gtsam.Pose3, vb, T_BinW: gtsam.Pose3, xi):
        """Stacked 2N reprojection residual (unwhitened)."""
        calib = self.base.retract(np.asarray(xi, float))
        v_inI = pose.rotation().matrix().T @ np.asarray(vb, float)[0:3]
        proj = make_projector(calib.model, calib.intr)
        err = np.zeros(2 * self.n)
        T_td_cache = None                       # shared when tr ≈ 0 (global shutter)
        for i in range(self.n):
            t_i = calib.td + self.rho[i] * calib.tr
            if calib.tr == 0.0:
                if T_td_cache is None:
                    T_td_cache = _time_shift_pose(self.w_m, v_inI, t_i)
                T_td = T_td_cache
            else:
                T_td = _time_shift_pose(self.w_m, v_inI, t_i)
            T_CtoW = pose.compose(T_td).compose(calib.T)
            p_W = T_BinW.transformFrom(gtsam.Point3(*self.p_B[i]))
            try:
                px, _, _, _ = proj.project(T_CtoW, p_W, jac=False)
            except RuntimeError:                # cheirality / model range
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
        return err

    # ------------------------------------------------------------------
    def error_and_jacobians(self, pose, vb, T_BinW, xi):
        """Returns (error 2N, [H_pose 2N×6, H_vb 2N×9, H_board 2N×6,
        H_calib 2N×(8+n_intr)])."""
        xi = np.asarray(xi, float)
        calib = self.base.retract(xi)
        R_X = pose.rotation().matrix()
        v_inI = R_X.T @ np.asarray(vb, float)[0:3]
        proj = make_projector(calib.model, calib.intr)
        n_intr = calib.n_intr
        R_B = T_BinW.rotation().matrix()
        T_C = calib.T
        Ad_TCinv = T_C.inverse().AdjointMap()
        Jr_xiT = se3_right_jacobian(xi[0:6])    # retract chain around base.T

        err = np.zeros(2 * self.n)
        H_pose = np.zeros((2 * self.n, 6))
        H_vb = np.zeros((2 * self.n, 9))
        H_board = np.zeros((2 * self.n, 6))
        H_calib = np.zeros((2 * self.n, calib.DIM))

        for i in range(self.n):
            t_i = calib.td + self.rho[i] * calib.tr
            A = exp_so3(self.w_m * t_i)
            T_td = gtsam.Pose3(gtsam.Rot3(A), gtsam.Point3(*(v_inI * t_i)))
            M = T_td.compose(T_C)
            T_CtoW = pose.compose(M)
            p_W = np.asarray(T_BinW.transformFrom(gtsam.Point3(*self.p_B[i])), float)

            try:
                px, J_pose, J_point, J_intr = proj.project(T_CtoW, p_W)
            except RuntimeError:                # cheirality: zero rows
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            r = slice(2 * i, 2 * i + 2)

            # board: p_W = B·p_B  (gtsam transformFrom pose Jacobian)
            H_board[r, 0:3] = J_point @ (-R_B @ skew(self.p_B[i]))
            H_board[r, 3:6] = J_point @ R_B

            # pose: transport through M, + the v_inI = R_Xᵀ·v rotation coupling
            H_pose[r] = J_pose @ M.inverse().AdjointMap()
            xi_v = np.zeros((6, 3))
            xi_v[3:6] = A.T * t_i               # δp_td = δv_inI·t in T_td right coords
            H_pose[r, 0:3] += J_pose @ Ad_TCinv @ xi_v @ skew(v_inI)

            # vb: velocity columns only (δv_inI = R_Xᵀ·δv)
            H_vb[r, 0:3] = J_pose @ Ad_TCinv @ xi_v @ R_X.T

            # calib: extrinsic ξ_T retract chain; td/tr via T_td's time velocity
            H_calib[r, 0:6] = J_pose @ Jr_xiT
            xi_t = np.concatenate([self.w_m, A.T @ v_inI])   # dξ_td/dt (right tangent)
            col_t = J_pose @ Ad_TCinv @ xi_t
            H_calib[r, 6] = col_t
            H_calib[r, 7] = self.rho[i] * col_t
            H_calib[r, 8:8 + n_intr] = J_intr

        return err, [H_pose, H_vb, H_board, H_calib]

    def _numerical_jacobians(self, pose, vb, T_BinW, xi, eps: float = 1e-6):
        """Central-difference Jacobians (built-in cross-check for the analytic)."""
        _, blocks = numerical_jacobian(
            lambda p, v, b, x: self.error(p, v, b, x), [pose, vb, T_BinW, xi], eps)
        return blocks

    # ------------------------------------------------------------------
    def make_factor(self, k_pose, k_vb, k_board, k_calib, sigma_px: float = 1.0,
                    huber_px: float = 0.0):
        """Build the 4-key gtsam.CustomFactor (isotropic pixel noise).

        huber_px > 0 wraps the noise in a Huber M-estimator with the threshold
        given in PIXELS (converted to whitened units internally) — for real
        detections with outliers; sim tests keep the plain quadratic model.
        """
        keys = gtsam.KeyVector([k_pose, k_vb, k_board, k_calib])
        noise = gtsam.noiseModel.Isotropic.Sigma(2 * self.n, sigma_px)
        if huber_px > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber_px / sigma_px),
                noise)

        def err_func(this, values, H):
            pose = values.atPose3(k_pose)
            vb = values.atVector(k_vb)
            board = values.atPose3(k_board)
            xi = values.atVector(k_calib)
            if H is None:
                return self.error(pose, vb, board, xi)
            e, Js = self.error_and_jacobians(pose, vb, board, xi)
            for i in range(4):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
