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
"""Frozen copy of `mvis.calibration.factors.camera_interp_factor` (S0, never
edited). Only the imports change (projector, `_rs_row_offset` and the lie
group functions come from the frozen copies; `CameraCalib` is the live value
class). `_numerical_jacobians` is dropped (FD helper, not an oracle).
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.camera_calib import CameraCalib   # value class
from .ref_camera_factor import _rs_row_offset
from .ref_projection import make_projector
from .ref_lie_groups import (exp_so3, log_so3, skew, se3_right_jacobian,
                             so3_right_jacobian, so3_right_jacobian_inv,
                             so3_left_jacobian_inv)


def _interp_pose(pose_k: gtsam.Pose3, pose_k1: gtsam.Pose3, lam: float):
    """T(lam) plus the pieces the Jacobians reuse."""
    R_k = pose_k.rotation().matrix()
    p_k = np.asarray(pose_k.translation(), float)
    p_k1 = np.asarray(pose_k1.translation(), float)
    phi = log_so3(R_k.T @ pose_k1.rotation().matrix())
    A = exp_so3(lam * phi)
    R_lam = R_k @ A
    p_lam = (1.0 - lam) * p_k + lam * p_k1
    return gtsam.Pose3(gtsam.Rot3(R_lam), gtsam.Point3(*p_lam)), phi, A, R_lam


class CameraCalibInterpFactor:
    """Per-frame stacked reprojection over [pose_k, pose_k1, board, calib(ξ16)]."""

    def __init__(self, uv_meas, p_FinB, dt_frame: float, dt_interval: float,
                 calib_base: CameraCalib, image_size, rolling_mode: int = 0,
                 board_pids=None, n_board_points: int = 0):
        self.uv = np.asarray(uv_meas, float).reshape(-1, 2)
        self.p_B = np.asarray(p_FinB, float).reshape(-1, 3)
        assert len(self.uv) == len(self.p_B), "one board corner per measurement"
        assert dt_interval > 0
        # Estimated board shape (optional): `board_pids` are the corners' board
        # point ids; the factor then takes a trailing key holding one height
        # offset per board point (length n_board_points), added along the board
        # normal: p_B_i + [0, 0, h[pid_i]].
        self.pids = None if board_pids is None else np.asarray(board_pids, int).reshape(-1)
        self.n_bp = int(n_board_points)
        if self.pids is not None:
            assert len(self.pids) == len(self.uv) and self.n_bp > int(self.pids.max())
        self.dt_frame = float(dt_frame)
        self.dt = float(dt_interval)
        self.base = calib_base
        self.w_px, self.h_px = int(image_size[0]), int(image_size[1])
        self.rolling_mode = int(rolling_mode)
        self.n = len(self.uv)
        self.rho = np.array([_rs_row_offset(v, self.h_px, self.rolling_mode)
                             for v in self.uv[:, 1]])
        self.keys = None                        # set by make_factor

    # ------------------------------------------------------------------
    def _lam(self, calib: CameraCalib, i: int) -> float:
        return (self.dt_frame + calib.td + self.rho[i] * calib.tr) / self.dt

    @property
    def uses_heights_key(self) -> bool:
        return self.pids is not None

    def _pB(self, h):
        """Board points of this frame's corners, with the estimated heights."""
        if self.pids is None or h is None:
            return self.p_B
        out = self.p_B.copy()
        out[:, 2] += np.asarray(h, float)[self.pids]
        return out

    def error(self, pose_k: gtsam.Pose3, pose_k1: gtsam.Pose3,
              T_BinW: gtsam.Pose3, xi, h=None):
        """Stacked 2N reprojection residual (unwhitened)."""
        return self._residuals(pose_k, pose_k1, T_BinW, xi, h)[0]

    def _residuals(self, pose_k, pose_k1, T_BinW, xi, h=None):
        """(stacked 2N unwhitened residual, per-corner validity mask). Invalid
        corners (cheirality / model-range guard) keep zero rows in the residual
        but are False in the mask so stats can exclude them."""
        calib = self.base.retract(np.asarray(xi, float))
        proj = make_projector(calib.model, calib.intr)
        err = np.zeros(2 * self.n)
        valid = np.zeros(self.n, bool)
        pB = self._pB(h)
        T_I_cache = None                        # shared when tr == 0
        for i in range(self.n):
            if calib.tr == 0.0 and T_I_cache is not None:
                T_I = T_I_cache
            else:
                T_I, _phi, _A, _R = _interp_pose(pose_k, pose_k1,
                                                 self._lam(calib, i))
                if calib.tr == 0.0:
                    T_I_cache = T_I
            T_CtoW = T_I.compose(calib.T)
            p_W = T_BinW.transformFrom(gtsam.Point3(*pB[i]))
            try:
                px, _, _, _ = proj.project(T_CtoW, p_W, jac=False)
            except RuntimeError:                # cheirality / model range
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            valid[i] = True
        return err, valid

    def corner_norms(self, pose_k, pose_k1, T_BinW, xi, h=None):
        """Per-corner reprojection error norms (px), Kalibr convention; NaN
        where the guard skipped the corner (excluded from stats, not zero)."""
        err, valid = self._residuals(pose_k, pose_k1, T_BinW, xi, h)
        norms = np.linalg.norm(err.reshape(-1, 2), axis=1)
        norms[~valid] = np.nan
        return norms

    def corner_norms_values(self, values: gtsam.Values):
        """corner_norms at a gtsam Values, using the keys from make_factor."""
        k = self.keys
        h = values.atVector(k[4]) if self.uses_heights_key else None
        return self.corner_norms(values.atPose3(k[0]), values.atPose3(k[1]),
                                 values.atPose3(k[2]), values.atVector(k[3]), h)

    # ------------------------------------------------------------------
    def error_and_jacobians(self, pose_k, pose_k1, T_BinW, xi, h=None):
        """(error 2N, [H_pose_k 2N×6, H_pose_k1 2N×6, H_board 2N×6,
        H_calib 2N×(8+n_intr), (H_heights 2N×n_board_points when estimated)])."""
        xi = np.asarray(xi, float)
        calib = self.base.retract(xi)
        proj = make_projector(calib.model, calib.intr)
        n_intr = calib.n_intr
        R_B = T_BinW.rotation().matrix()
        T_C = calib.T
        Ad_TCinv = T_C.inverse().AdjointMap()
        Jr_xiT = se3_right_jacobian(xi[0:6])

        p_k = np.asarray(pose_k.translation(), float)
        p_k1 = np.asarray(pose_k1.translation(), float)
        dp = p_k1 - p_k

        err = np.zeros(2 * self.n)
        H_pk = np.zeros((2 * self.n, 6))
        H_pk1 = np.zeros((2 * self.n, 6))
        H_board = np.zeros((2 * self.n, 6))
        H_calib = np.zeros((2 * self.n, calib.DIM))
        H_h = np.zeros((2 * self.n, self.n_bp)) if self.uses_heights_key else None
        pB = self._pB(h)
        n_B = R_B[:, 2]                         # d p_W / d h  (board normal in W)

        cache = {}
        for i in range(self.n):
            lam = self._lam(calib, i)
            if lam in cache:
                T_I, phi, A, R_lam, M_k, M_k1, zeta_lam = cache[lam]
            else:
                T_I, phi, A, R_lam = _interp_pose(pose_k, pose_k1, lam)
                Jr_lp = so3_right_jacobian(lam * phi)
                # state blocks (right tangent of T(lam) per right tangent of X)
                M_k = np.zeros((6, 6))
                M_k[0:3, 0:3] = A.T - lam * Jr_lp @ so3_left_jacobian_inv(phi)
                M_k[3:6, 3:6] = (1.0 - lam) * A.T
                M_k1 = np.zeros((6, 6))
                M_k1[0:3, 0:3] = lam * Jr_lp @ so3_right_jacobian_inv(phi)
                M_k1[3:6, 3:6] = lam * exp_so3((1.0 - lam) * phi)
                zeta_lam = np.concatenate([phi, R_lam.T @ dp])
                cache[lam] = (T_I, phi, A, R_lam, M_k, M_k1, zeta_lam)

            T_CtoW = T_I.compose(T_C)
            p_W = np.asarray(T_BinW.transformFrom(gtsam.Point3(*pB[i])), float)

            try:
                px, J_pose, J_point, J_intr = proj.project(T_CtoW, p_W)
            except RuntimeError:                # cheirality: zero rows
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            r = slice(2 * i, 2 * i + 2)
            J_T = J_pose @ Ad_TCinv            # w.r.t. T(lam) right tangent

            H_pk[r] = J_T @ M_k
            H_pk1[r] = J_T @ M_k1
            H_board[r, 0:3] = J_point @ (-R_B @ skew(pB[i]))
            H_board[r, 3:6] = J_point @ R_B
            if H_h is not None:
                H_h[r, self.pids[i]] = J_point @ n_B
            H_calib[r, 0:6] = J_pose @ Jr_xiT
            col_t = J_T @ zeta_lam / self.dt   # d/dtd through lam
            H_calib[r, 6] = col_t
            H_calib[r, 7] = self.rho[i] * col_t
            H_calib[r, 8:8 + n_intr] = J_intr

        Hs = [H_pk, H_pk1, H_board, H_calib]
        if H_h is not None:
            Hs.append(H_h)
        return err, Hs

    # ------------------------------------------------------------------
    def make_factor(self, k_pose0, k_pose1, k_board, k_calib, k_heights=None,
                    sigma_px: float = 1.0, huber_px: float = 0.0):
        """4-key gtsam.CustomFactor (5 with the board-heights key; isotropic
        pixel noise, optional Huber)."""
        self.keys = [k_pose0, k_pose1, k_board, k_calib]
        if self.uses_heights_key:
            assert k_heights is not None, "board_pids needs the board-heights key"
            self.keys.append(k_heights)
        keys = gtsam.KeyVector(self.keys)
        noise = gtsam.noiseModel.Isotropic.Sigma(2 * self.n, sigma_px)
        if huber_px > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber_px / sigma_px),
                noise)

        def err_func(this, values, H):
            pose_k = values.atPose3(k_pose0)
            pose_k1 = values.atPose3(k_pose1)
            board = values.atPose3(k_board)
            xi = values.atVector(k_calib)
            h = values.atVector(k_heights) if self.uses_heights_key else None
            if H is None:
                return self.error(pose_k, pose_k1, board, xi, h)
            e, Js = self.error_and_jacobians(pose_k, pose_k1, board, xi, h)
            for i in range(len(Js)):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
