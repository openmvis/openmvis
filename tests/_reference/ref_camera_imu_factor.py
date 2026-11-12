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
"""Frozen copy of `mvis.calibration.factors.camera_imu_factor` (S0, never
edited). Only the imports change (projector, `_rs_row_offset`, `ImuSegment`,
lie groups all come from the frozen copies); `_intr` freezes the retracted
ImuModel through `wrap_intr` (S0 adapter). `CameraCalib` is the live value
class (a plain container; its `retract` is not touched by any step).
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.camera_calib import CameraCalib   # value class
from mvis.imu import create_imu_model                           # value object
from .ref_camera_factor import _rs_row_offset
from .ref_imu_segment import ImuSegment
from .ref_imu_model import wrap_intr
from .ref_projection import make_projector
from .ref_lie_groups import skew, se3_right_jacobian


class CameraCalibImuFactor:
    """Per-frame stacked reprojection over [pose_k, vb_k, board, calib(ξ), (ξ_imu)]."""

    def __init__(self, uv_meas, p_FinB, dt_anchor: float, segment: ImuSegment,
                 t_anchor: float, g_inW, calib_base: CameraCalib, image_size,
                 rolling_mode: int = 0, gyro: str = "calibrated",
                 dw_column: bool = True, frame_dt: bool = False,
                 board_pids=None, n_board_points: int = 0):
        self.uv = np.asarray(uv_meas, float).reshape(-1, 2)
        self.p_B = np.asarray(p_FinB, float).reshape(-1, 3)
        assert len(self.uv) == len(self.p_B), "one board corner per measurement"
        # Estimated board shape (optional, as in CameraCalibInterpFactor): one
        # height offset per board point in a trailing key, added along the
        # board normal at the corners' ids.
        self.pids = None if board_pids is None else np.asarray(board_pids, int).reshape(-1)
        self.n_bp = int(n_board_points)
        if self.pids is not None:
            assert len(self.pids) == len(self.uv) and self.n_bp > int(self.pids.max())
        assert gyro in ("calibrated", "raw")
        self.dt_anchor = float(dt_anchor)
        self.seg = segment
        self.t_anchor = float(t_anchor)
        self.g = np.asarray(g_inW, float)
        self.base = calib_base
        self.w_px, self.h_px = int(image_size[0]), int(image_size[1])
        self.rolling_mode = int(rolling_mode)
        self.gyro = gyro
        self.dw_column = bool(dw_column)
        self.frame_dt = bool(frame_dt)
        self.n = len(self.uv)
        self.rho = np.array([_rs_row_offset(v, self.h_px, self.rolling_mode)
                             for v in self.uv[:, 1]])
        self.imu_base = create_imu_model(segment.model)   # ξ24 base (identity)
        self.keys = None                                   # set by make_factor
        self.key_dt = None

    # ------------------------------------------------------------------
    @property
    def uses_imu_key(self) -> bool:
        return self.gyro == "calibrated"

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

    def _intr(self, xi_imu):
        if not self.uses_imu_key or xi_imu is None:
            return None                                    # identity intrinsics
        return wrap_intr(self.imu_base.retract(np.asarray(xi_imu, float)))   # S0 adapter

    def deltas(self, calib: CameraCalib, dt_f=None) -> np.ndarray:
        """Signed capture offsets from the anchor state per corner (s)."""
        d_f = 0.0 if dt_f is None else float(np.asarray(dt_f).ravel()[0])
        return self.dt_anchor + calib.td + d_f + self.rho * calib.tr

    def _poses(self, pose_k, vb, calib, xi_imu, jac, dt_f=None):
        """Per-corner (T_i, Propagated, Δ_i) plus R_k, p_k, v_k."""
        R_k = pose_k.rotation().matrix()
        p_k = np.asarray(pose_k.translation(), float)
        vb = np.asarray(vb, float)
        v_k, b_g, b_a = vb[0:3], vb[3:6], vb[6:9]
        dl = self.deltas(calib, dt_f)
        props = self.seg.propagate(self.t_anchor, dl, b_g, b_a,
                                   self._intr(xi_imu), jac=jac)
        poses = []
        for i in range(self.n):
            P, d = props[i], dl[i]
            R_i = R_k @ P.D_R
            p_i = p_k + v_k * d + 0.5 * self.g * d * d + R_k @ P.D_p
            poses.append(gtsam.Pose3(gtsam.Rot3(R_i), gtsam.Point3(*p_i)))
        return poses, props, dl, R_k, v_k

    def error(self, pose_k, vb, T_BinW, xi, xi_imu=None, dt_f=None, h=None):
        """Stacked 2N reprojection residual (unwhitened)."""
        return self._residuals(pose_k, vb, T_BinW, xi, xi_imu, dt_f, h)[0]

    def _residuals(self, pose_k, vb, T_BinW, xi, xi_imu=None, dt_f=None, h=None):
        calib = self.base.retract(np.asarray(xi, float))
        proj = make_projector(calib.model, calib.intr)
        poses, _p, _d, _R, _v = self._poses(pose_k, vb, calib, xi_imu, jac=False,
                                            dt_f=dt_f)
        err = np.zeros(2 * self.n)
        valid = np.zeros(self.n, bool)
        pB = self._pB(h)
        for i in range(self.n):
            T_CtoW = poses[i].compose(calib.T)
            p_W = T_BinW.transformFrom(gtsam.Point3(*pB[i]))
            try:
                px, _, _, _ = proj.project(T_CtoW, p_W, jac=False)
            except RuntimeError:                # cheirality / model range
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            valid[i] = True
        return err, valid

    def corner_norms(self, pose_k, vb, T_BinW, xi, xi_imu=None, dt_f=None, h=None):
        """Per-corner reprojection error norms (px); NaN where the guard
        skipped the corner."""
        err, valid = self._residuals(pose_k, vb, T_BinW, xi, xi_imu, dt_f, h)
        norms = np.linalg.norm(err.reshape(-1, 2), axis=1)
        norms[~valid] = np.nan
        return norms

    def args_from_values(self, values: gtsam.Values):
        """(pose_k, vb, board, xi, xi_imu|None, dt_f|None, h|None) at a gtsam
        Values, using the keys from make_factor."""
        k = self.keys
        q = 4
        xi_imu = dt_f = h = None
        if self.uses_imu_key:
            xi_imu = values.atVector(k[q]); q += 1
        if self.frame_dt:
            dt_f = values.atVector(k[q]); q += 1
        if self.uses_heights_key:
            h = values.atVector(k[q])
        return (values.atPose3(k[0]), values.atVector(k[1]), values.atPose3(k[2]),
                values.atVector(k[3]), xi_imu, dt_f, h)

    def corner_norms_values(self, values: gtsam.Values):
        """corner_norms at a gtsam Values, using the keys from make_factor."""
        return self.corner_norms(*self.args_from_values(values))

    # ------------------------------------------------------------------
    def error_and_jacobians(self, pose_k, vb, T_BinW, xi, xi_imu=None, dt_f=None,
                            h=None):
        """(error 2N, [H_pose 2N×6, H_vb 2N×9, H_board 2N×6,
        H_calib 2N×DIM, (H_imu 2N×24 when calibrated), (H_dt 2N×1 when frame_dt),
        (H_heights 2N×n_board_points when the board shape is estimated)])."""
        xi = np.asarray(xi, float)
        calib = self.base.retract(xi)
        proj = make_projector(calib.model, calib.intr)
        n_intr = calib.n_intr
        R_B = T_BinW.rotation().matrix()
        T_C = calib.T
        Ad_TCinv = T_C.inverse().AdjointMap()
        Jr_xiT = se3_right_jacobian(xi[0:6])

        poses, props, dl, R_k, v_k = self._poses(pose_k, vb, calib, xi_imu, jac=True,
                                                 dt_f=dt_f)

        err = np.zeros(2 * self.n)
        H_pose = np.zeros((2 * self.n, 6))
        H_vb = np.zeros((2 * self.n, 9))
        H_board = np.zeros((2 * self.n, 6))
        H_calib = np.zeros((2 * self.n, calib.DIM))
        H_imu = np.zeros((2 * self.n, 24)) if self.uses_imu_key else None
        H_dt = np.zeros((2 * self.n, 1)) if self.frame_dt else None
        H_h = np.zeros((2 * self.n, self.n_bp)) if self.uses_heights_key else None
        pB = self._pB(h)
        n_B = R_B[:, 2]                         # d p_W / d h  (board normal in W)

        for i in range(self.n):
            P, d, T_I = props[i], dl[i], poses[i]
            T_CtoW = T_I.compose(T_C)
            p_W = np.asarray(T_BinW.transformFrom(gtsam.Point3(*pB[i])), float)
            try:
                px, J_pose, J_point, J_intr = proj.project(T_CtoW, p_W)
            except RuntimeError:                # cheirality: zero rows
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            r = slice(2 * i, 2 * i + 2)
            J_T = J_pose @ Ad_TCinv            # w.r.t. T_i right tangent
            DRt = P.D_R.T
            R_i = T_I.rotation().matrix()

            # anchor state: Ad(T_Δ⁻¹)
            M = np.zeros((6, 6))
            M[0:3, 0:3] = DRt
            M[3:6, 0:3] = -DRt @ skew(P.D_p)
            M[3:6, 3:6] = DRt
            H_pose[r] = J_T @ M

            # velocity (world) and biases
            H_vb[r, 0:3] = J_T[:, 3:6] @ (R_i.T * d)
            H_vb[r, 3:9] = J_T @ np.vstack([P.J_R_b, DRt @ P.J_p_b])

            if H_imu is not None and self.dw_column:
                H_imu[r] = J_T @ np.vstack([P.J_R_in, DRt @ P.J_p_in])

            H_board[r, 0:3] = J_point @ (-R_B @ skew(pB[i]))
            H_board[r, 3:6] = J_point @ R_B
            if H_h is not None:
                H_h[r, self.pids[i]] = J_point @ n_B
            H_calib[r, 0:6] = J_pose @ Jr_xiT
            v_W = v_k + self.g * d + R_k @ P.D_v
            zeta_t = np.concatenate([P.dtheta_dDelta, R_i.T @ v_W])
            col_t = J_T @ zeta_t
            H_calib[r, 6] = col_t
            if H_dt is not None:
                H_dt[r, 0] = col_t                  # δ_f enters exactly like td
            H_calib[r, 7] = self.rho[i] * col_t
            H_calib[r, 8:8 + n_intr] = J_intr

        Hs = [H_pose, H_vb, H_board, H_calib]
        if H_imu is not None:
            Hs.append(H_imu)
        if H_dt is not None:
            Hs.append(H_dt)
        if H_h is not None:
            Hs.append(H_h)
        return err, Hs

    # ------------------------------------------------------------------
    def make_factor(self, k_pose, k_vb, k_board, k_calib, k_imu=None, k_dt=None,
                    k_heights=None, sigma_px: float = 1.0, huber_px: float = 0.0):
        """4- to 7-key gtsam.CustomFactor (isotropic pixel noise, optional Huber)."""
        ks = [k_pose, k_vb, k_board, k_calib]
        if self.uses_imu_key:
            assert k_imu is not None, "calibrated gyro needs the IMU-intrinsics key"
            ks.append(k_imu)
        if self.frame_dt:
            assert k_dt is not None, "frame_dt needs the per-frame time-offset key"
            ks.append(k_dt)
        self.key_dt = k_dt if self.frame_dt else None   # explicit: heights may follow
        if self.uses_heights_key:
            assert k_heights is not None, "board_pids needs the board-heights key"
            ks.append(k_heights)
        self.keys = list(ks)
        keys = gtsam.KeyVector(ks)
        noise = gtsam.noiseModel.Isotropic.Sigma(2 * self.n, sigma_px)
        if huber_px > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber_px / sigma_px),
                noise)
        n_keys = len(ks)

        def err_func(this, values, H):
            args = self.args_from_values(values)
            if H is None:
                return self.error(*args)
            e, Js = self.error_and_jacobians(*args)
            for i in range(n_keys):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
