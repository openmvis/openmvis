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
"""Frozen copies of the pipeline helpers the diagnostics / init steps rewrite
(S0, never edited): `cam_imu_init._interp_vec` / `_slice_imu` (K8a),
`pipeline_cam_imu._propagate_state` / `_densify_velocities` (K8c) and
`_frame_corner_norms` / `_reproj_rms` / `_reproj_corner_norms` (D1, D2).

Only the imports change: the preintegrator is the frozen `ref_aci` copy. The
gtsam key symbols (X, B0, _Cc) and `_imu_model(cfg)` are plain constants /
config parsing and stay imported from the live pipeline.
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.imu import create_imu_model                                  # value object
from mvis.calibration.pipeline_cam_imu import X, B0, _Cc, _imu_model   # key symbols
from .ref_aci import ACIPreintegrator


# --- cam_imu_init.py ------------------------------------------------------

def _interp_vec(t_query, t, v):
    """Per-axis linear interpolation of a (N,3) signal."""
    return np.column_stack([np.interp(t_query, t, v[:, i]) for i in range(3)])


def _slice_imu(t_imu, w_m, a_m, t0, t1):
    """IMU samples covering [t0, t1], boundary samples linearly interpolated."""
    t_imu = np.asarray(t_imu, float)
    inside = (t_imu > t0) & (t_imu < t1)
    t = np.concatenate([[t0], t_imu[inside], [t1]])
    w = np.vstack([_interp_vec([t0], t_imu, w_m),
                   np.asarray(w_m)[inside],
                   _interp_vec([t1], t_imu, w_m)])
    a = np.vstack([_interp_vec([t0], t_imu, a_m),
                   np.asarray(a_m)[inside],
                   _interp_vec([t1], t_imu, a_m)])
    return t, w, a


# --- pipeline_cam_imu.py --------------------------------------------------

def _propagate_state(T0: gtsam.Pose3, v0, t0, t1, t_imu, w_m, a_m, b_g, b_a,
                     cfg, g_inW):
    """ACI-propagate (T0, v0) at t0 to t1 (identity intrinsics)."""
    pre = ACIPreintegrator(cfg.sigma_g, cfg.sigma_bg, cfg.sigma_a, cfg.sigma_ba,
                           model=_imu_model(cfg), avg_meas=True)
    pre.reset(np.asarray(b_g, float), np.asarray(b_a, float))
    pre.set_intrinsics(create_imu_model(_imu_model(cfg)))
    t, w, a = _slice_imu(t_imu, w_m, a_m, t0, t1)
    pre.feed_imu_interval(t, w, a)
    R0 = T0.rotation().matrix()
    p0 = np.asarray(T0.translation(), float)
    dt = pre.Dt
    R1 = R0 @ pre.D_R
    v1 = v0 + g_inW * dt + R0 @ pre.D_v
    p1 = p0 + v0 * dt + 0.5 * g_inW * dt * dt + R0 @ pre.D_p
    return gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(*p1)), v1


def _densify_velocities(t_states, T_states, seed, t_imu, w_m, a_m,
                        cfg, g_inW):
    """Per-state velocity init: ACI propagation from the nearest seed keyframe
    at or before each state (constant extrapolation outside seed coverage)."""
    t_key = np.asarray(seed.t_key, float)
    v_key = np.asarray(seed.v_inW, float)
    v = np.empty((len(t_states), 3))
    for i, t in enumerate(t_states):
        j = int(np.searchsorted(t_key, t, side='right')) - 1
        if j < 0:
            v[i] = v_key[0]
            continue
        if t - t_key[j] < 1e-9:
            v[i] = v_key[j]
            continue
        # propagate v only (pose comes from PnP); cap at the next keyframe
        _T, v1 = _propagate_state(
            seed.T_ItoW[j], v_key[j], t_key[j], t, t_imu, w_m, a_m,
            seed.b_g, seed.b_a, cfg, g_inW)
        v[i] = v1
    return v


def _frame_corner_norms(fac, c, j, values):
    """Per-corner error norms of one board-mode vision factor at `values`.
    Factors built through make_factor know their own keys (interp AND
    IMU-propagated, any board); the legacy path assumes [X(j-1), X(j), B0]."""
    if getattr(fac, "keys", None) is not None:
        return fac.corner_norms_values(values)
    return fac.corner_norms(values.atPose3(X(j - 1)), values.atPose3(X(j)),
                            values.atPose3(B0), values.atVector(_Cc(c)))


def _reproj_rms(cam_factors, values):
    """Per-frame RMS of per-corner error norms (px) at the given Values, plus
    the camera index per frame (cam_factors entries are (c, j, factor)).
    Guard-skipped corners are excluded; a frame with none valid is NaN."""
    out = np.empty(len(cam_factors))
    cams = np.empty(len(cam_factors), int)
    for k, (c, j, fac) in enumerate(cam_factors):
        e = _frame_corner_norms(fac, c, j, values)
        out[k] = (np.sqrt(np.nanmean(e ** 2))
                  if np.isfinite(e).any() else np.nan)
        cams[k] = c
    return out, cams


def _reproj_corner_norms(cam_factors, values):
    """Kalibr-convention population for reprojection stats: per-corner error
    norms (px) of ALL frames concatenated, plus the camera index per corner.
    Guard-skipped (NaN) corners are dropped from the pool."""
    norms, cams = [], []
    for c, j, fac in cam_factors:
        e = _frame_corner_norms(fac, c, j, values)
        v = np.isfinite(e)
        norms.append(e[v])
        cams.append(np.full(int(v.sum()), c, int))
    if not norms:
        return np.empty(0), np.empty(0, int)
    return np.concatenate(norms), np.concatenate(cams)
